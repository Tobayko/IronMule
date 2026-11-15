"""CPU6-M: today's CPU path against the fast BLAS and the CPU stacked engine, through the product (BACKLOG CPU6).

Arms, each on a fresh worker started as `ironmule serve` starts it:

* A, today's CPU path: the reference worker with `--compute-dtype dequantize` on the stock BLAS;
* B, A with `--cpu-blas openblas`;
* C, `--engine dense` (the CPU stacked plan) with `--cpu-blas openblas`.

Each repetition offers the same 8 requests at once to each arm, in a rotated order, with A twice (its A/A noise).
`--gate` adds the numeric plans' perplexity gate: the dequantized model on OpenBLAS against the same on the stock
BLAS. `--analyse REPORT` applies the rule fixed in the entry before the data.

`--plan agent` is CPU6-A (step 3): the agent's own setup, made by `ironmule serve --agent on`'s `_agent_setup` on a
fresh store, is arm S, against A, in rotated order of A S A.

    python tools/cpu6_measure.py --model mlx-community/SmolLM2-135M-Instruct-8bit --out cpu6.json --gate
    python tools/cpu6_measure.py --plan agent --out cpu6a.json
    python -I tools/cpu6_measure.py --analyse cpu6.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUESTS = 8
LIMIT = 64
REPETITIONS = 5
ORDER = ("A", "B", "C", "A")
RESAMPLES = 2000
SEED = 20261110
PASS_B, PASS_C = 1.5, 3.0  # the 95 % lower bound on tokens per second over A
ORDER_AGENT = ("A", "S", "A")
PASS_S = 3.0  # CPU6-A: the agent's own setup over A
QUALITY_BOUND = 1.005
ARMS_EXTRA: dict[str, dict[str, Any]] = {}  # CPU6-A: "S", the agent's own setup, filled at run time
ARMS = {"A": {"compute_dtype": "dequantize"},
        "B": {"compute_dtype": "dequantize", "cpu_blas": "openblas"},
        "C": {"execution_variant": "current_engine", "engine_configuration": "dense_stacked", "cpu_blas": "openblas"}}


def prompts() -> list[str]:
    topics = ("animals that live in the sea", "uses for a paperclip", "reasons the sky looks blue",
              "steps to boil an egg", "facts about the moon", "ways to save water at home",
              "tips for learning a language", "parts of a bicycle")
    return [f"List {3 + i % 3} {topic}, one short line each." for i, topic in enumerate(topics)]


def rotation(repetitions: int = REPETITIONS, order: tuple[str, ...] = ORDER) -> list[list[str]]:
    return [list(order[k % len(order):] + order[:k % len(order)]) for k in range(repetitions)]


def agent_options(setup: dict[str, Any]) -> dict[str, Any]:
    """The worker options `ironmule serve` starts for the agent's setup (CPU6 step 3)."""
    from ironmule_product.cli import _engine_options

    blas = {"cpu_blas": setup["cpu_blas"]} if setup.get("cpu_blas") else {}
    if setup.get("engine") == "dense":
        return {"execution_variant": "current_engine", **_engine_options("dense", setup.get("width")), **blas}
    if setup.get("engine") != "reference":
        raise ValueError(f"CPU6-A serves the reference or the dense engine, not {setup.get('engine')!r}")
    return {**({"compute_dtype": setup["compute_dtype"]} if setup.get("compute_dtype") else {}), **blas}


def _spec(model_id: str):
    from ironmule_product.types import ModelSpec

    snapshot = sorted(glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*")))[-1]
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(snapshot, "*.safetensors")))
    return ModelSpec(model_id, os.path.basename(snapshot), str(Path(snapshot).resolve()), size)


def run_arm(spec: Any, arm: str, texts: list[str], limit: int) -> dict[str, Any]:
    """One arm on a fresh worker: every request offered at once; span, first tokens, answers."""
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest

    store = ProductStore(Path(tempfile.mkdtemp(prefix="cpu6-")) / "state")
    store.setup("server")
    store.set_request_timeout(7200)
    store.register_model(spec)
    options = dict(ARMS[arm]) if arm in ARMS else dict(ARMS_EXTRA[arm])
    options.pop("startup_timeout", None)
    backend = MLXWorkerClient(spec, startup_timeout=3600, **options)
    started = time.perf_counter()
    ready = backend.start()
    startup = time.perf_counter() - started
    service = ProductService(store, backend=backend, spec=spec)
    out: dict[int, dict[str, Any]] = {}

    def ask(index: int, offered: float) -> None:
        request = GenerationRequest(spec.model_id, (("user", texts[index]),), max_tokens=limit,
                                    request_id=f"r{index}", stream=True)
        first, text, tokens, error = None, [], None, None
        try:
            for event in service.stream(request):
                if not isinstance(event, dict):
                    continue
                if event.get("text"):
                    first = first if first is not None else time.perf_counter() - offered
                    text.append(event["text"])
                if event.get("type") == "done":
                    tokens = event.get("completion_tokens")
        except Exception as exc:  # noqa: BLE001 - recorded, and it makes the arm incomplete
            error = type(exc).__name__
        out[index] = {"first_s": first, "end_s": time.perf_counter() - offered, "tokens": tokens,
                      "text": "".join(text), "error": error}

    try:
        offered = time.perf_counter()
        threads = [threading.Thread(target=ask, args=(i, offered)) for i in range(len(texts))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        health = service.health()
    finally:
        service.close()
    rows = [out[i] for i in range(len(texts))]
    span = max(r["end_s"] for r in rows)
    complete = all(r["error"] is None and r["tokens"] for r in rows)
    engine = ready.get("engine") if isinstance(ready.get("engine"), dict) else {}
    return {"arm": arm, "startup_s": startup, "span_s": span, "complete": complete,
            "tokens": sum(r["tokens"] or 0 for r in rows), "tokens_per_s": sum(r["tokens"] or 0 for r in rows) / span,
            "mean_first_s": statistics.fmean(r["first_s"] for r in rows) if complete else None,
            "execution": health.get("execution"), "cpu_blas": health.get("cpu_blas"),
            "dense_plan": {k: engine.get("dense_plan", {}).get(k) for k in
                           ("kernels", "stacked_rows_identical", "reference_rel_error", "slots", "capacity")}
            if engine.get("dense_plan") else None,
            "requests": rows}


GATE_CHILD = r"""
import json, sys
from types import SimpleNamespace
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx_lm import load
from ironmule import cpu_blas
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos, nll_per_chunk
from ironmule.numeric_plans import dequantize_model
model, tokenizer = load(sys.argv[1])
dequantize_model(model)
rows = nll_per_chunk(SimpleNamespace(model=model), _text_ids(tokenizer), CHUNKS, CHUNK_TOKENS, gate_bos(tokenizer))
print(json.dumps({"serving": cpu_blas.active()["serving"], "nll": rows}))
"""


def gate(model_id: str) -> dict[str, Any]:
    """The perplexity gate: the dequantized model on OpenBLAS against the same on the stock BLAS."""
    from ironmule import cpu_blas
    from ironmule.numeric_choice import ratio_interval

    variables, record = cpu_blas.environment()
    if not variables:
        return {"ran": False, "reason": record.get("reason")}
    result = {}
    for name, extra in (("stock", {}), ("openblas", variables)):
        env = {**os.environ, **extra, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"}
        started = time.perf_counter()
        done = subprocess.run([sys.executable, "-c", GATE_CHILD, model_id], env=env, capture_output=True, text=True,
                              timeout=7200, check=False)
        if done.returncode != 0:
            return {"ran": False, "reason": f"the {name} gate child failed: {done.stderr[-300:]}"}
        result[name] = {**json.loads(done.stdout.strip().splitlines()[-1]), "seconds": time.perf_counter() - started}
    interval = ratio_interval(result["stock"]["nll"], result["openblas"]["nll"])
    return {"ran": True, "serving": {k: result[k]["serving"] for k in result}, **interval,
            "seconds": {k: result[k]["seconds"] for k in result}}


def _bootstrap(values: list[float], seed: int) -> tuple[float, float, float]:
    """Geometric mean of per-repetition ratios and its 95 % percentile interval."""
    logs = [math.log(v) for v in values]
    rng = random.Random(seed)
    boot = sorted(math.exp(statistics.fmean(rng.choice(logs) for _ in logs)) for _ in range(RESAMPLES))
    return math.exp(statistics.fmean(logs)), boot[int(0.025 * RESAMPLES)], boot[int(0.975 * RESAMPLES) - 1]


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """The rule fixed in BACKLOG CPU6 before the data."""
    problems = []
    reps = report.get("repetitions") or []
    if len(reps) < REPETITIONS:
        problems.append(f"{len(reps)} of {REPETITIONS} repetitions")
    cells = [cell for rep in reps for cell in rep]
    if any(not cell["complete"] for cell in cells):
        problems.append("a request failed or returned nothing")
    if problems:
        return {"valid": False, "problems": problems, "verdict": "invalid"}
    per = []
    for rep in reps:
        a = [c for c in rep if c["arm"] == "A"]
        b = next(c for c in rep if c["arm"] == "B")
        c = next(c for c in rep if c["arm"] == "C")
        a_tps = statistics.fmean(x["tokens_per_s"] for x in a)
        a_first = statistics.fmean(x["mean_first_s"] for x in a)
        per.append({"b_tps": b["tokens_per_s"] / a_tps, "c_tps": c["tokens_per_s"] / a_tps,
                    "b_first": b["mean_first_s"] / a_first, "c_first": c["mean_first_s"] / a_first,
                    "aa_first": abs(a[0]["mean_first_s"] - a[1]["mean_first_s"]) / a_first,
                    "aa_tps": abs(a[0]["tokens_per_s"] - a[1]["tokens_per_s"]) / a_tps})
    noise = statistics.fmean(p["aa_first"] for p in per)
    out: dict[str, Any] = {"valid": True, "aa_noise_first": noise,
                           "aa_noise_tokens_per_s": statistics.fmean(p["aa_tps"] for p in per)}
    for arm, bound in (("b", PASS_B), ("c", PASS_C)):
        tps = _bootstrap([p[f"{arm}_tps"] for p in per], SEED)
        first = _bootstrap([p[f"{arm}_first"] for p in per], SEED + 1)
        out[arm.upper()] = {"tokens_per_s": tps, "first": first,
                            "passes_tokens": tps[1] > bound, "passes_first": first[2] <= 1 + noise}
    quality = report.get("gate") or {}
    out["gate"] = {k: quality.get(k) for k in ("ran", "ratio", "interval", "passes", "reason")}
    gate_ok = bool(quality.get("ran")) and quality["interval"][1] <= QUALITY_BOUND
    for arm in ("B", "C"):
        out[arm]["passes"] = out[arm]["passes_tokens"] and out[arm]["passes_first"] and gate_ok
    out["verdict"] = {arm: "pass" if out[arm]["passes"] else "fail" for arm in ("B", "C")}
    first_a = [c for rep in reps for c in rep if c["arm"] == "A"][0]["requests"]
    same = {arm: sum(r["text"] == a["text"] for rep in reps for cell in rep if cell["arm"] == arm
                     for r, a in zip(cell["requests"], first_a)) for arm in ("A", "B", "C")}
    out["answers_equal_to_first_a"] = same
    return out


def analyse_agent(report: dict[str, Any]) -> dict[str, Any]:
    """CPU6-A's rule, fixed in BACKLOG CPU6 before the data."""
    setup = report.get("setup") or {}
    problems = []
    reps = report.get("repetitions") or []
    if len(reps) < REPETITIONS:
        problems.append(f"{len(reps)} of {REPETITIONS} repetitions")
    if any(not cell["complete"] for rep in reps for cell in rep):
        problems.append("a request failed or returned nothing")
    if problems:
        return {"valid": False, "problems": problems, "verdict": "invalid"}
    same_as_a = (setup.get("engine") == "reference" and setup.get("compute_dtype") == "dequantize"
                 and not setup.get("cpu_blas"))
    per = []
    for rep in reps:
        a = [c for c in rep if c["arm"] == "A"]
        s_ = next(c for c in rep if c["arm"] == "S")
        a_tps, a_first = (statistics.fmean(x[k] for x in a) for k in ("tokens_per_s", "mean_first_s"))
        per.append({"tps": s_["tokens_per_s"] / a_tps, "first": s_["mean_first_s"] / a_first,
                    "aa_first": abs(a[0]["mean_first_s"] - a[1]["mean_first_s"]) / a_first})
    noise = statistics.fmean(p["aa_first"] for p in per)
    tps, first = _bootstrap([p["tps"] for p in per], SEED), _bootstrap([p["first"] for p in per], SEED + 1)
    passes = not same_as_a and tps[1] > PASS_S and first[2] <= 1 + noise
    return {"valid": True, "setup": {k: setup.get(k) for k in ("engine", "compute_dtype", "cpu_blas", "width", "reason")},
            "chose_other_than_a": not same_as_a, "aa_noise_first": noise, "tokens_per_s": tps, "first": first,
            "verdict": "pass" if passes else "fail"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlx-community/SmolLM2-135M-Instruct-8bit")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--analyse", type=Path)
    parser.add_argument("--plan", choices=("blas", "agent"), default="blas")
    args = parser.parse_args(argv)
    if args.analyse:
        report = json.loads(args.analyse.read_text())
        print(json.dumps((analyse_agent if report.get("protocol") == "cpu6-a" else analyse)(report), indent=1))
        return 0
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    setup, setup_s = None, None
    if args.plan == "agent":
        os.environ["IRONMULE_HOME"] = tempfile.mkdtemp(prefix="cpu6a-store-")  # a fresh store: nothing stored
    import mlx.core as mx

    spec = _spec(args.model)
    if args.plan == "agent":
        from ironmule_product.cli import _agent_setup

        started = time.perf_counter()
        setup = _agent_setup(spec)
        setup_s = time.perf_counter() - started
        ARMS_EXTRA["S"] = agent_options(setup)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report: dict[str, Any] = {"protocol": "cpu6-a" if setup else "cpu6-m", "model": args.model,
                              "revision": spec.revision, "commit": commit,
                              "machine": {"machine": platform.machine(), "cpus": os.cpu_count(),
                                          "processor": platform.processor(), "mlx": mx.__version__},
                              "requests": REQUESTS, "limit": args.limit,
                              "order": rotation(args.repetitions, ORDER_AGENT if setup else ORDER),
                              "repetitions": []}
    if setup:
        report.update(setup=json.loads(json.dumps(setup, default=str)), setup_seconds=setup_s, arm_s=ARMS_EXTRA["S"])
    texts = prompts()
    for k, order in enumerate(report["order"]):
        cells = []
        for arm in order:
            cell = run_arm(spec, arm, texts, args.limit)
            cells.append(cell)
            print(json.dumps({"repetition": k, "arm": arm, "tokens_per_s": round(cell["tokens_per_s"], 2),
                              "mean_first_s": cell["mean_first_s"], "complete": cell["complete"]}), flush=True)
        report["repetitions"].append(cells)
        if args.out:
            args.out.write_text(json.dumps(report, indent=1))
    if args.gate:
        report["gate"] = gate(args.model)
        print(json.dumps({"gate": {k: report["gate"].get(k) for k in ("ratio", "interval", "passes", "reason")}}),
              flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps((analyse_agent if setup else analyse)(report), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
