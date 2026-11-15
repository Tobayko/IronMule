"""GG1-H: IronMule's ggml engine through the product against `llama-server`, same CPU, model and GGUF (BACKLOG GG).

Arm L is LC1's `llama-server` (8 parallel slots, 4 threads, greedy, HTTP); arm G is `ProductService` with the ggml
engine on the same GGUF, speculation on, after its quality gate passed. Load: CPU6's 8 prompts at once, 64 tokens,
5 blocks of L G L; quiet: 16 Dolly prompts from index 14,800 one at a time, 128 tokens, one block of L G L.
`--analyse REPORT` applies the rule fixed in the entry before the data.

    python tools/gg1_measure.py --server PATH/llama-server --gguf MODEL.gguf --dolly DIR --out gg1h.json
    python -I tools/gg1_measure.py --analyse gg1h.json
"""

from __future__ import annotations

import argparse
import json
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
LOAD_BLOCKS, LOAD_LIMIT = 5, 64
QUIET_FIRST, QUIET_SIZE, QUIET_LIMIT = 14_800, 16, 128
ORDER = ("L", "G", "L")
GG3_ORDER = ("L0", "Ls", "G", "Ls", "L0")
GG3_QUIET = {None: 14_100, "Qwen/Qwen3-0.6B": 13_600, "Qwen/Qwen3-1.7B": 13_700}
RESAMPLES, GOAL = 2000, 1.2


def run_ggml(spec: Any, gguf: str, gate: dict[str, Any], texts: list[str], at_once: bool, limit: int) -> dict[str, Any]:
    """One arm on a fresh worker with the ggml engine: the prompts one after another, or all offered at once."""
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest

    store = ProductStore(Path(tempfile.mkdtemp(prefix="gg1h-")) / "state")
    store.setup("server")
    store.set_request_timeout(3600)
    store.register_model(spec)
    backend = MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="ggml", ggml_gguf=gguf,
                              ggml_gate=gate, speculate=True, startup_timeout=600)
    started = time.perf_counter()
    backend.start()
    startup = time.perf_counter() - started
    service = ProductService(store, backend=backend, spec=spec)
    out: dict[int, dict[str, Any]] = {}
    tag = f"{time.monotonic_ns()}"

    def ask(index: int, offered: float) -> None:
        request = GenerationRequest(spec.model_id, (("user", texts[index]),), max_tokens=limit,
                                    request_id=f"g{tag}-{index}", stream=True)
        first, text, tokens, error = None, [], None, None
        try:
            for event in service.stream(request):
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "token":
                    first = first if first is not None else time.perf_counter() - offered
                    text.append(event.get("text") or "")
                if event.get("type") == "done":
                    tokens = event.get("completion_tokens")
        except Exception as exc:  # noqa: BLE001 - recorded, and it makes the cell incomplete
            error = type(exc).__name__
        out[index] = {"first_s": first, "end_s": time.perf_counter() - offered, "tokens": tokens,
                      "text": "".join(text), "error": error}

    try:
        if at_once:
            offered = time.perf_counter()
            threads = [threading.Thread(target=ask, args=(i, offered)) for i in range(len(texts))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for i in range(len(texts)):
                ask(i, time.perf_counter())
        health = service.health()
    finally:
        service.close()
    rows = [out[i] for i in range(len(texts))]
    return {"arm": "G", "at_once": at_once, "startup_s": startup, "requests": rows,
            "complete": all(r["error"] is None and r["tokens"] and r["first_s"] is not None for r in rows),
            "execution": health.get("execution"), "ggml": health.get("ggml"), "speculation": health.get("speculation")}


def _interval(values: list[float], seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(RESAMPLES))
    return [statistics.fmean(values), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]]


def _rate(row: dict[str, Any]) -> float | None:
    span = row["end_s"] - row["first_s"] if row["first_s"] is not None else 0
    return (row["tokens"] - 1) / span if row["tokens"] and row["tokens"] > 1 and span > 0 else None


def _compare(load: list, quiet: list, g: int, refs: tuple[int, int]) -> dict[str, Any]:
    """G (cell `g` of each block) over the mean of the two cells `refs`, by GG1-H's rule."""
    def block_rate(cell):
        return sum(r["tokens"] for r in cell["requests"]) / max(r["end_s"] for r in cell["requests"])

    def block_first(cell):
        return statistics.fmean(r["first_s"] for r in cell["requests"])

    a, b = refs
    load_tps = _interval([block_rate(c[g]) / ((block_rate(c[a]) + block_rate(c[b])) / 2) for c in load])
    load_first = _interval([block_first(c[g]) / ((block_first(c[a]) + block_first(c[b])) / 2) for c in load])
    load_aa = statistics.fmean(abs(block_first(c[b]) / block_first(c[a]) - 1) for c in load)
    tps, first, aa_first = [], [], []
    for c in quiet:
        for r1, rg, r2 in zip(c[a]["requests"], c[g]["requests"], c[b]["requests"]):
            x1, xg, x2 = _rate(r1), _rate(rg), _rate(r2)
            if None in (x1, xg, x2):
                continue
            tps.append(xg / ((x1 + x2) / 2))
            first.append(rg["first_s"] / ((r1["first_s"] + r2["first_s"]) / 2))
            aa_first.append(abs(r2["first_s"] / r1["first_s"] - 1))
    quiet_tps, quiet_first = _interval(tps), _interval(first)
    checks = {"quiet_tokens_per_s": quiet_tps[1] >= GOAL, "load_tokens_per_s": load_tps[1] >= GOAL,
              "quiet_first": quiet_first[2] <= 1 + statistics.fmean(aa_first),
              "load_first": load_first[2] <= 1 + load_aa}
    return {"quiet": {"prompts": len(tps), "tokens_per_s": quiet_tps, "first": quiet_first,
                      "l_aa_first": statistics.fmean(aa_first)},
            "load": {"tokens_per_s": load_tps, "first": load_first, "l_aa_first": load_aa}, "checks": checks,
            "verdict": "goal reached" if all(checks.values()) else "goal not reached"}


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """The rule fixed in BACKLOG GG (GG1-H) before the data: G over the mean of L's two runs. GG3 (five cells a block,
    L0 Ls G Ls L0): the same rule against L0 (the goal as fixed in GG) and against Ls (the fair comparison)."""
    load, quiet = report.get("load", []), report.get("quiet", [])
    gg3 = str(report.get("protocol", "")).startswith("gg3")
    problems = []
    if len(load) != LOAD_BLOCKS or len(quiet) != 1:
        problems.append("missing blocks")
    if any(len(block) != (len(GG3_ORDER) if gg3 else len(ORDER)) for block in load + quiet):
        problems.append("missing cells")
    if not all(cell["complete"] for block in load + quiet for cell in block):
        problems.append("a request failed or returned nothing")
    if not report.get("gate", {}).get("passes"):
        problems.append("the gate did not pass")
    if problems:
        return {"valid": False, "problems": problems, "verdict": "invalid"}
    if not gg3:
        return {"valid": True, **_compare(load, quiet, 1, (0, 2)), "gate": report["gate"]}
    goal, fair = _compare(load, quiet, 2, (0, 4)), _compare(load, quiet, 2, (1, 3))
    return {"valid": True, "gate": {k: report["gate"].get(k) for k in ("ratio", "interval", "settings", "passes")},
            "ls_options": report.get("ls_options"), "against_l0": goal, "against_ls": fair,
            "verdict": goal["verdict"] if goal["verdict"] == fair["verdict"] else f"fair: {fair['verdict']}"}


def llama_options(settings: dict[str, bool]) -> tuple[str, ...]:
    """GG3's Ls: the `llama-server` flags of the engine's setting (4227c9b's spelling)."""
    return ((("-fa",) if settings.get("flash_attn") else ()) + (("-kvu",) if settings.get("kv_unified") else ())
            + (("-ctk", "f32", "-ctv", "f32") if settings.get("kv_f32") else ()))


def model_spec(model: str | None) -> Any:
    """The registered model: SmolLM2's MLX snapshot (GG1-GG3-F) or another cached Hugging Face snapshot (GG3-Q)."""
    import sp2_measure

    if not model:
        return sp2_measure._spec()
    from ironmule_product.types import ModelSpec

    root = Path(os.path.expanduser(f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots"))
    snapshot = sorted(root.iterdir())[-1].resolve()
    return ModelSpec(model, snapshot.name, str(snapshot), sum(f.stat().st_size for f in snapshot.glob("*.safetensors")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server")
    parser.add_argument("--gguf")
    parser.add_argument("--dolly", type=Path)
    parser.add_argument("--gate-cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analyse", type=Path)
    parser.add_argument("--smoke", action="store_true", help="tiny blocks on prompts the protocol never uses")
    parser.add_argument("--plan", choices=("gg1", "gg2", "gg3"), default="gg1",
                        help="gg2: GG2-H's quiet prompts (14,500); gg3: GG3's arms L0 Ls G Ls L0")
    parser.add_argument("--model", help="gg3: a registered Hugging Face snapshot (GG3-Q), else SmolLM2 (GG3-F)")
    args = parser.parse_args(argv)
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1))
        return 0
    sys.path[:0] = [str(ROOT), str(ROOT / "training"), str(ROOT / "tools")]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import cpu6_measure
    import lc1_measure
    import rl2_headroom

    from ironmule.ggml_engine import file_sha256, gate

    spec = model_spec(args.model)
    gguf = str(Path(args.gguf).resolve())
    checked = gate(spec.snapshot_path, gguf, cache=args.gate_cache)
    print(json.dumps({"gate": {k: checked.get(k) for k in ("ratio", "interval", "passes")}}), flush=True)
    texts, digest = rl2_headroom.dolly_prompts(args.dolly)
    quiet_first = {"gg1": QUIET_FIRST, "gg2": 14_500, "gg3": GG3_QUIET[args.model]}[args.plan]
    blocks, quiet_size, load_limit, quiet_limit, first_index = (
        (1, 2, 16, 16, quiet_first + (100 if args.plan == "gg1" else 50)) if args.smoke
        else (LOAD_BLOCKS, QUIET_SIZE, LOAD_LIMIT, QUIET_LIMIT, quiet_first))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    version = subprocess.run([args.server, "--version"], capture_output=True, text=True)
    report: dict[str, Any] = {"protocol": f"{args.plan}-h" + ("-smoke" if args.smoke else ""), "commit": commit,
                              "gguf_sha256": file_sha256(gguf), "gate": checked, "dolly_sha256": digest,
                              "llama_server": (version.stdout + version.stderr).strip()[-200:],
                              "machine": {"machine": platform.machine(), "cpus": os.cpu_count()}, "load": [],
                              "quiet": [], "model": spec.model_id, "revision": spec.revision}
    options = llama_options(checked.get("settings") or {})
    if args.plan == "gg3":
        report["ls_options"] = list(options)
    if not checked.get("passes"):
        if args.out:
            args.out.write_text(json.dumps(report))
        raise SystemExit("the GGUF did not pass the gate: GG1-H does not run")
    plan = ([("load", cpu6_measure.prompts(), True, load_limit)] * blocks
            + [("quiet", texts[first_index:first_index + quiet_size], False, quiet_limit)])
    for kind, chunk, at_once, limit in plan:
        cells = []
        for arm in (GG3_ORDER if args.plan == "gg3" else ORDER):
            cell = (run_ggml(spec, gguf, checked, chunk, at_once, limit) if arm == "G"
                    else lc1_measure.run_llama(args.server, gguf, chunk, at_once, limit,
                                               options if arm == "Ls" else ()))
            cell["arm"] = arm
            cells.append(cell)
            print(json.dumps({kind: len(report[kind]), "arm": arm, "complete": cell["complete"],
                              "tokens": sum(r["tokens"] or 0 for r in cell["requests"]),
                              "seconds": round(max(r["end_s"] for r in cell["requests"]), 2)}), flush=True)
        report[kind].append(cells)
        if args.out:
            args.out.write_text(json.dumps(report))
    report["analysis"] = analyse(report) if not args.smoke else None
    if args.out:
        args.out.write_text(json.dumps(report))
    print(json.dumps(report["analysis"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
