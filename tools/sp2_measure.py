"""SP2-H: speculation with a learned window through the product, against the same stacked engine without it.

Arms, each on a fresh worker started as `ironmule serve` starts it, with the agent's CPU6-A2 setup named
(`dequantize` weights in the CPU stacked plan at width 8, OpenBLAS):

* A, `--speculate off`;
* S, `--speculate context` (the window policy, from its priors at each start).

Quiet: 3 blocks of 16 Dolly prompts, one request at a time, A S A per block. Load: 3 blocks of 8 prompts offered
at once, A S A. `--analyse REPORT` applies the rule fixed in BACKLOG SP2 before the data.

    python tools/sp2_measure.py --dolly DIR --out sp2h.json
    python tools/sp2_measure.py --plan sp3 --dolly DIR --out sp3h.json   # SP3-H: policy version 2, fresh prompts
    python -I tools/sp2_measure.py --analyse sp2h.json
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
MODEL, REVISION = "mlx-community/SmolLM2-135M-Instruct-8bit", "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
LIMIT = 128
QUIET_BLOCKS, QUIET_SIZE, QUIET_FIRST = 3, 16, 13_000
LOAD_BLOCKS, LOAD_SIZE, LOAD_FIRST = 3, 8, 13_100
PLANS = {"sp2": (QUIET_FIRST, LOAD_FIRST), "sp3": (13_200, 13_300)}  # SP3-H: prompts no earlier run used
ORDER = ("A", "S", "A")
RESAMPLES = 2000
QUIET_BOUND, LOAD_TOKENS_FLOOR, LOAD_FIRST_BOUND = 1.3, 0.98, 1.02
BASE = {"execution_variant": "current_engine", "engine_configuration": "dense_stacked", "dense_width": 8,
        "cpu_blas": "openblas"}
ARMS = {"A": dict(BASE), "S": {**BASE, "speculate": True}}


def _spec():
    from ironmule_product.types import ModelSpec

    snapshot = Path(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/"
                                       f"{REVISION}")).resolve()
    size = sum(f.stat().st_size for f in snapshot.glob("*.safetensors"))
    return ModelSpec(MODEL, REVISION, str(snapshot), size)


def run_arm(spec: Any, arm: str, texts: list[str], at_once: bool, limit: int = LIMIT) -> dict[str, Any]:
    """One arm on a fresh worker: the prompts one after another, or all offered at once."""
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest

    store = ProductStore(Path(tempfile.mkdtemp(prefix="sp2h-")) / "state")
    store.setup("server")
    store.set_request_timeout(7200)
    store.register_model(spec)
    backend = MLXWorkerClient(spec, startup_timeout=3600, **ARMS[arm])
    started = time.perf_counter()
    ready = backend.start()
    startup = time.perf_counter() - started
    service = ProductService(store, backend=backend, spec=spec)
    out: dict[int, dict[str, Any]] = {}

    def ask(index: int, offered: float) -> None:
        request = GenerationRequest(spec.model_id, (("user", texts[index]),), max_tokens=limit,
                                    request_id=f"r{index}", stream=True)
        first, text, ids, tokens, error = None, [], [], None, None
        try:
            for event in service.stream(request):
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "token":
                    first = first if first is not None else time.perf_counter() - offered
                    text.append(event.get("text") or "")
                    ids.append(event.get("token_id"))
                if event.get("type") == "done":
                    tokens = event.get("completion_tokens")
        except Exception as exc:  # noqa: BLE001 - recorded, and it makes the arm incomplete
            error = type(exc).__name__
        out[index] = {"offered_s": offered - base, "first_s": first, "end_s": time.perf_counter() - offered,
                      "tokens": tokens, "token_ids": ids, "text": "".join(text), "error": error}

    try:
        base = time.perf_counter()
        if at_once:
            threads = [threading.Thread(target=ask, args=(i, base)) for i in range(len(texts))]
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
    engine = ready.get("engine") if isinstance(ready.get("engine"), dict) else {}
    return {"arm": arm, "at_once": at_once, "startup_s": startup, "requests": rows,
            "complete": all(r["error"] is None and r["tokens"] and r["first_s"] is not None for r in rows),
            "cpu_blas": health.get("cpu_blas"), "speculation": health.get("speculation"),
            "dense_plan": {k: (engine.get("dense_plan") or {}).get(k) for k in
                           ("kernels", "plan_version", "verify_windows_identical", "speculation", "width")}}


def _rate(row: dict[str, Any]) -> float | None:
    """A request's tokens per second, from its first token to its end."""
    span = row["end_s"] - row["first_s"] if row["first_s"] is not None else 0
    return (row["tokens"] - 1) / span if row["tokens"] and row["tokens"] > 1 and span > 0 else None


def _interval(values: list[float], seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(RESAMPLES))
    return [statistics.fmean(values), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]]


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["token_ids"] and b["token_ids"] and None not in a["token_ids"] + b["token_ids"]:
        return a["token_ids"] == b["token_ids"]
    return a["text"] == b["text"]


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """The rule fixed in BACKLOG SP2 (SP2-H) before the data."""
    problems: list[str] = []
    quiet, load = report.get("quiet", []), report.get("load", [])
    if len(quiet) != QUIET_BLOCKS or len(load) != LOAD_BLOCKS:
        problems.append("missing blocks")
    cells = [cell for block in quiet + load for cell in block]
    if not all(cell["complete"] for cell in cells):
        problems.append("a request failed or returned nothing")
    unequal = sum(not _same(s_row, a_row) for block in quiet + load
                  for s_row, a_row in zip(block[1]["requests"], block[0]["requests"]))
    out: dict[str, Any] = {"answers_unequal": unequal}
    if problems:
        return {**out, "valid": False, "problems": problems, "verdict": "invalid"}
    tps, first, aa_tps, aa_first = [], [], [], []
    for a1, s, a2 in quiet:
        for r1, rs, r2 in zip(a1["requests"], s["requests"], a2["requests"]):
            x1, xs, x2 = _rate(r1), _rate(rs), _rate(r2)
            if None in (x1, xs, x2):
                continue
            tps.append(xs / ((x1 + x2) / 2))
            first.append(rs["first_s"] / ((r1["first_s"] + r2["first_s"]) / 2))
            aa_tps.append(abs(x2 / x1 - 1))
            aa_first.append(abs(r2["first_s"] / r1["first_s"] - 1))
    out["quiet"] = {"prompts": len(tps), "tokens_per_s": _interval(tps), "first": _interval(first),
                    "aa_tokens_per_s": statistics.fmean(aa_tps), "aa_first": statistics.fmean(aa_first)}

    def block_rate(cell):
        rows = cell["requests"]
        return sum(r["tokens"] for r in rows) / max(r["end_s"] for r in rows)

    def block_first(cell):
        return statistics.fmean(r["first_s"] for r in cell["requests"])

    load_tps = [block_rate(s) / ((block_rate(a1) + block_rate(a2)) / 2) for a1, s, a2 in load]
    load_first = [block_first(s) / ((block_first(a1) + block_first(a2)) / 2) for a1, s, a2 in load]
    load_aa_first = statistics.fmean(abs(block_first(a2) / block_first(a1) - 1) for a1, _, a2 in load)
    out["load"] = {"tokens_per_s": [statistics.fmean(load_tps), min(load_tps), max(load_tps)],
                   "first": [statistics.fmean(load_first), min(load_first), max(load_first)],
                   "aa_first": load_aa_first}
    checks = {"answers_equal": unequal == 0,
              "quiet_tokens_per_s": out["quiet"]["tokens_per_s"][1] >= QUIET_BOUND,
              "quiet_first": out["quiet"]["first"][2] <= 1 + out["quiet"]["aa_first"],
              "load_tokens_per_s": min(load_tps) >= LOAD_TOKENS_FLOOR,
              "load_first": max(load_first) <= LOAD_FIRST_BOUND + load_aa_first}
    out.update(valid=True, checks=checks, verdict="pass" if all(checks.values()) else "fail")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dolly", type=Path, help="a directory holding (or to receive) dolly.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analyse", type=Path)
    parser.add_argument("--smoke", action="store_true", help="one tiny block of each, not the protocol's run")
    parser.add_argument("--plan", choices=tuple(PLANS), default="sp2")
    args = parser.parse_args(argv)
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1))
        return 0
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "training"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import mlx.core as mx

    import rl2_headroom

    texts, digest = rl2_headroom.dolly_prompts(args.dolly)
    spec = _spec()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    quiet_blocks, quiet_size, load_blocks, load_size, limit = (
        (1, 2, 1, 2, 16) if args.smoke else (QUIET_BLOCKS, QUIET_SIZE, LOAD_BLOCKS, LOAD_SIZE, LIMIT))
    from ironmule.speculation import POLICY_VERSION

    quiet_first, load_first = PLANS[args.plan]
    report: dict[str, Any] = {"protocol": f"{args.plan}-h" + ("-smoke" if args.smoke else ""), "model": MODEL,
                              "policy_version": POLICY_VERSION,
                              "revision": REVISION, "commit": commit, "dolly_sha256": digest, "limit": limit,
                              "machine": {"machine": platform.machine(), "cpus": os.cpu_count(),
                                          "mlx": mx.__version__},
                              "quiet": [], "load": []}
    shift = 1_000 if args.smoke else 0  # a smoke reads prompts the protocol's run never uses
    for kind, blocks, size, first_index, at_once in (("quiet", quiet_blocks, quiet_size, quiet_first + shift, False),
                                                     ("load", load_blocks, load_size, load_first + shift, True)):
        for b in range(blocks):
            chunk = texts[first_index + b * size:first_index + (b + 1) * size]
            cells = []
            for arm in ORDER:
                cell = run_arm(spec, arm, chunk, at_once, limit)
                cells.append(cell)
                print(json.dumps({kind: b, "arm": arm, "complete": cell["complete"],
                                  "tokens": sum(r["tokens"] or 0 for r in cell["requests"]),
                                  "seconds": round(max(r["end_s"] + r["offered_s"] for r in cell["requests"]), 1),
                                  "speculation": (cell["speculation"] or {}).get("learned", {}).get("accepted")}),
                      flush=True)
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
