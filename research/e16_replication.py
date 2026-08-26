"""E16: does the W=4 gain survive real OS process boundaries?

Preregistered at commit a35cb36, research/raw/E16_preregistration.md
(SHA-256 0ec4a1eb39a92458ea196e2768b3d81703138e4f6f8bd0e3b68a57b380f9bcd8).

Every replicate is a new OS process, fully terminated before the next begins. The
parent performs no model work: it spawns, reads one JSON line, and waits for exit.

Only two arms, sequential and grouped W=4, exactly as E15 measured them. No new
width, no new candidate, no controller, no optimisation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

PROCESSES = 5
WARMUP, REPEATS = 2, 3
WIDTH = 4
MAIN_WORKLOADS = ["homogeneous", "heterogeneous", "staggered"]
ALL_WORKLOADS = MAIN_WORKLOADS + ["terse"]
PLANS = ["strict", "reusable"]
SEED = 20260825
THETA = 0.10
GROWTH_LIMIT = 0.10          # A1, A2
REPEAT_DRIFT_LIMIT = 0.03    # A3
CHILD_MEMORY_CEILING = 12 * 1024**3
E15_REFERENCE = {  # for rule 3, taken from research/raw/E15_summary.json
    ("homogeneous", "strict"): 0.1666, ("homogeneous", "reusable"): 0.1713,
    ("heterogeneous", "strict"): 0.1552, ("heterogeneous", "reusable"): 0.1618,
    ("staggered", "strict"): 0.1540, ("staggered", "reusable"): 0.1499,
    ("terse", "strict"): 0.0918, ("terse", "reusable"): 0.1452,
}


def rss_kb() -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def checkpoint(label: str, engine=None) -> dict:
    """RSS plus the finer MLX counters plus proxies for persistent Python structures."""
    import mlx.core as mx
    bodies = 0
    if engine is not None:
        bodies = len(getattr(engine, "_bodies", {}) or {}) or (1 if engine._compiled else 0)
    return {"label": label, "rss_kb": rss_kb(),
            "mx_active": mx.get_active_memory(), "mx_cache": mx.get_cache_memory(),
            "mx_peak": mx.get_peak_memory(),
            "py_blocks": sys.getallocatedblocks(), "gc_counts": list(gc.get_count()),
            "compiled_bodies": bodies}


# --------------------------------------------------------------------------- child

def run_child(spec: dict) -> dict:
    sys.path.insert(0, str(HERE))
    import mlx.core as mx
    from e15_service import (WORKLOADS, build_requests, run_grouped, run_sequential,
                             snapshot)
    from ironmule import bench
    from ironmule.tune import DEFAULT_MODEL, _eos_ids, load_engine
    from e15_service import KNOBS

    workload, plan, index = spec["workload"], spec["plan"], spec["index"]
    marks = [checkpoint("process_start")]

    engine, tok = load_engine(DEFAULT_MODEL, KNOBS)
    eos = _eos_ids(tok)
    marks.append(checkpoint("after_model_load", engine))

    requests, capacity, prefill_ns = build_requests(engine, tok, workload, plan)
    spec_wl = WORKLOADS[workload]
    for r, arrival, cap in zip(requests, spec_wl["arrivals"], spec_wl["caps"]):
        r.arrival_ms, r.cap = arrival, cap
    marks.append(checkpoint("after_prefill", engine))

    for _ in range(WARMUP):
        run_sequential(engine, requests, capacity, eos)
        run_grouped(engine, requests, capacity, eos, WIDTH)
    marks.append(checkpoint("after_warmup", engine))

    base = run_sequential(engine, requests, capacity, eos)
    reference = snapshot(requests, base, capacity, with_hashes=True)

    rng = random.Random(SEED + index)
    runs = []
    for repeat in range(REPEATS):
        arms = ["sequential", f"grouped{WIDTH}"]
        rng.shuffle(arms)
        for arm in arms:
            run = (run_sequential(engine, requests, capacity, eos) if arm == "sequential"
                   else run_grouped(engine, requests, capacity, eos, WIDTH))
            snap = snapshot(requests, run, capacity, with_hashes=(repeat == 0))
            runs.append({"arm": run["strategy"], "repeat": repeat,
                         "wall_ns": run["wall_ns"], "idle_ns": run["idle_ns"],
                         "mean_realised_width": st.mean(x["width_realised"] for x in run["rounds"]),
                         "tokens_generated": sum(r["token_count"] - 1 for r in snap),
                         "requests": snap})
        marks.append(checkpoint(f"after_repeat_{repeat}", engine))

    failures = []
    for entry in runs:
        for got, want in zip(entry["requests"], reference):
            for field in ("tokens", "token_count", "stop_reason"):
                if got[field] != want[field]:
                    failures.append({"kind": field, "arm": entry["arm"],
                                     "repeat": entry["repeat"], "rid": got["rid"]})
                    break
            else:
                if got["kv_hash"] and want["kv_hash"] and got["kv_hash"] != want["kv_hash"]:
                    failures.append({"kind": "kv_state", "arm": entry["arm"],
                                     "repeat": entry["repeat"], "rid": got["rid"]})
    marks.append(checkpoint("process_end", engine))

    return {"workload": workload, "plan": plan, "index": index, "pid": os.getpid(),
            "capacity": capacity, "prefill_ms": [p / 1e6 for p in prefill_ns],
            "reference_tokens": [r["tokens"] for r in reference],
            "reference": reference, "runs": runs, "failures": failures,
            "checkpoints": marks, "environment": bench.environment()}


# -------------------------------------------------------------------------- parent

def spawn(workload: str, plan: str, index: int, timeout: int = 900) -> dict | None:
    """One replicate, one OS process, terminated before the next begins."""
    spec = json.dumps({"workload": workload, "plan": plan, "index": index})
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", spec],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(HERE.parent), env={**os.environ, "PYTHONPATH": str(HERE.parent),
                                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        return {"workload": workload, "plan": plan, "index": index,
                "crashed": True, "stderr": proc.stderr[-2000:]}
    return json.loads(line[2:])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    parser.add_argument("--stage", choices=["pilot", "main"], default="main")
    parser.add_argument("--processes", type=int, default=PROCESSES)
    args = parser.parse_args(argv)

    if args.child:
        print("@@" + json.dumps(run_child(json.loads(args.child)), default=str))
        return 0

    sys.path.insert(0, str(HERE))
    from ironmule import bench
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        print(f"ABORT gpu_busy: {busy}")
        return 2
    env = bench.environment()
    if env["power_source"] != "AC":
        print(f"ABORT power_source={env['power_source']}")
        return 2

    pilot = args.stage == "pilot"
    workloads = ["homogeneous"] if pilot else ALL_WORKLOADS
    plans = ["strict"] if pilot else PLANS
    processes = 2 if pilot else args.processes

    started = time.perf_counter()
    replicates, aborted = [], None
    for workload in workloads:
        for plan in plans:
            for index in range(processes):
                if time.perf_counter() - started > 60 * 60:
                    aborted = "wall_limit"
                    break
                child = spawn(workload, plan, index)
                replicates.append(child)
                if child.get("crashed"):
                    print(f"  CHILD CRASHED {workload}/{plan}#{index}", flush=True)
                    continue
                peak = max(m["mx_peak"] for m in child["checkpoints"])
                if peak > CHILD_MEMORY_CEILING:
                    aborted = "child_memory_ceiling"
                seq = st.median(r["wall_ns"] for r in child["runs"] if r["arm"] == "sequential")
                grp = st.median(r["wall_ns"] for r in child["runs"] if r["arm"] != "sequential")
                print(f"  {workload:14s} {plan:8s} #{index}  pid {child['pid']}  "
                      f"G {1-grp/seq:+.4f}  peak {peak/1e9:.2f} GB  "
                      f"fail {len(child['failures'])}", flush=True)
            if aborted:
                break
        if aborted:
            break

    payload = {"experiment": "E16", "stage": args.stage, "aborted": aborted,
               "replicates": replicates, "processes_per_condition": processes,
               "preregistration_sha256": "0ec4a1eb39a92458ea196e2768b3d81703138e4f6f8bd0e3b68a57b380f9bcd8",
               "prereg_commit": "a35cb36cb6475291a0dd601e7f7c96b9935e54c9",
               "wall_seconds": time.perf_counter() - started, "environment": env}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"E16_results_{args.stage}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {path} ({payload['wall_seconds']:.0f}s, aborted={aborted})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
