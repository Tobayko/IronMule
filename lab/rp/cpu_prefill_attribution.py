"""Time native single-sequence prefill inside a fresh head-bound CPU serving audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "gguf", "weights", "prior", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior.read_bytes())
    base = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()
    if args.out.exists() or base != prior["commit"] or digest(args.weights) != prior["binding"]["scratch_thread_head"]["weights_sha256"]:
        raise ValueError("expected a new output, original runtime and original head")
    cpus = set(prior["binding"]["affinity"])
    if not cpus.issubset(os.sched_getaffinity(0)):
        raise ValueError("previous CPU set unavailable")
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", IRONMULE_GGML_GPU_LAYERS="0", CUDA_VISIBLE_DEVICES="", IRONMULE_PILOT="1")
    sys.path.insert(0, str(args.repo.resolve()))
    os.chdir(args.repo)
    from training import serving_trial as st

    if digest(Path(st.__file__)) != prior["evaluator_sha256"]:
        raise ValueError("different serving evaluator")
    measured = []
    trace = None
    prefill = st.ge.GgmlEngine._prefill
    original = st.run_wave

    def timed(engine, ids, seq, start=0):
        if trace is None:
            return prefill(engine, ids, seq, start)
        threads = engine.threads
        t = time.monotonic_ns()
        result = prefill(engine, ids, seq, start)
        trace.append({"tokens": len(ids) - start, "threads": threads, "seconds": (time.monotonic_ns() - t) / 1e9})
        return result

    def wave(*a, **kw):
        nonlocal trace
        trace = []
        try:
            result = original(*a, **kw)
            measured.append({"wall_s": result["wall_s"], "prefill": trace, "output_sha256": result["output_sha256"],
                             "requests": result["requests"], "ttfts_s": result["ttfts_s"]})
            return result
        finally:
            trace = None

    st.ge.GgmlEngine._prefill, st.run_wave = timed, wave
    workload = prior["workload"]
    try:
        run = st.measure(SimpleNamespace(gguf=args.gguf, threads=prior["binding"]["threads"], clients=workload["clients"],
                         seed=workload["seed"], tokens=workload["max_tokens"], waves=workload["waves"], revision=base,
                         require_provider="cpu", thread_candidate=args.weights, profile_dir=None))
    finally:
        st.ge.GgmlEngine._prefill, st.run_wave = prefill, original
    windows = measured[1:]  # standard evaluator begins with a separate warmup
    if (len(windows) != workload["waves"] or run["output_sha256"] != prior["output_sha256"]
            or run["correct"] != run["completed"] or run["runtime_source_sha256"] != prior["runtime_source_sha256"]
            or any(run["binding"].get(k) != v for k, v in prior["binding"].items() if k != "env_sha256")):
        raise ValueError("timed work, serving profile, source or correctness changed")
    record = {"schema": "ironmule.cpu-prefill-attribution.v1", "kind": "measured", "method": "GgmlEngine._prefill",
              "base_commit": base, "evaluator_sha256": prior["evaluator_sha256"], "source_sha256": run["runtime_source_sha256"]["ggml_engine.py"],
              "prior_sha256": digest(args.prior), "instrument_sha256": digest(Path(__file__)), "binding": run["binding"],
              "workload": workload, "measured_choices": run["scratch_measured_decisions"], "correct": run["correct"],
              "tokens": run["tokens"], "qualify_s": run["scratch_qualification"]["search_and_warmup_s"],
              "windows": windows, "inclusive_seconds": sum(c["seconds"] for w in windows for c in w["prefill"]),
              "threads_by_call_count": dict(Counter(c["threads"] for w in windows for c in w["prefill"])),
              "total_window_seconds": sum(w["wall_s"] for w in windows)}
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
