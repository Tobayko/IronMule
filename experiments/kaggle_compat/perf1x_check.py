"""PERF1-X: does `native` load and serve through IronMule's engine on a card its weights nearly fill?

Usage: python perf1x_check.py MODEL_ID REVISION OUT.json

Loads through `load_engine(..., compute_dtype="native")` with the untuned knobs, records the
device's reported info, whether `head_skip_prefill` was switched on for the head's float16
copy and why (`native_admission`), then serves `ironmule benchmark`'s workload once (6 strict
requests x 48 tokens, interactive mode) and records the tokens, the wall and the peak memory.
A failure is recorded with its exception class and message head, not raised.
"""
import json
import sys
import time

import mlx.core as mx

import ironmule
from ironmule import benchmark as bench
from ironmule.tune import load_engine

MODEL_ID, REVISION, OUT = sys.argv[1:4]
started = time.time()
report = {"schema": "ironmule.perf1x-check.v1", "model_id": MODEL_ID, "revision": REVISION,
          "device_info": dict(mx.device_info()), "performance_claim": False}
try:
    engine, tokenizer = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION, compute_dtype="native")
    report["native_admission"] = engine.native_admission
    report["head_skip_prefill"] = engine.knobs.head_skip_prefill
    with ironmule.Runtime(engine, tokenizer, model_id=MODEL_ID) as rt:
        requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", 6, 48)
        results, snapshot = bench._run(rt, ironmule, ironmule.InteractiveMode(), requests)
        report["tokens"] = [[int(t) for t in r.tokens] for r in results]
        report["outer_wall_ms"] = snapshot["outer_wall_ms"]
    report["ok"] = True
except Exception as exc:  # noqa: BLE001 - the harness records, the notebook judges
    report["ok"] = False
    report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
report["peak_memory_bytes"] = int(mx.get_peak_memory())
report["seconds"] = round(time.time() - started, 1)
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1, default=str)
print(json.dumps({k: v for k, v in report.items() if k != "tokens"}, default=str))
