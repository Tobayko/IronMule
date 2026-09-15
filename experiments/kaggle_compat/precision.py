"""PORT1-C diagnostic: bf16 model computed in float32 on a device that emulates bf16.

Usage: python precision.py MODEL_ID REVISION TUNED_KNOBS_JSON OUT.json

Arms on `ironmule benchmark`'s workload (6 strict requests, 48 tokens), one warmup and
three measured runs each: A16 baseline knobs, interactive, model dtype (bf16); A32 the
same in float32; D32 tuned knobs + throughput in float32. Quality proxy: teacher-forced
top-1 agreement of float32 with bf16 on the bf16 continuations, and generated-token
agreement. float32 changes numerics, so this is never a token-identity claim.
"""
import gc
import json
import statistics as st
import sys
import time

import mlx.core as mx

import ironmule
from ironmule import benchmark as bench
from ironmule.tune import load_engine

MODEL_ID, REVISION, KNOBS, OUT = sys.argv[1:5]
report = {"schema": "ironmule.port1-precision.v1", "model_id": MODEL_ID, "revision": REVISION,
          "device": str(mx.default_device()), "arms": {}, "performance_claim": False}


def arm(name, knobs, mode, dtype):
    engine, tokenizer = load_engine(MODEL_ID, knobs, revision=REVISION)
    if dtype is not None:
        engine.model.set_dtype(dtype)  # floating parameters only; packed weights stay uint32
    rt = ironmule.Runtime(engine, tokenizer, model_id=MODEL_ID)
    walls, outputs = [], None
    for index in range(4):
        requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", 6, 48)
        results, snapshot = bench._run(rt, ironmule, mode(), requests)
        outputs = [[int(t) for t in r.tokens] for r in results]
        if index:
            walls.append(snapshot["outer_wall_ms"])
    prompts = [list(r.prompt_ids) for r in requests]
    report["arms"][name] = {"walls_ms": walls, "median_ms": st.median(walls), "tokens": outputs,
                            "peak_memory_bytes": int(mx.get_peak_memory())}
    print(name, round(st.median(walls)), flush=True)
    return rt, prompts, outputs


def top1(rt, sequences):
    """Argmax at every position of prompt+continuation, one forward per sequence."""
    rows = []
    for ids in sequences:
        logits = rt.engine.model(mx.array([ids]))
        rows.append(mx.argmax(logits[0], axis=-1).tolist())
    return rows


started = time.time()
rt16, prompts, out16 = arm("A16", ironmule.BASELINE, ironmule.InteractiveMode, None)
sequences = [p + o for p, o in zip(prompts, out16)]
forced16 = top1(rt16, sequences)
rt16.close()
del rt16  # the 12B bf16 model must leave the device before the float32 copy loads (run 5 OOM)
gc.collect()
mx.clear_cache()
rt32, _, out32 = arm("A32", ironmule.BASELINE, ironmule.InteractiveMode, mx.float32)
forced32 = top1(rt32, sequences)
rt32.close()
del rt32
gc.collect()
mx.clear_cache()
rtd, _, outd = arm("D32", ironmule.Knobs(**json.loads(KNOBS)), ironmule.ThroughputMode, mx.float32)
rtd.close()

positions = [(a == b) for f16, f32, p in zip(forced16, forced32, prompts)
             for a, b in zip(f16[len(p) - 1:], f32[len(p) - 1:])]


def first_divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


report.update({
    "ratios": {"A32/A16": report["arms"]["A32"]["median_ms"] / report["arms"]["A16"]["median_ms"],
               "D32/A16": report["arms"]["D32"]["median_ms"] / report["arms"]["A16"]["median_ms"]},
    "teacher_forced_top1_agreement": sum(positions) / len(positions),
    "generated_identical_requests": sum(a == b for a, b in zip(out16, out32)),
    "first_divergence_A32": [first_divergence(a, b) for a, b in zip(out16, out32)],
    "D32_equals_A32": outd == out32,
    "seconds": time.time() - started,
})
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1)
print(json.dumps({k: report[k] for k in ("ratios", "teacher_forced_top1_agreement",
                                         "generated_identical_requests", "first_divergence_A32",
                                         "D32_equals_A32")}, indent=1))
