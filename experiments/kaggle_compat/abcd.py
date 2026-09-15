"""PORT1: both IronMule axes on one device, the CUDA counterpart of Apple's B39d.

Usage: python abcd.py MODEL_ID REVISION KNOBS_JSON OUT.json [WARMUP REPEATS]

A baseline knobs + interactive, B baseline + throughput, C tuned knobs + interactive,
D tuned + throughput. The workload is `ironmule benchmark`'s (6 strict requests, 48
tokens). Arm order rotates through a balanced Latin square; every arm must reproduce
arm A's tokens, stop reasons and counts, or its ratios are not reported as valid.
"""
import json
import sys
import time

import ironmule
from ironmule import benchmark as bench
from ironmule.tune import load_engine

MODEL_ID, REVISION, KNOBS, OUT = sys.argv[1:5]
WARMUP, REPEATS = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (2, 6)
REQUESTS, MAX_TOKENS = 6, 48
tuned = ironmule.Knobs(**json.loads(KNOBS))
runtimes = {}
for name, knobs in (("baseline", ironmule.BASELINE), ("tuned", tuned)):
    engine, tokenizer = load_engine(MODEL_ID, knobs, revision=REVISION)
    runtimes[name] = ironmule.Runtime(engine, tokenizer, model_id=MODEL_ID)
ARMS = {"A": ("baseline", ironmule.InteractiveMode), "B": ("baseline", ironmule.ThroughputMode),
        "C": ("tuned", ironmule.InteractiveMode), "D": ("tuned", ironmule.ThroughputMode)}
ORDERS = [["A", "B", "C", "D"], ["B", "D", "A", "C"], ["C", "A", "D", "B"], ["D", "C", "B", "A"]]
raw = {arm: [] for arm in ARMS}
reference = None
mismatches = []


def execute(arm, phase, index):
    global reference
    runtime_name, mode = ARMS[arm]
    rt = runtimes[runtime_name]
    requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", REQUESTS, MAX_TOKENS)
    results, snapshot = bench._run(rt, ironmule, mode(), requests)
    outputs = [bench._output_record(r, tuple(rt.backend.eos_ids)) for r in results]
    if reference is None and arm == "A":
        reference = outputs
    if reference is not None and outputs != reference:
        mismatches.append({"arm": arm, "phase": phase, "repeat": index,
                           "diff": bench.output_diff(reference, outputs)[:3]})
    raw[arm].append({"phase": phase, "repeat": index, "outer_wall_ms": snapshot["outer_wall_ms"],
                     "fallbacks": snapshot.get("fallbacks"), "mean_realised_width": snapshot.get("mean_realised_width")})


started = time.time()
execute("A", "reference", -1)
for index in range(WARMUP + REPEATS):
    phase = "warmup" if index < WARMUP else "measure"
    for arm in ORDERS[index % len(ORDERS)]:
        execute(arm, phase, index)
walls = {arm: [s["outer_wall_ms"] for s in raw[arm] if s["phase"] == "measure"] for arm in ARMS}
ratios = {f"{c}/{b}": bench._paired_ratio(walls[c], walls[b])
          for c, b in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("D", "C"))}
report = {"schema": "ironmule.port1-abcd.v1", "model_id": MODEL_ID, "revision": REVISION,
          "tuned_knobs": tuned.as_dict(), "requests": REQUESTS, "max_tokens": MAX_TOKENS,
          "warmup": WARMUP, "repeats": REPEATS, "orders": ORDERS, "token_identity": not mismatches,
          "mismatches": mismatches, "wall_ms": {arm: bench._sample_summary(v) for arm, v in walls.items()},
          "wall_ratios": ratios, "raw": raw, "seconds": time.time() - started,
          "device": str(__import__("mlx.core").core.default_device()), "performance_claim": False}
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1, default=str)
for name, ratio in ratios.items():
    print(f"{name} {ratio['median_ratio']:.4f} [{ratio['ci_low']:.4f}; {ratio['ci_high']:.4f}]")
print("token identity", not mismatches)
for rt in runtimes.values():
    rt.close()
