# Experimental CPU Pilot startup head for W4 on Linux

This head selects a complete, locally measured execution profile before native graph capture.
The reference is Qwen3-32B on its individually bound GPU, with the calibrated W4 library
and dispatch table, warm-up, padding, prefill settings and CPU Pilot step costs. Selection
scores the measured stock, Q4-only and Q4+Q6 paths. A profile can serve only with a local
whole-serving qualification. The first A100 receipt recognizes the complete incumbent.
An alternate bundle enters the eligible set only after a six-pair qualification on the same
target and mixed serving workload. Each request in the selected
engine still passes through its normal serving Pilot. The implementation uses the existing
Pilot's NumPy forward pass on CPU for each of three independently initialized heads.

Fit against the exact native order, adjacent CUPTI profile and W4 calibration receipt.
The fit records the receipt identity, held-out timings, checkpoint and prediction changes
when the width-16 group is added. It keeps every 1–8-row measurement in one tile family,
calibrates margins at 24 and holds out width 32. All 6,324 individual kernel attempts of
the first A100 calibration remain diagnostic inputs to the underlying dispatch table.
The startup head learns from eleven end-to-end native step medians for each of its three profiles.

```sh
OPENBLAS_NUM_THREADS=1 python -m training.pilot_w4_profiles fit \
  --order "$STATE/order.json" --receipt "$W4_CACHE/measurement.json" \
  --weights "$STATE/startup.npz" --report "$STATE/startup-fit.json"
```

PR 57's startup timing puts successive token positions of **one sequence** after a 64-token
prefix. On the canonical A100 sample, all eleven measurements favor Q4+Q6. Held-out
32-row times were 61.68 ms stock, 38.81 ms Q4 and 33.64 ms Q4+Q6. The three learned heads
disagreed at width 32 and abstained. The normal 32-client serving geometry also differs from
these training examples. A GPU decision using this first checkpoint will preserve the local
incumbent for independent-stream serving. These data establish no serving gain over PR 57.

An independently budgeted follow-up measures 1, 8, 16 and 32 separate streams. Acquire
exclusive use of the same lease after the independent qualification completes. Source the
target's runner environment and run in its trusted framework. `BASE` is the immutable
reference checkout. `WORKER_ROOT` is the restricted RunPod work directory. The incumbent's
`calibration.json` must already exist from this order. The command writes six resumable
worker reports: stock, Q4, Q4+Q6, then the reverse order. It times eight complete native
steps per cell after warm-up and verifies every greedy token across arms and repetitions.

```sh
python -m training.pilot_w4_profiles collect \
  --order "$STATE/order.json" --receipt "$W4_CACHE/measurement.json" \
  --state "$STATE/context-probe" --checkout "$BASE" --gguf "$GGUF" \
  --worker-root "$WORKER_ROOT" --reference-lib "$W4_CACHE/kernel.so" \
  --table "$W4_CACHE/dispatch.table" --calibration "$STATE/trial/calibration.json"

OPENBLAS_NUM_THREADS=1 python -m training.pilot_w4_profiles fit \
  --order "$STATE/order.json" --receipt "$W4_CACHE/measurement.json" \
  --contexts "$STATE/context-probe/contexts.json" --weights "$STATE/stream.npz" \
  --report "$STATE/stream-fit.json"
```

The context fit groups both orders and all widths by prompt family. It excludes changed
tokens, overlapping winner intervals and the identical one-stream synthetic short cases.
It first learns the 32- and 96-token groups, adds 224, calibrates on 512 and holds out the
actual serving prompt mix for seed 92741. The check measures clean native decode in these
states and preserves the separate complete-serving gate and ordinary Pilot overhead.

```sh
python -m training.pilot_w4_profiles serve \
  --order "$STATE/order.json" --receipt "$W4_CACHE/measurement.json" \
  --weights "$STATE/stream.npz" --state "$STATE/stream-serve" --gguf "$GGUF" \
  --checkout "$BASE" --worker-root "$WORKER_ROOT" --reference-lib "$W4_CACHE/kernel.so" \
  --table "$W4_CACHE/dispatch.table" --calibration "$STATE/trial/calibration.json" \
  --reference-arm "$STATE/trial/ref-0.json" --seed 92741
```

The serving runner revalidates the exact hardware, software, model and whole-profile receipt.
It records the proposal, margin, eligible profiles, checkpoint-and-selection time, actual
mask registers, full delivered outputs, wall time, TTFT and factual score. Unknown workloads,
different bindings and unqualified proposals cannot activate another W4 bundle. A binding
change requires a new local reference. The outer campaign budgets and schedules target jobs.

If a new context fit proposes stock or Q4-only, use `qualify --name stock` or `qualify --name q4`
with the same `--order`, `--receipt`, `--calibration`, model, checkout and worker arguments as
`collect`, and a fresh `--state` directory. This compares two complete bundles in six pairs
using the frozen native serving evaluator. The reader checks the two expected mask registers
separately, then applies the existing output, quality, wall-time, TTFT and memory gates.
A subsequent experimental `serve --qualified DIRECTORY` replays the underlying measurements
before the startup head may select that entire alternate profile.

The separate repeated-startup experiment uses the same CPU Pilot to choose between qualified
source profiles for a fresh worker. Its model trains on complete 32-client mixed starts and
holds out the 18-client factual starts. See [training, source and measurement replay](../lab/pk1/restart-trial.md).
