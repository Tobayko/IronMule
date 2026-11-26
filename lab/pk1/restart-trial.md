# Reusing locally qualified W4 measurements on restart

This experiment measures the time from a fresh Python worker process to delivery of its first
complete, correct 32-client mixed wave on Qwen3-32B Q4_K_M. It uses the canonical PR 57
source with the W4 receipt writer fix. The language model, native GGML libraries, W4 library,
dispatch table, flags and GPU stay fixed. It also measures a second mixed wave on the same
engine and grades eighteen independent factual answers.

The planner reads both K traces from the earlier pod 37 archive. Between the corresponding
startup trace markers, W4 width selection took 18.064 and 18.088 seconds. PK2 calibration took
18.642 and 18.631 seconds in those same starts. The current deployment explicitly pins its
cuBLAS threshold at 128 rows. Two fresh K diagnostics on the current GPU independently chose
identical Q4 and Q6 width masks for the same source and binary. The planner ranks startup
phases with retained comparable decisions and sends the W4 phase to the coding worker.
The estimated opportunity concerns a restart after an already qualified deployment.

```sh
python -m lab.pk1.restart_order plan --canonical "$CANONICAL/order.json" \
    --previous "$PREVIOUS/order.json" --checkout "$BASE" --output "$STATE/order.json"
python -m lab.pk1.restart_order author --order "$STATE/order.json" --checkout "$BASE" \
    --output "$STATE/authoring/candidate.py"
```

Each input order requires the adjacent `order.profile.json` from the same actual capture.
The author request contains the source-addressed constructor, its existing calibration
method, measured phase observations and required receipt contract. A missing hosted reply
leaves an immutable request under `$STATE/authoring/author`. An authorized Codex worker
records its response for that request; rerunning `author` produces the candidate Python source.

On the owned Linux RunPod, stage the trusted framework, immutable reference checkout, model,
bindings, exact local W4 kernel and table, dedicated worker UID and restricted worker directory.
The JSON configuration uses the existing reserved lease and the paths documented for the native
campaign. The controller can stage without admitting GPU work. Run measurement with exclusive
access to the GPU and its CPU allocation.

On this image, point the restart broker's `LD_LIBRARY_PATH` first to the installed NVIDIA
wheel directories `nvidia/cublas/lib` and `nvidia/cuda_runtime/lib`. They are under
`/usr/local/lib/python3.11/dist-packages`. The startup worker launches an ordinary Python
interpreter with those paths. After serving it identifies the mapped inodes of cuBLAS,
cuBLASLt and the CUDA runtime, hashes their bytes and compares them to the successful
incumbent diagnostic. The original system-toolkit stack aborted during incumbent calibration.

```sh
python -m lab.rp.restart_campaign --config "$TARGET" --order "$STATE/order.json" \
    --state "$STATE" --checkout "$BASE" --stage-only
python -m lab.rp.restart_campaign --config "$TARGET" --order "$STATE/order.json" \
    --state "$STATE" --checkout "$BASE" --publish --wait-seconds 3600
```

On a fresh target measurement, two unmodified A/A starts must first complete the workload
with identical outputs and effective masks. An independent broker derives the restart receipt
from the first graded observation. The receipt binds the model and its mounted-file identity,
actual CUDA and native runtime, both kernel artifacts, CPU affinity, engine geometry, flags,
original calibration function and returned answers. Fresh immutable copies are mounted
read-only for each child. The optional source change requires the independently selected
`IRONMULE_KG_RESTART_RECEIPT` and matching `IRONMULE_KG_RESTART_SHA256`. A stale, separately
hash-pinned receipt and an absent receipt must both trigger the original measurement.
The original explicit kernel scope and disabled-mask path retain their precedence.

Six fresh process pairs alternate order and use different fixed seeds. Process ticks include
interpreter startup and model initialization. The first-wave clock stops after all streams
have rendered their final text. The broker records the total worker lifetime as an independent
timing check. Both arms must complete the same 32 requests, with matching generated IDs,
stops, rendered text and actual scheduling counters. The evaluator grades the eighteen
factual answers outside the worker. Every cold pair must improve. Median cold speedup must
reach 1.02 and exceed the relative A/A spread. Median warm time and 95th-percentile first-token
ratios must be at most 1.02. Host and observed device memory ratios must be at most 1.05.
Two additional fresh process pairs reverse execution order. Their first admitted requests are
eighteen seeded factual questions, followed immediately by eighteen different factual questions
on the warmed worker. The evaluator regrades all answers, verifies equal output tokens and
measures cold time through the last completed fact and cold p95 through the first visible tokens.
Both cold factual pairs must improve, median speedup must reach 1.02 and the median warmed
factual latency and p95 ratios must stay within 1.02. Warm factual latency and p95 from the six
original mixed-first pairs must satisfy that same threshold.
A startup gain applies only to this repeated-deployment workload. Warm serving and HTTP
throughput require their own measurements.

Reinvoking the campaign reads the saved order, authorship, A/A, fixtures and finished arms.
Each published arm is complete and hash-checked. The control host can replay the gate from the
captured measurements without a GPU. Source publication, target activation, withdrawal and
a fresh-reference rollback exercise use the scoped release commands. Inspect the target
runner entry points with `python -m lab.pk1.restart_release --help`. The current lease,
watchdogs and cost journal remain authoritative throughout.
