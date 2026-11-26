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

After export, replay the frozen evaluator from the portable archive on an ordinary CPU.
Supply the final campaign journal to include the entire paid lease, development work, idle
time and conservative billing tail in a break-even scenario.

```sh
python -m lab.pk1.restart_payback lab/autonomous/results/a100-restart/"$ORDER_ID" \
    --journal lab/rp/results/campaign-costs.json
```

The calculation uses elapsed seconds through the first completed workload and the full
worker lifetime. It reports each absolute median difference and uses the smaller value to
estimate how many matching restarts would recover the investment. Actual billed savings
also require the worker to release paid capacity earlier.

After a successful source qualification, fit the startup head used by the existing CPU Pilot.

```sh
python -m training.pilot_restart_profiles --order "$STATE/order.json" --checkout "$BASE" \
    --trial "$STATE/trial" --candidate-source "$STATE/authoring/candidate.py" \
    --weights "$STATE/startup-head.npz" --report "$STATE/startup-fit.json"
```

The trainer starts three small networks from independent random weights. The six mixed-first
pairs form one family. Two factual-first pairs form a distinct held-out family and never enter
the gradient or training-margin calculation. The fit records the shipped reference policy,
fixed reuse rule, untrained and trained scores, and CPU fitting time. Reuse can win in both
families. These observations therefore cannot establish superiority over a fixed reuse rule.
The product broker supplies the verified merged release, a pinned checkpoint SHA and a valid
local receipt. Before constructing the worker it calls the existing Pilot's whole-profile
selector. Its decision includes complete checkpoint-load and inference latency. The controller
falls back to the measured reference for a new family, unresolved prediction, unavailable
receipt, different runtime binding or withdrawn source. The recorded previous state represents
random initialization before any measured profile was admitted.
The fit also records a separate weights-only comparison: all three initial networks and the
three trained networks score the exact same admitted hardware, family, complete profiles and
margin threshold. For this comparison the product broker can pass `ablate_weights=True`.
Its receipt records all scores, the selected profile, why any ensemble abstained and the time
to load and evaluate the checkpoint.

### Qualify and use ordinary HTTP startup

After the scoped source PR merges and a fresh native target pair passes, the campaign can run a separate product gate. It uses the normal ProductService HTTP frontend and its ordinary GGUF worker. On the control host, add `--publish --serve-product --product-dedicated --product-port 8232 --product-timeout 2400 --product-lifetime 120` to the restart campaign command. Alternatively supply `--product-reference-url http://127.0.0.1:8231` for a separately verified ordinary same-model server to receive requests beyond the qualified lane. The campaign uses the existing reservation and measurement lock. Both sources must first pass their current on-device product quality gate.

Six distinct holdout seeds run in alternating order with a fresh worker on every arm. Each first batch contains 18 independently graded streamed answers. The wall clock starts before CPU profile selection, source checks, worker construction and the HTTP server start. It ends when every answer has completed. The second, warm batch contains 18 fresh factual tasks, followed by 32 ordinary PR57 performance completions with equal output, token counts and stop reasons. The earlier controller must select the reference. The newly trained controller must select the candidate using the pinned weights, cache and active release. Promotion requires six cold throughput wins, a median improvement of at least 2%, cold p95 TTFT parity, and no regression in median warm correct completions per second or warm p95 TTFT. The median 32-client performance wall time and TTFT may increase by at most 2%. Peak RSS and observed GPU memory may increase by at most 5% in each pair.

The live endpoint receives a fresh factual batch, persists the actual CPU decision, source and streamed work, and serves ordinary requests for the bounded lifetime. An independent exercise starts the selected source, withdraws HTTP admission in a separate state, and repeats exactly the same requests after the CPU controller selects the reference. A failed integrity check, worker failure or three matched timing regressions withdraws HTTP use. Each finished arm has its own recoverable JSON record. The control host copies the results to `product/`.

### Offline replay of the measured A100 experiment

The complete Qwen3-32B Q4_K_M source qualification used the A100-SXM4-80GB with both W4
masks at 0xffffffff. The median paired cold-start saving was 17.805 seconds and the median
speedup was 2.1592. The six paired warm batch-time and p95 first-token ratios were 0.997 and
0.998 at their medians. The factual-first runs finished eighteen graded answers in 30.445 and
30.287 seconds with the incumbent, and 12.436 and 12.348 seconds with the candidate. Their
second factual batches had median wall and p95 ratios of 1.011 and 1.003. Qualification used
920.48 target seconds. At the observed median cold saving, 52 similar restarts would amortize
that qualification alone. The experiment does not measure the frequency of future restarts.

From a checkout containing the evidence and the qualification branch, reconstruct the candidate
and replay all timing, output and source checks before refitting on the CPU:

```sh
order=d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
evidence=lab/autonomous/results/a100-restart/$order
replay=$(mktemp -d)
git worktree add --detach "$replay/base" 33dc7bf544954f6c79a8a15536c0d6f5724b26a2
patch --silent --batch --output "$replay/candidate.py" "$replay/base/ironmule/ggml_engine.py" < "$evidence/candidate.diff"
python -m training.pilot_restart_profiles --order "$evidence/order.json" \
    --checkout "$replay/base" --trial "$evidence" --candidate-source "$replay/candidate.py" \
    --weights "$replay/startup-head.npz" --report "$replay/fit.json"
sha256sum "$replay/startup-head.npz" "$replay/candidate.py"
```

The verified Linux CPU replay produced model SHA-256
`a490979b3ebb77a2bcae6659c7ae44b70a813acb4fb05be4a2da7c19d7a46254` and source SHA-256
`80a41c59bfe8088b305529cca5164134c25d4b5a0ce9572877cb25a2b7d269fd`.
