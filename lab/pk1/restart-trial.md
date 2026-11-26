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

A single control command also accepts the two observations directly. It derives the order,
submits it to the configured Codex author, builds the response in the isolated checkout,
collects the actual target pairs, trains the CPU head, publishes a passing source and checks
activation and withdrawal on the reserved target. It then serves matched work with the initial
and learned weights. Use a persistent state directory and the already reserved target config.

```sh
OPENBLAS_NUM_THREADS=1 python -m lab.rp.restart_campaign --config "$TARGET" \
    --observations "$PREVIOUS/order.json" "$CANONICAL/order.json" --checkout "$BASE" \
    --state "$STATE" --author cli --publish --exercise-pilot --wait-seconds 7200
```

The subscription CLI requires working authorization to execute a Codex request. This campaign
used the hosted Codex author. In that environment, select `--author hosted`, let the hosting
worker fulfill the persisted request, and resume the same command. A repeated invocation
validates the original evidence and reuses its recorded author output and completed target
jobs. The RunPod lease and both watchdogs continue to enforce the campaign deadline.

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

Six distinct holdout seeds run in alternating order with a fresh worker on every arm. Each first batch contains 18 independently graded streamed answers. The wall clock starts before CPU profile selection, source checks, worker construction and the HTTP server start. It ends when every answer has completed. The second, warm batch contains 18 fresh factual tasks, followed by 32 ordinary PR57 performance completions with equal output, token counts and stop reasons. An ordinary POST reaches the product listener before any model worker exists. The broker observes up to 250 ms of arrivals. It admits a batch of 18 requests after a 10 ms quiet window when their shortest, median and longest prompt sizes fall in the coarse UTF-8 byte-length bins measured on the local native reference. Repeated text is eligible, and each caller has its own request ID. A different batch size, prompt-length pattern or cold-start setting uses the reference. Initial and trained weights run with identical admitted profiles and confidence rules. The initial weights must select the reference and the fitted weights must select the candidate. Client-observed TTFT therefore includes waiting for the batch, CPU inference, worker construction and actual generation. Promotion requires six cold throughput wins, a median improvement of at least 2%, cold p95 TTFT parity, and no regression in median warm correct completions per second or warm p95 TTFT. The median 32-client performance wall time and TTFT may increase by at most 2%. Peak RSS and observed GPU memory may increase by at most 5% in each pair.

The live endpoint publishes readiness and waits for real incoming requests. Its first arriving batch drives the CPU decision and worker launch. An idle window records zero completions and opens no model worker. Add `--verify-product-client` to explicitly launch a separate client that submits 18 new inventory questions over ordinary HTTP. The verifier checks the actual server-assigned SSE request IDs, distinct process identity, arrival and startup times, reply text and elapsed cost against the passive listener receipt. An independent exercise starts the selected source, withdraws HTTP admission in a separate state, and repeats exactly the same requests after the CPU controller selects the reference. A failed integrity check, worker failure or three matched timing regressions withdraws HTTP use. Each finished arm has its own recoverable JSON record. The two product quality gates retain separate on-target stage caches for restarts. The control host copies the results to `product/`.

The first A100 product attempt stopped inside the stock reference tokenizer's child process. The outer diagnostic labeled its Python `ImportError` as a missing dependency. The inner traceback identified `mlx.core` failing to allocate CUDA managed memory while the reference package initialized. An independent direct tokenizer process produced 8,193 IDs. The second product environment uses a separately attested MLX 0.32.5 CPU distribution for Python imports alongside the same GGUF CUDA backend. Only the four MLX distribution directories are readable through this overlay. Its 322 files are bound by a content manifest. The worker verifies the imported MLX module, generated library, GGUF, mapped CUDA libraries, device and runtime settings. Both sources must pass the original product quality gates and six paired HTTP trials in this environment.

The vocabulary-only tokenizer can complete while GGML reports that CUDA initialization failed in that nested process. The product gate therefore runs each of its original tokenizer, tuning and numerical/factual child scripts in its own restricted process. The ordinary gate controls the order, stock and candidate environments, input, exit status, scoring and thresholds. The trusted broker verifies the unchanged scripts and settings. Each child verifies a CUDA device registered by the native GGML library and the requested full offload before running. Each arm saves these witnesses in `gate-devices-*.json`.

Pass `--product-preflight "$STATE/product-preflight.json"` to the product campaign only after the original nested reference tokenizer has completed successfully on the corrected environment. The trusted control host checks the original failure, source and GATE_IDS digests, exit status, complete ID count, factual signature and package provenance. Receipts are written to a separate `product-v2/` directory. The original decisions remain available in `product/`.

The explicit-width second attempt used the real A100 for both tokenization and the stock numerical reference. The stock reference offloaded 65/65 layers, constructed an unshared 32-sequence context with flash attention off, and aborted in the batched pointer CUDA kernel before scoring. Its strict failure remains in `product-v2/`. The historical K1 server starts differently. The PR57 launch exports the sequence context and leaves the number of sequences unset. The product CLI runs the unchanged numerical and factual gate at the GGUF engine's default of eight sequences. It then calls `_ggml_streams` on the measured settings and `choose_sequences` on the available hardware. The serving worker receives that measured width.

Add `--original-product-width` for this distinct `product-v3/` attempt. Both arms must pass the original gate and then independently select 32 in the native width probe, with an actual 32-sequence measured rate. Every selected setting must match the qualified startup release. The unchanged HTTP checks then apply at serving width 32. Each arm checkpoints after its complete ordinary gate and releases the shared measurement lock. `--stop-after-product-gate reference` ends a run after the historical incumbent's gate, allowing another campaign to verify or schedule its next workload.

Use `--stop-after-product-phase qualify` to release a shared GPU after the six uninterrupted paired HTTP measurements. Use `--stop-after-product-phase exercise` to pause after the rollback exercise. Resuming the same campaign verifies the stored receipts before starting the next pending phase.

On the A100, the original source at `33dc7bf` passed the 16-chunk numerical gate with a likelihood ratio of 0.99405 and a 95% interval of [0.98511, 1.00085]. The upper limit was 1.005. The stock, repeated stock and selected settings each answered 18/18 graded questions correctly. The separate calibration measured 358.49, 696.26, 1,078.66 and 759.78 decode tokens/s at 8, 16, 32 and 64 sequences, respectively. It selected 32 with flash attention and separate KV streams. The numerical/factual gate took 339.3 seconds and the width measurement took 150.0 seconds. These measurements cover the bound W4 binary with a fixed 128-row cuBLAS threshold. The historical K1 serving script also calibrated that threshold during startup. The six HTTP holdouts measure end-user throughput and latency separately. [Original-source quality and width evidence](../autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3/product-v3/)

### Offline replay of the measured A100 experiment

The complete Qwen3-32B Q4_K_M source qualification used the A100-SXM4-80GB with both W4
masks at 0xffffffff. The median paired cold-start saving was 17.805 seconds and the median
speedup was 2.1592. The six paired warm batch-time and p95 first-token ratios were 0.997 and
0.998 at their medians. The factual-first runs finished eighteen graded answers in 30.445 and
30.287 seconds with the incumbent, and 12.436 and 12.348 seconds with the candidate. Their
second factual batches had median wall and p95 ratios of 1.011 and 1.003. Qualification used
920.48 target seconds. At the observed median cold saving, 52 similar restarts would amortize
that qualification alone. The experiment does not measure the frequency of future restarts.

The offline command rebuilds the candidate from the qualification branch and verifies every
timing, output and source receipt in the evidence checkout before refitting on the CPU.

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

### Test a source change to fresh startup calibration over merged PR 60

The complete local PR 60 worker is the reference for the next experiment. Both arms
use its admitted W4 cache and real CPU Pilot. The subsequent sampler still measures
the GPU at every width on every start. Its order and two blind pre-authoring
observations are in [the sampling archive](../autonomous/results/a100-calibration-sampling/README.md).

For the source qualification prepare the immutable merged checkout on both computers
and provide the existing campaign target configuration. The target must already have
the very same active, fresh PR 60 release, kernel, original cache receipt, model and
restricted worker. The campaign consumes the reserved target and snapshots successful
and rejected arms to its control-side state. An unanswered author request remains in
the same state for the hosted Codex worker.

```sh
d=lab/autonomous/results/a100-calibration-sampling
prior=lab/autonomous/results/a100-calibration-stability
r=lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
python -m training.experience_archive build --output /tmp/ironmule-experience.jsonl
python -m lab.rp.followup_campaign --config "$TARGET" --checkout "$MERGED_PR60" \
  --remote-checkout "$REMOTE_MERGED_PR60" --state "$SAMPLING_STATE" \
  --prior-dir "$prior" --forecast "$d/prediction.json" \
  --observations "$d/97351.json" "$d/97352.json" --control "$r/control.json" \
  --weights "$r/startup-head.npz" --archive /tmp/ironmule-experience.jsonl \
  --author hosted --publish --wait-seconds 3600
```

Qualification begins with two original reference processes. Every per-width price in
these A/A processes must agree within 2%. Two separate diagnostic workers count actual
native calls and time them. They verify the complete table delivered by the real engine,
the measured widths and the interval test that admits a shorter measurement. The
candidate must restore the original number of fresh measurements for absent, expired,
wrong-digest and foreign profiles. An independent diagnostic injects a variable delay
into real calls at width eight and verifies withdrawal to the full measurement.

Six counterbalanced mixed pairs and two factual-first pairs then use uninstrumented
GGUF calls. The broker charges its outer verification and CPU profile decision to the
cold clock. Paired dispatch tables, generated output and completed work must match.
Each of the 16 measured prices has a 2% tolerance against the independently measured
original. The original cold, warm, p95, factual and memory requirements remain in effect.
An eligible source can proceed to its own research PR, fresh target admission and actual
rollback exercise. The forecast's 256-start scenario had a negative return after trial
cost. Refining the estimate requires actual serving measurements and a known restart
frequency.

### Run one adaptive calibration experiment on the active PR 60 source

The discovery inputs below include eight historical starts, two prospective price studies,
and two more blind workers. The coordinator recreates the price forecast, source method and
decision from these same inputs. A declined forecast ends locally. An eligible order goes to
the configured author once. On a reserved target a real candidate runs against the current
PR 60 worker, with the identical trained startup head in both arms. The existing Linux
watchdogs, CPU/GPU measurement lock, numerical controls, source-bound PR checks, fresh target
and rollback checks apply. The control host saves each complete worker receipt while polling.
The signed-in subscription CLI can run unattended where its authorization works. The hosted
worker used here consumes the pending request and resumes with the same state directory.

```sh
data=lab/autonomous/results
parent="$data/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3"
probe="$data/a100-calibration-sampling"
python -m training.experience_archive build --output "$KNOWLEDGE_JSONL"
OPENBLAS_NUM_THREADS=1 python -m lab.rp.followup_campaign --config "$TARGET" \
    --checkout "$PR60_CHECKOUT" --remote-checkout "$PR60_TARGET_TREE" --state "$FOLLOWUP_STATE" \
    --prior-dir "$data/a100-calibration-stability" --forecast "$probe/prediction.json" \
    --observations "$probe/97351.json" "$probe/97352.json" --control "$parent/control.json" \
    --weights "$parent/startup-head.npz" --archive "$KNOWLEDGE_JSONL" \
    --author hosted --publish --wait-seconds 7200
```

The local checkout is pinned at `63db7745998383509602601d46ae4bbff0076648`. The corresponding
target export must pass the parent receipt's entire runtime manifest. The target configuration
still names the original reference tree so withdrawal can run the appropriate preceding source.
A fresh machine requires its own hardware measurements, release and binding.
