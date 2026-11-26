# Training and measuring the agent

For GGUF device placement, train and replay the C++17 controller through `ironmule hardware calibrate`. The same measurement loop runs in a live server during idle periods. [Hardware controller training and RunPod/Kaggle estimates](hardware-placement.md)

This folder holds what trains, simulates and measures the IronMule agent ([docs/AGENT.md](../docs/AGENT.md)).
The agent itself ships in `ironmule_product/`. Nothing here is imported by the product.

| File | What it does |
| :-- | :-- |
| `ag1_online.py` | The agent, unchanged, against a simulated server priced by a machine's fitted costs. Live traces, idle replays and qualification run on a simulated clock. `--prompts` adds real prompt texts with one limit for all (AG2). |
| `rl2_headroom.py` | The hardware harness. It runs the product's own service with the agent attached, records the online phase, then compares the default, the fixed rule and the frozen agent on held-out traces. `--plan ag1 --profile NAME` picks a protocol, `--write-kernel` writes a Kaggle kernel, and `--analyse` judges a report. |
| `ag4_routing.py` | AG4 step 2: the product's router between a single path and a stacked engine on one card, against each path alone, priced by a hardware measurement. |
| `rl4_h9_replay.py` | Trains two dispatch checkpoints from zero, selects one on validation traffic, and evaluates both against the fixed arms, simple rules and the published checkpoint on unseen quiet, burst and spike traces. |
| `rl4_agent.py` | RL4, the offline REINFORCE agent whose policy format the product's learned dispatch (LD1) and the agent share. |

Their tests are in [tests/agent/](../tests/agent/). The tests run without a model or GPU, also in CI:

```bash
python -m pytest tests/agent
```

## Replaying the H9 dispatch training

Run these from the repository root. Each command trains two controllers from zero for 16,000 episodes per seed on one CPU core. They use simulated requests and the published timing coefficients for the two-T4 Qwen3-1.7B H9 pool.

```sh
python -m training.rl4_h9_replay --out rl4-replay.json
python -m training.rl4_h9_replay --width-followup --out rl6-replay.json
```

Sixty validation traces per phase select one seed. Each frozen controller then runs on 128 unseen traces per phase, sharing the same arrivals with every baseline. The follow-up reads the first study's archived checkpoint, adds 16 trainable group-width weights, and uses a new test set. The objective adds mean time to first token and mean time to completion.

The width controller selected a cap of two on every held-out decision with at least two requests waiting. During spikes this cut simulated cost by 6.01% against the first study's frozen controller, while mean time to first token rose from 8.32 to 11.15 simulated seconds. Against the best fixed policy in each phase, cost rose 0.29% in quiet traffic and fell 0.62% in bursts and 0.88% in spikes. Both studies saved their checkpoints for replay in the simulator. Hardware qualification would require fresh on-device measurements. [First study](../evidence/runtime/rl4-h9-holdout.json) · [Width follow-up](../evidence/runtime/rl6-h9-width-holdout.json)

## Forecasting the next native workload

Use the normal local GGML bindings and a SmolLM2-135M-Instruct Q4_K_M GGUF. The study opens the native engine on
two CPU threads, checks that widths four and eight produce identical calibration output, and observes 864 real
requests. Each set of eight arrives together. The three workloads have short prompts, longer distinct prompts or
longer prompts with a shared session prefix. All have a three-token generation limit. The controller sees timings
and token counts. Phase labels and copy targets stay in the study.

```sh
IRONMULE_GGML_GPU_LAYERS=0 IRONMULE_GGML_THREADS=2 IRONMULE_GGML_PREFIX_REUSE=1 IRONMULE_PILOT=0 \
  python training/ag10_replay_study.py --gguf /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --state-dir ./new-ag10-state --output ./ag10-replay.json
```

The sampler forecast all 18 held-out blocks before they arrived. It found matching history for eight.
Across four traffic measurements, mean absolute log₂ forecast error fell from 1.010 to 0.976 (3.33%).
Short requests improved from 0.962 to 0.786 and distinct long prompts from 0.734 to 0.723. Follow-ups with
shared prefixes worsened from 1.333 to 1.419. All 576 held-out first words passed the copy check. All requests
ran at width four, providing 1,401 measured processing steps. The agent rebuilt to width eight after the last
block. Measuring the serving effect of that switch requires subsequent traffic. [Per-block evidence and fitted weights](../evidence/runtime/smollm2-135m-cpu-dream-replay.json)

### Serving the proposed width and testing its rollback

The continuation read the exact 864-request checkpoint on the same CPU, verified the model, libraries, engine,
controller and calibration outputs, and restored four slots through the normal restart guard for an unfinished
trial. After 32 fresh requests with shared prefixes, the controller chose eight slots again. Three independent
contexts received identical token IDs in groups of eight. Each block rotated the order of the adaptive arm,
fixed four slots and fixed eight slots. Workload labels served only to score the results.

| Period | Requests per arm | Adaptive tok/s · p95 TTFT | Fixed 4 tok/s · p95 TTFT | Fixed 8 tok/s · p95 TTFT |
| :-- | --: | --: | --: | --: |
| Two blocks actually served at eight slots | 64 | 32.21 · 1,323 ms | 93.11 · 229 ms | 68.05 · 306 ms |
| Four subsequent phases, adaptive arm back at four slots | 384 | 22.72 · 1,900 ms | 21.93 · 2,028 ms | 21.92 · 1,920 ms |

Rates divide total generated tokens by active wave time. The fresh adaptive context lost its cached prefixes on
rebuild. Its two width switches added another 1.697 seconds outside active requests. All arms matched the complete
outputs on both eight-slot trial blocks. After 94 measured updates at eight slots, the corrected latency model
withdrew the trial. The next 384 requests rotated through short, document, cached follow-up, then short prompts.
All arms copied the correct first word on 383 of 384. The adaptive and fixed-four outputs matched on every block.
Fixed eight differed on one of twelve blocks, with the correct first words. The adaptive arm served those 384
requests at the same width as fixed four. Its measured difference from fixed four does not establish a scheduling
gain. The report includes request-level time to first token and output digests. [Continuation evidence](../evidence/runtime/smollm2-135m-cpu-dream-continuation.json)

On the matching machine and runtime, replay the saved study state into a **new** directory. The archived state
belongs to this experiment and includes only numeric traffic, fitted weights, decisions and hardware binding.

```sh
IRONMULE_GGML_GPU_LAYERS=0 IRONMULE_GGML_THREADS=2 IRONMULE_GGML_PREFIX_REUSE=1 IRONMULE_PILOT=0 \
  python training/ag10_replay_followup.py --gguf /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prior-report evidence/runtime/smollm2-135m-cpu-dream-replay.json \
  --prior-state evidence/runtime/ag10-dream-checkpoint --state-dir ./new-ag10-continuation \
  --output ./ag10-continuation.json
```

## Testing grouped prompt processing on recorded HTTP traffic

The optional selector reads a single completed IronMule prefill HTTP study. The two eligible arms use the same
GGUF and separate KV streams. One processes prompts individually, the other can group them in a native batch.

```sh
python -m training.prefill_selection prefill-http.json --output prefill-cross-validation.json
```

A fold holds out an entire server repeat, including its measured concurrency levels. The selector groups
observations by device and model binding, active sequence layout, concurrency, output limit and the histogram
of prompt lengths in powers of two. To select grouping, at least two other matching repeats must have run the
two arms in both orders. Every such repeat must show actual native grouping, identical greedy output and stops,
an informative baseline for each factual category, preservation of every correct answer, at least 3% more
generated tokens per second and p95 time to first token no later than split processing.
The held-out repeat records the selected action and its own measurements. A failed held-out prediction and a
newly lost factual answer remain in the report. A repeated timed prompt across folds invalidates that workload.

The producer can emit a hash of the loaded worker libraries, driver, device and visible execution settings.
Historical reports with verified in-run GPU placement and missing worker hashes can support a descriptive
comparison within their one run. A promising result calls for a new bound local holdout. This offline command
reads no other engine's measurements and installs no serving rules.

On the NVIDIA L40S, Qwen3-32B Q4_K_M completed one verified split-stream round at concurrency 1, 8 and 32.
It generated 37.74, 113.52 and 161.13 tokens/s, with p95 times to first token of 0.240, 1.148 and 4.406 seconds.
All 18 factual checks passed at each level. The grouped arm started with a different KV layout and produced zero
comparable measurements. The recorded decision is insufficient evidence. [Verified cells and rejection](../evidence/runtime/qwen3-32b-l40s-prefill-selection.json)

## The workflow

Every step below follows [AGENTS.md](../AGENTS.md), "Performance and evidence". A result counts only when it was
measured on the hardware it claims, and its protocol was fixed before its data.

1. **Write the protocol first.** Write it in [docs/BACKLOG.md](../docs/BACKLOG.md), under the agent's entry (AG1,
   AG2): what changes, the profile, decisions, kill rule and expected session hours. A change after data is
   labelled post hoc.
2. **Simulate.** A gate passes in simulation before hardware is spent, for example S-LP:

   ```bash
   python training/ag1_online.py --traces 300 --out sim.json
   ```

   ```bash
   python training/ag1_online.py --traces 300 --prompts prompts.jsonl --limit 512 --out sim.json
   ```

   `--profile v-g12` mirrors a hardware protocol's per-card sizes for a dry run of its online phase.

   A prompts file has one `{"prompt": ..., "tokens": ...}` per line. It is built from a public dataset, kept
   outside the repository, and never committed.
3. **Smoke locally.** A smoke checks the tool and measures nothing:

   ```bash
   python training/rl2_headroom.py --plan ag1 --profile cpu-lp --smoke --model mlx-community/SmolLM2-135M-Instruct-8bit --out smoke.json
   ```

4. **Run on Kaggle.** Push the commit first, then write a kernel pinned to its full SHA:

   ```bash
   python training/rl2_headroom.py --write-kernel kernel/ --owner tobayko --commit FULL_SHA --plan ag1 --profile t4-g12-v
   ```

   ```bash
   kaggle kernels push -p kernel/
   ```

   For a smoke kernel, add `'--smoke', '--model', MODEL` to the last line of `kernel/run.py`, and add `-smoke` to
   the id in `kernel/kernel-metadata.json`. Kaggle runs spend the owner's weekly GPU quota. State the expected
   session hours first, and poll the status at most every two minutes.
5. **Analyse and record.** Read a report with the harness's own judge:

   ```bash
   python -I training/rl2_headroom.py --analyse rl2_headroom_report.json
   ```

   Downloaded output goes into a new, empty directory outside the repository and is read with `python -I`.
   Record the verdict under the entry, and note that the research repository's ledger entry is still owed.
   Archive a finished kernel's source and output in `tobayko/ironmule-notebook-archive` before deleting it.

## Profiles (`--plan ag1 --profile ...`)

| Profile | Machine and model | What it tests |
| :-- | :-- | :-- |
| `t4`, `t4b`-`t4d` | T4, Qwen3 1.7B, tuned engine | AG1 V-V4: order by limit and width |
| `cpu`, `cpu2`, `cpu3` | CPU, SmolLM2-135M | AG1 V-CPU, V5-CPU: order only (capacity 1) |
| `cpu-lp` | CPU, SmolLM2-135M, Dolly prompts | AG2 smoke path (real prompts, one limit) |
| `t4-g12-lp` | 2 x T4, Gemma 3 12B `native`, Dolly | AG2 gate (smoke) |
| `t4-g12-v` | 2 x T4, Gemma 3 12B `native`, Dolly, limit 256 | AG2 V-G12 (replaced before its data by KB2) |

A new protocol gets a new profile on fresh seeds. A test checks that no two profiles share a seed.

## Fitting the Pilot thread head from paired CPU decode

The GGUF per-step Pilot can test a freshly trained thread head in its existing idle qualification loop.
This experiment covers plain decode on the local GGML CPU backend. Each action changes the thread count
of an otherwise fixed decode call. Context, row count, the shipped Pilot's base decision and foreign CPU
load identify the regime. Other decisions continue through their existing qualified paths.

From an activated IronMule environment, source the CPU backend's env file and set `MODEL` to the pinned
SmolLM2-135M-Instruct-Q4_K_M GGUF described in [hardware-placement.md](hardware-placement.md).
Use a directory on local storage for the following files.

```bash
export OPENBLAS_NUM_THREADS=1
export IRONMULE_GGML_GPU_LAYERS=0
mkdir -p pilot-run
python -m training.pilot_threads collect --gguf "$MODEL" --state pilot-run/state --trials pilot-run/trials.json
python -m training.pilot_threads fit --trials pilot-run/trials.json --weights pilot-run/threads.npz --report pilot-run/replay.json
python -m training.pilot_threads exercise --gguf "$MODEL" --state pilot-run/state --weights pilot-run/threads.npz --report pilot-run/serving.json
```

Collection includes both orders of the available thread profiles on the same inputs, with eight greedy tokens per
sequence and a digest of the generated IDs. Interrupted or unmatched series stay in the file and are excluded
from fitting. A second collection with `--extend` adds a pass, preserving the first, on the same bound engine.
The fit uses one CPU process, seed 20261126, two 128-unit tanh layers and eight scores for log speedup relative to
the shipped Pilot's measured action. For four cores, some of the eight thread fractions denote the same action.
Context bins 1–3 provide the first training sample, bins 4–5 add experience and bins 6–7 are held out. All widths,
prompt repeats and arms for a bin stay together. The report compares the resulting frozen controller to the
previous fit, the shipped Pilot, one thread and all threads. These are decode times for fixed work. Replay does
not measure first-token latency or qualify deployment.

`exercise` runs the native engine and its local idle gate on a bounded stream of requests, then times four
counterbalanced complete generations. Its JSON includes first-token time, elapsed time, output equality,
the number of qualified thread decisions and search plus warm-up cost. For a server configured with the
same model, native binary, CPU and engine geometry, set `IRONMULE_PILOT_THREAD_CANDIDATE` to the checkpoint
path and `IRONMULE_PILOT_LEARN=1`. During plain decode the candidate chooses between its locally qualified action
and the shipped Pilot's thread count. A foreign checkpoint cannot propose actions. A fresh process rechecks
the saved suggestions. Removing `IRONMULE_PILOT_THREAD_CANDIDATE` at restart restores the standard local
thread learner. The GGUF weights and GPU kernels are separate from the controller checkpoint.

Use `collect --slots 4` to measure the four-slot geometry used by the code evaluator. It samples one
and four simultaneous decode rows. During a follow-up on the same machine, `--extend --missing-only`
prioritizes incomplete context/width pairs. `exercise --foreign-workers 1` or `2` tests externally
contended CPU operation. Every worker terminates after the bounded audit. To profile real waves under a
new head and dispatch its resulting development order, pass the four-slot checkpoint to
`python -m training.optimization_run --gguf "$MODEL" --state pilot-run/optimization --run-id scratch --thread-candidate pilot-run/threads4.npz`.
The evaluator records the measured scratch choices, the cost of fresh target qualification and the output
check alongside the code trial. [Evaluation and activation](../docs/OPTIMIZATION_METHODS.md)

### Start an offline experiment from GGUF serving demand

Set `IRONMULE_EXPERIMENT_QUEUE` to a private directory before starting a GGUF server.
Each cohort of 32 fresh completed requests can produce one ticket during an idle spell.
The ticket contains the counts of prompts below and above the 96-token boundary
(96 enters the document bin), the model digest, an opaque ticket ID and the producer's
boot-ID digest. The publisher uses the width agent's existing memory when available.
In single-profile mode it keeps 32 length classes in memory. Profiling occurs only on
synthetic requests in the evaluator. The server keeps delivering answers with no evaluator attached.

Mount the same private queue on a dedicated Linux evaluation host with the same GGUF.
Use a filesystem supporting advisory locks and atomic replacement. Start one bounded consumer.
It can also run on the producer after the GGUF service has shut down.

```sh
python -m training.demand_worker --queue "$IRONMULE_EXPERIMENT_QUEUE" --state "$EVAL_STATE" \
  --gguf "$GGUF" --provider cpu --cores 2 --wait 60 --seconds 300
```

The consumer waits up to 60 seconds for one ticket, then gives its low-priority experiment
two eligible CPUs and a 300-second wall limit. It selects the hosted Codex mailbox to avoid
metered API calls. On the originating Linux machine the active publisher defers evaluation.
An evaluator seeing a new queue-enabled service on its own machine preempts the experiment.
Dedicate the evaluator to the trial because services using another queue are outside this guard.
`awaiting_author` leaves the same ticket in place. After the host returns a bound reply,
repeat the consumer command to resume paired qualification. This consumer runs on existing
hardware and uses the nonpublishing evaluation mode. [Publishing a qualified result](../docs/OPTIMIZATION_METHODS.md)

On the Xeon Platinum 8573C, a live product bridge using SmolLM2-135M-Instruct Q4_K_M answered
32 of 32 synthetic copy requests correctly. The idle publisher counted 16 short and 16 document
prompts. The consumer deferred while that bridge was open. On shutdown the offline evaluation
completed 16 of 16 discovery tasks, planned with the observed phase counts and produced a
development order in 9.62 seconds through the author handoff. The hosted Codex worker read the archived
list-copy rejection and returned the original tokenizer. The evaluator resumed from its pinned checkout,
recorded the no-change decision and consumed the ticket. [Hosted reply](../lab/autonomous/results/cpu-serving-demand/author-response.json) · [Measured ticket and order](../lab/autonomous/results/cpu-serving-demand/summary.json)

### Uncertainty in the scratch CPU thread decision

The opt-in scratch head acts on plain decode with equal-band contexts. Its status includes the selected thread count, host/runtime and model binding, context and slot capacities, actual sequence rows, absence of prompt, draft or padding rows, eligible thread counts and fallback reason. A new process requalifies each regime. The existing four paired windows must pass the output and timing gates. The head also compares its current selection cost with the smallest absolute time saved in any window, after deducting that window's entire difference between repeated reference timings. Both values are converted to microseconds per decode step. Where the reference is the only eligible profile, uncertainty is recorded as unknown. Check the serving trial's wall and first-visible-token times for the cost of the entire path.

Audit held-out choices against the original paired timings without refitting the weights.

```sh
python -m training.pilot_threads audit \
  --trials evidence/runtime/cpu-pilot-four-slot-trials.json \
  --weights evidence/runtime/cpu-pilot-four-slot-candidate.npz \
  --report four-slot-uncertainty.json
```

Of six observed held-out cells in two context families, two had complete matched measurements. In one, another thread count beat the selected profile in both repeats with disjoint times. In the other, the timings overlapped. This sample cannot estimate a calibrated probability of picking the fastest profile. The archive keeps the coverage denominator, unresolved outcomes and engineering bounds separately. [Held-out uncertainty audit](../evidence/runtime/cpu-pilot-four-slot-uncertainty.json)

### Updating a running CPU worker from archived measurements

An offline update imports the repository's measured thread profiles and an incoming trial file of
the same engine binding. Set `STATE` to the store root used by the GGUF worker. This command performs
a single bounded fit on one CPU process. Its published checkpoint becomes a proposal for that worker.

```bash
python -m training.pilot_archive --incoming pilot-run/trials.json --state "$STATE"
```

To replay an exported JSONL archive instead, pass `--archive experience.jsonl --binding BINDING`.
The worker reports its binding under `pilot.thread_learning.binding`. The local SQLite journal
retains observed and unresolved cells with their source digests. Selection keeps up to 16 recent
cells per context band, reserves bands 6–7 for replay and recomputes output/timing eligibility from the
raw arms. Repeated deliveries share one stored sample. Each distinct fit keeps its checkpoint, input
sample IDs and grouped report under `pilot/BINDING.training/`.

The trainer atomically replaces `pilot/BINDING.candidate.npz` when the fitted weights change.
A learner-enabled CPU worker checks that path in idle time. A changed head discards its predecessor's
serving qualifications and must earn local timing receipts before it can change a thread count.
`IRONMULE_PILOT_THREAD_CANDIDATE` selects an explicit checkpoint in place of the store path.
A real two-slot SmolLM2 worker reloaded a changed head after a second archive delivery and preserved
both answers through the reload. [Live reload observation](../evidence/runtime/cpu-pilot-archive-live-reload.json)

To replay the decision following the Xeon serving-demand experiment, run the command below. The first order had targeted tokenizer input conversion. Across the four observed waves, the largest remaining single Python function occupied 0.0214% of wall time. Under the planner's same-work assumption even removing all that self time fell below the existing 2% throughput gate. The worker saved its decision to defer further host-only proposals for those observations. A new binding, new source or additional serving evidence can change that decision.

```sh
evidence=lab/autonomous/results/cpu-serving-demand
python -m training.experiment_planner --observations "$evidence/waves" --prior-orders "$evidence/prior" --output /tmp/ironmule-followup.json
```
