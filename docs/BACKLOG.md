# Speed backlog

Every hypothesis that could plausibly make IronMule faster, including the ones that
are probably wrong. Nothing here is a result. Nothing here has been measured unless
it says so and names an experiment.

This file exists because the alternative is holding twenty half-ideas in one head and
re-deriving them badly six weeks later. An idea written down with its kill criterion
costs nothing to keep and can be refuted by anyone.

## PROD1 — Approved product implementation (2026-09-05)

The user approved the product-wrapper plan, real-hardware validation of every
locally available Gemma model, and publishing completed work to GitHub. The
cross-tree work list is `BACKLOG.md` PROD1. Mechanism: portable model/backend
contracts, corrected population-aware evaluation, measured configuration
interactions, and conservative RL outside the token path. Gate: installed-product
correctness plus paired end-to-end validation against the unchanged product and
stock mlx_lm references. Kill: missing evidence, resource/quality failures or no
net gain leaves the reference active. Historical sealed artifacts stay unchanged.

## Read this before optimising anything

**Making the runtime faster usually makes the headline number smaller.** The shipped
gain is `+11.81%` to `+19.24%`, and it is a *ratio*: recovered overhead over total step
time. Grouped batch-1 works by overlapping device execution with host submission
(`E14b`). Anything that makes host submission cheaper — `B8`, `B9`, `B10` — removes
exactly the thing the gain is measured against. The product gets faster and the
percentage falls.

That is not a reason to avoid those entries. It is a reason to state, before running
them, which number is being optimised: **absolute tokens per second**, or the
**grouping gain over an unchanged baseline**. They are not the same target and a fix
can move them in opposite directions. Publish both or the report is misleading.

## How to read an entry

Each has a **mechanism** (why it could work), the **evidence** for and against it from
the ledger, a **test**, and a **kill** — what result closes the entry for good. An
entry with no kill criterion is not a hypothesis, it is a wish.

Effort is calendar-honest for one person on one machine. Payoff is a guess and is
labelled as one.

## Release blockers imported from the 2026-08-27 runtime review

These entries are code-review hypotheses until a failing regression test or current
source proof confirms them. They are deliberately separate from the speed tiers: the
goal is a trustworthy `0.1.1` runtime and benchmark, not a larger headline gain.

### `R2` — Make request arrival, prefill and decode one measurable lifecycle

**Mechanism.** `engine_start_ns` now begins before prefill and remains nonnegative. The
remaining defect is that all prefills still run before request arrival/admission, and
the lifecycle lacks separate phase timestamps for admission, prefill start/finish and
decode. Delayed arrivals can therefore receive model work before they exist in the
simulated service timeline.

**Test.** Inject a deterministic clock/backend and assert the ordered timestamps
`request_received <= queue_entered <= engine_start <= prefill_start <= prefill_finished
<= first_token <= finished`; assert no prefill starts before `arrival_ms`.

**Kill.** The service owns admission and prefill scheduling, all durations derive from
those timestamps, and sequential/grouped results remain token-identical. This changes
the runtime architecture and requires the project's explicit architecture approval
before implementation.

### `R3` — Make the public benchmark balanced and end-to-end

**Mechanism.** The public benchmark now uses complete service `outer_wall_ms`, balanced
AB/BA ordering, and independent plans. Remaining gaps are one shared loaded
process/model, no stock `mlx_lm` arm, and prefill/decode phase diagnostics that are
planned but not yet present in snapshots.

**Test.** Fresh cache/plan instances per arm, at least two warmups, alternating AB/BA
orders in fresh processes, raw samples plus median/spread/interval, and a deliberate
token mismatch that must produce a non-zero exit code and a structured diff.

**Kill.** Primary throughput uses complete service wall time; executor/prefill/decode/
queue times remain diagnostic; the protocol is order-balanced and fails closed on
wrong answers. A stock `mlx_lm` arm is added only after its exact prompt/stop contract
is defined and architecture approval is recorded.

### `R8` — Turn correctness and packaging into automated release gates

**Mechanism.** The macOS workflow now runs remotely and its clean installed-wheel job
is green. The real-model fixture no longer skips every exception. What is left is the
gap this entry keeps confusing with CI: nothing asserted that a *first* run works. The
first two commands a new clone runs, `ironmule models` and `ironmule benchmark`, both
raised a raw `CacheNotFound` traceback on a machine with no Hugging Face cache — the
exact machine every new user is on, and the exact machine CI is.

**Test.** CI builds and installs the wheel in a clean environment, runs unit and CLI
smokes, and checks dependency metadata. Integration setup skips only enumerated model,
access or unavailable-Metal failures; all other exceptions fail. Subprocess tests assert
that a cache-less machine gets an actionable message and a non-zero exit, never a
traceback, and that `--help`/`doctor` still start when the MLX import itself is broken.

**Kill.** Remote clean package/CLI job green (done), first-run behaviour covered by
tests (done), and synthetic regressions covering `R1`–`R7` (open — only `R6`/`R7` have
a dedicated suite). Apple-Silicon model CI remains open until runner availability and
cost are explicitly approved.

### `S1` — Persistent local service with an explicit overload contract

**Mechanism.** A warm process with a real admission queue can expose completions and
chat completions without making callers embed the Python runtime. Streaming,
cancellation, queue limits, timeouts, backpressure, health/readiness and separate
interactive/throughput lanes are one service contract, not independent decorations.

**Test.** Loopback-only MVP with OpenAI-compatible request/stream shapes; bounded queue
property tests; cancellation/disconnect and overload tests; no request prefills before
admission; 1 h stability gate before any production claim.

**Kill.** Unbounded memory/queue growth, incorrect cancellation, token divergence from
the library path, or p95 latency outside a preregistered service budget. Architecture
approval is required before implementation.

### `C1` — Safe cache, chat and sampling expansion

**Mechanism.** Capacity buckets and an LRU prefix-cache budget can reduce mixed-prompt
memory, while multi-message templates and custom stops make the API useful. Sampling
must remain a separate seeded mode because exact-greedy guarantees do not transfer.

**Test.** Prefix mismatch rejects by default; per-bucket peak memory and hit rate;
model-family chat-template corpus; deterministic seeded-sampling distribution tests.

**Kill.** Any cross-tenant/prompt cache reuse, unbounded cache growth, exact-mode token
change, or no material memory reduction from bucketing. Architecture approval required.

### `Q1` — Expand the evidence matrix and run sustained-load gates

**Mechanism.** Current evidence is narrow in chip, model family, quantisation, context,
output length, concurrency and workload. A matrix plus smoke/1 h/6 h/24 h and burst
profiles can separate compatibility from performance and stability.

**Test.** Record exact model revision, hardware, framework, power, RSS/MLX peak/swap,
TTFT p50/p95/p99, throughput, queue depth, fallbacks and cache hits for every cell.

**Kill.** A cell without raw data, repeats, correctness gate or a comparable baseline
cannot extend the validity domain. Hardware/model acquisition and long runs require
explicit resource approval.

### `D1` — Reproducible community bundles and release supply chain

**Mechanism.** A local bundle with console output, raw JSON, fingerprints, checksums and
privacy preview enables external replication; clean build/install, dependency scanning,
SBOM and signed artifacts make releases inspectable.

**Test.** Offline bundle round-trip and redaction tests; clean wheel install/CLI smoke;
release dry run with tag/version equality. Submission is always opt-in.

**Kill.** Hidden upload, private prompt leakage, non-reproducible metadata, or unsigned/
unverifiable final artifacts. Publishing a tag/release requires separate user approval.

### `L1` — Clarify the source-available licence before enterprise claims

**Mechanism.** Developer/company summaries and concrete SaaS, consulting, internal-use,
fork and commercial-contact examples reduce ambiguity; independent legal review is the
authority, not repository code.

**Test.** Counsel-reviewed text and examples agree with `LICENSE.md`; no telemetry or
phone-home enforcement is introduced.

**Kill.** This entry cannot close on an engineering opinion. It closes only with the
user-approved legal review and resulting documents.

| | idea | effort | guessed payoff | correctness risk |
| :-- | :-- | :-- | :-- | :-- |
| `B1` | Width sweep at 27B | hours | 0 – 5% | low, gated by token identity |
| `B2` | Group the `lm_head` only | days | 1 – 3% at 4B, less at 27B | medium |
| `B3` | Unroll k decode steps into one graph | days | 2 – 8% | medium, this axis broke tokens once |
| `B4` | Wire the weights against page pressure | hours | 0 – 10% under load | none |
| `B5` | Fill the group on purpose | hours | 0 – 5% at short answers | none, latency cost |
| `B6` | Cost ratio of `M=4` vs `M=1` against model size | hours | 0, it is a precondition | none |
| `B26` | Qwen3.8 27B: same size, different family | hours | 0, it separates two explanations | none |
| `B30` | Widen Qwen grouped batch-1 groups to 5/6 | days | 0 – 10% | medium, throughput/correctness |
| `B8` | Native decode loop, no Python per operation | weeks | **at most 32/25/17% at 1B/4B/12B** (`B24`) | low |
| `B9` | Record the decode step once, replay it | weeks | **at most 32/25/17% at 1B/4B/12B** (`B24`) | low |
| `B10` | Fewer kernels per step | weeks | **at most 0.7% of GPU time** (`B24S`) | low |
| `B11` | Layer-level pipelining across the group | weeks | 5 – 15% | medium |
| `B12` | Jump the `M=8` valley to width 16 | days | up to 40% throughput | **high** |
| `B13` | Speculative decoding with a real draft model | weeks | 1.5 – 3x at 27B, **rejected once at 4B** | low if verified greedily |
| `B14` | A draft head on the target model | months | 2 – 3x | low if verified greedily |
| `B16` | Lower or mixed weight precision | days | 20 – 40% | **high**, quality |
| `B17` | KV cache quantisation | days | 0 – 5% | medium |
| `B18` | Adaptive layer skip | weeks | 10 – 40% | **high**, quality |
| `B19` | Exploit Gemma's 5:1 local/global layers | days | 0 – 10% memory and bandwidth | medium |
| `B20` | `lm_head` on the Neural Engine | months | 10 – 16% at 4B | medium |
| `B21` | Small projections on the CPU while the GPU runs | weeks | 0 – 10% | medium |
| `B22` | Two processes, one GPU | days | 0, it is a control | none |
| `B23` | Weight layout tuned for `M=1` | weeks | 0 – 20% | low |
| `B24` | Real GPU counters instead of wall clock | **answered 2026-09-09** | 0, it is instrumentation | none |
| `B27` | Evidence-bound execution strategies | audit first, then weeks | 0 immediate; prevents regressions and unsafe reuse | low for audit, medium for routing |
| `B41` | Pad the reduction dimension to 512 | **measured 2026-09-09** | **3.4% at 12B**, -5.8% at 1B, 0 at 4B | medium, logit difference |
| `B42` | `K=3840`-specialised matvec, bit-identical | **confirmed 2026-09-09** | **7-10% kernel, 1.0-1.3% model** (`B43`) | none, identity proven |
| `B44` | The same kernel as an opt-in knob, default off | **integrated 2026-09-09** | **0.8% shipped** (`B44M`) | none, identity proven |
| `B45` | One weight sweep for two requests | **prototype 2026-09-09** | **17.7% vs serial** (`B45M`) | none, identity proven; latency trade |
| `B46` | The same pairing against the shipped throughput path | **measured 2026-09-09** | **12.7-12.8% vs `ThroughputMode`** (`B46`) | none, identity proven; no latency cost |
| `B47` | That pairing as an opt-in mode, default off | **opt-in ready 2026-09-09** | **12.5% through the shipped surface** (`B47U`) | none; cancellation untested |
| `B48` | Sharing `K=4096` and `K=15360` as well | **goal not met 2026-09-09** | kernel yes, **~1% product**, gate was 5% | none, identity proven |
| `B49` | One sweep for four requests at `K=3840` | **kernel no-go 2026-09-09** | **11% slower than two shared pairs** | none, identity proven |

---

## Opened 2026-08-29, from making the repository usable by strangers

These are not release blockers. They came out of running the project the way someone
who just cloned it would, and out of the review that followed.

### `P1` — Ask before querying the operating system

**Mechanism.** Four places shell out to the OS without asking: `hw.py:39` `sysctl`,
`hw.py:51` `system_profiler`, `bench.py:33-40` `pmset`/`sw_vers`, and `tune.py:185`
`ps -Ao pid=,rss=,comm=,args=`. None of it leaves the machine, and `ps` sees only this
user's own processes — but a stranger who cloned this cannot see that and has to take
it on faith. A one-time stored opt-in plus a `--no-probe` path makes the promise
checkable instead of asserted. The project owner has asked for this; it is wanted, but
deliberately not a release blocker.

**Test.** With no stored opt-in, no code path runs any of the four calls;
`doctor`/`tune`/`benchmark` ask once and record the answer. A test that patches
`subprocess.run` fails as soon as a call happens without consent.

**Kill.** The gate breaks existing fingerprints or profiles, or leaves `doctor` unable
to diagnose a fresh machine — the command that exists to answer "why does this not
work" must not be the one that needs setup first. Then the promise is kept another
way, by documenting the four calls instead of gating them.

### `Q2` — Run the self-tuning loop once, for real

**Mechanism.** `tune()` is the core of the self-optimisation the README describes, and
it has never run end to end anywhere in this project: `~/.ironmule` does not exist on
this machine, and `test_r6_r7.py` stubs the engine, `probe` and `gpu_busy`, so what is
covered is the control flow, not the run. Unknown: whether the coordinate descent
completes, whether it finds anything above baseline, and whether token identity holds
across every candidate it tries.

**Test.** A preregistered run on the M1 Max with `gemma-3-4b-it-4bit` cached and on
mains power. Record every candidate with its knobs, time and token match; the profile
written; the gain; total runtime. Then a second start that loads the profile instead
of tuning again.

**Kill.** The run aborts, finds no candidate above baseline, or any candidate changes
tokens. Then self-tuning is not the feature the README advertises and that claim comes
out before the next release.

**Do not mistake a winner for a bug.** `readback_every` is the likeliest candidate to
be kept, and that is correct behaviour. The predecessor project's cycle 17 measured it
at ratio `0.9581`, faster in every pair, and rejected it only against that experiment's
own preregistered 5% bar. `tune` keeps anything below `KEEP_IF_RATIO_BELOW = 0.995`
(`tune.py:80`), so the same number qualifies here. Two knobs genuinely cannot win and
would indicate a broken harness: `prefill_into_fixed` (E1 bounds the prize at 1.47 ms
of 537 ms, ratio `0.9973`) and `speculate_k` (ratio above `1.0` on MLX 0.32).

### `Q3` — Adaptive optimizer method selection and replay

**Mechanism.** Reuse `tune.Knobs`/`SEARCH` and the evidence-bound execution
surfaces from B27 to make optimizer-method choice itself evidence-bound. Start
offline with no runtime import: define a durable state/action/outcome/failure/
uncertainty schema and an information-gain signal, then replay recorded studies
before allowing any method to influence tuning. Treat reinforcement learning as a
hypothesis, not an assumption; it is justified only if the data show a sequential
benefit that simpler search methods cannot provide.

**Test.** At equal evaluation budgets, compare the current coordinate descent and
`BASE` against seeded Random, a Bayesian/surrogate method, and a contextual bandit.
Use disjoint `TRAIN`/`VALIDATION`/sealed `HOLDOUT` group splits by study, model,
hardware and time, and report best-known outcome, regret, uncertainty calibration,
failure recovery and replay determinism. Keep all safety gates external to the
optimizer: exact output/token/stop/count identity, resource and leak checks,
fingerprint/domain validity, timeouts and rollback must pass before a candidate is
considered. No runtime import, routing, persistence or activation is part of this
experiment.

**Kill.** Insufficient real coverage or sealed-holdout evidence means data
collection only. If a simple method is equal or better at the same budget, do not
introduce RL. Any output divergence, correctness failure, resource leak, unsafe
failure handling or non-reproducible replay stops the experiment and leaves the
current deterministic coordinate descent unchanged.

**Current result (2026-08-31).** The real-data replay is `DATA_INSUFFICIENT` for
adaptive method comparison and `NOT_APPLICABLE` for offline RL. The frozen dataset
has SHA-256 `f67d975788763e4238019a3be7afa5394efbe2f2faea3a96a927e7cf522f2e33`,
dataset ID `d4ae0c148e826de85c7aa5338f892b5571481a105f558d463e9d041f63dc82b7`,
14 observations, 12 actions and 160 B36 raw timing samples. Q2 contributes a
validation trajectory, B36 a sealed holdout, and there are no training rows; the
counts are inventory facts, not statistical qualification. Q3 remains open. The
next missing evidence is separated into (1) a complete raw counterfactual action
panel for coordinate/random/BO-surrogate replay, (2) independent grouped contexts
with comparable panels for generalisation and contextual bandits, and (3) a measured
sequential horizon before RL can become applicable.

### `Q4` — Evidence-bound hierarchical RL and two-stage hybrid optimizer

**Mechanism.** The Q3 contract has a durable offline state/action/outcome schema, but
its ten-knob action space cannot express the execution/scheduling evidence that made
E14b and E16 useful. Q4 joins the existing Q2, B35, B36, B27, E14b, E16 and X1
knowledge through content-hash migration while keeping two closed spaces separate:
`KnobAction` for the ten runtime knobs and `ExecutionStrategyAction` for plan, mode,
grouping, synchronization, cache and workload policy. A hierarchical conservative
offline-RL policy can then learn an H=17 sequential trajectory (`11 KNOB_DELTA`, five
`STRATEGY_SELECT`, one `REVALIDATE`) before final revalidation, with evaluator-owned
correctness/resource gates, separate knob/strategy value heads and failure-risk critic,
behaviour prior, exact propensities, action masks and uncertainty/OPE checks. The
`HybridOptimizer` is two-stage and shadow-only until a later architecture decision.

**Test (remaining hardware/evaluation work).** The offline contracts, corpus importer,
replay methods, strict OPE, dataset gate, shadow envelope and foreign replay boundary
are complete; the durable implementation report is
`research/raw/Q4_implementation_report_20260901.md`. Remaining work is explicitly
named collection only: on the local M1 Max, collect 24 entirely new contexts spanning
Gemma 1B/4B/12B (`no27`) in new `Q4_TRAIN`/`Q4_VALIDATION`/`Q4_SEALED_HOLDOUT`
groups by study/model/manifest/workload/hardware/runtime/time; build 12-knob and
plan-matching five-strategy panels; collect 72 complete H17 trajectories (1224
transitions); then run the equal-budget BASELINE/current-coordinate/seeded-random/
BO/surrogate/contextual-bandit/OFFLINE_RL evaluation once on the direct sealed panels.
Historical `Q3_VALIDATION`/`Q3_SEALED_HOLDOUT` and E11 `LEDGER_ONLY` rows remain
outside Q4. The Stage-2 panel is 12 knob actions × 5 matching strategies = 60 exact
cells/context, in 12 anchor phases; S11/S12 are separate risk probes. Each collection
phase is separately preregistered and explicitly user-started, with 16 candidate
decisions per context (11 knob-delta + 5 strategy; BASE external), exact identities,
propensities, p95/request gates, grouped metrics, recovery and replay checks. The
runtime remains unable to import, select, write or activate from the shadow report.

For each context, Stage 2 uses only the five strategies matching its frozen plan:
S01--S05 for StrictOneShotPlan or S06--S10 for ReusableSessionPlan; the opposite plan
is not exact/safe. The Stage-2 interaction panel is the full 60-cell cross-product,
collected in 12 separate knob-anchor phases; each phase uses five fresh processes, one
per strategy under that knob anchor (five arm cells, two warmups and five measured
repeats), and any missing pair forces BASE fallback. The shared BASE reference is
external to the method budget. The H17
trajectory has explicit `KNOB_DELTA`/`STRATEGY_SELECT`/`REVALIDATE` stages; every delta
changes exactly one legal knob field, accepted targets update the current action and
rejected/failed targets leave it unchanged. Partial aborts are terminal at the current
step and complete only at step 16. Knob FQI and strategy-immediate value heads are
separate; the hybrid never scalar-adds them.

**Kill.** Keep the current deterministic coordinate path and mark Q4
`DATA_INSUFFICIENT` if any split, full panel, independent context or measured H=17
horizon is absent. Mark RL `NOT_APPLICABLE` until the sequential horizon exists. Reject RL
permanently for this decision when the best simpler method is equal or better at the
same budget, RL does not beat it on sealed-holdout reward and both time-to-best and
experiments-to-best, regression is worse, or uncertainty/OPE support is uncalibrated.
The direct grouped advantage must have a 95% lower bound `>+2pp`; the equivalence
margin is `1pp`. A DR estimate with opposite sign or absolute disagreement `>2pp`
invalidates `RL_WINS`. `time_to_best`/`experiments_to_best` use original cost
`c <= 1.01*c_oracle`; unsafe/censored and safe `c > 1.02*c_BASE` rates use separate
exact denominators.
Any token/output/stop/state divergence, safety/resource/cleanup failure, split
leakage, fabricated/imputed/summary-only label, nondeterministic replay, unauthorized
runtime import, profile write, routing or activation is terminal; BASE/current
coordinate remains unchanged. WIS uses ratio clip 10 and grouped five-fold DR by
complete context/group hash (all trajectories in a context co-fold); any overlap/support
failure yields `OPE_UNSUPPORTED`. Do not retry or pool Q3c/Q3d/Q3e/Q3f, promote X1/B35/
B27 summaries or exploratory true batching, combine E14b's `+18.02%` and `+20.05%`,
or invent foreign-Mac measurements.

### `R14` — Process-inventory/group-gone order interaction in integration/release quality

**Mechanism.** The combined integration order can change macOS process-inventory,
group-gone and cleanup timing observations even when the Q3d/Q3f cleanup logic is
correct in isolation. The initial full-pytest failure was a separate test-harness
`sys.modules["ironmule"]` pollution issue; the private `tests/q4_offline_loader.py`
namespace fix resolved that without product changes.

**Test.** Run the full integration collection without Qwen or 27B, and run the exact
macOS cleanup tests in isolation: `tests/test_q3d_stability_gate.py::test_real_macos_process_identity_and_cleanup_reap`
and `tests/test_q3f_child_guard.py::test_q3f_real_cleanup_keeps_external_process_alive`.
Record process inventory, group-gone ordering, timing and cleanup evidence for both
combined and isolated runs.

**Kill.** Keep this integration/release/collection-quality issue open until two
consecutive full integration runs are green, or a deterministic order/timing proof
identifies and fixes the interaction. Neither isolated passes nor a Q4-only `55/55`
pass closes this entry; no Qwen/27B result is inferred. The user-authorized merge may
proceed with R14 explicitly open; this entry is not a Q4 performance or model result.

### `Q3a` — Path interaction: final Q2 incumbent versus `fused_argmax`

**Mechanism.** Q2 evaluated `fused_argmax` early and then retained
`compiled_fixed_cache=True`, `head_skip_prefill=True`, and `readback_every=2`.
This six-process, balanced fresh-process A/B pilot asks the narrower path-dependence
question: does enabling `fused_argmax=True` change the result only after the final
incumbent has been assembled? Use only the locally cached
`mlx-community/gemma-3-4b-it-4bit`, two warmups and seven measured repeats per arm.
The dry-run is the default; execution is capped by the preregistered conservative
`6 * 35 s + 60 s = 270 s` bound and must stop before start if the bound exceeds
300 s. AC power, low-power off, nominal thermal state, no competing model process,
swap and resource ceilings, three-sample load average (`max <= 4.0`, spread `<= 1.0`),
strict process inventory, exact token/count/stop-reason/determinism, complete raw
evidence, and runtime/model/environment identity are hard gates. The parent owns a
monotone 300-second deadline; each child is capped at 35 seconds and the worker
phase at 240 seconds. A passing result is information gain about path interaction
only; there is no profile activation or promotion.

**Kill.** Refuse before start on any unknown preflight state, missing local model,
missing model revision/manifest, competing model process, or runtime bound above
300 s. During execution, timeout, crash, swap/resource breach, malformed process
inventory, load-gate breach, incomplete raw data, or any token/stop/count mismatch
records `FAILED` with `BASE` fallback and stops the pilot. A clean run still cannot
promote a profile; classify the paired total ratio before measurement as `GAIN` only
when `ci_high < 0.995`, `LOSS` only when `ci_low > 1.005`,
`PRACTICALLY_NEUTRAL` only for a complete CI inside `[0.995, 1.005]`, otherwise
`INCONCLUSIVE`. Do not claim a direct statistical comparison with early Q2.

**P2 safety debt (runtime lifecycle).** `ab.run` must retain partial child records
and terminate the entire child process group on timeout; kill/cleanup failure is a
hard `FAILED` result, never a short successful run. Kill when a timeout leaves an
orphan process or the raw record cannot identify the completed children.

**P2 safety debt (evaluator-owned identity).** The runtime must expose per-repeat
physical/logical tokens, counts, stop reasons, capacities, RSS and resource gates
without letting the optimizer infer missing values. Kill when a new execution path
can pass validation with absent or self-asserted identity/resource evidence.

**P2 safety debt (streaming worker output).** The current worker uses bounded
`Popen` pipes and a 512 KiB cap, but `communicate()` still buffers the complete stream
before the cap is checked. Replace this with a tempfile/selector-backed bounded reader
that preserves progress markers and terminates the worker group on overflow. Kill when
an overflow can block the producer, lose a completed-child marker, or leave an orphan.

### `R10` — An aborted run must not look like a finished one

**Mechanism.** `e14b_arms.py:243` breaks the block loop on the memory guard and reports
it with a `print` to stdout. The result file records nothing: a truncated run carries
`runs: 1` and is otherwise shaped exactly like a complete four-block one. A reader who
has the JSON but not the console log cannot tell a cut-short experiment from a
deliberately short one, and the analysis that follows rests silently on a quarter of the
intended samples. Found by living through it: the 12B leg of a scaling run aborted, and
only the terminal output said so.

**Test.** A run that hits the guard writes a machine-readable record into its own result
file — the reason, the block index reached, and the value that tripped it. An analysis
helper refuses to summarise a file carrying such a record unless the caller acknowledges
it. A synthetic run with the guard set below the first block's peak produces that record
rather than a plain short file.

**Kill.** Result files are bound to preregistration hashes, so a schema change
invalidates the comparison the file was written for. This closes only if the record can
be added without breaking existing readers — an additive optional key — or with an
explicit decision to version the schema. If neither is acceptable, the fallback is that
the guard raises instead of breaking, so an aborted run produces no result file at all
rather than a plausible one.

**Related, and it bites before the guard's reporting bug does.** The threshold is a
hard-coded `12 * 1024**3`. Gemma 3 12B's true per-block peak is `17.51 GB`, measured on
block 1, which has nothing accumulated to inflate it. So 12B trips the guard honestly,
with or without the peak reset, and a 27B 4-bit model at roughly `15 GiB` of weights
cannot be measured either. On a 32 GB machine that holds both comfortably, this harness
runs only the smallest of the three cached models to completion — which is also part of
why the scaling evidence in this repository rests on 4B. Raising the number is a
decision about swap safety and needs its own entry with a kill criterion, most usefully
with the threshold as a parameter and direct swap monitoring as the criterion rather
than another constant in the source.

### `R9` — `ironmule.tune` is the function, not the module

**Mechanism.** `__init__.py:35` rebinds the name `tune` from the submodule to the
function it exports. `import ironmule.tune as m` therefore yields the function; the
module is reachable only through `importlib.import_module("ironmule.tune")`. It cost
one debugging round while writing the `gpu_busy` regression test, and it will cost the
same to anyone writing against the package.

**Test.** One test that performs both accesses and asserts the type of each, so the
behaviour is pinned for as long as it exists.

**Kill.** Renaming breaks the public API. This closes with a major version bump, or
with the decision to document the quirk permanently rather than change it.

---

## Tier 0 — measured and rejected. Re-open only under the rule below.

### Rejected is not forbidden (project rule, 2026-09-10)

Tier 0, `NO-GO` and every closed kill entry are **historical evidence, not a permanent
ban**. They record what a specific mechanism cost on a specific fingerprint, and that is
exactly how far they reach.

Any of them may be re-opened when one of three things is new:

* **new hardware evidence** — a different chip, memory size, MLX/mlx-lm build or model
  revision than the one the entry was measured on;
* **a changed mechanism** — the reason the entry died no longer applies, stated
  explicitly against the old entry's own kill criterion;
* **a new implementation** — a different code path, kernel or execution route, not a
  re-run of the same one.

Re-opening costs one thing: the new entry must name the old entry, quote the old kill
criterion, and say which of the three conditions above is met. Repeating an experiment
with no such statement is still forbidden, because that is what wastes GPU time. Refusing
a method *only* because an older attempt failed is equally forbidden, because that is what
freezes a runtime.

- **`B15` exact-but-pruned `lm_head` (2026-09-09).** Measured offline on
  `gemma-3-4b-it-4bit` over 72 real greedy decode steps: with a correct
  Cauchy-Schwarz cluster bound, `95.5%` of the 262,208 rows survive at `k=16384`
  and `98.5%` at `k=4096`, against a kill threshold of `25%`. The token was exact
  on 72 of 72 steps, so the bound is right and the geometry is the problem. The
  bound needs a cluster radius of `0.106`; k-means reaches `0.387`. It is not a
  clustering failure: sampling 256 rows, only `1.25` rows on average lie within
  the required radius of any row, and the mean nearest-neighbour distance is
  `0.235`. Achieving the radius would need roughly one cluster per row, and the
  centroids alone would then cost more than the rows they replace. Row norms are
  nearly uniform (`0.006` to `1.085`, median `1.000`), so the pure norm bound
  prunes `0.02%`. Experiment `B15_4B_offline_bound_20260909_attempt1`; do not
  re-run with a different clustering.

- **PROD8 12B 1077-token integration screen (2026-09-07).** First real stock-reference request hit the frozen 6 s host deadline (6.002551 s observed); owned worker reaped, no product/HTTP run. Experiment `PROD8_12B_long_context_20260907_attempt1`; no shorter-context or relaxed-limit retry. Phase-aware follow-up hypothesis: root `BACKLOG.md` PROD9.

- **PROD2/PROD6 12B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol, exact outputs/resources and three normal worker exits; no qualified net gain at any cap under the frozen noise rule. Experiment `PROD6_12B_installed_20260907_attempt1`; no adoption or same-design retry. Results: `docs/PROD6_12B_RESULTS_2026-09-07.md`.

Listed so the next person does not spend a week rediscovering them.

- **PROD2/PROD6 1B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol and clean worker exits, but no qualified signal above the frozen noise floor; no adoption. Experiment `PROD6_1B_installed_20260907_attempt1`; do not repeat for a luckier result.

- **PROD2/PROD3 4B short-prompt calibration (2026-09-07).** Complete installed 90-call protocol, exact outputs and resources valid, but all three caps remain below the frozen noise qualification; no adoption. See `PROD3_4B_installed_20260907_attempt1` and `docs/PROD3_RESULTS_2026-09-07.md`; do not rerun the same design for a luckier result.

- **`Q3c` direct replication attempts (2026-08-31).** Run 1 was refused before
  a phase because load `8.294921875 > 8` (raw SHA-256
  `5270c0f38e50984cd26223aa2a9817982fc5a1861ddbe2caa3cff98393c9e8d5`); run 2
  aborted after `105` samples / `27.395 s`, swap `2,353,654,661 B` to
  `2,625,172,930 B`, delta `271,518,269 B` (`258.94 MiB > 128 MiB`), with
  cleanup unverified. No timings, identity, performance result or promotion
  exists. Raw run 2 SHA-256 is
  `d94db80402254c87c0e4a0128cf802e1eaa59d42c4459c2f208077f48c38b8df`; retain
  both records and do not rerun Q3c.

- **`Q3d` model-free recovery gate (2026-08-31).** The gate itself passed:
  `61` samples over `60.020192667 s`, maximum adjacent gap
  `1.013944625 s`, swap `2,651,722,874 B` throughout and delta `0 B`.
  Raw SHA-256 is
  `4699a49b174db31580a9701ef2075f8b1964d309b0f857dd7779fb230cfccb83`
  (`34,144` bytes); the summary SHA-256 is
  `3b43e267000ba15b9d9079d9f118e59c1cd51dbcdfecc067c20995b01a0a1c3e`
  (`970` bytes). The one permitted Q3c invocation was then refused before
  `Popen` because macOS `/bin/ps` rejected `sid` (`rc=1`, `ps: sid: keyword not
  found`). No model, inference process, timing, identity or performance data
  exists. Final state is `Q3C_FAILED` with `BASE/current incumbent` fallback;
  do not repeat Q3d or pool its gate with Q3c. Q3e is the separately frozen
  portability-repair path.

- **`Q3e` portable process probe and one Q3c invocation (2026-08-31).** The
  repair removed unsupported macOS `ps sid` and the exact Phase-R run completed
  with token/resource identity intact. Its descriptive incumbent/BASE total
  ratio was `0.857466859207542`, CI
  `[0.8551668079699586, 0.8611021999710893]` (`14.2533140792%` faster), but
  cleanup conservatively rejected four stable unrelated same-UID launchd
  services that appeared after the worker baseline. The raw result is
  `research/raw/Q3e_q3c_final_20260831.json`, SHA-256
  `1df6c81dc824911016e687883c535f1ec314f3e03b51303b04c38ae71bb6f4ea`, size
  `2,205,857` bytes; the terminal note is
  `research/raw/Q3e_terminal_result_20260831.md`. Status is `FAILED`, Phase N
  did not run, and fallback remains `BASE/current incumbent`. Do not pool or
  rerun Q3e; Q3f is the separately frozen attribution path.

- **`Q3f` terminal attribution path (2026-09-01).** The single permitted Q3f
  execution reached the exact local Gemma 4B worker only far enough to fail with
  `ABRunError: child 0 start callback failed`; no child marker, ledger, timing,
  token, Phase-N or accepted performance evidence exists. All 14 preflight
  checks passed, AC/thermal/low-power gates were green, and swap stayed at
  `2,609,643,520 B` across 25 samples with zero delta. Cleanup reaped the
  worker and rejected the run closed because guard/ledger evidence was
  unavailable and same-UID Spotify Helper PID `52017` could not be proven
  unrelated. Raw SHA-256 is
  `e82accdbd52857e6201fa2b34984765e61658ecfd3956d5c903a49c1e6de70a9`
  (`2,487,533` bytes); terminal note is
  `research/raw/Q3f_terminal_result_20260901.md`. Status is `FAILED`, with
  `BASE/current incumbent` fallback. Do not retry Q3f, pool its record or say
  that historical Q2 speed was reproduced; any child-visibility race fix needs
  new authorization and preregistration.

- **`R11/R12/E15` fork-per-block memory-integrity path (b700377).** Closed by the
  complete four-block E15 after-file (SHA-256
  `d14875e43ee800d8f1a29af966b8adad56245a414dd204f202a48b81d1f91b5c`): four fresh
  PIDs, flat per-block peaks, no swap growth, no correctness divergence. This is an
  engineering memory result only, not a speed claim; do not repeat the exact attempt.

- **`B25` KV cache reallocation during decode.** Nothing reallocates. The fixed-shape
  cache is allocated once per `serve()`: `mx.zeros` appears only in
  `_empty_fixed_state`, and `_caches_from_state` wraps the existing arrays rather than
  copying them. Measured, not just read: writing 56 tokens through `FixedKVCache`
  moved active memory from `65,644 B` to `32,876 B` — it *fell* by exactly one full
  keys+values copy (`32,768 B`), which is the warmup's double buffer being released
  once MLX takes the `slice_update` donation. Shape constant throughout.
  `tests/test_cache_allocation.py` holds both properties. The predecessor's `4.4263%`
  from cache growth copies (candidate 21) is gone with the growing cache it was
  measured on, so there is nothing here to claim.

  **One caveat worth carrying, because it shapes how the tuned gain reads.**
  `FixedKVCache.update_and_fetch` builds `mx.array(0)` and `mx.stack(...)` on every
  call, and `make_mask` an `mx.arange(capacity)` — per layer, per decode step. Under
  `compiled_fixed_cache=True` these are bound once when `mx.compile` traces the body,
  so they cost nothing. Under the untuned `BASELINE` they are rebuilt every step. Part
  of that knob's `0.9679` is therefore host work that stops happening, not GPU work
  that got faster — which is the documented purpose of `mx.compile`, and `E5` already
  put `3.3 ms` of host work on each step. How the ratio splits between constant rebuild
  and kernel fusion is not measured. The tuned gain against `BASELINE` is real, because
  `BASELINE` is what an untuned install actually runs; it is not a gain against a
  well-optimised floor.

- **B27e mirrored cross-commit control.** Four fresh 4B processes, source-surface
  digest `ec242c…`, all correctness/resource gates green. OLD/D1 block ratios were
  within 2%; mirrored D1/OLD made D1 appear 5.8–7.9% faster. Final
  `ORDER_OR_TEMPORAL_DRIFT`; a consistent D1 slowdown was not reproduced, but B27d
  remains formally inconclusive. Do not rerun the same two-block unconditioned design.

**Two lists, one of them closed.** The entries below marked `E*` come from this
project's ledger. The ones marked `cycle *` come from the predecessor project, whose
candidate catalogue is `EXPERIMENT_BACKLOG.md` in the source repository — 24 candidates
over 17 sealed cycles, most of them closed. That file is history and takes no new
entries; everything current goes here.

- **Prompt-lookup speculation.** `E0c`: `2.9x` slower, acceptance `0.17` per drafted
  token. The workload does not repeat itself enough. This does **not** kill `B13`,
  which uses a different draft source.
- **Projection fusion as a decode win.** `E5`: paired across six processes, decode
  ratio `0.9990` with a CI containing 1.0. Kept for its `-1.10%` prefill effect only.
  `E0b`'s `+2.4%` was noise.
- **A dedicated `M=1` fast path.** `E2`: the apparent `M=1` cost was a `~0.49 ms`
  `eval`+`synchronize` round trip, not kernel behaviour.
- **`prefill_into_fixed`.** `E1`: the phase it would optimise costs `1.47 ms` of
  `537 ms`. Killed before it was benchmarked.
- **Padding `M=322` to 384.** `E2`: `~4.74 ms` predicted against `4.43 ms` measured.
- **An adaptive width controller.** Realised width already adapts; nothing measured
  beats a fixed 4. Revisit only if `B1` finds the optimum is model-dependent.
- **A draft model drafting for a larger target** (predecessor candidate 11, 1B for 4B).
  Measured and rejected, recorded as `0.560x`. **The sign convention on that row is
  ambiguous** — elsewhere in that file a ratio below 1.0 marks a win — so re-derive the
  direction from the cycle's raw data before trusting it either way. See `B13`, which is
  the same idea at a target/draft ratio the predecessor never tried.
- **`mx.compile` over decode subgraphs on a growing cache** (predecessor candidate 12).
  `-23.8%` dispatch and **wrong tokens from position 2**. Rejected, and the older
  device-model compile numbers were invalidated with it. The same idea on a *fixed-shape*
  cache is what this runtime ships: cycle 16 measured `0.9296` with identical tokens
  across 18 arm executions, and `E0a` reproduced it. The lesson is about the cache, not
  about compiling. See `B3`.
- **A custom Metal kernel chosen without profiler evidence** (predecessor candidate 14).
  Locked because cycle 9 localised no single kernel hotspot. Still the right rule:
  `B24` before `B10` or `B23`, or it is guesswork with a compiler attached.
- **Bundled host readback**, reading the stop token only every `N` steps (predecessor
  candidate 18, cycle 17). Readback 8 was faster in every pair, but `0.9581` missed a
  preregistered 5% threshold, so the recorded decision is
  `no_clear_speedup_baseline_retained` and the `4.19%` is calculated rather than claimed.
  A valid negative result. It also points the wrong way for the scaling problem: a fixed
  per-step host cost is a *smaller* share of a longer step, so 27B would show less, not
  more. Do not re-run it hoping for a better draw.
- **B28 / `qwen_native_true_batch_v1` rejected at the correctness gate.** Widths 2, 3
  and 4 produced exact visible tokens and stop reasons with no fallbacks, but the final
  hybrid `kv_hash` differed from the sequential reference. Swap delta was `0 B`; no
  token-rate or other performance measurement is valid. Do not route this path.
- **B29c / `qwen_native_b1_v1` below target.** Widths 2, 3 and 4 passed exact
  correctness, final state and 16-token continuation with no fallback and zero swap
  delta. Candidate `16.0722` versus Interactive `15.6740` (`1.02541x`) and versus
  Throughput `16.0687` (`1.000219x`) remain below the `1.10` gate. No route.

- **B40 width sweep (experiment B40, `INCONCLUSIVE`).** W2/W4 and W3/W4 were
  directionally slower on all six blocks, but material epoch drift prevented a
  valid selection; all 18 children and safety gates were clean. No retry is
  authorized, and no W2/W3/W4 timing may be cherry-picked or treated as a
  selected profile. W4 remains the unchanged operational baseline.

---

## Tier 1 — cheap, grounded, worth doing first

### `B56` — A per-request objective, not a per-runtime one

**Mechanism.** `B55` measured the latency/throughput trade at `+7.7%` aggregate tokens
per second for `+74.9%` median per-request latency at two ready requests. `objective` is
currently set once, at `AppleRuntime.load`. A server answering a chat stream and a batch
job from the same loaded model wants both at once, and the objective is a dispatch-time
fact the caller already knows — it can travel on the `Request` the way an execution plan
does. The router would then group only the requests whose callers accepted the trade.

**Test.** Add the field, route a mixed dispatch where some requests carry `latency` and
some `throughput`, and measure against two fixed arms under the `B55` protocol: token
identity per request, per-request median latency for each objective class, and aggregate
throughput. Sixteen rotated blocks, `2%` equivalence margin.

**Kill.** If splitting a dispatch by objective costs more than it saves — because the
throughput class loses the width the latency class took away — the objective stays a
runtime-level setting and this entry closes with the measured width curve.

### `B57` — Qualify a kernel per shape, through the profile

**Mechanism.** `k3840_matvec` is admitted at load, against one `K`, by
`Engine.admit_k3840`. `ironmule/kernel_registry.py` already derives one MLX kernel name
per whole specification, so several specialisations can coexist without the
ml-explore/mlx#3832 collision that `B54` closed. What is missing is the record that says
*which shapes* a kernel is qualified for on this fingerprint, so the router can select
one per dispatch instead of per load. The tuned profile already carries exactly this
shape of record for the service strategy (`ironmule/service_strategy.py`): strategy,
validity range, correctness contract, evidence run IDs.

**Test.** Extend that record to kernels, qualify one shape end to end against the library
path with bit-exact outputs, and measure the routed selection against a fixed
library-only arm under the `B55` protocol. Only shapes with stored evidence may activate.

**Kill.** If per-shape selection cannot be made bit-exact against the library path for
the shapes it would select, or if the selection cost exceeds the kernel's gain at every
qualified shape, kernels stay load-time knobs.

### `B58` — Does the paired route survive real concurrency?

**Mechanism.** `B45` to `B50` measured `PairedThroughputMode` on a request *pair*:
`12.7-12.8%` earlier completion. `B52` measured that choosing it from the profile costs
nothing against naming it. Neither measured it under a stream of arrivals, where the
number of ready requests moves between rounds and a partner may not exist when the round
starts. `B55` never exercised it at all — this machine's profile carries no
service-strategy record, so `AutomaticMode` correctly kept the established mode
throughout, which means the routed paired path is currently untested end to end.

**Test.** Write a service-strategy record for this fingerprint from the `B45`/`B46`
evidence, then run the `B55` protocol with staggered arrivals so ready counts vary within
a dispatch. Gate on token identity per request, `paired_steps` versus `solo_steps`, and
per-request latency against the grouped path.

**Kill.** If the paired route wins only when both requests are ready at dispatch — a
condition a real arrival stream rarely meets — it stays an explicitly named mode and is
never routed to automatically.

### `B59` — Is a native Metal path faster than `mx.fast.metal_kernel`?

**Mechanism.** Custom kernels currently reach the GPU through
`mx.fast.metal_kernel`, which compiles from source at first use and dispatches through
MLX's own primitive. A native MLX extension or a prebuilt `.metallib` removes the
source-compile step and part of the dispatch wrapper. Whether that is measurable at
decode shapes is unknown; the reasonable prior is that it is not, because decode at width
one is bandwidth bound (`E4`: `104` to `324 GB/s` over `1.4` to `360 MB`) and dispatch is
not the binding constraint. This entry exists to measure that rather than assume it in
either direction.

**Test.** Build the same `k3840` matvec three ways — `mx.fast.metal_kernel`, a native
extension, a prebuilt `.metallib` — assert byte-identical outputs against the library
path on the captured inputs, and compare dispatch and end-to-end decode time under the
`B55` protocol.

**Kill.** If the three paths are within the `2%` equivalence margin at decode shapes,
`mx.fast.metal_kernel` stays the only kernel path and this entry closes. "Closer to the
hardware" is not a result.

### `B64` — Now that `wired_fraction` can be confirmed, does it pay?

**Mechanism.** `B62` removed the trap: the knob no longer starts a subprocess in a guarded
child, so a candidate that keeps it can now reach and survive a confirmation. Nothing about
that says it is worth keeping. The only evidence that it buys anything is one screening
ratio, `0.9034`, from the `12B` run that then died — a single-process screening number,
never confirmed, and taken on a machine whose memory behaviour under a wired limit is
exactly what is in question. `B61` re-screened without it and reached `0.9218` on other
knobs, so the two numbers are not comparable and must not be subtracted.

**Test.** Its own preregistered study, on the `12B` where the screening once kept it: the
standard paired confirmation of the current confirmed candidate against the same candidate
plus `wired_fraction=0.6`, with the wired limit's effect on free memory, swap and MLX active
and peak bytes recorded around every child, and the existing swapout and free-memory gates
unchanged. A wired limit of `20.6 GB` on a `34.36 GB` machine is the thing being measured,
not a side condition, so the resource trace is a result and not a diagnostic.

**Kill.** If the interval against the unmodified candidate does not lie entirely below `1.0`,
or if the run cannot pass the resource gates it is measured under, the knob is removed from
`ironmule.tune.SEARCH` rather than left in it. A knob that survives confirmation but never
wins one does not belong in a search that feeds a profile.

### `B80` — Continual learning during ordinary use, with nothing to explore

**Mechanism.** `B79` closed the loop once, on evidence gathered by studies. Every session it
learned from was a preregistered measurement with its own reference arm, its own A/A control
and its own gates. Ordinary use has none of those: a user's dispatches are all candidate or all
reference, never both, so nothing in them estimates a ratio. The open question is whether a
running installation can add *valid* local evidence without ever exploring — without serving a
single request on a path chosen to learn from rather than to answer it.

**Test.** The only honest source is a paired measurement the user did not pay for: a reference
arm run when the machine is otherwise idle, against the candidate arm from the same period,
under the same gates `B76` used, with the A/A control that decides whether either can be read.
Gate on the controller's state moving only through the same `decide_state` every other path
uses, on `B79`'s kill criteria staying armed throughout, and on the interval that qualified the
action being re-checked against the widened evidence after every update.

**Kill.** If valid evidence cannot be produced without exploring in the user path, continual
learning stops here and the controller stays a thing that is updated by studies. A preference
learned from unpaired production traffic is a preference learned from whatever else the machine
was doing, which `B77` already measured as worse than knowing nothing.

**Not authorised by `B79`.** `B79_LEARNED_DISPATCH_CONFIRMED` says a qualified action can be
dispatched safely. It says nothing about earning a qualification during use.

**Dead end, measured, do not re-run.** `B76`: a contextual model over `load_1min`,
`memory_free_percent` and `swap_used_gb` predicting the `(4, 8)` stack ratio on this machine.
Fourteen sessions, ridge at `alpha 1.0`, sealed prospectively, re-scored by `B77`. Prediction
error `0.0087` against a constant model's `0.0058`, coverage `0.75` against `1.00`, cumulative
regret `0.2037` against `0.0750`. Its error is larger than the entire between-session spread it
exists to predict. Those three features carry no signal about this effect at this spread. A
different feature set needs a mechanism first, not another fit.

**Dead end, measured, do not re-run.** `B77`: choosing between a plain mean and a Bayesian
posterior on this evidence. Identical action sequence across fourteen sessions, identical
regret, identical coverage, a prediction-error gap of `1.3%` of the between-session `SD`. The
data does not distinguish them and no further comparison will until the spread moves.

### `B73` — A second Mac, which is what most of this now needs

**Mechanism.** Four separate entries have reached the same wall. `B71`'s `H1` cannot be
tested on one machine. `B72`'s hardware axis does not vary. `B69`'s confirmed geometry is
bound to one fingerprint and nobody knows whether it is a property of this chip or of Apple
GPUs. `B66`'s width step at `16` and its submission-boundary cost are the same. One machine
has produced everything it can.

**Test.** Run, on a Mac with a materially different memory system, exactly what already
exists and in this order: `B71`'s probe set, then a tune, then `B57`'s composition study, then
`B69`'s stack proof for `(4, 8)`. Nothing new is written for it. Every harness already fails
closed on an unknown fingerprint, so the second machine's results cannot contaminate this
one's.

**Kill.** If the second machine's vector and its stack results agree in sign with this one's,
`H1` survives its first real test and a third machine becomes worth arguing about. If they
disagree, the vector is a description of one machine and the entries built on it say so. Either
outcome is a result; not having a second machine is not.

**Sharpened by `B76` and `B77`.** The comparison target is no longer one confirmation. Fourteen
sessions on this M1 Max put `single_short` at `0.9647` with a between-session `SD` of `0.0069`,
so a second machine is measured against a distribution and a disagreement has a scale. `B77`
then found that what separates `B69`'s `0.8469` from those sessions is dispersion, not speed:
`B69`'s A/A control ran at a half width of `0.0848` against `B76`'s `SD` of `0.0069`, and its
candidate interval overlaps its own control. So the second machine reports its A/A dispersion
and its machine state **before** any magnitude it produces is quoted, and a magnitude whose A/A
arm fails the `B75` gate is recorded without being quoted at all.

### `B66` — Silicon characterisation, against the best confirmed complete stack

**Mechanism.** Every lever measured so far moves work around the device. None has asked what
the device wants: threadgroup size, grid geometry, cache locality, how many command queues
are worth having, register pressure, and how all of those change with shape. `E4` measured
decode at width one as bandwidth bound between `104` and `324 GB/s` over `1.4` to `360 MB`,
which is the prior — it says dispatch geometry should not matter much at decode, and says
nothing about prefill, about grouped widths, or about the paired path's shared sweep.

**Test.** Profile the shapes this runtime actually executes, then compare each candidate
geometry against the *best confirmed complete stack* for that workload class, never against
a microbenchmark and never against a sum of earlier percentages. On `4B` that reference is
the confirmed composition profile; on `12B` the throughput classes have no confirmed stack
yet — `B63` blocked — so either that resolves first or `12B` is characterised against `A_t`
and `B` and said to be so.

**Kill.** A geometry that does not beat its own reference as a fully executed stack does not
enter one, however good it looks in isolation. If decode geometry turns out to be as
bandwidth-bound as `E4` predicts, that closes the decode half of the entry and the work
continues on prefill and on grouped widths, or stops.

**Released.** `B67` confirmed `12B` stack `C`, so both models now have a confirmed complete
stack to measure against: on `4B` the composition profile's adopted stacks, on `12B` `C` for
the four throughput classes and `B` for `session_warm`. Every other class on both models
keeps its reference, and a geometry candidate for one of those is measured against that
reference, not against a stack that was never adopted.

**Not combinatorial.** Only candidates a profile run actually points at. A geometry that
looks good in a microbenchmark and is not indicated by the profile does not get a run.
### `B53` — A leaked default device, not a kernel defect — closed

**The cause.** `tests/engine/test_ironmule.py` calls `mx.set_default_device(mx.cpu)` eight
times so a small model never competes for the GPU, and restores it none. `pytest-xdist`
hands a worker whole files in sequence, so a worker that ran that file kept the CPU as its
default for every later file it was given. `mx.quantized_matmul` follows the default
device; a custom Metal kernel can only run on the GPU. The comparison then held one arm on
the CPU and one on the GPU and called the difference a bit-identity failure.

**Proved on the retained bytes.** All three dumps from the reproduction:

| running `quantized_matmul` on | matches |
| :-- | :-- |
| the CPU | the stored **library** bytes, exactly |
| the GPU | the stored **kernel** bytes, exactly |

Three states, each in a fresh process on the same stored inputs: the CPU default
reproduces the failure, an untouched default is clean, and setting the CPU and putting it
back is clean. Experiment `B53_device_trigger_20260910`.

**Nothing was wrong with the kernels or with MLX.** The transcription and the specialised
kernel were byte-identical to each other throughout, and MLX's quantised matmul is correct
on each device. Two devices were compared as if they were one.

**Repair, at the cause.** An autouse fixture in the leaking file records the default device
and puts it back after every test. `ironmule/fast.py::_self_check` restores it in a
`finally` as well; it is a script entry point and was never part of `fuse_projections`.
`tests/test_qmv_k3840.py` now states its call contract: it asserts the default device is
the GPU and says why, so a future leak fails as a leak rather than looking like a kernel
defect. No test was removed, no tolerance loosened, and nothing forces the GPU globally.

**Remedy demonstrated.** The leaking file and the comparison file run in one process, in
the worker's own order, and pass. The new contract check fires, with the device named,
when the default is left on the CPU. `B51` rerun in full: ten cases, 22 requests, logit
bit patterns per step and the whole KV state per request identical to independent library
runs (`B51S_identity_gap_20260910`). Unit suite `5691` tests, `0` failures.

**Closed.** The original inputs stay lost; the cause is established on the reproduction
whose inputs were kept. Resolution `B53_resolution_20260910`.

**What it cost, and the lesson.** Five phases of diagnosis treated the library as the
reference and the kernels as the candidate, and the entry twice had to withdraw a
conclusion drawn from that framing. The first probe that saved its inputs settled it in
one comparison. A tripwire that keeps the failing data is worth more than any number of
clean repetitions.

### `B54` — One Metal kernel name, several modules — closed

**Confirmed upstream.** `ml-explore/mlx#3832`: the custom-kernel library cache is keyed by
kernel name in `Device::get_library(name_, …)`, and the stale-source invalidation works
across `eval` boundaries but **not inside one batch**. Affected releases `0.31.1`,
`0.31.2` and `0.32.0`. Reproduced here on `0.32.0` with a two-line kernel: the `+100.0f`
variant returned the `+1.0f` result.

**Fixed by deriving the name.** `ironmule/kernel_registry.py` builds every Metal kernel in
this repository under a name that is a digest over the whole specification: base name,
source, header, input and output names, the row-contiguity and atomic flags, the compile
options and the template values. Equal source bytes are not equal specifications, so the
options and template are in the digest too. The digest is computed once per specialisation
at import, never per call and never per token, and the registry refuses an identifier that
would stand for a second specification. Eight kernels are now registered, one per
specialisation; the copies whose sources are byte-identical collapse onto one name, which
is the case MLX handles correctly.

**Arithmetic untouched.** No kernel source, no compile option and no MLX version changed.
Only the key MLX caches under.

**Verified on the device after the rename.** `B51` rerun in full: ten cases, 22 requests,
logit bit patterns per step and the whole KV state per request identical to independent
library runs, same step counts as the run before the rename
(`B51R_identity_gap_20260910`). The limited non-regression preregistered for the one
runtime path the rename touches: the paired path keeps its advantage over the shipped
throughput path at `0.8751`, CI `0.8714 – 0.8772`, eight of eight blocks under the `0.90`
gate, A/A control passing, zero swapouts (`B54R_paired_nonregression_20260910`).

**Left as a finding, not changed.** `tools/b42_qmv_kernel.py` and `ironmule/qmv_k3840.py`
both define `COMPILE_OPTIONS = {"math_mode": "safe"}` and never pass it, so those kernels
compile under MLX's default math mode. Passing it would change the arithmetic, which this
work was not allowed to do. The digest records the options actually passed, which is
`None`, so the discrepancy can no longer hide.

### `B52` — The profile chooses the service mode, only when asked

**The question.** `B50` produced rules a person can follow. Can the runtime follow them
itself, from the tuned profile, without ever turning itself on?

**Not a knob.** A service mode changes how requests are grouped, not how a kernel
computes, so it is not a `Knobs` field. `ironmule/service_strategy.py` puts a versioned
record beside the knobs, `ironmule.tuned_profile.service_strategy.v1`: the strategy, the
admitted range (hardware fingerprint, model identity, MLX and mlx_lm versions, and the
minimum and maximum number of simultaneously ready requests), the correctness contract,
and the run ids that evidence it. A record missing any field reads back as absent, so an
incomplete record and an older profile mean exactly the same thing: choose nothing.

**Three gates, all of which must open.** An explicit `automatic_service_mode=True` at
load; a complete record in the profile; and this machine, model, library build and ready
count inside the admitted range. Ready means ready: a request that has not arrived yet, or
that finished during prefill, is not counted as a partner, and the status reports the
group size beside it. Only facts known at decision time enter, so the response
a request will eventually produce is not used and no length is predicted. The admitted
range is the ready count, `2` to `4`, which is exactly what `B50` measured. Nothing waits
for a partner. If the profile admits the paired path and the loaded model then refuses
admission, the choice falls back to the established mode rather than through to the
sequential safety net.

**One decision per `serve`.** The ready count is only known once the sessions exist,
which is also the last moment before any token is produced. Measured: `9` decisions for
`9` calls in the timed part, `3` for `3` in the user path. Nothing is added per token.

**Migration on a working copy only.** Writing goes through `tune.save_profile`, the
existing authorised path. The tool refuses to run unless `IRONMULE_HOME` points at a
working copy. This machine has no product profile at all — `~/.ironmule/profiles.json`
did not exist before the run and does not exist after it.

**User path, Gemma 12B, real hardware.** Without the opt-in the record is never read.
With it: one request keeps the established mode, two share `23` steps, four run as two
pairs sharing `46`, and a pair whose partner has not arrived yet keeps the established
mode on one ready request out of a group of two. Every output is identical to an
independent `InteractiveMode` run. A record naming a foreign MLX version is refused with
the reason naming the field. Assigning `ThroughputMode` returns the runtime to the
established mode.

**What the automation costs, preregistered at ±2 per cent.** Eight rotated blocks, an
A/A control, a 95 per cent bootstrap over 10.000 resamples. The release run is
`B52R_automatic_selection_release_20260909`, measured on the tree after the stabilisation
repairs and carrying a source binding over every file it measures. Automatic over manually
naming the same strategy: median `0.9973`, CI `0.9937 – 1.0018`, inside the margin. A/A
control `1.0055`, CI `0.9958 – 1.0068`. Zero swapouts, no fallback.

**One earlier execution is lost.** The first run wrote the same output path as the second
and was overwritten; it is unrecoverable and none of its numbers enters a release claim.
`B52P_evidence_provenance_20260909` records the loss, the sources searched for it, and the
checksums of what survived. Raw records are now written once and atomically, and every
measuring record carries its own code binding. An interval that
merely contains `1.0` was fixed in advance as *not* equivalence, which is why the margin
was set before the run. **READY, off by default.**

**Closed.** The paired path's own gain is `B46` and `B50` and is not re-derived here.
Experiment `B52_automatic_selection_20260909`, tool `tools/b52_automatic_selection.py`,
tests `tests/test_service_strategy.py`.

### `B51` — The eight load cases, bit-identical in logits and KV state

**The gap.** `B50` compared tokens, stop reasons and text. Two different distributions
can share an argmax, so that is weaker than equal logits. The bit-level identity came
from `B45` and `B46` and held for their prompts, not for the eight load cases. It was not
carried over.

**What was compared, outside every timed region.** Each case decodes twice, once on the
shipped body and once through the paired step: the full logit bit pattern of every step,
the whole KV state of every request, the token sequences, the stop reason and the step
count.

**Two real end-token cases added.** A full stop or a forced stop is not evidence. One
request stopped on a genuine end token after `19` steps while the other ran to `64`,
which exercises the pair-to-single transition and the later lone request. In the second
case both stopped on a real end token, at `19` and `17` steps.

**Result: ten cases, 22 requests, everything identical.** Logits, KV state, tokens, stop
reason and step count all match, in every case, including the four-request case running
as two pairs.

**Not in this run.** Arrival times are not varied, so a partner joining late is not
exercised here despite the record's method note; that case is `B47`, on real model
computation. Cancellation mid-flight stays open: `ironmule.service.Request` carries no
cancel handle and `serve()` runs to completion. An interface limit, not a pass.

**Closed.** Experiment `B51_identity_gap_20260909`, tool `tools/b51_identity_gap.py`.

### `B49` — Width four shares one sweep and loses to two shared pairs

**The question.** `B45` shares one weight sweep between two requests. Four ready requests
could ride one sweep instead of two. The generator already took the width, so the
per-request arithmetic is untouched: same thread mapping, same eight values per thread,
same 256-value blocks, same `qdot`, same partial-sum order, same `simd_sum`. Only the
register demand grows, from 24 scalars per thread to 48.

**Not a repeat.** `E3`'s decode-width sweep and `B1` are about the scheduler's group
width. This is about how many activations ride one weight load inside one kernel call.

**Kernel level, 18 real projections with four different real activations each, 903 MB per
sweep, 20 rotated blocks:**

| comparison | median | CI95 | blocks under `0.95` |
| :-- | --: | :-- | --: |
| four shared / four library calls | `0.8061` | `0.7970 – 0.8115` | 20 / 20 |
| two shared pairs / four library calls | `0.7216` | `0.7165 – 0.7429` | 20 / 20 |
| **four shared / two shared pairs** | **`1.1084`** | **`1.1014 – 1.1205`** | **0 / 20** |

All four outputs are bit-identical to independent library calls. A/A control passes, `0`
swapouts. **KERNEL NO-GO.** Width four still beats the library, but it loses to the width
two it would replace, by `11%`, in every single block.

**The cause is not measured.** The kernel source shows twice as many scalars per thread,
and that is the obvious suspect, but counting scalars in source is not a measurement of
hardware register usage or spilling. Metal exposes no register or spill counter through
`xctrace` here and the occupancy stream is too large to export proportionately. No
counter was invented, so register pressure remains an unconfirmed explanation.

**No model test, and none pending.** The brief gates it on sufficient measured
potential. Width four is slower than what it would replace, so the product comparison and
the two confirmation sessions do not apply: potential gate not passed. They are not open
work.

**Closed.** Investigation complete, performance decision NO-GO for the tested candidate.
A negative finding closes the entry.

**Status.** The shipped two-request path is untouched, `share_aligned` stays `False`, and
width four exists only as study tooling. It was never wired into a mode.

### `B48` — Sharing the aligned projections too: kernel yes, product no

**A different kernel, not a widened one.** `K=4096` and `K=15360` are multiples of 512,
so the library runs `qmv_fast_impl`: 16 values per thread, 512-value blocks, no tail path
and no `used_out_row` step-back. Enlarging the `K=3840` kernel would have changed the
reduction order, so this is a separate transcription, checked byte for byte against the
library before a second activation was added.

**Where the paired path spends its time now (`B48P`).** With `K=3840` already shared:
our shared kernel `33.9%` of GPU time, library `qmv_fast` (both aligned families)
`22.3%`, library plain `qmv` (the `lm_head`) `7.3%`, prefill matrix matmuls `34.8%`.
Splitting the `qmv_fast` share by bytes puts `down_proj` at about `17.6%` and `o_proj` at
`4.7%`, so `K=15360` was taken first.

**Kernel level, real matrices well past the cache, 20 rotated blocks:**

| family | bytes per sweep | shared / two library calls | sharing alone | bit-identical |
| :-- | --: | --: | --: | :-- |
| `K=15360` | `199 MB` | `0.7874` `[0.7728, 0.7910]` | `0.9392` `[0.9242, 0.9496]` | yes |
| `K=4096` | `159 MB` | `0.8110` `[0.8040, 0.8244]` | `0.9245` `[0.9148, 0.9465]` | yes |

**KERNEL GO** for both. Note the split: most of the `shared / library` figure is the
transcription itself, and the sharing term alone is `6.1%` and `7.6%`, far below the
`17.9%` the `K=3840` shape gave. `qmv_fast` already reads more per thread, so there is
less duplicate traffic left to remove.

**Derived beforehand (`B48D`).** Applying those sharing terms to the measured shares puts
the available saving at `1.4%` of paired GPU time. That is under the `5%` wall-clock
gate, and it was written down before the product run rather than after it.

**Product level, two preregistered sessions against the current opt-in path:**

| | session 1 | session 2 |
| :-- | --: | --: |
| candidate / `PairedThroughputMode()` | `0.9903` `[0.9823, 0.9937]` | `0.9920` `[0.9879, 0.9970]` |
| blocks under `0.95` | 1 / 10 | 0 / 10 |
| A/A null control | passes | passes |
| completion latency | `0.992`, `0.990` | `0.992`, `0.991` |

Identity clean in both, a genuine EOS observed in both, `0` swapouts. **GOAL NOT MET.**
The extension is real and statistically separated from `1.0`, worth about `1%`, against a
`5%` goal. The derivation and the measurement agree, which is the useful part.

**A correction.** The combination was measured, not the two families separately, and it
was described here as an upper bound for either alone. That does not follow: register
pressure and scheduling can make two changes interact, so a single family is not
guaranteed to land below the pair. Neither single variant was measured, and no claim is
made about them. The byte-based split of the `qmv_fast` share is likewise an estimate,
not a measurement: both shapes run the same kernel and the profile aggregates by name.

**What this does not say.** The total against `ThroughputMode` was not remeasured and is
not implied; percentages from separate studies are not added.

**Status.** The `K=3840` opt-in path is untouched and still the qualified one. The
extension exists as `PairedThroughputMode(share_aligned=True)`, off by default, and stays
off: it did not earn its gate.

### `B47` — The paired path as an opt-in mode, off by default

**The surface.** `PairedThroughputMode` sits beside `InteractiveMode` and
`ThroughputMode` in `ironmule/service.py` and is chosen the same way. There is no CLI
flag because modes are a library choice in this runtime; `docs/PAIRED_OPT_IN.md` records
the actual calls. `paired_status(mode)` answers for any mode, so *disabled* is a real
answer, and it separates enabled, admitted, steps that shared a load, and steps taken
alone for want of a partner. Admission runs once at load; nothing per token hashes a
model or re-checks a version.

**Operational cases, on real model computation (`B47`).**

| case | result |
| :-- | :-- |
| genuine EOS on one side | request one stopped on a real end token after 19 steps while the other ran to 64; tokens and stop reasons identical |
| unequal lengths | 6 and 20 tokens, identical, and the pair returned to the single path |
| late partner | 4 solo steps, then 13 paired: a partner joins only at a step boundary |
| lone request | 0 paired steps, 7 solo: it never waits |
| injected fault in the shared step | existing fallback caught it, both requests completed with no duplicate tokens |

**The gap that stays open.** Cancellation mid-flight could not be exercised:
`ironmule.service.Request` has no cancel handle and `serve()` runs to completion. Reported
as a gap rather than a pass, and no new server was built to manufacture one.

**The surface costs nothing measurable (`B47U`), 8 rotated blocks, one run:**

| comparison | median | CI95 |
| :-- | --: | :-- |
| opt-in mode / shipped `ThroughputMode` | `0.8748` | `0.8729 – 0.8782` |
| opt-in mode / the `B46` research build | `0.9983` | `0.9928 – 1.0034` |

The gate still holds at `0.8748`, and the interval against the research build contains
`1.0`, so wrapping it in a service mode added no measurable work to the request path.
A/A control passes, `0` swapouts.

**User walk.** Load with the default and status reports disabled. Name the mode, run one
request: `0` paired steps. Run a pair: paired steps recorded, output identical to an
independent library run. Switch back to `ThroughputMode`, output still identical. Reload:
default off again. **OPT-IN READY.**

**Status.** Default off. Nothing activates it, no commit, no push, and the separate
`k3840_matvec` knob is a different feature that stays untouched and off.

### `B46` — Pairing against the shipped throughput path

**What the earlier number could not say.** `B45` measured `17.7%` against running two
requests one after the other. The product reference is not serial execution: it is
`ThroughputMode`, which already groups ready requests and submits them asynchronously.
This entry measures against that.

**How it attaches.** `ironmule/paired_research.py` subclasses the shipped
`AsyncGroupedB1Executor`, so admission, sessions, stop rules, telemetry and the
sequential fallback are the existing ones; only the group's inner step changes, through a
new `_step_group` hook that leaves the shipped behaviour identical. With `share=False`
the same pairing runs with separate projections, which is the control. A lone request
takes the ordinary single path and nothing waits for a partner. 240 projections admitted
under the full model, hardware, library and shape gate.

**Correctness, before any timing.** At step level, logit bit patterns and KV digests
match the shipped decode body for both requests, in both `share` modes. At service level,
tokens, stop reasons and output text match `A` across a single request, simultaneous
arrival, staggered arrival, unequal output lengths and a long run.

**Two preregistered sessions, 10 rotated blocks, one resident model:**

| | session 1 | session 2 |
| :-- | --: | --: |
| `C` shared / `A` shipped | `0.8734` `[0.8705, 0.8749]` | `0.8719` `[0.8688, 0.8745]` |
| `B` paired only / `A` | `0.9773` | `0.9782` |
| sharing alone, `C` / `B` | `0.8924` | `0.8927` |
| A/A null control | passes | passes |
| completion latency, `C` / `A` | `0.873`, `0.856` | `0.872`, `0.856` |

Both intervals lie entirely below the `0.90` gate, in every block. **PRODUCT GO**: the
same pair of requests completes `12.7` to `12.8%` sooner than on the shipped throughput
path, with no fallbacks, `0` swapouts and peak memory unchanged.

**Where the win comes from.** Pairing by itself buys `2.2%`, because `A` already groups.
The shared weight load buys `10.7%`. Against the serial baseline `B45` used, the
scheduling term looked far larger; against the real product path it nearly disappears.

**No latency cost this time.** Both requests finish sooner, `12.8%` and `14.5%`, and
service TTFT is unchanged at about `230 ms` in every arm. The serial comparison in `B45`
made the first request wait; against a path that already groups, it does not.

**Validity.** Two fixed prompts, 24 tokens each, greedy, one machine, MLX `0.32.0`, this
Gemma 12B revision, 4-bit weights. Only `K=3840` projections at single-token decode are
shared; prefill is untouched. The long-output case never reached a natural EOS, so stop
behaviour is verified for length stops only. Load and compile time are outside the
numbers.

**Strongest remaining bottleneck.** The projections that are not shared: `o_proj` at
`K=4096`, `down_proj` at `K=15360`, plus attention and the output head.

**Status.** Research mode, not a default. No knob was added to the shipped surface, the
`k3840_matvec` knob is untouched and still off, and nothing activates this.

### `B45` — One weight sweep, two requests, bit-identical per request

**Why this is not `qmv_wide_impl`.** MLX already streams several vectors past one weight
group, and declines to here: `use_qmv_wide` needs architecture generation 15 or newer for
affine quantisation and this M1 Max is `applegpu_g13s`. It is also a different
computation, decoding each group into registers and splitting a row across `k_lanes` with
a shuffle ladder. This entry keeps `qmv_impl`'s thread mapping, `qdot` expression, block
size, partial-sum order and `simd_sum` reduction, and only adds a second activation. What
is shared is the load, not the arithmetic. `B28` changed the computation and failed on the
KV hash; `B29c` overlapped whole sessions and still read the weights once per session.

**Kernel (`B45K`), 18 real projections with two different real activations each, 20
rotated blocks:**

| comparison | median | CI95 | blocks at or under `0.90` |
| :-- | --: | :-- | --: |
| shared width 2 / two library calls | `0.7335` | `0.7280 – 0.7486` | 20 / 20 |
| same kernel, called twice / library | `0.9022` | `0.8852 – 0.9121` | 9 / 20 |
| sharing alone (shared / called twice) | `0.8206` | `0.8132 – 0.8243` | 20 / 20 |

Bit identity holds on every case; a difference would have locked the variant before any
timing. A/A control passes, `0` swapouts. **KERNEL GO**, well past the gate. Width 4 was
not attempted: the gate was set for width 2 and cleared there.

**Potential (`B45P`, derived).** The `K=3840` projections are `61.7%` of a 12B step
(`B24` × `B24S`). Applying the measured kernel ratio to that share projects `0.836` for a
pair, upper bound `0.845`, so the model gate was worth testing.

**Model (`B45M`), two prompts, separate KV caches, separate attention, shared projections
only, two sessions of 10 rotated blocks:**

| session | shared / serial | CI95 | blocks at or under `0.90` | scheduling alone | sharing alone |
| :-- | --: | :-- | --: | --: | --: |
| 1 | `0.8233` | `0.8125 – 0.8270` | 10 / 10 | `0.9027` | `0.9127` |
| 2 | `0.8228` | `0.8125 – 0.8262` | 10 / 10 | `0.9035` | `0.9113` |

Both sessions clear the gate, both are bit-identical in tokens, logits and KV state for
both requests, `0` swapouts, peak memory `7.46 GB`. **MODEL GO**: a pair of requests
finishes `17.7%` sooner than running them one after the other.

**The split matters.** Of that `17.7%`, roughly `9.7%` is the paired loop's scheduling,
which shares nothing, and roughly `8.7%` is the shared weight load. Reporting the total
as a weight-reuse win would be wrong.

**What this is not.** Not a faster single chat: pairing makes the first request wait for
the second, and the serial arm delivers request one at about half the pair's wall time.
Not a comparison against IronMule's grouped server path; the baseline here is the
unmodified model call run twice. Not a general model gain.

**Strongest remaining bottleneck.** The projections that are not shared. `o_proj` at
`K=4096` and `down_proj` at `K=15360` still read their weights once per request, and with
attention, norms and the output head they are the `38%` of a step this cannot touch.

**Status.** Research prototype. Nothing is wired into a runtime path, no knob was added,
and the shipped `k3840_matvec` is untouched and still off.

### `B44` — The `K=3840` kernel as an opt-in knob, off by default

**What shipped.** One knob, `k3840_matvec`, added to the existing `Knobs` dataclass and
default `False`. The kernel is the `B42`/`B43` candidate copied unchanged;
`tests/test_qmv_k3840_integration.py` asserts the shipped source still contains the
qualified one. No second runtime, no new configuration mechanism.

**Admission is narrow and paid once.** `ironmule/qmv_k3840.py` checks the hardware
fingerprint, the MLX and mlx_lm versions, the model identity digest and the architecture,
then every projection's bit width, group size, dtypes, output width, scale geometry and
buffer sizes. `K == 3840` alone admits nothing. It runs inside `load_engine`, after the
identity is attached and before any token; an `Engine` built directly has no identity and
therefore stays on the library path. Nothing in the per-token path hashes a model,
re-checks a version or adds a synchronisation. A refusal raises rather than silently
substituting, and `disable()` restores the original module objects for a fallback.

**What is never routed there.** Prefill and every multi-token input take the library
call on the same buffers, verified byte for byte. Sampling, batching, grouped execution
and any shape outside the admitted set never reach the kernel.

**Correctness of the integration (`B44`).** The model was loaded twice through the
ordinary route, knob off and knob on, over four preregistered prompts including a
96-step continuation. Tokens, the logit bit pattern of every step, the whole KV cache,
the stop behaviour and the step count are identical in all four, including the prompt
that stopped early at 23 steps. 241 projections admitted, 0 declined.

**Speed of the integration (`B44M`), preregistered, 20 paired blocks of 32 steps, one
resident model, one run:**

| | median | CI95 | blocks below 1 |
| :-- | --: | :-- | --: |
| candidate / reference | `0.9921` | `0.9909 – 0.9950` | 18 / 20 |
| A/A null control | passes, interval contains `1.0` | | |

Reference step `32.67 ms`, candidate `32.45 ms`, drift `1.019`, `0` swapouts, peak
memory `7.33 GB`. **INTEGRATION GO.**

**The integration keeps most of the win, not all of it.** `B43` confirmed `1.0` to
`1.3%` for the study prototype; the shipped path measures `0.8%`, interval `0.5` to
`0.9%`. Where the remainder goes is not established: the two were measured in different
harnesses and never against each other, so "the module boundary costs it" is a
hypothesis, not a finding. The knob stays `False`; nothing activates it.

**Validity.** One prompt for the timing, four for correctness, one machine, MLX `0.32.0`,
this Gemma 12B revision, 4-bit weights, greedy, batch 1, ungrouped single-token decode.
The `B43` model GO stays bounded by its own conditions and is not extended by this entry.

### `B42` — A `K=3840` quantised matvec kernel, bit-identical by construction

**Mechanism.** `B24S` put `99.79%` of a 12B decode step's GPU time in the quantised
matrix-vector kernel, and `B41` showed MLX runs its slow variant there because `3840` is
not a multiple of 512. `B41`'s padding bought speed at the cost of bit identity. This
entry keeps the arithmetic and specialises the kernel instead: no padding, no switch to
`qmv_fast`, same reduction order.

**The source proves the identity before any measurement.** In MLX `0.32.0`,
`qmv_impl` uses `values_per_thread = 8` and `block_size = 256`. Its loop runs while
`k < in_vec_size - block_size`, so at `K = 3840` fourteen of fifteen blocks go through
`qdot` and the fifteenth through `qdot_safe` with `remaining` clamped to exactly `8`.
At `N == values_per_thread`, `load_vector_safe` and `qdot_safe` are character-identical
to `load_vector` and `qdot`. Fixing the loop at fifteen `qdot` blocks therefore keeps
every partial sum in the same order while dropping the clamp, the branch and the
bounds arithmetic. Compiled with `math_mode: "safe"`, no extra fast-math freedom.

**Bit identity, measured.** A transcription of `qmv_impl` with runtime dimensions
reproduces `mx.quantized_matmul` byte for byte first, so the harness is proven before
anything is specialised. Then, on 21 real Gemma 12B projections with 21 real decode
activations captured from a live run (`B42C`), both the transcription and the
specialisation match the reference bytes on every case. In the model, four separate
runs agree on tokens, on the first-step logits digest and on the whole KV cache digest.

**Kernel level, two runs of 40 interleaved blocks over 903 MB of real weights:**

| comparison | run 1 median | run 1 CI95 | run 2 median | run 2 CI95 |
| :-- | --: | :-- | --: | :-- |
| `k3840` / transcription | `0.9037` | `0.882 – 0.922` | `0.9263` | `0.887 – 0.960` |
| `k3840` / `mx.quantized_matmul` | `0.9493` | `0.901 – 0.995` | `0.9192` | `0.851 – 0.969` |

Both runs put the interval entirely below `1.0`. The kernel is `7` to `10%` faster than
the code it was specialised from.

**Model level, preregistered at 20 blocks of 32 steps, one resident model, no extension:**

| run | ratio | CI95 | blocks below 1 | rule met |
| :-- | --: | :-- | --: | :-- |
| attempt1 | `0.9458` | `0.9325 – 0.9563` | 16 / 20 | yes |
| attempt2 | `0.9549` | `0.9028 – 1.0128` | 14 / 20 | no |

The medians agree at `4.5` to `5.4%` faster, but only the first run's interval clears
`1.0`. Peak memory `7.33 GB`, swap unchanged by the arm. **Kernel: GO. Model: UNCLEAR
at this point**, resolved by `B43` below.

**`B43` — confirmation, two preregistered sessions in fresh processes (2026-09-09).**
Readiness first: a twelve-second probe before the study counted `0` swapouts and `0`
pageouts against `9.6 GB` of occupied swap, so occupancy alone is not paging. That probe
runs after the earlier sessions, not during them, and therefore says nothing about what
caused their drift; that cause is still unknown. Both sessions then ran on a quiet
machine, one resident model,
20 paired blocks of 32 greedy steps, three arms including an A/A null control, balanced
order, 95% percentile bootstrap over the paired blocks.

| session | reference step | candidate step | paired ratio | CI95 | blocks below 1 | A/A CI95 | drift |
| :-- | --: | --: | --: | :-- | --: | :-- | --: |
| 1 | `31.48 ms` | `31.06 ms` | `0.9872` | `0.9844 – 0.9899` | 20 / 20 | `0.9971 – 1.0050` | `1.020` |
| 2 | `31.45 ms` | `31.14 ms` | `0.9900` | `0.9876 – 0.9914` | 18 / 20 | `0.9973 – 1.0052` | `1.020` |

Both intervals lie entirely below `1.0`, both null controls contain `1.0`, both sessions
report `0` swapouts, drift of `1.02`, peak memory `7.33 GB`, and identical tokens,
logits, KV state and stop behaviour. **Model: GO.**

The honest size of the win is `1.0` to `1.3%` of a decode step, not the `4.5` to `5.4%`
the earlier runs showed. Those ran at step times of `43` to `61 ms` against these `31 ms`;
the drift inflated the ratio rather than the kernel earning it. The earlier runs are kept
as recorded and are not pooled with these.

**Why the model result is noisier than the kernel result.** The machine held `11.2 GB`
of swap and a load average near `4` throughout, from processes this study does not own.
Step times drifted from `33 ms` to `61 ms` across runs while the paired ratios stayed
put. The effect survives that drift in direction and fails it in significance, and no
run was extended to fix that.

**Rejected on the way (`B42R`).** Full unrolling of the fifteen blocks: `7.136x` slower,
bit identity intact. Dispatching with `init_value=0`: one extra dispatch per call,
`68.1` command buffers per step against `61.5`, and the model arm turned negative. Two
resident models: `14.52 GB` peak and a step time of `49 ms`, against `7.33 GB` and
`33 ms` with one.

**Next.** Replicate the model arm on a quiet machine before any activation decision.
Nothing here is wired into the standard path.

### `B41` — Pad the reduction dimension so the fast quantised kernel runs

**Mechanism.** `B24S` measured that 99.3 to 99.8% of a decode step's GPU time is one
kernel family, the quantised matrix-vector product, and that MLX picks between two
variants of it. `affine_qmv_fast` runs only when the reduction dimension is a multiple
of 512. The three local Gemmas fall differently on that boundary, and the measured time
shares follow it exactly:

| model | hidden | intermediate | plain `qmv` | `qmv_fast` |
| :-- | --: | --: | --: | --: |
| 1B | `1152` (no) | `6912` (no) | `98.6%` | `0.8%` |
| 4B | `2560` (yes) | `10240` (yes) | `0.0%` | `99.6%` |
| 12B | `3840` (no) | `15360` (yes) | `74.1%` | `25.7%` |

**Price of the boundary (`B41K`).** A microbenchmark on the device, eight interleaved
blocks with alternating direction, `N=4096`, batch 1: `4096` reaches `362 GB/s` while
`3840`, `4032` and `4160` all sit at `306` to `310`. The jump is `17.6%` and it sits on
the boundary alone, not on size. Those absolute numbers are cache-assisted: at `N=4096`
the weights are `8.4 MB` and a chain rereads them inside the `48 MB` system level cache
(`B41C`). The comparison is unaffected, both arms ran under the same conditions, and the
model-level result below reads `7.2 GB` per token, far past any cache.

**The bandwidth ceiling this competes against (`B41C`).** Same kernel, `K=4096`, `N`
swept so the weight set crosses the cache, five interleaved blocks:

| weights | resident in | median | best |
| --: | :-- | --: | --: |
| `9.4 MB` | cache | `264.5 GB/s` | `315.8` |
| `18.9 MB` | cache | `358.0` | `367.2` |
| `37.7 MB` | cache | `406.2` | `409.2` |
| `75.5 MB` | memory | `319.9` | `324.6` |
| `151.0 MB` | memory | `334.1` | `338.5` |
| `302.0 MB` | memory | `342.2` | `343.7` |

`409 GB/s` is the highest this project has ever measured and it is a cache read. Out of
memory, where a model's weights actually live, the ceiling is `334` to `344 GB/s`. That
confirms the `324 GB/s` floor `B24R` borrowed from `E4`: the roofline it derived stands,
and a 12B step's device time is `80%` of a ceiling that is real.

**The padding is arithmetically exact.** Appending zero packed weights with zero scales
and zero biases makes `dequantize` return exactly `0.0` for the new columns, and the
input is zero there, so every added product is `0 * 0`. No existing byte is touched and
nothing is requantised. `tests/test_padded_qmv.py` asserts the zero.

**Result on a real model (`B41P`).** 12B, 241 padded linears, all `3840 -> 4096`, six
interleaved order-balanced blocks of 20 greedy steps:

| | median step | paired ratio |
| :-- | --: | --: |
| reference | `31.34 ms` | |
| padded | `30.29 ms` | `0.9657` |

Faster in six blocks of six, ratios `0.9485` to `0.9831`, and the two arms do not
overlap: reference `31.14` to `31.82 ms`, candidate `30.09` to `30.62`. **`3.4%` off a
12B decode step.** The 4B null control padded nothing, as every 4B shape is already
aligned, and measured `-0.44%` with a logit difference of exactly `0.0`: the harness
carries no bias of its own.

**What it costs, and the open gate.** Tokens were identical over 20 steps, but the
maximum absolute logit difference is `0.53`, and in the microbenchmark `2.1e-07` on a
128-row case. The added columns contribute nothing; the difference is the other kernel
reducing in a different order. Under `B2`'s rule, any nonzero logit difference makes
this a plan change, not a drop-in optimisation. It also costs `6.7%` more weight bytes
and one pad kernel per distinct input vector.

**Where it does not work.** 1B measured `-5.8%`, slower. Its `1152` pads to `1536`, a
`33%` byte increase that the `17.6%` bandwidth gain cannot repay, and its host share is
the largest of the three. The entry is size-dependent: it wins only where the distance
to the next boundary is small relative to the gain.

**Next.** Qualify it as an explicit opt-in plan against the exact-output contract, or
hold the padded weights so that the per-call pad disappears. Do not activate it while
the logit difference is unqualified.

### `B38` — Exact Gemma 12B core-profile activation/canary

**Mechanism.** B36 qualifies the non-mutating core profile only for the exact
Gemma 3 12B revision, prompt, host and full-hash/prefault protocol. A separate
activation/canary decision must preserve that fingerprint scope rather than
turning the exploratory result into automatic routing.

**Test.** Obtain explicit architecture approval first. Then canary only the
exact revision/quantisation/hardware identity, with fail-closed model/code/
environment/workload fingerprints, correctness, memory, swap, crash and
rollback gates. Keep activation opt-in and compare against the unchanged
baseline under a preregistered rollout budget.

**Kill.** Any identity drift, missing gate evidence, token/stop divergence,
resource regression, inconclusive canary or unapproved architecture change
blocks activation. No activation is allowed now.

### `B1` — Is width 4 still the ceiling at 27B?

**Mechanism.** `W=4` was fixed by `E2`, `E3` and `E14b`, all three measured on 4B.
The pathological `M=8` regime is a property of a matrix shape, and 27B's shapes are
not close (hidden `2560 -> 5376`, intermediate `10240 -> 21504`). `E4` showed larger
matrices reach much higher achieved bandwidth.

**Test.** Sweep 2, 4, 6, 8, 12 at 27B under the standard protocol; repeat the winner
three times; check token identity at every width.

**Kill.** Width 4 wins at 27B too. Then the ceiling belongs to the kernel rather than
the shape, `LIMITS.md` gains a sentence saying it was checked at two scales, and the
`M=8` story is settled. Full detail in [`SCALING.md`](SCALING.md).

### `B2` — Group the `lm_head`, and only the `lm_head`

**Mechanism.** Grouping deliberately never changes a tensor shape. But the four
sessions in a group each run `lm_head` separately at `M=1`, and `E3` measured
`1.307 ms` at width 1 against `1.126 ms` per token at width 4. One matmul is stacked;
the trunk, the cache and the per-session state stay untouched.

**Evidence against.** Rows of a matmul are arithmetically independent, but bit
identity across a shape change is not guaranteed — tiling can alter reduction order.
`E9` is the precedent: plans that differ by `4.31` logits still chose the same token,
and `E10` recorded that as luck rather than a guarantee.

**Test.** Stack only the `lm_head` input across the group. Compare logits bitwise
against the ungrouped run, not just the tokens, across at least the 756-request
corpus `E12` used.

**Kill.** Any nonzero logit difference. Then it is a plan change, not an optimisation,
and belongs behind an explicit plan the way `ReusableSessionPlan` is. Note the payoff
shrinks with model size: `lm_head` is roughly 16% of a 4B step and about 5% of a 27B
step, so this is the wrong direction for the scaling problem.

### `B3` — Unroll k decode steps into one compiled graph

**Mechanism.** `E5` measured `dispatch_us = 6.41` and ~510 kernels per decode step,
which is `3.3 ms` of host work per step — and it accounts for the whole gap between
the trunk's achieved `195 GB/s` and the `324 GB/s` ceiling `E4` measured. Compiling
four sequential steps into one graph submits once instead of four times.

**Evidence against, and it has already bitten once.** Greedy decoding is
data-dependent: step `n+1` needs step `n`'s token. Unrolling requires the argmax and the
cache write to live inside the compiled body, which `E0a` already did for one step. EOS
handling has to become a mask rather than a branch.

The predecessor project compiled decode subgraphs on a *growing* cache and got
**wrong tokens from position 2** (candidate 12), which invalidated a set of earlier
compile measurements along with it. Compiling on a *fixed-shape* cache then worked —
cycle 16, `0.9296`, identical tokens over 18 arm executions — and that is what this
runtime ships. Unrolling several steps is the next move along the same axis, and it is
the axis where a shape assumption has already produced silently wrong output once.

**Test.** Unroll 2 and 4 steps, paired A/B across fresh processes like `E5`, with
token identity as a gate.

**Kill.** No improvement beyond noise, or the compile cost per shape makes it a loss
for short answers.

**B3-U2 pilot (2026-08-28).** The fixed-shape two-step correctness pilot completed
all eight isolated processes and 240 measured requests with identical tokens and
final states, zero fallback, zero Swap delta and no relevant crash. This is not a
performance result. The parent did not persist a separate pre/post host-state record
for every child, so confirmation is blocked and the pilot is not retried or pooled.
Public evidence: `research/raw/B3-U2_public_summary_20260828.json`.

**B3a next gate.** Persist real parent pre/post host state and strictly validate
request IDs and checkpoint events in a newly authorised pilot. Kill on any missing
state/event, token or state mismatch, fallback, cache growth, resource failure or
incomplete pair. No confirmation or activation follows automatically.

### `B4` — Wire the weights against page pressure

**Mechanism.** 27B peaks at `16.78 GB` on a 32 GB machine that also runs an editor, a
browser and, in this project's case, a second agent. If the OS evicts weight pages
between steps, decode pays a page fault at 4 KB granularity. MLX exposes a wired
memory limit; nothing in this repo has ever set it.

**Test.** Standard protocol at 27B with and without a raised wired limit, under a
deliberately loaded machine, three runs each. Record swap activity, not just time.

**Kill.** No difference on an idle machine and no difference under load. Cheap enough
to be worth the certainty either way — and if it *is* the explanation for some of the
27B spread (`0.73pp`, twice 4B's), that matters for every other measurement here.

### `B5` — Fill the group on purpose

**Mechanism.** `LIMITS.md`: with 2–3 token answers the realised width fell to `1.83`
and the gain dropped below threshold in `E15` and `E16`. The executor never waits to
fill a group, by design. A bounded wait — a few milliseconds — would raise realised
width where it collapses.

**Evidence against.** Median latency already worsens by 26% to 54% under grouping.
This makes it worse on purpose, and `E15`'s workload is where it would apply.

**Test.** A wait bounded at 2, 5 and 10 ms on the `E15` short-answer workload. Report
realised width, throughput **and** the full latency distribution, not the median.

**Kill.** Throughput gain smaller than the added tail latency at every bound. Likely
outcome, and the entry exists so the trade is measured once instead of argued.

### `B6` — How does the `M=4` cost ratio move with model size?

**Mechanism.** Not an optimisation — a precondition for `B12`, `B13` and `B14`. At 4B,
`E3` measured a width-4 forward costing `2.545x` a width-1 forward for four tokens.
A larger model is more bandwidth bound and its weight traffic is identical at `M=4`,
so the ratio should *fall* toward 1.0 as the model grows. If it does, every
multi-token-per-forward idea gets cheaper exactly where the grouping gain got worse.

**Test.** `E3`'s isolated forward sweep repeated at 12B and 27B. Two hours of GPU time.

**Kill.** The ratio does not fall. Then `B13` and `B14` lose most of their predicted
advantage at scale and drop several tiers.

### `B26` — Qwen3.8 27B, to separate model size from model family

**Mechanism.** 4B, 12B and 27B are all Gemma 3, so size and family are fully
confounded and the falling gain in [`SCALING.md`](SCALING.md) has two live explanations.
`mlx-community/Qwen3.8-27B-4bit` is the cleanest available discriminator: the same
parameter count as the Gemma 3 27B already measured, at **4 bit, group size 64** —
identical quantisation, so the validity box changes in one dimension instead of three.

The shapes are close enough to compare and different enough to matter:

| | hidden | intermediate | layers | kv heads | vocab |
| :-- | --: | --: | --: | --: | --: |
| Gemma 3 27B | 5376 | 21504 | 62 | 16 | 262144 |
| Qwen3.8 27B | 5120 | 17408 | 64 | **4** | 248320 |

Near-identical depth and width, and **a quarter of the KV heads**. If the gain lands on
the Gemma line, the trend belongs to size and is worth preregistering. If it does not,
it belongs to architecture, and the four-fold difference in KV traffic is the first
place to look.

**Test.** The standard protocol — strict plan, 6 requests, 48 max tokens, three runs —
with no knob touched, exactly as X1 was run. Escalate the way the Gemma 27B run did: a
two-request probe first, then half size, then full. Report peak memory separately; this
checkpoint carries a vision tower the text path never uses, so it is not comparable with
Gemma's `16.78 GB` without saying so.

**Before starting.** The checkpoint is roughly 16 GB on disk and needs comparable
headroom in unified memory. The Gemma 3 27B weights were deleted to make room, which
costs nothing evidential — X1's nine raw 27B result files are in `research/raw/` — but
it does mean a re-measurement of the Gemma side is a 16 GB download away, not a command
away.

**Kill.** Nothing. Either answer closes a confound that currently limits every
conclusion in `SCALING.md`, which is why this sits in Tier 1 despite a guessed payoff of
zero percent.

**Compatibility result.** The separate X2 gate qualified the type-preserving hybrid
adapter and leaves this performance/family study open for its preregistered,
three-repeat measurements.

### `B30` — Widen Qwen grouped batch-1 groups to five and six

**Mechanism.** The simultaneous six-request Qwen workload currently forms width-4
plus width-2 independent fixed-state batch-1 groups. A Qwen-only scheduler ceiling
of 5/6 can submit more independent B=1 forwards before the existing barrier while
leaving fixed cache tensors, model calls and grouping semantics unchanged. Gemma
stays at width 4.

**Success gate.** On the strict six-request, 48-token workload, candidate versus
Interactive paired median executor/decode rate must be at least `1.10`, with no
regression versus current Throughput. Physical/visible tokens, stops, counts and
final `kv_hash` must match fresh Sequential and current Throughput controls; memory,
swap and fallback gates must pass.

**Test.** Run widths 4, 5 and 6 through the mandatory pre-timing gate, then the
standard balanced two-warmup/three-repeat pilot. Keep model calls at independent
B=1 and do not add native conversion, cache merging or compilation.

**Kill.** Any divergence, state/hash mismatch, fallback, memory/swap pressure,
candidate below `1.10`, regression versus current Throughput, or Gemma regression.

**B30 correctness/resource event (2026-08-27).** The real integration gate passed
at widths 4/5/6 with exact tokens, stops, counts and state hashes, zero fallbacks,
exit `0`, and `152.16 s` wall time. No token-ID artifact exists, so this is not a
product or performance claim. Swap rose from `505.75 MiB` to `2676.69 MiB`
(`+2170.94 MiB`); free-memory telemetry moved `82%` to `87%`, with no crash.
B30a is therefore not run until reboot/resource reset; all preregistered gates and
thresholds remain unchanged. Raw: `B30_correctness_gate_20260827.json`.

**B30a pilot result (2026-08-27).** The mandatory correctness gate again passed
at widths 4/5/6 with exact output/state and zero fallback. Warmup 0 produced
Interactive `15.8483`, Throughput `16.4353`, and candidate `16.4240` executor
tokens/s. The safety checkpoint then stopped the pilot: swap delta was
`314111427 B` and peak MLX memory `23882126950 B`. No measured repeat ran;
classification is `INCONCLUSIVE` and no performance claim is made. B30b is
required after reboot/resource reset; gates and thresholds remain unchanged.

---

## Tier 2 — structural. Weeks, and they change the shape of the runtime.

### `B8` — A decode loop that does not go through Python per operation

**Mechanism.** `6.41 µs` per dispatch is not a GPU number. ~510 kernels per step at
that rate is `3.3 ms` of host time per decode step, against a `~10.1 ms` trunk. A
loop written against MLX's C++ API, with no interpreter and no GIL between kernels,
should submit the same graph in a fraction of that.

**Evidence for.** `E14b`'s clearest result: arm B's host submission per request *rises*
from `6.24` to `9.16 ms` as the group grows, while completion wait collapses to
`1.27 ms`. **Host submission is the saturating resource**, not the GPU.

**The catch.** This is the entry that most directly shrinks the published gain. If
host submission stops being the bottleneck, there is much less left to overlap. Decide
which number is being reported before starting — see the warning at the top.

**Test.** Port one decode step, measure it standalone against the Python path first.
No integration until that microbenchmark says the premise is right.

**`B24` discharged this dependency on 2026-09-09.** The four-way wall-clock split
could not separate host from device cost; the Metal System Trace can. Measured on an
ungrouped batch-1 step, host time is `2.72 ms` at 1B, `3.27` at 4B and `5.50` at 12B,
against steps of `8.48`, `13.23` and `32.86 ms`. That is the ceiling for anything here:
at most `32% / 25% / 17%`, falling as the model grows. Encoders per step are `16 / 22 /
35`, but encoders are not dispatches, so the `~510 kernels` premise is still unmeasured.

**Kill.** Under `2x` improvement on the isolated step. Then `6.41 µs` is Metal's
enqueue cost rather than Python's, and `B9` becomes the only remaining route.

### `B9` — Record the decode step once, replay it every step

**Mechanism.** The decode step is a fixed sequence of kernels over fixed shapes with a
fixed cache — the exact case Metal's indirect command buffers exist for. Encode once,
replay per step, and the per-dispatch host cost goes away almost entirely.

**Evidence against.** MLX does not expose this today. It means patching MLX or
building alongside it, and it welds the runtime to one framework version — which the
fingerprint mechanism was built to police, not to encourage.

**Test.** Prototype outside IronMule: a hand-built ICB replaying one transformer
block, timed against the same block through MLX.

**`B24` discharged this dependency on 2026-09-09.** The four-way wall-clock split
could not separate host from device cost; the Metal System Trace can. Measured on an
ungrouped batch-1 step, host time is `2.72 ms` at 1B, `3.27` at 4B and `5.50` at 12B,
against steps of `8.48`, `13.23` and `32.86 ms`. That is the ceiling for anything here:
at most `32% / 25% / 17%`, falling as the model grows. Encoders per step are `16 / 22 /
35`, but encoders are not dispatches, so the `~510 kernels` premise is still unmeasured.

**Kill.** The replay is not meaningfully faster, or the shapes turn out not to be
stable enough across steps to reuse an encoding. Highest ceiling of anything in Tier 2
and the highest chance of being abandoned halfway.

### `B10` — Fewer kernels per step

**Mechanism.** 510 kernels for 34 layers is roughly 15 per layer. `E5` measured what
happens to a naive attempt: fusing q/k/v removed 102 matmul dispatches but added 170
`mx.split` slices, a net `+68` kernels, and the bandwidth gain and dispatch loss
cancelled exactly. The lesson is that fusion must not reintroduce kernels — a real
custom kernel for the block, not a rearrangement of existing primitives.

**Largely closed by `B24S` (2026-09-09).** The shader timeline puts `99.3` to `99.8%`
of a decode step's GPU time in the quantised matrix-vector kernel. Everything this entry
would fuse away, norms and adds and copies together, is `0.2` to `0.7%`. Fewer kernels
is no longer the question; the one kernel that matters is, and `B41` is what came of
looking at it. The dispatch count itself is still unmeasured, and no longer needed to
size this entry.

**`B24` discharged this dependency on 2026-09-09.** The four-way wall-clock split
could not separate host from device cost; the Metal System Trace can. Measured on an
ungrouped batch-1 step, host time is `2.72 ms` at 1B, `3.27` at 4B and `5.50` at 12B,
against steps of `8.48`, `13.23` and `32.86 ms`. That is the ceiling for anything here:
at most `32% / 25% / 17%`, falling as the model grows. Encoders per step are `16 / 22 /
35`, but encoders are not dispatches, so the `~510 kernels` premise is still unmeasured.

**Kill.** Kernel count is already near the floor for the primitives available, or a
fused block kernel underperforms the library's tuned matmuls — which is the usual
outcome and should be the expectation.

### `B11` — Pipeline layers across the group instead of whole steps

**Mechanism.** Grouping currently submits four whole steps and waits once. Interleaving
at layer granularity — layer `i` of session A, then layer `i` of session B — keeps the
same shapes but gives the scheduler a finer-grained stream and lets one session's layer
`i+1` overlap another's layer `i` synchronisation.

**Evidence against.** `E14b` showed the gain already comes from overlap that MLX's
async submission achieves on its own. This may be re-implementing what the runtime
already does, one level up and slower.

**Test.** A microbenchmark on two sessions before any executor change.

**Kill.** No improvement over the current whole-step grouping at width 4. Likely, and
cheap to establish.

### `B12` — Jump over the `M=8` valley

**Mechanism.** `E3`'s own table, which nobody has acted on:

| width | full | ms/token |
| --: | --: | --: |
| 1 | 11.909 | 11.909 |
| 4 | 30.307 | **7.577** |
| 8 | 66.856 | 8.357 |
| 16 | 71.807 | **4.488** |

Width 16 is the best point in the entire sweep, by a wide margin. `M=8` is a valley,
not a ceiling — and every conclusion in this project stopped at the near side of it.

**Evidence against, and it is serious.** This requires true tensor batching, which
`E14b` measured producing a reproducible one-token divergence at `b = 8` — row 3,
index 6, `1437` against `1580`, in all four blocks (blocks, not OS processes — see
`E15` limitation `M2`). `E14b` also measured `C8` as
*worse* than `C4` on throughput. Correctness decides before speed does.

**Test.** `E14b`'s arm C extended to `b = 16` and `b = 32`, correctness first: if
divergence grows with batch size, the entry closes on that alone. Only if token
identity holds at 16 does the throughput number mean anything.

**Kill.** Divergence at 16 or beyond. Then `LIMITS.md`'s "no true tensor batching"
becomes permanent rather than provisional, and `E3`'s width-16 row is a curiosity
about kernels rather than a route.

---

## Tier 3 — algorithmic. Real speedups with a quality cost that must be bounded.

### `B13` — Speculative decoding with an actual draft model

**Mechanism.** `E0c` rejected speculation, but it drafted with prompt-lookup n-grams
and got `0.17` acceptance. A small model drafting for a large one is a different
proposition, and `E3` already measured the verification side: four tokens verified in
one `M=4` forward cost `2.545x` a single step, so break-even sits at about `2.6`
accepted tokens out of four.

**This has been tried once already and rejected.** The predecessor project ran a 1B
draft against a 4B target and recorded `0.560x` (candidate 11, and see the sign-convention
warning in Tier 0). What is different here is only the ratio: a draft costs roughly the
same either way, while the target it saves grows `6.75x` from 4B to 27B, and per `B6` the
`M=4` verification premium should shrink at the same time. **That is a reason to
re-measure, not a reason to assume the earlier result does not apply.** Read the cycle's
raw numbers first; if the rejection was about acceptance rather than about the target
being too small, none of this changes it.

**Why it points the right way.** The bigger the target model, the cheaper the draft is
in relative terms, and per `B6` the `M=4` verification ratio should fall as well. This
is the one entry whose payoff *grows* exactly where the grouping gain shrank. Drafting
`k=3` and verifying at `M=4` also lands on `E3`'s optimum by construction.

**Correctness.** Greedy verification is exact: a speculative step either produces the
token the target would have produced or falls back. That is a structural guarantee of
the same kind `E9` found for chunked prefill, not a statistical one.

**Test.** Measure acceptance first, on this project's own workloads, before any
integration. Acceptance is the whole entry.

**Kill.** Acceptance below `0.65` at `k=3`. Then the arithmetic never closes.

### `B14` — A draft head trained on the target's own hidden state

**Mechanism.** The EAGLE/Medusa family: instead of a separate draft model, a small head
on the target's last hidden state proposes the next few tokens. No second set of
weights to hold, much higher acceptance than an independent draft, same exact greedy
verification.

**A shortcut worth checking first.** `mlx-community/Qwen3.8-27B-MTP-4bit` ships a
multi-token-prediction head already trained. If `B26` is run anyway, that variant makes
this entry testable without training anything — measure acceptance on it before deciding
whether the idea is worth building.

**Evidence against.** It needs training, which nothing in this project currently does,
and the head is model-specific — a new head per model, per quantisation. That fights
the fingerprint discipline directly.

**Test.** Only after `B13` shows the verification arithmetic closes. Sequence matters.

**Kill.** `B13` fails, or the head cannot be trained to acceptance meaningfully above
an independent draft's.

### `B16` — Lower or mixed precision

**Mechanism.** Decode is bandwidth bound. Fewer bits is directly fewer bytes: 3-bit
weights would be ~25% less traffic. Mixed precision — 3-bit for the large MLP
matrices, 4-bit for attention — is the usual compromise.

**Evidence against.** This changes the model, not the runtime, and every performance
number in this repo is scoped to `4 bit, group size 64` by the fingerprint. Quality
would need `E13`'s paired design, on a set that is not contaminated the way SQuAD is.

**Kill.** Quality loss beyond a preregistered bound. It should be preregistered
*before* the speed is measured, or the temptation to accept the loss is obvious.

### `B17` — Quantise the KV cache

**Mechanism.** Decode reads the KV cache as well as the weights. At 2048 context this
is small next to 13.5 GB of 27B weights; at 8192 it is not, and `LIMITS.md` puts the
hard refusal there.

**Test.** Measure the KV share of per-step traffic at 2048 and 8192 first. If it is
under 10%, close the entry without implementing anything.

**Kill.** KV traffic is a small share at every context this runtime allows.

### `B18` — Skip layers that do not change the answer

**Mechanism.** 27B has 62 layers. Early-exit and layer-skip work claims many decode
steps converge well before the last layer.

**Evidence against.** No exactness guarantee exists, unlike `B13` and `B15`. This is a
quality/speed trade dressed as an optimisation, and it interacts with the fixed cache:
a skipped layer still needs its KV written or the next step's attention is wrong.

**Kill.** Any implementation that cannot state what it costs on `E13`'s design.
Listed for completeness and deliberately ranked below everything exact.

### `B19` — Use Gemma's 5:1 local-to-global layer pattern

**Mechanism.** Gemma 3 alternates sliding-window local attention with global layers.
The local layers never need KV beyond 1024 tokens. If the fixed cache allocates full
capacity for every layer, five sixths of the layers are holding — and reading — KV
they cannot attend to.

**Evidence.** `E12` deliberately spanned the 1024 boundary and found reuse bit-exact
across it, so the boundary is understood. Whether the allocation exploits it is a
different question and one nobody has checked.

**Test.** Read what the fixed cache actually allocates per layer. Possibly a
five-minute answer with no code at all.

**Kill.** MLX already handles it. Quite likely, and worth the five minutes.

---

## Tier 4 — the wild ones

Written down because the cost of recording a bad idea is one paragraph, and because
two of these are only wild in effort, not in mechanism.

### `B20` — Put `lm_head` on the Neural Engine

**Mechanism.** The ANE sits completely idle during every measurement in this repo. It
is a second compute unit on the same unified memory. `lm_head` is one large,
self-contained matmul with no data dependency on the rest of the block until the end
of the step — the one piece of the model that could plausibly run somewhere else
*while the GPU works on the next step's trunk*.

**Evidence against.** Reachable only through CoreML, which means a converted model
graph, its own quantisation story, and a bridge on the critical path. The ANE's own
latency for a `262144 x 2560` matmul is unknown and could easily exceed the GPU's
`1.3 ms`.

**Test.** Time the isolated matmul through CoreML on the ANE before anything else. One
number decides the entry.

**Kill.** ANE slower than `1.3 ms`, or the transfer cost exceeds the overlap gained.

### `B21` — Small projections on the CPU while the GPU does the big ones

**Mechanism.** `E4`'s most underused finding: `k_proj` and `v_proj` are `1.4 MB` and
achieve `103.7 GB/s`, a third of what the same hardware reaches on large matrices. At
`M=1` these are matrix-vector products — the case Apple's AMX units are built for —
and `E2` measured a `~0.49 ms` fixed cost just to dispatch to the GPU at all.

**Evidence against.** Synchronising CPU and GPU per layer would cost far more than
`1.4 MB` of traffic, so this only works if a whole set of small operations moves
together and stays there.

**Test.** Time a `1.4 MB` 4-bit matvec on the CPU against `E4`'s GPU number. Again,
one microbenchmark decides it.

**Kill.** CPU not faster on the isolated operation, which is the likely result.

### `B22` — Two processes, one GPU

**Mechanism.** Not a speedup — a control. If two independent processes each running
batch-1 achieve overlap comparable to grouping inside one process, then the gain is
the GPU scheduler's and the executor is taking credit for the operating system's work.

**Evidence for running it anyway.** `E16` already runs across forty OS processes and
found the gain replicates *within* each. This asks the complementary question and it
is the cheapest possible attack on the project's central claim.

**Test.** Two processes at 4B (`2.78 GB` each, comfortable), same total request load,
against one process grouping at width 2.

**Kill.** Nothing — either answer is worth knowing, and a project that publishes
negative results should be willing to aim one at itself.

### `B23` — A weight layout built for `M=1`

**Mechanism.** Quantised weights are stored for general matmul. At `M=1` every kernel
reads a full tile to use one row of activations. A layout swizzled for matrix-vector
access might reach closer to `E4`'s `324 GB/s` on the small matrices that currently
manage `104`.

**Evidence against.** It means a custom Metal kernel competing with a tuned library
implementation, and a weight conversion step at load time.

**Kill.** A hand-written matvec kernel does not beat `quantized_matmul` at `M=1` on the
small shapes. Expect it not to.

### `B24` — Stop measuring the GPU with a wall clock

**Mechanism.** Not a speedup — instrumentation, and `B10` depends on it. `LIMITS.md`
records that kernel counts are retired because MLX exposes no machine-readable dispatch
counter, and that `completion_wait` is a wait rather than GPU time. Metal's own
counter sampling and Instruments give real per-kernel GPU time.

**Why it is ranked here.** Everything in Tier 2 is currently reasoned from `6.41 µs`
and "~510 kernels", numbers that came from careful inference rather than measurement.
Several entries above could be answered in an afternoon with a real profile, and one
of them might be answered *differently*.

**Kill.** Nothing. This is the entry to do first if Tier 2 is ever seriously attempted.

**B24 answered (2026-09-09).** Instruments 16.0 `Metal System Trace` records
what MLX does not expose. Joining the compute-channel GPU intervals to the
process's own command buffer ids, over 32 measured greedy decode steps after a
quiet gap, three runs per model:

| model | GPU share of step | GPU ms | host ms | step ms | encoders/step | command buffers/step |
| :-- | --: | --: | --: | --: | --: | --: |
| `gemma-3-1b-it-4bit` | `0.684` | `5.81` | `2.72` | `8.48` | `16.0` | `16.1` |
| `gemma-3-4b-it-4bit` | `0.753` | `9.96` | `3.27` | `13.23` | `22.0` | `26.3` |
| `gemma-3-12b-it-4bit` | `0.833` | `27.46` | `5.50` | `32.86` | `35.0` | `56.1` |

The device, not the host, holds the majority of an ungrouped batch-1 decode step,
and its share **rises** with model size while host time stays nearly flat. That
caps every host-side entry: removing all host time is worth at most `32%` at 1B,
`25%` at 4B and `17%` at 12B, before any of it is actually removable.

**What the device time then says about direction (`B24R`, derived).** With device
time measured, the weight sweep can be priced. Bytes are the language model's own
parameters as loaded; the floor divides them by `324 GB/s`, the best bandwidth the
ledger has ever achieved (`E4`, on a large matmul, so the floor is optimistic and KV
traffic is not counted):

| model | weight bytes read per token | achieved | share of best measured | bandwidth floor | device slack | host |
| :-- | --: | --: | --: | --: | --: | --: |
| `gemma-3-1b-it-4bit` | `732.5 MB` | `126.0 GB/s` | `38.9%` | `26.7%` of step | `41.9%` | `32.0%` |
| `gemma-3-4b-it-4bit` | `2.561 GB` | `257.1 GB/s` | `79.4%` | `59.7%` of step | `15.5%` | `24.7%` |
| `gemma-3-12b-it-4bit` | `7.186 GB` | `261.7 GB/s` | `80.8%` | `67.5%` of step | `16.1%` | `16.7%` |

At 4B and 12B the device already runs at about `80%` of the best bandwidth this project
has ever measured, and two thirds of a 12B step is an unavoidable single sweep of the
weights. Host work and device slack together are `40%` at 4B and `33%` at 12B, and both
shares shrink as the model grows. Nothing that reorganises execution can pass that;
only reading fewer bytes per token can — speculation, true batching, or KV geometry.
The exception is 1B, where achieved bandwidth is `38.9%` and the slack is the largest
single share of the step: small matmuls, the `E4` regime, not a host problem.
Experiment `B24R_roofline_20260909_attempt1`.

**Kernel level (`B24S`).** Adding the `Metal GPU Counters` instrument to the same
template turns on the shader timeline, which records every shader run with its name and
duration. Sixteen measured steps per model, compute shaders of the traced process only:

| model | shader intervals/step | quantised matmul calls/step | matmul share of GPU time | everything else | copy intervals/step | copy share of GPU time |
| :-- | --: | --: | --: | --: | --: | --: |
| 1B | `68.1` | `183` | `99.32%` | `0.68%` | `9.2` (13.5%) | `0.371%` |
| 4B | `100.6` | `239` | `99.60%` | `0.40%` | `15.7` (15.6%) | `0.0008%` |
| 12B | `277.2` | `337` | `99.79%` | `0.21%` | `4.6` (1.6%) | `0.00002%` |

Two things follow. **There is no kernel overhead to remove.** Norms, adds, rope,
attention and every copy together are `0.2` to `0.7%` of GPU time, so a perfect
superoptimiser that deleted all of them would return under half a percent of a step.
**Copies exist but are free.** They are up to `15.6%` of the intervals and never more
than `0.4%` of the time, which closes the "replace a copy with address arithmetic" idea
on this path: there is nothing there to win.

The matmul call count is exact and comes from outside the trace, by counting
`QuantizedLinear` invocations in one step: `26 x 7 + 1`, `34 x 7 + 1`, `48 x 7 + 1`. The
shader timeline reports fewer intervals than that, so an interval aggregates dispatches
of one shader within a kick. Interval counts are a floor on kernels, never the dispatch
count, and the `~510 kernels per step` figure stays unmeasured. What the timeline does
settle is where the time goes, and that answer is unambiguous. It also found the split
between `affine_qmv` and `affine_qmv_fast` that became `B41`.

**What it does not settle.** Encoders are not dispatches — one compute encoder can
hold several kernels — so the `~510 kernels per step` figure is neither confirmed
nor refuted. That needs `Shader Timeline: Enabled`, which this recording had off.
The measurement is one ungrouped session; `E14b`'s host saturation was measured
with four grouped sessions and is not contradicted by this. Experiment
`B24_metal_trace_series_20260909_attempt1`, `tools/b24_decode_workload.py` and
`tools/b24_trace_report.py`; traces stay outside the repository.

**B24 capture smoke (2026-08-27).** Installed MLX `0.32` exposed start/stop
capture support and memory counters, but no public machine-readable counter or
profile names were identified. The first tiny smoke failed because the capture
layer was not inserted. A retry with `MTL_CAPTURE_ENABLED=1` succeeded for a
tiny 64-element matmul and produced `/private/tmp/ironmule_b24_enabled_smoke.gputrace`;
there was no timing/performance claim and no crash. The trace is intentionally
not copied into the repository. That decode trace has since been taken with
`xctrace` rather than Xcode, see the entry above; what remains open is only the
dispatch count, which needs `Shader Timeline: Enabled`. Apple's
[GPU counter statistics guidance](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
and [Metal developer tools](https://developer.apple.com/metal/tools/); MLX's
available [active-memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html)
and [peak-memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_peak_memory.html)
APIs remain allocation diagnostics, not GPU counter names.

---

## How these compose

`B13` drafts three tokens and verifies at `M=4` — which is `E3`'s measured optimum, and
the same width the executor already groups at. Grouped batch-1 and speculation are not
alternatives: four grouped sessions each verifying a four-token draft is the same
scheduler with a different unit of work.

`B8` and `B9` shrink the published gain while making the product faster. `B13` and
`B14` raise both. If the goal is a number that goes up *and* means something, the
speculative branch is the honest one to push.

`B24` and `B24R` now put numbers on that sentence. Every host entry competes for
`24.7%` of a 4B step and `16.7%` of a 12B one, and that share falls as the model grows.
Every kernel entry competes for the `15.5%` and `16.1%` of device slack above a
bandwidth floor the device already reaches four fifths of. The entries that read fewer
bytes per token are the only ones not bounded by either number, which puts `B13`, `B14`
and `B19` ahead of `B8`, `B9` and `B10` on measured grounds rather than on taste.

## Rules that apply to every entry here

**Preregister before measuring anything that could ship.** `X1` is exploratory because
it was not preregistered, and it stays labelled that way. An entry graduating from this
file to the ledger needs a sealed preregistration, a threshold fixed in advance, and
repeats across separate OS processes, the way `E16` was run.

**Correctness gates come first and they are gates, not tie-breakers.** `B12`'s ceiling
is the highest in Tier 2 and it is still ranked behind entries with a third of the
payoff, because `E14b` found a reproducible token divergence and that is not a rounding
error to be negotiated with.

**A finished entry leaves this file.** Delete it from its tier the moment it is
answered — this is a list of open work, and an entry that survives its own answer makes
the list lie about how much is left. The result does not disappear, it moves:

- **It worked and shipped** -> a full entry in [`research/LEDGER.md`](../research/LEDGER.md),
  and a line in [`LIMITS.md`](LIMITS.md) if it changed the validity domain.
- **It failed, or it was measured and rejected** -> one line in Tier 0 above, with the
  number and the experiment ID. Tier 0 is the only part of this file that grows.
- **It turned out to be someone else's problem** -> delete it and say so in the commit.

**Add what you learn while you are here.** Anything discovered mid-experiment that
could make the runtime faster belongs in this file the same day, even half-formed,
even if it is probably wrong — with a mechanism and a kill criterion, because an idea
without a kill criterion is a wish. The same goes for a route that turned out to be
blocked: write it into Tier 0 so nobody walks into it twice. This file is only worth
reading if it is the current state of what is known, not a snapshot of one afternoon.
