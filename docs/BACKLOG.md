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

### MOLE1 — Remaining agent-level MCP comparison (2026-09-12)

**Mechanism.** Provenance-bound same-task plans may amortize model orchestration and
validation across repeated `repository_context_bundle.v1` requests. The local MVP is
complete in `ironmole_mcp/`; results moved to `research/LEDGER.md` MOLE1 and
`ironmole_mcp/docs/RESULTS.md`. IronMule inference remains unchanged.

**Remaining test.** Supply real direct-agent and actual Programmatic Tool Calling
adapters, record final responses, actual usage/pricing and handoff/continuation costs,
and evaluate whole held-out repositories against the strongest stored workflow.
Local synthetic MCP timings cannot replace these measurements.

**Kill.** Any contract/permission violation or no amortized advantage inside the
preregistered quality, cost and latency limits leaves adaptation unqualified. Missing
agent evidence leaves the full hypothesis unanswered. LOCAL-1/LOCAL-2 already found
no adaptive benefit in their local regime; do not repeat those unchanged experiments.

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
| `B41` | Pad the reduction dimension to 512 | **measured 2026-09-09** | **3.4% at 12B**, -5.8% at 1B, 0 at 4B | medium, logit difference |

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

### `R15` — The competing-process gate matches directory names

**Mechanism.** `competing_model_process` (research/q3b_residual_swap_canary.py) blocks when a
blocker token (`ironmule`, `mlx`, `gemma`, `qwen`, `llama`, ...) is a substring of any
same-user command line. On 2026-09-26 a web dev server under a directory named "Ironmule
Website Design" tripped it (ledger R14), so every Q3-gated run and q3f's real cleanup test are
blocked on that machine by a process that loads no model. Matching the executable name and
the script/module argument instead of the whole command line would keep real model processes
and drop directory-name matches.

**Test.** The detector's unit tests plus a case per blocker token: a model process by
executable, by `python -m`, by script path, and a non-model process under a directory that
contains the token. Both directions must hold.

**Kill.** Any sealed Q3 artefact hashes this module or its token semantics, or a real model
process could slip through the narrower match. Then the gate stays as it is and the
precondition skip in q3f is the answer.

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

**P2 safety debt (runtime lifecycle) — answered 2026-09-26 (agent decision).** `ab.run`
passes every completed child record as `ABRunError.partial_children` on timeout, start
failure, malformed output and non-zero exit, and a failed cleanup raises (`child ... and
cleanup failed`) rather than returning a short run (`tests/engine/test_ab.py`). A whole
process group needs no signal: the child runs under the q3f guard, which refuses and
records every process operation (`subprocess.Popen`, `os.system`, `os.fork`, `os.forkpty`,
...) before the package imports
(`test_q3f_guard_blocks_and_records_every_process_operation_in_isolated_child`), so a
timed-out child has no descendants to orphan. Reopen if the guard's operation set shrinks.

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

---

## Tier 0 — measured and rejected. Re-open only under the rule below.

- **MOLE1 local adaptive route and indexed preparation.** Experiments
  `MOLE1-LOCAL-1-20260912-attempt1` and `MOLE1-LOCAL-2-20260912-attempt1`:
  no adaptive benefit over C; LOCAL-2 preparation missed its 5% benefit gate.
  Full results: `ironmole_mcp/docs/RESULTS.md`. Agent-level comparisons remain open.

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

- **`B48` sharing the aligned projections too (2026-09-09).** Kernel GO for `K=4096` and
  `K=15360`, bit-identical, but the product gain against `PairedThroughputMode()` was
  `0.9903` and `0.9920` against a `5%` goal. `share_aligned` stays `False`. Full entry in
  `research/LEDGER.md`.

- **`B49` one weight sweep for four requests (2026-09-09).** Bit-identical, but four shared
  over two shared pairs is `1.1084`, CI `1.1014 – 1.1205`, slower in 20 of 20 blocks. Wider
  sharing is not the lever at `K=3840`. Full entry in `research/LEDGER.md`.

- **`B56` a per-request objective (2026-09-10).** Implemented, default unchanged, `NO_GO`
  in `B56` and `B56b` (`research/LEDGER.md`).

- **`B68` MLX command buffer limits (2026-09-10).** `MLX_MAX_OPS_PER_BUFFER` and
  `MLX_MAX_MB_PER_BUFFER` swept in 48 fresh processes on 4B and 12B: no interval below
  `1.0`, three outright losses, `DEFAULT_WINS` on both models, MLX `50/50` stays.
  Experiment `B68_command_buffer_sweep_20260910`. Proposed again on 2026-09-23 as a first
  step; not re-run, because nothing about the mechanism or the hardware changed.

- **`B10` fewer kernels per step (2026-09-09).** Bounded by `B24S`: norms, adds, rope,
  attention and copies together are `0.2` to `0.7%` of decode GPU time, so fusing all of
  them away returns under `0.7%`. Experiment `B24_metal_trace_series_20260909_attempt1`.

- **Fusing `lm_head` with argmax (proposed 2026-09-23; derived, not measured).** The
  saving is the logits write and re-read, about `0.5` to `1 MB` per token against
  `2.561 GB` of weights read per 4B token (`B24R`), and argmax sits inside the `0.40%`
  non-matmul share of 4B GPU time (`B24S`). That bound is below every equivalence margin in
  use. The `fused_argmax` knob already moves argmax into the compiled body; `E11` rejected
  it and `Q3a` asks whether that changes on the final incumbent. Re-open only with a
  profile that shows the head's epilogue as a cost.

- **Judging a module by searching its source text (`B82`).** Four studies wrote checks that
  matched the module's own prose rather than its code. Structural claims are checked on the
  syntax tree. See `research/LEDGER.md` B82.

- **`B76` a contextual state model for the `(4, 8)` stack ratio (2026-09-11).** Ridge over
  `load_1min`, `memory_free_percent` and `swap_used_gb`: prediction error `0.0087` against a
  constant model's `0.0058`, larger than the whole between-session spread. Those features
  carry no signal; a new feature set needs a mechanism first.

- **`B77` plain mean against a Bayesian posterior (2026-09-11).** Identical action sequence
  and regret over fourteen sessions; the data cannot tell them apart until the spread moves.

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

- **`R11` measurement is gated on swap, not on a byte count.** The literal
  `12 * 1024**3` was wrong in both directions: it refused `gemma-3-12b-it-4bit` at a
  `17.51 GB` block that never swapped, and it passed a run that had to be discarded
  because swap climbed `2816 MB`. `ironmule/bench.py` now carries `MemoryGate`, which
  aborts on a *delta* against the run's own swap baseline — an absolute value would
  refuse every run on a machine already carrying backlog — with a coarse peak backstop
  derived from `hw.memsize` rather than typed in. Wired into `e14b_arms.py`,
  `e15_service.py` and `e12_window_falsification.py`; `e16_replication.py` keeps its
  per-child ceiling, which was always coherent because it forks. The self-check replays
  both reference runs from `B7`'s recorded numbers, so the gate is tested against what
  actually happened rather than against what sounds reasonable. A gate that can read
  neither swap nor installed memory reports itself `inert` instead of silently passing.

- **`R9` `ironmule.tune` resolves to the function, not the module.** `__init__.py:35`
  re-exports `tune` from `.tune`, rebinding the name in the package namespace, so
  `import ironmule.tune as m` yields the function and the module is reachable only via
  `importlib.import_module("ironmule.tune")`. Renaming would break the public API, so
  this is documented rather than changed — the decision its kill criterion allowed for.
  Checked across the package: exactly one of fifteen submodules is shadowed this way.
  `tests/engine/test_ironmule.py` asserts that, so a second one cannot appear unnoticed; the
  quirk stays a footnote instead of becoming a pattern.

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

## Known consequence of the 2026-09-11 repository layout change

The research packages moved from the repository root into `research/`. Two sealed
studies name a file by its old path inside their own provenance input list, so they
can still be **replayed and verified** but can no longer **re-seal**:

- `experiments/head_skip_formal/study.py` hashes `friday_h1/statistics.py`. Collecting
  provenance now raises `unavailable provenance input`. Its 16 records verify, its
  script and preregistration hashes match, and its verdict stands.
- `docs/F1_INTEGRATION_VORREGISTRIERUNG.md` links `friday_optimizer/integration.py`.
  The document is a hashed provenance input and must not be edited, so the link stays
  as the record of the tree it was written against.

Neither is fixed by editing a sealed file. Re-running either study on the current tree
would be a new study with its own preregistration, which is what a moved source tree
means anyway.

## Tier 1 — cheap, grounded, worth doing first

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

**Since written (2026-09-10).** The record exists: `ironmule/silicon_profile.py` (`B70`)
loads, matches and reports a fail-closed per-class profile and may not change a route. `B69`
confirmed one geometry, `(4, 8)`, against the stack; it is not activated. What stays open is
this entry's question: letting a qualified per-shape record select a kernel at dispatch.

**What the kernel set may grow into (external proposal, 2026-09-23).** A small library of
variants, each qualified for one shape family and described by shape, dtype, bits and group
size, layout, hardware fingerprint, MLX version and compile options; `B54`'s name digest lets
them coexist. A variant may change only how independent outputs are mapped to threads and
threadgroups. The order in which one output's partial sums are reduced stays the library's,
which is what made `B42` and `B69` provable. Selection is by fixed rules over stored evidence
first; a learned selector enters only if it beats that table including its own cost, which
`B76`'s contextual model did not manage on this machine. A bounded template search over the
same axes (`docs/PRODUCT_RESEARCH_2026-09-05.md`, section 3) comes only once this entry has
more than one qualified variant for some shape: new candidates start locked, run through the
controlled kernel worker, and follow the rule from `B66` that only a shape the profile points
at gets a run.

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

**Sharpened 2026-09-23, from an external proposal.** The hypothesis is not "a faster
language" but less host work around the same GPU operations: same kernels, same dtypes, same
weights, only the submission path changes. Two routes, both native:

- MLX `0.32.0` ships `mx.export_function` and `mx.import_function` (checked 2026-09-23). An
  exported compiled decode step could be replayed from C++ (`mlx::core::import_function`) or
  from Rust through `mlx-c`, without re-implementing the model. First gate: a Gemma decode
  step with its fixed KV state exports, imports and produces the same logit bits and KV bytes
  as the Python path. If it does not, the only route left is re-writing the model forward
  natively, which is a different and far larger entry.
- C++ and Rust are equal candidates; C++ sits closer to MLX and keeps a first prototype
  smaller. A Python extension is acceptable only at the outer boundary: one call per decode
  step, or per `k` steps, never a call back into Python per layer. A native loop that
  re-enters Python per layer does not test this hypothesis.

Measure host time with the `B24` trace protocol, not the wall clock alone:
`compiled_fixed_cache` already removes much per-step Python work (the `B25` caveat in
Tier 0), so what remains of the `2.72` to `5.50 ms` may be MLX's own encoding rather than
the interpreter. The `~510 kernels` premise is still unmeasured; a run with
`Shader Timeline: Enabled` is what would count them.

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

**Measured once on other hardware (`PERF1`, 2026-09-23).** On a Kaggle T4 a draft model
reached acceptance `0.61`, below this kill (`perf1-run3-1d84848f`). That closes the entry for
that CUDA cell only; Apple and a 27B target remain unmeasured.

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

**Sharpened 2026-09-23.** Three things changed since this was written. `B24R` prices the
prize: device slack above the bandwidth floor is `15.5%` of a 4B step and `16.1%` of a 12B
step, but `41.9%` at 1B, so a layout that helps small matrices has most room on the smallest
model. `B42` showed a hand-written kernel can beat the library at one shape without a new
layout (`7` to `10%` kernel, `1.0` to `1.3%` model), and `B66`/`B69` found a byte-identical
`(4, 8)` geometry for `K=3840, N=15360`. A layout candidate is measured against the best of
those at the same shape, not against `quantized_matmul`.

The layout is a reversible repacking of the existing quantised bits, scales and biases,
never a requantisation: unpacking returns the original bytes, and the reduction order stays
the library's so identity can be proven the way `B42` proved it. The artifact is bound to
model revision, quantisation, kernel version and layout version, and fails closed on any
mismatch. On a 32 GB machine it replaces one projection family, not a second copy of the
model. Repacking pays only after `repack cost / saving per token` tokens, stated per workload
class. Hidden copies count as cost: `mx.fast.metal_kernel(..., ensure_row_contiguous=True)`
copies a non-contiguous input.

**Kill, added.** No gain over the best existing kernel for the same shape, or a break-even
beyond the token counts the workload classes actually produce.

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
