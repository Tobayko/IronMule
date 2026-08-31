# Speed backlog

Every hypothesis that could plausibly make IronMule faster, including the ones that
are probably wrong. Nothing here is a result. Nothing here has been measured unless
it says so and names an experiment.

This file exists because the alternative is holding twenty half-ideas in one head and
re-deriving them badly six weeks later. An idea written down with its kill criterion
costs nothing to keep and can be refuted by anyone.

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
| `B8` | Native decode loop, no Python per operation | weeks | 10 – 25% absolute | low |
| `B9` | Record the decode step once, replay it | weeks | 10 – 30% absolute | low |
| `B10` | Fewer kernels per step | weeks | 5 – 15% | low |
| `B11` | Layer-level pipelining across the group | weeks | 5 – 15% | medium |
| `B12` | Jump the `M=8` valley to width 16 | days | up to 40% throughput | **high** |
| `B13` | Speculative decoding with a real draft model | weeks | 1.5 – 3x at 27B, **rejected once at 4B** | low if verified greedily |
| `B14` | A draft head on the target model | months | 2 – 3x | low if verified greedily |
| `B15` | Exact-but-pruned `lm_head` | weeks | up to 16% at 4B, ~5% at 27B | low if the bound is proved |
| `B16` | Lower or mixed weight precision | days | 20 – 40% | **high**, quality |
| `B17` | KV cache quantisation | days | 0 – 5% | medium |
| `B18` | Adaptive layer skip | weeks | 10 – 40% | **high**, quality |
| `B19` | Exploit Gemma's 5:1 local/global layers | days | 0 – 10% memory and bandwidth | medium |
| `B20` | `lm_head` on the Neural Engine | months | 10 – 16% at 4B | medium |
| `B21` | Small projections on the CPU while the GPU runs | weeks | 0 – 10% | medium |
| `B22` | Two processes, one GPU | days | 0, it is a control | none |
| `B23` | Weight layout tuned for `M=1` | weeks | 0 – 20% | low |
| `B24` | Real GPU counters instead of wall clock | days | 0, it is instrumentation | none |
| `B27` | Evidence-bound execution strategies | audit first, then weeks | 0 immediate; prevents regressions and unsafe reuse | low for audit, medium for routing |

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

## Architecture track — audit first, not a performance claim

### `B27` — Evidence-bound execution strategies

**Mechanism.** The repository already contains qualified fast paths, fingerprints,
raw measurements and conservative runtime fallbacks, but they are not yet expressed
through one explicit contract equivalent to `ExecutionStrategy + ValidityDomain +
EvidenceRecord = TrustedExecutionProfile`. Representing the existing paths through
that contract should let the runtime select only strategies whose evidence matches the
current hardware, model, framework and workload, while a regression gate prevents a
cleaner architecture from silently discarding a measured gain. The first stage is
read-only inventory and baseline capture; it changes no routing decision.

**Test.** Inventory the ledger, preregistrations, raw results and all runtime selection
boundaries; reproduce the current `main` correctness and performance baseline; produce
a gap analysis; then, only after architecture approval, encode one existing qualified
path without changing its observable selection or fallback behaviour. Require exact
token/stop/count equivalence and a before/after regression comparison under the same
fingerprint and workload.

**Kill/Pivot.** If the current evidence cannot be reconstructed with raw samples and
provenance, if the abstraction requires a parallel execution implementation, or if it
weakens fail-closed correctness/fallback semantics, stop at the audit and keep the
current deterministic routing. Any evidence-domain mismatch must yield
`REVALIDATION_REQUIRED` or the baseline, never an inferred promotion.

**Current result.** D1 was approved and implemented as a non-imported stdlib-only
contract on commit `0b14eb6`; it has no persistence, selection or activation path.
B27d preserved token/resource correctness and found no 12B regression, but its 4B
post/pre screen was common-mode 5.7–6.4% slower in both Interactive and Throughput
arms and therefore remains `INCONCLUSIVE_POTENTIAL_REGRESSION`. B27e did not reproduce
a consistent D1 slowdown: block 0 was within 2%, while mirrored block 1 made the
first-running D1 appear 5.8–7.9% faster than OLD. Its frozen class is
`ORDER_OR_TEMPORAL_DRIFT`, so B27d formally remains inconclusive. The execution
surfaces are byte-identical and D1 stays off the runtime import graph; no neutrality,
causality, qualification or activation claim is made.

Approved D2 is complete: Runtime fingerprint v2 and tuned-profile conditions v2 bind
exact local revision, complete manifest, architecture, quantisation and tokenizer;
incomplete legacy profiles fail closed. The same-day D2b 4B/12B post screen had exact
identities/outputs/resources, zero domain drift or hard failures and passed every 5%
gate, yielding `NO_REGRESSION_OBSERVED`. D2 added no D1 persistence, strategy
selection, routing or activation. This closes the former `R6` identity entry. B27
remains open only at the next explicit architecture decision, not by extending D2 or
rerunning the same control.

---

## Tier 0 — already dead. Do not re-run these.

Listed so the next person does not spend a week rediscovering them.

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

**Depends on `B24`, and it is a hard dependency, not a preference.** The `B7` scaling
run found that `submission_ns` is not host work: on identical work and shapes, arm B
submits `73.53 ms` then waits `10.11`, while arm A submits `50.85` and waits `48.79`.
The larger window is larger *because the device runs inside it*. The four-way split
therefore measures wall-clock windows, not host and device cost separately — so the
question every Tier 2 entry turns on, how much of a decode step is Python, cannot be
answered with that instrument at all. Size these with real GPU counters or not at all.

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

**Depends on `B24`, and it is a hard dependency, not a preference.** The `B7` scaling
run found that `submission_ns` is not host work: on identical work and shapes, arm B
submits `73.53 ms` then waits `10.11`, while arm A submits `50.85` and waits `48.79`.
The larger window is larger *because the device runs inside it*. The four-way split
therefore measures wall-clock windows, not host and device cost separately — so the
question every Tier 2 entry turns on, how much of a decode step is Python, cannot be
answered with that instrument at all. Size these with real GPU counters or not at all.

**Kill.** The replay is not meaningfully faster, or the shapes turn out not to be
stable enough across steps to reuse an encoding. Highest ceiling of anything in Tier 2
and the highest chance of being abandoned halfway.

### `B10` — Fewer kernels per step

**Mechanism.** 510 kernels for 34 layers is roughly 15 per layer. `E5` measured what
happens to a naive attempt: fusing q/k/v removed 102 matmul dispatches but added 170
`mx.split` slices, a net `+68` kernels, and the bandwidth gain and dispatch loss
cancelled exactly. The lesson is that fusion must not reintroduce kernels — a real
custom kernel for the block, not a rearrangement of existing primitives.

**Test.** Count kernels honestly first. `LIMITS.md` records that MLX exposes no
machine-readable dispatch counter, so this depends on `B24`.

**Depends on `B24`, and it is a hard dependency, not a preference.** The `B7` scaling
run found that `submission_ns` is not host work: on identical work and shapes, arm B
submits `73.53 ms` then waits `10.11`, while arm A submits `50.85` and waits `48.79`.
The larger window is larger *because the device runs inside it*. The four-way split
therefore measures wall-clock windows, not host and device cost separately — so the
question every Tier 2 entry turns on, how much of a decode step is Python, cannot be
answered with that instrument at all. Size these with real GPU counters or not at all.

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

### `B15` — An `lm_head` that is exact but does not read all 262k rows

**Mechanism.** Gemma 3's vocabulary is ~262k, so `lm_head` moves `377 MB` at 4B and
about `704 MB` at 27B. Greedy decoding needs only the argmax, not the full logit
vector. Cluster the vocabulary once, compute cluster upper bounds, and only evaluate
rows in clusters whose bound can still win. With a correct bound the result is
**exactly** the same token, not an approximation.

**Evidence against.** `E3` measured `lm_head` at `288 GB/s`, the best achieved
bandwidth in the whole model — this is the efficient part. The pruning logic adds
kernels, and `E5` is the cautionary tale about adding kernels to save bandwidth.

**Test.** Offline first: on real hidden states, what fraction of rows survives a
correct bound? If it is not below a quarter, stop.

**Kill.** Survival fraction too high, or the added dispatches eat the saving. Note the
share falls with model size (~16% at 4B, ~5% at 27B), so this is the wrong direction
for the scaling problem even if it works.

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

**B24 capture smoke (2026-08-27).** Installed MLX `0.32` exposed start/stop
capture support and memory counters, but no public machine-readable counter or
profile names were identified. The first tiny smoke failed because the capture
layer was not inserted. A retry with `MTL_CAPTURE_ENABLED=1` succeeded for a
tiny 64-element matmul and produced `/private/tmp/ironmule_b24_enabled_smoke.gputrace`;
there was no timing/performance claim and no crash. The trace is intentionally
not copied into the repository. B24 remains open for one model decode trace and
Xcode analysis, using Apple's [GPU counter statistics guidance](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
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
