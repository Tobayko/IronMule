# B27 — Evidence-driven execution layer: gap analysis

## Decision first

IronMule already contains most of the mechanisms needed for an evidence-driven
execution layer, but they are split across runtime knobs, caller-owned plans, service
modes, fingerprints, tuner profiles, benchmark JSON and the research ledger. The
smallest safe next change is therefore **not** a new executor and not automatic
routing. It is one pure, fail-closed evidence contract that names the existing paths
without changing which path runs.

The proposed first implementation step requires architecture approval under the
repository rules. Until approved, current routing, profile loading, plans, modes and
fallbacks remain unchanged.

## Evidence used

- base commit `d422fdb00fced3238dfaa6b5e9e993294adb72cd`;
- complete `docs/BACKLOG.md` including Tier-0 dead ends;
- `docs/LIMITS.md`, `research/LEDGER.md`, runtime documentation and review audits;
- ProjectAtlas summaries/slices for Runtime, Executor, Tuner, Fingerprinting,
  Telemetry, Benchmarking and their tests;
- read-only inventory of the branch and preserved local unpublished evidence;
- B27a2 engineering baselines on the exact cached Gemma 3 4B and 12B revisions.

The corpus inventory found 134 occurrences / 92 unique artifacts. Fifty-one are
local-only or ignored and remain local. Of 72 valid JSON artifacts, structural field
presence ranges from 25/72 for baseline/candidate to 64/72 for resources; this alone
rules out treating the existing directory as one homogeneous dataset.

## Existing foundation mapped to the target abstraction

| Target concept | Existing implementation | What is already strong | Missing contract |
| --- | --- | --- | --- |
| `ExecutionStrategy` | `Knobs`, `ExecutionPlan`, `InteractiveMode` / `ThroughputMode`, `SequentialExecutor` / `AsyncGroupedB1Executor` | Existing successful paths are explicit and tested; plans are caller-owned; grouped B1 never changes tensor shape | No single stable strategy ID or record joins plan, mode, knobs, cache, scheduling, sync, memory and graph policy |
| `ValidityDomain` | `ironmule.fingerprint`, `tune.conditions`, hardware probe | Hardware, OS, framework, model ID, plan, mode and workload drift are partly represented and fail closed in several paths | Runtime load does not populate exact model revision/architecture/quantisation; workload buckets and system conditions are inconsistent between fingerprint and tuner profile |
| `EvidenceRecord` | public benchmark JSON, research raw JSON, preregistrations, reviews, ledger | Raw samples, token IDs, stop reasons, balanced orders, resources and negative results exist | No canonical schema, status vocabulary, provenance rule or append-only import boundary; many records are summaries or partials |
| `TrustedExecutionProfile` | tuner `profiles.json`, `load_profile`, `revalidate`, runtime tuned-knob load | Stored profiles are compatibility-checked and canary-revalidated; plan and mode are not silently selected | Profile evidence is overwrite-oriented, lacks exact revision/quantisation, has no formal evidence status and does not distinguish code regression from evidence drift |

The current mechanisms should be adapted, not copied. In particular, an evidence
strategy should reference `Knobs.key()`, `plan_kind()` and the explicit service mode;
it must not reimplement decode, cache or scheduling.

## Protected behavior — must not change

1. `StrictOneShotPlan` and `ReusableSessionPlan` remain explicit caller decisions;
   automatic plan substitution stays forbidden.
2. `InteractiveMode` and `ThroughputMode` remain explicit latency/throughput choices;
   a selector may advise in shadow mode later but may not silently change them.
3. Grouped batch-1 retains independent shapes, width ceiling 4, fair rotation and
   sequential restart fallback.
4. Fixed-shape cache, head-skip, projection fusion and Qwen hybrid-cache compatibility
   remain their existing implementations, with their existing fail-closed boundaries.
5. Token IDs, stop reason, physical/visible count, deterministic state, state isolation,
   no fallback and resource gates precede every speed classification.
6. Local unpublished evidence stays local; public summaries do not masquerade as raw
   measurements.
7. No stock-MLX comparison is claimed until an identical tokenizer/prompt/stop/token
   and process contract is implemented and tested.

The B27a2 protection references are:

| Cell | Interactive outer p50 | Throughput outer p50 | wall ratio [95% CI] | rate ratio [95% CI] | MLX peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 4B | `4141.79 ms` | `3492.57 ms` | `0.84374 [0.83977; 0.84889]` | `1.18520 [1.17802; 1.19080]` | `2.785 GB` |
| Gemma 3 12B | `11306.16 ms` | `10098.55 ms` | `0.87871 [0.87331; 0.91457]` | `1.13804 [1.09349; 1.14508]` | `7.801 GB` |

Both have exact arm identity, zero fallback/correctness errors and zero swap. They are
engineering baselines, not new qualification claims and not pooled with B39d/B40.

## Gaps, ordered by dependency

### G1 — Exact identity is incomplete

`Runtime.load()` stores `model_id` but currently constructs `Runtime` without resolved
revision or quantisation. `fingerprint.build()` can carry quantisation only if the
caller supplies it. `tune.conditions()` also lacks revision, architecture, tokenizer
and model-manifest identity. A trusted profile cannot exist until this is fixed.

### G2 — Status and evidence quality are heterogeneous

The ledger, JSON artifacts and public summaries contain several vocabularies including
`MEASURED`, `QUALIFIED`, `INCONCLUSIVE`, bespoke experimental verdicts and absent
statuses. `valid_for_performance` appears explicitly in only nine of the 72 JSON
records; explicit `activation_allowed=false` appears in eleven and never true. Missing
is not equivalent to false or safe.

### G3 — Profiles are not evidence records

The tuner stores one mutable winner under hardware/model. It preserves useful trial and
confirmation data, but it does not provide immutable EvidenceRecord identity,
preregistration binding, raw-sample references, evaluator separation or invalidation
history.

### G4 — No explicit regression gate

Qualified paths are documented in the ledger and specialized harnesses, but a normal
Runtime/Executor/Telemetry change has no machine-readable gate that resolves the
applicable reference domain and classifies either `CODE_REGRESSION` or
`EVIDENCE_DRIFT`. B27a2 supplies two current-main protection cells but does not itself
implement such a gate.

### G5 — Stock `mlx_lm` fairness is unimplemented

The public benchmark explicitly compares Interactive and Throughput modes inside one
loaded IronMule runtime. It does not compare stock `mlx_lm`. Fresh-process-per-arm and
stock-arm contracts remain open; adding a third arm without exact prompt,
tokenization, stop, output-count, decoding and process equivalence would create a
marketing number rather than a reference.

### G6 — Profiler evidence is partial

The phase/roofline helper correctly separates measured, derived and inconclusive
values, but no canonical producer populates all required traffic and phase inputs.
Kernel count and absolute dispatch time remain retired inferences. This blocks custom
kernel or native-decode attribution, not the evidence-schema step.

### G7 — Researcher, reviewer and evaluator are not system boundaries

Preregistrations and reviews demonstrate the roles operationally, but the code has no
typed separation preventing a researcher candidate from assigning its own terminal
status. Runtime activation must remain outside this first step.

## Smallest proposed architecture change (approval required)

Add one stdlib-only module, provisionally `ironmule/evidence.py`, plus unit tests. It
would contain immutable, JSON-serializable records only:

1. `EvidenceStatus` with exactly `HYPOTHESIS`, `QUALIFIED`, `REJECTED`,
   `INCONCLUSIVE`, `INVALIDATED`, `REVALIDATION_REQUIRED`.
2. `ExecutionStrategy`, referencing existing plan kind, service mode and `Knobs.key()`
   plus explicit cache/scheduling/synchronisation/memory/graph/prefix policy fields.
3. `ValidityDomain`, requiring chip, RAM, GPU configuration, OS, MLX/mlx-lm,
   model ID, exact revision, architecture, quantisation, cache type, workload buckets,
   concurrency and relevant system conditions.
4. `EvidenceRecord`, binding baseline/candidate strategy IDs, domain digest,
   preregistration/code/model/environment hashes, raw sample references, correctness,
   resource metrics, uncertainty and evaluator-owned status.
5. `TrustedExecutionProfile`, constructible only from `QUALIFIED` evidence and
   returning `REVALIDATION_REQUIRED` on any domain mismatch.

Adapters would only read the existing `Knobs`, plan, mode, fingerprint and B27 public
summary shapes. The module would not import MLX, write a profile, select a strategy,
change `Runtime.load`, modify fallback, or execute a model. This makes the abstraction
reviewable before persistence or routing exists.

### Tests and kill criteria for the proposed step

- missing/unknown fields fail closed;
- exact revision and quantisation are mandatory;
- an unqualified or self-classified record cannot create a trusted profile;
- any identity mismatch produces `REVALIDATION_REQUIRED`;
- canonical serialization/digests are deterministic;
- adapters preserve existing plan/mode/knob IDs without substituting them;
- no import edge from current runtime modules to the new module in this step;
- full serial non-integration suite remains green;
- 4B and 12B B27 protection cells are rerun under a separately sealed post-change
  protocol, with exact tokens, zero fallback/swap and raw variation reported.

**Kill/pivot:** if the type layer needs a second executor/cache implementation, cannot
represent the existing qualified paths without lossy aliases, weakens the caller-owned
plan/mode rule, or cannot fail closed on domain mismatch, stop at this audit. Keep the
current deterministic routing and address identity/schema gaps independently.

## Later steps, deliberately not approved by the first change

- append-only evidence store and idempotent legacy import;
- regression evaluator and protected-reference registry;
- fair fresh-process stock `mlx_lm` arm;
- researcher/reviewer/evaluator workflow;
- shadow recommendation and local profile selector;
- any runtime activation, automatic routing, learning model or autonomous experiment.

Each is a separate architecture or hardware decision with its own preregistration and
rollback boundary.
