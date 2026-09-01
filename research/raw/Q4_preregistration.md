# Q4 preregistration — evidence-bound hierarchical RL and the two-stage hybrid optimizer

**Status:** architecture and data-collection contract only. Frozen before Q4
implementation or hardware execution on 2026-09-01.

This document is a protocol, not a result. It authorises no model process, runtime
selection, profile write, routing, activation, upload, download, installation or
hardware measurement. Q4 implementation must be completed and reviewed first. Each
hardware collection phase requires a separate user-confirmed start and its own
pre-registration/hash before it can run.

## 1. Primary question and claim boundary

The primary question is whether a conservative, hierarchical **offline RL** policy can
choose a sequence of safe runtime configurations that beats simpler methods at equal
evaluation budget on a sealed holdout. The policy is deliberately treated as a
hypothesis. A working Q4 implementation is not evidence that RL works.

The secondary question is whether a two-stage `HybridOptimizer` can combine:

1. a low-level optimizer over the existing ten runtime knobs; and
2. a high-level optimizer over execution and scheduling strategies.

The hybrid is shadow-only during Q4. It may produce a signed, inspectable recommendation
in an offline report, but it may not select a runtime path, alter a profile, change a
caller-owned plan/mode, execute a candidate, or bypass BASE. The Q4 result is a method
comparison, not a shipping or generalisation claim.

The only claim that could close Q4 is:

> On the declared local evidence domain and sealed groups, the named RL method has a
> reproducible, statistically supported advantage over the best simpler comparator at
> the same budget, with no correctness, safety, leakage or regression violation.

There is no claim about every Mac, another model family, the Apple Neural Engine,
absolute GPU work, or a universal optimum. A foreign Mac can extend the evidence domain
only through the signed-bundle protocol in section 13.
Even a successful Q4 comparison is a **local pilot** result for the declared M1 Max,
three local Gemma sizes and frozen workload strata; it is not a universal or cross-Mac
claim.

The final trajectory horizon is **H=17**: steps 0--10 are eleven knob-delta
evaluations, steps 11--15 are five strategy selections for the final knob under the
matching plan pool, and step 16 is terminal revalidation. The equal method budget is
the same 16 candidate decisions; the shared BASE reference is external.

## 2. Existing knowledge: import, use and limits

The corpus is migrated read-only by content hash. Duplicate files, summaries, partials,
reviews, preregistrations, failed runs and negative results remain separate references;
none is silently upgraded or dropped. A result is usable as a performance label only if
its raw samples, evaluator-owned gates, exact identity and frozen preregistration meet
the Q4 schema. The following table is binding.

| Source | Q4 role | What it contributes | What it cannot contribute |
| --- | --- | --- | --- |
| Q2 self-tuning | historical `Q3_VALIDATION` only | Ten-field knob identity, ordered search behaviour, accepted incumbent, exact-token constraint, confirmation structure; Q2 confirmed ratio `0.8568` | No Q4 split; Q2 trial rows lack full raw repeats/action snapshots; Q2 prose does not overwrite raw identity |
| E11 self-tuning | `LEDGER_ONLY` | Independent confirmation that coordinate descent found the known knob winners and rejected known dead controls | Ledger-only: no Q4 observations, transitions, propensities or labels |
| B35 clean archive | historical `Q3_VALIDATION` only | 1B/4B/12B core-profile screen, AB/BA process observations, token/determinism context | Exploratory; hardware identity and evaluator-owned no-crash/raw stop evidence are incomplete; 12B order-sensitive; invalid first 1B attempt is retained but never scored |
| B36 / B36a | historical `Q3_SEALED_HOLDOUT` only | 16 pairs/32 children, 160 measured samples, full repeat-level token/stop/state/resource evidence, exact Gemma 12B domain; qualified non-mutating core profile | One model/workload and repeated two-action comparison, not a broad panel or sequential trajectory; activation remains false; never a Q4 holdout row |
| B27 corpus / D1 | strategy schema and provenance prior | `ExecutionStrategy`, `ValidityDomain`, evaluator-owned evidence semantics, 134 rows/92 contents/72 valid JSON, explicit activation-disabled boundary | Summary-only/partial/legacy artifacts are not labels; B27d is `INCONCLUSIVE_POTENTIAL_REGRESSION`, B27e is `ORDER_OR_TEMPORAL_DRIFT`; no D1 routing |
| E14b | `PRIOR_ONLY` strategy mechanism prior | At `b=4`, grouped submission/sync `+18.02%` and true-batch increment `+20.05%`; exact at widths 2/4 in the teacher-forced pilot | No action conversion; true batching at `b=8` diverged and increased caller latency; one-interpreter/teacher-forced design is not a service qualification or promotion record |
| E15/E16 | `PRIOR_ONLY` grouped-B1 context/failure prior | Throughput/latency trade-off, width fill behaviour, terse-workload miss, 40-process replication context | No action conversion; E16 frozen class is `CONFOUNDED_BY_PROCESS_STATE` despite substantive replication; no adaptive-controller or cross-Mac conclusion |
| X1 | `PRIOR_ONLY` model-size trend | Descriptive 4B/12B/27B trend `+19.24%/+15.42%/+11.81%`, realised width 4 | Not preregistered, one Mac/family/build, three runs; no TRAIN/VALIDATION/HOLDOUT label and no threshold evidence |
| Q3c/Q3d/Q3e/Q3f | censored failure/safety examples | Safety aborts, cleanup uncertainty, process-probe/child-visibility failure modes | Never pool, impute or turn into a reward; no historical Q2 reproduction or new performance claim |

The frozen Q3 audit remains authoritative: dataset SHA-256
`f67d975788763e4238019a3be7afa5394efbe2f2faea3a96a927e7cf522f2e33`, dataset ID
`d4ae0c148e826de85c7aa5338f892b5571481a105f558d463e9d041f63dc82b7`, 14 observations,
12 actions, 160 B36 samples, two complete safe observations, and zero TRAIN rows.
Q4 must not report these counts as statistical qualification. Historical names are
`Q3_VALIDATION` and `Q3_SEALED_HOLDOUT`; E11 is `LEDGER_ONLY`. The Q3 raw split is
preserved; Q4 collection creates a new dataset ID and the namespaced new splits
`Q4_TRAIN`, `Q4_VALIDATION` and `Q4_SEALED_HOLDOUT`. No Q2/B35/B36/E11 artifact is a
Q4 split row.

The prior strategy records have explicit missing cells: E14b is a teacher-forced
decode-step study rather than a complete service panel; E15/E16 cover only their
declared workload/plan strata and do not cover every width, plan, mode, model or exact
knob × strategy combination. These gaps are not filled by labels or factorization;
Q4 measures every required cell anew.

## 3. Architecture: hierarchical, conservative and offline

The proposed internal method name is **EB-HCORL** (Evidence-Bound Hierarchical
Conservative Offline RL). The name is descriptive, not a scientific priority claim.
It combines a discrete action mask, behaviour cloning prior, conservative value penalty,
an ensemble uncertainty estimate, a separate failure-risk critic and doubly robust
off-policy evaluation. It may be implemented with the repository's existing local
dependencies only; installing a model, package or service needs separate explicit user
approval.

The hierarchy is:

```text
state/context -> Stage 1: knob policy -> complete knob action
                         -> Stage 2: strategy policy -> execution/scheduling action
                         -> evaluator-owned gates -> outcome/reward -> next state
```

Stage 1 is conditioned on the current knob configuration and chooses one member of the
closed knob catalogue. Stage 2 is conditioned on the selected knob action, workload
objective and current service state and chooses one member of the closed strategy
catalogue. The policies never receive authority to execute an action. The evaluator
owns correctness, resources, rollback, identity and outcome status.

The learning rule is fixed before labels are viewed:

- separate critics/value heads: a knob FQI head operates only on steps 0--10 and a
  strategy contextual-immediate head operates only on steps 11--15; there is no Bellman
  backup or cross-unit reward transfer from strategy to knob or vice versa;
- a five-member deterministic value ensemble, with a pessimistic lower confidence bound
  (LCB) for performance and an upper confidence bound (UCB) for failure risk. Ensemble
  folds are assigned by complete context/group hash; all trajectories from one context
  co-fold and are never split by trajectory hash;
- invalid, out-of-domain, unexplored or uncertain actions are masked, not assigned an
  invented bad reward;
- exact frozen hyperparameters: knob FQI `gamma=0.9`, ridge `alpha=1`, 20 iterations,
  convergence tolerance `1e-9`, ensemble `5` deterministic context/group-hash folds,
  behaviour penalty `lambda=0.1`, and minimum support `3` grouped contexts; the
  strategy immediate ridge head uses `alpha=1` and no Bellman discount/backup;
- a weighted importance estimate (WIS, ratio clip `10`) and grouped five-fold doubly
  robust (DR) estimator are used only when overlap/support diagnostics pass. Folds are
  split by complete context/group hash, never by individual transitions; all trajectories
  from a context co-fold. Otherwise the report says `OPE_UNSUPPORTED`;
- deterministic seed `Q4-RL-20260901`, fixed fold order, fixed action ordering and
  fixed tie-break `lexicographic(action_id)`;
- no neural model, network service or online fine-tuning is part of Q4.

The policy score is never allowed to trade away exactness. A candidate with a failed
hard gate is a failure/censored transition, not a low-scoring success. Safety and BASE
fallback stay outside the policy and remain the final authority.

For the knob head, the fitted-Q target is frozen as

```text
y = r + 0.9 * (not_knob_terminal) * max_supported Q_prev
loss = sum((Q - y)^2) + 1 * ||w||^2
```

The knob head runs exactly 20 iterations and stops only at tolerance `1e-9`. The
strategy head is a separate contextual immediate ridge fit on steps 11--15; it has no
Bellman target and no cross-unit backup. The hybrid reports the two head/vector outputs
without scalar-adding them.

For either supported head, the frozen conservative score is

```text
Q_LCB = ensemble_mean - 1.0 * ensemble_sd - 1 / sqrt(grouped_support + 1)
```

and the behaviour-cloning penalty added during selection is

```text
lambda * -log(max(behaviour_propensity, 1e-6))       where lambda = 0.1
```

The selected-action score is exactly
`Q_LCB - 0.1 * (-log(max(behaviour_propensity, 1e-6)))`, equivalently
`Q_LCB + 0.1*log(max(behaviour_propensity, 1e-6))`.

The failure critic's frozen upper confidence bound is

```text
failure_UCB = min(1, (failures + 1) / (trials + 2)
                     + sqrt(log(20) / (2 * max(trials, 1))))
```

An action is allowed only when its failure UCB is no greater than the incumbent
failure UCB, its grouped support is at least 3, its state is in-domain and the external
evaluator gates pass. These bounds mask unsupported actions; they never manufacture a
negative reward for missing data.

The strict tabular state encoder is fixed before collection. Its exact feature-vector
ordering is: intercept; model-size one-hot `[1B,4B,12B]`; memory-bucket one-hot;
GPU-core-bucket one-hot; prompt-bucket one-hot; output-bucket one-hot;
concurrency-bucket one-hot; objective one-hot `[LATENCY,THROUGHPUT]`; plan one-hot
`[StrictOneShotPlan,ReusableSessionPlan]`; workload-stratum one-hot
`[homogeneous,heterogeneous,staggered,terse]`; arrival-pattern categorical
`[homogeneous,heterogeneous,staggered,terse]`; current-action one-hot; and
remaining-budget scaled to `[0,1]`. No interactions are admitted unless explicitly
listed above. Unknown model size, memory, GPU-core, prompt/output/concurrency bucket,
objective, plan, workload stratum, arrival pattern or action identity is
`OUT_OF_DOMAIN` and is masked; it is never mapped to an "other" bucket. The exact
same ordering is used for every ridge design matrix.

## 4. Closed action spaces

### 4.1 `ironmule.q4_knob_action.v1`

The canonical knob configuration is exactly the existing ten-field
`KnobAction` vocabulary, with no callable or executable payload:

```json
{
  "schema": "ironmule.q4_knob_action.v1",
  "fuse_projections": false,
  "compiled_fixed_cache": false,
  "fused_argmax": false,
  "head_skip_prefill": false,
  "prefill_into_fixed": false,
  "readback_every": 1,
  "speculate_k": 0,
  "speculate_ngram": 3,
  "capacity_slack": 0,
  "wired_fraction": 0.0,
  "action_id": "sha256(canonical semantic object)"
}
```

Booleans are strict booleans; integer fields are non-negative except
`readback_every >= 1`; `wired_fraction` is finite in `[0,1]`. `action_id` is the
lowercase SHA-256 of canonical UTF-8 JSON with sorted keys, compact separators and
without `action_id`. The Q4 panel must mirror the source-of-truth defaults and search
values in `ironmule/adaptive.py`/`ironmule/tune.py`; a prose duplicate is invalid.
The known candidate values are:

```text
compiled_fixed_cache: [True]
fused_argmax:         [True]
head_skip_prefill:    [True]
prefill_into_fixed:   [True]
readback_every:       [2, 4, 8]
speculate_k:          [4]
capacity_slack:       [128]
wired_fraction:       [0.6]
fuse_projections:     [True]
speculate_ngram:      [3] (declared, not searched by the current Q2 tuner)
```

`prefill_into_fixed` and `speculate_k` remain expected dead controls from Tier 0;
either appearing as a safe winner is a harness failure. A complete panel contains the
BASE action plus every declared candidate exactly once per context. The panel is not
complete merely because a raw file has many samples.

### 4.1a `ironmule.q4_knob_delta.v1`

Each of the eleven Stage-1 decisions at steps 0--10 is a strict delta record, not an
executable configuration:

```json
{
  "schema": "ironmule.q4_knob_delta.v1",
  "stage": "KNOB_DELTA",
  "source_action_id": "current complete KnobAction ID",
  "target_action_id": "legal complete KnobAction ID",
  "changed_field": "one of the ten knob names",
  "target_value": "canonical value for changed_field",
  "action_id": "sha256(canonical semantic object)"
}
```

Exactly one field must differ between `source_action_id` and `target_action_id`, and
the target must be in the frozen knob catalogue. An already-evaluated target cannot be
repeated in the same context/trajectory. On an accepted safe result, the target becomes
the current action; on rejection, failure or censoring, the current action remains the
source. In every transition, `previous_action_id` is the current absolute action ID,
not the delta ID. The evaluator verifies these update semantics; the RL policy cannot
mutate them.

### 4.2 `ironmule.q4_execution_strategy_action.v1`

The strategy action is a closed, path-free description. Its canonical fields are:

```json
{
  "schema": "ironmule.q4_execution_strategy_action.v1",
  "stage": "STRATEGY_SELECT",
  "semantic_class": "exact|risk_probe",
  "plan_kind": "StrictOneShotPlan|ReusableSessionPlan",
  "service_mode": "InteractiveMode|ThroughputMode",
  "prefill_policy": "strict_one_shot|reusable_session",
  "decode_policy": "greedy",
  "scheduling_policy": "sequential|async_grouped_b1|true_batch_risk_probe",
  "grouping_policy": "none|grouped_batch1|true_batch_risk_probe",
  "grouping_width": 1,
  "synchronization_policy": "per_request|group_barrier",
  "cache_policy": "standard|fixed_shape|prefix_reuse",
  "prefix_reuse_policy": "disabled|exact_reuse",
  "memory_policy": "existing",
  "compile_graph_policy": "existing",
  "strategy_class": "EXISTING_EXECUTION_STRATEGY|RISK_PROBE",
  "action_id": "sha256(canonical semantic object)"
}
```

The safe policy catalogue across all contexts is exactly S01--S10 below. Every field is
frozen, including the existing executor semantics; no field is inferred from a label.
For one context, the safe pool is exactly five actions matching its frozen plan kind:
Strict contexts use S01--S05; Reusable contexts use S06--S10. The opposite plan is
not an exact/safe action for that context. The canonical `action_id` is SHA-256 of the
complete semantic object, never hand-authored.
`grouping_width` is restricted to `1..4`; width `8` is not a Q4 policy candidate
because E14b found a reproducible token divergence and throughput regression there.
S11/S12 are separate risk probes, not `ExecutionStrategy` values, are excluded from
safe-panel completeness and are never emitted by a learned policy.

| Label | Plan | Mode | Prefill/decode | Scheduling/grouping | Width | Sync | Cache/prefix | Class |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| S01 | Strict | Interactive | strict_one_shot/greedy | sequential/none | 1 | per_request | standard/disabled | EXISTING_EXECUTION_STRATEGY |
| S02 | Strict | Throughput | strict_one_shot/greedy | async_grouped_b1/grouped_batch1 | 1 | group_barrier | standard/disabled | EXISTING_EXECUTION_STRATEGY |
| S03 | Strict | Throughput | strict_one_shot/greedy | async_grouped_b1/grouped_batch1 | 2 | group_barrier | standard/disabled | EXISTING_EXECUTION_STRATEGY |
| S04 | Strict | Throughput | strict_one_shot/greedy | async_grouped_b1/grouped_batch1 | 3 | group_barrier | standard/disabled | EXISTING_EXECUTION_STRATEGY |
| S05 | Strict | Throughput | strict_one_shot/greedy | async_grouped_b1/grouped_batch1 | 4 | group_barrier | standard/disabled | EXISTING_EXECUTION_STRATEGY |
| S06 | Reusable | Interactive | reusable_session/greedy | sequential/none | 1 | per_request | fixed_shape/exact_reuse | EXISTING_EXECUTION_STRATEGY |
| S07 | Reusable | Throughput | reusable_session/greedy | async_grouped_b1/grouped_batch1 | 1 | group_barrier | fixed_shape/exact_reuse | EXISTING_EXECUTION_STRATEGY |
| S08 | Reusable | Throughput | reusable_session/greedy | async_grouped_b1/grouped_batch1 | 2 | group_barrier | fixed_shape/exact_reuse | EXISTING_EXECUTION_STRATEGY |
| S09 | Reusable | Throughput | reusable_session/greedy | async_grouped_b1/grouped_batch1 | 3 | group_barrier | fixed_shape/exact_reuse | EXISTING_EXECUTION_STRATEGY |
| S10 | Reusable | Throughput | reusable_session/greedy | async_grouped_b1/grouped_batch1 | 4 | group_barrier | fixed_shape/exact_reuse | EXISTING_EXECUTION_STRATEGY |
| S11 | Strict | Throughput | strict_one_shot/greedy | true_batch_risk_probe/true_batch_risk_probe | 4 | group_barrier | standard/disabled | RISK_PROBE |
| S12 | Reusable | Throughput | reusable_session/greedy | true_batch_risk_probe/true_batch_risk_probe | 4 | group_barrier | fixed_shape/exact_reuse | RISK_PROBE |

The frozen canonical action IDs for this exact field order and value vocabulary are:

| Label | Canonical `action_id` |
| --- | --- |
| S01 | `712a6d6ea2cf1bb588fcd74a509f52dac5015b08f3b4bc5ae067232592c3a56a` |
| S02 | `a97214109ad4b9f3c74ee0d3cc69a9925ae63b53d44e23ab4ca4801905e0d7ce` |
| S03 | `9e9dc84b74669691fbcf8e4d4e617fc0bf09a0ebd310a276985cbfe46a75cc5f` |
| S04 | `f27fcef617ab253adab4e86cc4a1a09f6a65cee9ef77ecbcb35ed9bff98ac25f` |
| S05 | `b72cbd0476ec1014c3ac2dcd81163f543a0963a2c34f223c8746edd4bcfd754a` |
| S06 | `5167fcbde4d9bf61ed89b711f2fbd366536e098502721762b1eed48677a3804c` |
| S07 | `37b54ee14dd36c11cf31b49eea0471f32b90069b694858d6ecfe3240dedb8a90` |
| S08 | `b100e7dde084b673fddfb15677021fcdb14639b99d5b6d1ab07999deee6ec4c2` |
| S09 | `9ce45f487d9c9ca362532b61c30227a039261b3e0c360ad4d277bc20e0f250db` |
| S10 | `3c61c473394070a7bc77bf41518ad463a9a3cf7b4d688f7613368a802d44a123` |
| S11 | `8a9b3914099c51543335468f3bec9fc901ad8a047be26f02c74818b26c1c5608` *(risk probe)* |
| S12 | `3c1fd4624bfbff81a87782ce87673f39f1fcaf8f806c8f1a02123596ef20758d` *(risk probe)* |

S02 and S07 explicitly mean the existing `AsyncGroupedB1Executor` at width 1, not a
new sequential alias. S11/S12 retain E14b's true-batch information, including the
approximate `+20.05%` incremental result at width 4, but are risk probes outside the
safe strategy pool and policy budget. True Batch is not an existing `ExecutionStrategy`.
The approximately `+18.02%` grouped submission/synchronization effect remains separate;
Q4 must not multiply or add headline gains from incompatible workloads.

The safe action maps to the existing `ExecutionStrategy` vocabulary and may not invent
a cache, plan, mode, kernel or hardware capability. `TrustedExecutionProfile`
construction remains impossible from a risk probe, exploratory, summary-only or RL
record alone.

### 4.2a `ironmule.q4_risk_observation.v1`

S11 and S12 are recorded only through this non-reward schema:

```text
schema, risk_observation_id, state_digest, action, risk_probe_id, failure_code,
evaluator_gates, evidence_ids, recorded_at
```

`risk_probe_id` is exactly S11 or S12; `action` is the complete frozen risk-probe
descriptor; `failure_code` is evaluator-owned and may be `null` only when every risk
gate passes. Risk observations feed only the failure-risk critic. They are excluded
from reward, OPE, safe-panel completeness, policy action support and the 16-action
method budget. Q3 failures are retained failure examples and are never action labels.

### 4.3 Hybrid action identity

`ironmule.q4_hybrid_action.v1` contains exactly:

```json
{
  "schema": "ironmule.q4_hybrid_action.v1",
  "knob_action_id": "sha256",
  "strategy_action_id": "sha256",
  "stage_order": ["KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"],
  "action_id": "sha256(canonical semantic object)"
}
```

The frozen Stage-2 interaction panel measures the full exact cross-product of all 12
knob actions × the five plan-matching safe strategies (60 cells per context), with
each cell carrying full identity. Composition is eligible only from a measured exact
cross-product; no factorized assumption may fill it. Any unmeasured pair forces Stage-2
BASE fallback and is ineligible for composition. The same knob/strategy pair is never
treated as two observations merely because it was run in a different order. A failed
or missing pair remains missing/censored. No action ID can contain a path, prompt,
code, process command, callback or executable payload.

The targeted interaction panel includes these exact complete knob configurations:

```text
BASE:      all ten fields at the `KnobAction.baseline()` defaults
Q2-current: compiled_fixed_cache=True, head_skip_prefill=True,
            readback_every=2; all other eight fields at BASE defaults
```

The Q2-current values are one named anchor in the full panel, not a claim that Q2 is
Q4 training data; the Q2 artifact remains historical `Q3_VALIDATION`.

## 5. State, transition and dataset schemas

All Q4 records are strict JSON objects with no unknown keys, finite numeric values,
lowercase SHA-256 digests and deterministic canonical IDs. `null` means unobserved; it
must not be converted to zero, a median or a guessed reward.

### 5.1 `ironmule.q4_context.v1`

Required fields:

```text
schema, study_digest, model_digest, model_manifest_digest, workload_digest,
hardware_digest, runtime_digest, time_digest, objective_class, workload_stratum,
arrival_pattern, context_id
```

`objective_class` is exactly `LATENCY` or `THROUGHPUT` and is fixed by the workload
manifest before measurement. `workload_stratum` and `arrival_pattern` are each exactly
one of `homogeneous`, `heterogeneous`, `staggered` or `terse`, fixed before collection.
`time_digest` is the digest of a predeclared collection batch (not an arbitrary wall-
clock timestamp chosen after a result); `study_digest` is stable across all methods,
repeats and stages in that study. `context_id` is SHA-256 of all other fields. The leakage
group key is the canonical tuple, in this order:

```text
study × model × model_manifest × workload × hardware × runtime × time
```

The group key, not a caller-provided split label, controls the split. Every repeat,
arm, order, bootstrap, profile/trial row, preregistration, review and derived summary
from that group stays together. A context or group cannot cross
`Q4_TRAIN`, `Q4_VALIDATION` or `Q4_SEALED_HOLDOUT`; historical Q3 names are never
accepted as Q4 split values.

### 5.1a `ironmule.q4_state.v1`

Required fields are `schema`, `state_digest`, `context_id`, `stage`, `step_index`,
`model_size`, `memory_bucket`, `gpu_core_bucket`, `prompt_bucket`, `output_bucket`,
`concurrency_bucket`, `objective_class`, `plan_kind`, `workload_stratum`,
`arrival_pattern`, `knob_action_id` and `strategy_candidate_index`.
`workload_stratum` and `arrival_pattern` are
required categorical fields, each one of `homogeneous`, `heterogeneous`, `staggered`
or `terse`. For
`KNOB_DELTA`, `knob_action_id` is the current complete configuration; for
`STRATEGY_SELECT` and `REVALIDATE` it is mandatory context, so Stage 2 cannot be
factorized away from Stage 1. Unknown bucket values or a missing Stage-2 knob ID make
the state `OUT_OF_DOMAIN` and mask its actions. `strategy_candidate_index` is `0..4`
only at steps 11--15 and `null` for all other steps. `state_digest` is SHA-256 of the
canonical semantic object without `state_digest`.

### 5.1b `ironmule.q4_trajectory.v1`

Required fields are `schema`, `trajectory_id`, `context_id`, `split`, `horizon`,
`trajectory_status`, `transition_ids`, `terminal_step_index` and `partial_abort`.
`trajectory_status` is exactly `RUNNING`, `COMPLETE` or `ABORTED`; `horizon` is 17;
for Q4 records `split` is exactly `Q4_TRAIN`, `Q4_VALIDATION` or
`Q4_SEALED_HOLDOUT` (historical Q3 names are import metadata only);
`COMPLETE` requires exactly 17 ordered transition IDs, `terminal_step_index=16`, a
terminal `REVALIDATE` transition and `partial_abort=null`. `ABORTED` requires the
partial-abort object below, a terminal current-step marker and `terminal_step_index`
equal to that object's value. A partial trajectory is retained for failure/recovery
analysis and never promoted to an RL training trajectory.

### 5.2 `ironmule.q4_transition.v1`

Required fields:

```text
schema, transition_id, trajectory_id, context, stage, step_index, horizon,
state_digest, action_space, action_id, previous_action_id, outcome_id,
next_state_digest, terminal, split, evidence_ids, behaviour_propensity,
behaviour_policy_digest, seed, decision_budget_index, strategy_candidate_index,
partial_abort
```

`stage` is exactly one of `KNOB_DELTA`, `STRATEGY_SELECT` or `REVALIDATE`;
`step_index` starts at zero and increases by one; `horizon` is exactly **17** for a
complete trajectory: eleven `KNOB_DELTA` evaluations at steps 0--10, five
`STRATEGY_SELECT` evaluations at steps 11--15, and one `REVALIDATE` evaluation at step
16. For a complete transition, `terminal=true` is allowed only at `step_index=16`; a
partial abort instead records its terminal marker inside
`partial_abort` at the current step.
Each `STRATEGY_SELECT` candidate is evaluated independently under the same final knob
and against the same plan-matched strategy BASE reference; no strategy incumbent is
updated between steps 11--15.

The five strategy actions are selected for the final accepted knob from step 10 and
must come from that context's matching five-action plan pool; an opposite-plan or
risk-probe action is invalid.
`strategy_candidate_index` is exactly `0..4` for steps 11--15 and `null` otherwise.
`previous_action_id` is the final knob action ID for each of steps 11--15 (the
last-stage marker), not the preceding strategy candidate; step 16 uses the final
selected strategy marker. For the strategy reward, `c_current` is fixed to the same-
knob strategy BASE reference for every candidate.
`trajectory_status=COMPLETE` is allowed only when step 16 is present, terminal and all
17 transitions are complete and safe. An abort may retain a partial trajectory, but it
is not a complete RL trajectory and is never padded with fabricated transitions.

`behaviour_propensity` is a finite number with `0 < p <= 1`, and
`behaviour_policy_digest` is a required lowercase SHA-256 digest of the frozen policy.
The deterministic coordinate policy has propensity `p=1` for the selected action only;
because it assigns no positive propensity to counterfactual actions, its counterfactual
OPE is `OPE_UNSUPPORTED`. Q4 TRAIN collection uses only controlled seeded uniform safe
exploration: for `k` legal safe actions, every selected action records the exact
`behaviour_propensity=1/k` (risk probes excluded). Unknown, rounded or reconstructed
propensities fail closed.

When a trajectory aborts, `partial_abort` is required and has exactly this schema;
for a non-aborted transition it is `null`:

```json
{
  "schema": "ironmule.q4_partial_abort.v1",
  "status": "PARTIAL_ABORT",
  "failure_state": "PREFLIGHT|KNOB_DELTA|STRATEGY_SELECT|REVALIDATE|CLEANUP|UNKNOWN",
  "terminal": true,
  "terminal_step_index": 0,
  "failure_reason": "known non-empty reason",
  "fallback": "BASE",
  "cleanup_verified": false,
  "raw_artifact_ids": []
}
```

`terminal_step_index` is the current step where the failure became known (`0..16`),
not the intended horizon. `failure_state` and `failure_reason` are evaluator-owned;
`cleanup_verified` is never inferred. A partial-abort record is terminal at its current
step, remains censored and counts toward failure/recovery metrics, but can never make a
trajectory complete. `terminal_step_index=16` still requires a valid final revalidation
to be `COMPLETE`.

`previous_action_id`, `next_state_digest`, `outcome_id` and `evidence_ids` are required
even for a failed transition; an unknown value is recorded as an explicit failure of
the collection contract. `transition_id` is a canonical SHA-256 over the semantic
fields. `behaviour_policy_digest` and `seed` identify the collection policy; they are not
permission to execute it.

### 5.3 `ironmule.q4_outcome.v1`

This extends the existing `AdaptiveOutcome` without weakening it. Required groups:

- evaluator-owned status: `MEASURED|FAILED|INCONCLUSIVE|REJECTED|INVALIDATED|DATA_INSUFFICIENT`;
- raw references: one or more `ArtifactRef` records with SHA-256 and quality;
- measured timing: `total_ns`, `prefill_ns`, `decode_ns` and any strategy-specific
  throughput/TTFT/p50/p95 values, each with samples, median and uncertainty;
- exact correctness: logical/physical/visible token identity, token count, stop reason,
  deterministic state/KV identity and capacity;
- resources: MLX active/peak, RSS peak, swap before/after/delta, timeout/crash,
  fallback count, worker/reap status and hard-gate result;
- rollback: `NOT_REQUIRED|APPLIED|FAILED|NOT_ATTEMPTED`;
- fixed provenance: preregistration, code, model, manifest, environment and workload
  digests, plus distinct researcher/reviewer/evaluator IDs.

`MEASURED` is not synonymous with eligible. A complete safe outcome requires measured
raw samples, all evaluator-owned gates, exact identity, no fallback/crash/timeout,
successful rollback state and non-empty uncertainty. Failed, partial, summary-only,
exploratory and unreviewed outcomes remain visible and cannot be used as successful
labels.

### 5.4 `ironmule.q4_dataset.v1`

Required top-level fields:

```text
schema, preregistration_sha256, dataset_id, source_artifacts, action_pools,
contexts, transitions, outcomes, split_manifest, seed_manifest, no_invented_performance
```

`dataset_id` is the canonical SHA-256 of the complete sorted semantic dataset. Source
artifacts are content-hash references, never absolute local paths. `no_invented_performance`
must be `true`. The dataset must preserve duplicate/conflict rejection, group leakage
checks, action-pool completeness, status ownership and all failed observations.

## 6. Data migration and split contract

Migration is offline and read-only. It deduplicates on content hash, then groups before
splitting. It never rewrites a raw artifact or changes its historical status.

Historical imported rows retain namespaced `Q3_VALIDATION`, `Q3_SEALED_HOLDOUT` or
`LEDGER_ONLY`; they never enter a Q4 split. The target Q4 collection has three new,
disjoint splits, with the following minimum design:

| Split | Contexts | Trajectories/context | Model coverage | Purpose |
| --- | ---: | ---: | --- | --- |
| `Q4_TRAIN` | 12 | 3 | 1B, 4B, 12B; four contexts/model | fit behaviour/value/risk models |
| `Q4_VALIDATION` | 6 | 3 | 1B, 4B, 12B; two contexts/model | seed/hyperparameter/model-selection lock |
| `Q4_SEALED_HOLDOUT` | 6 | 3 | 1B, 4B, 12B; two contexts/model | one-time final comparison |

This is **24 new independent contexts** and **72 complete trajectories**, each with
horizon **17**, for a minimum of **1224 complete transitions** before `OFFLINE_RL`
can become structurally eligible. Each context must use one of the four frozen workload strata below, with no
stratum repeated under the same full group identity:

```text
homogeneous / StrictOneShotPlan
heterogeneous / StrictOneShotPlan
staggered / ReusableSessionPlan
terse / ReusableSessionPlan
```

The assignment of the four strata to the 24 model-specific contexts is frozen by the
split manifest before collection. Every model appears in every split. A panel is
complete only if all 12 knob actions and exactly the five plan-matching safe strategy
actions appear with complete evaluator-owned outcomes in that context. S11/S12 risk
probes are excluded from panel completeness and policy support. The existing Q2/B35/
B36/Q3 data do not satisfy this new minimum automatically.

Historical artifacts are imported with their status and role from section 2. Q2/B35
stay `Q3_VALIDATION`, B36 stays `Q3_SEALED_HOLDOUT`, E11 stays `LEDGER_ONLY`, and
Q3's zero TRAIN count remains true for Q3. No historical summary or trajectory is
relabelled as `Q4_TRAIN`, `Q4_VALIDATION` or `Q4_SEALED_HOLDOUT`.

## 7. Measurement and hardware collection contract

No collection phase starts implicitly. The user must explicitly start that named phase
after the implementation and its tests are reviewed. The phase records the exact
preregistration hash before any model import.

The local collection domain is initially only:

```text
Apple M1 Max, 32 GiB unified memory, AC power, Low Power off,
nominal thermal state, exact local Gemma 3 1B/4B/12B snapshots,
MLX/mlx-lm/runtime/code revisions recorded by digest, greedy decoding.
```

Local availability of Gemma 1B, 4B and 12B for the planned panel is confirmed. The 27B
model is explicitly out of Q4's required panel (`no27`). No 27B download, installation
or run is implied. Existing X1 27B rows remain prior-only context.

Hard preflight refusal conditions are:

- unknown hardware/model/runtime/manifest/workload identity;
- not AC, Low Power on, non-nominal thermal state, or unknown power/thermal result;
- an active Claude/Claude Code model or other competing inference/model process;
- free memory below 40% of installed memory, start swap above 4 GiB, or unknown swap;
- load maximum above 4.0 or three-sample load spread above 1.0;
- dirty/ambiguous Git or preregistration hash mismatch, absent evaluator binding, or
  an output path that could overwrite an existing artifact.

Each collection phase is separately pre-registered, hashed and explicitly started by
the user for one declared `context × stage` or `context × knob-anchor × strategy`;
no phase is started implicitly. Each such phase has
an independent hard wall deadline **1800 s (30 minutes)**, child timeout 120 s and
bounded output 512 KiB per child. These are per-phase limits; Q4 makes no aggregate
30-minute claim across contexts, models, panels or trajectories. The parent owns one
monotone deadline; no retry or automatic respawn is allowed. Live gates require swap
delta `<=256 MiB` (decrease allowed), MLX peak and conservative child RSS `<60%` of
installed memory, complete process-group reap, and no residual competing model process.
A failure stops that phase, retains all partial markers and falls back to BASE for
reporting.

Explicit maximum child processes per phase are frozen as follows:

```text
knob panel:                  12 children (BASE + 11 declared candidates)
safe strategy panel:          5 children (one process per plan-matching strategy)
risk probes (separate phase): 2 children (S11/S12; outside policy budget)
trajectory knob phase:       11 children (one trajectory's KNOB_DELTA steps)
trajectory strategy phase:    5 children (one trajectory's STRATEGY_SELECT steps)
trajectory revalidate phase:  1 child (one trajectory's REVALIDATE step)
```

For the full Stage-2 interaction panel, all 12 knob actions × five plan-matching safe
strategies (60 cells/context) are collected as 12 separately preregistered
knob-anchor phases. Each phase uses five fresh strategy processes, one process per
strategy under that knob anchor: exactly five arm cells, each with two warmups and five
measured repeats. Its hard ceiling is five child processes, 120 seconds per child and
600 seconds for that phase. The shared BASE reference remains external to the method
budget and identical across methods. Any unmeasured pair forces BASE fallback and is
ineligible for composition. S11/S12 are the separate two-child risk phase. Each
trajectory is three separately preregistered/user-started
context×trajectory subphases: an 11-child knob phase (`11 × 120 s = 1320 s`), a
5-child strategy phase (`5 × 120 s = 600 s`) and a one-child revalidation phase
(`1 × 120 s = 120 s`), each below 1800 s. There is no single 39-child trajectory phase.
The `trajectory_id`, `context_id`, `study_digest` and predeclared collection-batch
`time_digest` remain stable across all three subphases.

Each action uses exactly two warmups and five measured repeats unless a later phase
pre-registration freezes a stricter number. Repeats are fresh, paired where applicable,
and preserve per-repeat logical/physical/visible tokens, counts, stop reasons, capacity,
timings, memory and gates. Seeds are fixed as `Q4-RL-20260901/<phase>/<context>/<rep>`;
no seed is chosen after inspecting a result. Strategy p95 is computable only with at
least **20 request-level samples per action**; otherwise it is `NOT_COMPUTABLE` and
cannot satisfy an RL success criterion.

All Q4 panel and trajectory rows are entirely new, even when an old action or context
looks identical. Existing Q2, B35 and B36 observations are not Q4 rows and cannot
consume Q4 budget. Their hashes may be cited as historical provenance only. A missing
new row is missing; no budget is increased after seeing a result.

## 8. Fixed objectives, rewards and metrics

All primary quantities are computed per context first and then aggregated with equal
context weight. Repeats are not treated as independent contexts.

Define one positive lower-is-better cost `c` per context/objective (wall time for
`KNOB_DELTA`, p95 full-response latency for `LATENCY`, and the reciprocal of physical
tokens/s for `THROUGHPUT`). For each step 0--10 the knob head uses incremental wall
reward. For each step 11--15 the strategy head uses the objective-specific immediate
reward. In both units, the incremental reward is frozen as:

```text
r_t = log(c_current / c_candidate)
```

Here `current` is the previous complete safe action in the same unit. For each of the
independent Stage-2 steps 11--15, `c_current` is fixed to the same-final-knob,
plan-matched strategy BASE reference; it is not updated by earlier strategy candidates
and is never BASE under a different knob. Higher reward is better. The terminal
revalidation reward is separately frozen:

```text
r_terminal = log(c_BASE / c_final)
```

The knob head receives no strategy reward and the strategy head receives no knob
Bellman target. The hybrid reports a two-head reward vector and never scalar-adds the
heads. The strategy baseline is always measured under the same final knob.

The strategy objective class in the context fixes the cost before measurement:

- `LATENCY`: primary cost `candidate_p95_full_response_ms`;
- `THROUGHPUT`: primary cost `1 / candidate_physical_tokens_per_second`, with p95
  full-response inflation `<=10%` as a hard guard.

TTFT, median latency, aggregate throughput, realized grouping width, prefill/decode
time, memory, swap, fallbacks and uncertainty are diagnostic or guard metrics. Group
wall time is never divided by width. Strategy comparisons report the Pareto front of
throughput and p95 latency; they do not convert E14b's `+18.02%` and `+20.05%` into one
new headline number. Per-action p95 is the nearest-rank estimator over at least 20
request-level samples; its interval is a grouped bootstrap over complete contexts, not
over individual requests.

At equal evaluation budgets, every method reports:

1. best safe outcome and context-normalized regret against the sealed panel oracle;
2. `time_to_best`: first decision index whose original cost satisfies
   `c <= 1.01 * c_oracle` (not a log-reward threshold);
3. `experiments_to_best`: number of evaluated actions to the same original-cost
   threshold `c <= 1.01 * c_oracle`;
4. two separate exact-denominator rates: `unsafe_censored_rate` is failed,
   timed-out, fallback, unknown or incomplete outcomes divided by all attempted action
   cells; `safe_regression_rate` is complete safe outcomes with `c > 1.02*c_BASE`
   divided by complete safe outcomes;
5. uncertainty calibration: 90% interval coverage, interval width, log score and
   support/overlap diagnostics;
6. failure recovery: BASE fallback, rollback result and whether the next legal decision
   is available within one transition;
7. replay determinism: byte-identical action sequence, scores and report under the
   same dataset, seed and implementation digest.

The `1%` time-to-best tolerance, `2%` regression threshold and `10%` throughput-context
p95 guard are fixed protocol values, not tuned after results. The direct grouped RL
advantage has a frozen equivalence margin of **1 percentage point**; values inside
`[-1pp,+1pp]` are `TIE_NO_RL`. An oracle is computed only
from complete safe raw cells in the sealed panel. If no oracle exists, the metric is
`NOT_COMPUTABLE`, not imputed.

### Off-policy evaluation

WIS uses the recorded behaviour propensities and clips each per-decision importance
ratio to **10** before trajectory weighting. DR uses exactly **five grouped folds**,
where the fold assignment is `complete_context_or_group_hash mod 5`; all trajectories
from one context co-fold and no context/group may contribute transitions to both fit and
evaluation folds. WIS and DR are reported per context and then aggregated equally. If
any evaluated action lacks the minimum support of three grouped contexts, if a
propensity is absent/invalid, if a fold is empty, or if overlap or positivity fails,
every OPE result is `OPE_UNSUPPORTED`. In particular, the deterministic coordinate
trajectory (propensity 1 only for the selected action) cannot support counterfactual
OPE. OPE is permitted only for controlled Q4 TRAIN/VALIDATION rows with valid
propensities; Q4_SEALED_HOLDOUT is direct-panel-only and is never used to fit or select
an OPE model. Before freezing the RL model, supported validation WIS/DR must have the
same sign as the direct validation estimate and an absolute difference `<=2pp`.

For validation diagnostics only, a supported DR estimate with opposite sign to the
direct validation estimate or absolute disagreement greater than `2pp` blocks model
freeze. The sealed-holdout claim uses direct grouped panels only; no DR-vs-holdout
criterion exists. Its direct grouped advantage must have a 95% interval lower bound
greater than `+2pp` against the validation-selected simpler comparator.

## 9. Equal-budget method comparison

The comparison set is exactly:

```text
BASELINE
CURRENT_COORDINATE
SEEDED_RANDOM
BO
SURROGATE
CONTEXTUAL_BANDIT
OFFLINE_RL
```

All methods receive the same context, action masks, raw evidence, split manifest and
per-context cap of **16 candidate decisions**: **11 knob-delta** decisions plus **5
plan-matching strategy** decisions. The shared BASE reference is measured once per
context, is external to this 16-action budget and is byte-identical for every method.
The two risk probes S11/S12 are a separate safety phase outside this method budget and
are never counted as policy actions. All methods receive the same trajectory horizon.
`SEEDED_RANDOM` uses seeds `Q4-RANDOM-0..4`; all other methods use the seed manifest
and deterministic tie-break above. `BASELINE` consumes the shared BASE reference and
may not use hidden tuned data.

The five strategy decisions are plan-matched: S01--S05 for a StrictOneShot context or
S06--S10 for a ReusableSession context. The opposite plan is not an exact/safe action
in that context and is excluded before method scoring.

`CURRENT_COORDINATE` is the existing Q2/E11 order and keep rule, with no changes to
`SEARCH`, `KEEP_IF_RATIO_BELOW` or confirmation semantics; its selected action has
behaviour propensity 1 and its counterfactual OPE is unsupported. Q4 TRAIN uses
controlled seeded uniform safe exploration with exact propensities. Q4 validation and
sealed-holdout collection behaviour/propensities are frozen and deterministic but are
never used as TRAIN. BO and SURROGATE are
deterministic finite-catalogue methods with a declared prior and no network/model
dependency. `CONTEXTUAL_BANDIT` is eligible only when every split has multiple
independent contexts with comparable panels. `OFFLINE_RL` is eligible only after the
sequential-horizon gate below; until then its result is `NOT_APPLICABLE`, not a poor
score.

The behaviour policies are frozen by split. `Q4_TRAIN` uses seeded uniform safe
exploration without replacement, with seed `Q4-TRAIN/<context>` and exact propensity
`1/remaining_legal_safe_actions` for the scheduled action. `Q4_VALIDATION` uses the
same without-replacement policy with seed `Q4-VAL/<context>` and exact propensity
`1/remaining_legal_safe_actions`; these rows are valid for grouped OPE and model
selection but never become TRAIN. `Q4_SEALED_HOLDOUT` is unread until final direct
evaluation and uses frozen lexicographic full-panel order, propensity `1` for the
scheduled action, direct panel scoring only and no OPE fit, selection or estimate.

The final sealed evaluation is one pass, no interim inspection, no seed search and no
budget extension. A method that fails safety or replay gates is marked failed and does
not receive a replacement evaluation.

## 10. RL eligibility and success criteria

Offline RL changes from `NOT_APPLICABLE` to `STRUCTURALLY_ELIGIBLE` only if all of these
are true:

1. all 24 new contexts and 72 complete horizon-17 trajectories exist (1224 complete
   transitions); historical Q3 rows do not satisfy this count;
2. every split has complete grouped action panels for both spaces and every model 1B/4B/
   12B is represented;
3. every transition has evaluator-owned outcome, next-state identity, action identity,
   behaviour propensity, behaviour-policy digest, seed, uncertainty and raw evidence;
4. no context/group crosses splits, and deduplication/content hashes are complete;
5. BASE and each simpler comparator have a complete safe oracle-comparable panel;
6. failure/rollback rows are retained and the risk critic has non-empty observed failure
   support; no missing outcome has been imputed;
7. overlap/support diagnostics pass: at least `3` grouped contexts support each evaluated
   action for OPE, with the deterministic-coordinate exception explicitly marked
   `OPE_UNSUPPORTED`.

RL is a Q4 success only if, against the best eligible simpler method selected on
validation and frozen before holdout:

- the sealed-holdout primary reward has a 95% grouped bootstrap lower bound strictly
  greater than **+2 percentage points**;
- median `time_to_best` and median `experiments_to_best` are both computable and lower;
  `NOT_COMPUTABLE` is a failure to establish the RL advantage, not a tie exemption;
- regression rate is no worse than the simpler method by more than **1 percentage point**;
- uncertainty coverage is within `[0.80, 0.98]` and support diagnostics pass;
- before model freeze, supported validation WIS/DR has the same sign as the direct
  validation estimate and differs by no more than `2pp`; sealed-holdout OPE is never
  fit, selected or used as a contradiction check and remains direct-panel-only;
- all exact correctness, resource, rollback, process and replay gates pass; and
- the result holds in at least two of the three model strata without a third-model
  safety or correctness failure.

The 2-point, 1-point and `[0.80,0.98]` criteria are frozen before data collection.
They are not claims of universal statistical power. If a simpler method is equal or
better within the declared equivalence margin, RL is killed as a product direction and
the simpler method remains the research baseline. If a simpler method wins, there is
no RL activation, routing or profile promotion. This is a **local-pilot** result only;
it does not establish a universal or cross-Mac policy.

## 11. Two-stage HybridOptimizer contract

The implementation plan must keep these stages separately inspectable:

1. **KnobOptimizer:** proposes complete `q4_knob_action.v1` candidates with coordinate,
   random, BO, surrogate and RL policies sharing the same panel/evaluator. It may not
   alter execution plans or strategy modes.
2. **StrategyOptimizer:** receives the selected/observed knob configuration and proposes
   `q4_execution_strategy_action.v1` candidates. It may not rewrite knob fields or
   silently substitute `InteractiveMode`/`ThroughputMode`/plans.

The hybrid report contains both IDs, both value estimates, both uncertainty intervals,
the selected objective class and the safety decision. In Q4 the result is always
`SHADOW_RECOMMENDATION`; the runtime consumes none of it. A recommendation is only
`eligible_for_future_revalidation` when it is backed by raw, evaluator-owned,
qualified evidence in the exact domain. X1, E14b, B35, B27 summaries and any Q3 failure
can inform priors or risk masks but cannot create that status.

## 12. Safety, correctness and promotion boundary

The evaluator is external to the optimizer and must verify, per repeat:

- exact logical/physical/visible token IDs, counts, stop reasons, capacity and
  deterministic state/KV identity;
- exact model revision/manifest, runtime/code, workload, hardware and power identity;
- no timeout, crash, fallback, swap/resource breach, process leak or failed rollback;
- complete raw evidence, reviewer/evaluator role separation and preregistration hash.

Any unknown, malformed, racy, missing or self-asserted evidence fails closed. A worker
or child visibility problem is a collection failure, not permission to wait longer or
to infer success. Q3f is terminal and must not be retried or pooled.

No Q4 record can activate a profile. Activation, routing, persistent storage and any
change to BASE/current coordinate require a later architecture decision, a separately
sealed experiment and explicit user approval. A Q4 result may at most recommend a
future revalidation cell.

## 13. Foreign-Mac evidence and calibration

External hardware is not available to this protocol by assertion. A foreign result is
accepted for local offline calibration only as an
`ironmule.q4_foreign_bundle.v1` object containing exactly:

```text
schema, bundle_id, exporter_id, host_class, hardware_digest, model_digest,
model_manifest_digest, runtime_digest, code_digest, workload_digest,
preregistration_sha256, raw_artifacts, reviewer_record_sha256,
signature_algorithm, signer_key_fingerprint, signature, exported_at_utc
public_key_id
```

`signature_algorithm` is exactly `Ed25519`; no alternate or unsigned signature is
accepted. The `signer_key_fingerprint` and public-key ID must resolve through an
explicit, local, user-approved trust store before verification. The signature covers
canonical bundle bytes and all referenced artifact hashes. The bundle must include raw
samples, full repeat-level correctness/resource gates and an independent
evaluator/reviewer record. Summary-only, screenshot-only, manually typed or unsigned
numbers are `UNTRUSTED_FOREIGN_EVIDENCE` and cannot calibrate a policy.

Calibration is a held-out prior update, not pooling: foreign bundles are assigned a
new hardware group, never copied into the local TRAIN split, and first produce
`REVALIDATION_REQUIRED` until a local exact-identity probe matches. A missing or
unverifiable foreign bundle is recorded as missing. **Current foreign evidence is
`MISSING`.** No external Mac measurement may be fabricated, simulated as measured, or
substituted by an estimate.

## 14. Implementation order before any hardware

This is the Q4 implementation plan, not permission to execute it:

1. **Freeze and audit:** verify this file's SHA-256, preserve Q3/B27 raw files, build a
   read-only content-hash inventory and a migration report with all statuses/limits.
2. **Offline contracts:** implement strict serializers/validators for the Q4
   schemas, action catalogues, split/group checks, canonical IDs and failure retention;
   keep the module stdlib/offline and outside the runtime import graph.
3. **Replay/evaluator:** add deterministic replay for all seven methods, equal-budget
   accounting, oracle/regret/time-to-best metrics, uncertainty/failure diagnostics and
   byte-identical reruns. No model/runtime execution is used here.
4. **Hierarchical RL:** implement EB-HCORL with action masks, behaviour prior,
   conservative ensemble/risk critic and OPE support checks. Validate only on synthetic
   contract fixtures and migrated historical records; never treat fixtures as evidence.
5. **Shadow hybrid:** implement the two-stage report and prove it cannot import runtime,
   select a plan/mode, write profiles or execute a candidate. This remains shadow-only.
6. **Separate collection preregistrations:** write/hash one phase for every
   `context × stage` (knob panel, safe strategy panel, separate S11/S12 risk probes or
   trajectory). Obtain explicit user start for each; require AC and the
   no-Claude/no-competing-model gates in section 7. Each phase is capped at 30 minutes
   independently; no aggregate 30-minute claim is made.
7. **Sealed evaluation:** import only the frozen collection artefacts, lock validation
   choices, run one sealed holdout comparison and append results without editing this
   protocol. Foreign bundles, if any, are separately signed/calibrated.
8. **Decision:** classify `RL_WINS`, `SIMPLER_WINS`, `TIE_NO_RL`, `DATA_INSUFFICIENT`,
   `SAFETY_FAILURE` or `OPE_UNSUPPORTED`. Only `RL_WINS` can open a later architecture
   decision; it still does not activate anything automatically.

## 15. Kill criteria and non-negotiable limits

Q4 stops as `DATA_INSUFFICIENT` when any action panel, independent grouped context,
full split or measured horizon is absent. This is a useful result and not a reason to
pool incompatible evidence or lower gates.

RL is permanently rejected for this Q4 decision if a simpler method is equal or better
at equal budget, if holdout advantage is not reproduced, if either time-to-best or
experiments-to-best is not lower, if regression risk is worse, or if uncertainty/OPE
support is not calibrated. The two-stage hybrid then remains a shadow diagnostic.

Any output/token/stop/count/state divergence, resource leak, swap/timeout/crash,
unknown cleanup, split leakage, forged/summary-only evidence, nondeterministic replay,
budget mismatch, unauthorized runtime import, profile write, route change or automatic
activation is a terminal Q4 failure. BASE/current coordinate remains unchanged.

No Q4 result may:

- combine E14b's two mechanisms into one unqualified percentage;
- promote X1, B35, B27 summaries or exploratory true batching;
- retry Q3c/Q3d/Q3e/Q3f or pool their safety records;
- claim a 27B result when Q4 is explicitly `no27`;
- claim a foreign-Mac result without the signed raw bundle; or
- infer an RL win from modelled, synthetic or imputed observations.

The final report must state raw artifact hashes, dataset ID, split manifest hash,
implementation hash, seeds, method budgets, all missing/failed rows, direct versus
derived metrics, and the exact reason for any non-eligibility. No UI is part of this
contract; a future UI may display already-written local evidence only after a separate
decision.
