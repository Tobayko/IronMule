# Q4 implementation plan — RL-first, shadow-only

This plan is subordinate to [`research/raw/Q4_preregistration.md`](../research/raw/Q4_preregistration.md).
It describes implementation order and gates; it does not authorize a model run,
hardware collection, package/model download, runtime routing or profile activation.

## Completion contract before hardware

Q4 implementation is complete only when the offline evaluator can replay the frozen
contract and fail closed on malformed, incomplete, unsafe, leaked or non-deterministic
records. The implementation must stay outside the runtime import graph until a later
architecture decision.

The collection target is 24 entirely new contexts (Q4_TRAIN 12, Q4_VALIDATION 6,
Q4_SEALED_HOLDOUT 6; 4/2/2 per model for Gemma 1B/4B/12B), three H17 trajectories per
context, 72 trajectories and 1224 transitions. The resulting claim is a local-pilot
claim only. Foreign evidence is `MISSING` until an Ed25519 signature, public-key ID and
fingerprint resolve through a local user-approved trust store.

## Current status after implementation

`DONE`: offline Q4 contracts, 11 candidate specs, dynamic knob-delta/no-repeat state,
12 interaction anchors, context-bound Outcomes, panel/trajectory/risk records, reward
derivation exclusions, strict stage-vector OPE, dataset gate, external-Ed25519 shadow
envelope and foreign replay registry are implemented and covered by 55/55 tests.
`VERIFIED_POST_AMENDMENT`: the stable historical import saw 285 inputs, skipped two
derived implementation-report files, retained 195 unique contents and one eligible
historical artifact. Its dataset ID is
`42de861095f7050a7c572ee2ab97ed253e649e790c115d65b2cc1e4e2f6c766b` with semantic
payload SHA `05a592140db776c48423a3caec8d646e28216c232a0c3d60483af1d3901b35ed`;
the temporary import file SHA is `a188d4e2fd299ea615706e2ed0292bdf78dfaa1def388f0e869b82cc38558f35`
and is not repository evidence. `PENDING_USER_START`: all Q4 panel cells and trajectories
are new and unmeasured; current counts are 0 Q4 transitions and 0 TRAIN rows.
`OFFLINE_RL` therefore remains `DATA_INSUFFICIENT`/`NOT_APPLICABLE`; no Q4 speed
improvement claim exists. The durable evidence is recorded in
[`research/raw/Q4_implementation_report_20260901.md`](../research/raw/Q4_implementation_report_20260901.md).

Each trajectory is split into three separately preregistered/user-started phases:
11 knob children (`11×120s=1320s`), 5 strategy children (`5×120s=600s`) and 1
revalidation child (`120s`), each under 1800 seconds. Context, trajectory, study and
predeclared batch-time digests remain stable across those phases.

## Work packages

1. **[DONE — post-amendment verification] Corpus migration.** The read-only importer
   deduplicates by content hash,
   preserves raw/summary/partial/exploratory/failure quality and status, records source
   preregistration/code/model/environment/workload identities, and emits a new Q4
   dataset ID. Keep Q3's `DATA_INSUFFICIENT`/`NOT_APPLICABLE` result unchanged.

2. **[DONE] Two closed action spaces.** Strict canonical records exist for the existing
   ten-field knob action and the ten-entry safe execution/scheduling strategy catalogue.
   Keep S11/S12 as two separate risk probes outside policy completeness and budget.
   Validate legal combinations, width `1..4`, objective class, action-pool completeness
   and hybrid `(knob_action_id, strategy_action_id)` identity. S02/S07 must map to the
   existing `AsyncGroupedB1Executor` at width 1; True Batch is not an existing
   `ExecutionStrategy`. Do not include 27B in the required panel.

3. **[DONE] State/transition/evaluator layer.** Q4 context, state, trajectory,
   transition, outcome, partial-abort and signed-foreign-bundle schemas. Enforce the
   seven-part group key `study/model/manifest/workload/hardware/runtime/time`, with a
   predeclared batch `time_digest` and stable `study_digest`. Enforce new-only
   `Q4_TRAIN/Q4_VALIDATION/Q4_SEALED_HOLDOUT` splits, all split isolation, horizon
   `H=17` (`11 KNOB_DELTA + 5 STRATEGY_SELECT + 1 REVALIDATE`), terminal completion only
   at step 16, evaluator-owned gates, raw references, rollback and no reward imputation.
   Preserve failed and censored transitions, including terminal partial-abort state at
   the current step.
   The exact tabular feature order is intercept, model-size one-hot, memory bucket,
   GPU-core bucket, prompt/output/concurrency buckets, objective, plan,
   workload-stratum, arrival-pattern, current-action one-hot and scaled remaining
   budget; unknown categories are OOD and masked, with no unlisted interactions.

4. **[DONE] Deterministic replay baselines.** Equal-budget replay exists for BASELINE,
   current coordinate, seeded random, deterministic BO, surrogate and contextual
   bandit. Fix seeds, fold order, action ordering and lexicographic tie-breaks. Record
   `0 < behaviour_propensity <= 1` and `behaviour_policy_digest` on every transition.
   Coordinate's selected-action propensity is 1 and makes counterfactual OPE
   unsupported; Q4 TRAIN uses controlled seeded uniform safe exploration with exact
   propensities. Validation uses seed `Q4-VAL/<context>` without replacement with
   `1/remaining` propensities; holdout uses frozen lexicographic order with propensity
   1 and direct-only scoring. Produce best outcome, oracle regret, time-to-best, experiments-to-best,
   regression rate, calibration, support, recovery and byte-identical replay reports.
   No runtime import.

5. **[DONE — data-gated] EB-HCORL.** Two separate RL heads exist: knob FQI only on steps 0--10 and
   strategy contextual-immediate ridge only on steps 11--15; neither head receives
   Bellman/reward backup from the other unit. Knob FQI uses `gamma=0.9`, ridge
   `alpha=1`, 20 iterations, tolerance `1e-9`, BC penalty `lambda=0.1`, a five-member
   deterministic context/group-hash ensemble, pessimistic performance LCB, separate
   failure-risk UCB, action masks and OPE diagnostics. The knob target/loss are
   `y=r+0.9*(not_knob_terminal)*max_supported(Q_prev)` and
   `sum((Q-y)^2)+1*||w||^2`, for 20 iterations/tolerance `1e-9`; the strategy head is
   contextual immediate ridge only on steps 11--15, with no Bellman or cross-unit
   backup; each strategy candidate uses the same final-knob strategy BASE reference.
   Selection is `Q_LCB - 0.1*(-log(max(propensity,1e-6)))`.
   Freeze `Q_LCB = mean - 1.0*sd - 1/sqrt(grouped_support+1)` and
   `failure_UCB = min(1,(failures+1)/(trials+2)+sqrt(log(20)/(2*max(trials,1))))`;
   allow an action only when its risk UCB is no greater than the incumbent's.
   Freeze WIS ratio clip `10` and grouped five-fold DR by complete context/group hash;
   all trajectories from a context co-fold. Require minimum support `3` grouped
   contexts and mask OOD/unsupported actions. `OPE_UNSUPPORTED` is a valid outcome;
   unsupported OPE must not be converted into a direct performance claim.

6. **[DONE — shadow-only] Shadow HybridOptimizer.** The knob stage passes only its canonical action
   identity and evidence to the strategy stage, then emit a signed
   `SHADOW_RECOMMENDATION`. Stage 2 state must contain `knob_action_id`. Measure the
   full exact panel of all 12 knob actions × five plan-matching safe strategies (60
   cells/context) in 12 separately preregistered anchor×strategy phases; any unmeasured
   pair forces Stage-2 BASE fallback and is ineligible for composition. Prove by tests
   that the hybrid cannot execute an action, import MLX/runtime, change a plan/mode,
   write a profile or activate a route.

7. **[DONE — offline] Synthetic and historical verification.** Synthetic fixtures cover schema,
   leakage, malformed-input, determinism, action-mask, failure-recovery and metric
   tests. Replay historical Q2/B35/B36/B27/E14b/E16/X1 records with their original
   restrictions and names (`Q3_VALIDATION`, `Q3_SEALED_HOLDOUT`, `LEDGER_ONLY` for
   E11); do not upgrade prior-only, summary-only or Q3 rows to Q4 split labels.

8. **[PENDING USER START] Collection gates.** After code review, freeze/hash one separate preregistration
   for each collection phase: complete panels, independent contexts and sequential
   trajectories. Request explicit user start for each phase. Require AC, Low Power off,
   nominal thermal state, no Claude/Claude Code model or competing inference process,
   exact local identity and the 30-minute per-context×stage wall bound. The phase
   child ceilings are 12 knob-panel, 5 plan-matching safe-strategy (one strategy per
   child under the knob anchor), 2 separate risk-probe, and per-trajectory subphases of 11
   knob, 5 strategy and 1 revalidation child; the three trajectories per context are
   separately preregistered and there is no aggregate 30-minute claim.

9. **[BLOCKED BY DATA] Sealed decision.** Lock validation choices, import only frozen raw evidence, run
   the one-pass sealed comparison and append an immutable result. Only `RL_WINS` can
   open a later architecture decision; it cannot activate anything automatically.

## Verification gates

- strict round-trip and canonical SHA tests for every Q4 schema;
- cross-split group/context leakage rejection;
- incomplete panel and missing-horizon rejection;
- H17 stage enum and partial-abort terminal-state validation;
- exact behaviour propensities, coordinate `OPE_UNSUPPORTED`, WIS clip-10 and grouped
  five-fold DR support/overlap rejection;
- failure, timeout, fallback and rollback retention without reward fabrication;
- action-mask, risk-probe exclusion and true-batch promotion rejection;
- equal-budget accounting and seeded replay byte equality;
- grouped bootstrap and oracle `NOT_COMPUTABLE` behaviour when a safe panel is absent;
- nearest-rank p95 only with at least 20 request-level samples per action;
- direct grouped RL lower-bound `>+2pp`, equivalence margin `1pp`, DR contradiction
  (opposite sign or gap `>2pp`), original-cost time-to-best `<=1.01*oracle`, and
  separate unsafe/censored versus safe-regression denominators;
- uncertainty/OPE support diagnostics and calibration bounds;
- no runtime/MLX/model/network imports and no profile/persistence writes;
- `git diff --check` and the serial non-integration suite before any collection.

## Stop and rollback

Any failed gate leaves BASE/current coordinate unchanged and records a durable failure
reason. Missing raw evidence, identity drift, process ambiguity, unsafe resource state,
split leakage, nondeterministic replay or unauthorized side effect is terminal for the
affected phase. Do not retry Q3c/Q3d/Q3e/Q3f, pool their records, or fabricate foreign
Mac measurements. Later collection work requires a new explicit phase decision.
