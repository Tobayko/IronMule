# Q4 implementation report — 2026-09-01

## Result first

The Q4 software path is implemented and tested as an offline, shadow-only research
boundary. It has produced **zero new Q4 speed gain** and no RL result. The honest
current status is `DATA_INSUFFICIENT` for adaptive method comparison and
`NOT_APPLICABLE` for `OFFLINE_RL` until the preregistered new panels and sequential
trajectories exist.

No model process, hardware/MLX run, 27B run, download, installation, runtime
activation, profile write, routing change or UI action occurred in producing this
report.

## Implemented contract surface

The offline implementation is split into immutable contracts, corpus import, replay
methods, foreign verification and a shadow optimizer:

- 11 `CandidateSpec` policy slots plus dynamic one-field `KnobDelta` transitions and
  no-repeat state tracking;
- 12 absolute interaction anchors, including the exact Q2-current anchor;
- dataset records for contexts, states, trajectories, transitions, outcomes,
  `risk_observations` and 60-cell `panel_cells`;
- transitions bound to both `candidate_id` and `reference_outcome_id`;
- evaluator-owned, context-bound Outcomes with raw timing, request-level and
  throughput samples;
- canonical `RewardRecord` plus explicit derivation records that exclude unsafe,
  missing, censored, summary-only and mismatched evidence;
- full exact Stage-2 cross-product: 12 knob anchors × 5 plan-matching safe strategies
  = 60 cells per context;
- strict stage-vector OPE with WIS clip 10 and grouped five-fold DR; unsupported
  overlap/propensity returns `OPE_UNSUPPORTED`;
- dataset-gated `OFFLINE_RL`, with no eligibility from historical import alone;
- signed shadow recommendation envelope using an external Ed25519 provider; and
- foreign replay registry with user-approved key/trust binding and duplicate/replay
  rejection.

The runtime remains outside this boundary. The HybridOptimizer can only return a
`SHADOW_RECOMMENDATION`; it cannot execute, route, activate or persist a profile.

## Offline import verification

The stable post-amendment verification imported **285 inputs**. The two derived
implementation-report files were explicitly skipped from evidence identity, leaving
**195 unique contents**. Exactly **1 historical artifact** was eligible under its
original raw/identity/resource gates. The import contained **0 Q4 transitions** and
**0 Q4 TRAIN rows**. Foreign evidence is **`MISSING`**.

The final historical-import dataset ID is
`42de861095f7050a7c572ee2ab97ed253e649e790c115d65b2cc1e4e2f6c766b` and its semantic
payload SHA-256 is
`05a592140db776c48423a3caec8d646e28216c232a0c3d60483af1d3901b35ed`. The temporary
import file SHA-256 is
`a188d4e2fd299ea615706e2ed0292bdf78dfaa1def388f0e869b82cc38558f35`; it is temporary
verification output only and is not repository raw evidence. The earlier dataset ID
`dfb48cd030fd61a64a7a0f006d099682b349410b8e5761c1d1a593c80490a6cc` and hash prefix
`85c90b…` are retained only as superseded pre-amendment verification.

## Existing speed evidence (not Q4 results)

These values are historical context and are not new Q4 performance claims:

| Evidence | Historical result | Status and limitation |
| --- | --- | --- |
| Q2 self-tuning | `+14.57%`, confirmed ratio `0.8568`, exact tokens | Historical Q3 validation; not Q4 TRAIN and not a new Q4 gain |
| E14b grouped B/A | `+18.02%` | Prior-only mechanism evidence; teacher-forced pilot |
| E14b true batch C/B | `+20.05%`; C/A `+34.47%` | Prior-only; width 8 had reproducible divergence/unsafe behaviour; no Q4 promotion |
| X1 Gemma 4B / 12B | `+19.24%` / `+15.42%` | Exploratory, unpreregistered, prior-only; no Q4 27B panel |
| B36 Gemma 12B | `+7.29%`, ratio `0.927147`, 95% CI `[0.919736, 0.930375]` | Qualified raw evidence, activation remains disabled and it is not a Q4 row |

Q4 itself has **no speed improvement claim yet**. RL has not been trained or judged
against a holdout.

## Verification and remaining gate

- Offline implementation verification: **55/55 tests passed**.
- `xcodebuild -checkFirstLaunchStatus`: exit 0.
- ProjectAtlas runtime: v3, index available for the b7 worktree; final refresh
  completed after the documentation changes.
- The Q4 structural gate remains false until there are 24 entirely new contexts,
  72 complete H17 trajectories and 1224 complete transitions, with the required
  12-anchor × 5-strategy panels and split isolation.
- Therefore the report status is `DATA_INSUFFICIENT` / `NOT_APPLICABLE`, not a
  negative RL performance result.

The local SQuAD artifact and all existing raw JSON remain untouched. Foreign-Mac
evidence cannot be inferred or fabricated; only a verified, user-approved Ed25519
bundle may enter the calibration/replay boundary later.
