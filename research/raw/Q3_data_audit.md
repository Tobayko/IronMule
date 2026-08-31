# Q3 offline data audit

Frozen audit date: 2026-08-31. This is an inventory decision, not a performance
result. No model, MLX runtime, UI, persistence, profile activation, or executable
action was used.

## Decision

The real inventory is `DATA_INSUFFICIENT` for Bayesian optimisation, surrogate
learning, contextual-bandit evaluation, current coordinate descent, and seeded
random comparison: no complete safe counterfactual action panel is present. The
baseline is structurally replayable only when a complete safe baseline observation
exists. Offline RL is `NOT_APPLICABLE`: Q3 v1 contains static knob actions and no
measured sequential horizon. These are structural minimums, not statistical
qualification or performance claims.

## Counts and identity

| Source | State/action/outcome units | Raw timing repeats | Action coverage | Identity / gate status |
| --- | ---: | ---: | --- | --- |
| Q2 profile archive | 1 final profile + 11 adaptive trial rows | 0 trial raw repeats; 6 confirmation pair ratios | final action has 10/10 knobs | model/runtime identity recorded; adaptive trajectory only |
| B35 archive | 6 clean process runs, 12 arm rows | 60 measured + 24 warmups | 2 actions (baseline/core), 3 models × AB/BA | token/determinism gates true; hardware identity incomplete |
| B36 archive | 16 paired observations, 32 children | 160 measured + 64 warmups | 2 actions, one model/workload context | 32/32 children and 16/16 pairs passed all gates |
| B27 benchmark archive | 8 complete raw benchmark files | 96 arm repeats / 576 request outputs | `interactive` vs `throughput`, not ten tuning knobs | model/runtime/workload present; chip/HW fingerprint absent |

Q2 has ten final knob fields and complete model conditions, but each trial stores only
one aggregate timing tuple (`total_ns`, `prefill_ns`, `decode_ns`), not raw repeats or
full action snapshots. Its six confirmation ratios are derived summaries.

B35's seven archived JSON files contain six clean files and one invalidated file:
`B35_gemma1b_AB_20260828.json` was contaminated by overlapping broad filesystem
search and has `valid_for_metrics=false`. Every clean arm has ten knob fields, five
timing repeats, two warmups, one token list, and no per-repeat stop-reason/visible-token
records. The raw worker lacks complete hardware identity and encodes `no_crash` as a
worker boolean rather than an independently machine-checked raw gate.

B36's full raw file has SHA-256
`8566c6bd1ca2c82bfa2635a5bf38765d55ee5860de0772dd6edcbeb1d6441706` (archive object
`8566c6bd1ca2c82bfa2635a5bf38765d55ee5860de0772dd6edcbeb1d6441706-B36_gemma12b_results_20260828.json`).
All measured repeats contain logical, physical, and visible token arrays, stop reason,
decode steps, and total/prefill/decode timings. The exact context is one Gemma 3 12B
snapshot, one 322-token prompt, Apple M1 Max 32 GB, MLX 0.32.0, mlx-lm 0.31.3, AC
power. The candidate action is repeated, not a broad action panel.

Every measured outcome must carry an evaluator-owned `hard_gates_passed` value. Q3
does not infer or weaken this value; failed, timed-out, crashed, fallback, rejected,
and invalidated observations remain in the dataset and count toward coverage failures.

The B27 corpus inventory reports 134 artifact rows, 92 unique contents, 42 duplicate
groups, 72 valid JSON artifacts, and 51 local-only/untracked artifacts. Its dataset
SHA-256 is `ee414c9ee51c6e583ada094444ce66d5e22dca6c15c197dda1d7cd004e30bf32`.
The B27 public summaries explicitly keep activation disabled; B27d is
`INCONCLUSIVE_POTENTIAL_REGRESSION` and B27e is `ORDER_OR_TEMPORAL_DRIFT`.

## Q3 replay artifact

The offline adapter produced the ignored local artifact
`research/raw/Q3_replay_dataset.json` with file SHA-256
`f67d975788763e4238019a3be7afa5394efbe2f2faea3a96a927e7cf522f2e33`, mode `0600`,
and dataset ID
`d4ae0c148e826de85c7aa5338f892b5571481a105f558d463e9d041f63dc82b7`. It contains
14 observations (12 Q2 `VALIDATION`, 2 B36 `SEALED_HOLDOUT`), an explicit 12-action
pool, 160 B36 raw samples, 2 complete safe observations, and zero `TRAIN` rows.
The resulting structural eligibility is frozen as: `BASELINE` structurally eligible;
`CURRENT_COORDINATE`, `SEEDED_RANDOM`, `BO`, `SURROGATE`, and
`CONTEXTUAL_BANDIT` `DATA_INSUFFICIENT`; `OFFLINE_RL` `NOT_APPLICABLE`.

Key source hashes:

* Q2 preregistration: `ab8a0740ae42600f80fa1a4f2f01aa751fd8b62ff7d9430fa8b5f36f9d64aef0`.
* Q2 run log: `a3c0839fa15f605718275f99e11ba2f4e8dab2a0a23568c5d276fd7f2f26eac9`.
* Archived Q2 profile: `0a1104b248b4aaf532ee8ef7d9c9c0c06196dde0c5111450ee9386358d15509b`.
* B36 preregistration: `7bf3997b19dc55d3b75be977c0da8d42d6ab554232ce2bf40617429c478897a4`.
* B36a clarification: `ee5b3e9b250d75eb69ed6e38f9661f656da743098bef318966dc055099c9e492`.

## Leakage-safe replay split

Deduplicate by content hash before splitting. Keep every repeat, both arms, AB/BA
order, bootstrap result, summary, profile/trial row, preregistration, and review in
the same group. The group key is:

`experiment × model_manifest_digest × workload/prompt_digest × hardware_digest × runtime/code_digest`.

Use pre-Q2 schema-compatible exploration for `TRAIN`; put complete B35 model groups
(AB and BA together) in `VALIDATION`; keep all B36 pairs as one sealed
`SEALED_HOLDOUT`. Keep Q2 as one sealed adaptive trajectory, never as training rows.
The replay dataset carries an immutable unique `action_pool`; a panel is complete only
when every pooled action appears exactly once, safely and fully measured, in one
context. The structural minimum is not statistical qualification.

## Cheapest decisive missing evidence

Three needs are kept separate:

1. `complete_raw_counterfactual_action_panel_in_isolated_fresh_processes`: the cheapest
   decisive need for current coordinate, seeded random, and local BO/surrogate replay.
2. `independent_grouped_contexts_with_comparable_action_panels`: required for
   generalisation and contextual-bandit evaluation.
3. `measured_sequential_horizon`: required before offline RL can be applicable; Q3 v1
   is static and therefore currently `NOT_APPLICABLE`.

Each is a data-collection request only; none is an executable optimizer action.
