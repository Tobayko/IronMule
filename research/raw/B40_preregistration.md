# B40 — Core ThroughputMode width sweep

Experiment ID: B40
Registered: 2026-08-28
Status: exploratory width-selection only; no activation

B40 tests only the core configuration (`compiled_fixed_cache=True` and
`head_skip_prefill=True`) while varying `ThroughputMode.max_width`. The three
explicit arms are W2, W3 and W4 with `max_width` 2, 3 and 4 respectively. The
exact local Gemma 3 12B snapshot and X1-strict workload remain frozen: six
fresh q0-q5 requests, `StrictOneShotPlan` per request, max_tokens 48 and greedy
decoding. No B39 data or timing, conditioner, purge, cache mutation or retry
is allowed.

Six fresh serial blocks use orders `W2 W3 W4`, `W3 W4 W2`, `W4 W2 W3`,
`W4 W2 W3`, `W3 W4 W2`, `W2 W3 W4`. Each arm process loads the model once,
performs two warmups and five measured repeats. The parent writes atomic
partial evidence after each child. Parent and child use schemas
`ironmule.b40.v1` and `ironmule.b40.child.v1` and bind this file's SHA-256.

Before every child, AC power, low-power-off, nominal thermal state, no
competing model process and process-start Swap at most `268435456 B` are
required. Every child must provide complete token/stop/correctness/identity,
checkpoint, crash, residual and post-state evidence. Swap delta must remain at
most `268435456 B`; all RSS and MLX checkpoint values must remain at or below
12 GiB. Per-block MLX peak ratios `W2/W4` and `W3/W4` must be at most `1.10`.
RSS ratios are recorded but are not per-block hard gates.

After all six blocks, RSS is evaluated only with the position-balanced metric:
each arm/position cell must have exactly two non-start RSS peaks; its median is
`m[a,p]`, and the arm value is the geometric mean over the three positions.
Global `W2/W4` and `W3/W4`, every position residual and every mirrored epoch
ratio for pairs `(0,5)`, `(1,4)` and `(2,3)` must lie in
`[1/1.10, 1.10]`. Missing, nonfinite, wrong-cardinality or failed RSS evidence
is `INCONCLUSIVE`, never `REJECTED`.

Performance uses five-repeat block medians. Candidate wall ratios are W2/W4
and W3/W4 (lower is better); physical and visible rate ratios use candidate/W4
(higher is better). Each comparison uses a deterministic 10,000-resample
bootstrap with descriptive 95% and Bonferroni-conservative 97.5% fields. A
candidate qualifies only when its wall median is below 0.995, its upper 97.5%
wall bound is below 1, and both rate lower 97.5% bounds exceed 1.

If both candidates qualify, both are reported and the lower wall-median
candidate is selected. `RETAIN_WIDTH4` is allowed only when both candidates
have robust practical misses (wall lower 97.5% bound above 1 with median at
least 1). Ambiguous results are `INCONCLUSIVE`. Complete resource-clean
performance misses are otherwise `REJECTED`; resource, RSS, identity,
correctness, crash, drift or final-H2 failures are `INCONCLUSIVE`.

`valid_for_performance` is true only for complete `QUALIFIED` or
`RETAIN_WIDTH4` results. `activation_allowed` is always false. No automatic
routing or B39 follow-on is permitted. The historical X1 `+15.42%` rate is a
descriptive flag only: rate ratio `1.1542`, equivalent wall ratio
`0.866400970369`, and is never a B40 gate.
