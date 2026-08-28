# B39d — Performance main after RSS-order diagnostic

Experiment ID: B39d
Parent: B39
Registered: 2026-08-28
Status: exploratory performance qualification only; no activation

B39d is a new run and does not reuse or pool B39, B39b or B39c timings or
evidence. It uses the exact local Gemma 3 12B snapshot and X1-strict workload:
four explicit arms, six fresh q0-q5 requests, `StrictOneShotPlan` per request,
max_tokens 48 and greedy decoding. No cache conditioner, purge, wired/cache
mutation or other host-state mutation is allowed.

The eight frozen balanced block orders are ABDC, BCAD, CDBA, DACB, DACB,
CDBA, BCAD and ABDC. Each block has four fresh serial operating-system child
processes, one model load per child, two warmups and five measured repeats.
The parent performs no retries and writes an atomic partial sidecar after every
child. The child schema is `ironmule.b39d.child.v1` and the result schema is
`ironmule.b39d.v1`.

Before every child, the parent requires stable AC/non-low-power/nominal-thermal
preflight, no competing model process and process-start Swap at most
`268435456 B` (`256 MiB`). Every child must retain complete checkpoint,
correctness, identity, model/code/preregistration, crash, residual and post-state
evidence. Every RSS and MLX checkpoint must be finite and at most 12 GiB, and
process-start-to-end Swap delta must be at most `268435456 B`. Relative MLX
peak ratios `C/A` and `D/B` must be at most `1.10` in every block. Relative RSS
ratios are recorded but are not per-block hard gates.

RSS is evaluated only after all eight blocks complete. For each child, the RSS
observation is the maximum of its non-process-start checkpoints. For every arm
and process position, exactly two observations are required; their median is
`m[a,p]`. The arm value is the geometric mean over the four positions,
`G[a] = exp(mean_p(log(m[a,p])))`. Global RSS `C/A = G[C]/G[A]` and
`D/B = G[D]/G[B]` must be within `[1/1.10, 1.10]`. Every position residual
`m[a,p]/G[a]` and every matched epoch ratio for pairs `(0,7)`, `(1,6)`,
`(2,5)`, `(3,4)` must be within that reciprocal band. Missing, nonfinite or
wrong-cardinality RSS evidence is `INCONCLUSIVE`; an RSS failure never becomes
`REJECTED` and supports no arm attribution.

Performance uses the existing B39 block-median statistic unchanged. The
co-primary `D/A` and `D/B` wall-time ratios must each have median below 0.995
and an upper Bonferroni-conservative 97.5% bootstrap bound below 1; physical
and visible rate lower 97.5% bounds must exceed 1. Descriptive B/A, C/A, D/C,
interaction, absolute endpoints/deltas and the existing position/order/epoch
drift diagnostics are retained. Complete resource-clean data that misses a
performance target is `REJECTED`; CI overlap, drift, resource, RSS, identity,
correctness, crash or final-H2 failure is `INCONCLUSIVE`.

The top-level result is `QUALIFIED` or `REJECTED` only when all 32 children,
resource gates, the global RSS gate and final H2 pass. Otherwise it is
`INCONCLUSIVE`. `valid_for_performance` is true only for those complete
resource-clean `QUALIFIED`/`REJECTED` results. `activation_allowed` is always
false; no routing or automatic B39 main follow-on is permitted.
