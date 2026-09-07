# R2: population accounting correction

The original campaign drew 400 actions but stored outcomes for only 352. The
48 omitted `persistent_process` draws have zero target-policy probability for
each of the four evaluated policies. Removing those draws from the denominator
while retaining the original logging propensities inflated the population IPS
estimate.

`experiments/r2_campaign/corrected_evaluation.py` reconstructs the frozen draw
schedule and validates it against the read-only record chain. Missing records
are accepted only for the documented omitted action. It rejects missing
target-supported observations, changed propensities and duplicate indices.

The separate `corpus_evaluation_corrected.json` keeps the original population
of 400 and the original holdout indices 320–399: 80 draws, 70 observed outcomes.
For example, full-corpus head-skip IPS is approximately 0.127579 rather than
0.144976. This is an estimator correction, not an additional runtime gain; the
descriptive per-action median gain is a different statistic.

The historical holdout remains inconclusive: three targets still have effective
sample sizes 7, 7 and 5 against the frozen minimum of 30. No policy is trained,
activated or retrospectively qualified. The original evaluator and report are
unchanged. Individual-row bootstrap intervals are explicitly diagnostic under
an unverified IID assumption; cross-commit timing comparability is not proved.
The known failed/retried attempt at index 378 remains a documented limitation.

Validation: six focused arithmetic and reconstruction-guard regressions. The
source SQLite database is opened read-only and its integrity/hash chain checked.
No GPU, model or performance benchmark is run by this correction.
