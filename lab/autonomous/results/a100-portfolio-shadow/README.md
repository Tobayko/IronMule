# Two complete portfolio observations on the PR 60 A100 worker

Both independently started Qwen3-32B Q4_K_M workers used the trained startup Pilot
and the qualified W4 masks. Each ran the original startup calibrations, completed 32
cold and 32 warm generations, passed all 18 factual checks, and made 73 Pilot
decisions. The broker occupied 75.607 seconds of the existing A100 lease, or $0.04624.
The discovery protocol and complete donor vectors were frozen before either worker ran.

| Original worker and context | Sentinel drift | Largest error across 22 original prices | Decision |
| --- | ---: | ---: | --- |
| 97451, initial | 0.0875% | 1.929% | Accepted numerical prediction |
| 97452, initial | 0.1606% | 0.640% | Abstained |
| 97452, after delivery | 12.720% | 0.715% | Abstained; outside startup scope |

The accepted comparison exceeded the narrower historical error envelope. A fixed,
latest-donor rule also stayed within 2% at both starts. The incumbent continued to
use fresh prices throughout all delivered work. The independent, post-delivery
width-one sentinels cost 0.441 s. The subsequent two complete original methods cost
4.937 s. Its standalone pass price was 15.39% higher than width one in the directly
following complete pass calibration on the same engine. The observables include
hardware/affinity, timestamps, load, device snapshots, logical geometry and the
effective thread count. The exact age of the historical donors and the complete
internal graph/cache states remain unknown.

The field-read inventory found 266 reads of the padding action and 292 reads of
Pilot costs across the six delivered waves. No direct scheduling read of the pass
price table occurred in those waves. Under the measured PS1 setting the PR 60
engine returns before that scheduling lookup. The two full pass calibrations took
2.684 and 2.689 s. The ordinary product worker sends static metadata at readiness.
Its first streamed terminal frame calls the bridge counters, which export all six
current pass prices. The direct engine harness separately saves the same six prices
before completion. These consumer paths are verified against the original source
bytes. Product-v3's SSE refusal yielded zero complete A/B receipts.

Merely moving the full calibration to first use leaves it due by the first ordinary
completed response and can clear a live native context. The conditional upper
bound before accounting for this mandatory work is 687.2 s over the previous
hypothetical 256 starts. The prior complete source trial cost 815.0 s; actual restart
frequency is unknown. No verified full-work saving, source order, or activation
followed. Both original comparison rules and the earlier source rejection remain
recorded.

## CPU replay

From the framework repository, with the merged PR 60 source present locally:

```sh
git worktree add --detach /tmp/ironmule-pr60-portfolio 63db7745998383509602601d46ae4bbff0076648
d=lab/autonomous/results/a100-portfolio-shadow
r=lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
export OPENBLAS_NUM_THREADS=1
python -m training.portfolio_shadow forecast --data lab/autonomous/results --control "$r/control.json" --out /tmp/portfolio-forecast.json
cmp "$d/prediction.json" /tmp/portfolio-forecast.json
python -m training.portfolio_shadow grade --data lab/autonomous/results --control "$r/control.json" --weights "$r/startup-head.npz" --forecast "$d/prediction.json" --observation "$d/97451.json" --observation "$d/97452.json" --out /tmp/portfolio-grade.json
cmp "$d/result.json" /tmp/portfolio-grade.json
python -m training.portfolio_usage --data lab/autonomous/results --control "$r/control.json" --weights "$r/startup-head.npz" --checkout /tmp/ironmule-pr60-portfolio --out /tmp/portfolio-usage.json
cmp "$d/usage-replan.json" /tmp/portfolio-usage.json
```
