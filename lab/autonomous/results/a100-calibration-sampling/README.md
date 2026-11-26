# An order selected from measured calibration work

The prior trial predicted the discrete return twice. Its original three-sample variant
abstained at 15 of 16 widths in both starts. The complete raw series showed why that
information was useful: several early observations still included settling time after
a width change. The programmer-supplied next search family varied warmups and fresh
measurement counts. The planner ranked all ten admissible prefixes using both whole
prior starts. Every prefix is charged all its warmups and measurements; only original
calls beyond its end can count as potentially avoidable work.

Before either new seed ran, the planner committed to two warmups and two fresh timing
samples at every width. It required the same result at every corner of an empirical
price interval and at most 2% error against each full original median. Two distinct,
independently initialized A100 startup workers ran the complete unchanged original
measurement. Each used the actual, previously trained CPU startup head, ran 32 cold
and 32 warm completions, and answered all 18 factual questions correctly.

* 97351 passed at all 16 widths, with 0.4001% maximum price error. The measured retained
  prefix took 1.46872 s. Its unused tail in the original measurement took 0.68169 s.
* 97352 passed at all 16 widths, with 0.4498% maximum price error. The measured retained
  prefix took 1.46743 s. Its unused tail took 0.68156 s.

The observations caused the planner to issue the source-specific order in decision.json.
Its identity is 4e87c90dad8d9f21db28a69501613f44a98b9399b1b5838316f276978de288e7.
The two original holds took 69.71 s including controller and broker, about $0.04264 of
the existing lease. The separate qualification estimate is at least 792.67 s. Under
the smaller modeled saving, recouping that qualification alone requires 1,164 similar
future starts (about 1,371 including the preceding two 71.65 s and 69.71 s research
pairs). Startup frequency is unknown. Actual source and full request performance require
the separate isolated source trial. The empirical bounds do not give a coverage guarantee
on different workloads, devices, driver states or future load.

To reconstruct the original forecast and resulting decision on CPU from the same
merged checkout used in the neighboring calibration-stability replay:

```sh
d=lab/autonomous/results/a100-calibration-sampling
prior=lab/autonomous/results/a100-calibration-stability
r=lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
src=/tmp/ironmule-pr60-replay
OPENBLAS_NUM_THREADS=1 python -m training.calibration_sampling forecast --prior-dir "$prior" \
  --control "$r/control.json" --weights "$r/startup-head.npz" --checkout "$src" \
  --out /tmp/ironmule-sampling-forecast.json
cmp "$d/prediction.json" /tmp/ironmule-sampling-forecast.json
OPENBLAS_NUM_THREADS=1 python -m training.calibration_sampling decide --forecast "$d/prediction.json" \
  --observation "$d/97351.json" --observation "$d/97352.json" \
  --control "$r/control.json" --weights "$r/startup-head.npz" --checkout "$src" \
  --out /tmp/ironmule-sampling-order.json
cmp "$d/decision.json" /tmp/ironmule-sampling-order.json
```

## Physical source verdict

The exported 22-worker target trial regraded the authored source on the A100. Its
instrumented screen observed 64 real calibration calls across 16 widths, with 32 fresh
measurement calls. Missing, expired and foreign profiles, a wrong pin and injected
jitter all restored full fresh measurement. The regular trial then ran six mixed
and two factual-first pairs. Across those pairs each source returned 256 cold mixed
and 256 warm mixed completions (8,192 output tokens in each cohort of waves) and
answered 216/216 independently graded factual prompts correctly.

| Physical quantity | PR 60 reference | Proposed source |
| --- | ---: | ---: |
| Six mixed cold waves, total s including startup and dispatch | 100.533 | 104.491 |
| Six complete mixed workers, total s | 216.983 | 219.788 |
| Median mixed cold wave including startup and dispatch, s | 16.775 | 16.200 |
| Median full mixed worker, s | 36.188 | 35.675 |
| Median warm mixed p95 time to first token, s | 2.311 | 2.310 |
| Median first factual wave including startup and dispatch, s | 13.927 | 13.066 |

Paired mixed median speedup was 1.03244. Seeds 96854 and 96855 regressed, so the
unchanged all-pairs requirement rejected the source. At 96854 the adaptive measurement
ran in 1.584 s and the first wave itself was within 3 ms of the baseline. The
initialization interval was 4.990 s longer. These workers did not record the causal
subphase. At 96855 ambiguity triggered the correct full fallback, with 160 reported
calls and 3.823 s of calibration. Every measured mixed cold and warm wave reported zero
padded tokens. The retained counters give no first-read timestamps for the scheduler's
price tables, no per-factual-wave padding count, and do not quantify warm-up effects.

The protected source trial took 815.02 s ($0.49850 of the existing lease). The two
information cohorts and target trial together account for $0.58496. Earlier exploratory
source attribution accounted for a separate $0.08839. The prior active release continues.
The source has no realized payback. The terminal control-host replan updates the
source-scoped uncertainty estimate from the measured adaptive/fallback modes,
records both regressions and closes this order without booking more resources.

From the framework checkout, reproduce the accounting with

```sh
d=lab/autonomous/results/a100-calibration-sampling
order=4e87c90dad8d9f21db28a69501613f44a98b9399b1b5838316f276978de288e7
OPENBLAS_NUM_THREADS=1 python -m lab.pk1.sampling_accounting \
  --data lab/autonomous/results --folder "$d/trial-$order" --out /tmp/ironmule-sampling-accounting.json
cmp "$d/target-accounting.json" /tmp/ironmule-sampling-accounting.json
```
