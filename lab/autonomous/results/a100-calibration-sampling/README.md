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
future starts (about 1,372 including the preceding two 71.65 s and 69.71 s research
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
