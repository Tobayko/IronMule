# Prospective calibration return prediction

Eight source- and hardware-matched PR 60 arms provided six mixed starts and two independent factual-first starts. The forecast evaluated the recorded result of every optional calibration it could identify in the original constructor. Exact full precision was retained. It selected a new two-start information probe for the discrete padding table. Each of the pilot's continuous cost profiles and native pass curves had eight different results. The protected W4 calibration and inactive cuBLAS calibration did not produce an optional measured return on the active cached route.

The uniform Beta(1,1) prior gave an 18.18% chance of at least one miss in two new starts, conditional on exchangeability within the bound regime. Avoiding one prospective 22-worker trial if a stable-return hypothesis failed was the conditional information benefit. Its estimated 144.12 s exceeded the 72.06 s probe cost. The observed 71.65 s cost about $0.04382 of the existing lease. Future frequency of comparable restarts is unknown.

The pilot chose the current complete PR 60 profile before both independent new A100 starts. Both served 32 cold and 32 warm requests plus 18 correct factual answers. Each returned the predicted empty padding table. Their 16 unrounded fresh width prices differed. The retained observer records all six original timing samples per width, the actual median-of-five result and the direct public price-table side effect. Indirect and native graph and KV state was unobserved.

The separately registered three-sample simulation reproduced the final empty dispatch in both starts. Its empirical intervals allowed alternative choices at 15 of 16 widths. It therefore abstained both times. A different source arm from the original publication previously produced a nonempty table. The two holds qualify an exact observed forecast, with no authorization to reuse dynamic prices or to alter serving.

For replay, prepare the merged checkout as described in the neighboring a100-post-restart-60 directory, then use

```sh
d=lab/autonomous/results/a100-calibration-stability
r=lab/autonomous/results/a100-restart/d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3
src=/tmp/ironmule-pr60-replay
OPENBLAS_NUM_THREADS=1 python -m training.calibration_stability forecast \
  --history "$r" --control "$r/control.json" --checkout "$src" --hourly-usd 2.201904761904762 \
  --phase-prior lab/pk1/results/pod32/raw/ab2/trace_k1.txt \
  --phase-prior lab/pk1/results/pod32/raw/ab2/trace_k2.txt --out /tmp/ironmule-calibration-forecast.json
cmp "$d/prediction.json" /tmp/ironmule-calibration-forecast.json
OPENBLAS_NUM_THREADS=1 python -m training.calibration_stability holdout --forecast "$d/prediction.json" \
  --observation "$d/97251.json" --observation "$d/97252.json" --control "$r/control.json" \
  --weights "$r/startup-head.npz" --checkout "$src" --out /tmp/ironmule-calibration-holdout.json
cmp "$d/holdout.json" /tmp/ironmule-calibration-holdout.json
OPENBLAS_NUM_THREADS=1 python -m training.calibration_information --forecast "$d/prediction.json" \
  --observation "$d/97251.json" --observation "$d/97252.json" --control "$r/control.json" \
  --weights "$r/startup-head.npz" --checkout "$src" --out /tmp/ironmule-calibration-information.json
cmp "$d/information-decision.json" /tmp/ironmule-calibration-information.json
```
