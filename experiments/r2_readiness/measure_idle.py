"""What this machine's load actually looks like, so a threshold can mean something.

`ReadinessPolicy` blocks every campaign point at `max_load_1m = 0.75` and
`max_cpu_percent = 35.0`. Neither is reachable here, and the reason is a unit
mismatch rather than a busy machine:

* `load_1m` is compared raw (`readiness.py:301-303`, `normalize_load_by_cpus`
  defaults to `False`), so `0.75` means 7.5 % utilisation on a ten-core machine
  while the observed idle floor is `4.0`-`6.0` (`BACKLOG.md` G1).
* `cpu_percent` is the **sum** of every process's `%cpu` (`readiness.py:583`,
  `sum(cpu_values)`), where 100 % is one core. A ten-core machine can read
  `1000`, so a `35.0` ceiling is below a single busy core.

This records the distribution instead of guessing at it. The **minimum** over the
window is the idle estimate -- the same trick `friday_serve/throttle.py` uses,
and the only honest one on a machine that cannot be guaranteed quiet for ten
minutes. Mean and median are reported next to it so the reader can see how busy
the window actually was.

No GPU, no model, no claim about performance. Run:

    python experiments/r2_readiness/measure_idle.py --execute --minutes 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

STUDY_ID = "r2-readiness-idle-20260904-01"
INTERVAL_SECONDS = 5.0


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "min": None, "p05": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": round(ordered[0], 4),
        "p05": round(ordered[max(0, int(0.05 * len(ordered)) - 1)], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.fmean(ordered), 4),
        "max": round(ordered[-1], 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--minutes", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78

    import os

    from friday_optimizer.readiness import MacSystemProbe

    probe = MacSystemProbe()
    deadline = time.monotonic() + args.minutes * 60.0
    loads: list[float] = []
    cpus: list[float] = []
    errors: list[str] = []

    while time.monotonic() < deadline:
        snapshot = probe.sample()
        if snapshot.load_1m is not None:
            loads.append(float(snapshot.load_1m))
        if snapshot.cpu_percent is not None:
            cpus.append(float(snapshot.cpu_percent))
        errors.extend(snapshot.errors)
        time.sleep(INTERVAL_SECONDS)

    cpu_count = os.cpu_count() or 0
    report = {
        "study_id": STUDY_ID,
        "minutes": args.minutes,
        "interval_seconds": INTERVAL_SECONDS,
        "cpu_count": cpu_count,
        "load_1m": _summary(loads),
        # Units: sum over processes, 100 == one core. Divided by the core count
        # it becomes a fraction of the whole machine, which is what a "percent"
        # ceiling reads like to anyone setting one.
        "cpu_percent_summed": _summary(cpus),
        "cpu_percent_of_machine": _summary([value / cpu_count for value in cpus])
        if cpu_count
        else _summary([]),
        "probe_errors": sorted(set(errors)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_claim": False,
        "note": (
            "The window was not guaranteed quiet; read min/p05 as the idle "
            "estimate and median/mean as how busy this particular window was."
        ),
    }
    out = Path(__file__).resolve().parent / "idle.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": "measured", "report": str(out), **report["load_1m"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
