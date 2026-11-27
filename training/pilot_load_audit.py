"""Measure both CPU load scopes and their cost in the shipped Pilot's decision entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from ironmule.pilot import Pilot


def audit(profile, samples):
    raw = profile.read_bytes()
    measured = json.loads(raw)["profile"]
    previous = os.environ.get("IRONMULE_PILOT_CPU_LOAD")
    pilots = {}
    rows = []
    try:
        for scope in ("host", "affinity"):
            os.environ["IRONMULE_PILOT_CPU_LOAD"] = scope
            pilots[scope] = Pilot(SimpleNamespace(drafter=None), measured["step_costs"], max_cap=measured["capacity"])
        for i in range(samples):
            time.sleep(0.06)
            row = {}
            scopes = list(pilots)
            for scope in reversed(scopes) if i % 2 else scopes:
                pilot = pilots[scope]
                t = time.perf_counter_ns()
                action = pilot.decide(1, 0, 0, 97)
                elapsed = time.perf_counter_ns() - t
                foreign = pilot.load.foreign
                t = time.perf_counter_ns()
                for _ in range(32):
                    pilot.decide(1, 0, 0, 97)
                cached = time.perf_counter_ns() - t
                row[scope] = {"foreign_cores": foreign, "shipped_decision": action,
                              "decision_with_refresh_us": elapsed / 1000, "cached_decision_mean_us": cached / 32000}
            rows.append(row)
    finally:
        for pilot in pilots.values():
            pilot.close()
        if previous is None:
            os.environ.pop("IRONMULE_PILOT_CPU_LOAD", None)
        else:
            os.environ["IRONMULE_PILOT_CPU_LOAD"] = previous
    return {"schema": "ironmule.cpu-load-scope-audit.v1", "affinity": sorted(os.sched_getaffinity(0)),
            "host_cpus": os.cpu_count(), "cost_profile_sha256": hashlib.sha256(raw).hexdigest(),
            "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), None),
            "profile_source": profile.name, "samples": rows,
            "medians": {scope: {key: statistics.median(r[scope][key] for r in rows)
                                for key in ("foreign_cores", "decision_with_refresh_us", "cached_decision_mean_us")}
                        for scope in pilots},
            "scope": "Actual passive observations and shipped Pilot decisions on a fixed request state. "
                     "The affinity option adds a scoped reader for a future separately qualified head. "
                     "Neither arm serves a language model in this observation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True, help="existing measured CPU thread trial JSON")
    p.add_argument("--samples", type=int, choices=range(2, 33), default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    from training.pilot_threads import save

    save(args.out, audit(args.profile, args.samples))


if __name__ == "__main__":
    main()
