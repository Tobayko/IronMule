"""Measurement environment and interleaved A/B harness.

Two things every number in this repository needs: the machine state it was taken
under, and an arm order that cannot smuggle drift into the result. Thermal state
and power source are recorded because the same M1 Max is a different machine on
battery, in low power mode, or after twenty minutes of load.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RESEARCH = Path(__file__).resolve().parent.parent / "research"
RAW = RESEARCH / "raw"


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def environment() -> dict[str, Any]:
    """Everything that can move a timing without any code changing."""
    batt = _run(["pmset", "-g", "batt"])
    therm = _run(["pmset", "-g", "therm"])
    env: dict[str, Any] = {
        "power_source": "AC" if "AC Power" in batt else "battery" if batt else "unknown",
        "low_power_mode": "1" in _run(["pmset", "-g", "lowpowermode"]).split()[-1:] if _run(["pmset", "-g", "lowpowermode"]) else None,
        "thermal": {},
        "loadavg": os.getloadavg(),
        "os": _run(["sw_vers", "-productVersion"]).strip(),
        "python": sys.version.split()[0],
    }
    for line in therm.splitlines():
        if "=" in line:
            key, _, value = line.strip().partition("=")
            env["thermal"][key.strip().split()[-1]] = value.strip()
    try:
        import mlx.core as mx
        import mlx_lm
        env["mlx"] = mx.__version__ if hasattr(mx, "__version__") else None
        env["mlx_lm"] = mlx_lm.__version__
        env["mlx_peak_bytes"] = mx.get_peak_memory()
    except ImportError:
        pass
    env["git_commit"] = _run(["git", "-C", str(RESEARCH.parent), "rev-parse", "HEAD"]).strip()
    env["git_dirty"] = bool(_run(["git", "-C", str(RESEARCH.parent), "status", "--porcelain"]).strip())
    return env


def interleave(arms: list[str], processes: int) -> list[list[str]]:
    """Alternate the arm order per process so drift cannot favour one arm."""
    orders = []
    for index in range(processes):
        order = list(arms) if index % 2 == 0 else list(reversed(arms))
        orders.append(order)
    return orders


def summarise(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def paired_ratio(candidate: list[float], baseline: list[float], resamples: int = 10000,
                 seed: int = 20260825) -> dict[str, float]:
    """Median paired ratio with a bootstrap interval. Pairs stay pairs when resampled."""
    import random
    pairs = [c / b for c, b in zip(candidate, baseline)]
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        medians.append(statistics.median(draw))
    medians.sort()
    return {
        "median_ratio": statistics.median(pairs),
        "ci_low": medians[int(0.025 * resamples)],
        "ci_high": medians[int(0.975 * resamples)],
        "pairs": pairs,
    }


def record(experiment_id: str, payload: dict[str, Any]) -> Path:
    """Raw measurements are written before any interpretation happens."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{experiment_id}.json"
    payload = dict(payload, environment=environment(), recorded_at=time.time())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def _self_check() -> None:
    assert interleave(["a", "b"], 4) == [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]]
    s = summarise([3.0, 1.0, 2.0, 5.0, 4.0])
    assert s["median"] == 3.0 and s["min"] == 1.0 and s["max"] == 5.0 and s["n"] == 5
    r = paired_ratio([9.0, 8.0, 10.0], [10.0, 10.0, 10.0], resamples=200)
    assert 0.85 < r["median_ratio"] < 0.95, r
    assert r["ci_low"] <= r["median_ratio"] <= r["ci_high"]
    env = environment()
    assert env["power_source"] in ("AC", "battery", "unknown")
    print("bench self-check ok:", env["power_source"], env["thermal"], "load", round(env["loadavg"][0], 2))


if __name__ == "__main__":
    _self_check()
