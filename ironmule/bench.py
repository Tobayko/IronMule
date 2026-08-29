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


def swap_used_bytes() -> int | None:
    """Bytes currently swapped out, or None if the OS will not say.

    `vm.swapusage` reports like `total = 4096.00M  used = 2816.75M  free = 1279.25M`.
    Only `used` matters: a large swap *file* on an idle machine costs nothing, while a
    small one being written to invalidates every timing on the box.
    """
    field = _run(["sysctl", "-n", "vm.swapusage"]).partition("used =")[2].strip().split()
    if not field:
        return None
    text, scale = field[0].replace(",", "."), {"K": 1024, "M": 1024**2, "G": 1024**3}
    try:
        return int(float(text.rstrip("KMG")) * scale.get(text[-1], 1))
    except ValueError:
        return None


# A run is invalid once the machine starts swapping, whatever its allocation looked
# like. B7 measured 12B at a 17.51 GB peak with swap flat at 0.06 MB and the timings
# were clean; its confirmation run stayed under every byte ceiling and was discarded
# because swap climbed 2816 MB and every cell slowed by a uniform 1.10-1.15x. See R11.
SWAP_DELTA_LIMIT = 256 * 1024**2

# Coarse backstop against an unbounded allocation, not a measurement gate. Derived from
# installed memory so it does not have to be retyped per machine or per model size.
PEAK_CEILING_FRACTION = 0.6


def installed_memory_bytes() -> int | None:
    try:
        return int(_run(["sysctl", "-n", "hw.memsize"]).strip())
    except ValueError:
        return None


class MemoryGate:
    """Decide whether a measurement is still trustworthy, and record why.

    Constructed before the first block, then `check()`ed after each one. It returns a
    record rather than a bare bool: a guard that only aborts leaves an aborted run
    looking exactly like a short one, which is R10. Callers should put `record` into
    their result file once the schema question there is settled.

    `read_swap` is injectable so the two reference cases in R11 can be tested without
    having to push a real machine into swap.
    """

    def __init__(self, *, swap_delta_limit: int = SWAP_DELTA_LIMIT,
                 peak_ceiling: int | None = None, read_swap=swap_used_bytes,
                 read_installed=installed_memory_bytes):
        self._read_swap = read_swap
        self.swap_delta_limit = swap_delta_limit
        if peak_ceiling is None:
            # `read_installed` is injectable for the same reason `read_swap` is: the
            # case worth testing is a machine that answers neither, and passing
            # `peak_ceiling=None` cannot express that — it means "derive one".
            installed = read_installed()
            peak_ceiling = int(installed * PEAK_CEILING_FRACTION) if installed else None
        self.peak_ceiling = peak_ceiling
        self.baseline_swap = self._read_swap()
        # Both halves can be unavailable at once — an OS that reports neither swap nor
        # installed memory leaves `check()` unable to refuse anything. That is a run
        # with no gate at all, which must be visible afterwards rather than inferred
        # from an absence of aborts.
        self.inert = self.baseline_swap is None and peak_ceiling is None
        self.record: dict[str, Any] = {
            "baseline_swap_bytes": self.baseline_swap,
            "swap_delta_limit": swap_delta_limit,
            "peak_ceiling": peak_ceiling,
            "inert": self.inert,
            "blocks": [],
            "aborted": None,
        }
        if self.inert:
            print("WARNING: memory gate is inert — neither swap nor installed memory "
                  "could be read; this run has no memory condition", flush=True)

    def check(self, block_index: int, peak_bytes: int | None) -> str | None:
        """None to continue, else a human-readable reason to stop."""
        swap = self._read_swap()
        delta = None if swap is None or self.baseline_swap is None else swap - self.baseline_swap
        reason = None
        if delta is not None and delta > self.swap_delta_limit:
            reason = (f"swap grew {delta / 1024**2:.0f} MB above this run's baseline "
                      f"(limit {self.swap_delta_limit / 1024**2:.0f} MB); timings are not comparable")
        elif (self.peak_ceiling is not None and peak_bytes is not None
              and peak_bytes > self.peak_ceiling):
            reason = (f"peak {peak_bytes / 1024**3:.2f} GiB exceeds the backstop "
                      f"{self.peak_ceiling / 1024**3:.2f} GiB")
        self.record["blocks"].append({
            "block": block_index, "swap_bytes": swap, "swap_delta_bytes": delta,
            "peak_bytes": peak_bytes, "aborted_here": reason is not None,
        })
        if reason:
            self.record["aborted"] = {"block": block_index, "reason": reason}
        return reason


def environment() -> dict[str, Any]:
    """Everything that can move a timing without any code changing."""
    batt = _run(["pmset", "-g", "batt"])
    therm = _run(["pmset", "-g", "therm"])
    env: dict[str, Any] = {
        "power_source": "AC" if "AC Power" in batt else "battery" if batt else "unknown",
        "low_power_mode": "1" in _run(["pmset", "-g", "lowpowermode"]).split()[-1:] if _run(["pmset", "-g", "lowpowermode"]) else None,
        "thermal": {},
        "loadavg": os.getloadavg(),
        "swap_used_bytes": swap_used_bytes(),
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
    # R11's two reference cases, replayed from B7's recorded numbers. A gate that only
    # sounds reasonable is what this replaces, so both must come out the recorded way.
    MB, GB = 1024**2, 1024**3
    ceiling = int(32 * GB * PEAK_CEILING_FRACTION)

    # B7's 12B run: peak 17.51 GB per block, swap flat at 0.06 MB. It completed cleanly
    # and the old 12 GiB literal refused it. Four blocks must now pass.
    flat = MemoryGate(peak_ceiling=ceiling, read_swap=lambda: int(0.06 * MB))
    for block in range(4):
        assert flat.check(block, int(17.51 * GB)) is None, f"12B must not abort at block {block}"
    assert flat.record["aborted"] is None
    assert len(flat.record["blocks"]) == 4

    # B7's discarded confirmation run: every block under any byte ceiling, but swap
    # climbed to 2816 MB. The old guard passed it; this one must not.
    # First value is consumed as the run's own baseline, then one quiet block, then the
    # swap file grows the way it did during that run.
    climbing = iter([1 * MB, 60 * MB, 1800 * MB, 2816 * MB])
    pressured = MemoryGate(peak_ceiling=ceiling, read_swap=lambda: next(climbing))
    assert pressured.check(0, int(8.43 * GB)) is None, "must not abort before swap moves"
    assert pressured.check(1, int(8.43 * GB)) is not None, "must abort once swap climbs"
    assert "swap grew" in pressured.record["aborted"]["reason"]
    assert pressured.record["blocks"][-1]["aborted_here"] is True

    # The backstop still catches an unbounded allocation with swap perfectly flat.
    runaway = MemoryGate(peak_ceiling=ceiling, read_swap=lambda: 0)
    assert "backstop" in (runaway.check(0, int(25 * GB)) or "")

    # An OS that will not report swap must not silently disable the gate's other half.
    blind = MemoryGate(peak_ceiling=ceiling, read_swap=lambda: None)
    assert blind.check(0, int(17.51 * GB)) is None
    assert "backstop" in (blind.check(1, int(25 * GB)) or "")
    assert blind.record["inert"] is False, "one half missing is not the same as no gate"

    # Both halves missing is a run with no memory condition at all. It is allowed —
    # refusing to measure on an unfamiliar OS is worse — but it must say so.
    deaf = MemoryGate(read_swap=lambda: None, read_installed=lambda: None)
    assert deaf.inert is True and deaf.record["inert"] is True
    assert deaf.check(0, int(99 * GB)) is None, "an inert gate refuses nothing, by definition"

    env = environment()
    assert env["power_source"] in ("AC", "battery", "unknown")
    assert env["swap_used_bytes"] is None or env["swap_used_bytes"] >= 0
    print("bench self-check ok:", env["power_source"], env["thermal"], "load", round(env["loadavg"][0], 2),
          "swap", None if env["swap_used_bytes"] is None else f"{env['swap_used_bytes'] / 1024**2:.0f} MB")


if __name__ == "__main__":
    _self_check()
