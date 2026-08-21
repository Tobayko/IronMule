#!/usr/bin/env python3
"""Characterize the cooldown effect: why the first sample after a pause is slow.

H0.1 found that the first main sample after its 20 s cooldown was roughly twice
the median in all six sessions.  This tool measures how that penalty depends on
pause length, how many samples it takes to decay, and what causes it.

Every repetition is **internally paired**: the ratio is the first sample against
the steady state *of the same repetition*, so shared drift cancels.  That is the
same principle the A/A calibration showed to be decisive.

Pause lengths are visited in a deterministic SHA-256 shuffled order so that any
drift over the run cannot masquerade as a pause-length effect.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_SEED = 4051312678
SHAPE = 2048
SAMPLES_PER_BURST = 12
STEADY_FROM = 6
DEFAULT_PAUSES = (0.0, 0.05, 0.25, 0.75, 2.0, 5.0, 20.0)
DEFAULT_REPS = 8
# One-sided: only a slower-than-steady sample counts as contaminated.
STEADY_TOLERANCE = 0.10
GPU_WORK_BUDGET_S = 120.0
WALL_BUDGET_S = 30 * 60


# tools/ is loaded as loose scripts, not a package, so make the directory
# importable before pulling in the shared preconditions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import release_gate, require_ac_power  # noqa: E402


def shuffled_plan(pauses: tuple[float, ...], reps: int) -> list[tuple[float, int]]:
    """Deterministic interleaving of pause lengths; no library RNG involved."""

    plan = [(pause, index) for pause in pauses for index in range(reps)]
    plan.sort(key=lambda item: hashlib.sha256(f"{item[0]}:{item[1]}".encode()).hexdigest())
    return plan


def normalize(burst: list[int]) -> list[float]:
    """Express a burst relative to its own steady state."""

    if len(burst) <= STEADY_FROM:
        raise ValueError("burst is too short to have a steady state")
    steady = statistics.median(burst[STEADY_FROM:])
    if steady <= 0:
        raise ValueError("steady state must be positive")
    return [value / steady for value in burst]


def excess_samples(profile: list[float], tolerance: float = STEADY_TOLERANCE) -> float:
    """Time lost to the cooldown, expressed in steady-state sample equivalents.

    A "first index that stays settled" rule was tried first and abandoned: this
    device has no steady state that holds a +-10% band, so that rule reported
    nine contaminated samples even after a zero-second pause.  Summing the
    excess instead needs no clean cutoff to exist -- an occasional jittery
    sample adds a little, a genuine cooldown ramp adds a lot.

    Only the leading run of slow samples counts; the sum stops at the first
    sample that has returned to the steady band.
    """

    total = 0.0
    for value in profile:
        if value <= 1.0 + tolerance:
            break
        total += value - 1.0
    return round(total, 3)


def _self_check() -> int:
    """Offline checks of the decision logic; no GPU, no MLX."""

    flat = [100] * 12
    assert normalize(flat) == [1.0] * 12

    # A settled burst has no excess at all.
    assert excess_samples([1.0] * 12) == 0.0
    assert excess_samples([1.04] * 12) == 0.0

    # A ramp is charged its full overshoot: 2.0 + 0.5 sample equivalents.
    assert excess_samples([3.0, 1.5] + [1.0] * 10) == 2.5

    # Samples below steady state are jitter and cost nothing.
    assert excess_samples([0.90, 0.93] + [0.96] * 10) == 0.0

    # A later spike is not charged to the cooldown; only the leading run counts.
    assert excess_samples([1.0] * 10 + [1.5, 1.0]) == 0.0

    # A longer cooldown must cost strictly more than a shorter one.
    assert excess_samples([4.0, 2.0, 1.4] + [1.0] * 9) > excess_samples([2.0, 1.2] + [1.0] * 10)

    plan_a = shuffled_plan((0.0, 1.0), 3)
    plan_b = shuffled_plan((0.0, 1.0), 3)
    assert plan_a == plan_b, "plan must be deterministic"
    assert sorted(plan_a) == sorted([(p, i) for p in (0.0, 1.0) for i in range(3)])
    assert plan_a != sorted(plan_a), "plan must actually interleave"

    try:
        normalize([1, 2, 3])
    except ValueError:
        pass
    else:  # pragma: no cover - guard against a silently accepted short burst
        raise AssertionError("short burst must be refused")

    print(json.dumps({"self_check": "pass", "checks": 11}))
    return 0


def run(pauses: tuple[float, ...], reps: int) -> dict[str, object]:
    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT))
    from friday_h0.benchmark import _generate_fixture

    fixture = _generate_fixture(np, FIXTURE_SEED, shape=SHAPE)
    a = mx.array(fixture.a)
    b = mx.array(fixture.b)
    mx.eval(a, b)
    mx.synchronize()
    reference = np.array(mx.matmul(a, b), copy=False).astype(np.float32)

    def burst() -> list[int]:
        timings = []
        for _ in range(SAMPLES_PER_BURST):
            start = time.perf_counter_ns()
            value = mx.matmul(a, b)
            mx.eval(value)
            mx.synchronize()
            timings.append(time.perf_counter_ns() - start)
        return timings

    produced = mx.matmul(a, b)
    mx.eval(produced)
    if float(np.abs(np.array(produced, copy=False).astype(np.float32) - reference).max()) != 0.0:
        raise SystemExit("workload is not reproducible; refusing to report timing")

    for _ in range(3):
        burst()

    gpu_seconds = 0.0
    started = time.monotonic()
    profiles: dict[float, list[list[float]]] = {pause: [] for pause in pauses}
    for pause, _index in shuffled_plan(pauses, reps):
        if time.monotonic() - started > WALL_BUDGET_S:
            raise SystemExit("refused: wall budget exceeded")
        if pause > 0:
            time.sleep(pause)
        timings = burst()
        gpu_seconds += sum(timings) / 1e9
        if gpu_seconds > GPU_WORK_BUDGET_S:
            raise SystemExit("refused: GPU work budget exceeded")
        profiles[pause].append(normalize(timings))

    results = []
    for pause in pauses:
        rows = profiles[pause]
        median_profile = [
            statistics.median([row[index] for row in rows]) for index in range(SAMPLES_PER_BURST)
        ]
        first = [row[0] for row in rows]
        results.append(
            {
                "pause_seconds": pause,
                "first_sample_ratio": statistics.median(first),
                "first_sample_min": min(first),
                "first_sample_max": max(first),
                "median_profile": [round(value, 4) for value in median_profile],
                "excess_sample_equivalents": excess_samples(median_profile),
                "repetitions": len(rows),
            }
        )

    worst = max(entry["excess_sample_equivalents"] for entry in results)
    return {
        "samples_per_burst": SAMPLES_PER_BURST,
        "steady_from_sample": STEADY_FROM,
        "steady_tolerance": STEADY_TOLERANCE,
        "by_pause": results,
        "worst_excess_sample_equivalents": worst,
        "gpu_work_seconds": round(gpu_seconds, 3),
        "wall_seconds": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_cooldown_effect", allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    args = parser.parse_args(argv)

    gated = release_gate(args, _self_check)
    if gated is not None:
        return gated
    if not 2 <= args.reps <= 50:
        raise SystemExit("reps must be between 2 and 50")

    power = require_ac_power()
    report = run(DEFAULT_PAUSES, args.reps)
    report["power_source"] = power
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
