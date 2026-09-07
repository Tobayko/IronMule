"""Pure, model-free summary statistics shared by measurement code."""

from __future__ import annotations

import math
import random
import statistics as _statistics
from collections.abc import Sequence
from numbers import Real
from typing import Any


def _finite_values(values: Sequence[Real], *, name: str, allow_zero: bool) -> list[Real]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    result: list[Real] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must contain real finite numbers")
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must contain real finite numbers") from exc
        if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must contain finite {qualifier} numbers")
        result.append(value)
    return result


def summarise(samples: Sequence[Real]) -> dict[str, Any]:
    """Return stable descriptive statistics for finite non-negative samples.

    Zero remains valid for historical generic measurements.  Timing ratios use
    :func:`paired_ratio`, which requires strictly positive values because it
    divides by each baseline.
    """
    ordered = sorted(_finite_values(samples, name="samples", allow_zero=True))
    return {
        "n": len(ordered),
        "median": _statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "stdev": _statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def paired_ratio(
    candidate: Sequence[Real],
    baseline: Sequence[Real],
    resamples: int = 10000,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Return paired candidate/baseline ratios and a median bootstrap interval."""
    candidates = _finite_values(candidate, name="candidate", allow_zero=False)
    baselines = _finite_values(baseline, name="baseline", allow_zero=False)
    if len(candidates) != len(baselines):
        raise ValueError("candidate and baseline must have equal lengths")
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    pairs = [candidate_value / baseline_value for candidate_value, baseline_value in zip(candidates, baselines)]
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        medians.append(_statistics.median(draw))
    medians.sort()
    return {
        "median_ratio": _statistics.median(pairs),
        "ci_low": medians[int(0.025 * resamples)],
        "ci_high": medians[int(0.975 * resamples)],
        "pairs": pairs,
    }


__all__ = ["paired_ratio", "summarise"]
