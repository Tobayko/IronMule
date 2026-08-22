"""Frozen robust estimators and hierarchical paired bootstrap."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import (
    BASELINE_PRECEDENCE,
    BASELINE_TIE_FRACTION,
    BOOTSTRAP_REPETITIONS,
)


class StatisticsError(ValueError):
    """Raw timing data does not satisfy the preregistered estimator contract."""


def _positive(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise StatisticsError(f"{name} must contain finite positive timings")
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise StatisticsError("percentile input is invalid")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    samples = _positive(values, "timings")
    median = statistics.median(samples)
    return {
        "median_ns": median,
        "mad_ns": statistics.median(abs(value - median) for value in samples),
        "p95_ns": percentile(samples, 0.95),
        "min_ns": min(samples),
        "max_ns": max(samples),
    }

def session_ratio(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    candidate_values = _positive(candidate, "candidate")
    baseline_values = _positive(baseline, "baseline")
    if len(candidate_values) != len(baseline_values):
        raise StatisticsError("paired timing lengths differ")
    logs = [math.log(c / b) for c, b in zip(candidate_values, baseline_values, strict=True)]
    return math.exp(statistics.median(logs))


def hierarchical_ratio(
    sessions: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    if len(sessions) != 3 or repetitions != BOOTSTRAP_REPETITIONS:
        raise StatisticsError("hierarchical estimator requires three sessions and 10,000 resamples")
    log_sessions: list[list[float]] = []
    session_ratios: list[float] = []
    for candidate, baseline in sessions:
        candidate_values = _positive(candidate, "candidate")
        baseline_values = _positive(baseline, "baseline")
        if len(candidate_values) != len(baseline_values):
            raise StatisticsError("paired timing lengths differ")
        logs = [
            math.log(c / b)
            for c, b in zip(candidate_values, baseline_values, strict=True)
        ]
        log_sessions.append(logs)
        session_ratios.append(math.exp(statistics.median(logs)))
    estimate = math.exp(statistics.median(math.log(value) for value in session_ratios))
    generator = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(repetitions):
        sampled_session_logs: list[float] = []
        for _session_slot in range(len(log_sessions)):
            source = log_sessions[generator.randrange(len(log_sessions))]
            resample = [source[generator.randrange(len(source))] for _ in range(len(source))]
            sampled_session_logs.append(statistics.median(resample))
        bootstrap.append(math.exp(statistics.median(sampled_session_logs)))
    return {
        "ratio": estimate,
        "ci95_low": percentile(bootstrap, 0.025),
        "ci95_high": percentile(bootstrap, 0.975),
        "session_ratios": session_ratios,
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions,
    }


def select_baseline(session_results: Sequence[Mapping[str, Sequence[float]]]) -> dict[str, Any]:
    if len(session_results) != 3:
        raise StatisticsError("baseline selection requires exactly three sessions")
    summaries: dict[str, dict[str, Any]] = {}
    for name in BASELINE_PRECEDENCE:
        per_session: list[float] = []
        for session in session_results:
            if set(session) != set(BASELINE_PRECEDENCE):
                raise StatisticsError("baseline characterization arms differ")
            per_session.append(statistics.median(_positive(session[name], name)))
        aggregate = math.exp(statistics.median(math.log(value) for value in per_session))
        summaries[name] = {"aggregate_median_ns": aggregate, "session_medians_ns": per_session}
    fastest = min(value["aggregate_median_ns"] for value in summaries.values())
    eligible = {
        name
        for name, value in summaries.items()
        if value["aggregate_median_ns"] <= fastest * (1.0 + BASELINE_TIE_FRACTION)
    }
    selected = next(name for name in BASELINE_PRECEDENCE if name in eligible)
    return {
        "selected": selected,
        "tie_fraction": BASELINE_TIE_FRACTION,
        "eligible": [name for name in BASELINE_PRECEDENCE if name in eligible],
        "arms": summaries,
    }
