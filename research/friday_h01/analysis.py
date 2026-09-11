"""Deterministic real-time engineering-envelope analysis for H0.1 v2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_sha256
from .constants import (
    ACF_LAGS,
    BURN_IN_SAMPLES,
    CHANGEPOINT_MAX_SPLIT,
    CHANGEPOINT_MIN_SPLIT,
    GATE_LIMITS,
    MAIN_SAMPLES,
    PHASE,
    SCHEMA_VERSION,
    SESSION_COMPLETE_STATUS,
    SESSION_INVALID_STATUS,
    SHORT_LABEL,
    STUDY,
    TOTAL_SAMPLES,
)
from .protocol import ProtocolError, validate_manifest, validate_trace


def _median(values: Sequence[float | int]) -> float:
    if not values:
        raise ProtocolError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


def _theil_sen(times: Sequence[float], values: Sequence[float]) -> float:
    slopes = [
        (values[right] - values[left]) / (times[right] - times[left])
        for left in range(len(values) - 1)
        for right in range(left + 1, len(values))
    ]
    return _median(slopes)


def _max_changepoint(values: Sequence[float]) -> tuple[int, float]:
    selected_split = CHANGEPOINT_MIN_SPLIT
    selected_effect = math.expm1(
        _median(values[selected_split:]) - _median(values[:selected_split])
    )
    for split in range(CHANGEPOINT_MIN_SPLIT + 1, CHANGEPOINT_MAX_SPLIT + 1):
        effect = math.expm1(_median(values[split:]) - _median(values[:split]))
        if abs(effect) > abs(selected_effect):
            selected_split = split
            selected_effect = effect
    return selected_split, selected_effect


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = ((cursor + 1) + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return 0.0
    return covariance / math.sqrt(left_ss * right_ss)


def _spearman_acf(values: Sequence[float], lag: int) -> float:
    return _pearson(_average_ranks(values[:-lag]), _average_ranks(values[lag:]))


def _gate(observed: float, name: str) -> dict[str, float | str]:
    operator, limit = GATE_LIMITS[name]
    passes = observed <= limit if operator == "<=" else observed >= limit
    return {
        "status": "pass" if passes else "fail",
        "observed": observed,
        "operator": operator,
        "limit": limit,
    }


def _result_base(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study": STUDY,
        "run_id": manifest["run_id"],
        "manifest_sha256": canonical_sha256(manifest),
        "trace_sha256": None,
        "status": SESSION_INVALID_STATUS,
        "conclusion": "invalid_input",
        "action": "no_h0_conclusion",
        "h0_reclassification": False,
        "promotion_applicable": False,
        "error": None,
        "sample_accounting": None,
        "metrics": None,
        "gates": None,
    }


def _invalid_result(manifest: Mapping[str, Any], error: ProtocolError) -> dict[str, Any]:
    result = _result_base(manifest)
    result["error"] = {
        "code": "trace_contract_violation",
        "message": str(error)[:512] or "trace contract violation",
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def analyze_trace(manifest: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    """Characterize one session; only a contract fault changes its completion status."""

    checked_manifest = validate_manifest(manifest)
    try:
        checked_trace = validate_trace(checked_manifest, trace)
    except ProtocolError as exc:
        return _invalid_result(checked_manifest, exc)

    main = checked_trace["samples"][BURN_IN_SAMPLES:]
    if len(main) != MAIN_SAMPLES or len(checked_trace["samples"]) != TOTAL_SAMPLES:
        return _invalid_result(checked_manifest, ProtocolError("sample accounting is not registered"))

    starts = [sample["start_ns"] for sample in main]
    first_start = starts[0]
    times_seconds = [(start - first_start) / 1_000_000_000.0 for start in starts]
    observed_span_seconds = times_seconds[-1]
    if observed_span_seconds <= 0.0:
        return _invalid_result(checked_manifest, ProtocolError("main real-time span is not positive"))

    log_durations = [math.log(sample["duration_ns"]) for sample in main]
    labels = [sample["gap_label"] for sample in main]
    pace_medians = {
        label: _median(
            [
                value
                for value, candidate in zip(log_durations, labels, strict=True)
                if candidate == label
            ]
        )
        for label in sorted(set(labels))
    }
    residuals = [
        value - pace_medians[label]
        for value, label in zip(log_durations, labels, strict=True)
    ]
    actual_gaps = [sample["gap_end_ns"] - sample["gap_start_ns"] for sample in main]
    overshoots = [
        actual - sample["requested_gap_ns"]
        for actual, sample in zip(actual_gaps, main, strict=True)
    ]

    slope = _theil_sen(times_seconds, residuals)
    trajectory_effect = math.expm1(slope * observed_span_seconds)
    changepoint_split, changepoint_effect = _max_changepoint(residuals)
    acf_values = {lag: _spearman_acf(residuals, lag) for lag in ACF_LAGS}
    ess_denominator = 1.0 + 2.0 * sum(max(0.0, value) for value in acf_values.values())
    effective_sample_size = min(float(MAIN_SAMPLES), max(1.0, MAIN_SAMPLES / ess_denominator))

    short_median = _median(
        [value for value, label in zip(log_durations, labels, strict=True) if label == SHORT_LABEL]
    )
    long_median = _median(
        [value for value, label in zip(log_durations, labels, strict=True) if label != SHORT_LABEL]
    )
    pace_effect = math.expm1(long_median - short_median)
    normalized = [
        math.exp(value - pace_medians[label])
        for value, label in zip(log_durations, labels, strict=True)
    ]
    tail_ratio = max(normalized) / _median(normalized)

    metrics = {
        "transform": "natural_log_ns",
        "residual_sha256": canonical_sha256(residuals),
        "actual_gap_sha256": canonical_sha256(actual_gaps),
        "trajectory": {
            "slope_per_second": slope,
            "effect_ratio": trajectory_effect,
            "observed_span_seconds": observed_span_seconds,
        },
        "changepoint": {"split": changepoint_split, "effect_ratio": changepoint_effect},
        "acf": {f"lag{lag}": acf_values[lag] for lag in ACF_LAGS},
        "effective_sample_size": effective_sample_size,
        "pace_effect_ratio": pace_effect,
        "tail_ratio": tail_ratio,
        "gap_adherence": {
            "max_overshoot_ns": max(overshoots),
            "median_overshoot_ns": _median(overshoots),
        },
    }
    gates = {
        "trend": _gate(abs(trajectory_effect), "trend"),
        "changepoint": _gate(abs(changepoint_effect), "changepoint"),
        "acf": _gate(max(abs(value) for value in acf_values.values()), "acf"),
        "ess": _gate(effective_sample_size, "ess"),
        "pacing": _gate(abs(pace_effect), "pacing"),
        "tail": _gate(tail_ratio, "tail"),
    }
    result = _result_base(checked_manifest)
    result.update(
        {
            "trace_sha256": canonical_sha256(checked_trace),
            "status": SESSION_COMPLETE_STATUS,
            "conclusion": "session_characterized",
            "error": None,
            "sample_accounting": {
                "trace_samples": TOTAL_SAMPLES,
                "burn_in_samples": BURN_IN_SAMPLES,
                "main_samples": MAIN_SAMPLES,
                "analysis_samples": MAIN_SAMPLES,
                "dropped_samples": 0,
                "adaptive_stop": False,
                "outlier_deletion": False,
            },
            "metrics": metrics,
            "gates": gates,
        }
    )
    result["decision_sha256"] = canonical_sha256(result)
    return result


__all__ = ["analyze_trace"]
