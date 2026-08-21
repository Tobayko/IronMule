"""Pure deterministic statistics for the H0 contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .constants import (
    AA_BOOTSTRAP_SEEDS,
    BOOTSTRAP_REPLICATES,
    ENGINEERING_EQUIVALENCE_BAND,
    SESSION_RATIO_BAND,
)


class StatisticsError(ValueError):
    """Raised for empty, non-finite, non-positive, or structurally invalid data."""


def _finite_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise StatisticsError(f"{name} must be a numeric sequence")
    result = tuple(values)
    if not result:
        raise StatisticsError(f"{name} must not be empty")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise StatisticsError(f"{name} contains a non-finite or non-numeric value")
    return tuple(float(value) for value in result)


def median(values: Sequence[float]) -> float:
    """Return the exact middle or arithmetic middle of sorted finite values."""

    ordered = sorted(_finite_values(values, "values"))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile in the closed interval [0, 1]."""

    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not math.isfinite(float(fraction)):
        raise StatisticsError("fraction must be finite")
    if not 0.0 <= float(fraction) <= 1.0:
        raise StatisticsError("fraction must be in [0, 1]")
    ordered = sorted(_finite_values(values, "values"))
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mad(values: Sequence[float]) -> float:
    """Return the median absolute deviation from the sample median."""

    sample = _finite_values(values, "values")
    center = median(sample)
    return median([abs(value - center) for value in sample])


def iqr(values: Sequence[float]) -> float:
    """Return Q3-Q1 using the contract's linear percentile interpolation."""

    return percentile(values, 0.75) - percentile(values, 0.25)


def sample_standard_deviation(values: Sequence[float]) -> float:
    """Return sample SD; one observation has SD zero by convention."""

    sample = _finite_values(values, "values")
    if len(sample) == 1:
        return 0.0
    center = sum(sample) / len(sample)
    return math.sqrt(sum((value - center) ** 2 for value in sample) / (len(sample) - 1))


def _pairs(sessions: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    if isinstance(sessions, Mapping):
        sessions = sessions.get("sessions")
    if sessions is None or isinstance(sessions, (str, bytes, bytearray)):
        raise StatisticsError("sessions must be a sequence or an object with sessions")
    result = []
    for session_index, session in enumerate(sessions):
        if isinstance(session, Mapping):
            session = session.get("blocks")
        if session is None:
            raise StatisticsError(f"session {session_index} has no blocks")
        blocks = []
        for block_index, pair in enumerate(session):
            if isinstance(pair, Mapping):
                if set(pair) != {"a", "b"}:
                    raise StatisticsError(f"session {session_index} block {block_index} must have a and b")
                a, b = pair["a"], pair["b"]
            else:
                if not isinstance(pair, Sequence) or len(pair) != 2:
                    raise StatisticsError(f"session {session_index} block {block_index} must be a pair")
                a, b = pair
            if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise StatisticsError("paired times must be numeric")
            if not math.isfinite(float(a)) or not math.isfinite(float(b)) or float(a) <= 0 or float(b) <= 0:
                raise StatisticsError("paired times must be finite and positive")
            blocks.append((float(a), float(b)))
        if not blocks:
            raise StatisticsError(f"session {session_index} has no blocks")
        result.append(tuple(blocks))
    if not result:
        raise StatisticsError("at least one session is required")
    return tuple(result)


def session_log_ratio(session: Any) -> float:
    """Return median_b(log(t_B/t_A)) for one session."""

    blocks = _pairs([session])[0]
    return median([math.log(b / a) for a, b in blocks])


def session_ratio(session: Any) -> float:
    """Return exp(median_b(log(t_B/t_A))) for one session."""

    return math.exp(session_log_ratio(session))


def set_ratio(session_ratios: Sequence[float]) -> float:
    """Return exp(median_s(log(R_s))) for one registered set."""

    ratios = _finite_values(session_ratios, "session_ratios")
    if any(ratio <= 0 for ratio in ratios):
        raise StatisticsError("session ratios must be positive")
    return math.exp(median([math.log(ratio) for ratio in ratios]))


class _SplitMix64:
    """Small version-independent deterministic PRNG for replayable bootstrap draws."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= self._MASK:
            raise StatisticsError("bootstrap seed must be an unsigned 64-bit integer")
        self.state = seed

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def below(self, upper: int) -> int:
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
            raise StatisticsError("random upper bound must be positive")
        span = 1 << 64
        limit = span - (span % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def hierarchical_bootstrap(
    sessions: Any,
    seed: int,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, ...]:
    """Resample sessions and blocks, reconstructing the exact set ratio each time."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise StatisticsError("replicates must be a positive integer")
    normalized = _pairs(sessions)
    rng = _SplitMix64(seed)
    results = []
    session_count = len(normalized)
    for _ in range(replicates):
        selected_session_logs = []
        for _ in range(session_count):
            source_session = normalized[rng.below(session_count)]
            block_logs = []
            for _ in range(len(source_session)):
                a, b = source_session[rng.below(len(source_session))]
                block_logs.append(math.log(b / a))
            selected_session_logs.append(median(block_logs))
        results.append(math.exp(median(selected_session_logs)))
    return tuple(results)


def _set_summary(sessions: Any, seed: int, replicates: int) -> dict[str, Any]:
    normalized = _pairs(sessions)
    session_logs = [median([math.log(b / a) for a, b in session]) for session in normalized]
    ratios = [math.exp(value) for value in session_logs]
    bootstraps = hierarchical_bootstrap(normalized, seed, replicates=replicates)
    return {
        "r_aa": set_ratio(ratios),
        "ci_lower": percentile(bootstraps, 0.025),
        "ci_upper": percentile(bootstraps, 0.975),
        "session_ratios": ratios,
        "session_log_ratios": session_logs,
        "session_log_sd": sample_standard_deviation(session_logs),
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "prng": "splitmix64_v1",
        "percentile_method": "linear_type7",
    }


def _validate_aa_set(sessions: Any, name: str) -> None:
    normalized = _pairs(sessions)
    if len(normalized) != 3:
        raise StatisticsError(f"{name} must contain exactly 3 sessions")
    if any(len(session) != 30 for session in normalized):
        raise StatisticsError(f"{name} must contain exactly 30 paired blocks per session")


def aa_gate(
    characterization: Any,
    confirmation: Any,
) -> dict[str, Any]:
    """Evaluate both A/A sets as an engineering null gate only."""

    _validate_aa_set(characterization, "characterization")
    _validate_aa_set(confirmation, "confirmation")
    sets = {
        "characterization": _set_summary(
            characterization, AA_BOOTSTRAP_SEEDS["characterization"], BOOTSTRAP_REPLICATES
        ),
        "confirmation": _set_summary(
            confirmation, AA_BOOTSTRAP_SEEDS["confirmation"], BOOTSTRAP_REPLICATES
        ),
    }
    low, high = ENGINEERING_EQUIVALENCE_BAND
    session_low, session_high = SESSION_RATIO_BAND
    failures = []
    for name, summary in sets.items():
        if not low <= summary["r_aa"] <= high:
            failures.append(f"{name}.r_aa_outside_engineering_band")
        if not low <= summary["ci_lower"] <= summary["ci_upper"] <= high:
            failures.append(f"{name}.ci_outside_engineering_band")
        if not summary["ci_lower"] <= 1.0 <= summary["ci_upper"]:
            failures.append(f"{name}.ci_does_not_contain_one")
        for index, ratio in enumerate(summary["session_ratios"]):
            if not session_low <= ratio <= session_high:
                failures.append(f"{name}.session_{index}_outside_session_band")
    return {
        "classification": "tie" if not failures else "h0_invalid",
        "h0_valid": not failures,
        "engineering_equivalence_gate": True,
        "scientific_equivalence_claim": False,
        "prng": "splitmix64_v1",
        "percentile_method": "linear_type7",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sets": sets,
        "failures": failures,
    }
