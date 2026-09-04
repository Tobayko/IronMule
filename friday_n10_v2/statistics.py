"""Deterministic paired estimators and hierarchical bootstrap for N10-v2."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


class StatisticsError(ValueError):
    """Raw measurements cannot support the preregistered estimator."""


class Sha256CounterRng:
    """Deterministic counter RNG with unbiased bounded integer draws."""

    def __init__(self, seed: int, *, domain: str) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise StatisticsError("seed must be an unsigned 64-bit integer")
        if not isinstance(domain, str) or not domain or len(domain) > 128:
            raise StatisticsError("RNG domain must be bounded non-empty text")
        self.seed = seed
        self.domain = domain.encode("ascii", errors="strict")
        self.counter = 0
        self._buffer: list[int] = []

    def _word(self) -> int:
        if self._buffer:
            return self._buffer.pop()
        if self.counter >= 1 << 64:
            raise StatisticsError("RNG counter exhausted")
        material = (
            len(self.domain).to_bytes(2, "big")
            + self.domain
            + self.seed.to_bytes(8, "big")
            + self.counter.to_bytes(8, "big")
        )
        self.counter += 1
        digest = hashlib.sha256(material).digest()
        words = [int.from_bytes(digest[offset : offset + 8], "big") for offset in range(0, 32, 8)]
        self._buffer.extend(reversed(words[1:]))
        return words[0]

    def draw(self, bound: int) -> int:
        if isinstance(bound, bool) or not isinstance(bound, int) or bound < 1:
            raise StatisticsError("draw bound must be a positive integer")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self._word()
            if value < limit:
                return value % bound

    def shuffle(self, values: list[Any]) -> None:
        for index in range(len(values) - 1, 0, -1):
            selected = self.draw(index + 1)
            values[index], values[selected] = values[selected], values[index]


def balanced_orders(count: int, *, seed: int, domain: str) -> list[str]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 2 or count % 2:
        raise StatisticsError("paired order count must be a positive even integer")
    result = ["ab"] * (count // 2) + ["ba"] * (count // 2)
    Sha256CounterRng(seed, domain=domain).shuffle(result)
    return result


def _median(values: Sequence[float | int]) -> float:
    if not values:
        raise StatisticsError("median requires at least one value")
    return float(statistics.median(values))


def session_metrics(blocks: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if len(blocks) < 2:
        raise StatisticsError("a paired session requires at least two blocks")
    a_values: list[int] = []
    b_values: list[int] = []
    log_ratios: list[float] = []
    for index, block in enumerate(blocks):
        a_ns = block.get("a_ns")
        b_ns = block.get("b_ns")
        if (
            isinstance(a_ns, bool)
            or not isinstance(a_ns, int)
            or a_ns <= 0
            or isinstance(b_ns, bool)
            or not isinstance(b_ns, int)
            or b_ns <= 0
        ):
            raise StatisticsError(f"block {index} contains an invalid duration")
        a_values.append(a_ns)
        b_values.append(b_ns)
        log_ratios.append(math.log(b_ns / a_ns))
    log_centre = _median(log_ratios)
    ratio = math.exp(log_centre)
    return {
        "blocks": len(blocks),
        "ratio": ratio,
        "effect_percent": 100.0 * (ratio - 1.0),
        "median_a_ns": int(statistics.median(a_values)),
        "median_b_ns": int(statistics.median(b_values)),
        "sd_log_ratio": statistics.stdev(log_ratios),
    }


def hierarchical_bootstrap(
    sessions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    seed: int,
    draws: int,
    confidence: float,
) -> dict[str, float | int]:
    if len(sessions) < 2 or any(len(blocks) < 2 for blocks in sessions):
        raise StatisticsError("hierarchical bootstrap needs at least two non-trivial sessions")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise StatisticsError("bootstrap draws must be at least 100")
    if not isinstance(confidence, float) or not 0.5 < confidence < 1.0:
        raise StatisticsError("confidence must be a float between 0.5 and 1")
    raw_logs = [
        [math.log(float(block["b_ns"]) / float(block["a_ns"])) for block in blocks]
        for blocks in sessions
    ]
    point = math.exp(_median([_median(values) for values in raw_logs]))
    rng = Sha256CounterRng(seed, domain="n10v2-hierarchical-bootstrap-v1")
    replays: list[float] = []
    for _ in range(draws):
        centres: list[float] = []
        for _session_index in range(len(raw_logs)):
            selected = raw_logs[rng.draw(len(raw_logs))]
            centres.append(_median([selected[rng.draw(len(selected))] for _ in selected]))
        replays.append(math.exp(_median(centres)))
    replays.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(draws - 1, int(math.floor(tail * draws))))
    high_index = max(0, min(draws - 1, int(math.ceil((1.0 - tail) * draws)) - 1))
    return {
        "ratio": point,
        "ci_low": replays[low_index],
        "ci_high": replays[high_index],
        "confidence": confidence,
        "draws": draws,
        "seed": seed,
        "sessions": len(sessions),
    }


__all__ = [
    "Sha256CounterRng",
    "StatisticsError",
    "balanced_orders",
    "hierarchical_bootstrap",
    "session_metrics",
]
