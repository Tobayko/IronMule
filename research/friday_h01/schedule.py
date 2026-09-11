"""Materialize the fixed Design A schedule with unbiased SHA-256 shuffling."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from .canonical import (
    CanonicalError,
    bounded_text,
    canonical_json_bytes,
    canonical_sha256,
    exact_int64,
    exact_keys,
    int64,
    nonnegative_int64,
)
from .constants import (
    BURN_IN_BLOCKS,
    BURN_IN_SAMPLES,
    LONG_GAP_NS,
    LONG_LABEL,
    MAIN_BLOCKS,
    MAIN_SAMPLES,
    SCHEMA_VERSION,
    SAMPLES_PER_BLOCK,
    SCHEDULE_ALGORITHM,
    SESSION_SPECS,
    SHORT_GAP_NS,
    SHORT_LABEL,
    TOTAL_SAMPLES,
)


class ScheduleError(ValueError):
    """Raised when schedule identity or materialization is not exact."""


class Sha256CounterRng:
    """Deterministic counter RNG with rejection sampling for bounded integers."""

    def __init__(self, seed: int, *, domain: str) -> None:
        self.seed = int64(seed, "rng seed", minimum=0)
        if not isinstance(domain, str) or not domain or len(domain) > 128:
            raise ScheduleError("rng domain must be bounded non-empty text")
        self.domain = domain.encode("ascii", errors="strict")
        self.counter = 0

    def _word(self) -> int:
        if self.counter > (1 << 63) - 1:
            raise ScheduleError("rng counter exhausted")
        material = (
            b"friday_h01.sha256_counter_v1\x00"
            + len(self.domain).to_bytes(2, "big")
            + self.domain
            + self.seed.to_bytes(8, "big", signed=False)
            + self.counter.to_bytes(8, "big", signed=False)
        )
        self.counter += 1
        return int.from_bytes(hashlib.sha256(material).digest(), "big")

    def draw(self, bound: int) -> int:
        bound = int64(bound, "rng bound", minimum=1)
        space = 1 << 256
        limit = space - (space % bound)
        while True:
            candidate = self._word()
            if candidate < limit:
                return candidate % bound

    def shuffle(self, values: list[str]) -> None:
        for index in range(len(values) - 1, 0, -1):
            selected = self.draw(index + 1)
            values[index], values[selected] = values[selected], values[index]


_SCHEDULE_KEYS = frozenset({"schema_version", "algorithm", "session_id", "seed", "entries", "sha256"})
_ENTRY_KEYS = frozenset(
    {
        "sample_index",
        "phase",
        "phase_index",
        "block_index",
        "position",
        "gap_label",
        "requested_gap_ns",
    }
)


def _entries(session_id: str, seed: int) -> list[dict[str, Any]]:
    rng = Sha256CounterRng(seed, domain=f"{SCHEDULE_ALGORITHM}:{session_id}")
    entries: list[dict[str, Any]] = []
    sample_index = 0
    for phase, blocks in (("burn_in", BURN_IN_BLOCKS), ("main", MAIN_BLOCKS)):
        phase_index = 0
        for block_index in range(blocks):
            labels = [SHORT_LABEL, SHORT_LABEL, LONG_LABEL, LONG_LABEL]
            rng.shuffle(labels)
            for position, label in enumerate(labels):
                entries.append(
                    {
                        "sample_index": sample_index,
                        "phase": phase,
                        "phase_index": phase_index,
                        "block_index": block_index,
                        "position": position,
                        "gap_label": label,
                        "requested_gap_ns": SHORT_GAP_NS if label == SHORT_LABEL else LONG_GAP_NS,
                    }
                )
                sample_index += 1
                phase_index += 1
    return entries


def materialize_schedule(session_id: str) -> dict[str, Any]:
    if session_id not in SESSION_SPECS:
        raise ScheduleError("session id is not registered")
    seed = SESSION_SPECS[session_id][2]
    body = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": SCHEDULE_ALGORITHM,
        "session_id": session_id,
        "seed": seed,
        "entries": _entries(session_id, seed),
    }
    return {**body, "sha256": canonical_sha256(body)}


def validate_schedule(value: Any, *, name: str = "schedule") -> dict[str, Any]:
    try:
        schedule = exact_keys(value, _SCHEDULE_KEYS, name)
        exact_int64(schedule["schema_version"], f"{name}.schema_version", SCHEMA_VERSION)
        bounded_text(schedule["algorithm"], f"{name}.algorithm", maximum=64)
        session_id = schedule["session_id"]
        if not isinstance(session_id, str) or session_id not in SESSION_SPECS:
            raise ScheduleError(f"{name}.session_id is not registered")
        exact_int64(schedule["seed"], f"{name}.seed", SESSION_SPECS[session_id][2])
        entries = schedule["entries"]
        if (
            not isinstance(entries, Sequence)
            or isinstance(entries, (str, bytes, bytearray))
            or len(entries) != TOTAL_SAMPLES
        ):
            raise ScheduleError(f"{name}.entries has the wrong sample count")
        for index, entry_value in enumerate(entries):
            entry_name = f"{name}.entries[{index}]"
            entry = exact_keys(entry_value, _ENTRY_KEYS, entry_name)
            nonnegative_int64(entry["sample_index"], f"{entry_name}.sample_index", maximum=TOTAL_SAMPLES - 1)
            bounded_text(entry["phase"], f"{entry_name}.phase", maximum=16)
            nonnegative_int64(entry["phase_index"], f"{entry_name}.phase_index", maximum=MAIN_SAMPLES - 1)
            nonnegative_int64(entry["block_index"], f"{entry_name}.block_index", maximum=MAIN_BLOCKS - 1)
            nonnegative_int64(entry["position"], f"{entry_name}.position", maximum=SAMPLES_PER_BLOCK - 1)
            bounded_text(entry["gap_label"], f"{entry_name}.gap_label", maximum=32)
            nonnegative_int64(entry["requested_gap_ns"], f"{entry_name}.requested_gap_ns")
        expected = materialize_schedule(session_id)
        if schedule != expected:
            raise ScheduleError(f"{name} differs from the registered materialization")
        if sum(entry["phase"] == "burn_in" for entry in entries) != BURN_IN_SAMPLES:
            raise ScheduleError(f"{name} burn-in count is not registered")
        if sum(entry["phase"] == "main" for entry in entries) != MAIN_SAMPLES:
            raise ScheduleError(f"{name} main count is not registered")
        canonical_json_bytes(schedule)
        return expected
    except CanonicalError as exc:
        raise ScheduleError(str(exc)) from exc


__all__ = ["ScheduleError", "Sha256CounterRng", "materialize_schedule", "validate_schedule"]
