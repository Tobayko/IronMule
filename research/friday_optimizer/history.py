"""Immutable, redacted session history for the Friday optimizer.

The history layer is deliberately a very small adapter over
``OptimizationMemoryV2``.  It does not have a second file format or a second
source of truth: events are encoded as ordinary v2 records and all reads use a
read-only SQLite connection.  Event payloads are bounded and are never allowed
to contain prompts, model output, or process logs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping

#: An immutable empty mapping, shared. Spelled as a factory because a
#: dataclass default must be hashable on Python 3.11 and ``mappingproxy``
#: is not; 3.12 relaxed that check to list/dict/set only (gh-96151), which
#: is why the bare default worked here and nowhere else.
_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})

from .canonical import CanonicalJSONError, canonical_bytes, loads_strict, sha256_hex
from .memory import OptimizationMemoryV2, ReadOnlyMemoryView
from .records import DataPhase, OptimizationRecord, QualityClass, RecordKind

MAX_EVENT_PAYLOAD_BYTES = 256 * 1024
_HASH = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_HISTORY_KINDS = frozenset(
    {
        RecordKind.SYSTEM,
        RecordKind.IMPORT,
        RecordKind.DATASET,
        RecordKind.BENCHMARK,
        RecordKind.PROFILE,
        RecordKind.PROMOTION,
    }
)
HISTORY_KIND_PHASE = MappingProxyType({
    RecordKind.SYSTEM: DataPhase.FEATURE,
    RecordKind.IMPORT: DataPhase.FEATURE,
    RecordKind.DATASET: DataPhase.FEATURE,
    RecordKind.BENCHMARK: DataPhase.LABEL,
    RecordKind.PROFILE: DataPhase.LABEL,
    RecordKind.PROMOTION: DataPhase.LABEL,
})

# Keys are checked by name rather than by value.  That makes it impossible to
# accidentally persist a secret after a caller adds a new nested structure.
_PRIVATE_KEY_PARTS = frozenset(
    {"prompt", "prompts", "log", "logs", "stdout", "stderr", "completion", "response", "output", "text", "path", "source", "input", "request"}
)
_REASON = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_PAYLOAD_KEYS = {
    RecordKind.SYSTEM: frozenset({"code", "ok", "details", "value", "version"}),
    RecordKind.IMPORT: frozenset({"inventory_hash", "sink_hash", "records_seen", "records_written", "records_idempotent", "code"}),
    RecordKind.DATASET: frozenset({"records", "card", "code"}),
    RecordKind.BENCHMARK: frozenset({"status", "qualified", "ratios", "confidence_intervals", "no_activation", "registry_hash", "reasons", "metrics", "fingerprint_hash", "dataset_hash", "candidate_id", "evidence_hash", "code"}),
    RecordKind.PROFILE: frozenset({"profile_id", "profile_hash", "fingerprint_hash", "candidate_id", "metrics", "qualified", "code"}),
    RecordKind.PROMOTION: frozenset({"profile_id", "profile_hash", "fingerprint_hash", "state", "rollback", "candidate_id", "decision", "reason_code", "code"}),
}


class HistoryError(ValueError):
    """An event is malformed or cannot be safely read."""


class HistoryIntegrityError(HistoryError):
    """A stored history row failed strict decoding or integrity checks."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _private_key(key: str) -> bool:
    parts = {part for part in re.split(r"[^a-z0-9]+", key.lower()) if part}
    return bool(parts & _PRIVATE_KEY_PARTS)


def _redact(value: Any, *, depth: int = 0) -> Any:
    """Drop sensitive fields while retaining a deterministic JSON value.

    The event API accepts evidence summaries from untrusted callers.  Dropping
    private keys is safer than attempting to inspect or truncate their values;
    a caller can still bind omitted raw evidence with ``evidence_hash``.
    """

    if depth > 16:
        raise HistoryError("history payload exceeds maximum nesting depth")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HistoryError("history payload keys must be strings")
            if not _private_key(key):
                result[key] = _redact(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item, depth=depth + 1) for item in value]
    return value


def _hash(value: str | None, field: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise HistoryError(f"{field} must be a lowercase SHA-256 hash or empty")
    return value


def _identifier(value: str | None, field: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise HistoryError(f"{field} must be a bounded safe identifier")
    return value


def _kind(value: RecordKind | str) -> RecordKind:
    try:
        result = value if isinstance(value, RecordKind) else RecordKind(value)
    except (TypeError, ValueError) as exc:
        raise HistoryError("invalid history event kind") from exc
    if result not in _HISTORY_KINDS:
        raise HistoryError("history kind is not permitted")
    return result


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """One immutable history event.

    Hash fields bind external evidence without embedding sensitive raw data.
    ``event_id`` is content-addressed when omitted, making retries naturally
    idempotent across processes.
    """

    event_id: str = ""
    kind: RecordKind | str = RecordKind.SYSTEM
    session_id: str = ""
    fingerprint_hash: str = ""
    dataset_hash: str = ""
    candidate_id: str = ""
    candidate_hash: str = ""
    code_hash: str = ""
    evidence_hash: str = ""
    state: str = ""
    reason: str = ""
    uncertainty: float | None = None
    ood: bool = False
    profile_hash: str = ""
    rollback: bool = False
    payload: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    created_at: str = ""

    def __post_init__(self) -> None:
        kind = _kind(self.kind)
        object.__setattr__(self, "kind", kind)
        values = {
            "session_id": _identifier(self.session_id, "session_id"),
            "candidate_id": _identifier(self.candidate_id, "candidate_id"),
            "state": _identifier(self.state, "state"),
            "reason": self.reason,
        }
        for name in ("reason",):
            if not isinstance(values[name], str) or len(values[name]) > 1024:
                raise HistoryError(f"{name} must be a bounded string")
            if values[name] and not _REASON.fullmatch(values[name]):
                raise HistoryError("reason must be a bounded safe code")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        for name in ("fingerprint_hash", "dataset_hash", "candidate_hash", "code_hash", "evidence_hash", "profile_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        if not isinstance(self.ood, bool) or not isinstance(self.rollback, bool):
            raise HistoryError("ood and rollback must be booleans")
        if self.uncertainty is not None:
            if isinstance(self.uncertainty, bool) or not isinstance(self.uncertainty, (int, float)):
                raise HistoryError("uncertainty must be a finite number or empty")
            import math
            if not math.isfinite(float(self.uncertainty)) or not 0 <= float(self.uncertainty) <= 1:
                raise HistoryError("uncertainty must be between zero and one")
            object.__setattr__(self, "uncertainty", float(self.uncertainty))
        if not isinstance(self.payload, Mapping) or isinstance(self.payload, (str, bytes, bytearray)):
            raise HistoryError("payload must be a JSON object")
        cleaned = _redact(dict(self.payload))
        unknown_payload = set(cleaned) - _PAYLOAD_KEYS[kind]
        if unknown_payload:
            raise HistoryError("payload contains fields outside the event-kind allowlist")
        try:
            encoded = canonical_bytes(cleaned, max_bytes=MAX_EVENT_PAYLOAD_BYTES, max_depth=16, max_items=2048, max_string_bytes=32 * 1024)
        except (CanonicalJSONError, TypeError, ValueError) as exc:
            raise HistoryError("event payload is not bounded canonical JSON") from exc
        object.__setattr__(self, "payload", _freeze(cleaned))
        if not isinstance(self.created_at, str) or len(self.created_at) > 40:
            raise HistoryError("created_at must be a bounded string")
        if self.created_at:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", self.created_at):
                raise HistoryError("created_at must be RFC3339 UTC")
            try:
                parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
                if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                    raise ValueError
            except ValueError as exc:
                raise HistoryError("created_at must be RFC3339 UTC") from exc
        event_id = self.event_id
        if event_id:
            object.__setattr__(self, "event_id", _identifier(event_id, "event_id"))
        else:
            body = self._body_dict(include_event_id=False)
            object.__setattr__(self, "event_id", "event:" + sha256_hex(canonical_bytes(body, max_bytes=MAX_EVENT_PAYLOAD_BYTES))[:48])
        if len(encoded) > MAX_EVENT_PAYLOAD_BYTES:
            raise HistoryError("event payload exceeds size bound")

    def _body_dict(self, *, include_event_id: bool = True) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kind": self.kind.value,
            "session_id": self.session_id,
            "fingerprint_hash": self.fingerprint_hash,
            "dataset_hash": self.dataset_hash,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "code_hash": self.code_hash,
            "evidence_hash": self.evidence_hash,
            "state": self.state,
            "reason": self.reason,
            "uncertainty": self.uncertainty,
            "ood": self.ood,
            "profile_hash": self.profile_hash,
            "rollback": self.rollback,
            "payload": _thaw(self.payload),
            "created_at": self.created_at,
        }
        if include_event_id:
            body["event_id"] = self.event_id
        return body

    @property
    def event_hash(self) -> str:
        return sha256_hex(canonical_bytes(self._body_dict()))

    @property
    def record_id(self) -> str:
        return "history:" + self.event_id

    @property
    def payload_hash(self) -> str:
        return sha256_hex(canonical_bytes(_thaw(self.payload), max_bytes=MAX_EVENT_PAYLOAD_BYTES))

    @property
    def payload_bytes(self) -> bytes:
        return canonical_bytes(_thaw(self.payload), max_bytes=MAX_EVENT_PAYLOAD_BYTES)

    # Short aliases keep the public event vocabulary pleasant while the
    # serialized names remain explicit and schema-stable.
    @property
    def fingerprint(self) -> str:
        return self.fingerprint_hash

    @property
    def dataset(self) -> str:
        return self.dataset_hash

    @property
    def candidate(self) -> str:
        return self.candidate_id

    @property
    def code(self) -> str:
        return self.code_hash

    @property
    def evidence(self) -> str:
        return self.evidence_hash

    def as_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, **self._body_dict(include_event_id=False)}

    def to_record(self) -> OptimizationRecord:
        body = self.as_dict()
        return OptimizationRecord(
            record_id="history:" + self.event_id,
            kind=self.kind,
            quality=QualityClass.ENGINEERING,
            phase=HISTORY_KIND_PHASE[self.kind],
            payload=body,
            source_hash=self.evidence_hash or None,
            created_at=self.created_at,
        )

    as_record = to_record

    @classmethod
    def from_record(cls, row: Mapping[str, Any]) -> "SessionEvent":
        try:
            raw = row["payload"]
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            if isinstance(raw, (bytes, bytearray)):
                raw = loads_strict(bytes(raw), max_bytes=MAX_EVENT_PAYLOAD_BYTES)
            if not isinstance(raw, Mapping):
                raise HistoryIntegrityError("stored event payload is not an object")
            event = cls.from_dict(raw)
            try:
                stored_record_id = row["record_id"]
            except (KeyError, IndexError):
                stored_record_id = None
            if stored_record_id not in {"history:" + event.event_id, event.event_id}:
                raise HistoryIntegrityError("stored history record identity mismatch")
            return event
        except HistoryError:
            raise
        except (KeyError, TypeError, ValueError, CanonicalJSONError) as exc:
            raise HistoryIntegrityError("stored history row is malformed") from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SessionEvent":
        if not isinstance(value, Mapping):
            raise HistoryError("event must be an object")
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if set(value) - allowed:
            raise HistoryError("event contains unknown fields")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class HistoryWriter:
    """Append events to one ``OptimizationMemoryV2`` instance."""

    memory: OptimizationMemoryV2

    def __post_init__(self) -> None:
        if not isinstance(self.memory, OptimizationMemoryV2):
            raise TypeError("HistoryWriter requires OptimizationMemoryV2")

    def append(self, event: SessionEvent) -> SessionEvent:
        if not isinstance(event, SessionEvent):
            raise TypeError("append expects SessionEvent")
        self.memory.append(event.to_record())
        return event

    def append_many(self, events: Iterable[SessionEvent]) -> tuple[SessionEvent, ...]:
        values = tuple(events)
        if any(not isinstance(event, SessionEvent) for event in values):
            raise TypeError("append_many expects SessionEvent values")
        self.memory.append_many(event.to_record() for event in values)
        return values

    write = append
    write_many = append_many
    record = append


@dataclass(frozen=True, slots=True)
class HistoryReader:
    """Read immutable events through ``OptimizationMemoryV2.read_connection``."""

    memory: OptimizationMemoryV2 | ReadOnlyMemoryView

    def __post_init__(self) -> None:
        if not isinstance(self.memory, (OptimizationMemoryV2, ReadOnlyMemoryView)):
            raise TypeError("HistoryReader requires a memory read API")

    def recent(self, limit: int = 100, *, session_id: str | None = None) -> tuple[SessionEvent, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 10_000:
            raise ValueError("limit must be between 0 and 10000")
        if not self.memory.verify_chain():
            raise HistoryIntegrityError("optimization memory hash chain is invalid")
        # OptimizationMemory also stores imported BENCHMARK records.  The
        # event envelope marker prevents those source records being decoded as
        # SessionEvent values.
        clauses = ["kind IN (?, ?, ?, ?, ?, ?)", "json_extract(payload, '$.event_id') IS NOT NULL"]
        values: list[Any] = [item.value for item in sorted(_HISTORY_KINDS, key=lambda item: item.value)]
        if session_id is not None:
            session_id = _identifier(session_id, "session_id")
            clauses.append("json_extract(payload, '$.session_id') = ?")
            values.append(session_id)
        with self.memory.read_connection() as conn:
            rows = conn.execute(
                "SELECT record_id, payload FROM optimization_records WHERE " + " AND ".join(clauses) + " ORDER BY seq DESC LIMIT ?",
                (*values, limit),
            ).fetchall()
        return tuple(SessionEvent.from_record(row) for row in rows)

    def all(self, *, limit: int = 10_000) -> tuple[SessionEvent, ...]:
        return self.recent(limit)

    read_recent = recent
    read = recent


__all__ = [
    "MAX_EVENT_PAYLOAD_BYTES",
    "HistoryError",
    "HistoryIntegrityError",
    "SessionEvent",
    "HistoryWriter",
    "HistoryReader",
    "HISTORY_KIND_PHASE",
]
