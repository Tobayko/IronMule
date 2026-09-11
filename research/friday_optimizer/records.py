"""Typed records accepted by Optimization Memory v2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import canonical_bytes, sha256_hex

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_RFC3339_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)$"
)


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class RecordKind(_StringEnum):
    ENVIRONMENT = "environment"
    WORKLOAD = "workload"
    CANDIDATE = "candidate"
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    BENCHMARK = "benchmark"
    PROFILE = "profile"
    PROMOTION = "promotion"
    SYSTEM = "system"
    IMPORT = "import"
    DATASET = "dataset"


class QualityClass(_StringEnum):
    FORMAL = "formal"
    ENGINEERING = "engineering"
    EXPLORATORY = "exploratory"
    LEGACY_SUMMARY = "legacy_summary"
    INVALID = "invalid"
    QUARANTINED = "quarantined"


class DataPhase(_StringEnum):
    FEATURE = "feature"
    LABEL = "label"


# More explicit name for external clients and hidden consumers.
FeatureLabelPhase = DataPhase


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON containers to close mutation/TOCTOU gaps."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _enum_value(value: str | Enum, enum_type: type[Enum], field: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or {enum_type.__name__}")
    try:
        return enum_type(value).value
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"invalid {field} {value!r}; expected one of {allowed}") from exc


@dataclass(frozen=True, slots=True)
class OptimizationRecord:
    """Immutable, typed input record.

    ``payload`` contains only the evidence body.  It is serialized and hashed
    before insertion; generated chain metadata never mutates the payload.
    """

    record_id: str
    kind: RecordKind
    quality: QualityClass
    phase: DataPhase
    payload: Mapping[str, Any]
    source_hash: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not _ID_RE.fullmatch(self.record_id):
            raise ValueError("record_id must be a bounded non-empty safe identifier")
        kind = _enum_value(self.kind, RecordKind, "kind")
        quality = _enum_value(self.quality, QualityClass, "quality")
        phase = _enum_value(self.phase, DataPhase, "phase")
        object.__setattr__(self, "kind", RecordKind(kind))
        object.__setattr__(self, "quality", QualityClass(quality))
        object.__setattr__(self, "phase", DataPhase(phase))
        if not isinstance(self.payload, Mapping) or isinstance(self.payload, (str, bytes)):
            raise TypeError("payload must be a JSON object mapping")
        # Force validation now, before any transaction starts.
        frozen_payload = _freeze(dict(self.payload))
        payload_bytes = canonical_bytes(frozen_payload)
        object.__setattr__(self, "payload", frozen_payload)
        if self.source_hash is not None:
            if not isinstance(self.source_hash, str) or not _HASH_RE.fullmatch(self.source_hash):
                raise ValueError("source_hash must be a lowercase SHA-256 hex digest")
        if self.created_at is not None:
            if not isinstance(self.created_at, str) or len(self.created_at) > 40:
                raise TypeError("created_at must be a bounded RFC3339 UTC string or empty")
            if self.created_at and not _RFC3339_UTC_RE.fullmatch(self.created_at):
                raise ValueError("created_at must be RFC3339 UTC (ending in Z) or empty")
            if self.created_at:
                try:
                    parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("created_at is not a valid UTC timestamp") from exc
                if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
                    raise ValueError("created_at must carry a UTC offset")
        if self.created_at is None:
            object.__setattr__(self, "created_at", "")
        # Keep this useful assertion close to construction so a custom Mapping
        # cannot change between validation and insertion.
        if sha256_hex(payload_bytes) != self.payload_hash:
            raise AssertionError("payload changed during record construction")

    @property
    def payload_bytes(self) -> bytes:
        """Canonical UTF-8 payload bytes."""

        return canonical_bytes(self.payload)

    @property
    def payload_hash(self) -> str:
        """SHA-256 hash of :attr:`payload_bytes`."""

        return sha256_hex(self.payload_bytes)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record input (without chain metadata)."""

        result: dict[str, Any] = {
            "record_id": self.record_id,
            "kind": self.kind.value,
            "quality": self.quality.value,
            "phase": self.phase.value,
            "payload": _thaw(self.payload),
        }
        if self.source_hash is not None:
            result["source_hash"] = self.source_hash
        if self.created_at is not None:
            result["created_at"] = self.created_at
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizationRecord":
        """Construct a record from a strict, JSON-compatible mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("record must be an object mapping")
        try:
            return cls(
                record_id=value["record_id"],
                kind=value["kind"],
                quality=value["quality"],
                phase=value["phase"],
                payload=value["payload"],
                source_hash=value.get("source_hash"),
                created_at=value.get("created_at"),
            )
        except KeyError as exc:
            raise ValueError(f"record missing required field: {exc.args[0]}") from exc


Record = OptimizationRecord
