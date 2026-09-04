"""Transactional bridge from corpus records into Optimization Memory v2."""

from __future__ import annotations

import hashlib
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes
from .corpus import EVIDENCE_CONTRACTS, CorpusInventory, NormalizedRecord, QualityClass, EvidenceContract
from .memory import OptimizationMemoryV2
from .records import DataPhase, OptimizationRecord, QualityClass as MemoryQuality, RecordKind


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_VALUE_BYTES = 64 * 1024


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(child) for child in value]
    return value


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            result.extend(_flatten(value[key], child))
        return result
    return [(prefix, value)]


def _compact(value: Any) -> Any:
    """Return a bounded JSON value, hashing oversized arrays/objects."""

    plain = _thaw(value)
    try:
        encoded = canonical_bytes(plain, max_bytes=_MAX_VALUE_BYTES, max_items=100_000)
    except ValueError:
        encoded = canonical_bytes(plain, max_bytes=64 * 1024 * 1024, max_items=2_000_000)
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
    return plain


def _selected(record: NormalizedRecord, fields: Sequence[str]) -> dict[str, Any]:
    flattened = dict(_flatten(record.data))
    return {
        field: _compact(flattened[field])
        for field in sorted(set(fields))
        if field in flattened
    }


def _safe_source_hash(record: NormalizedRecord) -> str | None:
    if _HASH_RE.fullmatch(record.source_sha256 or ""):
        return record.source_sha256
    return None


def _memory_quality(record: NormalizedRecord) -> MemoryQuality:
    return MemoryQuality(record.quality.value)


def _validated_contract(record: NormalizedRecord) -> EvidenceContract | None:
    if not _HASH_RE.fullmatch(record.source_sha256 or "") or record.source_sha256 != (record.source_sha256 or "").lower():
        if record.training_eligible or record.contract_id is not None:
            raise ValueError("source_sha256 must be lowercase SHA-256")
    if record.contract_id is None:
        if record.contract_verified or record.training_eligible:
            raise ValueError("eligible/self-declared record lacks a registered EvidenceContract")
        return None
    contract = next((item for item in EVIDENCE_CONTRACTS if item.contract_id == record.contract_id), None)
    if contract is None:
        raise ValueError(f"unregistered EvidenceContract: {record.contract_id}")
    if record.contract_version != contract.version or record.contract_hash != contract.contract_hash:
        raise ValueError("EvidenceContract version/hash mismatch")
    if dict(record.contract_bound_constants) != dict(contract.bound_constants):
        raise ValueError("EvidenceContract bound constants mismatch")
    if record.logical_source_file != contract.source_basename:
        raise ValueError("EvidenceContract logical source file mismatch")
    source_name = Path(record.source_path.split("#", 1)[0]).name
    if source_name != contract.source_basename and not re.fullmatch(r"[0-9a-f]{64}-" + re.escape(contract.source_basename), source_name):
        raise ValueError("EvidenceContract source path mismatch")
    flattened = dict(_flatten(record.data))
    expected_features = {path for path in flattened if any(fnmatch.fnmatchcase(path, pattern) for pattern in contract.feature_paths)}
    expected_labels = {path for path in flattened if any(fnmatch.fnmatchcase(path, pattern) for pattern in contract.label_paths)}
    if set(record.feature_fields) != expected_features or set(record.label_fields) != expected_labels:
        raise ValueError("EvidenceContract feature/label field mismatch")
    if not record.identity_contract_valid or not contract.identity_satisfied(record.data):
        raise ValueError("EvidenceContract identity requirements are incomplete")
    if not record.source_integrity_verified or not record.archive_manifest_verified or not record.identity_complete:
        raise ValueError("source integrity/manifest/identity gate failed")
    return contract


def _memory_id(kind: str, record: NormalizedRecord) -> str:
    digest = hashlib.sha256(
        canonical_bytes(
            {"kind": kind, "record_id": record.record_id},
            max_bytes=4096,
        )
    ).hexdigest()
    return f"bridge:{kind}:{digest}"


def normalized_to_memory_records(record: NormalizedRecord) -> tuple[OptimizationRecord, ...]:
    """Convert one source into compact import/features/label records."""

    contract = _validated_contract(record)
    source_hash = _safe_source_hash(record)
    quality = _memory_quality(record)
    provenance = {
        "source_path": record.source_path,
        "source_kind": record.source_kind,
        "source_sha256": record.source_sha256,
        "source_verified": bool(record.source_verified),
        "manifest_verified": record.manifest_verified,
        "content_fingerprint": record.content_fingerprint,
        "quality": record.quality.value,
        "feature_fields": list(record.feature_fields),
        "label_fields": list(record.label_fields),
        "missing_fields": list(record.missing_fields),
        "censored_fields": list(record.censored_fields),
        "training_eligible": record.training_eligible,
        "contract_id": record.contract_id,
        "contract_version": record.contract_version,
        "contract_hash": record.contract_hash,
        "contract_bound_constants": dict(record.contract_bound_constants),
    }
    result: list[OptimizationRecord] = [
        OptimizationRecord(
            record_id=_memory_id("import", record),
            kind=RecordKind.IMPORT,
            quality=quality,
            phase=DataPhase.FEATURE,
            payload={"bridge_version": 1, "record": provenance},
            source_hash=source_hash,
            created_at="",
        )
    ]
    if not record.training_eligible:
        return tuple(result)
    identity = {
        "study_id": record.study_id,
        "run_id": record.run_id,
        "observed_time": record.observed_time,
        "hardware_fingerprint": record.hardware_fingerprint,
        "model_fingerprint": record.model_fingerprint,
        "workload_fingerprint": record.workload_fingerprint,
        "prompt_family": record.prompt_family,
    }
    common = {
        "bridge_version": 1,
        "source_record_id": record.record_id,
        "content_fingerprint": record.content_fingerprint,
        "identity": identity,
        "contract_id": record.contract_id,
        "contract_version": record.contract_version,
        "contract_hash": record.contract_hash,
    }
    result.extend(
        (
            OptimizationRecord(
                record_id=_memory_id("environment", record),
                kind=RecordKind.ENVIRONMENT,
                quality=quality,
                phase=DataPhase.FEATURE,
                payload={**common, "environment": {key: identity[key] for key in ("hardware_fingerprint", "model_fingerprint", "observed_time")}},
                source_hash=source_hash,
                created_at="",
            ),
            OptimizationRecord(
                record_id=_memory_id("workload", record),
                kind=RecordKind.WORKLOAD,
                quality=quality,
                phase=DataPhase.FEATURE,
                payload={**common, "workload": {key: identity[key] for key in ("workload_fingerprint", "prompt_family", "study_id")}},
                source_hash=source_hash,
                created_at="",
            ),
            OptimizationRecord(
                record_id=_memory_id("candidate", record),
                kind=RecordKind.CANDIDATE,
                quality=quality,
                phase=DataPhase.FEATURE,
                payload={**common, "candidate": _selected(record, record.feature_fields)},
                source_hash=source_hash,
                created_at="",
            ),
            OptimizationRecord(
                record_id=_memory_id("label", record),
                kind=RecordKind.BENCHMARK,
                quality=quality,
                phase=DataPhase.LABEL,
                payload={**common, "labels": _selected(record, record.label_fields)},
                source_hash=source_hash,
                created_at="",
            ),
        )
    )
    return tuple(result)


@dataclass
class MemoryRecordSink:
    """A fully buffering, deterministic, one-transaction memory sink."""

    memory: OptimizationMemoryV2
    last_records: tuple[OptimizationRecord, ...] = ()
    last_snapshot_hash: str | None = None

    def accept_many(self, records: Sequence[NormalizedRecord]) -> None:
        ordered = tuple(
            sorted(
                records,
                key=lambda record: (record.source_path, record.content_fingerprint, record.record_id),
            )
        )
        converted = tuple(item for record in ordered for item in normalized_to_memory_records(record))
        # Every validation and conversion above happens before append_many.  A
        # memory implementation guarantees that append_many is one transaction.
        self.memory.append_many(converted)
        self.last_records = converted
        self.last_snapshot_hash = hashlib.sha256(
            canonical_bytes(
                [record.as_dict() for record in converted],
                max_bytes=64 * 1024 * 1024,
                max_items=2_000_000,
            )
        ).hexdigest()


def import_inventory(inventory: CorpusInventory, memory: OptimizationMemoryV2) -> MemoryRecordSink:
    """Import an already completed inventory through the transactional sink."""

    sink = MemoryRecordSink(memory)
    sink.accept_many(inventory.records)
    return sink


__all__ = [
    "MemoryRecordSink",
    "import_inventory",
    "normalized_to_memory_records",
]
