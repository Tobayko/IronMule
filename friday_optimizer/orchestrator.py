"""Offline control-plane facade for Friday's Gemma optimizer.

This module coordinates existing, deterministic components.  It deliberately
does not probe hardware, launch a model, activate a profile, download anything,
or mutate a source corpus.  The only writes are explicit memory/history writes
requested by the caller; ordinary inspection and shadow evaluation are
read-only.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .candidates import CandidateRegistry
from .canonical import canonical_bytes, sha256_hex
from .adapters import DEFAULT_DISCOVERY_ROOTS
from .corpus import CorpusAuditor, CorpusInventory, NormalizedRecord
from .dataset import DatasetBuilder, DatasetSnapshot
from .evaluator import (
    CorrectnessResult,
    Evaluator,
    MetricSample,
    ResourceResult,
    ShadowDecision,
)
from .fingerprint import ExactFingerprint
from .history import HistoryReader, HistoryWriter, SessionEvent
from .memory import OptimizationMemoryV2, ReadOnlyMemoryView
from .profiles import AtomicProfileStore, ProfileError, ProfileMode
from .readiness import HardwareLease, ReadinessGate
from .records import RecordKind
from .record_bridge import import_inventory as bridge_import_inventory
from .session import SessionController, StageSpec, SubprocessStageRunner


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


def _hash(value: str | None, name: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash or empty")
    return value


def _reason_code(value: str) -> str:
    """Turn untrusted evaluator prose into a bounded, non-path history code."""
    import re
    text = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(value)).lower().strip("_")
    if not text:
        return "unknown"
    if len(text) > 96:
        return "reason_" + sha256_hex(text)[:16]
    return text


def _text_digest(value: Any) -> str:
    """Digest text fields without placing paths or prose in the identity."""
    return hashlib.sha256(str(value or "").encode("utf-8", "surrogatepass")).hexdigest()


def _sequence_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        item = str(value).encode("utf-8", "surrogatepass")
        digest.update(len(item).to_bytes(4, "big"))
        digest.update(item)
    return digest.hexdigest()


def _value_digest(value: Any) -> str:
    try:
        return sha256_hex(canonical_bytes(value, max_bytes=64 * 1024, max_items=4096, max_depth=16))
    except (TypeError, ValueError):
        return _text_digest(value)


def _inventory_identity_hash(inventory: CorpusInventory) -> str:
    """Return a bounded, payload-free identity for one corpus inventory.

    This intentionally streams one compact canonical item at a time.  The
    source payloads and absolute paths never enter the digest material; their
    identities are represented by hashes and bounded metadata instead.
    """
    if not isinstance(inventory, CorpusInventory):
        raise TypeError("inventory must be CorpusInventory")
    if len(inventory.records) > 100_000 or len(inventory.files) > 100_000 or len(inventory.issues) > 100_000:
        raise ValueError("inventory identity exceeds its bounded row limit")
    digest = hashlib.sha256()

    def feed(value: Mapping[str, Any]) -> None:
        encoded = canonical_bytes(value, max_bytes=16 * 1024, max_items=512, max_string_bytes=1024)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)

    feed({
        "schema": "friday.inventory-identity.v1",
        "files": len(inventory.files),
        "records": len(inventory.records),
        "issues": len(inventory.issues),
        "quality_counts": inventory.quality_counts,
        "exclusions": inventory.exclusions,
        "duplicates": inventory.duplicate_count,
    })
    for row in inventory.files:
        feed({
            "kind": "file",
            "relative_path": _text_digest(row.relative_path),
            "root_name": _text_digest(row.root_name),
            "file_kind": row.kind,
            "size": row.size_bytes,
            "sha256": row.sha256,
            "dev": row.st_dev,
            "ino": row.st_ino,
            "read_error": _text_digest(row.read_error),
            "excluded": row.excluded_reason,
        })
    for record in sorted(inventory.records, key=lambda item: (item.record_id, item.content_fingerprint, item.source_path)):
        feed({
            "kind": "record",
            "record_id": _text_digest(record.record_id),
            "source_path": _text_digest(record.source_path),
            "source_kind": record.source_kind,
            "data_hash": _value_digest(record.data),
            "content_fingerprint": record.content_fingerprint,
            "source_sha256": record.source_sha256,
            "source_fingerprint": record.source_fingerprint,
            "quality": record.quality.value,
            "features": _sequence_digest(record.feature_fields),
            "labels": _sequence_digest(record.label_fields),
            "missing": _sequence_digest(record.missing_fields),
            "censored": _sequence_digest(record.censored_fields),
            "duplicate_of": _text_digest(record.duplicate_of),
            "dirty": record.dirty,
            "contract_id": record.contract_id or "",
            "contract_version": record.contract_version,
            "contract_hash": record.contract_hash or "",
            "contract_constants": _value_digest(record.contract_bound_constants),
            "identity_flags": {
                "source_verified": record.source_verified,
                "manifest_verified": record.manifest_verified,
                "contract_verified": record.contract_verified,
                "source_integrity_verified": record.source_integrity_verified,
                "archive_manifest_verified": record.archive_manifest_verified,
                "identity_contract_valid": record.identity_contract_valid,
                "identity_complete": record.identity_complete,
            },
            "logical_source_hash": _text_digest(record.logical_source_file),
            "study": _text_digest(record.study_id),
            "run": _text_digest(record.run_id),
            "observed": _text_digest(record.observed_time),
            "hardware": _text_digest(record.hardware_fingerprint),
            "model": _text_digest(record.model_fingerprint),
            "workload": _text_digest(record.workload_fingerprint),
            "prompt_family": _text_digest(record.prompt_family),
        })
    for issue in sorted(inventory.issues, key=lambda item: (item.path, item.code, item.detail)):
        feed({"kind": "issue", "path": _text_digest(issue.path), "code": issue.code, "detail": _text_digest(issue.detail), "terminal": issue.terminal})
    if inventory.manifest is None:
        feed({"kind": "manifest", "present": False})
    else:
        manifest = inventory.manifest
        feed({
            "kind": "manifest",
            "present": True,
            "terminal": manifest.terminal,
            "ok": manifest.ok,
            "missing": _sequence_digest(manifest.missing),
            "hash_mismatches": _sequence_digest(manifest.hash_mismatches),
            "byte_mismatches": _sequence_digest(manifest.byte_mismatches),
            "path_violations": _sequence_digest(manifest.path_violations),
            "invalid_entries": _sequence_digest(manifest.invalid_entries),
            "duplicate_entries": manifest.duplicate_entries,
            "verified_paths": _sequence_digest(manifest.verified_paths),
            "logical_by_stored": _sequence_digest(f"{key}={value}" for key, value in sorted(manifest.logical_by_stored.items())),
        })
    for schema in sorted(inventory.sqlite, key=lambda item: str(item.get("path", ""))):
        tables = schema.get("tables", ())
        compact_tables = []
        for table in tables if isinstance(tables, (list, tuple)) else ():
            if isinstance(table, Mapping):
                compact_tables.append({"name": _text_digest(table.get("name")), "columns": tuple(table.get("columns", ())), "row_count": table.get("row_count"), "count_error": _text_digest(table.get("count_error"))})
        feed({"kind": "sqlite", "path": _text_digest(schema.get("path")), "tables": tuple(sorted(compact_tables, key=lambda item: item["name"])), "identity": _value_digest(schema.get("identity"))})
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Immutable filesystem configuration.

    Paths are resolved once and never created by the constructor.  This keeps
    ``doctor`` and ``status`` genuinely read-only, even on a fresh checkout.
    """

    root: str | os.PathLike[str]
    memory_path: str | os.PathLike[str] | None = None
    profile_path: str | os.PathLike[str] | None = None
    history_limit: int = 100

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError("optimizer root must be an existing directory")
        memory = Path(self.memory_path).expanduser() if self.memory_path is not None else root / ".friday-data" / "optimizer-memory.sqlite3"
        profile = Path(self.profile_path).expanduser() if self.profile_path is not None else root / ".friday-data" / "optimizer-profiles.json"
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "memory_path", memory.resolve())
        object.__setattr__(self, "profile_path", profile.resolve())
        if isinstance(self.history_limit, bool) or not isinstance(self.history_limit, int) or not 1 <= self.history_limit <= 10_000:
            raise ValueError("history_limit must be between 1 and 10000")

    @property
    def project_root(self) -> Path:
        return self.root  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {"root": str(self.root), "memory_path": str(self.memory_path), "profile_path": str(self.profile_path), "history_limit": self.history_limit}


OrchestratorConfig = OptimizerConfig


@dataclass(frozen=True, slots=True)
class DoctorReport:
    ok: bool
    root: str
    paths: Mapping[str, str] = field(default_factory=dict)
    schemas: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None
    fingerprint_exact: bool | None = None
    readiness: Any = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool) or not isinstance(self.root, str):
            raise TypeError("invalid DoctorReport")
        object.__setattr__(self, "paths", MappingProxyType(dict(self.paths)))
        object.__setattr__(self, "schemas", MappingProxyType(_freeze(dict(self.schemas))))
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))

    @property
    def ready(self) -> bool:
        return self.ok and not self.reasons

    @property
    def path_ok(self) -> bool:
        return self.root != "" and Path(self.root).is_dir()

    @property
    def schema_ok(self) -> bool:
        memory = self.schemas.get("memory", {})
        return not memory.get("exists", False) or bool(memory.get("chain_ok", False))

    @property
    def memory_schema(self) -> Mapping[str, Any]:
        return self.schemas.get("memory", {})

    @property
    def profile_schema(self) -> Mapping[str, Any]:
        return self.schemas.get("profile", {})

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "root": self.root, "paths": dict(self.paths), "schemas": _thaw(self.schemas), "fingerprint": self.fingerprint, "fingerprint_exact": self.fingerprint_exact, "readiness": self.readiness, "reasons": list(self.reasons)}


@dataclass(frozen=True, slots=True)
class ImportReport:
    ok: bool
    source_root: str
    records_seen: int = 0
    records_written: int = 0
    records_idempotent: int = 0
    inventory_hash: str = ""
    sink_hash: str = ""
    memory_snapshot_hash: str = ""
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool) or not isinstance(self.source_root, str):
            raise TypeError("invalid ImportReport")
        object.__setattr__(self, "inventory_hash", _hash(self.inventory_hash, "inventory_hash"))
        object.__setattr__(self, "sink_hash", _hash(self.sink_hash, "sink_hash"))
        object.__setattr__(self, "memory_snapshot_hash", _hash(self.memory_snapshot_hash, "memory_snapshot_hash"))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.records_seen, self.records_written, self.records_idempotent)):
            raise TypeError("record counts must be non-negative integers")
        object.__setattr__(self, "issues", tuple(str(item) for item in self.issues))

    @property
    def imported(self) -> int:
        return self.records_written

    @property
    def records_imported(self) -> int:
        return self.records_written

    @property
    def idempotent(self) -> bool:
        return self.records_idempotent > 0 and self.records_written == 0

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "source_root": self.source_root, "records_seen": self.records_seen, "records_written": self.records_written, "records_idempotent": self.records_idempotent, "inventory_hash": self.inventory_hash, "sink_hash": self.sink_hash, "memory_snapshot_hash": self.memory_snapshot_hash, "issues": list(self.issues)}


@dataclass(frozen=True, slots=True)
class OptimizerStatus:
    ok: bool
    memory_exists: bool = False
    chain_ok: bool = False
    record_count: int = 0
    profile: Mapping[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    last_events: tuple[SessionEvent, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool) or not isinstance(self.memory_exists, bool) or not isinstance(self.chain_ok, bool):
            raise TypeError("invalid OptimizerStatus")
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count < 0:
            raise TypeError("record_count must be a non-negative integer")
        object.__setattr__(self, "profile", MappingProxyType(_freeze(dict(self.profile))))
        object.__setattr__(self, "last_events", tuple(self.last_events))
        object.__setattr__(self, "dataset_hash", _hash(self.dataset_hash, "dataset_hash"))

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        return self.last_events

    @property
    def memory_integrity(self) -> bool:
        return self.chain_ok

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "memory_exists": self.memory_exists, "chain_ok": self.chain_ok, "record_count": self.record_count, "profile": _thaw(self.profile), "dataset_hash": self.dataset_hash, "last_events": [event.as_dict() for event in self.last_events], "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ShadowRequest:
    """Raw, immutable evidence supplied to one offline shadow evaluation."""

    fingerprint: ExactFingerprint
    candidate_id: str
    baseline_samples: tuple[MetricSample, ...] = ()
    candidate_samples: tuple[MetricSample, ...] = ()
    aa_baseline_samples: tuple[MetricSample, ...] = ()
    aa_control_samples: tuple[MetricSample, ...] = ()
    aa_pairs: tuple[tuple[MetricSample, MetricSample], ...] = ()
    resources: tuple[ResourceResult, ...] = ()
    baseline_correctness: CorrectnessResult | None = None
    candidate_correctness: CorrectnessResult | None = None
    correctness: tuple[CorrectnessResult, CorrectnessResult] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    qualified: tuple[str, ...] = ()
    session_id: str = "shadow"
    dataset_hash: str = ""
    code_hash: str = ""
    write_history: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, ExactFingerprint):
            raise TypeError("shadow requests require ExactFingerprint")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id is required")
        if len(self.candidate_id) > 256 or not isinstance(self.session_id, str) or len(self.session_id) > 256:
            raise ValueError("shadow identifiers are too long")
        for name in ("baseline_samples", "candidate_samples", "aa_baseline_samples", "aa_control_samples", "resources"):
            values = tuple(getattr(self, name))
            expected = MetricSample if "samples" in name else ResourceResult
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(f"{name} contains an invalid evidence value")
            object.__setattr__(self, name, values)
        pairs = tuple(self.aa_pairs)
        if any(len(pair) != 2 or any(not isinstance(item, MetricSample) for item in pair) for pair in pairs):
            raise TypeError("aa_pairs contains an invalid evidence value")
        object.__setattr__(self, "aa_pairs", tuple((pair[0], pair[1]) for pair in pairs))
        if self.correctness is not None:
            if len(self.correctness) != 2 or any(not isinstance(item, CorrectnessResult) for item in self.correctness):
                raise TypeError("correctness must contain two CorrectnessResult values")
            object.__setattr__(self, "correctness", tuple(self.correctness))
        for name in ("baseline_correctness", "candidate_correctness"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, CorrectnessResult):
                raise TypeError(f"{name} is invalid")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        if any(not isinstance(item, str) or not item or len(item) > 256 for item in self.qualified):
            raise TypeError("qualified contains an invalid candidate id")
        object.__setattr__(self, "parameters", _freeze(dict(self.parameters)))
        object.__setattr__(self, "qualified", tuple(self.qualified))
        if not self.dataset_hash or not self.code_hash:
            raise ValueError("dataset_hash and code_hash are mandatory for shadow evaluation")
        object.__setattr__(self, "dataset_hash", _hash(self.dataset_hash, "dataset_hash"))
        object.__setattr__(self, "code_hash", _hash(self.code_hash, "code_hash"))
        if not isinstance(self.write_history, bool):
            raise TypeError("write_history must be bool")

    @property
    def fingerprint_hash(self) -> str:
        return self.fingerprint.fingerprint_hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint.as_dict(),
            "candidate_id": self.candidate_id,
            "baseline_samples": [s.as_dict() for s in self.baseline_samples],
            "candidate_samples": [s.as_dict() for s in self.candidate_samples],
            "aa_baseline_samples": [s.as_dict() for s in self.aa_baseline_samples],
            "aa_control_samples": [s.as_dict() for s in self.aa_control_samples],
            "aa_pairs": [[a.as_dict(), b.as_dict()] for a, b in self.aa_pairs],
            "resources": [r.as_dict() for r in self.resources],
            "baseline_correctness": None if self.baseline_correctness is None else self.baseline_correctness.as_dict(),
            "candidate_correctness": None if self.candidate_correctness is None else self.candidate_correctness.as_dict(),
            "correctness": None if self.correctness is None else [item.as_dict() for item in self.correctness],
            "parameters": _thaw(self.parameters),
            "qualified": list(self.qualified),
            "dataset_hash": self.dataset_hash,
            "code_hash": self.code_hash,
            "session_id": self.session_id,
            "write_history": self.write_history,
        }

    to_dict = as_dict

    @property
    def request_hash(self) -> str:
        return sha256_hex(canonical_bytes(self.as_dict(), max_bytes=4 * 1024 * 1024, max_items=100_000))

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.as_dict(), max_bytes=4 * 1024 * 1024, max_items=100_000)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShadowRequest":
        if not isinstance(value, Mapping):
            raise TypeError("shadow request must be an object")
        allowed = {"fingerprint", "candidate_id", "baseline_samples", "candidate_samples", "aa_baseline_samples", "aa_control_samples", "aa_pairs", "resources", "baseline_correctness", "candidate_correctness", "correctness", "parameters", "qualified", "session_id", "dataset_hash", "code_hash", "write_history"}
        if set(value) - allowed:
            raise ValueError("shadow request contains unknown fields")
        def samples(name: str) -> tuple[MetricSample, ...]:
            raw = value.get(name, ())
            if not isinstance(raw, (list, tuple)):
                raise TypeError(name + " must be a sequence")
            if any(not isinstance(item, Mapping) for item in raw):
                raise TypeError(name + " contains an invalid item")
            return tuple(MetricSample(**dict(item)) for item in raw)
        def correctness(name: str) -> CorrectnessResult | None:
            raw = value.get(name)
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise TypeError(name + " must be an object")
            return CorrectnessResult(**dict(raw))
        raw_pairs = value.get("aa_pairs", ())
        if not isinstance(raw_pairs, (list, tuple)):
            raise TypeError("aa_pairs must be a sequence")
        pairs = tuple((MetricSample(**dict(pair[0])), MetricSample(**dict(pair[1]))) for pair in raw_pairs if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(item, Mapping) for item in pair))
        if len(pairs) != len(raw_pairs):
            raise TypeError("aa_pairs contains an invalid item")
        raw_resources = value.get("resources", ())
        if not isinstance(raw_resources, (list, tuple)) or any(not isinstance(item, Mapping) for item in raw_resources):
            raise TypeError("resources contains an invalid item")
        raw_correctness = value.get("correctness")
        pair_correctness = None
        if raw_correctness is not None:
            if not isinstance(raw_correctness, (list, tuple)) or len(raw_correctness) != 2 or any(not isinstance(item, Mapping) for item in raw_correctness):
                raise TypeError("correctness must contain two objects")
            pair_correctness = (CorrectnessResult(**dict(raw_correctness[0])), CorrectnessResult(**dict(raw_correctness[1])))
        return cls(
            fingerprint=ExactFingerprint.from_mapping(value["fingerprint"]),
            candidate_id=value["candidate_id"],
            baseline_samples=samples("baseline_samples"),
            candidate_samples=samples("candidate_samples"),
            aa_baseline_samples=samples("aa_baseline_samples"),
            aa_control_samples=samples("aa_control_samples"),
            aa_pairs=pairs,
            resources=tuple(ResourceResult(**dict(item)) for item in raw_resources),
            baseline_correctness=correctness("baseline_correctness"),
            candidate_correctness=correctness("candidate_correctness"),
            correctness=pair_correctness,
            parameters=value.get("parameters", {}),
            qualified=tuple(value.get("qualified", ())),
            session_id=value.get("session_id", "shadow"),
            dataset_hash=value.get("dataset_hash", ""),
            code_hash=value.get("code_hash", ""),
            write_history=value.get("write_history", False),
        )


def _read_memory(path: Path) -> tuple[bool, bool, int, str]:
    """Inspect an existing memory without opening it as a writer."""

    if not path.exists():
        return False, False, 0, "memory_missing"
    try:
        view = OptimizationMemoryV2.open_read_only(path)
        try:
            integrity = view.integrity()
            return True, integrity.ok, integrity.rows, "" if integrity.ok else (integrity.error or "memory_integrity_failed")
        finally:
            view.close()
    except Exception as exc:
        return True, False, 0, "memory_unreadable:" + type(exc).__name__


class OptimizerOrchestrator:
    """Read-only-by-default facade over corpus, dataset, evaluator and memory."""

    __slots__ = ("config", "memory", "registry", "evaluator", "_last_dataset")

    def __init__(self, config: OptimizerConfig | None = None, *, root: str | os.PathLike[str] | None = None, memory: OptimizationMemoryV2 | None = None, evaluator: Evaluator | None = None, registry: CandidateRegistry | None = None) -> None:
        if config is None:
            if root is None:
                raise TypeError("config or root is required")
            config = OptimizerConfig(root)
        if not isinstance(config, OptimizerConfig):
            raise TypeError("config must be OptimizerConfig")
        if memory is not None and not isinstance(memory, OptimizationMemoryV2):
            raise TypeError("memory must be OptimizationMemoryV2")
        self.config = config
        self.memory = memory
        if registry is not None and not isinstance(registry, CandidateRegistry):
            raise TypeError("registry must be CandidateRegistry")
        if registry is None and evaluator is not None:
            registry = getattr(evaluator, "registry", None)
            if not isinstance(registry, CandidateRegistry):
                raise TypeError("evaluator must expose CandidateRegistry")
        self.registry = registry or CandidateRegistry()
        if evaluator is None:
            self.evaluator = Evaluator(registry=self.registry)
        else:
            if not isinstance(evaluator, Evaluator):
                raise TypeError("evaluator must be Evaluator")
            if evaluator.registry is not self.registry:
                raise ValueError("orchestrator and evaluator must share one CandidateRegistry")
            self.evaluator = evaluator
        self._last_dataset: DatasetSnapshot | None = None

    def _open_memory(self) -> tuple[OptimizationMemoryV2, bool]:
        if self.memory is not None:
            return self.memory, False
        parent = Path(self.config.memory_path).parent
        if not parent.exists():
            # Directory creation is part of an explicit history/import write,
            # never of doctor/status construction.  Refuse a symlinked parent.
            parent.mkdir(parents=True, exist_ok=True)
        return OptimizationMemoryV2(self.config.memory_path), True  # type: ignore[arg-type]

    def _history(self, *, write: bool = False) -> tuple[HistoryWriter | HistoryReader, Any | None]:
        if write:
            memory, owned = self._open_memory()
            return HistoryWriter(memory), memory if owned else None
        if self.memory is not None:
            return HistoryReader(self.memory), None
        view = OptimizationMemoryV2.open_read_only(self.config.memory_path)
        return HistoryReader(view), view

    def doctor(self, *, fingerprint: ExactFingerprint | None = None, readiness: Any = None) -> DoctorReport:
        reasons: list[str] = []
        root = self.config.root
        if not root.is_dir():
            reasons.append("root_missing")
        memory_exists, chain_ok, _, memory_reason = _read_memory(Path(self.config.memory_path))
        if memory_exists and not chain_ok:
            reasons.append(memory_reason)
        profile_path = Path(self.config.profile_path)
        profile_status: dict[str, Any] = {"exists": profile_path.exists(), "valid": True}
        if profile_status["exists"] and profile_path.parent.is_dir():
            try:
                profile_status["value"] = AtomicProfileStore(self.config.profile_path).load()
            except (ProfileError, OSError) as exc:
                profile_status["valid"] = False
                reasons.append("profile_unreadable:" + type(exc).__name__)
        exact: bool | None = None
        fp_hash: str | None = None
        if fingerprint is not None:
            if not isinstance(fingerprint, ExactFingerprint):
                raise TypeError("fingerprint must be ExactFingerprint")
            exact = fingerprint.recommendation_allowed
            fp_hash = fingerprint.fingerprint_hash
            if not exact:
                reasons.append(fingerprint.ood_reason or "fingerprint_ood")
        readiness_value = None
        if readiness is not None:
            readiness_value = bool(getattr(readiness, "ready", readiness))
            if not readiness_value:
                reasons.append("readiness_not_ready")
        return DoctorReport(not reasons, str(root), {"memory": str(self.config.memory_path), "profile": str(self.config.profile_path)}, {"memory": {"exists": memory_exists, "chain_ok": chain_ok, "reason": memory_reason}, "profile": profile_status}, fp_hash, exact, readiness_value, tuple(dict.fromkeys(reasons)))

    def audit(self, root: str | os.PathLike[str] | None = None, *, roots: Sequence[str] | None = None) -> CorpusInventory:
        selected = Path(root).expanduser().resolve() if root is not None else self.config.root
        inventory = CorpusAuditor(selected, roots=tuple(roots) if roots is not None else DEFAULT_DISCOVERY_ROOTS).audit()
        # The optimizer's own memory/profile are control-plane state, not
        # evidence.  Excluding them keeps repeated audits stable after an
        # explicit import and prevents self-import recursion.
        internal: set[str] = set()
        for path in (Path(self.config.memory_path), Path(self.config.profile_path)):
            try:
                internal.add(path.resolve().relative_to(selected).as_posix())
            except ValueError:
                pass
        if internal:
            files = tuple(file for file in inventory.files if file.relative_path not in internal)
            records = tuple(record for record in inventory.records if record.source_path.split("#", 1)[0] not in internal)
            sqlite = tuple(item for item in inventory.sqlite if str(item.get("path", "")) not in internal)
            if len(files) != len(inventory.files) or len(records) != len(inventory.records) or len(sqlite) != len(inventory.sqlite):
                inventory = replace(inventory, files=files, records=records, sqlite=sqlite)
        return inventory

    def import_inventory(self, inventory: CorpusInventory, *, memory: OptimizationMemoryV2 | None = None) -> ImportReport:
        if not isinstance(inventory, CorpusInventory):
            raise TypeError("inventory must be CorpusInventory")
        target = memory or self.memory
        owned = False
        if target is None:
            parent = Path(self.config.memory_path).parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            target = OptimizationMemoryV2(self.config.memory_path)  # type: ignore[arg-type]
            owned = True
        before = target.integrity().rows
        inventory_hash = _inventory_identity_hash(inventory)
        try:
            sink = bridge_import_inventory(inventory, target)
            after = target.integrity().rows
            written = max(0, after - before)
            total = len(sink.last_records)
            sink_hash = sink.last_snapshot_hash or sha256_hex(canonical_bytes([], max_bytes=64 * 1024 * 1024))
            # The event is content-addressed by the source/sink hashes. Keep
            # its body stable across retries; the report still exposes each
            # call's new/idempotent counts.
            import_event = SessionEvent(event_id="import:" + sha256_hex(canonical_bytes({"inventory": inventory_hash, "sink": sink_hash}))[:48], kind=RecordKind.IMPORT, session_id="import", evidence_hash=sink_hash, state="imported", payload={"inventory_hash": inventory_hash, "sink_hash": sink_hash, "records_seen": len(inventory.records), "records_written": total, "records_idempotent": 0})
            HistoryWriter(target).append(import_event)
            report = ImportReport(True, inventory.root, len(inventory.records), written, max(0, total - written), inventory_hash, sink_hash, target.snapshot_hash())
            return report
        finally:
            if owned:
                target.close()

    def build_dataset(self, inventory: CorpusInventory | Iterable[NormalizedRecord], *, assignments: Mapping[str, str] | None = None, write_history: bool = False, session_id: str = "dataset") -> DatasetSnapshot:
        snapshot = DatasetBuilder(inventory).build(assignments=assignments)
        self._last_dataset = snapshot
        if write_history:
            event = SessionEvent(kind=RecordKind.DATASET, session_id=session_id, dataset_hash=snapshot.sha256, state="built", payload={"records": snapshot.card["coverage"], "card": snapshot.card.as_dict()})
            writer, owned = self._history(write=True)
            try:
                assert isinstance(writer, HistoryWriter)
                writer.append(event)
            finally:
                if owned is not None:
                    owned.close()
        return snapshot

    def status(self, *, fingerprint: ExactFingerprint | None = None, limit: int | None = None) -> OptimizerStatus:
        path = Path(self.config.memory_path)
        if self.memory is not None:
            integrity = self.memory.integrity()
            exists, chain_ok, count, reason = True, integrity.chain_ok, integrity.rows, integrity.error or ""
        else:
            exists, chain_ok, count, reason = _read_memory(path)
        profile: Mapping[str, Any] = {}
        profile_valid = True
        profile_path = Path(self.config.profile_path)
        try:
            if profile_path.exists() and profile_path.parent.is_dir():
                profile_store = AtomicProfileStore(profile_path)
                profile = profile_store.load()
            if fingerprint is not None and profile_path.exists() and profile_path.parent.is_dir():
                selection = AtomicProfileStore(profile_path).select(fingerprint.fingerprint_hash)
                profile = {**dict(profile), "selection": {"mode": selection.mode.value, "profile": None if selection.profile is None else selection.profile.as_dict(), "no_recommendation": selection.no_recommendation, "reason": selection.reason, "rollback_latched": selection.rollback_latched}}
        except (ProfileError, OSError) as exc:
            profile_valid = False
            reason = reason or "profile_unreadable:" + type(exc).__name__
        events: tuple[SessionEvent, ...] = ()
        history_valid = True
        if exists and chain_ok:
            owned = None
            try:
                reader, owned = self._history(write=False)
                assert isinstance(reader, HistoryReader)
                events = reader.recent(limit or self.config.history_limit)
            except Exception as exc:
                history_valid = False
                reason = reason or "history_unreadable:" + type(exc).__name__
            finally:
                if owned is not None:
                    owned.close()
        dataset_hash = ""
        for event in events:
            if event.kind == RecordKind.DATASET and event.dataset_hash:
                dataset_hash = event.dataset_hash
                break
        if self._last_dataset is not None:
            dataset_hash = self._last_dataset.sha256
        valid = bool(exists and chain_ok and profile_valid and history_valid and not reason)
        return OptimizerStatus(valid, exists, chain_ok, count, profile, dataset_hash, events, reason)

    def shadow(self, request: ShadowRequest | ExactFingerprint | None = None, candidate_id: str | None = None, baseline_samples: Sequence[MetricSample] = (), candidate_samples: Sequence[MetricSample] = (), **kwargs: Any) -> ShadowDecision:
        if isinstance(request, ShadowRequest):
            if any(value not in (None, (), []) for value in (candidate_id, baseline_samples, candidate_samples)) or kwargs:
                raise TypeError("provide either ShadowRequest or direct shadow arguments")
            req = request
        else:
            fingerprint = request if request is not None else kwargs.pop("fingerprint", None)
            if "baseline_samples" in kwargs:
                baseline_samples = kwargs.pop("baseline_samples")
            if "candidate_samples" in kwargs:
                candidate_samples = kwargs.pop("candidate_samples")
            req = ShadowRequest(fingerprint=fingerprint, candidate_id=candidate_id or kwargs.pop("candidate_id", ""), baseline_samples=tuple(baseline_samples), candidate_samples=tuple(candidate_samples), **kwargs)
        # Registry is the execution authority; evaluator remains the statistical
        # authority.  Neither path has an activation method.
        if not isinstance(req.fingerprint, ExactFingerprint):
            raise TypeError("shadow requires ExactFingerprint")
        decision = self.evaluator.evaluate(req.fingerprint, req.candidate_id, req.baseline_samples, req.candidate_samples, aa_baseline_samples=req.aa_baseline_samples or None, aa_control_samples=req.aa_control_samples or None, aa_pairs=req.aa_pairs or None, parameters=req.parameters or None, qualified=req.qualified, baseline_correctness=req.baseline_correctness, candidate_correctness=req.candidate_correctness, correctness=req.correctness, resources=req.resources)
        if req.write_history:
            reason_codes = [_reason_code(reason) for reason in decision.reasons]
            payload = {"status": decision.status, "qualified": decision.qualified, "ratios": dict(decision.baseline_ratios), "confidence_intervals": {key: list(value) for key, value in decision.confidence_intervals.items()}, "reasons": reason_codes, "no_activation": True, "registry_hash": self.registry.registry_hash}
            event_id = "shadow:" + sha256_hex(canonical_bytes({"fingerprint": req.fingerprint_hash, "candidate": req.candidate_id, "evidence": decision.evidence_hash}))[:48]
            event_reason = "qualified" if decision.qualified else (_reason_code(decision.reasons[0]) if decision.reasons else "no_recommendation")
            event = SessionEvent(event_id=event_id, kind=RecordKind.BENCHMARK, session_id=req.session_id, fingerprint_hash=req.fingerprint_hash, dataset_hash=req.dataset_hash, candidate_id=req.candidate_id, candidate_hash=sha256_hex(canonical_bytes({"candidate_id": req.candidate_id, "parameters": _thaw(req.parameters)})), code_hash=req.code_hash, evidence_hash=decision.evidence_hash, state=decision.status, reason=event_reason, uncertainty=None if decision.qualified else 1.0, ood=decision.status == "ood", payload=payload)
            writer, owned = self._history(write=True)
            try:
                assert isinstance(writer, HistoryWriter)
                writer.append(event)
            finally:
                if owned is not None:
                    owned.close()
        return decision

    def make_session(self, *, probe: Any = None, lease: HardwareLease | None = None, readiness: ReadinessGate | None = None, stage_runner: SubprocessStageRunner | None = None, profile_contract: Any = None, stage_specs: Mapping[str, StageSpec] | None = None, session_id: str = "session", adapter: Any = None, stage_authorization_gate: Any = None) -> SessionController:
        """Construct the one concrete, non-promoting session configuration."""
        if type(lease) is not HardwareLease:
            raise TypeError("a concrete HardwareLease is required")
        if type(readiness) is not ReadinessGate:
            raise TypeError("a concrete ReadinessGate is required")
        if type(stage_runner) is not SubprocessStageRunner:
            raise TypeError("a concrete SubprocessStageRunner is required")
        if profile_contract is None or any(not callable(getattr(profile_contract, name, None)) for name in ("validate_activation", "activate", "rollback")):
            raise TypeError("a concrete ProfileContract is required")
        from .ironmule_adapter import IronMuleTuneAdapter
        if type(adapter) is not IronMuleTuneAdapter:
            raise TypeError("a concrete IronMuleTuneAdapter is required")
        if stage_authorization_gate is None:
            stage_authorization_gate = adapter
        if stage_authorization_gate is not adapter:
            raise TypeError("stage authorization gate must be owned by the IronMuleTuneAdapter")
        if not isinstance(stage_specs, Mapping) or set(stage_specs) != {"calibrate", "test"} or any(not isinstance(value, StageSpec) for value in stage_specs.values()):
            raise TypeError("only calibrated shadow stage specs are allowed")
        for stage, spec in stage_specs.items():
            if spec.execute_authorized is not True:
                raise PermissionError("every session stage requires explicit execution authorization")
            if getattr(spec, "authorization_session_id", None) != session_id or not getattr(spec, "authorization_nonce", None) or not getattr(spec, "authorization_tag", None):
                raise PermissionError("every session stage requires adapter-issued authorization for this session")
            staged_name = getattr(spec, "stage", "")
            if staged_name != stage:
                raise ValueError("stage specification name does not match its slot")
            candidate_id = getattr(spec, "candidate_id", "")
            parameters = getattr(spec, "parameters", {})
            self.registry.resolve(candidate_id, parameters)
        return SessionController(probe=probe, lease=lease, readiness=readiness, stage_runner=stage_runner, profile_contract=profile_contract, stage_specs=stage_specs, session_id=session_id, auto_activate=False, stage_authorization_gate=stage_authorization_gate)

    def start_session(self, *, duration_minutes: int | None = None, user_started: bool = False, probe: Any = None, lease: HardwareLease | None = None, readiness: ReadinessGate | None = None, stage_runner: SubprocessStageRunner | None = None, profile_contract: Any = None, stage_specs: Mapping[str, StageSpec] | None = None, session_id: str = "session", supports_promotion: bool = False, adapter: Any = None, stage_authorization_gate: Any = None) -> Any:
        """Run only a configured, concrete shadow session after user start."""
        if not user_started:
            raise PermissionError("an explicit user start is required")
        if duration_minutes is None:
            raise TypeError("duration_minutes is required")
        if not isinstance(duration_minutes, int) or isinstance(duration_minutes, bool) or not 5 <= duration_minutes <= 30:
            raise ValueError("duration must be an integer from 5 through 30 minutes")
        if supports_promotion:
            raise PermissionError("IronMule promotion is not supported by this offline orchestrator")
        controller = self.make_session(probe=probe, lease=lease, readiness=readiness, stage_runner=stage_runner, profile_contract=profile_contract, stage_specs=stage_specs, session_id=session_id, adapter=adapter, stage_authorization_gate=stage_authorization_gate)
        return controller.run(duration_minutes, user_started=True)


OfflineOrchestrator = OptimizerOrchestrator


# Small functional facades are useful to integrations that do not need to keep
# an orchestrator object.  They preserve the same read-only defaults and do not
# create a memory database unless an explicit write operation is requested.
def doctor(root: str | os.PathLike[str], *, fingerprint: ExactFingerprint | None = None, readiness: Any = None) -> DoctorReport:
    return OptimizerOrchestrator(root=root).doctor(fingerprint=fingerprint, readiness=readiness)


def audit(root: str | os.PathLike[str], *, roots: Sequence[str] | None = None) -> CorpusInventory:
    return OptimizerOrchestrator(root=root).audit(root, roots=roots)


def build_dataset(records: CorpusInventory | Iterable[NormalizedRecord], *, assignments: Mapping[str, str] | None = None) -> DatasetSnapshot:
    if isinstance(records, CorpusInventory):
        root = records.root
    else:
        root = "."
    return OptimizerOrchestrator(root=root).build_dataset(records, assignments=assignments)


def import_inventory(inventory: CorpusInventory, memory: OptimizationMemoryV2) -> ImportReport:
    """Explicit functional import facade; the memory object owns the write."""
    if not isinstance(memory, OptimizationMemoryV2):
        raise TypeError("memory must be OptimizationMemoryV2")
    return OptimizerOrchestrator(root=inventory.root, memory=memory).import_inventory(inventory)


__all__ = [
    "OptimizerConfig", "OrchestratorConfig", "DoctorReport", "ImportReport", "OptimizerStatus", "ShadowRequest", "OptimizerOrchestrator", "OfflineOrchestrator", "doctor", "audit", "build_dataset", "import_inventory",
]
