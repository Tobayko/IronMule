"""Read-only evidence corpus audit and immutable normalization.

This module is intentionally independent from the eventual Optimization Memory
writer.  It makes an in-memory, provenance-rich view of existing evidence; no
source database or product directory is modified.
"""

from __future__ import annotations

import hashlib
import fnmatch
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from .adapters import (
    AdapterError,
    BoundsExceeded,
    DEFAULT_DISCOVERY_ROOTS,
    DiscoveredFile,
    DiscoveryLimits,
    SQLiteReadOnlyAdapter,
    discover_files,
    read_stable_bytes,
    read_bounded_json,
)
from .canonical import canonical_bytes
from .records import QualityClass


QUALITY_CLASSES: tuple[str, ...] = tuple(item.value for item in QualityClass)


class RecordSink(Protocol):
    """Minimal import boundary for a later immutable-memory writer."""

    def accept_many(self, records: Sequence["NormalizedRecord"]) -> None:
        ...


@dataclass(frozen=True)
class EvidenceContract:
    """Explicit, reviewed source-to-label mapping for one evidence family."""

    contract_id: str
    version: int
    source_basename: str
    feature_paths: tuple[str, ...]
    label_paths: tuple[str, ...]
    identity_paths: Mapping[str, str | tuple[str, ...]]
    bound_constants: Mapping[str, str] = field(default_factory=dict)
    identity_all_of: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    identity_one_of: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    identity_rules: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def contract_hash(self) -> str:
        return hashlib.sha256(
            canonical_bytes(
                {
                    "contract_id": self.contract_id,
                    "version": self.version,
                    "source_basename": self.source_basename,
                    "feature_paths": list(self.feature_paths),
                    "label_paths": list(self.label_paths),
                    "identity_paths": dict(self.identity_paths),
                    "bound_constants": dict(self.bound_constants),
                    "identity_all_of": {key: list(value) for key, value in self.identity_all_of.items()},
                    "identity_one_of": {key: list(value) for key, value in self.identity_one_of.items()},
                    "identity_rules": {key: dict(value) for key, value in self.identity_rules.items()},
                },
                max_bytes=64 * 1024,
            )
        ).hexdigest()

    def matches(self, source_path: str, payload: Mapping[str, Any], *, logical_file: str | None = None) -> bool:
        del payload
        if logical_file is not None:
            return logical_file == self.source_basename
        return Path(source_path.split("#", 1)[0]).name == self.source_basename

    def identity_satisfied(self, payload: Mapping[str, Any]) -> bool:
        flattened = dict(_flatten(payload))
        def valid(path: str) -> bool:
            value = flattened[path]
            rule = next((candidate for pattern, candidate in self.identity_rules.items() if fnmatch.fnmatchcase(path, pattern)), {})
            expected = rule.get("type")
            if expected == "string" and (not isinstance(value, str) or not value):
                return False
            if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                return False
            if expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                return False
            if rule.get("nonempty") and value in (None, "", [], {}):
                return False
            if "min" in rule and (not isinstance(value, (int, float)) or value < rule["min"]):
                return False
            if "max" in rule and (not isinstance(value, (int, float)) or value > rule["max"]):
                return False
            pattern = rule.get("format")
            if pattern and (not isinstance(value, str) or re.fullmatch(pattern, value) is None):
                return False
            return True
        for patterns in self.identity_all_of.values():
            if any(not any(fnmatch.fnmatchcase(path, pattern) and valid(path) for path in flattened) for pattern in patterns):
                return False
        for patterns in self.identity_one_of.values():
            if not any(any(fnmatch.fnmatchcase(path, pattern) and valid(path) for path in flattened) for pattern in patterns):
                return False
        return True


Q2_PROFILES_CONTRACT = EvidenceContract(
    contract_id="friday.q2_profiles.confirmation_ratio.v1",
    version=1,
    source_basename="Q2_profiles.json",
    feature_paths=(
        "*.conditions.chip",
        "*.conditions.fingerprint",
        "*.conditions.model_id",
        "*.conditions.model_identity_sha256",
        "*.conditions.execution_plan",
        "*.conditions.max_tokens",
        "*.conditions.prompt_tokens",
        "*.conditions.mlx",
        "*.conditions.mlx_lm",
        "*.conditions.quantisation.bits",
        "*.conditions.quantisation.group_size",
        "*.knobs.*",
    ),
    label_paths=(
        "*.confirmation.ratio.decode_ns.median_ratio",
        "*.confirmation.ratio.prefill_ns.median_ratio",
        "*.confirmation.ratio.total_ns.median_ratio",
    ),
    identity_paths={
        "hardware": "*.conditions.fingerprint",
        "model": ("*.conditions.model_id", "*.conditions.model_identity_sha256", "*.conditions.quantisation_sha256"),
        "workload": ("*.conditions.execution_plan", "*.conditions.max_tokens", "*.conditions.prompt_tokens"),
        "prompt": "*.conditions.max_tokens",
        "time": "*.tuned_at",
    },
    bound_constants={"study": "Q2", "run": "q2-profile-confirmation-v1"},
    identity_all_of={
        "model": ("*.conditions.model_id", "*.conditions.model_revision", "*.conditions.model_identity_sha256", "*.conditions.model_manifest_sha256", "*.conditions.quantisation_sha256", "*.conditions.tokenizer_sha256"),
        "hardware": ("*.conditions.fingerprint", "*.conditions.chip"),
        "workload": ("*.conditions.execution_plan", "*.conditions.max_tokens", "*.conditions.prompt_tokens"),
    },
    identity_one_of={"prompt": ("*.conditions.prompt_tokens", "*.conditions.max_tokens")},
    identity_rules={
        "*.conditions.chip": {"type": "string", "nonempty": True},
        "*.conditions.fingerprint": {"type": "string", "format": r"[0-9a-f]{16,64}"},
        "*.conditions.model_id": {"type": "string", "nonempty": True},
        "*.conditions.model_revision": {"type": "string", "nonempty": True},
        "*.conditions.model_identity_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
        "*.conditions.model_manifest_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
        "*.conditions.quantisation_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
        "*.conditions.tokenizer_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
        "*.conditions.execution_plan": {"type": "string", "nonempty": True},
        "*.conditions.max_tokens": {"type": "integer", "min": 1},
        "*.conditions.prompt_tokens": {"type": "integer", "min": 1},
    },
)


B27D_CONTRACT = EvidenceContract(
    contract_id="friday.b27d.interactive_summary.v1",
    version=1,
    source_basename="B27d_gemma4b_post_20260828.json",
    feature_paths=(
        "benchmark.arms.interactive.runtime_fingerprint.*",
        "benchmark.arms.interactive.workload.*",
    ),
    label_paths=(
        "benchmark.arms.interactive.summary.executor_wall_ms.median",
        "benchmark.arms.interactive.summary.physical_tokens_per_second.median",
    ),
    identity_paths={
        "hardware": "benchmark.arms.interactive.runtime_fingerprint.hardware_fingerprint",
        "model": ("benchmark.arms.interactive.runtime_fingerprint.model_id", "model_binding.model_manifest_sha256"),
        "workload": ("benchmark.arms.interactive.runtime_fingerprint.workload.plan", "benchmark.arms.interactive.runtime_fingerprint.workload.max_tokens"),
        "prompt": "benchmark.arms.interactive.runtime_fingerprint.workload.max_tokens",
        "time": "captured_at",
        "run": "experiment_id",
    },
    identity_all_of={
        "model": ("benchmark.arms.interactive.runtime_fingerprint.model_id", "model_binding.model_manifest_sha256"),
        "hardware": ("benchmark.arms.interactive.runtime_fingerprint.hardware_fingerprint", "benchmark.arms.interactive.runtime_fingerprint.chip"),
        "workload": ("benchmark.arms.interactive.runtime_fingerprint.workload.plan", "benchmark.arms.interactive.runtime_fingerprint.workload.max_tokens"),
    },
    identity_one_of={"prompt": ("benchmark.arms.interactive.runtime_fingerprint.workload.max_tokens",)},
    identity_rules={
        "benchmark.arms.interactive.runtime_fingerprint.hardware_fingerprint": {"type": "string", "format": r"[0-9a-f]{16,64}"},
        "benchmark.arms.interactive.runtime_fingerprint.chip": {"type": "string", "nonempty": True},
        "benchmark.arms.interactive.runtime_fingerprint.model_id": {"type": "string", "nonempty": True},
        "model_binding.model_manifest_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
        "benchmark.arms.interactive.runtime_fingerprint.workload.plan": {"type": "string", "nonempty": True},
        "benchmark.arms.interactive.runtime_fingerprint.workload.max_tokens": {"type": "integer", "min": 1},
    },
)


# B36/B39d do not expose a complete, contract-bound hardware identity in their
# result objects; registering them would manufacture a cross-device feature.
EVIDENCE_CONTRACTS: tuple[EvidenceContract, ...] = (Q2_PROFILES_CONTRACT, B27D_CONTRACT)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze(v) for key, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(v) for key, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


_MAX_CANONICAL_BYTES = 64 * 1024 * 1024
_MAX_CANONICAL_ITEMS = 2_000_000


def _canonical(value: Any) -> bytes:
    return canonical_bytes(
        _thaw(value),
        max_bytes=_MAX_CANONICAL_BYTES,
        max_items=_MAX_CANONICAL_ITEMS,
    )


def _canon_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            child = f"{prefix}.{key}" if prefix else key
            yield from _flatten(value[key], child)
    elif isinstance(value, (list, tuple)):
        # Lists are values for feature/label purposes.  We do not create an
        # unbounded field for every item.
        yield prefix, value
    else:
        yield prefix, value


def _find_value(payload: Any, names: Sequence[str]) -> Any:
    wanted = {name.lower() for name in names}
    for path, value in _flatten(payload):
        if path.rsplit(".", 1)[-1].lower() in wanted and value not in (None, ""):
            return value
    return None


def _string_value(payload: Any, names: Sequence[str], default: str = "unknown") -> str:
    value = _find_value(payload, names)
    if value is None:
        return default
    if isinstance(value, (dict, list, tuple)):
        return _canon_hash(value)[:24]
    return str(value)


_TIMING_TERMS = (
    "time",
    "latency",
    "ttft",
    "throughput",
    "tokens_per_second",
    "tokens_s",
    "duration",
    "elapsed",
    "p50",
    "p95",
    "median",
    "mean",
    "samples",
    "benchmark",
    "performance",
    "speed",
    "ratio",
    "regret",
)
_NON_LABEL_TERMS = (
    "created_at",
    "recorded_at",
    "timestamp",
    "start_time",
    "end_time",
)
_FEATURE_TERMS = (
    "hardware",
    "chip",
    "gpu",
    "cpu",
    "ram",
    "memory",
    "environment",
    "os",
    "metal",
    "mlx",
    "python",
    "compiler",
    "model",
    "quant",
    "tokenizer",
    "workload",
    "prompt",
    "shape",
    "stride",
    "dtype",
    "batch",
    "context",
    "candidate",
    "template",
    "action",
    "parameter",
    "operation",
    "seed",
    "version",
    "power",
    "dirty",
)


def _path_is_timing(path: str) -> bool:
    low = path.lower()
    leaf = low.rsplit(".", 1)[-1]
    if any(term in leaf for term in _NON_LABEL_TERMS):
        return False
    return any(term in low for term in _TIMING_TERMS)


def _collect_marker_fields(payload: Any, marker_names: set[str]) -> tuple[str, ...]:
    found: set[str] = set()
    for path, value in _flatten(payload):
        leaf = path.rsplit(".", 1)[-1].lower()
        if leaf in marker_names:
            if isinstance(value, Mapping):
                found.update(key for key in value if isinstance(key, str))
            elif isinstance(value, (list, tuple)):
                found.update(item for item in value if isinstance(item, str))
            elif value not in (None, False, ""):
                found.add(path)
    # Common nested format: {missing: {field: reason}}.
    for path, value in _flatten(payload):
        if path.lower().endswith(".missing_reason") and value not in (None, ""):
            found.add(path.rsplit(".", 1)[0])
        if path.lower().endswith(".censored") and value not in (None, False, ""):
            found.add(path.rsplit(".", 1)[0])
    return tuple(sorted(found))


def _contract_for(source_path: str, payload: Mapping[str, Any], *, logical_file: str | None = None) -> EvidenceContract | None:
    for contract in EVIDENCE_CONTRACTS:
        if contract.matches(source_path, payload, logical_file=logical_file):
            return contract
    return None


def _contract_paths(payload: Mapping[str, Any], patterns: Sequence[str]) -> tuple[str, ...]:
    available = [path for path, _ in _flatten(payload)]
    return tuple(sorted(path for path in available if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)))


def _contract_identity(payload: Mapping[str, Any], contract: EvidenceContract) -> dict[str, str]:
    flattened = dict(_flatten(payload))
    result: dict[str, str] = {}
    for name, patterns in contract.identity_paths.items():
        if isinstance(patterns, str):
            patterns = (patterns,)
        values: list[str] = []
        for pattern in patterns:
            candidates = sorted(path for path in flattened if fnmatch.fnmatchcase(path, pattern))
            if candidates:
                value = flattened[candidates[0]]
                values.append(
                    _canon_hash(value)[:24]
                    if isinstance(value, (Mapping, list, tuple))
                    else canonical_bytes(value, max_bytes=4096).decode("utf-8")
                )
        if values:
            result[name] = "|".join(values)
    return result


def _quality_for(
    path: str,
    payload: Any,
    *,
    parse_error: bool = False,
    partial: bool = False,
    trusted_source: bool = False,
) -> QualityClass:
    if partial:
        return QualityClass.QUARANTINED
    if parse_error:
        return QualityClass.INVALID
    low = path.lower()
    if "invalidated" in low or "invalid" in low:
        return QualityClass.INVALID
    if trusted_source and ("preregistration" in low or "pre_registration" in low):
        return QualityClass.FORMAL
    # A summary is safe for audit/coverage but never a measurable training
    # label.  Explicit raw sample markers override the filename heuristic.
    if "summary" in low:
        raw = _find_value(payload, ("raw_samples", "raw_measurements_available", "samples"))
        if raw in (None, False, 0, [], ""):
            return QualityClass.LEGACY_SUMMARY
    if isinstance(payload, Mapping):
        for key in ("formal_claim", "formal", "preregistered"):
            if trusted_source and payload.get(key) is True:
                return QualityClass.FORMAL
        for key in ("raw_samples", "raw_measurements", "measurements", "samples"):
            if trusted_source and key in payload and payload[key] not in (None, [], ""):
                return QualityClass.ENGINEERING
    return QualityClass.EXPLORATORY


@dataclass(frozen=True)
class NormalizedRecord:
    """An immutable source record with explicit feature/label provenance."""

    record_id: str
    source_path: str
    source_kind: str
    quality: QualityClass
    data: Mapping[str, Any]
    feature_fields: tuple[str, ...] = ()
    label_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    censored_fields: tuple[str, ...] = ()
    source_sha256: str = ""
    content_fingerprint: str = ""
    source_fingerprint: str = ""
    study_id: str = "unknown"
    run_id: str = "unknown"
    observed_time: str = "unknown"
    hardware_fingerprint: str = "unknown"
    model_fingerprint: str = "unknown"
    workload_fingerprint: str = "unknown"
    prompt_family: str = "unknown"
    dirty: bool | None = None
    duplicate_of: str | None = None
    source_verified: bool | None = None
    manifest_verified: bool = False
    contract_verified: bool | None = None
    source_integrity_verified: bool | None = None
    archive_manifest_verified: bool | None = None
    contract_id: str | None = None
    contract_version: int | None = None
    contract_hash: str | None = None
    contract_bound_constants: Mapping[str, str] = field(default_factory=dict)
    identity_contract_valid: bool = False
    logical_source_file: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quality", QualityClass(self.quality))
        object.__setattr__(self, "data", _freeze(dict(self.data)))
        for name in (
            "feature_fields",
            "label_fields",
            "missing_fields",
            "censored_fields",
        ):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))
        if not self.content_fingerprint:
            object.__setattr__(self, "content_fingerprint", _canon_hash(self.data))
        if not self.source_fingerprint:
            object.__setattr__(
                self,
                "source_fingerprint",
                hashlib.sha256(self.source_path.encode("utf-8")).hexdigest(),
            )
        if self.source_integrity_verified is None:
            object.__setattr__(self, "source_integrity_verified", self.source_verified if self.source_verified is not None else bool(self.source_sha256))
        object.__setattr__(self, "source_verified", self.source_integrity_verified)
        if self.archive_manifest_verified is None:
            object.__setattr__(self, "archive_manifest_verified", self.manifest_verified)
        object.__setattr__(self, "manifest_verified", bool(self.archive_manifest_verified))
        if self.contract_verified is None:
            object.__setattr__(self, "contract_verified", bool(self.feature_fields and self.label_fields))

    @property
    def training_eligible(self) -> bool:
        return self.quality in {
            QualityClass.FORMAL,
            QualityClass.ENGINEERING,
            QualityClass.EXPLORATORY,
        } and bool(self.label_fields) and not self.censored_fields and bool(self.source_integrity_verified) and bool(self.archive_manifest_verified) and bool(self.contract_verified) and bool(re.fullmatch(r"[0-9a-f]{64}", self.source_sha256 or "")) and self.contract_valid and self.identity_complete and self.identity_contract_valid

    @property
    def contract_valid(self) -> bool:
        flattened = dict(_flatten(self.data))
        if not self.feature_fields or not self.label_fields:
            return False
        if any(path not in flattened or flattened[path] is None for path in (*self.feature_fields, *self.label_fields)):
            return False
        for path in self.label_fields:
            value = flattened[path]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return False
        return bool(self.contract_verified)

    @property
    def identity_complete(self) -> bool:
        return all(
            value not in (None, "", "unknown")
            for value in (
                self.study_id,
                self.run_id,
                self.observed_time,
                self.hardware_fingerprint,
                self.model_fingerprint,
                self.workload_fingerprint,
                self.prompt_family,
            )
        )

    @property
    def summary_only(self) -> bool:
        return self.quality == QualityClass.LEGACY_SUMMARY or not self.label_fields

    def as_dict(self) -> dict[str, Any]:
        result = {
            "record_id": self.record_id,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "quality": self.quality.value,
            "data": _thaw(self.data),
            "feature_fields": list(self.feature_fields),
            "label_fields": list(self.label_fields),
            "missing_fields": list(self.missing_fields),
            "censored_fields": list(self.censored_fields),
            "source_sha256": self.source_sha256,
            "content_fingerprint": self.content_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "observed_time": self.observed_time,
            "hardware_fingerprint": self.hardware_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "prompt_family": self.prompt_family,
            "dirty": self.dirty,
            "duplicate_of": self.duplicate_of,
            "source_verified": self.source_verified,
            "manifest_verified": self.manifest_verified,
            "contract_verified": self.contract_verified,
            "source_integrity_verified": self.source_integrity_verified,
            "archive_manifest_verified": self.archive_manifest_verified,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "contract_hash": self.contract_hash,
            "contract_bound_constants": dict(self.contract_bound_constants),
            "identity_contract_valid": self.identity_contract_valid,
            "logical_source_file": self.logical_source_file,
        }
        return result


@dataclass(frozen=True)
class ManifestVerification:
    manifest_path: str
    entries_checked: int
    unique_content: int
    missing: tuple[str, ...] = ()
    hash_mismatches: tuple[str, ...] = ()
    byte_mismatches: tuple[str, ...] = ()
    path_violations: tuple[str, ...] = ()
    invalid_entries: tuple[str, ...] = ()
    duplicate_entries: int = 0
    verified_paths: tuple[str, ...] = ()
    logical_by_stored: Mapping[str, str] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return bool(
            self.missing
            or self.hash_mismatches
            or self.byte_mismatches
            or self.path_violations
            or self.invalid_entries
        )

    @property
    def ok(self) -> bool:
        return not self.terminal


@dataclass(frozen=True)
class CorpusIssue:
    path: str
    code: str
    detail: str
    terminal: bool = False


@dataclass(frozen=True)
class CorpusInventory:
    """Stable read-only result of a corpus audit."""

    root: str
    files: tuple[DiscoveredFile, ...]
    records: tuple[NormalizedRecord, ...]
    issues: tuple[CorpusIssue, ...] = ()
    manifest: ManifestVerification | None = None
    sqlite: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda row: row.relative_path)))
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda row: row.record_id)))
        object.__setattr__(self, "issues", tuple(sorted(self.issues, key=lambda row: (row.path, row.code, row.detail))))

    @property
    def quality_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in QUALITY_CLASSES}
        for record in self.records:
            counts[record.quality.value] += 1
        return counts

    @property
    def excluded_files(self) -> tuple[DiscoveredFile, ...]:
        return tuple(file for file in self.files if file.excluded_reason is not None)

    @property
    def exclusions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for file in self.excluded_files:
            assert file.excluded_reason is not None
            counts[file.excluded_reason] = counts.get(file.excluded_reason, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def usable_records(self) -> tuple[NormalizedRecord, ...]:
        return tuple(record for record in self.records if record.training_eligible)

    @property
    def duplicate_count(self) -> int:
        seen: set[str] = set()
        duplicates = 0
        for record in self.records:
            if record.content_fingerprint in seen:
                duplicates += 1
            seen.add(record.content_fingerprint)
        return duplicates

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": [
                {
                    "relative_path": row.relative_path,
                    "root_name": row.root_name,
                    "kind": row.kind,
                    "size_bytes": row.size_bytes,
                    "sha256": row.sha256,
                    "st_dev": row.st_dev,
                    "st_ino": row.st_ino,
                    "read_error": row.read_error,
                }
                for row in self.files
            ],
            "records": [record.as_dict() for record in self.records],
            "issues": [issue.__dict__ for issue in self.issues],
            "quality_counts": self.quality_counts,
            "duplicate_count": self.duplicate_count,
            "exclusions": self.exclusions,
            "manifest": None if self.manifest is None else self.manifest.__dict__,
        }


def _normalize_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str,
    source_kind: str,
    quality: QualityClass,
    source_sha256: str,
    duplicate_of: str | None = None,
    source_verified: bool | None = None,
    manifest_verified: bool = False,
    contract: EvidenceContract | None = None,
    logical_source_file: str | None = None,
) -> NormalizedRecord:
    frozen_payload = _freeze(dict(payload))
    flattened = tuple(_flatten(payload))
    explicit_features = payload.get("features") if isinstance(payload.get("features"), Mapping) else None
    explicit_labels = payload.get("labels") if isinstance(payload.get("labels"), Mapping) else None
    feature_fields: set[str] = set()
    label_fields: set[str] = set()
    if explicit_features is not None:
        if any(not isinstance(key, str) for key in explicit_features):
            raise TypeError("feature contract keys must be strings")
        feature_fields.update(
            key if key in dict(_flatten(payload)) else f"features.{key}"
            for key in explicit_features
        )
    if explicit_labels is not None:
        if any(not isinstance(key, str) for key in explicit_labels):
            raise TypeError("label contract keys must be strings")
        label_fields.update(
            key if key in dict(_flatten(payload)) else f"labels.{key}"
            for key in explicit_labels
        )
    if contract is not None:
        feature_fields.update(_contract_paths(payload, contract.feature_paths))
        label_fields.update(_contract_paths(payload, contract.label_paths))
    elif explicit_features is None and explicit_labels is None:
        for path, value in flattened:
            low = path.lower()
            if _path_is_timing(path):
                label_fields.add(path)
            elif any(term in low for term in _FEATURE_TERMS):
                feature_fields.add(path)
    # Never permit summary text, provenance timestamps or metadata to become a
    # measured target.  The values remain in ``data`` for auditability.
    label_fields = {
        path
        for path in label_fields
        if not any(term in path.lower() for term in _NON_LABEL_TERMS)
        and not path.lower().endswith(("summary", "description", "note"))
    }
    missing = set(_collect_marker_fields(payload, {"missing", "missing_fields"}))
    censored = set(_collect_marker_fields(payload, {"censored", "censoring", "censored_fields"}))
    for path, value in flattened:
        if value is None and (_path_is_timing(path) or path in feature_fields):
            missing.add(path)
    # A censored/aborted/timeout result cannot be used as a performance label;
    # retain it as a record with explicit censoring instead.
    status = _string_value(payload, ("status", "result_status", "classification"), "").lower()
    if any(term in status for term in ("timeout", "aborted", "cancel", "censor")):
        censored.add(status or "result")
    record_identity = {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "data": _thaw(frozen_payload),
        "quality": quality.value,
    }
    record_id = hashlib.sha256(_canonical(record_identity)).hexdigest()
    prompt_value = _find_value(payload, ("prompt_family", "prompt_template", "prompt_id"))
    prompt_family = str(prompt_value) if prompt_value not in (None, "") else "unknown"
    # Keep prompt content out of the grouping key unless the source explicitly
    # provides a family; a hash is enough for leakage detection.
    identity = _contract_identity(payload, contract) if contract is not None else {}
    if contract is not None:
        for name, value in contract.bound_constants.items():
            identity.setdefault(name, value)
    return NormalizedRecord(
        record_id=record_id,
        source_path=source_path,
        source_kind=source_kind,
        quality=quality,
        data=frozen_payload,
        feature_fields=tuple(feature_fields),
        label_fields=tuple(label_fields),
        missing_fields=tuple(missing),
        censored_fields=tuple(censored),
        source_sha256=source_sha256,
        content_fingerprint=_canon_hash(frozen_payload),
        study_id=identity.get("study", _string_value(payload, ("study_id", "study", "experiment_id"))),
        run_id=identity.get("run", _string_value(payload, ("run_id", "run", "session_id", "record_id"))),
        observed_time=identity.get("time", _string_value(payload, ("observed_at", "observed_at_unix_ns", "created_at", "timestamp"))),
        hardware_fingerprint=identity.get("hardware", _string_value(payload, ("hardware_fingerprint", "hardware_key", "hardware", "chip"))),
        model_fingerprint=identity.get("model", _string_value(payload, ("model_fingerprint", "model_revision", "model", "model_id"))),
        workload_fingerprint=identity.get("workload", _string_value(payload, ("workload_fingerprint", "workload_key", "workload", "operation"))),
        prompt_family=identity.get("prompt", prompt_family),
        dirty=_find_value(payload, ("git_dirty", "dirty")) if isinstance(_find_value(payload, ("git_dirty", "dirty")), bool) else None,
        duplicate_of=duplicate_of,
        source_verified=source_verified,
        manifest_verified=manifest_verified,
            contract_verified=contract is not None and bool(feature_fields and label_fields),
        source_integrity_verified=source_verified,
        archive_manifest_verified=manifest_verified,
        contract_id=contract.contract_id if contract is not None else None,
        contract_version=contract.version if contract is not None else None,
        contract_hash=contract.contract_hash if contract is not None else None,
        contract_bound_constants=dict(contract.bound_constants) if contract is not None else {},
        identity_contract_valid=bool(contract is not None and contract.identity_satisfied(payload)),
        logical_source_file=logical_source_file or (contract.source_basename if contract is not None else None),
    )


def verify_archive_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    limits: DiscoveryLimits | None = None,
    expected_identity: tuple[int, int, int] | None = None,
) -> ManifestVerification:
    """Verify archive manifest bytes, hashes, duplicate references and paths."""

    limits = limits or DiscoveryLimits()
    path = Path(manifest_path)
    try:
        payload = read_bounded_json(path, limits=limits, expected_identity=expected_identity)
    except AdapterError as exc:
        return ManifestVerification(str(path), 0, 0, invalid_entries=(str(exc),))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("entries"), list):
        return ManifestVerification(str(path), 0, 0, invalid_entries=("entries must be a list",))
    entries = payload["entries"]
    if len(entries) > limits.max_files:
        return ManifestVerification(str(path), 0, 0, invalid_entries=("manifest entry limit exceeded",))
    archive_root = path.parent.resolve()
    missing: list[str] = []
    hashes: list[str] = []
    byte_mismatches: list[str] = []
    violations: list[str] = []
    invalid: list[str] = []
    digest_paths: dict[str, str] = {}
    verified_paths: set[str] = set()
    logical_by_stored: dict[str, str] = {}
    stored: dict[str, Mapping[str, Any]] = {}
    duplicate_primary: set[str] = set()
    seen_primary: set[str] = set()
    duplicate_entries = 0
    # Register every primary path before checking duplicate references so a
    # manifest cannot smuggle a forward reference past the verifier.
    for item in entries:
        if isinstance(item, Mapping) and item.get("duplicate_of") is None and isinstance(item.get("stored_as"), str):
            stored[str(item["stored_as"])] = item
    for index, item in enumerate(entries):
        label = f"entry[{index}]"
        if not isinstance(item, Mapping):
            invalid.append(label)
            continue
        expected = item.get("sha256")
        logical_file = item.get("file")
        byte_count = item.get("bytes")
        stored_as = item.get("stored_as")
        duplicate_of = item.get("duplicate_of")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            invalid.append(label + ": sha256")
            continue
        if (
            not isinstance(logical_file, str)
            or not logical_file
            or Path(logical_file).is_absolute()
            or "\x00" in logical_file
            or Path(logical_file).name != logical_file
            or ".." in Path(logical_file).parts
        ):
            violations.append(str(logical_file))
            continue
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            invalid.append(label + ": bytes")
            continue
        if duplicate_of is not None:
            duplicate_entries += 1
            if not isinstance(duplicate_of, str):
                invalid.append(label + ": duplicate_of")
                continue
            duplicate_path = Path(duplicate_of)
            if duplicate_path.is_absolute() or ".." in duplicate_path.parts or "\x00" in duplicate_of:
                violations.append(duplicate_of)
                continue
            target = stored.get(duplicate_of)
            if target is None:
                missing.append(duplicate_of)
                continue
            if str(target.get("sha256", "")).lower() != expected.lower() or int(target.get("bytes", -1)) != byte_count:
                hashes.append(label + ": duplicate metadata")
            continue
        if not isinstance(stored_as, str) or not stored_as:
            invalid.append(label + ": stored_as")
            continue
        if stored_as in seen_primary:
            duplicate_primary.add(stored_as)
            invalid.append(label + ": duplicate stored_as")
        seen_primary.add(stored_as)
        candidate_rel = Path(stored_as)
        if candidate_rel.is_absolute() or ".." in candidate_rel.parts or "\x00" in stored_as:
            violations.append(stored_as)
            continue
        # The archive is content-addressed: the first directory and filename
        # must carry the claimed digest.  This also prevents two primary
        # entries from silently aliasing an unrelated path.
        if (
            len(candidate_rel.parts) < 2
            or candidate_rel.parts[0].lower() != expected[:2].lower()
            or not candidate_rel.name.lower().startswith(expected.lower())
        ):
            violations.append(stored_as)
            continue
        candidate = archive_root / candidate_rel
        try:
            candidate.relative_to(archive_root)
        except ValueError:
            violations.append(stored_as)
            continue
        # Inspect every path component without resolving it.  This rejects a
        # symlink in an intermediate directory as well as a symlink leaf.
        component = archive_root
        unsafe = False
        for part in candidate_rel.parts:
            component = component / part
            try:
                if os.path.islink(component):
                    unsafe = True
                    break
            except OSError:
                unsafe = True
                break
        if unsafe:
            violations.append(stored_as)
            continue
        try:
            st = candidate.lstat()
            if os.path.islink(candidate) or not candidate.is_file():
                violations.append(stored_as)
                continue
            size_mismatch = st.st_size != byte_count
            if size_mismatch:
                byte_mismatches.append(stored_as)
            raw, _ = read_stable_bytes(
                candidate,
                limits=limits,
                expected_identity=(int(st.st_dev), int(st.st_ino), int(st.st_size)),
            )
            actual = hashlib.sha256(raw).hexdigest()
            if actual.lower() != expected.lower():
                hashes.append(stored_as)
            elif not size_mismatch:
                digest_paths.setdefault(actual, stored_as)
                verified_paths.add(stored_as)
                logical_by_stored[stored_as] = logical_file
        except (FileNotFoundError, PermissionError):
            missing.append(stored_as)
        except (AdapterError, BoundsExceeded) as exc:
            invalid.append(f"{stored_as}: {type(exc).__name__}")
    return ManifestVerification(
        manifest_path=str(path),
        entries_checked=len(entries),
        unique_content=len(digest_paths),
        missing=tuple(sorted(set(missing))),
        hash_mismatches=tuple(sorted(set(hashes))),
        byte_mismatches=tuple(sorted(set(byte_mismatches))),
        path_violations=tuple(sorted(set(violations))),
        invalid_entries=tuple(sorted(set(invalid))),
        duplicate_entries=duplicate_entries + len(duplicate_primary),
        verified_paths=tuple(sorted(verified_paths)),
        logical_by_stored=dict(sorted(logical_by_stored.items())),
    )


class CorpusAuditor:
    """Bounded auditor for the four explicitly allowed project roots."""

    def __init__(
        self,
        project_root: str | os.PathLike[str],
        *,
        roots: Sequence[str] = DEFAULT_DISCOVERY_ROOTS,
        limits: DiscoveryLimits | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.roots = tuple(roots)
        self.limits = limits or DiscoveryLimits()

    def discover(self) -> tuple[DiscoveredFile, ...]:
        return discover_files(self.project_root, roots=self.roots, limits=self.limits)

    def audit(self, sink: RecordSink | None = None) -> CorpusInventory:
        issues: list[CorpusIssue] = []
        discovery_terminal = False
        try:
            files = self.discover()
        except BoundsExceeded as exc:
            files = tuple(getattr(exc, "partial_files", ()))
            discovery_terminal = True
            issues.append(CorpusIssue("<discovery>", "bounds_exceeded", str(exc), terminal=True))
        except AdapterError as exc:
            files = ()
            discovery_terminal = True
            issues.append(CorpusIssue("<discovery>", "discovery_error", str(exc), terminal=True))
        records: list[NormalizedRecord] = []
        sqlite_inventory: list[Mapping[str, Any]] = []
        manifest_result: ManifestVerification | None = None
        manifest_file = next(
            (
                file
                for file in files
                if file.path.name == "MANIFEST.json"
                and file.path.parent.name == "ironmule-evidence-archive"
                and not file.read_error
            ),
            None,
        )
        if manifest_file is not None:
            manifest_result = verify_archive_manifest(
                manifest_file.path,
                limits=self.limits,
                expected_identity=(manifest_file.st_dev, manifest_file.st_ino, manifest_file.size_bytes)
                if manifest_file.st_dev is not None and manifest_file.st_ino is not None
                else None,
            )
            if manifest_result.terminal:
                issues.append(
                    CorpusIssue(
                        manifest_file.relative_path,
                        "manifest_verification_failed",
                        "archive manifest has terminal integrity errors",
                        terminal=True,
                    )
                )
        seen_content: dict[str, str] = {}
        for file in files:
            relative = file.relative_path
            path = file.path
            if file.excluded_reason is not None:
                continue
            if file.read_error:
                issues.append(CorpusIssue(relative, "source_changed_during_discovery", file.read_error, terminal=True))
                records.append(
                    _normalize_payload(
                        {"source": relative, "read_error": file.read_error},
                        source_path=relative,
                        source_kind=file.kind,
                        quality=QualityClass.QUARANTINED,
                        source_sha256="",
                        source_verified=False,
                    )
                )
                continue
            if path.name == "MANIFEST.json" and path.parent.name == "ironmule-evidence-archive":
                continue
            in_archive = "ironmule-evidence-archive/" in relative
            archive_rel = ""
            if in_archive and manifest_file is not None:
                archive_rel = path.relative_to(manifest_file.path.parent).as_posix()
            archive_verified = bool(
                manifest_result is not None
                and manifest_result.ok
                and in_archive
                and archive_rel in manifest_result.verified_paths
            )
            if discovery_terminal:
                archive_verified = False
            if file.kind == "sqlite":
                pending_rows: list[tuple[int, Mapping[str, Any], QualityClass, EvidenceContract | None]] = []
                pending_schema: Mapping[str, Any] | None = None
                try:
                    with SQLiteReadOnlyAdapter(path, limits=self.limits) as adapter:
                        schema = adapter.schema_inventory()
                        pending_schema = {
                            "path": relative,
                            "tables": [
                                {"name": table.name, "columns": list(table.columns), "row_count": table.row_count, "count_error": table.count_error}
                                for table in schema.tables
                            ],
                            "identity": dict(schema.identity),
                        }
                        for index, row in enumerate(adapter.records()):
                            decode_error = row.get("_payload_decode_error")
                            decoded_payload = row.get("payload")
                            if decode_error or not isinstance(decoded_payload, Mapping):
                                payload = {"payload_decode_error": decode_error or "missing_known_payload"}
                                quality = QualityClass.INVALID
                            else:
                                # Keep row identity/status as bounded metadata,
                                # but never treat arbitrary columns as labels.
                                payload = dict(decoded_payload)
                                for key in ("record_id", "run_id", "study_id", "session_id", "status", "result_status", "created_at_unix_ns"):
                                    if key in row and key not in payload:
                                        payload[key] = row[key]
                                quality = _quality_for(relative, payload, trusted_source=archive_verified)
                            logical_file = manifest_result.logical_by_stored.get(archive_rel) if manifest_result is not None else None
                            contract = _contract_for(relative, payload, logical_file=logical_file) if archive_verified else None
                            if contract is not None and contract.feature_paths and contract.label_paths:
                                quality = QualityClass.ENGINEERING
                            if discovery_terminal:
                                quality = QualityClass.QUARANTINED
                            if in_archive and not archive_verified:
                                quality = QualityClass.QUARANTINED
                            pending_rows.append((index, payload, quality, contract))
                except AdapterError as exc:
                    issues.append(CorpusIssue(relative, "sqlite_read_error", str(exc), terminal=True))
                    records.append(_normalize_payload({"source": relative, "sqlite_error": str(exc)}, source_path=relative, source_kind="sqlite", quality=QualityClass.QUARANTINED, source_sha256=file.sha256, source_verified=False, manifest_verified=False))
                    continue
                if pending_schema is not None:
                    sqlite_inventory.append(pending_schema)
                for index, payload, quality, contract in pending_rows:
                    record = _normalize_payload(payload, source_path=f"{relative}#row:{index}", source_kind="sqlite", quality=quality, source_sha256=file.sha256, source_verified=bool(file.sha256), manifest_verified=archive_verified, contract=contract)
                    if record.content_fingerprint in seen_content:
                        record = _normalize_payload(payload, source_path=f"{relative}#row:{index}", source_kind="sqlite", quality=quality, source_sha256=file.sha256, duplicate_of=seen_content[record.content_fingerprint], source_verified=bool(file.sha256), manifest_verified=archive_verified, contract=contract)
                    else:
                        seen_content[record.content_fingerprint] = record.record_id
                    records.append(record)
                continue
            if path.suffix.lower() == ".json" or path.name.lower().endswith(".json.partial"):
                partial = path.name.lower().endswith(".partial")
                try:
                    payload = read_bounded_json(
                        path,
                        limits=self.limits,
                        expected_identity=(file.st_dev, file.st_ino, file.size_bytes)
                        if file.st_dev is not None and file.st_ino is not None
                        else None,
                        expected_sha256=file.sha256 or None,
                    )
                except AdapterError as exc:
                    quality = QualityClass.QUARANTINED if partial or in_archive else QualityClass.INVALID
                    detail = str(exc)
                    changed = "changed" in detail or "truncated" in detail or "disappeared" in detail
                    issues.append(CorpusIssue(relative, "quarantined" if partial else "invalid_json", detail, terminal=changed))
                    record = _normalize_payload(
                        {"parse_error": str(exc), "source": relative},
                        source_path=relative,
                        source_kind="json",
                        quality=quality,
                        source_sha256=file.sha256,
                        source_verified=False,
                    )
                    records.append(record)
                    continue
                payload_items: list[Mapping[str, Any]]
                if isinstance(payload, list):
                    payload_items = [item if isinstance(item, Mapping) else {"value": item} for item in payload]
                elif isinstance(payload, Mapping):
                    payload_items = [payload]
                else:
                    payload_items = [{"value": payload}]
                for index, item in enumerate(payload_items):
                    quality = _quality_for(relative, item, partial=partial, trusted_source=archive_verified)
                    logical_file = manifest_result.logical_by_stored.get(archive_rel) if manifest_result is not None else None
                    contract = _contract_for(relative, item, logical_file=logical_file) if archive_verified else None
                    if contract is not None and contract.feature_paths and contract.label_paths:
                        quality = QualityClass.ENGINEERING
                    if discovery_terminal:
                        quality = QualityClass.QUARANTINED
                    if in_archive and not archive_verified:
                        quality = QualityClass.QUARANTINED
                    record = _normalize_payload(
                        item,
                        source_path=relative if len(payload_items) == 1 else f"{relative}#item:{index}",
                        source_kind="json",
                        quality=quality,
                        source_sha256=file.sha256,
                        source_verified=bool(file.sha256),
                        manifest_verified=archive_verified,
                        contract=contract,
                        logical_source_file=logical_file,
                    )
                    duplicate = seen_content.get(record.content_fingerprint)
                    if duplicate:
                        record = _normalize_payload(
                            item,
                            source_path=record.source_path,
                            source_kind="json",
                            quality=quality,
                            source_sha256=file.sha256,
                            duplicate_of=duplicate,
                            source_verified=bool(file.sha256),
                            manifest_verified=archive_verified,
                            contract=contract,
                            logical_source_file=logical_file,
                        )
                    else:
                        seen_content[record.content_fingerprint] = record.record_id
                    records.append(record)
            elif path.suffix.lower() in {".md", ".txt", ".log"}:
                # Text evidence is retained as a summary-only record.  It is
                # never parsed for labels, so prose cannot become training data.
                try:
                    raw_text, _ = read_stable_bytes(
                        path,
                        limits=self.limits,
                        expected_identity=(file.st_dev, file.st_ino, file.size_bytes)
                        if file.st_dev is not None and file.st_ino is not None
                        else None,
                        expected_sha256=file.sha256 or None,
                    )
                    text = raw_text.decode("utf-8")
                    payload = {"summary_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "bytes": len(text.encode("utf-8"))}
                    text_quality = _quality_for(relative, payload, trusted_source=not in_archive or archive_verified)
                    if in_archive and not archive_verified:
                        text_quality = QualityClass.QUARANTINED
                    if discovery_terminal:
                        text_quality = QualityClass.QUARANTINED
                    if text_quality not in {QualityClass.FORMAL, QualityClass.ENGINEERING, QualityClass.EXPLORATORY}:
                        text_quality = QualityClass.LEGACY_SUMMARY
                    record = _normalize_payload(payload, source_path=relative, source_kind="text", quality=text_quality, source_sha256=file.sha256, source_verified=bool(file.sha256), manifest_verified=archive_verified)
                    records.append(record)
                except (AdapterError, OSError, UnicodeError) as exc:
                    issues.append(CorpusIssue(relative, "text_read_error", str(exc), terminal=True))
                    records.append(
                        _normalize_payload(
                            {"source": relative, "read_error": str(exc)},
                            source_path=relative,
                            source_kind="text",
                            quality=QualityClass.QUARANTINED,
                            source_sha256=file.sha256,
                            source_verified=False,
                            manifest_verified=False,
                        )
                    )
        if sink is not None:
            accept_many = getattr(sink, "accept_many", None)
            if not callable(accept_many):
                raise TypeError("RecordSink must provide transactional accept_many(records)")
            accept_many(tuple(records))
        return CorpusInventory(
            root=str(self.project_root),
            files=files,
            records=tuple(records),
            issues=tuple(issues),
            manifest=manifest_result,
            sqlite=tuple(sqlite_inventory),
        )

    build_inventory = audit

    scan = audit


def normalize_record(
    payload: Mapping[str, Any],
    *,
    source_path: str = "<memory>",
    source_kind: str = "json",
    quality: QualityClass | str = QualityClass.EXPLORATORY,
    source_sha256: str = "",
    source_verified: bool | None = None,
    manifest_verified: bool = False,
    contract: EvidenceContract | None = None,
    logical_source_file: str | None = None,
) -> NormalizedRecord:
    """Public constructor for tests and future adapters."""

    return _normalize_payload(
        payload,
        source_path=source_path,
        source_kind=source_kind,
        quality=QualityClass(quality),
        source_sha256=source_sha256,
        source_verified=source_verified,
        manifest_verified=manifest_verified,
        contract=contract,
        logical_source_file=logical_source_file,
    )


class ManifestVerifier:
    """Small state-free facade around :func:`verify_archive_manifest`."""

    def __init__(self, manifest_path: str | os.PathLike[str], *, limits: DiscoveryLimits | None = None):
        self.manifest_path = manifest_path
        self.limits = limits

    def verify(self) -> ManifestVerification:
        return verify_archive_manifest(self.manifest_path, limits=self.limits)


classify_quality = _quality_for


__all__ = [
    "CorpusAuditor",
    "CorpusInventory",
    "CorpusIssue",
    "B27D_CONTRACT",
    "EVIDENCE_CONTRACTS",
    "EvidenceContract",
    "ManifestVerification",
    "ManifestVerifier",
    "NormalizedRecord",
    "QUALITY_CLASSES",
    "QualityClass",
    "Q2_PROFILES_CONTRACT",
    "RecordSink",
    "classify_quality",
    "normalize_record",
    "verify_archive_manifest",
]
