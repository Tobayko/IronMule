"""Exact, append-only SQLite v1 evidence storage for H0.1.

The module is stdlib-only.  It deliberately has no default side effect: callers must
provide a concrete path, and merely importing it never creates the production database.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .canonical import (
    MAX_CANONICAL_BYTES,
    CanonicalError,
    bounded_text,
    canonical_json_bytes,
    exact_int64,
    exact_keys,
    int64,
    nonnegative_int64,
    positive_int64,
    strict_json_loads,
)
from .protocol import ProtocolError, validate_result
from .study import StudyError, validate_study_result

STORAGE_SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x48303131
DEFAULT_H01_DB_PATH = Path(".friday-data/h01.sqlite3")
MAX_DASHBOARD_ROWS = 200
MAX_PERSIST_BATCH = 200

ENTITY_KINDS = frozenset(
    {"paced_session", "paced_study", "legacy_h0_warmup_observation"}
)
ENTITY_STATUSES = {
    "paced_session": frozenset({"h01_session_complete", "h01_invalid"}),
    "paced_study": frozenset(
        {"h01_stationarity_supported", "h01_complete_unresolved", "h01_invalid"}
    ),
    "legacy_h0_warmup_observation": frozenset({"legacy_observation"}),
}
ACTION = "no_h0_conclusion"

_ENTITY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MIGRATION_PATH = Path(__file__).with_name("migrations") / "0001_initial.sql"

_BUNDLE_BODY_KEYS = frozenset(
    {
        "schema_version",
        "entity_id",
        "entity_kind",
        "status",
        "action",
        "created_at_unix_ns",
        "manifest_sha256",
        "trace_sha256",
        "result_sha256",
        "lineage_sha256",
        "manifest",
        "trace",
        "result",
        "lineage",
    }
)
_BUNDLE_KEYS = _BUNDLE_BODY_KEYS | {"bundle_sha256"}
_LEGACY_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "entity_id",
        "source_phase",
        "source_run_id",
        "source_mode",
        "source_status",
        "source_classification",
        "source_created_at_unix_ns",
        "observation_kind",
        "adapter",
        "registry_schema_version",
        "registry_sha256",
        "descriptor_sha256",
        "selector_sha256",
        "parser_id",
        "raw_warmup_sha256",
    }
)
_LEGACY_TRACE_KEYS = frozenset({"schema_version", "observation"})
_LEGACY_OBSERVATION_KEYS = frozenset(
    {
        "adapter",
        "source_status",
        "source_classification",
        "source_error_code",
        "warmup_ns",
        "statistics",
        "source_diagnostic",
        "registry_schema_version",
        "registry_sha256",
        "descriptor_sha256",
        "selector_sha256",
        "parser_id",
        "raw_warmup_sha256",
    }
)
_LEGACY_STATISTIC_KEYS = frozenset(
    {"count", "median_ns", "mad_ns", "iqr_ns", "last5_ns", "min_ns", "max_ns"}
)
_RATIONAL_KEYS = frozenset({"numerator", "denominator"})
_LEGACY_ADAPTERS = frozenset(
    {
        "completed_eager_warmup_v1",
        "warmup_unstable_diagnostic_v1",
        "completed_eager_warmup_w1v3",
    }
)
_LEGACY_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "conclusion",
        "interpretation",
        "action",
        "stationarity_supported",
        "paced_gate_applicable",
        "h0_reclassification",
        "promotion_applicable",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "parent_phase",
        "parent_run_id",
        "parent_manifest_sha256",
        "parent_result_sha256",
        "parent_bundle_sha256",
    }
)
_LEGACY_SOURCE_KEYS = _SOURCE_KEYS | frozenset(
    {
        "parent_evidence_sha256",
        "parent_code_sha256",
        "parent_spec_sha256",
        "parent_environment_sha256",
        "source_database_sha256",
        "registry_sha256",
        "descriptor_sha256",
        "selector_sha256",
        "raw_warmup_sha256",
    }
)
_STUDY_MANIFEST_KEYS = frozenset({"session_records"})
_STUDY_TRACE_KEYS = frozenset({"session_bindings"})

_ROW_COLUMNS = (
    "entity_id",
    "entity_kind",
    "status",
    "action",
    "created_at_unix_ns",
    "manifest_sha256",
    "trace_sha256",
    "result_sha256",
    "lineage_sha256",
    "bundle_sha256",
    "manifest_json",
    "trace_json",
    "result_json",
    "lineage_json",
    "bundle_json",
)
_SELECT_COLUMNS = ", ".join(_ROW_COLUMNS)
_PERSIST_ARGUMENT_KEYS = frozenset(
    {
        "entity_id",
        "entity_kind",
        "status",
        "created_at_unix_ns",
        "manifest",
        "trace",
        "result",
        "lineage",
    }
)


class StorageError(RuntimeError):
    """Base class for fail-closed H0.1 storage failures."""


class SchemaError(StorageError):
    """The SQLite file does not have the exact registered v1 schema."""


class BundleError(StorageError):
    """An evidence bundle violates canonical or semantic constraints."""


class StorageConflict(StorageError):
    """An existing entity ID is bound to different bytes."""


class ReadOnlyStorageError(StorageError):
    """A mutation was requested through a read-only handle."""


@dataclass(frozen=True)
class PersistenceOutcome:
    entity_id: str
    bundle_sha256: str
    state: str


@dataclass(frozen=True)
class _FileObjectIdentity:
    device: int
    inode: int
    uid: int
    mode: int


@dataclass(frozen=True)
class _PreparedPath:
    path: Path
    parent: Path
    parent_identity: _FileObjectIdentity
    file_identity: _FileObjectIdentity | None


@dataclass(frozen=True)
class _DatabaseBinding:
    path: Path
    parent: Path
    parent_identity: _FileObjectIdentity
    file_identity: _FileObjectIdentity


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any) -> tuple[str, str]:
    payload = canonical_json_bytes(value)
    return payload.decode("utf-8"), _sha256_bytes(payload)


def _canonical_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BundleError(f"{name} must be a canonical JSON object")
    try:
        payload = canonical_json_bytes(value)
        copied = strict_json_loads(payload)
    except CanonicalError as exc:
        raise BundleError(f"{name} is not bounded canonical JSON: {exc}") from exc
    if not isinstance(copied, dict):  # Defensive: Mapping was already checked above.
        raise BundleError(f"{name} must be a JSON object")
    return copied


def validate_entity_id(value: Any, name: str = "entity_id") -> str:
    try:
        text = bounded_text(value, name, maximum=160)
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if _ENTITY_ID_RE.fullmatch(text) is None:
        raise BundleError(f"{name} contains characters outside the registered ID alphabet")
    return text


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BundleError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_lineage(
    value: Mapping[str, Any], name: str = "lineage", *, legacy: bool = False
) -> dict[str, Any]:
    try:
        lineage = dict(exact_keys(value, _LEGACY_SOURCE_KEYS if legacy else _SOURCE_KEYS, name))
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if lineage["parent_phase"] != "H0":
        raise BundleError(f"{name}.parent_phase must be H0")
    validate_entity_id(lineage["parent_run_id"], f"{name}.parent_run_id")
    digest_fields = (
        _LEGACY_SOURCE_KEYS - {"parent_phase", "parent_run_id"}
        if legacy
        else _SOURCE_KEYS - {"parent_phase", "parent_run_id"}
    )
    for field in digest_fields:
        _sha256(lineage[field], f"{name}.{field}")
    return lineage


def _median_fraction(values: Sequence[Fraction]) -> Fraction:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _quantile_fraction(values: Sequence[Fraction], numerator: int) -> Fraction:
    ordered = sorted(values)
    position = Fraction((len(ordered) - 1) * numerator, 4)
    low = position.numerator // position.denominator
    high = low if position.denominator == 1 else low + 1
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def _rational(value: Fraction, name: str) -> dict[str, int]:
    try:
        numerator = exact_int64(value.numerator, f"{name}.numerator", value.numerator)
        denominator = exact_int64(value.denominator, f"{name}.denominator", value.denominator)
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if denominator <= 0 or gcd(numerator, denominator) != 1:
        raise BundleError(f"{name} must be a reduced rational with positive denominator")
    return {"numerator": numerator, "denominator": denominator}


def legacy_warmup_statistics(values: Any) -> dict[str, Any]:
    """Return the exact preregistered descriptive statistics for a warmup sequence."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise BundleError("legacy warmup_ns must be an array")
    if not 8 <= len(values) <= 16:
        raise BundleError("legacy warmup_ns must contain between 8 and 16 values")
    checked: list[int] = []
    for index, value in enumerate(values):
        try:
            checked.append(positive_int64(value, f"legacy warmup_ns[{index}]"))
        except CanonicalError as exc:
            raise BundleError(str(exc)) from exc
    fractions = [Fraction(value) for value in checked]
    median = _median_fraction(fractions)
    mad = _median_fraction([abs(value - median) for value in fractions])
    iqr = _quantile_fraction(fractions, 3) - _quantile_fraction(fractions, 1)
    return {
        "count": len(checked),
        "median_ns": _rational(median, "legacy statistics.median_ns"),
        "mad_ns": _rational(mad, "legacy statistics.mad_ns"),
        "iqr_ns": _rational(iqr, "legacy statistics.iqr_ns"),
        "last5_ns": checked[-5:],
        "min_ns": min(checked),
        "max_ns": max(checked),
    }


def _legacy_descriptor(adapter: Any) -> tuple[Any, Any]:
    """Resolve one adapter from the immutable import registry lazily.

    ``import_h0`` imports this module, so the lookup must happen only during
    bundle validation, after both modules have finished initializing.
    """

    if not isinstance(adapter, str) or adapter not in _LEGACY_ADAPTERS:
        raise BundleError("legacy adapter is not registered")
    try:
        from . import import_h0

        descriptor = next(
            descriptor
            for descriptor in import_h0.STATIC_ADAPTER_REGISTRY.descriptors
            if descriptor.adapter_id == adapter
        )
        return import_h0.STATIC_ADAPTER_REGISTRY, descriptor
    except (AttributeError, ImportError, StopIteration) as exc:
        raise BundleError("legacy adapter registry is unavailable") from exc


def _validate_legacy_diagnostic(value: Any, warmup_ns: list[int]) -> None:
    try:
        diagnostic = exact_keys(
            value, frozenset({"schema_version", "code", "details"}), "legacy source diagnostic"
        )
        exact_int64(
            diagnostic["schema_version"], "legacy source diagnostic.schema_version", 1
        )
        details = diagnostic["details"]
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if diagnostic["code"] != "warmup_unstable":
        raise BundleError("legacy source diagnostic code must be warmup_unstable")
    try:
        detail_map = exact_keys(details, frozenset({"warmups_ns"}), "legacy diagnostic details")
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if detail_map["warmups_ns"] != warmup_ns:
        raise BundleError("legacy diagnostic gate values do not mirror warmup_ns")


def _validate_common_result(result: Mapping[str, Any], status: str) -> None:
    if result.get("status") != status:
        raise BundleError("result.status does not mirror the row status")
    if result.get("action") != ACTION:
        raise BundleError("result.action must be no_h0_conclusion")
    if result.get("h0_reclassification") is not False:
        raise BundleError("result may not reclassify H0")
    if result.get("promotion_applicable") is not False:
        raise BundleError("result may not authorize promotion")


def _validate_entity_payload(
    entity_id: str,
    entity_kind: str,
    status: str,
    manifest: Mapping[str, Any],
    trace: Mapping[str, Any],
    result: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> None:
    if entity_kind not in ENTITY_KINDS:
        raise BundleError("entity_kind is not registered")
    if status not in ENTITY_STATUSES[entity_kind]:
        raise BundleError("status is not registered for entity_kind")
    _validate_common_result(result, status)
    checked_lineage = _validate_lineage(
        lineage, legacy=entity_kind == "legacy_h0_warmup_observation"
    )

    if entity_kind == "paced_session":
        if manifest.get("run_id") != entity_id or result.get("run_id") != entity_id:
            raise BundleError("paced_session entity_id must equal manifest/result run_id")
        try:
            validate_result(result, manifest, trace)
        except (CanonicalError, ProtocolError, ValueError, TypeError) as exc:
            raise BundleError(f"paced_session replay failed: {exc}") from exc
        source = manifest.get("source")
        if not isinstance(source, Mapping) or canonical_json_bytes(source) != canonical_json_bytes(
            checked_lineage
        ):
            raise BundleError("paced_session lineage does not match manifest.source")
        return

    if entity_kind == "paced_study":
        try:
            study_manifest = exact_keys(manifest, _STUDY_MANIFEST_KEYS, "study manifest")
            study_trace = exact_keys(trace, _STUDY_TRACE_KEYS, "study trace")
        except CanonicalError as exc:
            raise BundleError(str(exc)) from exc
        records = study_manifest["session_records"]
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
            raise BundleError("study manifest.session_records must be an array")
        try:
            validate_study_result(result, records)
        except (CanonicalError, ProtocolError, StudyError, ValueError, TypeError) as exc:
            raise BundleError(f"paced_study replay failed: {exc}") from exc
        if result.get("study_id") != entity_id:
            raise BundleError("paced_study entity_id must equal result.study_id")
        if canonical_json_bytes(study_trace["session_bindings"]) != canonical_json_bytes(
            result.get("session_bindings")
        ):
            raise BundleError("study trace does not mirror result.session_bindings")
        shared = result.get("shared_provenance")
        source = shared.get("source") if isinstance(shared, Mapping) else None
        if not isinstance(source, Mapping) or canonical_json_bytes(source) != canonical_json_bytes(
            checked_lineage
        ):
            raise BundleError("paced_study lineage does not match shared provenance")
        return

    try:
        legacy_manifest = exact_keys(manifest, _LEGACY_MANIFEST_KEYS, "legacy manifest")
        legacy_trace = exact_keys(trace, _LEGACY_TRACE_KEYS, "legacy trace")
        legacy_result = exact_keys(result, _LEGACY_RESULT_KEYS, "legacy result")
        exact_int64(legacy_manifest["schema_version"], "legacy manifest.schema_version", 1)
        exact_int64(legacy_trace["schema_version"], "legacy trace.schema_version", 1)
        exact_int64(legacy_result["schema_version"], "legacy result.schema_version", 1)
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if legacy_manifest["entity_id"] != entity_id:
        raise BundleError("legacy manifest.entity_id does not mirror entity_id")
    if legacy_manifest["source_phase"] != "H0":
        raise BundleError("legacy source_phase must be H0")
    if legacy_manifest["observation_kind"] != "warmup_observation":
        raise BundleError("legacy observation_kind must be warmup_observation")
    if legacy_manifest["source_mode"] != "eager_baseline":
        raise BundleError("legacy source_mode must be eager_baseline")
    validate_entity_id(legacy_manifest["source_run_id"], "legacy manifest.source_run_id")
    try:
        nonnegative_int64(
            legacy_manifest["source_created_at_unix_ns"],
            "legacy manifest.source_created_at_unix_ns",
        )
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if legacy_manifest["source_run_id"] != checked_lineage["parent_run_id"]:
        raise BundleError("legacy source run does not match parent lineage")
    adapter = legacy_manifest["adapter"]
    registry, descriptor = _legacy_descriptor(adapter)
    try:
        observation = exact_keys(
            legacy_trace["observation"], _LEGACY_OBSERVATION_KEYS, "legacy observation"
        )
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if observation["adapter"] != adapter:
        raise BundleError("legacy observation adapter does not mirror manifest")
    try:
        exact_int64(
            legacy_manifest["registry_schema_version"],
            "legacy manifest.registry_schema_version",
            1,
        )
        exact_int64(
            observation["registry_schema_version"],
            "legacy observation.registry_schema_version",
            1,
        )
        for field in (
            "registry_sha256",
            "descriptor_sha256",
            "selector_sha256",
            "raw_warmup_sha256",
        ):
            _sha256(legacy_manifest[field], f"legacy manifest.{field}")
            _sha256(observation[field], f"legacy observation.{field}")
        bounded_text(legacy_manifest["parser_id"], "legacy manifest.parser_id", maximum=160)
        bounded_text(observation["parser_id"], "legacy observation.parser_id", maximum=160)
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    expected_registry_sha = registry.registry_sha256
    expected_descriptor_sha = descriptor.descriptor_sha256
    expected_selector_sha = _sha256_bytes(canonical_json_bytes(descriptor.selector))
    expected_parser_id = descriptor.parser_id
    binding_fields = (
        ("registry_sha256", expected_registry_sha),
        ("descriptor_sha256", expected_descriptor_sha),
        ("selector_sha256", expected_selector_sha),
        ("parser_id", expected_parser_id),
    )
    for field, expected in binding_fields:
        if legacy_manifest[field] != expected or observation[field] != expected:
            raise BundleError(f"legacy {field} does not mirror the registered adapter")
    warmup = observation["warmup_ns"]
    statistics = legacy_warmup_statistics(warmup)
    expected_raw_warmup_sha = _sha256_bytes(canonical_json_bytes(warmup))
    if legacy_manifest["raw_warmup_sha256"] != expected_raw_warmup_sha:
        raise BundleError("legacy manifest.raw_warmup_sha256 does not replay from warmup_ns")
    if observation["raw_warmup_sha256"] != expected_raw_warmup_sha:
        raise BundleError("legacy observation.raw_warmup_sha256 does not replay from warmup_ns")
    for field in (
        "registry_sha256",
        "descriptor_sha256",
        "selector_sha256",
        "raw_warmup_sha256",
    ):
        if checked_lineage[field] != legacy_manifest[field]:
            raise BundleError(f"legacy lineage.{field} does not mirror manifest")
    try:
        declared_statistics = exact_keys(
            observation["statistics"], _LEGACY_STATISTIC_KEYS, "legacy statistics"
        )
        for name in ("median_ns", "mad_ns", "iqr_ns"):
            rational = exact_keys(
                declared_statistics[name], _RATIONAL_KEYS, f"legacy statistics.{name}"
            )
            numerator = int64(rational["numerator"], f"legacy statistics.{name}.numerator")
            denominator = positive_int64(
                rational["denominator"], f"legacy statistics.{name}.denominator"
            )
            if gcd(numerator, denominator) != 1:
                raise BundleError(f"legacy statistics.{name} is not reduced")
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    if canonical_json_bytes(declared_statistics) != canonical_json_bytes(statistics):
        raise BundleError("legacy statistics do not replay from warmup_ns")
    if adapter == "warmup_unstable_diagnostic_v1":
        expected_source = ("invalid", "invalid", "warmup_unstable")
        if len(warmup) != 16:
            raise BundleError("invalid legacy adapter requires exactly 16 warmups")
        _validate_legacy_diagnostic(observation["source_diagnostic"], list(warmup))
    else:
        expected_source = ("completed", "measurement_complete", None)
        expected_count = 11 if adapter == "completed_eager_warmup_v1" else 8
        if len(warmup) != expected_count or observation["source_diagnostic"] is not None:
            raise BundleError("completed legacy adapter has adapter-exact warmup evidence")
    actual_source = (
        legacy_manifest["source_status"],
        legacy_manifest["source_classification"],
        observation["source_error_code"],
    )
    if actual_source != expected_source:
        raise BundleError("legacy manifest source state is not registered for adapter")
    if (
        observation["source_status"] != legacy_manifest["source_status"]
        or observation["source_classification"] != legacy_manifest["source_classification"]
    ):
        raise BundleError("legacy observation source state does not mirror manifest")
    if legacy_result["conclusion"] != "historical_warmup_observation_only":
        raise BundleError("legacy conclusion is not registered")
    if legacy_result["interpretation"] != "descriptive_only":
        raise BundleError("legacy interpretation must be descriptive_only")
    if legacy_result["stationarity_supported"] is not False:
        raise BundleError("legacy result may not support stationarity")
    if legacy_result["paced_gate_applicable"] is not False:
        raise BundleError("legacy result may not satisfy a paced gate")


def build_bundle(
    *,
    entity_id: Any,
    entity_kind: Any,
    status: Any,
    created_at_unix_ns: Any,
    manifest: Any,
    trace: Any,
    result: Any,
    lineage: Any,
) -> dict[str, Any]:
    """Validate and materialize one detached canonical evidence bundle."""

    checked_id = validate_entity_id(entity_id)
    if not isinstance(entity_kind, str) or entity_kind not in ENTITY_KINDS:
        raise BundleError("entity_kind is not registered")
    if not isinstance(status, str) or status not in ENTITY_STATUSES[entity_kind]:
        raise BundleError("status is not registered for entity_kind")
    try:
        checked_created = nonnegative_int64(created_at_unix_ns, "created_at_unix_ns")
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc

    checked_manifest = _canonical_copy(manifest, "manifest")
    checked_trace = _canonical_copy(trace, "trace")
    checked_result = _canonical_copy(result, "result")
    checked_lineage = _canonical_copy(lineage, "lineage")
    _validate_entity_payload(
        checked_id,
        entity_kind,
        status,
        checked_manifest,
        checked_trace,
        checked_result,
        checked_lineage,
    )

    _, manifest_hash = _digest(checked_manifest)
    _, trace_hash = _digest(checked_trace)
    _, result_hash = _digest(checked_result)
    _, lineage_hash = _digest(checked_lineage)
    body = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "entity_id": checked_id,
        "entity_kind": entity_kind,
        "status": status,
        "action": ACTION,
        "created_at_unix_ns": checked_created,
        "manifest_sha256": manifest_hash,
        "trace_sha256": trace_hash,
        "result_sha256": result_hash,
        "lineage_sha256": lineage_hash,
        "manifest": checked_manifest,
        "trace": checked_trace,
        "result": checked_result,
        "lineage": checked_lineage,
    }
    try:
        body_bytes = canonical_json_bytes(body, maximum=MAX_CANONICAL_BYTES)
        bundle = {**body, "bundle_sha256": _sha256_bytes(body_bytes)}
        # The final bytes, rather than only the preimage, carry the hard 1 MiB cap.
        return strict_json_loads(canonical_json_bytes(bundle, maximum=MAX_CANONICAL_BYTES))
    except CanonicalError as exc:
        raise BundleError(f"bundle exceeds the canonical contract: {exc}") from exc


def _prepare_persistence_item(
    value: Mapping[str, Any], *, index: int
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BundleError(f"persistence batch item {index} must be an object with string keys")
    if set(value) != _PERSIST_ARGUMENT_KEYS:
        raise BundleError(f"persistence batch item {index} has unknown or missing fields")
    bundle = build_bundle(
        entity_id=value["entity_id"],
        entity_kind=value["entity_kind"],
        status=value["status"],
        created_at_unix_ns=value["created_at_unix_ns"],
        manifest=value["manifest"],
        trace=value["trace"],
        result=value["result"],
        lineage=value["lineage"],
    )
    manifest_json, _ = _digest(bundle["manifest"])
    trace_json, _ = _digest(bundle["trace"])
    result_json, _ = _digest(bundle["result"])
    lineage_json, _ = _digest(bundle["lineage"])
    try:
        bundle_json = canonical_json_bytes(bundle).decode("utf-8")
    except CanonicalError as exc:  # build_bundle already checked; defensive boundary.
        raise BundleError(str(exc)) from exc
    return (
        bundle,
        (
            bundle["entity_id"],
            bundle["entity_kind"],
            bundle["status"],
            bundle["action"],
            bundle["created_at_unix_ns"],
            bundle["manifest_sha256"],
            bundle["trace_sha256"],
            bundle["result_sha256"],
            bundle["lineage_sha256"],
            bundle["bundle_sha256"],
            manifest_json,
            trace_json,
            result_json,
            lineage_json,
            bundle_json,
        ),
    )


def _migration_sql() -> str:
    try:
        return _MIGRATION_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SchemaError(f"cannot read registered migration: {exc}") from exc


def _normalize_ddl(value: Any) -> str:
    return " ".join(str(value or "").split())


def _master_snapshot(connection: sqlite3.Connection) -> list[list[Any]]:
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    return [[row[0], row[1], row[2], _normalize_ddl(row[3])] for row in rows]


def _pragma_rows(connection: sqlite3.Connection, pragma: str, name: str) -> list[list[Any]]:
    if _ENTITY_ID_RE.fullmatch(name) is None:
        raise SchemaError("registered schema object has an unsafe identifier")
    rows = connection.execute(f'PRAGMA {pragma}("{name}")').fetchall()
    return [list(row) for row in rows]


def _schema_snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
    master = _master_snapshot(connection)
    tables = sorted(row[1] for row in master if row[0] == "table")
    indexes_by_table: dict[str, list[list[Any]]] = {}
    index_xinfo: dict[str, list[list[Any]]] = {}
    table_xinfo: dict[str, list[list[Any]]] = {}
    for table in tables:
        table_xinfo[table] = _pragma_rows(connection, "table_xinfo", table)
        index_rows = _pragma_rows(connection, "index_list", table)
        indexes_by_table[table] = index_rows
        for row in index_rows:
            index_name = row[1]
            if not isinstance(index_name, str) or len(index_name) > 160:
                raise SchemaError("schema index name is not bounded text")
            # SQLite-generated autoindex names use only this same safe alphabet.
            if _ENTITY_ID_RE.fullmatch(index_name) is None:
                raise SchemaError("schema index name has an unsafe identifier")
            index_xinfo[index_name] = _pragma_rows(connection, "index_xinfo", index_name)
    return {
        "application_id": connection.execute("PRAGMA application_id").fetchone()[0],
        "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
        "master": master,
        "table_xinfo": table_xinfo,
        "index_list": indexes_by_table,
        "index_xinfo": index_xinfo,
    }


def _expected_schema_snapshot() -> dict[str, Any]:
    expected = sqlite3.connect(":memory:", isolation_level=None)
    try:
        expected.executescript(_migration_sql())
        return _schema_snapshot(expected)
    except sqlite3.DatabaseError as exc:
        raise SchemaError(f"registered migration is invalid: {exc}") from exc
    finally:
        expected.close()


def _verify_schema(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        integrity_rows = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity_rows != [("ok",)]:
            raise SchemaError("SQLite integrity_check did not return exact ok")
        actual = _schema_snapshot(connection)
        expected = _expected_schema_snapshot()
    except sqlite3.DatabaseError as exc:
        raise SchemaError(f"cannot inspect SQLite schema: {exc}") from exc
    if actual != expected:
        raise SchemaError("SQLite schema differs from the exact registered v1 schema")
    return actual


def _object_identity(
    path: Path, *, expected: str, missing_ok: bool = False
) -> _FileObjectIdentity | None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise StorageError(f"required {expected} path does not exist") from None
    except OSError as exc:
        raise StorageError(f"cannot inspect {expected} path: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise StorageError(f"{expected} path may not be a symlink")
    if expected == "database" and not stat.S_ISREG(metadata.st_mode):
        raise StorageError("database path must be a regular file")
    if expected == "database parent" and not stat.S_ISDIR(metadata.st_mode):
        raise StorageError("database parent path must be a directory")
    return _FileObjectIdentity(
        device=int(metadata.st_dev),
        inode=int(metadata.st_ino),
        uid=int(getattr(metadata, "st_uid", -1)),
        mode=int(metadata.st_mode),
    )


def _secure_new_database_file(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise StorageError("new database path is not a regular file")
            os.fchmod(descriptor, 0o600)
            after = os.fstat(descriptor)
            if stat.S_IMODE(after.st_mode) != 0o600:
                raise StorageError("new database permissions are not exact 0600")
        finally:
            os.close(descriptor)
    except StorageError:
        raise
    except OSError as exc:
        raise StorageError(f"cannot secure new database file: {exc}") from exc


def _verify_private_database_mode(
    identity: _FileObjectIdentity, *, read_only: bool
) -> None:
    permissions = stat.S_IMODE(identity.mode)
    if permissions & 0o077:
        raise StorageError("database permissions expose group or other access")
    if not permissions & 0o400:
        raise StorageError("database owner read permission is required")
    if not read_only and not permissions & 0o200:
        raise StorageError("writable database requires owner write permission")


def _prepare_path(
    source: os.PathLike[str] | str, *, read_only: bool
) -> _PreparedPath:
    try:
        raw = os.fspath(source)
    except TypeError as exc:
        raise StorageError("database source must be a filesystem path") from exc
    if not isinstance(raw, str) or not raw or len(raw) > 4096 or "\x00" in raw:
        raise StorageError("database source must be bounded non-empty path text")
    if raw.lower().startswith("file:"):
        raise StorageError("caller-provided SQLite URIs are forbidden")
    path = Path(os.path.abspath(raw))
    parent = path.parent
    if not read_only:
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"cannot create database parent: {exc}") from exc
    parent_identity = _object_identity(parent, expected="database parent")
    file_identity = _object_identity(path, expected="database", missing_ok=not read_only)
    if read_only and file_identity is None:
        raise StorageError("read-only database source must be an existing regular file")
    if parent_identity is None:  # Unreachable but keeps the type boundary explicit.
        raise StorageError("database parent identity is unavailable")
    return _PreparedPath(path, parent, parent_identity, file_identity)


def _connect_database(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{quote(str(path), safe='/')}?mode=ro"
        return sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5.0)
    return sqlite3.connect(str(path), isolation_level=None, timeout=5.0)


def _database_list_identity(connection: sqlite3.Connection) -> _FileObjectIdentity:
    try:
        rows = connection.execute("PRAGMA database_list").fetchall()
    except sqlite3.DatabaseError as exc:
        raise StorageError(f"cannot inspect SQLite database_list: {exc}") from exc
    main_rows = [row for row in rows if row[1] == "main"]
    unexpected = [
        row
        for row in rows
        if row[1] != "main" and not (row[1] == "temp" and row[2] == "")
    ]
    if len(main_rows) != 1 or unexpected:
        raise StorageError("SQLite connection has an unexpected attached database")
    database_name = main_rows[0][2]
    if not isinstance(database_name, str) or not database_name or "\x00" in database_name:
        raise StorageError("SQLite main database path is unavailable")
    try:
        resolved = Path(database_name).absolute().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StorageError(f"cannot resolve SQLite main database path: {exc}") from exc
    identity = _object_identity(resolved, expected="database")
    if identity is None:  # Unreachable: missing_ok is false.
        raise StorageError("SQLite main database identity is unavailable")
    return identity


def _establish_binding(
    prepared: _PreparedPath, connection: sqlite3.Connection
) -> _DatabaseBinding:
    parent_identity = _object_identity(prepared.parent, expected="database parent")
    file_identity = _object_identity(prepared.path, expected="database")
    if parent_identity != prepared.parent_identity:
        raise StorageError("database parent identity changed while opening SQLite")
    if prepared.file_identity is not None and file_identity != prepared.file_identity:
        raise StorageError("database file identity changed while opening SQLite")
    if file_identity is None or _database_list_identity(connection) != file_identity:
        raise StorageError("SQLite main database does not match the requested file identity")
    return _DatabaseBinding(
        prepared.path,
        prepared.parent,
        parent_identity,
        file_identity,
    )


def _verify_file_binding(
    binding: _DatabaseBinding, connection: sqlite3.Connection
) -> None:
    if _object_identity(binding.parent, expected="database parent") != binding.parent_identity:
        raise StorageError("database parent identity changed after connection")
    if _object_identity(binding.path, expected="database") != binding.file_identity:
        raise StorageError("database file identity changed after connection")
    if _database_list_identity(connection) != binding.file_identity:
        raise StorageError("SQLite main database identity changed after connection")


def _verify_after_begin(
    binding: _DatabaseBinding, connection: sqlite3.Connection
) -> None:
    """Bind transaction checks to the same file object captured during open."""

    _verify_file_binding(binding, connection)


def _configure_connection(connection: sqlite3.Connection, *, read_only: bool) -> None:
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA recursive_triggers = ON")
    if read_only:
        connection.execute("PRAGMA query_only = ON")
    _verify_connection_profile(connection, read_only=read_only)


def _verify_connection_profile(
    connection: sqlite3.Connection, *, read_only: bool
) -> None:
    if connection.execute("PRAGMA recursive_triggers").fetchone() != (1,):
        raise StorageError("SQLite recursive_triggers did not become exact 1")
    expected_query_only = (1,) if read_only else (0,)
    if connection.execute("PRAGMA query_only").fetchone() != expected_query_only:
        raise ReadOnlyStorageError("SQLite query_only profile is not exact")


def _verified_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = connection.execute(
            f"SELECT rowid, {_SELECT_COLUMNS} FROM bundles ORDER BY rowid"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise BundleError(f"cannot read complete evidence rows: {exc}") from exc
    verified: list[dict[str, Any]] = []
    for raw in rows:
        try:
            rowid = nonnegative_int64(raw[0], "bundle rowid")
        except CanonicalError as exc:
            raise BundleError(str(exc)) from exc
        verified.append({"rowid": rowid, "bundle": _verify_row(raw[1:])})
    return verified


class Storage:
    """One verified SQLite handle; use as a context manager."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        binding: _DatabaseBinding,
        *,
        read_only: bool,
    ) -> None:
        self._connection = connection
        self._binding = binding
        self.path = binding.path
        self.read_only = read_only
        self._closed = False

    @classmethod
    def open(
        cls, source: os.PathLike[str] | str, *, read_only: bool = False
    ) -> "Storage":
        prepared = _prepare_path(source, read_only=read_only)
        connection = _connect_database(prepared.path, read_only=read_only)
        try:
            if not read_only and prepared.file_identity is None:
                _secure_new_database_file(prepared.path)
            _configure_connection(connection, read_only=read_only)
            binding = _establish_binding(prepared, connection)
            _verify_private_database_mode(binding.file_identity, read_only=read_only)
            # Only a path absent at pre-open may receive the registered initial schema.
            # Existing empty or incompatible files are never synthesized or repaired.
            if not read_only and prepared.file_identity is None:
                connection.executescript(_migration_sql())
                _verify_file_binding(binding, connection)

            connection.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")
            try:
                _verify_after_begin(binding, connection)
                _verify_connection_profile(connection, read_only=read_only)
                _verify_schema(connection)
                if not read_only:
                    _verified_rows(connection)
                _verify_file_binding(binding, connection)
                connection.execute("COMMIT")
            except Exception:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            return cls(connection, binding, read_only=read_only)
        except Exception:
            connection.close()
            raise

    def __enter__(self) -> "Storage":
        if self._closed:
            raise StorageError("storage handle is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("storage handle is closed")

    @contextmanager
    def read_transaction(self) -> Any:
        """Yield the exact schema from one file-bound, verified read snapshot."""

        self._ensure_open()
        if not self.read_only:
            raise ReadOnlyStorageError("read transaction requires a read-only handle")
        if self._connection.in_transaction:
            raise StorageError("nested read transactions are forbidden")
        self._connection.execute("BEGIN")
        try:
            _verify_after_begin(self._binding, self._connection)
            _verify_connection_profile(self._connection, read_only=True)
            schema = _verify_schema(self._connection)
            yield schema
            _verify_file_binding(self._binding, self._connection)
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def verify_schema(self) -> dict[str, Any]:
        self._ensure_open()
        if self._connection.in_transaction:
            _verify_file_binding(self._binding, self._connection)
            _verify_connection_profile(self._connection, read_only=self.read_only)
            return _verify_schema(self._connection)
        self._connection.execute("BEGIN" if self.read_only else "BEGIN IMMEDIATE")
        try:
            _verify_after_begin(self._binding, self._connection)
            _verify_connection_profile(self._connection, read_only=self.read_only)
            schema = _verify_schema(self._connection)
            _verify_file_binding(self._binding, self._connection)
            self._connection.execute("COMMIT")
            return schema
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def persist_bundle(
        self,
        *,
        entity_id: Any,
        entity_kind: Any,
        status: Any,
        created_at_unix_ns: Any,
        manifest: Any,
        trace: Any,
        result: Any,
        lineage: Any,
    ) -> PersistenceOutcome:
        return self.persist_bundles(
            (
                {
                    "entity_id": entity_id,
                    "entity_kind": entity_kind,
                    "status": status,
                    "created_at_unix_ns": created_at_unix_ns,
                    "manifest": manifest,
                    "trace": trace,
                    "result": result,
                    "lineage": lineage,
                },
            )
        )[0]

    def persist_bundles(
        self, values: Sequence[Mapping[str, Any]]
    ) -> tuple[PersistenceOutcome, ...]:
        """Persist one bounded batch atomically after rebuilding every bundle."""

        self._ensure_open()
        if self.read_only:
            raise ReadOnlyStorageError("cannot persist through read-only storage")
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise BundleError("persistence batch must be a sequence of bundle arguments")
        if len(values) > MAX_PERSIST_BATCH:
            raise BundleError("persistence batch exceeds the registered maximum")
        prepared = [
            _prepare_persistence_item(value, index=index)
            for index, value in enumerate(values)
        ]
        entity_ids = [bundle["entity_id"] for bundle, _row_values in prepared]
        if len(entity_ids) != len(set(entity_ids)):
            raise BundleError("persistence batch contains a duplicate entity_id")
        if not prepared:
            return ()

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            _verify_after_begin(self._binding, self._connection)
            _verify_connection_profile(self._connection, read_only=False)
            _verify_schema(self._connection)
            verified = _verified_rows(self._connection)
            existing_by_id = {
                row["bundle"]["entity_id"]: row["bundle"] for row in verified
            }
            for bundle, _row_values in prepared:
                existing = existing_by_id.get(bundle["entity_id"])
                if existing is not None and canonical_json_bytes(existing) != canonical_json_bytes(
                    bundle
                ):
                    raise StorageConflict("entity_id is already bound to different bytes")

            outcomes: list[PersistenceOutcome] = []
            for bundle, row_values in prepared:
                if bundle["entity_id"] in existing_by_id:
                    outcomes.append(
                        PersistenceOutcome(
                            bundle["entity_id"], bundle["bundle_sha256"], "idempotent"
                        )
                    )
                    continue
                self._connection.execute(
                    f"INSERT INTO bundles ({_SELECT_COLUMNS}) "
                    f"VALUES ({','.join('?' for _ in row_values)})",
                    row_values,
                )
                inserted = self._connection.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM bundles WHERE entity_id = ?",
                    (bundle["entity_id"],),
                ).fetchone()
                if inserted is None or canonical_json_bytes(
                    _verify_row(inserted)
                ) != canonical_json_bytes(bundle):
                    raise BundleError("inserted bundle failed in-transaction replay")
                existing_by_id[bundle["entity_id"]] = bundle
                outcomes.append(
                    PersistenceOutcome(
                        bundle["entity_id"], bundle["bundle_sha256"], "inserted"
                    )
                )
            _verify_file_binding(self._binding, self._connection)
            self._connection.execute("COMMIT")
            return tuple(outcomes)
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def _row(self, entity_id: Any) -> tuple[Any, ...] | None:
        self._ensure_open()
        checked_id = validate_entity_id(entity_id)
        return self._connection.execute(
            f"SELECT {_SELECT_COLUMNS} FROM bundles WHERE entity_id = ?", (checked_id,)
        ).fetchone()

    def get_verified_bundle(self, entity_id: Any) -> dict[str, Any] | None:
        row = self._row(entity_id)
        return None if row is None else _verify_row(row)

    def verified_rows(self) -> list[dict[str, Any]]:
        self._ensure_open()
        if not self._connection.in_transaction:
            raise StorageError("verified_rows requires an explicit transaction")
        return _verified_rows(self._connection)

    def count(self) -> int:
        self._ensure_open()
        value = self._connection.execute("SELECT COUNT(*) FROM bundles").fetchone()[0]
        try:
            return nonnegative_int64(value, "bundle count")
        except CanonicalError as exc:
            raise BundleError(str(exc)) from exc

    def counts_by(self, column: str) -> dict[str, int]:
        self._ensure_open()
        if column not in {"entity_kind", "status"}:
            raise StorageError("count grouping is not registered")
        rows = self._connection.execute(
            f"SELECT {column}, COUNT(*) FROM bundles GROUP BY {column} ORDER BY {column}"
        ).fetchall()
        result: dict[str, int] = {}
        for name, count in rows:
            if not isinstance(name, str):
                raise BundleError("stored grouping value is not text")
            try:
                result[name] = nonnegative_int64(count, f"count for {name}")
            except CanonicalError as exc:
                raise BundleError(str(exc)) from exc
        return result

    def recent(self, limit: Any) -> list[dict[str, Any]]:
        self._ensure_open()
        try:
            checked_limit = nonnegative_int64(limit, "recent limit", maximum=MAX_DASHBOARD_ROWS)
        except CanonicalError as exc:
            raise StorageError(str(exc)) from exc
        rows = self._connection.execute(
            "SELECT entity_id, entity_kind, status, created_at_unix_ns, bundle_sha256 "
            "FROM bundles ORDER BY created_at_unix_ns DESC, entity_id DESC LIMIT ?",
            (checked_limit,),
        ).fetchall()
        return [
            {
                "entity_id": row[0],
                "entity_kind": row[1],
                "status": row[2],
                "created_at_unix_ns": row[3],
                "bundle_sha256": row[4],
            }
            for row in rows
        ]

    def revision_material(
        self,
        verified_rows: Sequence[Mapping[str, Any]],
        schema_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_open()
        if not self._connection.in_transaction:
            raise StorageError("revision material requires an explicit transaction")
        identities: list[dict[str, Any]] = []
        previous_rowid = 0
        for index, row in enumerate(verified_rows):
            if not isinstance(row, Mapping) or set(row) != {"rowid", "bundle"}:
                raise BundleError("verified revision row has an unexpected shape")
            try:
                rowid = nonnegative_int64(row["rowid"], f"revision row {index}.rowid")
            except CanonicalError as exc:
                raise BundleError(str(exc)) from exc
            if rowid <= previous_rowid:
                raise BundleError("revision rows are not in strict rowid order")
            previous_rowid = rowid
            bundle = row["bundle"]
            if not isinstance(bundle, Mapping):
                raise BundleError("verified revision bundle is not an object")
            identities.append(
                {
                    "rowid": rowid,
                    "entity_id": bundle["entity_id"],
                    "bundle_sha256": bundle["bundle_sha256"],
                    "created_at_unix_ns": bundle["created_at_unix_ns"],
                }
            )
        return {
            "schema_sha256": _sha256_bytes(canonical_json_bytes(schema_snapshot)),
            "content_identities": identities,
        }


def _decode_canonical_json(value: Any, name: str) -> Any:
    if not isinstance(value, str):
        raise BundleError(f"stored {name} is not text")
    try:
        payload = value.encode("utf-8", errors="strict")
        decoded = strict_json_loads(payload)
        if canonical_json_bytes(decoded).decode("utf-8") != value:
            raise BundleError(f"stored {name} is not byte-canonical JSON")
        return decoded
    except (CanonicalError, UnicodeError) as exc:
        raise BundleError(f"stored {name} is invalid: {exc}") from exc


def _verify_row(row: Sequence[Any]) -> dict[str, Any]:
    if len(row) != len(_ROW_COLUMNS):
        raise BundleError("stored row has an unexpected column count")
    stored = dict(zip(_ROW_COLUMNS, row, strict=True))
    manifest = _decode_canonical_json(stored["manifest_json"], "manifest_json")
    trace = _decode_canonical_json(stored["trace_json"], "trace_json")
    result = _decode_canonical_json(stored["result_json"], "result_json")
    lineage = _decode_canonical_json(stored["lineage_json"], "lineage_json")
    bundle = _decode_canonical_json(stored["bundle_json"], "bundle_json")
    try:
        exact_keys(bundle, _BUNDLE_KEYS, "bundle")
        exact_int64(bundle["schema_version"], "bundle.schema_version", STORAGE_SCHEMA_VERSION)
    except CanonicalError as exc:
        raise BundleError(str(exc)) from exc
    for field, value in (
        ("entity_id", stored["entity_id"]),
        ("entity_kind", stored["entity_kind"]),
        ("status", stored["status"]),
        ("action", stored["action"]),
        ("created_at_unix_ns", stored["created_at_unix_ns"]),
        ("manifest_sha256", stored["manifest_sha256"]),
        ("trace_sha256", stored["trace_sha256"]),
        ("result_sha256", stored["result_sha256"]),
        ("lineage_sha256", stored["lineage_sha256"]),
        ("bundle_sha256", stored["bundle_sha256"]),
    ):
        if bundle.get(field) != value or type(bundle.get(field)) is not type(value):
            raise BundleError(f"stored {field} does not exactly mirror bundle JSON")
    for field, value in (
        ("manifest", manifest),
        ("trace", trace),
        ("result", result),
        ("lineage", lineage),
    ):
        if canonical_json_bytes(bundle.get(field)) != canonical_json_bytes(value):
            raise BundleError(f"stored {field} JSON does not mirror bundle JSON")
        expected_hash = _sha256_bytes(canonical_json_bytes(value))
        if stored[f"{field}_sha256"] != expected_hash:
            raise BundleError(f"stored {field} hash failed replay")
    body = dict(exact_keys(bundle, _BUNDLE_KEYS, "bundle"))
    bundle_hash = body.pop("bundle_sha256")
    if _sha256_bytes(canonical_json_bytes(body)) != bundle_hash:
        raise BundleError("stored bundle hash failed replay")
    replayed = build_bundle(
        entity_id=stored["entity_id"],
        entity_kind=stored["entity_kind"],
        status=stored["status"],
        created_at_unix_ns=stored["created_at_unix_ns"],
        manifest=manifest,
        trace=trace,
        result=result,
        lineage=lineage,
    )
    if canonical_json_bytes(replayed) != canonical_json_bytes(bundle):
        raise BundleError("stored bundle differs from semantic replay")
    return bundle


__all__ = [
    "ACTION",
    "BundleError",
    "DEFAULT_H01_DB_PATH",
    "ENTITY_KINDS",
    "ENTITY_STATUSES",
    "MAX_DASHBOARD_ROWS",
    "PersistenceOutcome",
    "ReadOnlyStorageError",
    "SQLITE_APPLICATION_ID",
    "STORAGE_SCHEMA_VERSION",
    "SchemaError",
    "Storage",
    "StorageConflict",
    "StorageError",
    "build_bundle",
    "validate_entity_id",
]
