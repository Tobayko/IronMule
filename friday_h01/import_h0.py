"""Read-only H0 inventory and exact, generation-bound legacy adapters.

Every candidate is replayed through the public H0 bundle verifier before the
value-independent registry selector is computed.  A matched descriptor then
selects one closed parser; no current-generation H0 normalizer or generic
warmup fallback participates in historical extraction.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from friday_h0.canonical import canonical_sha256 as h0_canonical_sha256
from friday_h0.protocol import (
    PRODUCTION_MANIFEST_BYTES,
    ProtocolError,
    close_manifest,
    parse_capped_json,
)
from friday_h0.storage import (
    PERSISTENCE_MAX_BUNDLE_BYTES,
    Storage as H0Storage,
    StorageError as H0StorageError,
)

from .canonical import (
    MAX_DEPTH,
    MAX_NODES,
    MAX_SEQUENCE,
    MAX_STRING,
    CanonicalError,
    bounded_text,
    canonical_json_bytes,
    canonical_sha256,
    exact_int64,
    exact_keys,
    finite_number,
    int64,
    nonnegative_int64,
    positive_int64,
)
from .storage import (
    BundleError,
    PersistenceOutcome,
    Storage,
    StorageError,
    build_bundle,
    legacy_warmup_statistics,
)

INVENTORY_SCHEMA = "friday_h01.h0_generation_inventory.v1"
REPORT_SCHEMA = "friday_h01.legacy_h0_import_report.v2"
SELECTION_RULE = "all_h0_v1_eager_baseline_common_result_ordered_created_at_run_id"
STRUCTURAL_FINGERPRINT_ALGORITHM = "sha256_recursive_json_structure_v1"
SCHEMA_TAG_ALGORITHM = "recursive_schema_version_paths_v1"
ADAPTER_DESCRIPTOR_SCHEMA_VERSION = 1
ADAPTER_REGISTRY_SCHEMA_VERSION = 1
ENTITY_BINDING_SCHEMA_VERSION = 1
LEGACY_MANIFEST_SCHEMA_VERSION = 1
LEGACY_TRACE_SCHEMA_VERSION = 1
LEGACY_RESULT_SCHEMA_VERSION = 1

RUNTIME_UNAVAILABLE_ADAPTER = "no_warmup_runtime_unavailable_v1"
COMPLETED_ADAPTER = "completed_eager_warmup_v1"
INVALID_ADAPTER = "warmup_unstable_diagnostic_v1"
W1V3_COMPLETED_ADAPTER = "completed_eager_warmup_w1v3"

OUTCOME_RECOGNIZED_EXCLUSION = "recognized_exclusion"
OUTCOME_LEGACY_OBSERVATION = "legacy_observation"
ADAPTER_OUTCOMES = frozenset(
    {OUTCOME_RECOGNIZED_EXCLUSION, OUTCOME_LEGACY_OBSERVATION}
)
PARSER_RUNTIME_UNAVAILABLE = "parse_no_warmup_runtime_unavailable_v1"
PARSER_COMPLETED_V1 = "parse_completed_eager_warmup_v1"
PARSER_WARMUP_UNSTABLE_V1 = "parse_warmup_unstable_diagnostic_v1"
PARSER_COMPLETED_W1V3 = "parse_completed_eager_warmup_w1v3"

MATCHED = "matched"
UNSUPPORTED_GENERATION = "unsupported_generation"
CLAIMED_KNOWN_MALFORMED = "claimed_known_malformed"
MATCH_STATES = frozenset({MATCHED, UNSUPPORTED_GENERATION, CLAIMED_KNOWN_MALFORMED})
RECOGNIZED_NO_WARMUP = "recognized_no_warmup_runtime_unavailable"
EXCLUSION_REASONS = frozenset({UNSUPPORTED_GENERATION, RECOGNIZED_NO_WARMUP})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}")
_SOURCE_STATUSES = frozenset({"completed", "invalid", "timeout", "worker_exit"})
_SELECTOR_KEYS = frozenset(
    {
        "schema_version",
        "structural_fingerprint_algorithm",
        "schema_tag_algorithm",
        "source_status",
        "parent_code_sha256",
        "parent_spec_sha256",
        "parent_environment_sha256",
        "result_structure_sha256",
        "result_schema_tags_sha256",
        "diagnostic_present",
        "diagnostic_structure_sha256",
        "diagnostic_schema_tags_sha256",
    }
)


class LegacyImportError(RuntimeError):
    """The H0 inventory or a claimed registered generation failed closed."""


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    uid: int
    mode: int
    size: int

    def value(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "uid": self.uid,
            "mode": self.mode,
            "size": self.size,
        }


@dataclass(frozen=True)
class _VerifiedCandidate:
    selection_index: int
    created_at_unix_ns: int
    closed_manifest: Any
    result: dict[str, Any]
    bundle: dict[str, Any]
    payload: dict[str, Any]
    code_sha256: str
    spec_sha256: str
    environment_sha256: str
    manifest_sha256: str
    common_result_payload_sha256: str


@dataclass(frozen=True)
class LegacyImportOutcome:
    """Canonical compatibility plan plus exact optional persistence outcomes."""

    report: dict[str, Any]
    bundles: tuple[dict[str, Any], ...]
    persistence: tuple[PersistenceOutcome, ...]


def _exact_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise LegacyImportError(f"{name} must be a bounded registered identifier")
    return value


def _exact_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LegacyImportError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _selector_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LegacyImportError("adapter selector must be an object with string keys")
    if set(value) != _SELECTOR_KEYS:
        raise LegacyImportError("adapter selector has unknown or missing keys")
    try:
        schema_version = int64(
            value["schema_version"],
            "adapter selector.schema_version",
            minimum=ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            maximum=ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
        )
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc
    if value["structural_fingerprint_algorithm"] != STRUCTURAL_FINGERPRINT_ALGORITHM:
        raise LegacyImportError("adapter selector uses an unknown structural algorithm")
    if value["schema_tag_algorithm"] != SCHEMA_TAG_ALGORITHM:
        raise LegacyImportError("adapter selector uses an unknown schema-tag algorithm")
    source_status = value["source_status"]
    if not isinstance(source_status, str) or source_status not in _SOURCE_STATUSES:
        raise LegacyImportError("adapter selector source_status is not registered")
    result = {
        "schema_version": schema_version,
        "structural_fingerprint_algorithm": STRUCTURAL_FINGERPRINT_ALGORITHM,
        "schema_tag_algorithm": SCHEMA_TAG_ALGORITHM,
        "source_status": source_status,
        "parent_code_sha256": _exact_sha256(
            value["parent_code_sha256"], "adapter selector.parent_code_sha256"
        ),
        "parent_spec_sha256": _exact_sha256(
            value["parent_spec_sha256"], "adapter selector.parent_spec_sha256"
        ),
        "parent_environment_sha256": _exact_sha256(
            value["parent_environment_sha256"],
            "adapter selector.parent_environment_sha256",
        ),
        "result_structure_sha256": _exact_sha256(
            value["result_structure_sha256"],
            "adapter selector.result_structure_sha256",
        ),
        "result_schema_tags_sha256": _exact_sha256(
            value["result_schema_tags_sha256"],
            "adapter selector.result_schema_tags_sha256",
        ),
        "diagnostic_present": value["diagnostic_present"],
        "diagnostic_structure_sha256": value["diagnostic_structure_sha256"],
        "diagnostic_schema_tags_sha256": value["diagnostic_schema_tags_sha256"],
    }
    if not isinstance(result["diagnostic_present"], bool):
        raise LegacyImportError("adapter selector.diagnostic_present must be bool")
    diagnostic_hashes = (
        result["diagnostic_structure_sha256"],
        result["diagnostic_schema_tags_sha256"],
    )
    if result["diagnostic_present"]:
        result["diagnostic_structure_sha256"] = _exact_sha256(
            diagnostic_hashes[0], "adapter selector.diagnostic_structure_sha256"
        )
        result["diagnostic_schema_tags_sha256"] = _exact_sha256(
            diagnostic_hashes[1], "adapter selector.diagnostic_schema_tags_sha256"
        )
    elif diagnostic_hashes != (None, None):
        raise LegacyImportError("absent diagnostic selector hashes must both be null")
    canonical_json_bytes(result)
    return result


@dataclass(frozen=True)
class AdapterDescriptor:
    """Immutable, versioned declaration for one exact historical generation."""

    adapter_id: str
    registry_schema_version: int
    outcome: str
    parser_id: str
    source_status: str
    parent_code_sha256: str
    parent_spec_sha256: str
    parent_environment_sha256: str
    result_structure_sha256: str
    result_schema_tags_sha256: str
    diagnostic_structure_sha256: str | None
    diagnostic_schema_tags_sha256: str | None

    def __post_init__(self) -> None:
        try:
            int64(
                self.registry_schema_version,
                "adapter registry_schema_version",
                minimum=ADAPTER_REGISTRY_SCHEMA_VERSION,
                maximum=ADAPTER_REGISTRY_SCHEMA_VERSION,
            )
        except CanonicalError as exc:
            raise LegacyImportError(str(exc)) from exc
        _exact_identifier(self.adapter_id, "adapter_id")
        if self.outcome not in ADAPTER_OUTCOMES:
            raise LegacyImportError("adapter outcome is not registered")
        _exact_identifier(self.parser_id, "parser_id")
        _selector_value(self.selector)

    @property
    def selector(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "structural_fingerprint_algorithm": STRUCTURAL_FINGERPRINT_ALGORITHM,
            "schema_tag_algorithm": SCHEMA_TAG_ALGORITHM,
            "source_status": self.source_status,
            "parent_code_sha256": self.parent_code_sha256,
            "parent_spec_sha256": self.parent_spec_sha256,
            "parent_environment_sha256": self.parent_environment_sha256,
            "result_structure_sha256": self.result_structure_sha256,
            "result_schema_tags_sha256": self.result_schema_tags_sha256,
            "diagnostic_present": self.diagnostic_structure_sha256 is not None,
            "diagnostic_structure_sha256": self.diagnostic_structure_sha256,
            "diagnostic_schema_tags_sha256": self.diagnostic_schema_tags_sha256,
        }

    @property
    def value(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "registry_schema_version": self.registry_schema_version,
            "adapter_id": self.adapter_id,
            "outcome": self.outcome,
            "parser_id": self.parser_id,
            "selector": self.selector,
        }

    @property
    def descriptor_sha256(self) -> str:
        return canonical_sha256(self.value)

    @classmethod
    def from_selector(
        cls,
        *,
        adapter_id: str,
        outcome: str,
        parser_id: str,
        selector: Mapping[str, Any],
    ) -> "AdapterDescriptor":
        checked = _selector_value(selector)
        return cls(
            adapter_id=adapter_id,
            registry_schema_version=ADAPTER_REGISTRY_SCHEMA_VERSION,
            outcome=outcome,
            parser_id=parser_id,
            source_status=checked["source_status"],
            parent_code_sha256=checked["parent_code_sha256"],
            parent_spec_sha256=checked["parent_spec_sha256"],
            parent_environment_sha256=checked["parent_environment_sha256"],
            result_structure_sha256=checked["result_structure_sha256"],
            result_schema_tags_sha256=checked["result_schema_tags_sha256"],
            diagnostic_structure_sha256=checked["diagnostic_structure_sha256"],
            diagnostic_schema_tags_sha256=checked["diagnostic_schema_tags_sha256"],
        )


@dataclass(frozen=True)
class AdapterMatch:
    state: Literal["matched", "unsupported_generation", "claimed_known_malformed"]
    adapter_id: str | None
    adapter_descriptor_sha256: str | None

    def value(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "adapter_id": self.adapter_id,
            "adapter_descriptor_sha256": self.adapter_descriptor_sha256,
        }


class AdapterRegistry:
    """Canonical immutable descriptor set with provenance-first matching."""

    def __init__(self, descriptors: tuple[AdapterDescriptor, ...]) -> None:
        if not isinstance(descriptors, tuple) or any(
            not isinstance(item, AdapterDescriptor) for item in descriptors
        ):
            raise LegacyImportError("adapter registry requires an exact descriptor tuple")
        ordered = tuple(sorted(descriptors, key=lambda item: item.adapter_id))
        if len({item.adapter_id for item in ordered}) != len(ordered):
            raise LegacyImportError("adapter registry contains duplicate adapter IDs")
        selectors = [canonical_sha256(item.selector) for item in ordered]
        if len(set(selectors)) != len(selectors):
            raise LegacyImportError("adapter registry contains duplicate selectors")
        self._descriptors = ordered
        self._value = {
            "schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
            "descriptors": [item.value for item in ordered],
        }
        self._sha256 = canonical_sha256(self._value)

    @property
    def descriptors(self) -> tuple[AdapterDescriptor, ...]:
        return self._descriptors

    @property
    def value(self) -> dict[str, Any]:
        return parse_capped_json(
            canonical_json_bytes(self._value), limit=PERSISTENCE_MAX_BUNDLE_BYTES
        )

    @property
    def registry_sha256(self) -> str:
        return self._sha256

    def descriptor(self, adapter_id: str, descriptor_sha256: str) -> AdapterDescriptor:
        checked_id = _exact_identifier(adapter_id, "adapter_id")
        checked_sha = _exact_sha256(descriptor_sha256, "adapter_descriptor_sha256")
        for descriptor in self._descriptors:
            if descriptor.adapter_id == checked_id:
                if descriptor.descriptor_sha256 != checked_sha:
                    raise LegacyImportError("matched descriptor hash is not the registry value")
                return descriptor
        raise LegacyImportError("matched adapter is absent from the registry")

    def match(self, selector: Mapping[str, Any]) -> AdapterMatch:
        checked = _selector_value(selector)
        provenance = (
            checked["parent_code_sha256"],
            checked["parent_spec_sha256"],
            checked["parent_environment_sha256"],
        )
        claimed = [
            descriptor
            for descriptor in self._descriptors
            if (
                descriptor.parent_code_sha256,
                descriptor.parent_spec_sha256,
                descriptor.parent_environment_sha256,
            )
            == provenance
        ]
        exact = [descriptor for descriptor in claimed if descriptor.selector == checked]
        if len(exact) == 1:
            descriptor = exact[0]
            return AdapterMatch(MATCHED, descriptor.adapter_id, descriptor.descriptor_sha256)
        if len(exact) > 1:
            raise LegacyImportError("adapter registry matched more than one descriptor")
        if claimed:
            return AdapterMatch(CLAIMED_KNOWN_MALFORMED, None, None)
        return AdapterMatch(UNSUPPORTED_GENERATION, None, None)


def _frozen_descriptor(
    *,
    adapter_id: str,
    outcome: str,
    parser_id: str,
    source_status: str,
    code_sha256: str,
    spec_sha256: str,
    environment_sha256: str,
    result_structure_sha256: str,
    result_schema_tags_sha256: str,
    diagnostic_structure_sha256: str | None = None,
    diagnostic_schema_tags_sha256: str | None = None,
) -> AdapterDescriptor:
    return AdapterDescriptor.from_selector(
        adapter_id=adapter_id,
        outcome=outcome,
        parser_id=parser_id,
        selector={
            "schema_version": ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "structural_fingerprint_algorithm": STRUCTURAL_FINGERPRINT_ALGORITHM,
            "schema_tag_algorithm": SCHEMA_TAG_ALGORITHM,
            "source_status": source_status,
            "parent_code_sha256": code_sha256,
            "parent_spec_sha256": spec_sha256,
            "parent_environment_sha256": environment_sha256,
            "result_structure_sha256": result_structure_sha256,
            "result_schema_tags_sha256": result_schema_tags_sha256,
            "diagnostic_present": diagnostic_structure_sha256 is not None,
            "diagnostic_structure_sha256": diagnostic_structure_sha256,
            "diagnostic_schema_tags_sha256": diagnostic_schema_tags_sha256,
        },
    )


STATIC_ADAPTER_REGISTRY = AdapterRegistry(
    (
        _frozen_descriptor(
            adapter_id=RUNTIME_UNAVAILABLE_ADAPTER,
            outcome=OUTCOME_RECOGNIZED_EXCLUSION,
            parser_id=PARSER_RUNTIME_UNAVAILABLE,
            source_status="invalid",
            code_sha256="246eb77ff4917122e54f5184ccb2cca174c079fd69e2c892d61a40f240fb333b",
            spec_sha256="a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac",
            environment_sha256="74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782",
            result_structure_sha256="5bd47782958e02d186b8daf166364e3c39d2588e63928750ba614e2a91164ecf",
            result_schema_tags_sha256="fa89b7dc8cacb04073b17d697f8dd10b4bb09a5b1dbdc4ea266cd4919761051f",
        ),
        _frozen_descriptor(
            adapter_id=COMPLETED_ADAPTER,
            outcome=OUTCOME_LEGACY_OBSERVATION,
            parser_id=PARSER_COMPLETED_V1,
            source_status="completed",
            code_sha256="5f62c419bac782ecc89fd5056b9070ab4789ea3b336f72c1ff7d351c5c5cc055",
            spec_sha256="a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac",
            environment_sha256="74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782",
            result_structure_sha256="397824478c7e9cbdc319387867d7d6623ac24a0e7671389fa54511fbba9fd658",
            result_schema_tags_sha256="fa89b7dc8cacb04073b17d697f8dd10b4bb09a5b1dbdc4ea266cd4919761051f",
        ),
        _frozen_descriptor(
            adapter_id=INVALID_ADAPTER,
            outcome=OUTCOME_LEGACY_OBSERVATION,
            parser_id=PARSER_WARMUP_UNSTABLE_V1,
            source_status="invalid",
            code_sha256="aae3245ee5df265ebbaa96cc3ccf7b60ec0292656e7abd79a98a6a188f3cad4c",
            spec_sha256="a713d6336f865724e4f0523760dc1abdaef7a5918313a034f375c29678b47bac",
            environment_sha256="74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782",
            result_structure_sha256="0747fd2dc5ee69a6acacca5b9188426b9957f5262853f749edbbfdab7c059857",
            result_schema_tags_sha256="46b900bf88117305d802b255556e00eea1b95e731564575faeda9dbcc148ce9b",
            diagnostic_structure_sha256="6d76f95584ab9c75d1bef8bd4686e418075dd0877beebf06c94400b7eb4c999a",
            diagnostic_schema_tags_sha256="fa89b7dc8cacb04073b17d697f8dd10b4bb09a5b1dbdc4ea266cd4919761051f",
        ),
        _frozen_descriptor(
            adapter_id=W1V3_COMPLETED_ADAPTER,
            outcome=OUTCOME_LEGACY_OBSERVATION,
            parser_id=PARSER_COMPLETED_W1V3,
            source_status="completed",
            code_sha256="101cdadfd1311bde541c65a91b59025e5aac7550055919e15bd267eb67cb68dc",
            spec_sha256="b53b112f97d12dacadaeb22b442bf321f7595fb376fc53a9855e149df9265851",
            environment_sha256="74ca2dac9550330e905bcbe0a96def9e4fdf2945eca10d342c341a2f5c08e782",
            result_structure_sha256="ef39c352ebf66e16da9c2e331cbf528444e2e8cfea871d04cb020ea23809a249",
            result_schema_tags_sha256="fa89b7dc8cacb04073b17d697f8dd10b4bb09a5b1dbdc4ea266cd4919761051f",
        ),
    )
)


def _identity(path: Path, name: str) -> _Identity:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise LegacyImportError(f"cannot stat {name}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise LegacyImportError(f"{name} must not be a symlink")
    expected = stat.S_ISDIR if name == "source database parent" else stat.S_ISREG
    if not expected(info.st_mode):
        raise LegacyImportError(f"{name} has the wrong file type")
    return _Identity(
        info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode), info.st_size
    )


def _source_path(value: os.PathLike[str] | str) -> tuple[Path, _Identity, _Identity]:
    if isinstance(value, str) and (value.startswith("file:") or "\x00" in value):
        raise LegacyImportError("source must be a filesystem path, not a URI or NUL path")
    try:
        path = Path(value).absolute()
    except (TypeError, ValueError, OSError) as exc:
        raise LegacyImportError("source database path is invalid") from exc
    return (
        path,
        _identity(path.parent, "source database parent"),
        _identity(path, "source database"),
    )


def _execute_target_path(
    value: os.PathLike[str] | str,
    *,
    source_path: Path,
    source_identity: _Identity,
) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise LegacyImportError("execute target must be a filesystem path") from exc
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw) > 4096
        or "\x00" in raw
        or raw.lower().startswith("file:")
    ):
        raise LegacyImportError("execute target must be bounded non-URI path text")
    try:
        path = Path(raw).absolute()
        parent_info = os.lstat(path.parent)
    except OSError as exc:
        raise LegacyImportError(f"cannot inspect execute target parent: {exc}") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise LegacyImportError("execute target parent must be a non-symlink directory")
    if path == source_path:
        raise LegacyImportError("execute target must differ from the H0 source")
    try:
        target_info = os.lstat(path)
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise LegacyImportError(f"cannot inspect execute target: {exc}") from exc
    if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISREG(target_info.st_mode):
        raise LegacyImportError("execute target must be absent or a non-symlink regular file")
    if (
        target_info.st_dev == source_identity.device
        and target_info.st_ino == source_identity.inode
    ):
        raise LegacyImportError("execute target aliases the H0 source file")
    return path


def _hash_source(path: Path, expected: _Identity) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            info = os.fstat(handle.fileno())
            actual = _Identity(
                info.st_dev,
                info.st_ino,
                info.st_uid,
                stat.S_IMODE(info.st_mode),
                info.st_size,
            )
            if actual != expected or not stat.S_ISREG(info.st_mode):
                raise LegacyImportError("source database identity changed before hashing")
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            final = os.fstat(handle.fileno())
            after = _Identity(
                final.st_dev,
                final.st_ino,
                final.st_uid,
                stat.S_IMODE(final.st_mode),
                final.st_size,
            )
            if after != expected:
                raise LegacyImportError("source database identity changed while hashing")
    except LegacyImportError:
        raise
    except OSError as exc:
        raise LegacyImportError(f"cannot hash source database: {exc}") from exc
    return digest.hexdigest()


def _verify_source_witness(
    path: Path,
    *,
    expected_parent: _Identity,
    expected_file: _Identity,
    expected_sha256: str,
) -> None:
    actual_path, actual_parent, actual_file = _source_path(path)
    same_parent_object = (
        actual_parent.device,
        actual_parent.inode,
        actual_parent.uid,
        actual_parent.mode,
    ) == (
        expected_parent.device,
        expected_parent.inode,
        expected_parent.uid,
        expected_parent.mode,
    )
    if (
        actual_path != path
        or not same_parent_object
        or actual_file != expected_file
        or _hash_source(path, expected_file) != expected_sha256
    ):
        raise LegacyImportError("source database identity or SHA-256 changed at execute boundary")


def _database_list_identity(connection: sqlite3.Connection) -> _Identity:
    rows = connection.execute("PRAGMA database_list").fetchall()
    main = [row for row in rows if row[1] == "main"]
    if len(main) != 1 or not isinstance(main[0][2], str) or not main[0][2]:
        raise LegacyImportError("SQLite did not expose one bounded main database path")
    return _identity(Path(main[0][2]), "source database")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LegacyImportError(f"{name} must be an object with string keys")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise LegacyImportError(f"{name} must be a JSON array")
    return value


def _parse_stored_json(value: Any, *, name: str, limit: int) -> Any:
    if not isinstance(value, str):
        raise LegacyImportError(f"{name} must be stored UTF-8 JSON text")
    try:
        return parse_capped_json(value.encode("utf-8", errors="strict"), limit=limit)
    except (UnicodeError, ProtocolError, ValueError, TypeError) as exc:
        raise LegacyImportError(f"{name} is not bounded strict JSON") from exc


def _verified_candidate(
    storage: H0Storage, *, selection_index: int, row: sqlite3.Row
) -> _VerifiedCandidate:
    run_id = row["run_id"]
    if not isinstance(run_id, str):
        raise LegacyImportError("selected H0 run_id is not text")
    try:
        created_at = nonnegative_int64(row["created_at_unix_ns"], f"{run_id} created_at")
    except CanonicalError as exc:
        raise LegacyImportError(f"{run_id} creation time is invalid") from exc
    manifest_value = _parse_stored_json(
        row["manifest_json"], name=f"{run_id} manifest", limit=PRODUCTION_MANIFEST_BYTES
    )
    try:
        closed = close_manifest(_mapping(manifest_value, f"{run_id} manifest"))
    except (ProtocolError, TypeError, ValueError) as exc:
        raise LegacyImportError(f"{run_id} manifest replay failed") from exc
    if closed.run_id != run_id or closed.mode != "eager_baseline":
        raise LegacyImportError(f"{run_id} selection mirrors do not match its manifest")
    payload_value = _parse_stored_json(
        row["payload_json"],
        name=f"{run_id} common_result payload",
        limit=PERSISTENCE_MAX_BUNDLE_BYTES,
    )
    payload = _mapping(payload_value, f"{run_id} common_result payload")
    bundle = _mapping(payload.get("bundle"), f"{run_id} evidence bundle")
    result = _mapping(bundle.get("result"), f"{run_id} result")
    child_names = ("raw_samples", "scalar_metrics", "correctness_metrics", "artifacts")
    children = {
        name: _array(bundle.get(name), f"{run_id} {name}") for name in child_names
    }
    try:
        verified = storage.verify_common_result_bundle(
            closed,
            result,
            raw_samples=children["raw_samples"],
            scalar_metrics=children["scalar_metrics"],
            correctness_metrics=children["correctness_metrics"],
            artifacts=children["artifacts"],
        )
    except (H0StorageError, ProtocolError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise LegacyImportError(f"{run_id} full H0 bundle replay failed") from exc
    if verified != "verified":
        raise LegacyImportError(f"{run_id} H0 verifier returned an unknown state")
    return _VerifiedCandidate(
        selection_index=selection_index,
        created_at_unix_ns=created_at,
        closed_manifest=closed,
        result=dict(result),
        bundle=dict(bundle),
        payload=dict(payload),
        code_sha256=_exact_sha256(row["code_sha256"], f"{run_id} stored code_sha256"),
        spec_sha256=_exact_sha256(row["spec_sha256"], f"{run_id} stored spec_sha256"),
        environment_sha256=_exact_sha256(
            row["environment_sha256"], f"{run_id} stored environment_sha256"
        ),
        manifest_sha256=_exact_sha256(
            row["manifest_hash"], f"{run_id} stored manifest_sha256"
        ),
        common_result_payload_sha256=_exact_sha256(
            row["payload_hash"], f"{run_id} stored common-result payload_sha256"
        ),
    )


def _read_verified_candidates(
    source: os.PathLike[str] | str,
) -> tuple[Path, _Identity, str, tuple[_VerifiedCandidate, ...]]:
    path, parent_before, file_before = _source_path(source)
    source_hash = _hash_source(path, file_before)
    candidates: list[_VerifiedCandidate] = []
    caught: Exception | None = None
    storage: H0Storage | None = None
    try:
        storage = H0Storage.open(path, read_only=True)
        connection = storage.connection
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise LegacyImportError("source SQLite query_only is not exact 1")
        if _database_list_identity(connection) != file_before:
            raise LegacyImportError("SQLite main database is not the preread source file")
        connection.execute("BEGIN")
        if _database_list_identity(connection) != file_before:
            raise LegacyImportError("SQLite main database changed after BEGIN")
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if len(integrity) != 1 or integrity[0][0] != "ok":
            raise LegacyImportError("source H0 database integrity_check is not exactly ok")
        rows = connection.execute(
            "SELECT r.run_id,r.created_at_unix_ns,r.manifest_json,r.manifest_hash,"
            "r.code_sha256,r.spec_sha256,r.environment_sha256,"
            "e.payload_json,e.payload_hash "
            "FROM runs AS r JOIN status_events AS e ON e.run_id=r.run_id "
            "WHERE r.mode=? AND e.event_kind='common_result' "
            "ORDER BY r.created_at_unix_ns,r.run_id",
            ("eager_baseline",),
        ).fetchall()
        # Every storage replay completes before the first fingerprint is derived.
        for selection_index, row in enumerate(rows):
            candidates.append(
                _verified_candidate(storage, selection_index=selection_index, row=row)
            )
        connection.execute("COMMIT")
    except Exception as exc:
        caught = exc
        if storage is not None and storage.connection.in_transaction:
            storage.connection.execute("ROLLBACK")
    finally:
        if storage is not None:
            storage.close()
    try:
        parent_after = _identity(path.parent, "source database parent")
        file_after = _identity(path, "source database")
        after_hash = _hash_source(path, file_after)
        if parent_after != parent_before or file_after != file_before or after_hash != source_hash:
            raise LegacyImportError("source database identity or SHA-256 changed during inventory")
    except LegacyImportError as identity_error:
        if caught is not None:
            raise identity_error from caught
        raise
    if caught is not None:
        if isinstance(caught, LegacyImportError):
            raise caught
        raise LegacyImportError("source H0 inventory failed closed") from caught
    return path, file_before, source_hash, tuple(candidates)


def _type_class(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is dict:
        return "object"
    if type(value) is list:
        return "array"
    raise LegacyImportError(f"unsupported structural value {type(value).__name__}")


def structural_fingerprint(value: Any) -> dict[str, Any]:
    """Hash exact JSON structure while excluding every scalar value."""

    digest = hashlib.sha256()
    counters = {name: 0 for name in ("null", "bool", "int", "float", "string", "object", "array")}
    node_count = 0
    maximum_depth = 0

    def size_token(size: int) -> bytes:
        return size.to_bytes(8, byteorder="big", signed=False)

    def visit(item: Any, depth: int) -> None:
        nonlocal node_count, maximum_depth
        node_count += 1
        maximum_depth = max(maximum_depth, depth)
        if node_count > MAX_NODES or depth > MAX_DEPTH:
            raise LegacyImportError("structural fingerprint exceeds node or depth bound")
        kind = _type_class(item)
        counters[kind] += 1
        if kind == "null":
            digest.update(b"N")
            return
        if kind == "bool":
            digest.update(b"B")
            return
        if kind == "int":
            try:
                int64(item, "structural JSON integer")
            except CanonicalError as exc:
                raise LegacyImportError(str(exc)) from exc
            digest.update(b"I")
            return
        if kind == "float":
            if not math.isfinite(item):
                raise LegacyImportError("structural JSON float must be finite")
            digest.update(b"F")
            return
        if kind == "string":
            if len(item) > MAX_STRING or "\x00" in item:
                raise LegacyImportError("structural JSON string exceeds its bound or contains NUL")
            digest.update(b"S")
            return
        if len(item) > MAX_SEQUENCE:
            raise LegacyImportError("structural JSON container exceeds its registered bound")
        if kind == "array":
            digest.update(b"L")
            digest.update(size_token(len(item)))
            for child in item:
                visit(child, depth + 1)
            digest.update(b"l")
            return
        digest.update(b"O")
        digest.update(size_token(len(item)))
        if any(type(key) is not str for key in item):
            raise LegacyImportError("structural JSON object key must be exact string")
        for key in sorted(item):
            try:
                encoded = key.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise LegacyImportError("structural JSON key is not valid UTF-8") from exc
            if len(key) > MAX_STRING or "\x00" in key:
                raise LegacyImportError("structural JSON key exceeds its bound or contains NUL")
            digest.update(b"K")
            digest.update(size_token(len(encoded)))
            digest.update(encoded)
            visit(item[key], depth + 1)
        digest.update(b"o")

    visit(value, 0)
    return {
        "algorithm": STRUCTURAL_FINGERPRINT_ALGORITHM,
        "sha256": digest.hexdigest(),
        "root_kind": _type_class(value),
        "node_count": node_count,
        "max_depth": maximum_depth,
        "type_counts": counters,
    }


def _pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def declared_schema_tags(value: Any) -> list[dict[str, Any]]:
    """Return exact declared ``schema_version`` tags without other values."""

    structural_fingerprint(value)
    records: list[dict[str, Any]] = []

    def visit(item: Any, path: str) -> None:
        if type(item) is dict:
            for key in sorted(item):
                child_path = f"{path}/{_pointer_segment(key)}"
                child = item[key]
                if key == "schema_version":
                    kind = _type_class(child)
                    record: dict[str, Any] = {"path": child_path, "type_class": kind}
                    if kind in {"null", "bool", "int", "float", "string"}:
                        record["value"] = child
                    else:
                        record["structural_sha256"] = structural_fingerprint(child)["sha256"]
                    records.append(record)
                visit(child, child_path)
        elif type(item) is list:
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")

    visit(value, "")
    canonical_json_bytes(records)
    return records


def _diagnostic(result: Mapping[str, Any]) -> tuple[bool, Any]:
    evidence = result.get("evidence")
    if not isinstance(evidence, Mapping):
        return False, None
    domain = evidence.get("benchmark_evidence")
    if not isinstance(domain, Mapping) or "failure_diagnostic" not in domain:
        return False, None
    return True, domain["failure_diagnostic"]


def _candidate_selector(
    candidate: _VerifiedCandidate,
    *,
    result_fingerprint: Mapping[str, Any],
    result_schema_tags: list[dict[str, Any]],
    diagnostic_present: bool,
    diagnostic_fingerprint: Mapping[str, Any] | None,
    diagnostic_schema_tags: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return _selector_value(
        {
            "schema_version": ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
            "structural_fingerprint_algorithm": STRUCTURAL_FINGERPRINT_ALGORITHM,
            "schema_tag_algorithm": SCHEMA_TAG_ALGORITHM,
            "source_status": candidate.result["status"],
            "parent_code_sha256": candidate.code_sha256,
            "parent_spec_sha256": candidate.spec_sha256,
            "parent_environment_sha256": candidate.environment_sha256,
            "result_structure_sha256": result_fingerprint["sha256"],
            "result_schema_tags_sha256": canonical_sha256(result_schema_tags),
            "diagnostic_present": diagnostic_present,
            "diagnostic_structure_sha256": (
                diagnostic_fingerprint["sha256"] if diagnostic_fingerprint is not None else None
            ),
            "diagnostic_schema_tags_sha256": (
                canonical_sha256(diagnostic_schema_tags)
                if diagnostic_schema_tags is not None
                else None
            ),
        }
    )


def _inventory_candidate(
    candidate: _VerifiedCandidate, registry: AdapterRegistry
) -> dict[str, Any]:
    result_fingerprint = structural_fingerprint(candidate.result)
    result_tags = declared_schema_tags(candidate.result)
    diagnostic_present, diagnostic = _diagnostic(candidate.result)
    diagnostic_fingerprint = structural_fingerprint(diagnostic) if diagnostic_present else None
    diagnostic_tags = declared_schema_tags(diagnostic) if diagnostic_present else None
    selector = _candidate_selector(
        candidate,
        result_fingerprint=result_fingerprint,
        result_schema_tags=result_tags,
        diagnostic_present=diagnostic_present,
        diagnostic_fingerprint=diagnostic_fingerprint,
        diagnostic_schema_tags=diagnostic_tags,
    )
    match = registry.match(selector)
    error = candidate.result.get("error")
    return {
        "selection_index": candidate.selection_index,
        "source_run_id": candidate.closed_manifest.run_id,
        "source_created_at_unix_ns": candidate.created_at_unix_ns,
        "source_status": candidate.result["status"],
        "source_classification": candidate.result["classification"],
        "source_action": candidate.result["action"],
        "source_error_code": error.get("code") if isinstance(error, Mapping) else None,
        "stored_hashes": {
            "code_sha256": candidate.code_sha256,
            "spec_sha256": candidate.spec_sha256,
            "environment_sha256": candidate.environment_sha256,
            "manifest_sha256": candidate.manifest_sha256,
            "result_sha256": _exact_sha256(
                candidate.payload.get("result_sha256"), "stored result_sha256"
            ),
            "evidence_sha256": h0_canonical_sha256(candidate.result["evidence"]),
            "bundle_sha256": _exact_sha256(
                candidate.payload.get("bundle_sha256"), "stored bundle_sha256"
            ),
            "common_result_payload_sha256": candidate.common_result_payload_sha256,
        },
        "result_structural_fingerprint": result_fingerprint,
        "result_declared_schema_tags": result_tags,
        "diagnostic_structural_fingerprint": diagnostic_fingerprint,
        "diagnostic_declared_schema_tags": diagnostic_tags,
        "selector": selector,
        "registry_match": match.value(),
        "full_bundle_verification": "verified",
    }


def _exact_object(value: Any, keys: frozenset[str], name: str) -> Mapping[str, Any]:
    try:
        return exact_keys(value, keys, name)
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc


def _exact_source_envelope(
    candidate: _VerifiedCandidate,
    *,
    status: str,
    classification: str,
    error_code: str | None,
) -> Mapping[str, Any]:
    result = candidate.result
    try:
        exact_int64(result.get("schema_version"), "historical result.schema_version", 1)
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc
    actual = (result.get("status"), result.get("classification"), result.get("action"))
    if actual != (status, classification, "baseline_fallback"):
        raise LegacyImportError("historical result state is not registered for its adapter")
    if result.get("mode") != "eager_baseline":
        raise LegacyImportError("historical result mode must be eager_baseline")
    error = result.get("error")
    if error_code is None:
        if error is not None:
            raise LegacyImportError("completed historical result.error must be null")
    else:
        checked_error = _exact_object(
            error, frozenset({"code", "message"}), "historical result.error"
        )
        if checked_error["code"] != error_code:
            raise LegacyImportError("historical result.error code is not registered")
        try:
            bounded_text(checked_error["message"], "historical result.error.message", maximum=4096)
        except CanonicalError as exc:
            raise LegacyImportError(str(exc)) from exc
    return _mapping(result.get("evidence"), "historical result.evidence")


def _warmup_at_registered_path(candidate: _VerifiedCandidate) -> Mapping[str, Any]:
    evidence = _exact_source_envelope(
        candidate,
        status="completed",
        classification="measurement_complete",
        error_code=None,
    )
    benchmark = _mapping(evidence.get("benchmark_evidence"), "historical benchmark_evidence")
    arms = _mapping(benchmark.get("arms"), "historical benchmark_evidence.arms")
    baseline = _mapping(arms.get("baseline"), "historical baseline arm")
    return _mapping(baseline.get("warmup"), "historical baseline warmup")


def _positive_warmup_values(value: Any, *, expected: int, name: str) -> list[int]:
    raw = _array(value, name)
    if len(raw) != expected:
        raise LegacyImportError(f"{name} must contain exactly {expected} values")
    checked: list[int] = []
    for index, item in enumerate(raw):
        try:
            checked.append(positive_int64(item, f"{name}[{index}]"))
        except CanonicalError as exc:
            raise LegacyImportError(str(exc)) from exc
    return checked


def _validate_warmup_samples(value: Any, warmup_ns: list[int], *, name: str) -> None:
    samples = _array(value, name)
    if len(samples) != len(warmup_ns):
        raise LegacyImportError(f"{name} length does not mirror durations_ns")
    sample_keys = frozenset({"phase", "sample_index", "unit", "value"})
    for index, raw in enumerate(samples):
        sample = _exact_object(raw, sample_keys, f"{name}[{index}]")
        try:
            exact_int64(sample["sample_index"], f"{name}[{index}].sample_index", index)
            measured = positive_int64(sample["value"], f"{name}[{index}].value")
        except CanonicalError as exc:
            raise LegacyImportError(str(exc)) from exc
        if sample["phase"] != "warmup" or sample["unit"] != "ns":
            raise LegacyImportError(f"{name}[{index}] phase/unit is not registered")
        if measured != warmup_ns[index]:
            raise LegacyImportError(f"{name}[{index}] does not mirror durations_ns")


def _validate_warmup_median(value: Any, warmup_ns: list[int], *, name: str) -> None:
    if type(value) is not float:
        raise LegacyImportError(f"{name} must retain the historical float type")
    try:
        observed = finite_number(value, name, minimum=0.0)
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc
    registered = float(sorted(warmup_ns[-5:])[2])
    if observed != registered:
        raise LegacyImportError(f"{name} does not replay from the last five durations")


def _parse_completed_warmup(
    candidate: _VerifiedCandidate, *, with_blocks: bool
) -> dict[str, Any]:
    warmup = _warmup_at_registered_path(candidate)
    keys = frozenset({"count", "durations_ns", "median_ns", "samples", "stable"})
    expected = 11
    if with_blocks:
        keys |= {"blocks"}
        expected = 8
    checked = _exact_object(warmup, keys, "historical baseline warmup")
    try:
        exact_int64(checked["count"], "historical warmup.count", expected)
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc
    if checked["stable"] is not True:
        raise LegacyImportError("historical warmup.stable must be exact true")
    values = _positive_warmup_values(
        checked["durations_ns"], expected=expected, name="historical warmup.durations_ns"
    )
    _validate_warmup_samples(checked["samples"], values, name="historical warmup.samples")
    _validate_warmup_median(checked["median_ns"], values, name="historical warmup.median_ns")
    if with_blocks:
        blocks = _array(checked["blocks"], "historical warmup.blocks")
        if len(blocks) != expected:
            raise LegacyImportError("historical W1v3 warmup requires exactly eight blocks")
        block_keys = frozenset(
            {
                "block_index",
                "block_ns",
                "evaluations",
                "max_eval_ns",
                "median_eval_ns",
                "min_eval_ns",
                "per_eval_ns",
            }
        )
        for index, raw in enumerate(blocks):
            block = _exact_object(raw, block_keys, f"historical warmup.blocks[{index}]")
            try:
                exact_int64(
                    block["block_index"],
                    f"historical warmup.blocks[{index}].block_index",
                    index,
                )
                block_ns = positive_int64(
                    block["block_ns"], f"historical warmup.blocks[{index}].block_ns"
                )
                evaluations = positive_int64(
                    block["evaluations"],
                    f"historical warmup.blocks[{index}].evaluations",
                    maximum=4096,
                )
                maximum = positive_int64(
                    block["max_eval_ns"],
                    f"historical warmup.blocks[{index}].max_eval_ns",
                )
                median = positive_int64(
                    block["median_eval_ns"],
                    f"historical warmup.blocks[{index}].median_eval_ns",
                )
                minimum = positive_int64(
                    block["min_eval_ns"],
                    f"historical warmup.blocks[{index}].min_eval_ns",
                )
                per_eval = positive_int64(
                    block["per_eval_ns"],
                    f"historical warmup.blocks[{index}].per_eval_ns",
                )
            except CanonicalError as exc:
                raise LegacyImportError(str(exc)) from exc
            if block_ns < 50_000_000 or not minimum <= median <= maximum:
                raise LegacyImportError(f"historical warmup.blocks[{index}] timing is invalid")
            if per_eval != max(1, int(round(block_ns / evaluations))):
                raise LegacyImportError(
                    f"historical warmup.blocks[{index}].per_eval_ns does not replay"
                )
            if per_eval != values[index]:
                raise LegacyImportError(
                    f"historical warmup.blocks[{index}] does not mirror durations_ns"
                )
    return {
        "warmup_ns": values,
        "raw_warmup_sha256": canonical_sha256(values),
        "source_diagnostic": None,
        "source_error_code": None,
    }


def _parse_no_warmup_runtime_unavailable(candidate: _VerifiedCandidate) -> None:
    evidence = _exact_source_envelope(
        candidate,
        status="invalid",
        classification="runtime_unavailable",
        error_code="runtime_unavailable",
    )
    benchmark = _exact_object(
        evidence.get("benchmark_evidence"), frozenset(), "runtime-unavailable benchmark_evidence"
    )
    if benchmark:
        raise LegacyImportError("runtime-unavailable benchmark evidence must be empty")
    return None


def _parse_warmup_unstable(candidate: _VerifiedCandidate) -> dict[str, Any]:
    evidence = _exact_source_envelope(
        candidate,
        status="invalid",
        classification="invalid",
        error_code="warmup_unstable",
    )
    benchmark = _exact_object(
        evidence.get("benchmark_evidence"),
        frozenset({"failure_diagnostic"}),
        "warmup-unstable benchmark_evidence",
    )
    diagnostic = _exact_object(
        benchmark["failure_diagnostic"],
        frozenset({"schema_version", "code", "details"}),
        "warmup-unstable diagnostic",
    )
    try:
        exact_int64(diagnostic["schema_version"], "warmup-unstable diagnostic.schema_version", 1)
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc
    if diagnostic["code"] != "warmup_unstable":
        raise LegacyImportError("warmup-unstable diagnostic code is not registered")
    details = _exact_object(
        diagnostic["details"], frozenset({"warmups_ns"}), "warmup-unstable details"
    )
    values = _positive_warmup_values(
        details["warmups_ns"], expected=16, name="warmup-unstable details.warmups_ns"
    )
    diagnostic_copy = parse_capped_json(
        canonical_json_bytes(diagnostic), limit=PERSISTENCE_MAX_BUNDLE_BYTES
    )
    return {
        "warmup_ns": values,
        "raw_warmup_sha256": canonical_sha256(values),
        "source_diagnostic": diagnostic_copy,
        "source_error_code": "warmup_unstable",
    }


def _parse_descriptor(
    candidate: _VerifiedCandidate, descriptor: AdapterDescriptor
) -> dict[str, Any] | None:
    if descriptor.parser_id == PARSER_RUNTIME_UNAVAILABLE:
        return _parse_no_warmup_runtime_unavailable(candidate)
    if descriptor.parser_id == PARSER_COMPLETED_V1:
        return _parse_completed_warmup(candidate, with_blocks=False)
    if descriptor.parser_id == PARSER_WARMUP_UNSTABLE_V1:
        return _parse_warmup_unstable(candidate)
    if descriptor.parser_id == PARSER_COMPLETED_W1V3:
        return _parse_completed_warmup(candidate, with_blocks=True)
    raise LegacyImportError("matched descriptor parser_id has no closed implementation")


def inventory_h0_generations(source: os.PathLike[str] | str) -> dict[str, Any]:
    """Inventory all eager H0 Common Results without extracting observations."""

    path, identity, source_hash, verified = _read_verified_candidates(source)
    candidates = [
        _inventory_candidate(candidate, STATIC_ADAPTER_REGISTRY) for candidate in verified
    ]
    counts = {
        "eligible": len(candidates),
        MATCHED: sum(row["registry_match"]["state"] == MATCHED for row in candidates),
        UNSUPPORTED_GENERATION: sum(
            row["registry_match"]["state"] == UNSUPPORTED_GENERATION for row in candidates
        ),
        CLAIMED_KNOWN_MALFORMED: sum(
            row["registry_match"]["state"] == CLAIMED_KNOWN_MALFORMED
            for row in candidates
        ),
    }
    body = {
        "schema_version": 1,
        "inventory_schema": INVENTORY_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "source_database_path": str(path),
        "source_database_identity": identity.value(),
        "source_database_sha256": source_hash,
        "adapter_registry_schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
        "adapter_registry_sha256": STATIC_ADAPTER_REGISTRY.registry_sha256,
        "adapter_descriptors": [
            {
                "adapter_id": descriptor.adapter_id,
                "outcome": descriptor.outcome,
                "parser_id": descriptor.parser_id,
                "descriptor_sha256": descriptor.descriptor_sha256,
            }
            for descriptor in STATIC_ADAPTER_REGISTRY.descriptors
        ],
        "counts": counts,
        "candidates": candidates,
    }
    report = {**body, "inventory_sha256": canonical_sha256(body)}
    try:
        parsed = parse_capped_json(
            canonical_json_bytes(report, maximum=PERSISTENCE_MAX_BUNDLE_BYTES),
            limit=PERSISTENCE_MAX_BUNDLE_BYTES,
        )
    except (CanonicalError, ProtocolError, TypeError, ValueError) as exc:
        raise LegacyImportError("canonical generation inventory construction failed") from exc
    if not isinstance(parsed, dict):
        raise LegacyImportError("canonical generation inventory is not an object")
    return parsed


def build_legacy_entity_binding(
    descriptor: AdapterDescriptor,
    *,
    adapter_registry_sha256: str,
    source_run_id: str,
    source_created_at_unix_ns: int,
    parent_manifest_sha256: str,
    parent_result_sha256: str,
    parent_evidence_sha256: str,
    parent_bundle_sha256: str,
    raw_warmup_sha256: str,
) -> dict[str, Any]:
    """Build the closed entity-ID binding without accepting raw warmup values."""

    if not isinstance(descriptor, AdapterDescriptor):
        raise LegacyImportError("entity binding requires an AdapterDescriptor")
    source_run_id = _exact_identifier(source_run_id, "source_run_id")
    try:
        created = nonnegative_int64(source_created_at_unix_ns, "source_created_at_unix_ns")
    except CanonicalError as exc:
        raise LegacyImportError(str(exc)) from exc
    identity = {
        "schema_version": ENTITY_BINDING_SCHEMA_VERSION,
        "adapter_registry_schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
        "adapter_registry_sha256": _exact_sha256(
            adapter_registry_sha256, "adapter_registry_sha256"
        ),
        "adapter_id": descriptor.adapter_id,
        "adapter_outcome": descriptor.outcome,
        "parser_id": descriptor.parser_id,
        "adapter_descriptor_sha256": descriptor.descriptor_sha256,
        "adapter_selector_sha256": canonical_sha256(descriptor.selector),
        "source_run_id": source_run_id,
        "source_created_at_unix_ns": created,
        "parent_manifest_sha256": _exact_sha256(
            parent_manifest_sha256, "parent_manifest_sha256"
        ),
        "parent_result_sha256": _exact_sha256(parent_result_sha256, "parent_result_sha256"),
        "parent_evidence_sha256": _exact_sha256(
            parent_evidence_sha256, "parent_evidence_sha256"
        ),
        "parent_bundle_sha256": _exact_sha256(parent_bundle_sha256, "parent_bundle_sha256"),
        "raw_warmup_sha256": _exact_sha256(raw_warmup_sha256, "raw_warmup_sha256"),
    }
    entity_id = f"legacy-h0-warmup-{canonical_sha256(identity)}"
    return {
        "schema_version": ENTITY_BINDING_SCHEMA_VERSION,
        "entity_id": entity_id,
        "identity": identity,
    }


def _legacy_bundle(
    candidate: _VerifiedCandidate,
    inventory_row: Mapping[str, Any],
    descriptor: AdapterDescriptor,
    extracted: Mapping[str, Any],
    *,
    source_database_sha256: str,
    registry: AdapterRegistry,
) -> dict[str, Any]:
    stored = _mapping(inventory_row.get("stored_hashes"), "inventory stored_hashes")
    raw_warmup_sha256 = _exact_sha256(
        extracted.get("raw_warmup_sha256"), "extracted raw_warmup_sha256"
    )
    selector_sha256 = canonical_sha256(descriptor.selector)
    binding = build_legacy_entity_binding(
        descriptor,
        adapter_registry_sha256=registry.registry_sha256,
        source_run_id=candidate.closed_manifest.run_id,
        source_created_at_unix_ns=candidate.created_at_unix_ns,
        parent_manifest_sha256=stored["manifest_sha256"],
        parent_result_sha256=stored["result_sha256"],
        parent_evidence_sha256=stored["evidence_sha256"],
        parent_bundle_sha256=stored["bundle_sha256"],
        raw_warmup_sha256=raw_warmup_sha256,
    )
    source_state = (
        candidate.result["status"],
        candidate.result["classification"],
        extracted["source_error_code"],
    )
    lineage = {
        "parent_phase": "H0",
        "parent_run_id": candidate.closed_manifest.run_id,
        "parent_manifest_sha256": stored["manifest_sha256"],
        "parent_result_sha256": stored["result_sha256"],
        "parent_evidence_sha256": stored["evidence_sha256"],
        "parent_bundle_sha256": stored["bundle_sha256"],
        "parent_code_sha256": candidate.code_sha256,
        "parent_spec_sha256": candidate.spec_sha256,
        "parent_environment_sha256": candidate.environment_sha256,
        "source_database_sha256": _exact_sha256(
            source_database_sha256, "source_database_sha256"
        ),
        "registry_sha256": registry.registry_sha256,
        "descriptor_sha256": descriptor.descriptor_sha256,
        "selector_sha256": selector_sha256,
        "raw_warmup_sha256": raw_warmup_sha256,
    }
    descriptor_fields = {
        "adapter": descriptor.adapter_id,
        "registry_schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
        "registry_sha256": registry.registry_sha256,
        "descriptor_sha256": descriptor.descriptor_sha256,
        "selector_sha256": selector_sha256,
        "parser_id": descriptor.parser_id,
        "raw_warmup_sha256": raw_warmup_sha256,
    }
    manifest = {
        "schema_version": LEGACY_MANIFEST_SCHEMA_VERSION,
        "entity_id": binding["entity_id"],
        "source_phase": "H0",
        "source_run_id": candidate.closed_manifest.run_id,
        "source_mode": "eager_baseline",
        "source_status": source_state[0],
        "source_classification": source_state[1],
        "source_created_at_unix_ns": candidate.created_at_unix_ns,
        "observation_kind": "warmup_observation",
        **descriptor_fields,
    }
    observation = {
        **descriptor_fields,
        "source_status": source_state[0],
        "source_classification": source_state[1],
        "source_error_code": source_state[2],
        "warmup_ns": list(extracted["warmup_ns"]),
        "statistics": legacy_warmup_statistics(extracted["warmup_ns"]),
        "source_diagnostic": extracted["source_diagnostic"],
    }
    payload = {
        "entity_id": binding["entity_id"],
        "entity_kind": "legacy_h0_warmup_observation",
        "status": "legacy_observation",
        "created_at_unix_ns": candidate.created_at_unix_ns,
        "manifest": manifest,
        "trace": {"schema_version": LEGACY_TRACE_SCHEMA_VERSION, "observation": observation},
        "result": {
            "schema_version": LEGACY_RESULT_SCHEMA_VERSION,
            "status": "legacy_observation",
            "conclusion": "historical_warmup_observation_only",
            "interpretation": "descriptive_only",
            "action": "no_h0_conclusion",
            "stationarity_supported": False,
            "paced_gate_applicable": False,
            "h0_reclassification": False,
            "promotion_applicable": False,
        },
        "lineage": lineage,
    }
    try:
        return build_bundle(**payload)
    except (BundleError, CanonicalError, TypeError, ValueError) as exc:
        raise LegacyImportError("legacy H0.1 evidence bundle construction failed") from exc


def _persistence_arguments(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: bundle[key]
        for key in (
            "entity_id",
            "entity_kind",
            "status",
            "created_at_unix_ns",
            "manifest",
            "trace",
            "result",
            "lineage",
        )
    }


def _verify_execute_target(
    target: Path, planned_bundles: tuple[dict[str, Any], ...]
) -> None:
    with Storage.open(target, read_only=True) as storage:
        with storage.read_transaction():
            verified = storage.verified_rows()
    stored_by_id = {
        row["bundle"]["entity_id"]: row["bundle"] for row in verified
    }
    for bundle in planned_bundles:
        stored = stored_by_id.get(bundle["entity_id"])
        if stored is None or canonical_json_bytes(stored) != canonical_json_bytes(bundle):
            raise LegacyImportError("persisted target does not replay the planned bundle")


def audit_h0_legacy_warmups(
    source: os.PathLike[str] | str,
    *,
    execute: bool = False,
    target: os.PathLike[str] | str | None = None,
) -> LegacyImportOutcome:
    """Build an exact compatibility plan and optionally persist it atomically."""

    if type(execute) is not bool:
        raise LegacyImportError("execute must be an exact boolean")
    if execute and target is None:
        raise LegacyImportError("adapter execute requires an explicit target database")
    if not execute and target is not None:
        raise LegacyImportError("adapter dry-run does not accept a target database")
    initial_path, initial_parent, initial_file = _source_path(source)
    path, source_identity, source_hash, verified = _read_verified_candidates(source)
    if path != initial_path or source_identity != initial_file:
        raise LegacyImportError("source database identity changed before compatibility audit")
    target_path = (
        _execute_target_path(
            target,
            source_path=path,
            source_identity=source_identity,
        )
        if execute and target is not None
        else None
    )
    inventory_rows = [
        _inventory_candidate(candidate, STATIC_ADAPTER_REGISTRY) for candidate in verified
    ]
    if any(
        row["registry_match"]["state"] == CLAIMED_KNOWN_MALFORMED
        for row in inventory_rows
    ):
        raise LegacyImportError("a claimed known generation is structurally malformed")
    records: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    for candidate, row in zip(verified, inventory_rows, strict=True):
        match = row["registry_match"]
        base = {
            "selection_index": row["selection_index"],
            "source_created_at_unix_ns": row["source_created_at_unix_ns"],
            "source_run_id": row["source_run_id"],
            "source_status": row["source_status"],
            "source_classification": row["source_classification"],
            "source_action": row["source_action"],
            "source_error_code": row["source_error_code"],
            "parent_manifest_sha256": row["stored_hashes"]["manifest_sha256"],
            "parent_result_sha256": row["stored_hashes"]["result_sha256"],
            "parent_evidence_sha256": row["stored_hashes"]["evidence_sha256"],
            "parent_bundle_sha256": row["stored_hashes"]["bundle_sha256"],
            "parent_code_sha256": row["stored_hashes"]["code_sha256"],
            "parent_spec_sha256": row["stored_hashes"]["spec_sha256"],
            "parent_environment_sha256": row["stored_hashes"]["environment_sha256"],
            "source_common_result_payload_sha256": row["stored_hashes"][
                "common_result_payload_sha256"
            ],
            "full_bundle_verification": "verified",
        }
        if match["state"] == UNSUPPORTED_GENERATION:
            records.append(
                {
                    **base,
                    "disposition": "excluded",
                    "adapter": None,
                    "adapter_outcome": None,
                    "parser_id": None,
                    "adapter_registry_schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
                    "adapter_registry_sha256": STATIC_ADAPTER_REGISTRY.registry_sha256,
                    "adapter_descriptor_sha256": None,
                    "adapter_selector_sha256": canonical_sha256(row["selector"]),
                    "raw_warmup_sha256": None,
                    "warmup_count": None,
                    "statistics": None,
                    "entity_id": None,
                    "h01_bundle_sha256": None,
                    "exclusion_reason": UNSUPPORTED_GENERATION,
                }
            )
            continue
        descriptor = STATIC_ADAPTER_REGISTRY.descriptor(
            match["adapter_id"], match["adapter_descriptor_sha256"]
        )
        extracted = _parse_descriptor(candidate, descriptor)
        descriptor_record = {
            "adapter": descriptor.adapter_id,
            "adapter_outcome": descriptor.outcome,
            "parser_id": descriptor.parser_id,
            "adapter_registry_schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
            "adapter_registry_sha256": STATIC_ADAPTER_REGISTRY.registry_sha256,
            "adapter_descriptor_sha256": descriptor.descriptor_sha256,
            "adapter_selector_sha256": canonical_sha256(descriptor.selector),
        }
        if descriptor.outcome == OUTCOME_RECOGNIZED_EXCLUSION:
            if extracted is not None:
                raise LegacyImportError("recognized exclusion parser returned observation data")
            records.append(
                {
                    **base,
                    **descriptor_record,
                    "disposition": "excluded",
                    "raw_warmup_sha256": None,
                    "warmup_count": None,
                    "statistics": None,
                    "entity_id": None,
                    "h01_bundle_sha256": None,
                    "exclusion_reason": RECOGNIZED_NO_WARMUP,
                }
            )
            continue
        if descriptor.outcome != OUTCOME_LEGACY_OBSERVATION or extracted is None:
            raise LegacyImportError("matched descriptor outcome/parser result is inconsistent")
        bundle = _legacy_bundle(
            candidate,
            row,
            descriptor,
            extracted,
            source_database_sha256=source_hash,
            registry=STATIC_ADAPTER_REGISTRY,
        )
        bundles.append(bundle)
        records.append(
            {
                **base,
                **descriptor_record,
                "disposition": "importable",
                "raw_warmup_sha256": extracted["raw_warmup_sha256"],
                "warmup_count": len(extracted["warmup_ns"]),
                "statistics": bundle["trace"]["observation"]["statistics"],
                "entity_id": bundle["entity_id"],
                "h01_bundle_sha256": bundle["bundle_sha256"],
                "exclusion_reason": None,
            }
        )
    body = {
        "schema_version": 2,
        "report_schema": REPORT_SCHEMA,
        "execution_mode": "adapter_execute" if execute else "adapter_dry_run",
        "selection_rule": SELECTION_RULE,
        "source_database_sha256": source_hash,
        "adapter_registry_schema_version": ADAPTER_REGISTRY_SCHEMA_VERSION,
        "adapter_registry_sha256": STATIC_ADAPTER_REGISTRY.registry_sha256,
        "counts": {
            "eligible": len(records),
            "importable": len(bundles),
            "excluded": len(records) - len(bundles),
        },
        "candidates": records,
    }
    report = {**body, "report_sha256": canonical_sha256(body)}
    parsed = parse_capped_json(
        canonical_json_bytes(report, maximum=PERSISTENCE_MAX_BUNDLE_BYTES),
        limit=PERSISTENCE_MAX_BUNDLE_BYTES,
    )
    if not isinstance(parsed, dict):
        raise LegacyImportError("canonical compatibility report is not an object")
    planned_bundles = tuple(bundles)
    persistence: tuple[PersistenceOutcome, ...] = ()
    if execute:
        if target_path is None:  # Guarded above; keeps the side-effect boundary explicit.
            raise LegacyImportError("adapter execute target was not resolved")
        _verify_source_witness(
            path,
            expected_parent=initial_parent,
            expected_file=source_identity,
            expected_sha256=source_hash,
        )
        try:
            if planned_bundles:
                arguments = tuple(
                    _persistence_arguments(bundle) for bundle in planned_bundles
                )
                with Storage.open(target_path) as storage:
                    persistence = storage.persist_bundles(arguments)
                expected_identities = tuple(
                    (bundle["entity_id"], bundle["bundle_sha256"])
                    for bundle in planned_bundles
                )
                actual_identities = tuple(
                    (outcome.entity_id, outcome.bundle_sha256)
                    for outcome in persistence
                )
                if actual_identities != expected_identities or any(
                    outcome.state not in {"inserted", "idempotent"}
                    for outcome in persistence
                ):
                    raise LegacyImportError("storage returned an invalid persistence outcome")
                _verify_execute_target(target_path, planned_bundles)
        except (LegacyImportError, StorageError, sqlite3.Error, CanonicalError, OSError) as exc:
            try:
                _verify_source_witness(
                    path,
                    expected_parent=initial_parent,
                    expected_file=source_identity,
                    expected_sha256=source_hash,
                )
            except LegacyImportError as source_error:
                raise source_error from exc
            if isinstance(exc, LegacyImportError):
                raise
            raise LegacyImportError("H0.1 batch persistence failed closed") from exc
        _verify_source_witness(
            path,
            expected_parent=initial_parent,
            expected_file=source_identity,
            expected_sha256=source_hash,
        )
    return LegacyImportOutcome(parsed, planned_bundles, persistence)


__all__ = [
    "ADAPTER_DESCRIPTOR_SCHEMA_VERSION",
    "ADAPTER_REGISTRY_SCHEMA_VERSION",
    "AdapterDescriptor",
    "AdapterMatch",
    "AdapterRegistry",
    "CLAIMED_KNOWN_MALFORMED",
    "COMPLETED_ADAPTER",
    "EXCLUSION_REASONS",
    "INVALID_ADAPTER",
    "INVENTORY_SCHEMA",
    "LegacyImportError",
    "LegacyImportOutcome",
    "MATCHED",
    "OUTCOME_LEGACY_OBSERVATION",
    "OUTCOME_RECOGNIZED_EXCLUSION",
    "PARSER_COMPLETED_V1",
    "PARSER_COMPLETED_W1V3",
    "PARSER_RUNTIME_UNAVAILABLE",
    "PARSER_WARMUP_UNSTABLE_V1",
    "RECOGNIZED_NO_WARMUP",
    "REPORT_SCHEMA",
    "SCHEMA_TAG_ALGORITHM",
    "SELECTION_RULE",
    "STATIC_ADAPTER_REGISTRY",
    "STRUCTURAL_FINGERPRINT_ALGORITHM",
    "UNSUPPORTED_GENERATION",
    "RUNTIME_UNAVAILABLE_ADAPTER",
    "W1V3_COMPLETED_ADAPTER",
    "audit_h0_legacy_warmups",
    "build_legacy_entity_binding",
    "declared_schema_tags",
    "inventory_h0_generations",
    "structural_fingerprint",
]
