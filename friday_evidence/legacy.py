"""One-way import of explicitly downgraded historical H1/H2 summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .canonical import canonical_sha256
from .registry import LEGACY_SOURCE_PATH, REGISTERED_TOOLS, SCHEMA_VERSION
from .storage import EvidenceStorage, PersistenceOutcome, StorageError


def _load(path: str | Path) -> tuple[str, str, list[dict[str, Any]]]:
    try:
        payload = Path(path).read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise StorageError("legacy summary source is unreadable") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "source_document", "records"
    }:
        raise StorageError("legacy summary source has an invalid envelope")
    if value["schema_version"] != SCHEMA_VERSION:
        raise StorageError("legacy summary schema version is not registered")
    if not isinstance(value["source_document"], str) or not value["source_document"]:
        raise StorageError("legacy source document is invalid")
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise StorageError("legacy summary source must contain records")
    if len(records) > 100:
        raise StorageError("legacy summary source exceeds its row limit")
    return value["source_document"], hashlib.sha256(payload).hexdigest(), records


def _provenance(
    tool: str, source_document: str, source_key: str, import_manifest_sha256: str
) -> dict[str, Any]:
    unavailable_code = {"availability": "unavailable", "reason": "pre-git historical result"}
    unavailable_environment = {
        "availability": "partial",
        "known": {"device": "Apple M1 Max", "memory_gb": 32},
        "unknown": ["complete package lock", "exact executable", "full OS fingerprint"],
    }
    unavailable_spec = {
        "availability": "not_formally_bound",
        "reason": "formal A/A gate and frozen MDE were incomplete before H1/H2 runs",
    }
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool,
        "workload_key": REGISTERED_TOOLS[tool],
        "provenance_kind": "legacy_summary",
        "source_document": source_document,
        "source_key": source_key,
        "import_manifest_sha256": import_manifest_sha256,
        "git_revision": "0" * 40,
        "git_dirty": True,
        "code_sha256": canonical_sha256(unavailable_code),
        "code": unavailable_code,
        "spec_sha256": canonical_sha256(unavailable_spec),
        "spec": unavailable_spec,
        "environment_sha256": canonical_sha256(unavailable_environment),
        "environment": unavailable_environment,
        "hardware_key": canonical_sha256(unavailable_environment["known"]),
        "hardware": unavailable_environment["known"],
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return provenance


def import_legacy_summaries(
    storage: EvidenceStorage, path: str | Path = LEGACY_SOURCE_PATH
) -> list[PersistenceOutcome]:
    if storage.read_only:
        raise StorageError("legacy import needs writable evidence storage")
    source_document, import_manifest_sha256, records = _load(path)
    outcomes: list[PersistenceOutcome] = []
    seen: set[tuple[str, str]] = set()
    for entry in records:
        if not isinstance(entry, dict) or set(entry) != {
            "source_key", "tool", "result_status", "observed_at_unix_ns", "report"
        }:
            raise StorageError("legacy record has an invalid shape")
        tool = entry["tool"]
        source_key = entry["source_key"]
        if tool not in REGISTERED_TOOLS or not isinstance(source_key, str):
            raise StorageError("legacy record has an invalid identity")
        identity = (tool, source_key)
        if identity in seen:
            raise StorageError("legacy source contains a duplicate identity")
        seen.add(identity)
        report = entry["report"]
        if not isinstance(report, dict):
            raise StorageError("legacy report must be an object")
        if report.get("evidence_grade") != "legacy_summary" or report.get(
            "raw_measurements_available"
        ) is not False:
            raise StorageError("legacy report must state its evidence downgrade")
        if report.get("formal_claim") is not False:
            raise StorageError("legacy report must explicitly reject a formal claim")
        outcomes.append(
            storage.persist(
                evidence_kind="legacy_summary",
                source_key=source_key,
                tool=tool,
                report=report,
                provenance=_provenance(
                    tool, source_document, source_key, import_manifest_sha256
                ),
                result_status=entry["result_status"],
                raw_measurements_available=False,
                observed_at_unix_ns=entry["observed_at_unix_ns"],
            )
        )
    return outcomes


__all__ = ["import_legacy_summaries"]
