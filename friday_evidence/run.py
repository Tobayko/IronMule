"""Fail-closed lifecycle binding a live measurement to evidence storage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from .canonical import canonical_sha256
from .provenance import collect_provenance
from .registry import DEFAULT_DATABASE_PATH, RAW_REPORT_FIELDS, REGISTERED_TOOLS
from .storage import EvidenceStorage, PersistenceOutcome

T = TypeVar("T", bound=dict[str, Any])


@dataclass(frozen=True)
class EvidenceBinding:
    tool: str
    database_path: Path
    observed_at_unix_ns: int
    provenance: dict[str, object]

    def _source_key(self, report: Mapping[str, Any]) -> str:
        return (
            f"native:{self.observed_at_unix_ns}:"
            f"{self.provenance['provenance_sha256']}:{canonical_sha256(report)}"
        )

    def persist(
        self,
        report: Mapping[str, Any],
        *,
        result_status: str,
        raw_measurements_available: bool,
    ) -> PersistenceOutcome:
        # A measurement is only valid while its checked-out code/spec/environment
        # identity stays unchanged from preflight through persistence.
        current = collect_provenance(self.tool, require_clean=True)
        if current["provenance_sha256"] != self.provenance["provenance_sha256"]:
            raise RuntimeError("measurement provenance changed while the run was active")
        with EvidenceStorage.open(self.database_path) as storage:
            return storage.persist(
                evidence_kind="native",
                source_key=self._source_key(report),
                tool=self.tool,
                report=report,
                provenance=self.provenance,
                result_status=result_status,
                raw_measurements_available=raw_measurements_available,
                observed_at_unix_ns=self.observed_at_unix_ns,
            )


def prepare_evidence(
    tool: str, *, database_path: str | Path = DEFAULT_DATABASE_PATH
) -> EvidenceBinding:
    """Verify storage and clean provenance before any measuring backend is imported."""

    if tool not in REGISTERED_TOOLS:
        raise ValueError(f"unregistered evidence tool: {tool}")
    path = Path(database_path)
    with EvidenceStorage.open(path, initialize=True):
        pass
    provenance = collect_provenance(tool, require_clean=True)
    return EvidenceBinding(
        tool=tool,
        database_path=path,
        observed_at_unix_ns=time.time_ns(),
        provenance=provenance,
    )


def _status(report: Mapping[str, Any]) -> str:
    for field in ("verdict", "status", "state"):
        value = report.get(field)
        if isinstance(value, str) and 1 <= len(value) <= 128:
            return value
    return "measurement_complete"


def run_persisted(
    tool: str,
    operation: Callable[[], T],
    *,
    raw_measurements_available: bool = True,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> T:
    """Run one already-authorized operation and persist before returning output."""

    binding = prepare_evidence(tool, database_path=database_path)
    try:
        report = operation()
        if not isinstance(report, dict):
            raise TypeError("measurement operation must return a report object")
        if report.get("formal_claim") not in (None, False):
            raise ValueError("schema v1 cannot emit a formal H1/H2 claim")
        report["formal_claim"] = False
        raw_field = RAW_REPORT_FIELDS[tool]
        raw_value = report.get(raw_field)
        if not isinstance(raw_value, list) or not raw_value:
            raise ValueError(f"native report lacks registered raw field: {raw_field}")
    except (Exception, SystemExit) as exc:
        failure = {
            "verdict": "measurement_failed",
            "failure_type": type(exc).__name__,
            "raw_measurements_available": False,
            "formal_claim": False,
        }
        # If even failure persistence cannot complete, that storage/provenance
        # error replaces the original: no unrecorded live attempt may look valid.
        binding.persist(
            failure,
            result_status="measurement_failed",
            raw_measurements_available=False,
        )
        raise
    outcome = binding.persist(
        report,
        result_status=_status(report),
        raw_measurements_available=raw_measurements_available,
    )
    report["evidence"] = {
        "record_id": outcome.record_id,
        "persistence_state": outcome.state,
        "database": str(binding.database_path),
    }
    return report


__all__ = ["EvidenceBinding", "prepare_evidence", "run_persisted"]
