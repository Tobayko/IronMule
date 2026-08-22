"""N10 evidence-bound policy selection with serial fail-closed behavior."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from friday_n10_v2.canonical import canonical_sha256
from friday_n10_v2.provenance import ProvenanceError, collect_provenance
from friday_n10_v2.runner import _private_regular_sha256, validate_predecessor_contract
from friday_n10_v2.storage import Storage, StorageError

from .constants import (
    BATCHED_PLAN,
    DEFAULT_N10_DATABASE_PATH,
    DTYPE,
    N10_CODE_SHA256,
    N10_DATABASE_SHA256,
    N10_DECISION_RECORD_ID,
    N10_DECISION_SHA256,
    N10_ENVIRONMENT_SHA256,
    N10_HARDWARE_SHA256,
    N10_PREREGISTRATION_SHA256,
    N10_PROVENANCE_SHA256,
    N10_SNAPSHOT_REVISION,
    N10_SPEC_SHA256,
    N10_STUDY_ID,
    OPERATION,
    OUTPUT_SHAPE,
    RHS_COUNT,
    SERIAL_PLAN,
    SHAPE,
)

Strategy = Literal["serial", "batched"]
IdentityProvider = Callable[[], Mapping[str, Any]]
EvidenceReader = Callable[[Path], Sequence[Mapping[str, Any]]]


class PolicyError(RuntimeError):
    """Evidence cannot authorize the optimized policy without weakening scope."""


@dataclass(frozen=True)
class Workload:
    operation: str
    dtype: str
    lhs_shape: tuple[int, int]
    rhs_shape: tuple[int, int]
    output_shape: tuple[int, int]
    rhs_count: int
    baseline_plan: str = SERIAL_PLAN
    candidate_plan: str = BATCHED_PLAN


REGISTERED_WORKLOAD = Workload(
    operation=OPERATION,
    dtype=DTYPE,
    lhs_shape=SHAPE,
    rhs_shape=SHAPE,
    output_shape=OUTPUT_SHAPE,
    rhs_count=RHS_COUNT,
)


@dataclass(frozen=True)
class PolicyEvidence:
    authorized: bool
    reason: str
    decision_record_id: str | None
    decision_sha256: str | None
    preregistration_sha256: str | None
    sealed_provenance_sha256: str | None
    formal_database_sha256: str | None
    formal_snapshot_revision: str | None
    evidence_records: int


@dataclass(frozen=True)
class PolicyDecision:
    strategy: Strategy
    plan: str
    reason: str
    evidence: PolicyEvidence


def _read_verified(path: Path) -> Sequence[Mapping[str, Any]]:
    before = _private_regular_sha256(path)
    if before != N10_DATABASE_SHA256:
        raise PolicyError("terminal N10 database hash differs")
    validate_predecessor_contract()
    with Storage.open(path, read_only=True) as storage:
        with storage.read_transaction():
            rows = storage.verified_records()
    revision = canonical_sha256(
        [
            [row["record_id"], row["payload_sha256"], row["provenance_sha256"]]
            for row in rows
        ]
    )
    after = _private_regular_sha256(path)
    if after != before or revision != N10_SNAPSHOT_REVISION:
        raise PolicyError("terminal N10 artifact changed or has a foreign revision")
    return rows


def _current_identity() -> Mapping[str, Any]:
    return collect_provenance(require_clean=False)


def _single(
    rows: Sequence[Mapping[str, Any]], kind: str
) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get("record_kind") == kind]
    if len(matches) != 1:
        raise PolicyError(f"exactly one {kind} record is required")
    return matches[0]


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("payload")
    if not isinstance(value, Mapping):
        raise PolicyError("verified N10 record has no payload object")
    return value


def _validate_history_shape(rows: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(row.get("record_kind") for row in rows)
    expected = {
        "preregistration": 1,
        "calibration_session": 6,
        "calibration_summary": 1,
        "confirmation_seal": 1,
        "confirmation_session": 6,
        "study_decision": 1,
    }
    if len(rows) != 16 or counts != Counter(expected):
        raise PolicyError("N10 history is not the terminal 16-record design")


def _validate_workload(preregistration: Mapping[str, Any]) -> None:
    specification = preregistration.get("study_specification")
    workload = specification.get("workload") if isinstance(specification, Mapping) else None
    expected = {
        "operation": OPERATION,
        "dtype": DTYPE,
        "lhs_shape": list(SHAPE),
        "rhs_shape": list(SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "rhs_count": RHS_COUNT,
        "baseline_plan": SERIAL_PLAN,
        "candidate_plan": BATCHED_PLAN,
    }
    if not isinstance(workload, Mapping) or any(
        workload.get(key) != value for key, value in expected.items()
    ):
        raise PolicyError("sealed N10 workload differs from the runtime scope")


def _validate_decision(row: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    required_gates = {
        "all_sessions_byte_identical": True,
        "gain_all_splits": True,
        "equivalence_all_splits": False,
        "regression_all_splits": False,
    }
    if (
        row.get("record_id") != N10_DECISION_RECORD_ID
        or row.get("formal_claim") is not True
        or payload.get("study_id") != N10_STUDY_ID
        or payload.get("decision_sha256") != N10_DECISION_SHA256
        or payload.get("preregistration_sha256") != N10_PREREGISTRATION_SHA256
        or payload.get("provenance_sha256") != N10_PROVENANCE_SHA256
        or payload.get("status") != "n10_gain_confirmed"
        or payload.get("action") != "permit_bounded_n10_runtime_prototype"
        or payload.get("claim") != "n10_batched_dispatch_is_faster_beyond_mde"
        or payload.get("claim_scope") != "one-device-one-workload-one-execution-plan"
        or payload.get("gates") != required_gates
    ):
        raise PolicyError("terminal N10 decision is not the registered authorization")


def _validate_identity(
    sealed: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if current.get("git_dirty") is not False:
        raise PolicyError("worktree_dirty")
    for field, expected, reason in (
        ("code_sha256", N10_CODE_SHA256, "n10_code_mismatch"),
        ("spec_sha256", N10_SPEC_SHA256, "n10_spec_mismatch"),
        ("environment_sha256", N10_ENVIRONMENT_SHA256, "environment_mismatch"),
        ("hardware_sha256", N10_HARDWARE_SHA256, "hardware_mismatch"),
    ):
        if sealed.get(field) != expected or current.get(field) != expected:
            raise PolicyError(reason)


def _fallback(reason: str, *, records: int = 0) -> PolicyEvidence:
    return PolicyEvidence(
        authorized=False,
        reason=reason,
        decision_record_id=None,
        decision_sha256=None,
        preregistration_sha256=None,
        sealed_provenance_sha256=None,
        formal_database_sha256=None,
        formal_snapshot_revision=None,
        evidence_records=records,
    )


def load_policy(
    evidence_path: str | Path = DEFAULT_N10_DATABASE_PATH,
    *,
    identity_provider: IdentityProvider = _current_identity,
    evidence_reader: EvidenceReader = _read_verified,
) -> PolicyEvidence:
    """Load and fully verify N10 once; any uncertainty becomes serial fallback."""

    rows: Sequence[Mapping[str, Any]] = ()
    try:
        rows = evidence_reader(Path(evidence_path))
        _validate_history_shape(rows)
        preregistration_row = _single(rows, "preregistration")
        decision_row = _single(rows, "study_decision")
        preregistration = _payload(preregistration_row)
        decision = _payload(decision_row)
        if (
            preregistration.get("preregistration_sha256") != N10_PREREGISTRATION_SHA256
            or preregistration.get("provenance_sha256") != N10_PROVENANCE_SHA256
            or preregistration_row.get("provenance_sha256") != N10_PROVENANCE_SHA256
        ):
            raise PolicyError("sealed N10 preregistration identity differs")
        _validate_workload(preregistration)
        _validate_decision(decision_row, decision)
        sealed = decision_row.get("provenance")
        if not isinstance(sealed, Mapping):
            raise PolicyError("terminal N10 provenance is unavailable")
        _validate_identity(sealed, identity_provider())
    except PolicyError as exc:
        reason = str(exc)
        if reason not in {
            "worktree_dirty",
            "n10_code_mismatch",
            "n10_spec_mismatch",
            "environment_mismatch",
            "hardware_mismatch",
        }:
            reason = "evidence_scope_mismatch"
        return _fallback(reason, records=len(rows))
    except (StorageError, ProvenanceError, OSError, ValueError, TypeError, KeyError):
        return _fallback("evidence_unavailable_or_invalid", records=len(rows))
    except Exception:
        # Policy loading is a release gate. Even an unanticipated verifier or
        # identity-provider failure must preserve the known-safe serial plan.
        return _fallback("evidence_unavailable_or_invalid", records=len(rows))

    return PolicyEvidence(
        authorized=True,
        reason="formal_n10_gain_exact_scope",
        decision_record_id=N10_DECISION_RECORD_ID,
        decision_sha256=N10_DECISION_SHA256,
        preregistration_sha256=N10_PREREGISTRATION_SHA256,
        sealed_provenance_sha256=N10_PROVENANCE_SHA256,
        formal_database_sha256=N10_DATABASE_SHA256,
        formal_snapshot_revision=N10_SNAPSHOT_REVISION,
        evidence_records=len(rows),
    )


def decision_for(evidence: PolicyEvidence, workload: Workload | None) -> PolicyDecision:
    if not evidence.authorized:
        return PolicyDecision("serial", SERIAL_PLAN, evidence.reason, evidence)
    if workload != REGISTERED_WORKLOAD:
        return PolicyDecision("serial", SERIAL_PLAN, "workload_out_of_scope", evidence)
    return PolicyDecision("batched", BATCHED_PLAN, evidence.reason, evidence)


__all__ = [
    "PolicyDecision",
    "PolicyError",
    "PolicyEvidence",
    "REGISTERED_WORKLOAD",
    "Workload",
    "decision_for",
    "load_policy",
]
