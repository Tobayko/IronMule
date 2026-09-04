"""Evidence-bound policy with baseline-only behavior on every uncertainty."""

from __future__ import annotations

import hashlib
import math
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from experiments.head_skip_formal import study

from .constants import (
    APPROVAL_SPEC_PATH,
    APPROVAL_SPEC_SHA256,
    BASELINE_PLAN,
    BATCH_SIZE,
    DEFAULT_FORMAL_DATABASE_PATH,
    DEFAULT_RUNTIME_DATABASE_PATH,
    FORMAL_CANDIDATE_ID,
    FORMAL_CHAIN_HEAD,
    FORMAL_CONFIRMATION_SEAL_SHA256,
    FORMAL_DATABASE_SHA256,
    FORMAL_DECISION_SHA256,
    FORMAL_PREREGISTRATION_DOCUMENT_SHA256,
    FORMAL_PREREGISTRATION_SHA256,
    FORMAL_PROVENANCE_SHA256,
    FORMAL_SCRIPT_SHA256,
    FORMAL_STUDY_ID,
    GPU_MAX_EXTRA_PEAK_BYTES,
    GPU_MAX_RATIO,
    GPU_MEASUREMENT_ORDERS,
    GPU_RUN_ID,
    HEAD_SKIP_PLAN,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_TOKENS,
    POLICY_MAX_INCREMENTAL_NS,
    POLICY_MAX_LOAD_NS,
    POLICY_MAX_MEDIAN_NS,
    POLICY_MAX_P95_NS,
    POLICY_RUN_ID,
    PREFILL_CHUNK,
    PROJECT_ROOT,
    PROMPT_CONTENT_SHA256,
    PROMPT_TOKENS,
    QUALIFICATION_ID,
)
from .history import History
from .provenance import collect_provenance

Strategy = Literal["baseline", "head_skip"]


class PolicyError(RuntimeError):
    """The optimized route cannot be authorized without weakening its scope."""


@dataclass(frozen=True)
class RequestScope:
    model_id: str
    model_revision: str
    prompt_content_sha256: str
    prompt_tokens: int
    prefill_chunk: int
    batch: int
    temperature: float
    prompt_logprobs: bool
    fixed_horizon: bool
    output_tokens: int


REGISTERED_SCOPE = RequestScope(
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    prompt_content_sha256=PROMPT_CONTENT_SHA256,
    prompt_tokens=PROMPT_TOKENS,
    prefill_chunk=PREFILL_CHUNK,
    batch=BATCH_SIZE,
    temperature=0.0,
    prompt_logprobs=False,
    fixed_horizon=True,
    output_tokens=OUTPUT_TOKENS,
)


@dataclass(frozen=True)
class FormalSnapshot:
    rows: tuple[Mapping[str, Any], ...]
    chain_head: str


@dataclass(frozen=True)
class PolicyEvidence:
    authorized: bool
    reason: str
    study_id: str | None
    decision_sha256: str | None
    preregistration_sha256: str | None
    formal_database_sha256: str | None
    formal_chain_head: str | None
    evidence_records: int
    qualification_id: str | None = None
    runtime_validation_record_id: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    strategy: Strategy
    plan: str
    reason: str
    evidence: PolicyEvidence


EvidenceReader = Callable[[Path], FormalSnapshot]
IdentityProvider = Callable[[], Mapping[str, Any]]
RuntimeReader = Callable[[Path], Sequence[Mapping[str, Any]]]


def _regular_sha256(path: Path) -> str:
    try:
        info = path.lstat()
        payload = path.read_bytes()
    except OSError as exc:
        raise PolicyError("required evidence input is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_size > 8 * 1024 * 1024
        or len(payload) != info.st_size
    ):
        raise PolicyError("required evidence input is unsafe or unstable")
    return hashlib.sha256(payload).hexdigest()


def _read_verified(path: Path) -> FormalSnapshot:
    if _regular_sha256(path) != FORMAL_DATABASE_SHA256:
        raise PolicyError("formal_database_mismatch")
    if (
        _regular_sha256(PROJECT_ROOT / study.SCRIPT_PATH) != FORMAL_SCRIPT_SHA256
        or _regular_sha256(PROJECT_ROOT / study.PREREGISTRATION_PATH)
        != FORMAL_PREREGISTRATION_DOCUMENT_SHA256
    ):
        raise PolicyError("formal_source_mismatch")
    before = FORMAL_DATABASE_SHA256
    with study.Storage.open(path, read_only=True) as storage:
        rows = tuple(storage.verified_records())
        head_row = storage.connection.execute(
            "SELECT record_sha256 FROM records ORDER BY seq DESC LIMIT 1"
        ).fetchone()
    after = _regular_sha256(path)
    head = head_row[0] if head_row is not None else None
    if after != before or head != FORMAL_CHAIN_HEAD:
        raise PolicyError("formal_database_changed_or_foreign")
    return FormalSnapshot(rows=rows, chain_head=head)


def _current_identity() -> Mapping[str, Any]:
    return study.collect_provenance(require_clean=False)


def _current_runtime_identity() -> Mapping[str, Any]:
    return collect_provenance(require_clean=True)


def _read_runtime_verified(path: Path) -> tuple[Mapping[str, Any], ...]:
    with History.open(path, read_only=True) as history:
        with history.read_transaction():
            return tuple(history.verified_records())


def _single(rows: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get("kind") == kind]
    if len(matches) != 1:
        raise PolicyError(f"exactly one {kind} record is required")
    return matches[0]


def _validate_history(snapshot: FormalSnapshot) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    rows = snapshot.rows
    expected = Counter(
        {
            "preregistration": 1,
            "calibration_session": 6,
            "calibration_summary": 1,
            "confirmation_seal": 1,
            "confirmation_session": 6,
            "study_decision": 1,
        }
    )
    if len(rows) != 16 or Counter(row.get("kind") for row in rows) != expected:
        raise PolicyError("formal history is not the terminal 16-record design")
    preregistration = _single(rows, "preregistration")
    decision = _single(rows, "study_decision")
    specification = preregistration.get("study_specification")
    workload = specification.get("workload") if isinstance(specification, Mapping) else None
    scope = specification.get("scope") if isinstance(specification, Mapping) else None
    expected_workload = {
        "prompt_content_sha256": PROMPT_CONTENT_SHA256,
        "prompt_tokens": PROMPT_TOKENS,
        "prefill_chunk": PREFILL_CHUNK,
        "batch": BATCH_SIZE,
        "correctness_tokens": OUTPUT_TOKENS,
        "arm_a": BASELINE_PLAN,
        "arm_b": HEAD_SKIP_PLAN,
        "primary_endpoint": "paired_prefill_duration_ratio_B_over_A",
    }
    if not isinstance(workload, Mapping) or any(
        workload.get(key) != value for key, value in expected_workload.items()
    ):
        raise PolicyError("formal workload differs from runtime scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("model_id") != MODEL_ID
        or scope.get("model_revision") != MODEL_REVISION
        or scope.get("sampling") != "greedy_fixed_horizon"
        or scope.get("prompt_logprobs") is not False
        or scope.get("claim_scope") != "one-device-one-model-one-prompt-one-prefill-plan"
    ):
        raise PolicyError("formal scope differs from runtime scope")
    required_gates = {
        "all_sessions_token_identical": True,
        "equivalence_all_splits": False,
        "gain_all_splits": True,
        "regression_all_splits": False,
    }
    if (
        preregistration.get("study_id") != FORMAL_STUDY_ID
        or preregistration.get("candidate_id") != FORMAL_CANDIDATE_ID
        or preregistration.get("preregistration_sha256")
        != FORMAL_PREREGISTRATION_SHA256
        or preregistration.get("provenance_sha256") != FORMAL_PROVENANCE_SHA256
        or decision.get("study_id") != FORMAL_STUDY_ID
        or decision.get("candidate_id") != FORMAL_CANDIDATE_ID
        or decision.get("decision_sha256") != FORMAL_DECISION_SHA256
        or decision.get("preregistration_sha256") != FORMAL_PREREGISTRATION_SHA256
        or decision.get("provenance_sha256") != FORMAL_PROVENANCE_SHA256
        or decision.get("confirmation_seal_sha256")
        != FORMAL_CONFIRMATION_SEAL_SHA256
        or decision.get("status") != "head_skip_gain_confirmed"
        or decision.get("action") != "permit_bounded_architecture_review"
        or decision.get("claim") != "prefill_head_skip_is_faster_beyond_mde"
        or decision.get("claim_scope")
        != "one-device-one-model-one-prompt-one-prefill-plan"
        or decision.get("formal_claim") is not True
        or decision.get("gates") != required_gates
    ):
        raise PolicyError("terminal decision is not the registered formal result")
    return preregistration, decision


def _validate_identity(
    sealed: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if current.get("git_dirty") is not False:
        raise PolicyError("worktree_dirty")
    for field, reason in (
        ("code_sha256", "formal_code_mismatch"),
        ("spec_sha256", "formal_spec_mismatch"),
        ("environment_sha256", "environment_mismatch"),
        ("hardware_sha256", "hardware_mismatch"),
        ("model_sha256", "model_snapshot_mismatch"),
    ):
        if not isinstance(sealed.get(field), str) or current.get(field) != sealed.get(field):
            raise PolicyError(reason)


def _fallback(reason: str, records: int = 0) -> PolicyEvidence:
    return PolicyEvidence(False, reason, None, None, None, None, None, records)


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    converted = float(value)
    return math.isfinite(converted) and (converted > 0 if positive else True)


def _formal_policy_projection(report: Mapping[str, Any]) -> bool:
    value = report.get("policy")
    return (
        isinstance(value, Mapping)
        and value.get("authorized") is True
        and value.get("study_id") == FORMAL_STUDY_ID
        and value.get("decision_sha256") == FORMAL_DECISION_SHA256
        and value.get("preregistration_sha256") == FORMAL_PREREGISTRATION_SHA256
        and value.get("formal_database_sha256") == FORMAL_DATABASE_SHA256
        and value.get("formal_chain_head") == FORMAL_CHAIN_HEAD
        and value.get("evidence_records") == 16
    )


def _validate_runtime_identity(
    rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any]
) -> None:
    if current.get("git_dirty") is not False:
        raise PolicyError("worktree_dirty")
    for row in rows:
        sealed = row.get("provenance")
        if not isinstance(sealed, Mapping) or sealed.get("git_dirty") is not False:
            raise PolicyError("runtime_provenance_invalid")
        for field in (
            "code_sha256",
            "spec_sha256",
            "environment_sha256",
            "hardware_sha256",
        ):
            if not isinstance(sealed.get(field), str) or sealed.get(field) != current.get(
                field
            ):
                raise PolicyError(f"runtime_{field}_mismatch")


def _validate_cpu_gate(row: Mapping[str, Any]) -> None:
    report = row.get("report")
    if not isinstance(report, Mapping):
        raise PolicyError("runtime_cpu_record_invalid")
    metrics = report.get("metrics")
    thresholds = report.get("thresholds")
    workload = report.get("workload")
    resources = report.get("resources")
    guard = resources.get("guard") if isinstance(resources, Mapping) else None
    if (
        report.get("kind") != "policy_overhead"
        or report.get("run_id") != POLICY_RUN_ID
        or report.get("status") != "policy_overhead_passed"
        or report.get("qualification_id") != QUALIFICATION_ID
        or not _formal_policy_projection(report)
        or not isinstance(workload, Mapping)
        or workload.get("power_source") != "ac_power"
        or not isinstance(guard, Mapping)
        or guard.get("duty_cycle_limit") != 0.15
        or not _finite_number(guard.get("gpu_work_seconds"))
        or float(guard["gpu_work_seconds"]) != 0.0
        or not isinstance(metrics, Mapping)
        or metrics.get("gate_passed") is not True
        or metrics.get("policy_load_gate_passed") is not True
        or not _finite_number(metrics.get("policy_median_ns"), positive=True)
        or float(metrics["policy_median_ns"]) > POLICY_MAX_MEDIAN_NS
        or not _finite_number(metrics.get("policy_p95_ns"), positive=True)
        or float(metrics["policy_p95_ns"]) > POLICY_MAX_P95_NS
        or not _finite_number(metrics.get("incremental_median_ns"))
        or float(metrics["incremental_median_ns"]) > POLICY_MAX_INCREMENTAL_NS
        or not _finite_number(report.get("policy_load_ns"), positive=True)
        or float(report["policy_load_ns"]) > POLICY_MAX_LOAD_NS
        or not isinstance(thresholds, Mapping)
        or thresholds.get("policy_max_median_ns") != POLICY_MAX_MEDIAN_NS
        or thresholds.get("policy_max_p95_ns") != POLICY_MAX_P95_NS
        or thresholds.get("policy_max_incremental_ns") != POLICY_MAX_INCREMENTAL_NS
        or thresholds.get("policy_max_load_ns") != POLICY_MAX_LOAD_NS
    ):
        raise PolicyError("runtime_cpu_gate_invalid")


def _validate_attempt(row: Mapping[str, Any]) -> None:
    report = row.get("report")
    if (
        not isinstance(report, Mapping)
        or report.get("kind") != "runtime_validation_attempt"
        or report.get("run_id") != GPU_RUN_ID
        or report.get("status") != "runtime_validation_started"
        or report.get("qualification_id") != QUALIFICATION_ID
        or not _formal_policy_projection(report)
    ):
        raise PolicyError("runtime_attempt_record_invalid")


def _validate_gpu_gate(row: Mapping[str, Any]) -> None:
    report = row.get("report")
    if not isinstance(report, Mapping):
        raise PolicyError("runtime_gpu_record_invalid")
    metrics = report.get("metrics")
    correctness = report.get("correctness")
    workload = report.get("workload")
    blocks = report.get("blocks")
    thresholds = report.get("thresholds")
    resources = report.get("resources")
    guard = resources.get("guard") if isinstance(resources, Mapping) else None
    if (
        report.get("kind") != "runtime_validation"
        or report.get("run_id") != GPU_RUN_ID
        or report.get("status") != "runtime_validation_passed"
        or report.get("qualification_id") != QUALIFICATION_ID
        or not _formal_policy_projection(report)
        or not isinstance(correctness, Mapping)
        or correctness.get("token_identical") is not True
        or correctness.get("candidate_path_exercised") is not True
        or not isinstance(workload, Mapping)
        or workload.get("prompt_tokens") != PROMPT_TOKENS
        or workload.get("output_tokens") != OUTPUT_TOKENS
        or workload.get("prefill_chunk") != PREFILL_CHUNK
        or workload.get("power_source") != "ac_power"
        or not isinstance(thresholds, Mapping)
        or thresholds.get("max_ratio") != GPU_MAX_RATIO
        or thresholds.get("max_extra_peak_bytes") != GPU_MAX_EXTRA_PEAK_BYTES
        or thresholds.get("duty_cycle") != 0.15
        or not isinstance(guard, Mapping)
        or guard.get("duty_cycle_limit") != 0.15
        or not _finite_number(guard.get("gpu_work_seconds"), positive=True)
        or not isinstance(metrics, Mapping)
        or metrics.get("gate_passed") is not True
        or metrics.get("policy_load_gate_passed") is not True
        or metrics.get("byte_identical") is not True
        or not _finite_number(metrics.get("ratio"), positive=True)
        or float(metrics["ratio"]) > GPU_MAX_RATIO
        or type(metrics.get("peak_memory_delta_bytes")) is not int
        or metrics["peak_memory_delta_bytes"] > GPU_MAX_EXTRA_PEAK_BYTES
        or type(metrics.get("swap_delta_bytes")) is not int
        or metrics["swap_delta_bytes"] > 0
        or not _finite_number(report.get("policy_load_ns"), positive=True)
        or float(report["policy_load_ns"]) > POLICY_MAX_LOAD_NS
        or not isinstance(blocks, list)
        or len(blocks) != len(GPU_MEASUREMENT_ORDERS)
        or any(not isinstance(block, Mapping) for block in blocks)
        or tuple(block.get("order") for block in blocks)
        != GPU_MEASUREMENT_ORDERS
        or any(block.get("candidate_plan") != HEAD_SKIP_PLAN for block in blocks)
    ):
        raise PolicyError("runtime_gpu_gate_invalid")


def _runtime_path_is_registered(path: str | Path) -> bool:
    return Path(path).absolute() == DEFAULT_RUNTIME_DATABASE_PATH.absolute()


def load_policy(
    evidence_path: str | Path = DEFAULT_FORMAL_DATABASE_PATH,
    *,
    evidence_reader: EvidenceReader = _read_verified,
    identity_provider: IdentityProvider = _current_identity,
) -> PolicyEvidence:
    """Verify the formal result and user-approved boundary; never raise open."""

    records = 0
    try:
        if _regular_sha256(APPROVAL_SPEC_PATH) != APPROVAL_SPEC_SHA256:
            raise PolicyError("approval_spec_mismatch")
        snapshot = evidence_reader(Path(evidence_path))
        records = len(snapshot.rows)
        preregistration, _decision = _validate_history(snapshot)
        sealed = preregistration.get("provenance")
        if not isinstance(sealed, Mapping):
            raise PolicyError("formal provenance is unavailable")
        _validate_identity(sealed, identity_provider())
    except PolicyError as exc:
        reason = str(exc)
        known = {
            "worktree_dirty",
            "formal_code_mismatch",
            "formal_spec_mismatch",
            "environment_mismatch",
            "hardware_mismatch",
            "model_snapshot_mismatch",
            "approval_spec_mismatch",
        }
        return _fallback(reason if reason in known else "evidence_scope_mismatch", records)
    except Exception:
        return _fallback("evidence_unavailable_or_invalid", records)
    return PolicyEvidence(
        True,
        "formal_gain_and_user_approval_exact_scope",
        FORMAL_STUDY_ID,
        FORMAL_DECISION_SHA256,
        FORMAL_PREREGISTRATION_SHA256,
        FORMAL_DATABASE_SHA256,
        FORMAL_CHAIN_HEAD,
        records,
    )


def load_gpu_qualification_policy(
    evidence_path: str | Path = DEFAULT_FORMAL_DATABASE_PATH,
    runtime_path: str | Path = DEFAULT_RUNTIME_DATABASE_PATH,
    *,
    evidence_reader: EvidenceReader = _read_verified,
    identity_provider: IdentityProvider = _current_identity,
    runtime_reader: RuntimeReader = _read_runtime_verified,
    runtime_identity_provider: IdentityProvider = _current_runtime_identity,
) -> PolicyEvidence:
    """Authorize the one GPU qualification only after the exact CPU gate."""

    formal = load_policy(
        evidence_path,
        evidence_reader=evidence_reader,
        identity_provider=identity_provider,
    )
    if not formal.authorized:
        return formal
    try:
        if not _runtime_path_is_registered(runtime_path):
            raise PolicyError("runtime_database_path_mismatch")
        rows = tuple(runtime_reader(Path(runtime_path)))
        if len(rows) != 1:
            raise PolicyError("runtime_cpu_gate_missing_or_history_not_fresh")
        _validate_runtime_identity(rows, runtime_identity_provider())
        _validate_cpu_gate(rows[0])
    except PolicyError as exc:
        return _fallback(str(exc), len(rows) if "rows" in locals() else 0)
    except Exception:
        return _fallback("runtime_evidence_unavailable_or_invalid")
    return formal


def load_runtime_policy(
    evidence_path: str | Path = DEFAULT_FORMAL_DATABASE_PATH,
    runtime_path: str | Path = DEFAULT_RUNTIME_DATABASE_PATH,
    *,
    evidence_reader: EvidenceReader = _read_verified,
    identity_provider: IdentityProvider = _current_identity,
    runtime_reader: RuntimeReader = _read_runtime_verified,
    runtime_identity_provider: IdentityProvider = _current_runtime_identity,
) -> PolicyEvidence:
    """Authorize normal use only after the exact terminal runtime qualification."""

    formal = load_policy(
        evidence_path,
        evidence_reader=evidence_reader,
        identity_provider=identity_provider,
    )
    if not formal.authorized:
        return formal
    try:
        if not _runtime_path_is_registered(runtime_path):
            raise PolicyError("runtime_database_path_mismatch")
        rows = tuple(runtime_reader(Path(runtime_path)))
        if len(rows) != 3:
            raise PolicyError("runtime_qualification_incomplete_or_nonterminal")
        if tuple(row.get("record_kind") for row in rows) != (
            "policy_overhead",
            "runtime_validation_attempt",
            "runtime_validation",
        ):
            raise PolicyError("runtime_qualification_sequence_invalid")
        _validate_runtime_identity(rows, runtime_identity_provider())
        _validate_cpu_gate(rows[0])
        _validate_attempt(rows[1])
        _validate_gpu_gate(rows[2])
        record_id = rows[2].get("record_id")
        if not isinstance(record_id, str) or len(record_id) != 64:
            raise PolicyError("runtime_validation_record_invalid")
    except PolicyError as exc:
        return _fallback(str(exc), len(rows) if "rows" in locals() else 0)
    except Exception:
        return _fallback("runtime_evidence_unavailable_or_invalid")
    return PolicyEvidence(
        True,
        "runtime_qualification_passed_exact_scope",
        formal.study_id,
        formal.decision_sha256,
        formal.preregistration_sha256,
        formal.formal_database_sha256,
        formal.formal_chain_head,
        formal.evidence_records,
        QUALIFICATION_ID,
        record_id,
    )


def decision_for(evidence: PolicyEvidence, scope: RequestScope | None) -> PolicyDecision:
    if not evidence.authorized:
        return PolicyDecision("baseline", BASELINE_PLAN, evidence.reason, evidence)
    if scope != REGISTERED_SCOPE:
        return PolicyDecision("baseline", BASELINE_PLAN, "request_out_of_scope", evidence)
    return PolicyDecision("head_skip", HEAD_SKIP_PLAN, evidence.reason, evidence)


__all__ = [
    "FormalSnapshot",
    "PolicyDecision",
    "PolicyError",
    "PolicyEvidence",
    "REGISTERED_SCOPE",
    "RequestScope",
    "decision_for",
    "load_gpu_qualification_policy",
    "load_policy",
    "load_runtime_policy",
]
