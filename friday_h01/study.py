"""Exact six-session replicated study aggregation for H0.1 v2."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import (
    CanonicalError,
    bounded_text,
    canonical_json_bytes,
    canonical_sha256,
    exact_int64,
    exact_keys,
)
from .constants import (
    GATE_LIMITS,
    PHASE,
    SCHEMA_VERSION,
    SESSION_COMPLETE_STATUS,
    SESSION_ORDER,
    STUDY,
    STUDY_STATUSES,
)
from .protocol import ProtocolError, validate_manifest, validate_result, validate_trace


class StudyError(ValueError):
    """Raised when a supplied study result cannot be validated or replayed."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_KEYS = frozenset({"manifest", "trace", "result"})
_STUDY_KEYS = frozenset(
    {
        "schema_version",
        "phase",
        "study",
        "study_id",
        "session_order",
        "session_count",
        "shared_provenance",
        "session_bindings",
        "failed_gate_count",
        "status",
        "conclusion",
        "action",
        "h0_reclassification",
        "promotion_applicable",
        "error",
        "decision_sha256",
    }
)
_PROVENANCE_KEYS = frozenset(
    {"study_spec_sha256", "code_sha256", "environment_sha256", "fixture", "source"}
)
_FIXTURE_KEYS = frozenset({"a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"})
_SOURCE_KEYS = frozenset(
    {
        "parent_phase",
        "parent_run_id",
        "parent_manifest_sha256",
        "parent_result_sha256",
        "parent_bundle_sha256",
    }
)
_BINDING_KEYS = frozenset(
    {
        "session_id",
        "run_id",
        "manifest_sha256",
        "trace_sha256",
        "result_sha256",
        "gates_sha256",
        "all_gates_pass",
        "failed_gates",
    }
)
_ERROR_KEYS = frozenset({"code", "message"})


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StudyError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "study_spec_sha256": manifest["study_spec_sha256"],
        "code_sha256": manifest["code_sha256"],
        "environment_sha256": manifest["environment_sha256"],
        "fixture": copy.deepcopy(manifest["fixture"]),
        "source": copy.deepcopy(manifest["source"]),
    }


def _study_base() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study": STUDY,
        "study_id": None,
        "session_order": list(SESSION_ORDER),
        "session_count": 0,
        "shared_provenance": None,
        "session_bindings": None,
        "failed_gate_count": None,
        "status": "h01_invalid",
        "conclusion": "invalid_input",
        "action": "no_h0_conclusion",
        "h0_reclassification": False,
        "promotion_applicable": False,
        "error": None,
    }


def _invalid_study(error: Exception) -> dict[str, Any]:
    result = _study_base()
    result["error"] = {
        "code": "study_contract_violation",
        "message": (str(error) or "study contract violation")[:512],
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def analyze_study(session_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Replay exactly C0,V0,C1,V1,C2,V2 and produce one terminal study status."""

    try:
        if (
            not isinstance(session_records, Sequence)
            or isinstance(session_records, (str, bytes, bytearray))
            or len(session_records) != len(SESSION_ORDER)
        ):
            raise StudyError("study requires exactly six ordered session records")
        bindings: list[dict[str, Any]] = []
        shared: dict[str, Any] | None = None
        failed_gate_count = 0
        for index, (record_value, expected_session_id) in enumerate(
            zip(session_records, SESSION_ORDER, strict=True)
        ):
            record = exact_keys(record_value, _RECORD_KEYS, f"study.sessions[{index}]")
            manifest = validate_manifest(record["manifest"])
            if manifest["session"]["id"] != expected_session_id:
                raise StudyError(
                    f"study.sessions[{index}] must be {expected_session_id}, not selective/reordered"
                )
            trace = validate_trace(manifest, record["trace"])
            result = validate_result(record["result"], manifest, trace)
            if result["status"] != SESSION_COMPLETE_STATUS:
                raise StudyError(f"study.sessions[{index}] is not a complete valid session")
            candidate_provenance = _provenance(manifest)
            if shared is None:
                shared = candidate_provenance
            elif candidate_provenance != shared:
                raise StudyError(f"study.sessions[{index}] has mixed provenance")
            failed_gates = [
                name for name in GATE_LIMITS if result["gates"][name]["status"] == "fail"
            ]
            failed_gate_count += len(failed_gates)
            bindings.append(
                {
                    "session_id": expected_session_id,
                    "run_id": manifest["run_id"],
                    "manifest_sha256": canonical_sha256(manifest),
                    "trace_sha256": canonical_sha256(trace),
                    "result_sha256": canonical_sha256(result),
                    "gates_sha256": canonical_sha256(result["gates"]),
                    "all_gates_pass": not failed_gates,
                    "failed_gates": failed_gates,
                }
            )
        if shared is None:
            raise StudyError("study shared provenance is absent")
        identity = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "study": STUDY,
            "session_order": list(SESSION_ORDER),
            "shared_provenance": shared,
            "session_bindings": bindings,
        }
        supported = failed_gate_count == 0
        result = _study_base()
        result.update(
            {
                "study_id": f"h01-study-{canonical_sha256(identity)}",
                "session_count": len(SESSION_ORDER),
                "shared_provenance": shared,
                "session_bindings": bindings,
                "failed_gate_count": failed_gate_count,
                "status": (
                    "h01_stationarity_supported"
                    if supported
                    else "h01_complete_unresolved"
                ),
                "conclusion": (
                    "replicated_stationarity_supported"
                    if supported
                    else "replicated_stationarity_not_supported"
                ),
                "error": None,
            }
        )
        result["decision_sha256"] = canonical_sha256(result)
        return result
    except (CanonicalError, ProtocolError, StudyError, TypeError, ValueError) as exc:
        return _invalid_study(exc)


def _validate_provenance(value: Any) -> None:
    provenance = exact_keys(value, _PROVENANCE_KEYS, "study_result.shared_provenance")
    for key in ("study_spec_sha256", "code_sha256", "environment_sha256"):
        _sha256(provenance[key], f"study_result.shared_provenance.{key}")
    fixture = exact_keys(provenance["fixture"], _FIXTURE_KEYS, "study_result.shared_provenance.fixture")
    for key in _FIXTURE_KEYS:
        _sha256(fixture[key], f"study_result.shared_provenance.fixture.{key}")
    source = exact_keys(provenance["source"], _SOURCE_KEYS, "study_result.shared_provenance.source")
    if source["parent_phase"] != "H0":
        raise StudyError("study_result.shared_provenance.source.parent_phase must be H0")
    bounded_text(source["parent_run_id"], "study_result.shared_provenance.source.parent_run_id", maximum=160)
    for key in ("parent_manifest_sha256", "parent_result_sha256", "parent_bundle_sha256"):
        _sha256(source[key], f"study_result.shared_provenance.source.{key}")


def _validate_study_structure(value: Any) -> dict[str, Any]:
    try:
        result = exact_keys(value, _STUDY_KEYS, "study_result")
        exact_int64(result["schema_version"], "study_result.schema_version", SCHEMA_VERSION)
        if result["phase"] != PHASE or result["study"] != STUDY:
            raise StudyError("study_result phase or study is not registered")
        if result["session_order"] != list(SESSION_ORDER):
            raise StudyError("study_result.session_order is not exact")
        status = result["status"]
        if not isinstance(status, str) or status not in STUDY_STATUSES:
            raise StudyError("study_result.status is not registered")
        if result["action"] != "no_h0_conclusion":
            raise StudyError("study_result.action is not registered")
        if result["h0_reclassification"] is not False or result["promotion_applicable"] is not False:
            raise StudyError("study result may not affect H0 or promotion")
        if status == "h01_invalid":
            exact_int64(result["session_count"], "study_result.session_count", 0)
            if result["study_id"] is not None:
                raise StudyError("invalid study_result.study_id must be null")
            if any(
                result[key] is not None
                for key in ("shared_provenance", "session_bindings", "failed_gate_count")
            ):
                raise StudyError("invalid study result must not expose partial evidence")
            if result["conclusion"] != "invalid_input":
                raise StudyError("invalid study conclusion is inconsistent")
            error = exact_keys(result["error"], _ERROR_KEYS, "study_result.error")
            bounded_text(error["code"], "study_result.error.code", maximum=64)
            bounded_text(error["message"], "study_result.error.message", maximum=512)
        else:
            exact_int64(
                result["session_count"],
                "study_result.session_count",
                len(SESSION_ORDER),
            )
            if not isinstance(result["study_id"], str) or not result["study_id"].startswith("h01-study-"):
                raise StudyError("study_result.study_id is not registered")
            _sha256(result["study_id"].removeprefix("h01-study-"), "study_result.study_id digest")
            _validate_provenance(result["shared_provenance"])
            bindings = result["session_bindings"]
            if (
                not isinstance(bindings, Sequence)
                or isinstance(bindings, (str, bytes, bytearray))
                or len(bindings) != len(SESSION_ORDER)
            ):
                raise StudyError("study_result.session_bindings must contain exactly six rows")
            counted_failed = 0
            for index, (binding_value, session_id) in enumerate(
                zip(bindings, SESSION_ORDER, strict=True)
            ):
                binding = exact_keys(
                    binding_value,
                    _BINDING_KEYS,
                    f"study_result.session_bindings[{index}]",
                )
                if binding["session_id"] != session_id:
                    raise StudyError(f"study_result.session_bindings[{index}].session_id is reordered")
                bounded_text(binding["run_id"], f"study_result.session_bindings[{index}].run_id", maximum=160)
                for key in ("manifest_sha256", "trace_sha256", "result_sha256", "gates_sha256"):
                    _sha256(binding[key], f"study_result.session_bindings[{index}].{key}")
                failed = binding["failed_gates"]
                if not isinstance(failed, list) or any(name not in GATE_LIMITS for name in failed):
                    raise StudyError(f"study_result.session_bindings[{index}].failed_gates is invalid")
                expected_failed = [name for name in GATE_LIMITS if name in set(failed)]
                if failed != expected_failed or len(failed) != len(set(failed)):
                    raise StudyError(f"study_result.session_bindings[{index}].failed_gates is not exact")
                if binding["all_gates_pass"] is not (not failed):
                    raise StudyError(f"study_result.session_bindings[{index}].all_gates_pass is inconsistent")
                counted_failed += len(failed)
            exact_int64(
                result["failed_gate_count"],
                "study_result.failed_gate_count",
                counted_failed,
            )
            supported = counted_failed == 0
            expected_status = "h01_stationarity_supported" if supported else "h01_complete_unresolved"
            expected_conclusion = (
                "replicated_stationarity_supported"
                if supported
                else "replicated_stationarity_not_supported"
            )
            if status != expected_status or result["conclusion"] != expected_conclusion:
                raise StudyError("study_result status/conclusion does not match all six gates")
            if result["error"] is not None:
                raise StudyError("complete study_result.error must be null")
            identity = {
                "schema_version": SCHEMA_VERSION,
                "phase": PHASE,
                "study": STUDY,
                "session_order": list(SESSION_ORDER),
                "shared_provenance": result["shared_provenance"],
                "session_bindings": result["session_bindings"],
            }
            if result["study_id"] != f"h01-study-{canonical_sha256(identity)}":
                raise StudyError("study_result.study_id does not replay")
        _sha256(result["decision_sha256"], "study_result.decision_sha256")
        body = {key: result[key] for key in result if key != "decision_sha256"}
        if result["decision_sha256"] != canonical_sha256(body):
            raise StudyError("study_result.decision_sha256 does not replay")
        canonical_json_bytes(result)
        return copy.deepcopy(dict(result))
    except CanonicalError as exc:
        raise StudyError(str(exc)) from exc


def validate_study_result(value: Any, session_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute all six sessions and require exact study-result replay."""

    supplied = _validate_study_structure(value)
    expected = analyze_study(session_records)
    _validate_study_structure(expected)
    if canonical_json_bytes(supplied) != canonical_json_bytes(expected):
        raise StudyError("study result replay differs from all six session records")
    return expected


__all__ = ["StudyError", "analyze_study", "validate_study_result"]
