"""Closed manifest, trace, and replayed session-result contracts for H0.1 v2."""

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
    finite_number,
    int64,
    nonnegative_int64,
    positive_int64,
)
from .constants import (
    ACF_LAGS,
    BURN_IN_SAMPLES,
    CHANGEPOINT_MAX_SPLIT,
    CHANGEPOINT_MIN_SPLIT,
    COOLDOWN_NS,
    GATE_LIMITS,
    INT64_MAX,
    LONG_GAP_NS,
    MAIN_BLOCKS,
    MAIN_SAMPLES,
    MAX_GAP_OVERSHOOT_NS,
    PHASE,
    RESULT_STATUSES,
    SAMPLES_PER_BLOCK,
    SCHEMA_VERSION,
    SESSION_COMPLETE_STATUS,
    SESSION_INVALID_STATUS,
    SESSION_SPECS,
    SHORT_GAP_NS,
    STUDY,
    TELEMETRY_MISSING_REASONS,
    TOTAL_SAMPLES,
)
from .schedule import ScheduleError, materialize_schedule, validate_schedule


class ProtocolError(ValueError):
    """Raised when a closed H0.1 protocol object is not exact."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "phase",
        "study",
        "run_id",
        "session",
        "schedule",
        "budgets",
        "gates",
        "fixture",
        "study_spec_sha256",
        "code_sha256",
        "environment_sha256",
        "source",
    }
)
_SESSION_KEYS = frozenset({"id", "cohort", "index", "seed"})
_BUDGET_KEYS = frozenset(
    {
        "burn_in_samples",
        "cooldown_ns",
        "main_samples",
        "main_blocks",
        "samples_per_block",
        "short_gap_ns",
        "long_gap_ns",
        "max_gap_overshoot_ns",
        "changepoint_min_split",
        "changepoint_max_split",
    }
)
_GATE_SPEC_KEYS = frozenset({"operator", "limit"})
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
_TRACE_KEYS = frozenset(
    {
        "schema_version",
        "phase",
        "study",
        "run_id",
        "manifest_sha256",
        "session_id",
        "schedule_sha256",
        "fixture",
        "study_spec_sha256",
        "code_sha256",
        "environment_sha256",
        "source",
        "cooldown",
        "telemetry",
        "samples",
    }
)
_COOLDOWN_KEYS = frozenset({"requested_ns", "observed_ns"})
_TELEMETRY_KEYS = frozenset({"thermal_state", "power_source"})
_TELEMETRY_VALUE_KEYS = frozenset({"value", "missing_reason"})
_SAMPLE_KEYS = frozenset(
    {
        "sample_index",
        "phase",
        "phase_index",
        "block_index",
        "position",
        "gap_label",
        "requested_gap_ns",
        "gap_start_ns",
        "gap_end_ns",
        "start_ns",
        "duration_ns",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "phase",
        "study",
        "run_id",
        "manifest_sha256",
        "trace_sha256",
        "status",
        "conclusion",
        "action",
        "h0_reclassification",
        "promotion_applicable",
        "error",
        "sample_accounting",
        "metrics",
        "gates",
        "decision_sha256",
    }
)
_ERROR_KEYS = frozenset({"code", "message"})
_ACCOUNTING_KEYS = frozenset(
    {
        "trace_samples",
        "burn_in_samples",
        "main_samples",
        "analysis_samples",
        "dropped_samples",
        "adaptive_stop",
        "outlier_deletion",
    }
)
_METRICS_KEYS = frozenset(
    {
        "transform",
        "residual_sha256",
        "actual_gap_sha256",
        "trajectory",
        "changepoint",
        "acf",
        "effective_sample_size",
        "pace_effect_ratio",
        "tail_ratio",
        "gap_adherence",
    }
)
_TRAJECTORY_KEYS = frozenset({"slope_per_second", "effect_ratio", "observed_span_seconds"})
_CHANGEPOINT_KEYS = frozenset({"split", "effect_ratio"})
_ACF_KEYS = frozenset({f"lag{lag}" for lag in ACF_LAGS})
_GAP_ADHERENCE_KEYS = frozenset({"max_overshoot_ns", "median_overshoot_ns"})
_GATE_RESULT_KEYS = frozenset({"status", "observed", "operator", "limit"})


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _fixed_budgets() -> dict[str, int]:
    return {
        "burn_in_samples": BURN_IN_SAMPLES,
        "cooldown_ns": COOLDOWN_NS,
        "main_samples": MAIN_SAMPLES,
        "main_blocks": MAIN_BLOCKS,
        "samples_per_block": SAMPLES_PER_BLOCK,
        "short_gap_ns": SHORT_GAP_NS,
        "long_gap_ns": LONG_GAP_NS,
        "max_gap_overshoot_ns": MAX_GAP_OVERSHOOT_NS,
        "changepoint_min_split": CHANGEPOINT_MIN_SPLIT,
        "changepoint_max_split": CHANGEPOINT_MAX_SPLIT,
    }


def _fixed_gate_specs() -> dict[str, dict[str, float | str]]:
    return {
        name: {"operator": operator, "limit": limit}
        for name, (operator, limit) in GATE_LIMITS.items()
    }


def _validate_fixture(value: Any, *, name: str = "fixture") -> dict[str, str]:
    fixture = exact_keys(value, _FIXTURE_KEYS, name)
    for key in _FIXTURE_KEYS:
        _sha256(fixture[key], f"{name}.{key}")
    body = {key: fixture[key] for key in ("a_sha256", "b_sha256", "metadata_sha256")}
    if fixture["fixture_sha256"] != canonical_sha256(body):
        raise ProtocolError(f"{name}.fixture_sha256 does not match component hashes")
    return dict(fixture)


def _validate_source(value: Any, *, name: str = "source") -> dict[str, str]:
    source = exact_keys(value, _SOURCE_KEYS, name)
    if source["parent_phase"] != "H0":
        raise ProtocolError(f"{name}.parent_phase must remain H0")
    bounded_text(source["parent_run_id"], f"{name}.parent_run_id", maximum=160)
    for key in ("parent_manifest_sha256", "parent_result_sha256", "parent_bundle_sha256"):
        _sha256(source[key], f"{name}.{key}")
    return dict(source)


def _manifest_identity(
    session_id: str,
    fixture: Mapping[str, str],
    study_spec_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    source: Mapping[str, str],
) -> dict[str, Any]:
    cohort, index, seed = SESSION_SPECS[session_id]
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study": STUDY,
        "session": {"id": session_id, "cohort": cohort, "index": index, "seed": seed},
        "schedule": materialize_schedule(session_id),
        "budgets": _fixed_budgets(),
        "gates": _fixed_gate_specs(),
        "fixture": dict(fixture),
        "study_spec_sha256": study_spec_sha256,
        "code_sha256": code_sha256,
        "environment_sha256": environment_sha256,
        "source": dict(source),
    }


def build_manifest(
    session_id: str,
    *,
    fixture: Mapping[str, str],
    study_spec_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    source: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic, provenance-bound manifest without executing work."""

    try:
        if session_id not in SESSION_SPECS:
            raise ProtocolError("session id is not registered")
        checked_fixture = _validate_fixture(fixture, name="manifest.fixture")
        checked_spec = _sha256(study_spec_sha256, "manifest.study_spec_sha256")
        checked_code = _sha256(code_sha256, "manifest.code_sha256")
        checked_environment = _sha256(environment_sha256, "manifest.environment_sha256")
        checked_source = _validate_source(source, name="manifest.source")
        identity = _manifest_identity(
            session_id,
            checked_fixture,
            checked_spec,
            checked_code,
            checked_environment,
            checked_source,
        )
        manifest = {**identity, "run_id": f"h01-{session_id.lower()}-{canonical_sha256(identity)}"}
        return validate_manifest(manifest)
    except CanonicalError as exc:
        raise ProtocolError(str(exc)) from exc


def validate_manifest(value: Any) -> dict[str, Any]:
    """Validate exact manifest closure and recompute every identity field."""

    try:
        manifest = exact_keys(value, _MANIFEST_KEYS, "manifest")
        exact_int64(manifest["schema_version"], "manifest.schema_version", SCHEMA_VERSION)
        if manifest["phase"] != PHASE or manifest["study"] != STUDY:
            raise ProtocolError("manifest phase or study is not registered")
        session = exact_keys(manifest["session"], _SESSION_KEYS, "manifest.session")
        session_id = session["id"]
        if not isinstance(session_id, str) or session_id not in SESSION_SPECS:
            raise ProtocolError("manifest.session.id is not registered")
        cohort, index, seed = SESSION_SPECS[session_id]
        exact_int64(session["index"], "manifest.session.index", index)
        exact_int64(session["seed"], "manifest.session.seed", seed)
        if dict(session) != {"id": session_id, "cohort": cohort, "index": index, "seed": seed}:
            raise ProtocolError("manifest.session differs from its registered identity")
        validate_schedule(manifest["schedule"], name="manifest.schedule")
        if manifest["schedule"] != materialize_schedule(session_id):
            raise ProtocolError("manifest.schedule is not registered for its session")
        budgets = exact_keys(manifest["budgets"], _BUDGET_KEYS, "manifest.budgets")
        for key, expected in _fixed_budgets().items():
            exact_int64(budgets[key], f"manifest.budgets.{key}", expected)
        gates = manifest["gates"]
        if not isinstance(gates, Mapping) or set(gates) != set(GATE_LIMITS):
            raise ProtocolError("manifest.gates is not closed")
        for name, expected in _fixed_gate_specs().items():
            gate = exact_keys(gates[name], _GATE_SPEC_KEYS, f"manifest.gates.{name}")
            finite_number(gate["limit"], f"manifest.gates.{name}.limit", minimum=0.0)
            if dict(gate) != expected:
                raise ProtocolError(f"manifest.gates.{name} differs from registration")
        fixture = _validate_fixture(manifest["fixture"], name="manifest.fixture")
        spec_hash = _sha256(manifest["study_spec_sha256"], "manifest.study_spec_sha256")
        code_hash = _sha256(manifest["code_sha256"], "manifest.code_sha256")
        environment = _sha256(manifest["environment_sha256"], "manifest.environment_sha256")
        source = _validate_source(manifest["source"], name="manifest.source")
        identity = _manifest_identity(
            session_id,
            fixture,
            spec_hash,
            code_hash,
            environment,
            source,
        )
        expected_run_id = f"h01-{session_id.lower()}-{canonical_sha256(identity)}"
        if manifest["run_id"] != expected_run_id:
            raise ProtocolError("manifest.run_id does not match canonical identity")
        canonical_json_bytes(manifest)
        return copy.deepcopy(dict(manifest))
    except (CanonicalError, ScheduleError) as exc:
        raise ProtocolError(str(exc)) from exc


def _default_telemetry() -> dict[str, dict[str, None | str]]:
    return {
        "thermal_state": {"value": None, "missing_reason": "not_collected"},
        "power_source": {"value": None, "missing_reason": "not_collected"},
    }


def _validate_telemetry(value: Any) -> dict[str, Any]:
    telemetry = exact_keys(value, _TELEMETRY_KEYS, "trace.telemetry")
    checked: dict[str, Any] = {}
    for name in sorted(_TELEMETRY_KEYS):
        item = exact_keys(telemetry[name], _TELEMETRY_VALUE_KEYS, f"trace.telemetry.{name}")
        reading = item["value"]
        reason = item["missing_reason"]
        if reading is None:
            if reason not in TELEMETRY_MISSING_REASONS:
                raise ProtocolError(f"trace.telemetry.{name} has no registered missing reason")
        else:
            bounded_text(reading, f"trace.telemetry.{name}.value", maximum=128)
            if reason is not None:
                raise ProtocolError(f"trace.telemetry.{name} must use value/reason XOR")
        checked[name] = {"value": reading, "missing_reason": reason}
    return checked


def _safe_add(left: int, right: int, name: str) -> int:
    if right > INT64_MAX - left:
        raise ProtocolError(f"{name} exceeds signed-int64")
    return left + right


def build_trace(
    manifest: Mapping[str, Any],
    durations_ns: Sequence[int],
    *,
    first_start_ns: int = 2_000_000_000,
    observed_cooldown_ns: int = COOLDOWN_NS,
    gap_overshoots_ns: Sequence[int] | None = None,
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize an offline trace envelope from already recorded values."""

    checked_manifest = validate_manifest(manifest)
    if (
        not isinstance(durations_ns, Sequence)
        or isinstance(durations_ns, (str, bytes, bytearray))
        or len(durations_ns) != TOTAL_SAMPLES
    ):
        raise ProtocolError(f"durations_ns must contain exactly {TOTAL_SAMPLES} entries")
    overshoots_value: Sequence[int] = (
        [0] * TOTAL_SAMPLES if gap_overshoots_ns is None else gap_overshoots_ns
    )
    if (
        not isinstance(overshoots_value, Sequence)
        or isinstance(overshoots_value, (str, bytes, bytearray))
        or len(overshoots_value) != TOTAL_SAMPLES
    ):
        raise ProtocolError(f"gap_overshoots_ns must contain exactly {TOTAL_SAMPLES} entries")
    try:
        durations = [positive_int64(value, f"durations_ns[{index}]") for index, value in enumerate(durations_ns)]
        overshoots = [
            nonnegative_int64(
                value,
                f"gap_overshoots_ns[{index}]",
                maximum=MAX_GAP_OVERSHOOT_NS,
            )
            for index, value in enumerate(overshoots_value)
        ]
        first_start = nonnegative_int64(first_start_ns, "first_start_ns")
        observed_cooldown = int64(
            observed_cooldown_ns,
            "observed_cooldown_ns",
            minimum=COOLDOWN_NS,
        )
    except CanonicalError as exc:
        raise ProtocolError(str(exc)) from exc

    entries = checked_manifest["schedule"]["entries"]
    samples: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        requested = entry["requested_gap_ns"]
        actual = _safe_add(requested, overshoots[index], f"sample {index} actual gap")
        if index == 0:
            if first_start < actual:
                raise ProtocolError("first_start_ns is earlier than the first recorded gap")
            gap_start = first_start - actual
        else:
            previous = samples[index - 1]
            gap_start = _safe_add(previous["start_ns"], previous["duration_ns"], f"sample {index} gap start")
            if entry["phase"] == "main" and previous["phase"] == "burn_in":
                gap_start = _safe_add(gap_start, observed_cooldown, f"sample {index} cooldown end")
        gap_end = _safe_add(gap_start, actual, f"sample {index} gap end")
        samples.append(
            {
                **entry,
                "gap_start_ns": gap_start,
                "gap_end_ns": gap_end,
                "start_ns": gap_end,
                "duration_ns": durations[index],
            }
        )

    trace = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study": STUDY,
        "run_id": checked_manifest["run_id"],
        "manifest_sha256": canonical_sha256(checked_manifest),
        "session_id": checked_manifest["session"]["id"],
        "schedule_sha256": checked_manifest["schedule"]["sha256"],
        "fixture": copy.deepcopy(checked_manifest["fixture"]),
        "study_spec_sha256": checked_manifest["study_spec_sha256"],
        "code_sha256": checked_manifest["code_sha256"],
        "environment_sha256": checked_manifest["environment_sha256"],
        "source": copy.deepcopy(checked_manifest["source"]),
        "cooldown": {"requested_ns": COOLDOWN_NS, "observed_ns": observed_cooldown},
        "telemetry": copy.deepcopy(telemetry) if telemetry is not None else _default_telemetry(),
        "samples": samples,
    }
    return validate_trace(checked_manifest, trace)


def _validate_trace_integer_structure(trace: Mapping[str, Any]) -> tuple[Mapping[str, Any], Sequence[Any]]:
    exact_int64(trace["schema_version"], "trace.schema_version", SCHEMA_VERSION)
    cooldown = exact_keys(trace["cooldown"], _COOLDOWN_KEYS, "trace.cooldown")
    exact_int64(cooldown["requested_ns"], "trace.cooldown.requested_ns", COOLDOWN_NS)
    int64(cooldown["observed_ns"], "trace.cooldown.observed_ns", minimum=COOLDOWN_NS)
    samples = trace["samples"]
    if (
        not isinstance(samples, Sequence)
        or isinstance(samples, (str, bytes, bytearray))
        or len(samples) != TOTAL_SAMPLES
    ):
        raise ProtocolError(f"trace.samples must contain exactly {TOTAL_SAMPLES} entries")
    for index, sample_value in enumerate(samples):
        name = f"trace.samples[{index}]"
        sample = exact_keys(sample_value, _SAMPLE_KEYS, name)
        nonnegative_int64(sample["sample_index"], f"{name}.sample_index", maximum=TOTAL_SAMPLES - 1)
        nonnegative_int64(sample["phase_index"], f"{name}.phase_index", maximum=MAIN_SAMPLES - 1)
        nonnegative_int64(sample["block_index"], f"{name}.block_index", maximum=MAIN_BLOCKS - 1)
        nonnegative_int64(sample["position"], f"{name}.position", maximum=SAMPLES_PER_BLOCK - 1)
        nonnegative_int64(sample["requested_gap_ns"], f"{name}.requested_gap_ns")
        nonnegative_int64(sample["gap_start_ns"], f"{name}.gap_start_ns")
        nonnegative_int64(sample["gap_end_ns"], f"{name}.gap_end_ns")
        nonnegative_int64(sample["start_ns"], f"{name}.start_ns")
        positive_int64(sample["duration_ns"], f"{name}.duration_ns")
    return cooldown, samples


def validate_trace(manifest: Mapping[str, Any], value: Any) -> dict[str, Any]:
    """Validate provenance, schedule, actual pacing, order, and real timestamps."""

    try:
        checked_manifest = validate_manifest(manifest)
        trace = exact_keys(value, _TRACE_KEYS, "trace")
        cooldown, samples = _validate_trace_integer_structure(trace)
        fixed = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "study": STUDY,
            "run_id": checked_manifest["run_id"],
            "manifest_sha256": canonical_sha256(checked_manifest),
            "session_id": checked_manifest["session"]["id"],
            "schedule_sha256": checked_manifest["schedule"]["sha256"],
            "study_spec_sha256": checked_manifest["study_spec_sha256"],
            "code_sha256": checked_manifest["code_sha256"],
            "environment_sha256": checked_manifest["environment_sha256"],
        }
        for key, expected in fixed.items():
            if trace[key] != expected:
                raise ProtocolError(f"trace.{key} does not match manifest")
        if _validate_fixture(trace["fixture"], name="trace.fixture") != checked_manifest["fixture"]:
            raise ProtocolError("trace.fixture does not match manifest")
        if _validate_source(trace["source"], name="trace.source") != checked_manifest["source"]:
            raise ProtocolError("trace.source does not match manifest")
        _validate_telemetry(trace["telemetry"])

        schedule_entries = checked_manifest["schedule"]["entries"]
        previous: Mapping[str, Any] | None = None
        for index, (sample_value, scheduled) in enumerate(zip(samples, schedule_entries, strict=True)):
            name = f"trace.samples[{index}]"
            sample = exact_keys(sample_value, _SAMPLE_KEYS, name)
            for key in ("phase", "gap_label"):
                bounded_text(sample[key], f"{name}.{key}", maximum=32)
            for key in scheduled:
                if sample[key] != scheduled[key]:
                    raise ProtocolError(f"{name}.{key} differs from schedule")
            if sample["gap_end_ns"] != sample["start_ns"]:
                raise ProtocolError(f"{name}.gap_end_ns must equal {name}.start_ns")
            if sample["gap_end_ns"] < sample["gap_start_ns"]:
                raise ProtocolError(f"{name} has a rebound/negative actual gap")
            actual_gap = sample["gap_end_ns"] - sample["gap_start_ns"]
            requested = sample["requested_gap_ns"]
            if actual_gap < requested:
                raise ProtocolError(f"{name}.actual_gap_ns is shorter than requested")
            if actual_gap > requested + MAX_GAP_OVERSHOOT_NS:
                raise ProtocolError(f"{name}.actual_gap_ns exceeds registered overshoot")
            if previous is not None:
                expected_start = _safe_add(previous["start_ns"], previous["duration_ns"], f"{name}.gap_start_ns")
                if sample["phase"] == "main" and previous["phase"] == "burn_in":
                    expected_start = _safe_add(expected_start, cooldown["observed_ns"], f"{name}.gap_start_ns")
                if sample["gap_start_ns"] != expected_start:
                    raise ProtocolError(f"{name}.gap_start_ns breaks exact continuity")
                if sample["start_ns"] <= previous["start_ns"]:
                    raise ProtocolError(f"{name}.start_ns is not strictly monotonic")
            previous = sample
        canonical_json_bytes(trace)
        return copy.deepcopy(dict(trace))
    except (CanonicalError, ScheduleError) as exc:
        raise ProtocolError(str(exc)) from exc


def _validate_accounting(value: Any) -> Mapping[str, Any]:
    accounting = exact_keys(value, _ACCOUNTING_KEYS, "result.sample_accounting")
    expected = {
        "trace_samples": TOTAL_SAMPLES,
        "burn_in_samples": BURN_IN_SAMPLES,
        "main_samples": MAIN_SAMPLES,
        "analysis_samples": MAIN_SAMPLES,
        "dropped_samples": 0,
    }
    for key, registered in expected.items():
        exact_int64(accounting[key], f"result.sample_accounting.{key}", registered)
    if accounting["adaptive_stop"] is not False or accounting["outlier_deletion"] is not False:
        raise ProtocolError("adaptive stop and outlier deletion must remain false")
    return accounting


def _validate_metrics(value: Any) -> None:
    metrics = exact_keys(value, _METRICS_KEYS, "result.metrics")
    trajectory = exact_keys(metrics["trajectory"], _TRAJECTORY_KEYS, "result.metrics.trajectory")
    changepoint = exact_keys(metrics["changepoint"], _CHANGEPOINT_KEYS, "result.metrics.changepoint")
    acf = exact_keys(metrics["acf"], _ACF_KEYS, "result.metrics.acf")
    adherence = exact_keys(metrics["gap_adherence"], _GAP_ADHERENCE_KEYS, "result.metrics.gap_adherence")
    int64(
        changepoint["split"],
        "result.metrics.changepoint.split",
        minimum=CHANGEPOINT_MIN_SPLIT,
        maximum=CHANGEPOINT_MAX_SPLIT,
    )
    nonnegative_int64(
        adherence["max_overshoot_ns"],
        "result.metrics.gap_adherence.max_overshoot_ns",
        maximum=MAX_GAP_OVERSHOOT_NS,
    )
    if metrics["transform"] != "natural_log_ns":
        raise ProtocolError("result.metrics.transform is not registered")
    _sha256(metrics["residual_sha256"], "result.metrics.residual_sha256")
    _sha256(metrics["actual_gap_sha256"], "result.metrics.actual_gap_sha256")
    finite_number(trajectory["slope_per_second"], "result.metrics.trajectory.slope_per_second")
    finite_number(trajectory["effect_ratio"], "result.metrics.trajectory.effect_ratio")
    finite_number(
        trajectory["observed_span_seconds"],
        "result.metrics.trajectory.observed_span_seconds",
        minimum=0.0,
    )
    finite_number(changepoint["effect_ratio"], "result.metrics.changepoint.effect_ratio")
    for lag in ACF_LAGS:
        finite_number(acf[f"lag{lag}"], f"result.metrics.acf.lag{lag}")
    finite_number(metrics["effective_sample_size"], "result.metrics.effective_sample_size", minimum=1.0)
    finite_number(metrics["pace_effect_ratio"], "result.metrics.pace_effect_ratio")
    finite_number(metrics["tail_ratio"], "result.metrics.tail_ratio", minimum=1.0)
    finite_number(
        adherence["median_overshoot_ns"],
        "result.metrics.gap_adherence.median_overshoot_ns",
        minimum=0.0,
    )


def _validate_gate_results(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(GATE_LIMITS):
        raise ProtocolError("result.gates is not closed")
    for name, (operator, limit) in GATE_LIMITS.items():
        gate = exact_keys(value[name], _GATE_RESULT_KEYS, f"result.gates.{name}")
        if gate["status"] not in {"pass", "fail"}:
            raise ProtocolError(f"result.gates.{name}.status is invalid")
        observed = finite_number(gate["observed"], f"result.gates.{name}.observed", minimum=0.0)
        registered_limit = finite_number(gate["limit"], f"result.gates.{name}.limit", minimum=0.0)
        if gate["operator"] != operator or registered_limit != limit:
            raise ProtocolError(f"result.gates.{name} differs from registration")
        expected_status = "pass" if (observed <= limit if operator == "<=" else observed >= limit) else "fail"
        if gate["status"] != expected_status:
            raise ProtocolError(f"result.gates.{name}.status is inconsistent")


def _validate_result_structure(
    manifest: Mapping[str, Any],
    value: Any,
    *,
    trace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        checked_manifest = validate_manifest(manifest)
        result = exact_keys(value, _RESULT_KEYS, "result")
        exact_int64(result["schema_version"], "result.schema_version", SCHEMA_VERSION)
        status = result["status"]
        if not isinstance(status, str) or status not in RESULT_STATUSES:
            raise ProtocolError("result.status is not registered")
        if status == SESSION_COMPLETE_STATUS:
            _validate_accounting(result["sample_accounting"])
            _validate_metrics(result["metrics"])
            _validate_gate_results(result["gates"])
        fixed = {
            "schema_version": SCHEMA_VERSION,
            "phase": PHASE,
            "study": STUDY,
            "run_id": checked_manifest["run_id"],
            "manifest_sha256": canonical_sha256(checked_manifest),
            "action": "no_h0_conclusion",
        }
        for key, expected in fixed.items():
            if result[key] != expected:
                raise ProtocolError(f"result.{key} does not match its fixed binding")
        if result["h0_reclassification"] is not False or result["promotion_applicable"] is not False:
            raise ProtocolError("H0.1 may not reclassify H0 or authorize promotion")
        if status == SESSION_INVALID_STATUS:
            if result["conclusion"] != "invalid_input" or result["trace_sha256"] is not None:
                raise ProtocolError("invalid session result conclusion or trace hash is inconsistent")
            error = exact_keys(result["error"], _ERROR_KEYS, "result.error")
            bounded_text(error["code"], "result.error.code", maximum=64)
            bounded_text(error["message"], "result.error.message", maximum=512)
            if any(result[key] is not None for key in ("sample_accounting", "metrics", "gates")):
                raise ProtocolError("invalid session result must not expose partial analysis")
        else:
            if result["conclusion"] != "session_characterized" or result["error"] is not None:
                raise ProtocolError("complete session result conclusion or error is inconsistent")
            _sha256(result["trace_sha256"], "result.trace_sha256")
            if trace is not None:
                checked_trace = validate_trace(checked_manifest, trace)
                if result["trace_sha256"] != canonical_sha256(checked_trace):
                    raise ProtocolError("result.trace_sha256 does not match trace")
        _sha256(result["decision_sha256"], "result.decision_sha256")
        body = {key: result[key] for key in result if key != "decision_sha256"}
        if result["decision_sha256"] != canonical_sha256(body):
            raise ProtocolError("result.decision_sha256 does not replay")
        canonical_json_bytes(result)
        return copy.deepcopy(dict(result))
    except (CanonicalError, ScheduleError) as exc:
        raise ProtocolError(str(exc)) from exc


def validate_result(
    value: Any,
    manifest: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay analysis from manifest+trace and require exact canonical equality."""

    supplied = _validate_result_structure(manifest, value)
    from .analysis import analyze_trace

    expected = analyze_trace(manifest, trace)
    _validate_result_structure(manifest, expected)
    if supplied["status"] == SESSION_COMPLETE_STATUS:
        _validate_result_structure(manifest, supplied, trace=trace)
    if canonical_json_bytes(supplied) != canonical_json_bytes(expected):
        raise ProtocolError("result replay differs from analyze_trace(manifest, trace)")
    return expected


__all__ = [
    "ProtocolError",
    "build_manifest",
    "build_trace",
    "validate_manifest",
    "validate_result",
    "validate_trace",
]
