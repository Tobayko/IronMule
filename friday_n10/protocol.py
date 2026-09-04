"""Closed, replayable N10-v1 preregistration and study-result contracts."""

from __future__ import annotations

import copy
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import (
    CanonicalError,
    bounded_text,
    canonical_json_bytes,
    canonical_sha256,
    exact_keys,
    finite_number,
    int64,
)
from .constants import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEEDS,
    CALIBRATION,
    CANDIDATE_COOLDOWN_SECONDS,
    CANDIDATE_COUNT,
    CONFIDENCE_LEVEL,
    CONFIRMATION,
    CONTINUOUS_GPU_LIMIT_SECONDS,
    DTYPE,
    DUTY_CYCLE_LIMIT,
    DUTY_WINDOW_SECONDS,
    FIXTURE_SEED,
    GPU_WORK_LIMIT_SECONDS,
    INTER_SESSION_COOLDOWN_SECONDS,
    MAXIMUM_CALIBRATED_MDE,
    MEASUREMENT_BLOCKS,
    MINIMUM_EFFECT_FLOOR,
    N_MATMULS,
    OPERAND_SEED,
    PHASE,
    REQUIRED_BREAK_SECONDS,
    SCHEMA_VERSION,
    SESSION_ORDER,
    SESSION_SPECS,
    SHAPE,
    STAGES,
    STUDY_ID,
    STUDY_NAME,
    WALL_LIMIT_SECONDS,
    WARMUP_PAIRS,
)
from .statistics import balanced_orders, hierarchical_bootstrap, session_metrics


class ProtocolError(ValueError):
    """A supplied object violates the sealed N10-v1 protocol."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_KEYS = frozenset({"block_index", "order", "a_ns", "b_ns"})
_CORRECTNESS_KEYS = frozenset(
    {
        "status",
        "max_abs_error",
        "reference_sha256",
        "a_output_sha256",
        "b_output_sha256",
    }
)
_BUDGET_KEYS = frozenset(
    {
        "passed",
        "gpu_work_seconds",
        "max_continuous_gpu_seconds",
        "cooldown_seconds",
        "required_break_seconds",
        "wall_seconds",
        "gpu_work_limit_seconds",
        "continuous_gpu_limit_seconds",
        "duty_cycle_limit",
        "wall_limit_seconds",
        "candidate_cooldown_seconds",
        "required_break_limit_seconds",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "cpu_process_ns",
        "rss_peak_bytes",
        "mlx_active_memory_bytes",
        "mlx_peak_memory_bytes",
        "mlx_cache_memory_bytes",
    }
)
_SESSION_KEYS = frozenset(
    {
        "schema_version",
        "phase",
        "study_id",
        "kind",
        "stage",
        "session_id",
        "cohort",
        "cohort_index",
        "preregistration_sha256",
        "confirmation_seal_sha256",
        "provenance_sha256",
        "power_source",
        "warmups",
        "measurements",
        "metrics",
        "correctness",
        "budget",
        "resources",
        "status",
        "formal_claim",
        "session_sha256",
    }
)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    digest = body.pop(field, None)
    _sha256(digest, field)
    expected = canonical_sha256(body)
    if digest != expected:
        raise ProtocolError(f"{field} does not replay")
    return digest


def _same(left: Any, right: Any, name: str) -> None:
    try:
        if canonical_json_bytes(left) != canonical_json_bytes(right):
            raise ProtocolError(f"{name} does not replay exactly")
    except CanonicalError as exc:
        raise ProtocolError(f"{name} is not canonical") from exc


def study_specification() -> dict[str, Any]:
    """Return the immutable prospective design, independent of live provenance."""

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "study_name": STUDY_NAME,
        "scope": {
            "device_count": 1,
            "operation_count": 1,
            "cross_device_claim": False,
            "end_to_end_model_claim": False,
            "claim_scope": "one-device-one-workload-one-execution-plan",
        },
        "hypothesis": {
            "primary_endpoint": "paired synchronized duration ratio B_over_A",
            "direction": "two_sided_with_gain_regression_and_equivalence_regions",
            "candidate_count": CANDIDATE_COUNT,
            "candidate_selected_from_prior_data": True,
            "selected_candidate": 10,
            "selection_record_id": "5d104d15eea14e82d6d90dc6d28de543858dcc73826a87f4e4c717ee1f24c26a",
            "selection_database_sha256": "70cbe45b846f3f06da57d5a7dd0a56270aab656dd1269df5737151053a0a6d91",
            "selection_snapshot_revision": "c3d1310e7b41ffb984e46cb8759018b9f52d0637cb2474a8d731ad9e52134e2b",
            "selection_method": "one_closed_gemma_round_then_deterministic_harness_ranking",
            "selection_model": "mlx-community/gemma-3-4b-it-4bit",
            "selection_model_revision": "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
            "model_excluded_from_confirmation": True,
            "fresh_confirmation_data_required": True,
        },
        "workload": {
            "operation": "matmul",
            "dtype": DTYPE,
            "lhs_shape": [SHAPE, SHAPE],
            "rhs_shape": [SHAPE, SHAPE],
            "output_shape": [SHAPE, SHAPE],
            "rhs_count": N_MATMULS,
            "fixture_seed": FIXTURE_SEED,
            "operand_seed": OPERAND_SEED,
            "baseline_plan": "serial_per_op_eval_and_sync",
            "candidate_plan": "enqueue_all_then_single_eval_and_sync",
        },
        "calibration": {
            "design": "A/A",
            "arm_a": "serial_callable_a",
            "arm_b": "serial_callable_b",
            "session_order": list(SESSION_ORDER),
            "sessions_per_cohort": 3,
            "mde_formula": "max(0.05,2*sd(session_ratio)*sqrt(2/3))",
            "maximum_mde": MAXIMUM_CALIBRATED_MDE,
        },
        "confirmation": {
            "design": "A/B",
            "arm_a": "serial_per_op_eval_and_sync",
            "arm_b": "enqueue_all_then_single_eval_and_sync",
            "session_order": list(SESSION_ORDER),
            "sessions_per_cohort": 3,
            "candidate_family": ["dispatch_n10_single_terminal_sync"],
            "positive_gate": "upper_C_and_upper_V_and_upper_all_below_1_minus_MDE",
            "negative_gate": "lower_C_and_lower_V_and_lower_all_above_1_plus_MDE",
            "equivalence_gate": "all_three_intervals_inside_1_plus_or_minus_MDE",
        },
        "schedule": {
            "warmup_pairs": WARMUP_PAIRS,
            "measurement_blocks": MEASUREMENT_BLOCKS,
            "arm_order": "balanced_sha256_fisher_yates_v1",
            "seed_derivation": "sha256_first_64_bits_big_endian_clear_msb",
            "seed_domain": "project-friday:h2-n10-v1:<label>",
            "session_seeds": {
                session_id: SESSION_SPECS[session_id][2]
                for session_id in SESSION_ORDER
            },
            "separate_process_per_session": True,
            "inter_session_cooldown_seconds": INTER_SESSION_COOLDOWN_SECONDS,
            "optional_stopping": False,
            "retry_failed_session": False,
        },
        "statistics": {
            "session_estimator": "exp(median(log(B_ns/A_ns)))",
            "study_estimator": "median(session_log_medians)",
            "interval": "deterministic_hierarchical_percentile_bootstrap",
            "confidence": CONFIDENCE_LEVEL,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seeds": dict(BOOTSTRAP_SEEDS),
            "minimum_effect_floor": MINIMUM_EFFECT_FLOOR,
            "multiplicity": "one_candidate_one_primary_endpoint_no_adjustment",
        },
        "correctness": {
            "requirement": "byte_identical_outputs",
            "maximum_absolute_error": 0.0,
            "timing_eligible_only_after_correctness": True,
        },
        "budgets": {
            "gpu_work_limit_seconds": GPU_WORK_LIMIT_SECONDS,
            "continuous_gpu_limit_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
            "required_break_seconds": REQUIRED_BREAK_SECONDS,
            "duty_window_seconds": DUTY_WINDOW_SECONDS,
            "duty_cycle_limit": DUTY_CYCLE_LIMIT,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "candidate_cooldown_seconds": CANDIDATE_COOLDOWN_SECONDS,
            "power_source": "ac_power",
        },
        "decision_policy": {
            "calibration_precedes_confirmation": True,
            "confirmation_requires_seal": True,
            "all_six_sessions_required": True,
            "failed_attempt_is_terminal": True,
            "no_custom_kernel": True,
            "no_generated_code": True,
        },
    }


def build_preregistration(provenance_sha256: str) -> dict[str, Any]:
    provenance = _sha256(provenance_sha256, "provenance_sha256")
    spec = study_specification()
    result = {
        "kind": "preregistration",
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "study_specification": spec,
        "study_spec_sha256": canonical_sha256(spec),
        "provenance_sha256": provenance,
        "status": "sealed_before_measurement",
        "formal_claim": False,
    }
    result["preregistration_sha256"] = canonical_sha256(result)
    return result


def validate_preregistration(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("preregistration must be an object")
    provenance = _sha256(value.get("provenance_sha256"), "provenance_sha256")
    expected = build_preregistration(provenance)
    _same(value, expected, "preregistration")
    return copy.deepcopy(expected)


def orders_for(stage: str, session_id: str, *, warmup: bool) -> list[str]:
    if stage not in STAGES or session_id not in SESSION_SPECS:
        raise ProtocolError("unknown stage or session")
    seed = SESSION_SPECS[session_id][2]
    count = WARMUP_PAIRS if warmup else MEASUREMENT_BLOCKS
    return balanced_orders(
        count,
        seed=seed,
        domain=f"n10v1:{stage}:{session_id}:{'warmup' if warmup else 'measurement'}",
    )


def _validate_blocks(
    value: Any, *, stage: str, session_id: str, warmup: bool
) -> list[dict[str, Any]]:
    expected_orders = orders_for(stage, session_id, warmup=warmup)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProtocolError("timing blocks must be an array")
    if len(value) != len(expected_orders):
        raise ProtocolError("timing block count is not preregistered")
    result: list[dict[str, Any]] = []
    for index, (raw, order) in enumerate(zip(value, expected_orders, strict=True)):
        block = exact_keys(raw, _BLOCK_KEYS, f"block[{index}]")
        if int64(block["block_index"], f"block[{index}].block_index", minimum=0) != index:
            raise ProtocolError("timing blocks are reordered")
        if block["order"] != order:
            raise ProtocolError("arm order does not match the sealed schedule")
        a_ns = int64(block["a_ns"], f"block[{index}].a_ns", minimum=1)
        b_ns = int64(block["b_ns"], f"block[{index}].b_ns", minimum=1)
        result.append({"block_index": index, "order": order, "a_ns": a_ns, "b_ns": b_ns})
    return result


def _validate_correctness(value: Any) -> dict[str, Any]:
    raw = exact_keys(value, _CORRECTNESS_KEYS, "correctness")
    if raw["status"] != "byte_identical":
        raise ProtocolError("only byte-identical output can enter timing")
    if finite_number(raw["max_abs_error"], "max_abs_error", minimum=0.0) != 0.0:
        raise ProtocolError("correctness error must be exactly zero")
    reference = _sha256(raw["reference_sha256"], "reference_sha256")
    a_digest = _sha256(raw["a_output_sha256"], "a_output_sha256")
    b_digest = _sha256(raw["b_output_sha256"], "b_output_sha256")
    if len({reference, a_digest, b_digest}) != 1:
        raise ProtocolError("byte-identical outputs must share one digest")
    return {
        "status": "byte_identical",
        "max_abs_error": 0.0,
        "reference_sha256": reference,
        "a_output_sha256": a_digest,
        "b_output_sha256": b_digest,
    }


def _validate_budget(value: Any) -> dict[str, Any]:
    raw = exact_keys(value, _BUDGET_KEYS, "budget")
    if raw["passed"] is not True:
        raise ProtocolError("a failed hardware budget cannot enter a session")
    numeric = {
        key: finite_number(raw[key], f"budget.{key}", minimum=0.0)
        for key in _BUDGET_KEYS - {"passed"}
    }
    fixed = {
        "gpu_work_limit_seconds": GPU_WORK_LIMIT_SECONDS,
        "continuous_gpu_limit_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
        "duty_cycle_limit": DUTY_CYCLE_LIMIT,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "candidate_cooldown_seconds": CANDIDATE_COOLDOWN_SECONDS,
        "required_break_limit_seconds": REQUIRED_BREAK_SECONDS,
    }
    if any(numeric[key] != expected for key, expected in fixed.items()):
        raise ProtocolError("hardware budget limits differ from the preregistration")
    if numeric["gpu_work_seconds"] > GPU_WORK_LIMIT_SECONDS:
        raise ProtocolError("GPU work budget was exceeded")
    if numeric["max_continuous_gpu_seconds"] > CONTINUOUS_GPU_LIMIT_SECONDS:
        raise ProtocolError("continuous GPU work budget was exceeded")
    if numeric["wall_seconds"] > WALL_LIMIT_SECONDS:
        raise ProtocolError("wall budget was exceeded")
    return {"passed": True, **numeric}


def _validate_resources(value: Any) -> dict[str, Any]:
    raw = exact_keys(value, _RESOURCE_KEYS, "resources")
    result: dict[str, Any] = {
        "cpu_process_ns": int64(raw["cpu_process_ns"], "cpu_process_ns", minimum=0),
        "rss_peak_bytes": int64(raw["rss_peak_bytes"], "rss_peak_bytes", minimum=0),
    }
    for name in (
        "mlx_active_memory_bytes",
        "mlx_peak_memory_bytes",
        "mlx_cache_memory_bytes",
    ):
        item = raw[name]
        result[name] = None if item is None else int64(item, name, minimum=0)
    return result


def build_session_payload(
    *,
    stage: str,
    session_id: str,
    preregistration_sha256: str,
    confirmation_seal_sha256: str | None,
    provenance_sha256: str,
    power_source: str,
    warmups: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    correctness: Mapping[str, Any],
    budget: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in STAGES or session_id not in SESSION_SPECS:
        raise ProtocolError("unknown stage or session")
    if power_source != "ac_power":
        raise ProtocolError("mains power is required")
    prereg = _sha256(preregistration_sha256, "preregistration_sha256")
    provenance = _sha256(provenance_sha256, "provenance_sha256")
    if stage == CALIBRATION:
        if confirmation_seal_sha256 is not None:
            raise ProtocolError("calibration cannot bind a confirmation seal")
        seal = None
        kind = "calibration_session"
    else:
        seal = _sha256(confirmation_seal_sha256, "confirmation_seal_sha256")
        kind = "confirmation_session"
    checked_warmups = _validate_blocks(
        warmups, stage=stage, session_id=session_id, warmup=True
    )
    checked_measurements = _validate_blocks(
        measurements, stage=stage, session_id=session_id, warmup=False
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "kind": kind,
        "stage": stage,
        "session_id": session_id,
        "cohort": SESSION_SPECS[session_id][0],
        "cohort_index": SESSION_SPECS[session_id][1],
        "preregistration_sha256": prereg,
        "confirmation_seal_sha256": seal,
        "provenance_sha256": provenance,
        "power_source": "ac_power",
        "warmups": checked_warmups,
        "measurements": checked_measurements,
        "metrics": session_metrics(checked_measurements),
        "correctness": _validate_correctness(correctness),
        "budget": _validate_budget(budget),
        "resources": _validate_resources(resources),
        "status": "session_complete",
        "formal_claim": False,
    }
    result["session_sha256"] = canonical_sha256(result)
    return result


def validate_session(value: Any) -> dict[str, Any]:
    raw = exact_keys(value, _SESSION_KEYS, "session")
    stage = bounded_text(raw["stage"], "stage", maximum=32)
    session_id = bounded_text(raw["session_id"], "session_id", maximum=8)
    rebuilt = build_session_payload(
        stage=stage,
        session_id=session_id,
        preregistration_sha256=raw["preregistration_sha256"],
        confirmation_seal_sha256=raw["confirmation_seal_sha256"],
        provenance_sha256=raw["provenance_sha256"],
        power_source=raw["power_source"],
        warmups=raw["warmups"],
        measurements=raw["measurements"],
        correctness=raw["correctness"],
        budget=raw["budget"],
        resources=raw["resources"],
    )
    _same(value, rebuilt, "session")
    return rebuilt


def _ordered_sessions(values: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
    if len(values) != len(SESSION_ORDER):
        raise ProtocolError("study stage requires exactly six sessions")
    checked = [validate_session(value) for value in values]
    if [value["session_id"] for value in checked] != list(SESSION_ORDER):
        raise ProtocolError("sessions are missing, duplicated, or reordered")
    if any(value["stage"] != stage for value in checked):
        raise ProtocolError("study stage mixes calibration and confirmation")
    prereg = {value["preregistration_sha256"] for value in checked}
    provenance = {value["provenance_sha256"] for value in checked}
    if len(prereg) != 1 or len(provenance) != 1:
        raise ProtocolError("sessions do not share one sealed provenance")
    return checked


def build_calibration_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = _ordered_sessions(values, CALIBRATION)
    ratios = [float(value["metrics"]["ratio"]) for value in sessions]
    ratio_sd = statistics.stdev(ratios)
    raw_mde = 2.0 * ratio_sd * math.sqrt(2.0 / 3.0)
    mde = max(MINIMUM_EFFECT_FLOOR, raw_mde)
    aggregate = hierarchical_bootstrap(
        [value["measurements"] for value in sessions],
        seed=BOOTSTRAP_SEEDS["calibration_all"],
        draws=BOOTSTRAP_DRAWS,
        confidence=CONFIDENCE_LEVEL,
    )
    gates = {
        "aggregate_interval_contains_one": bool(
            aggregate["ci_low"] <= 1.0 <= aggregate["ci_high"]
        ),
        "aggregate_bias_within_floor": abs(float(aggregate["ratio"]) - 1.0)
        <= MINIMUM_EFFECT_FLOOR,
        "mde_within_cap": mde <= MAXIMUM_CALIBRATED_MDE,
        "all_sessions_byte_identical": all(
            value["correctness"]["status"] == "byte_identical" for value in sessions
        ),
    }
    passed = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "kind": "calibration_summary",
        "stage": CALIBRATION,
        "preregistration_sha256": sessions[0]["preregistration_sha256"],
        "provenance_sha256": sessions[0]["provenance_sha256"],
        "session_sha256": [value["session_sha256"] for value in sessions],
        "session_ratios": ratios,
        "session_ratio_sd": ratio_sd,
        "raw_mde": raw_mde,
        "mde": mde,
        "mde_floor": MINIMUM_EFFECT_FLOOR,
        "mde_cap": MAXIMUM_CALIBRATED_MDE,
        "aggregate": aggregate,
        "gates": gates,
        "status": "calibration_passed" if passed else "calibration_failed",
        "formal_claim": False,
    }
    result["calibration_summary_sha256"] = canonical_sha256(result)
    return result


def validate_calibration_summary(
    value: Any, sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("calibration summary must be an object")
    _self_hash(value, "calibration_summary_sha256")
    expected = build_calibration_summary(sessions)
    _same(value, expected, "calibration summary")
    return expected


def build_confirmation_seal(
    calibration: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    checked = validate_calibration_summary(calibration, sessions)
    if checked["status"] != "calibration_passed":
        raise ProtocolError("failed calibration cannot open confirmation")
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "kind": "confirmation_seal",
        "stage": CONFIRMATION,
        "preregistration_sha256": checked["preregistration_sha256"],
        "calibration_summary_sha256": checked["calibration_summary_sha256"],
        "provenance_sha256": checked["provenance_sha256"],
        "candidate_id": "dispatch_n10_single_terminal_sync",
        "candidate_count": CANDIDATE_COUNT,
        "mde": checked["mde"],
        "session_order": list(SESSION_ORDER),
        "status": "confirmation_sealed",
        "formal_claim": False,
    }
    result["confirmation_seal_sha256"] = canonical_sha256(result)
    return result


def validate_confirmation_seal(
    value: Any,
    calibration: Mapping[str, Any],
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("confirmation seal must be an object")
    _self_hash(value, "confirmation_seal_sha256")
    expected = build_confirmation_seal(calibration, sessions)
    _same(value, expected, "confirmation seal")
    return expected


def _inside_interval(interval: Mapping[str, Any], low: float, high: float) -> bool:
    return float(interval["ci_low"]) >= low and float(interval["ci_high"]) <= high


def build_study_decision(
    values: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    sessions = _ordered_sessions(values, CONFIRMATION)
    seal_digest = _sha256(seal.get("confirmation_seal_sha256"), "confirmation_seal_sha256")
    if any(value["confirmation_seal_sha256"] != seal_digest for value in sessions):
        raise ProtocolError("confirmation sessions do not bind the supplied seal")
    if sessions[0]["preregistration_sha256"] != seal.get("preregistration_sha256"):
        raise ProtocolError("confirmation seal and sessions bind different preregistrations")
    if sessions[0]["provenance_sha256"] != seal.get("provenance_sha256"):
        raise ProtocolError("confirmation seal and sessions bind different provenance")
    mde = finite_number(seal.get("mde"), "sealed mde", minimum=MINIMUM_EFFECT_FLOOR)
    characterization = [
        value for value in sessions if value["cohort"] == "characterization"
    ]
    validation = [value for value in sessions if value["cohort"] == "validation"]
    intervals = {
        "characterization": hierarchical_bootstrap(
            [value["measurements"] for value in characterization],
            seed=BOOTSTRAP_SEEDS["confirmation_characterization"],
            draws=BOOTSTRAP_DRAWS,
            confidence=CONFIDENCE_LEVEL,
        ),
        "validation": hierarchical_bootstrap(
            [value["measurements"] for value in validation],
            seed=BOOTSTRAP_SEEDS["confirmation_validation"],
            draws=BOOTSTRAP_DRAWS,
            confidence=CONFIDENCE_LEVEL,
        ),
        "all": hierarchical_bootstrap(
            [value["measurements"] for value in sessions],
            seed=BOOTSTRAP_SEEDS["confirmation_all"],
            draws=BOOTSTRAP_DRAWS,
            confidence=CONFIDENCE_LEVEL,
        ),
    }
    low = 1.0 - mde
    high = 1.0 + mde
    gain = all(float(interval["ci_high"]) < low for interval in intervals.values())
    regression = all(float(interval["ci_low"]) > high for interval in intervals.values())
    equivalent = all(_inside_interval(interval, low, high) for interval in intervals.values())
    if gain:
        status = "n10_gain_confirmed"
        claim = "n10_batched_dispatch_is_faster_beyond_mde"
        action = "permit_bounded_n10_runtime_prototype"
    elif regression:
        status = "n10_regression_confirmed"
        claim = "n10_batched_dispatch_is_slower_beyond_mde"
        action = "reject_candidate"
    elif equivalent:
        status = "n10_equivalent_within_mde"
        claim = "n10_plans_are_equivalent_within_mde"
        action = "reject_candidate"
    else:
        status = "n10_inconclusive"
        claim = "n10_no_confirmatory_direction_cleared_all_split_gates"
        action = "stop_without_promotion"
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "kind": "study_decision",
        "stage": CONFIRMATION,
        "preregistration_sha256": sessions[0]["preregistration_sha256"],
        "confirmation_seal_sha256": seal_digest,
        "provenance_sha256": sessions[0]["provenance_sha256"],
        "session_sha256": [value["session_sha256"] for value in sessions],
        "mde": mde,
        "equivalence_bounds": {"low": low, "high": high},
        "intervals": intervals,
        "effect_percent": 100.0 * (float(intervals["all"]["ratio"]) - 1.0),
        "gates": {
            "gain_all_splits": gain,
            "regression_all_splits": regression,
            "equivalence_all_splits": equivalent,
            "all_sessions_byte_identical": all(
                value["correctness"]["status"] == "byte_identical" for value in sessions
            ),
        },
        "status": status,
        "claim": claim,
        "action": action,
        "claim_scope": "one-device-one-workload-one-execution-plan",
        "limitations": [
            "single_device",
            "single_fp16_shape",
            "exactly_ten_matmuls",
            "dispatch_plan_only",
            "candidate_selected_from_prior_exploration",
            "no_model_end_to_end_claim",
            "no_cross_device_claim",
        ],
        "formal_claim": True,
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def validate_study_decision(
    value: Any,
    sessions: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("study decision must be an object")
    _self_hash(value, "decision_sha256")
    expected = build_study_decision(sessions, seal)
    _same(value, expected, "study decision")
    return expected


def build_session_failure(
    *,
    stage: str,
    session_id: str,
    preregistration_sha256: str,
    confirmation_seal_sha256: str | None,
    provenance_sha256: str,
    failure_type: str,
) -> dict[str, Any]:
    if stage not in STAGES or session_id not in SESSION_SPECS:
        raise ProtocolError("unknown failed stage or session")
    if stage == CONFIRMATION:
        seal = _sha256(confirmation_seal_sha256, "confirmation_seal_sha256")
    elif confirmation_seal_sha256 is not None:
        raise ProtocolError("calibration failure cannot bind a confirmation seal")
    else:
        seal = None
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "study_id": STUDY_ID,
        "kind": "session_failure",
        "stage": stage,
        "session_id": session_id,
        "preregistration_sha256": _sha256(
            preregistration_sha256, "preregistration_sha256"
        ),
        "confirmation_seal_sha256": seal,
        "provenance_sha256": _sha256(provenance_sha256, "provenance_sha256"),
        "failure_type": bounded_text(failure_type, "failure_type", maximum=96),
        "status": "measurement_failed_terminal",
        "formal_claim": False,
    }
    result["failure_sha256"] = canonical_sha256(result)
    return result


def validate_session_failure(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("session failure must be an object")
    expected = build_session_failure(
        stage=value.get("stage"),
        session_id=value.get("session_id"),
        preregistration_sha256=value.get("preregistration_sha256"),
        confirmation_seal_sha256=value.get("confirmation_seal_sha256"),
        provenance_sha256=value.get("provenance_sha256"),
        failure_type=value.get("failure_type"),
    )
    _same(value, expected, "session failure")
    return expected


def validate_history(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replay a complete or partial append-only study history in insertion order."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ProtocolError("history must be an array")
    if not values:
        return []
    prereg = validate_preregistration(values[0])
    if values[0].get("kind") != "preregistration":
        raise ProtocolError("history must begin with preregistration")
    checked: list[dict[str, Any]] = [prereg]
    calibration_sessions: list[dict[str, Any]] = []
    confirmation_sessions: list[dict[str, Any]] = []
    calibration_summary: dict[str, Any] | None = None
    confirmation_seal: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    terminal_failure = False

    for raw in values[1:]:
        if terminal_failure or decision is not None:
            raise ProtocolError("terminal history cannot accept later records")
        kind = raw.get("kind")
        if kind in {"calibration_session", "confirmation_session"}:
            session = validate_session(raw)
            if session["preregistration_sha256"] != prereg["preregistration_sha256"]:
                raise ProtocolError("session binds a different preregistration")
            if session["provenance_sha256"] != prereg["provenance_sha256"]:
                raise ProtocolError("session provenance differs from preregistration")
            target = calibration_sessions if session["stage"] == CALIBRATION else confirmation_sessions
            if len(target) >= len(SESSION_ORDER):
                raise ProtocolError("study stage already contains all sealed sessions")
            if session["session_id"] != SESSION_ORDER[len(target)]:
                raise ProtocolError("sessions do not follow the sealed order")
            if session["stage"] == CALIBRATION:
                if calibration_summary is not None or confirmation_seal is not None:
                    raise ProtocolError("calibration is already closed")
            else:
                if confirmation_seal is None:
                    raise ProtocolError("confirmation started without a seal")
            target.append(session)
            checked.append(session)
            continue
        if kind == "session_failure":
            failure = validate_session_failure(raw)
            expected_ids = (
                calibration_sessions if failure["stage"] == CALIBRATION else confirmation_sessions
            )
            if len(expected_ids) >= len(SESSION_ORDER):
                raise ProtocolError("failure cannot follow a complete study stage")
            if failure["session_id"] != SESSION_ORDER[len(expected_ids)]:
                raise ProtocolError("failure does not identify the next sealed session")
            if failure["preregistration_sha256"] != prereg["preregistration_sha256"]:
                raise ProtocolError("failure binds a different preregistration")
            if failure["provenance_sha256"] != prereg["provenance_sha256"]:
                raise ProtocolError("failure provenance differs from preregistration")
            if failure["stage"] == CONFIRMATION and confirmation_seal is None:
                raise ProtocolError("confirmation failure occurred before sealing")
            terminal_failure = True
            checked.append(failure)
            continue
        if kind == "calibration_summary":
            if calibration_summary is not None or confirmation_seal is not None:
                raise ProtocolError("calibration summary is duplicated or late")
            calibration_summary = validate_calibration_summary(raw, calibration_sessions)
            checked.append(calibration_summary)
            continue
        if kind == "confirmation_seal":
            if calibration_summary is None or confirmation_seal is not None:
                raise ProtocolError("confirmation seal is missing its unique calibration")
            confirmation_seal = validate_confirmation_seal(
                raw, calibration_summary, calibration_sessions
            )
            checked.append(confirmation_seal)
            continue
        if kind == "study_decision":
            if confirmation_seal is None or decision is not None:
                raise ProtocolError("study decision is missing its unique seal")
            decision = validate_study_decision(raw, confirmation_sessions, confirmation_seal)
            checked.append(decision)
            continue
        raise ProtocolError("history contains an unknown record kind")
    return checked


__all__ = [
    "ProtocolError",
    "build_calibration_summary",
    "build_confirmation_seal",
    "build_preregistration",
    "build_session_failure",
    "build_session_payload",
    "build_study_decision",
    "orders_for",
    "study_specification",
    "validate_calibration_summary",
    "validate_confirmation_seal",
    "validate_history",
    "validate_preregistration",
    "validate_session",
    "validate_session_failure",
    "validate_study_decision",
]
