"""Pure, fail-closed aggregation of the registered H0 A/A sessions.

This module consumes a bounded ``{"manifest": ..., "result": ...}`` envelope for
each process.  It deliberately contains no runtime, worker, storage, or model
code.  All statistical decisions are delegated to :func:`statistics.aa_gate`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .benchmark import (
    H0_BATCH_MAX_NS,
    H0_BATCH_MIN_NS,
    H0_INITIAL_WARMUPS,
    H0_MAX_REPETITIONS,
    H0_MAX_WARMUPS,
    _balanced_order,
)
from .canonical import canonical_json_bytes
from .correctness_contract import (
    CORRECTNESS_CASE_NAMES,
    CORRECTNESS_FULL_KEYS,
    CORRECTNESS_HARD_CAPS,
    MEMORY_METRIC_NAMES,
    MEMORY_MAX_INT,
    SIGN_INVARIANT_KEYS,
    CorrectnessContractError,
    fixture_digest,
    validate_fixed_case,
    validate_memory_limit_contract,
    memory_name_allowed,
    missing_reason_allowed,
    validate_performance_case,
    validate_sign_invariant_case,
)
from .constants import (
    AA_BOOTSTRAP_SEEDS,
    AA_SESSION_SEEDS,
    BOOTSTRAP_REPLICATES,
    ENGINEERING_EQUIVALENCE_BAND,
    SESSION_RATIO_BAND,
)
from .manifest import ManifestError, validate_manifest
from .protocol import PRODUCTION_JSON_DEPTH, PRODUCTION_RESULT_BYTES, ProtocolError, close_manifest, validate_result
from .statistics import StatisticsError, aa_gate


MAX_SESSIONS = 3
MAX_BLOCKS = 30
MAX_INPUT_BYTES = 1 << 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SESSION_KEYS = frozenset({"manifest", "result"})
_RESULT_EVIDENCE_KEYS = frozenset(
    {
        "rss_peak_bytes",
        "rss_missing_reason",
        "benchmark_classification",
        "benchmark_action",
        "aggregation_required",
        "adapter_contract",
        "benchmark_evidence",
    }
)
_BENCHMARK_EVIDENCE_KEYS = frozenset(
    {
        "fixture",
        "correctness",
        "memory",
        "memory_limit",
        "memory_gate",
        "cache_state",
        "fresh_process_required",
        "aggregation_required",
        "compile_wrapper_setup_ns",
        "first_eval_compile_inclusive_ns",
        "total_elapsed_ns",
        "arms",
        "comparison",
        "raw_samples",
    }
)
_ARM_KEYS = frozenset({"baseline", "candidate"})
_BATCH_KEYS = frozenset(
    {"block_index", "batch_ns", "per_eval_ns", "repetitions", "position", "evaluation_ns", "synchronize_ns"}
)
_WARMUP_BLOCK_KEYS = frozenset({
    "block_index", "evaluations", "block_ns", "per_eval_ns",
    "median_eval_ns", "min_eval_ns", "max_eval_ns",
})
_COMPARISON_BLOCK_KEYS = frozenset(
    {"block_index", "first", "second", "baseline_per_eval_ns", "candidate_per_eval_ns", "ratio"}
)
_FIXTURE_KEYS = frozenset({"fixture_seed", "a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"})
_ADAPTER_CONTRACT = {
    "common_result_ready": False,
    "reason": "single-process measurements require aggregation before any global decision",
    "mapping": {
        "runtime_unavailable": "invalid/baseline_fallback",
        "invalid*": "invalid/baseline_fallback",
        "measurement_complete": "aggregation_required",
        "baseline_reference": "not_run",
    },
}
_CORRECTNESS_NAMES = (*CORRECTNESS_CASE_NAMES, "performance_fixture", "sign_invariant")
_CORRECTNESS_METRIC_KEYS = frozenset(
    {
        "abs_q50", "abs_q95", "abs_q99", "abs_max", "rel_q50", "rel_q95", "rel_q99", "rel_max",
        "rel_q99_abs_oracle_ge_1", "normalized_l2", "scaled_normalized_inf",
    }
)
_MEMORY_KEYS = frozenset(
    {"name", "api", "unit", "value", "missing_reason", "measurement_phase", "arm", "measured_at_ns", "reset_state"}
)
_MEMORY_LIMIT_KEYS = frozenset({"attempted", "hard_limit", "applied", "missing_reason"})
_MEMORY_PHASES = frozenset({"before_correctness", "after_calibration", "after_measurement", "after_timing"})
_MEMORY_ARMS = frozenset({"all", "baseline", "candidate"})
_COMPARISON_KEYS = frozenset(
    {
        "order", "blocks", "raw_samples", "ratio_statistics", "benchmark_classification", "action",
        "aggregation_required", "aggregation_gate", "global_decision", "comparison_kind",
    }
)
_RATIO_STAT_KEYS = frozenset({"count", "median_ratio", "mad_ratio", "iqr_ratio", "min_ratio", "max_ratio"})


class AggregationError(ValueError):
    """Raised internally for a malformed or non-evaluable session envelope."""


def _finite(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    # Integer-valued telemetry must stay within the same non-negative signed
    # int64 domain enforced by the runner/worker.  Do this before converting
    # to float: ``float(1 << 63)`` is finite and would otherwise erase the
    # distinction between the accepted maximum and the rejected upper bound.
    if isinstance(value, int) and not isinstance(value, bool) and not 0 <= value <= MEMORY_MAX_INT:
        raise AggregationError(f"{path}:integer_out_of_range")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite:
        raise AggregationError(f"{path}:nonfinite_or_non_numeric")
    number = float(value)
    if positive and number <= 0:
        raise AggregationError(f"{path}:not_positive")
    if nonnegative and number < 0:
        raise AggregationError(f"{path}:negative")
    return number


def _bounded_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MEMORY_MAX_INT:
        raise AggregationError(f"{path}:integer_out_of_range")
    return value


def _median(values: Sequence[int | float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise AggregationError("empty_statistical_sequence")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _mad(values: Sequence[int | float]) -> float:
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _iqr(values: Sequence[int | float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise AggregationError("empty_statistical_sequence")
    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return quantile(0.75) - quantile(0.25)


def _bounded_text(value: Any, limit: int = 256) -> str:
    try:
        text = str(value)
    except Exception:
        text = "aggregation:invalid_input"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _check_input_bounds(value: Any, *, path: str = "input", depth: int = 0, active: set[int] | None = None, nodes: list[int] | None = None) -> None:
    """Reject recursive, wide, huge-integer, and oversized values before JSON encoding."""

    if depth > PRODUCTION_JSON_DEPTH:
        raise AggregationError(f"{path}:depth_limit")
    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > 50_000:
        raise AggregationError("input:node_limit")
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value.encode("utf-8")) > 65_536:
            raise AggregationError(f"{path}:string_limit")
        return
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise AggregationError(f"{path}:integer_limit")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AggregationError(f"{path}:nonfinite")
        return
    if active is None:
        active = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise AggregationError(f"{path}:cycle")
        if len(value) > 10_000:
            raise AggregationError(f"{path}:mapping_limit")
        active.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise AggregationError(f"{path}:non_string_key")
                _check_input_bounds(child, path=f"{path}.{key}", depth=depth + 1, active=active, nodes=nodes)
        finally:
            active.remove(identity)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        identity = id(value)
        if identity in active:
            raise AggregationError(f"{path}:cycle")
        if len(value) > 10_000:
            raise AggregationError(f"{path}:sequence_limit")
        active.add(identity)
        try:
            for index, child in enumerate(value):
                _check_input_bounds(child, path=f"{path}[{index}]", depth=depth + 1, active=active, nodes=nodes)
        finally:
            active.remove(identity)
        return
    raise AggregationError(f"{path}:unsupported_value")


def _invalid(failures: Sequence[str], *, gates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    unique = list(dict.fromkeys(str(reason) for reason in failures))
    return {
        "schema_version": 1,
        "classification": "h0_invalid",
        "h0_valid": False,
        "engineering_equivalence_gate": False,
        "scientific_equivalence_claim": False,
        "promotion_gate_applicable": False,
        "seed_contract": "aa_gate",
        "bootstrap_seed_manifest_bound": False,
        "aggregation_contract_ready": False,
        "live_execution_authorized": False,
        "live_ready": False,
        "live_ready_reason": "live execution authorization is not granted by offline aggregation",
        "action": "baseline_reference",
        "next_step": "repair_and_remeasure",
        "gates": dict(gates or {}),
        "failures": unique[:256],
    }


def _bound_result(result: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = canonical_json_bytes(result)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _invalid(["result:not_canonical_json"])
    if len(payload) < PRODUCTION_RESULT_BYTES:
        return result
    return _invalid(["result:exceeds_production_result_limit"])


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AggregationError(f"{path}:not_object")
    return value


def _validate_memory(evidence: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    gate = evidence.get("memory_gate")
    if gate not in {"aggregation_required", "not_evaluable_missing_required_metric"}:
        raise AggregationError(f"{prefix}:memory_gate_invalid")
    rows = evidence.get("memory")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise AggregationError(f"{prefix}:memory_not_evaluable")
    entries: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    latest: dict[str, tuple[int | None, str | None]] = {}
    for index, row in enumerate(rows):
        item = _require_mapping(row, f"{prefix}.memory[{index}]")
        if set(item) != _MEMORY_KEYS:
            raise AggregationError(f"{prefix}.memory[{index}]:unknown_or_missing_keys")
        name = item.get("name")
        phase = item.get("measurement_phase")
        arm = item.get("arm")
        if (
            not memory_name_allowed(name)
            or not isinstance(phase, str) or phase not in _MEMORY_PHASES
            or not isinstance(arm, str) or arm not in _MEMORY_ARMS
        ):
            raise AggregationError(f"{prefix}.memory[{index}]:missing_name")
        if (phase == "after_calibration") != (arm in {"baseline", "candidate"}):
            raise AggregationError(f"{prefix}.memory[{index}]:phase_arm_semantics_invalid")
        if phase in {"before_correctness", "after_timing"} and arm != "all":
            raise AggregationError(f"{prefix}.memory[{index}]:phase_arm_semantics_invalid")
        identity = (name, phase, arm)
        if identity in identities:
            raise AggregationError(f"{prefix}.memory[{index}]:duplicate_metric")
        identities.add(identity)
        if not isinstance(item.get("api"), str) or not item["api"] or item.get("reset_state") != "not_reset_or_api_unavailable":
            raise AggregationError(f"{prefix}.memory[{index}]:api_or_reset_semantics_invalid")
        if item.get("unit") != "bytes":
            raise AggregationError(f"{prefix}.memory[{index}]:unit_not_bytes")
        value = item.get("value")
        reason = item.get("missing_reason")
        measured_at = item.get("measured_at_ns")
        if (
            isinstance(measured_at, bool)
            or not isinstance(measured_at, int)
            or not 0 < measured_at <= MEMORY_MAX_INT
        ):
            raise AggregationError(f"{prefix}.memory[{index}]:measured_at_invalid")
        if value is not None and reason is not None:
            raise AggregationError(f"{prefix}.memory[{index}]:value_and_missing_reason_both_set")
        if value is None and (not isinstance(reason, str) or not reason):
            raise AggregationError(f"{prefix}.memory[{index}]:missing_reason_required")
        if value is None and not missing_reason_allowed(reason):
            raise AggregationError(f"{prefix}.memory[{index}]:missing_reason_not_registered")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MEMORY_MAX_INT:
                raise AggregationError(f"{prefix}.memory[{index}].value:not_integer")
            latest[name] = (int(_finite(value, f"{prefix}.memory[{index}].value", nonnegative=True)), None)
        else:
            latest[name] = (None, reason)
        entries.append(dict(item))
    required = {"mlx_peak_memory", "rss"}
    available = {name: value for name, (value, _reason) in latest.items() if name in required and value is not None}
    unavailable = sorted(name for name in required if name not in available)
    missing = {name: latest.get(name, (None, "not_recorded"))[1] or "not_recorded" for name in unavailable}
    expected_gate = "aggregation_required" if not unavailable else "not_evaluable_missing_required_metric"
    if gate != expected_gate:
        raise AggregationError(f"{prefix}:memory_gate_not_reconstructed")
    status = "not_evaluable" if unavailable else "evaluable"
    return {
        "status": status,
        "gate": gate,
        "promotion_gate_applicable": False,
        "entries": entries,
        "required_metrics": {name: available[name] for name in sorted(required) if name in available},
        "missing_reasons": {name: missing.get(name, "not_recorded") for name in unavailable},
    }


def _validate_memory_limit(value: Any, prefix: str) -> dict[str, Any]:
    try:
        return validate_memory_limit_contract(value)
    except CorrectnessContractError as exc:
        raise AggregationError(f"{prefix}:{exc}") from exc


def _validate_arm(arm: Mapping[str, Any], prefix: str) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    allowed_arm = {"warmup", "repetitions", "batches", "statistics", "calibration_samples", "raw_samples"}
    if set(arm) != allowed_arm:
        raise AggregationError(f"{prefix}:unknown_or_missing_arm_keys")
    warmup = _require_mapping(arm["warmup"], f"{prefix}.warmup")
    if set(warmup) != {"count", "durations_ns", "stable", "median_ns", "samples", "blocks"}:
        raise AggregationError(f"{prefix}:warmup_contract_invalid")
    if not isinstance(warmup.get("stable"), bool) or not warmup["stable"]:
        raise AggregationError(f"{prefix}:warmup_not_stable")
    count = warmup.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or not H0_INITIAL_WARMUPS <= count <= H0_MAX_WARMUPS:
        raise AggregationError(f"{prefix}:warmup_count_invalid")
    durations = warmup.get("durations_ns")
    if not isinstance(durations, Sequence) or isinstance(durations, (str, bytes, bytearray)) or len(durations) != count:
        raise AggregationError(f"{prefix}:warmup_samples_invalid")
    for index, value in enumerate(durations):
        if _bounded_nonnegative_int(value, f"{prefix}.warmup.durations_ns[{index}]") <= 0:
            raise AggregationError(f"{prefix}.warmup.durations_ns[{index}]:not_positive")
    declared_samples = warmup.get("samples")
    if not isinstance(declared_samples, Sequence) or isinstance(declared_samples, (str, bytes, bytearray)) or len(declared_samples) != count:
        raise AggregationError(f"{prefix}:warmup_declared_samples_invalid")
    for index, sample in enumerate(declared_samples):
        item = _require_mapping(sample, f"{prefix}.warmup.samples[{index}]")
        if set(item) != {"phase", "sample_index", "value", "unit"} or item.get("phase") != "warmup" or item.get("sample_index") != index or item.get("unit") != "ns":
            raise AggregationError(f"{prefix}:warmup_declared_sample_contract_invalid")
        _finite(item.get("value"), f"{prefix}.warmup.samples[{index}].value", positive=True)
        if item["value"] != durations[index]:
            raise AggregationError(f"{prefix}:warmup_declared_sample_mismatch")
    blocks = warmup.get("blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes, bytearray)) or len(blocks) != count:
        raise AggregationError(f"{prefix}:warmup_blocks_invalid")
    for index, block in enumerate(blocks):
        item = _require_mapping(block, f"{prefix}.warmup.blocks[{index}]")
        if set(item) != _WARMUP_BLOCK_KEYS:
            raise AggregationError(f"{prefix}:warmup_block_contract_invalid")
        block_index = _bounded_nonnegative_int(item["block_index"], f"{prefix}.warmup.blocks[{index}].block_index")
        evaluations = _bounded_nonnegative_int(item["evaluations"], f"{prefix}.warmup.blocks[{index}].evaluations")
        block_ns = _bounded_nonnegative_int(item["block_ns"], f"{prefix}.warmup.blocks[{index}].block_ns")
        per_eval_ns = _bounded_nonnegative_int(item["per_eval_ns"], f"{prefix}.warmup.blocks[{index}].per_eval_ns")
        median_eval_ns = _bounded_nonnegative_int(item["median_eval_ns"], f"{prefix}.warmup.blocks[{index}].median_eval_ns")
        min_eval_ns = _bounded_nonnegative_int(item["min_eval_ns"], f"{prefix}.warmup.blocks[{index}].min_eval_ns")
        max_eval_ns = _bounded_nonnegative_int(item["max_eval_ns"], f"{prefix}.warmup.blocks[{index}].max_eval_ns")
        if (
            block_index != index
            or not 1 <= evaluations <= H0_MAX_REPETITIONS
            or block_ns < H0_BATCH_MIN_NS
            or per_eval_ns <= 0
            or not min_eval_ns <= median_eval_ns <= max_eval_ns
            or per_eval_ns != max(1, int(round(block_ns / evaluations)))
            or per_eval_ns != durations[index]
        ):
            raise AggregationError(f"{prefix}:warmup_block_cross_bind_invalid")
    warmup_median = _finite(warmup.get("median_ns"), f"{prefix}.warmup.median_ns", positive=True)
    if warmup_median != _median(durations[-5:]):
        raise AggregationError(f"{prefix}:warmup_median_not_reconstructed")
    if len(durations) >= 5:
        center = sorted(float(value) for value in durations[-5:])[2]
        if any(not 0.95 * center <= float(value) <= 1.05 * center for value in durations[-5:]):
            raise AggregationError(f"{prefix}:warmup_stability_rule_failed")
    repetition = _require_mapping(arm["repetitions"], f"{prefix}.repetitions")
    repetitions = repetition.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or not 1 <= repetitions <= H0_MAX_REPETITIONS:
        raise AggregationError(f"{prefix}:repetitions_invalid")
    if repetitions & (repetitions - 1):
        raise AggregationError(f"{prefix}:repetitions_not_power_of_two")
    batch_ns = _finite(repetition.get("batch_ns"), f"{prefix}.repetitions.batch_ns", positive=True)
    if not H0_BATCH_MIN_NS <= batch_ns <= H0_BATCH_MAX_NS:
        raise AggregationError(f"{prefix}:repetition_window_invalid")
    calibration = repetition.get("calibration_samples")
    if not isinstance(calibration, Sequence) or isinstance(calibration, (str, bytes, bytearray)) or not calibration:
        raise AggregationError(f"{prefix}:calibration_missing")
    for index, sample in enumerate(calibration):
        item = _require_mapping(sample, f"{prefix}.calibration[{index}]")
        _finite(item.get("value"), f"{prefix}.calibration[{index}].value", positive=True)
        if item.get("unit") != "ns" or item.get("phase") != "repetition_probe" or item.get("repetitions") != repetitions:
            raise AggregationError(f"{prefix}:calibration_unit_invalid")
    probes = repetition.get("probe_timings")
    if not isinstance(probes, Sequence) or len(probes) != repetitions:
        raise AggregationError(f"{prefix}:probe_timings_invalid")
    for index, value in enumerate(probes):
        _finite(value, f"{prefix}.probe_timings[{index}]", positive=True)
    batches = arm["batches"]
    if not isinstance(batches, Sequence) or len(batches) != MAX_BLOCKS:
        raise AggregationError(f"{prefix}:block_count_invalid")
    result: list[tuple[float, float]] = []
    positions: dict[int, str] = {}
    for index, batch in enumerate(batches):
        item = _require_mapping(batch, f"{prefix}.batches[{index}]")
        if set(item) != _BATCH_KEYS:
            raise AggregationError(f"{prefix}.batches[{index}]:unknown_or_missing_keys")
        if item["block_index"] != index or item["position"] not in {"first", "second"}:
            raise AggregationError(f"{prefix}.batches[{index}]:block_or_position_invalid")
        elapsed = _finite(item["batch_ns"], f"{prefix}.batches[{index}].batch_ns", positive=True)
        per_eval = _finite(item["per_eval_ns"], f"{prefix}.batches[{index}].per_eval_ns", positive=True)
        if item["repetitions"] != repetitions or not math.isclose(per_eval * repetitions, elapsed, rel_tol=1e-12, abs_tol=1e-6):
            raise AggregationError(f"{prefix}.batches[{index}]:repetition_binding_invalid")
        for field in ("evaluation_ns", "synchronize_ns"):
            values = item[field]
            if not isinstance(values, Sequence) or len(values) != repetitions:
                raise AggregationError(f"{prefix}.batches[{index}].{field}:invalid")
            for subindex, value in enumerate(values):
                _finite(value, f"{prefix}.batches[{index}].{field}[{subindex}]", positive=True)
        positions[index] = item["position"]
        result.append((per_eval, elapsed))
    raw = arm["raw_samples"]
    if not isinstance(raw, Sequence) or len(raw) > 10_000:
        raise AggregationError(f"{prefix}:raw_sample_limit")
    seen: set[tuple[int, str]] = set()
    calibration_seen: set[tuple[str, int]] = set()
    warmup_raw: list[Mapping[str, Any]] = []
    probe_raw: list[Mapping[str, Any]] = []
    for index, sample in enumerate(raw):
        item = _require_mapping(sample, f"{prefix}.raw_samples[{index}]")
        phase = item.get("phase")
        if phase == "warmup":
            if set(item) != {"phase", "sample_index", "value", "unit", "arm", "position"} or item["arm"] != prefix.rsplit(".", 1)[-1] or item["position"] != "calibration":
                raise AggregationError(f"{prefix}:warmup_raw_contract_invalid")
            sample_index = item["sample_index"]
            if isinstance(sample_index, bool) or not isinstance(sample_index, int) or not 0 <= sample_index < count or (phase, sample_index) in calibration_seen:
                raise AggregationError(f"{prefix}:warmup_raw_index_invalid")
            if sample_index != len(warmup_raw):
                raise AggregationError(f"{prefix}:warmup_raw_order_invalid")
            _finite(item["value"], f"{prefix}.raw_samples[{index}].value", positive=True)
            if item["unit"] != "ns":
                raise AggregationError(f"{prefix}:warmup_raw_unit_invalid")
            calibration_seen.add((phase, sample_index))
            warmup_raw.append(item)
            continue
        if phase == "repetition_probe":
            if set(item) != {"phase", "repetitions", "sample_index", "value", "unit", "arm", "position"} or item["arm"] != prefix.rsplit(".", 1)[-1] or item["position"] != "calibration":
                raise AggregationError(f"{prefix}:probe_raw_contract_invalid")
            sample_index = item["sample_index"]
            if isinstance(sample_index, bool) or not isinstance(sample_index, int) or not 0 <= sample_index < repetitions or item["repetitions"] != repetitions or (phase, sample_index) in calibration_seen:
                raise AggregationError(f"{prefix}:probe_raw_index_invalid")
            if sample_index != len(probe_raw):
                raise AggregationError(f"{prefix}:probe_raw_order_invalid")
            _finite(item["value"], f"{prefix}.raw_samples[{index}].value", positive=True)
            if item["unit"] != "ns":
                raise AggregationError(f"{prefix}:probe_raw_unit_invalid")
            calibration_seen.add((phase, sample_index))
            probe_raw.append(item)
            continue
        if phase != "measurement" or item.get("sample_kind") != "timing_batch":
            raise AggregationError(f"{prefix}:unknown_raw_sample_kind")
        if set(item) != {"phase", "sample_kind", "sample_index", "block_index", "arm", "position", "repetitions", "value", "unit"}:
            raise AggregationError(f"{prefix}:measurement_raw_contract_invalid")
        block = item.get("block_index")
        sample_index = item.get("sample_index")
        position = item.get("position")
        if isinstance(block, bool) or not isinstance(block, int) or block not in range(MAX_BLOCKS):
            raise AggregationError(f"{prefix}:raw_block_invalid")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index != block:
            raise AggregationError(f"{prefix}:raw_sample_index_invalid")
        if position not in {"first", "second"} or (block, position) in seen:
            raise AggregationError(f"{prefix}:raw_order_invalid")
        if item.get("arm") not in {"baseline", "candidate"} or item["arm"] != prefix.rsplit(".", 1)[-1]:
            raise AggregationError(f"{prefix}:raw_arm_invalid")
        if item.get("repetitions") != repetitions or item.get("unit") != "ns":
            raise AggregationError(f"{prefix}:raw_binding_invalid")
        value = _finite(item.get("value"), f"{prefix}.raw_samples[{index}].value", positive=True)
        if not math.isclose(value, result[block][1], rel_tol=0.0, abs_tol=1e-9):
            raise AggregationError(f"{prefix}:raw_value_mismatch")
        if position != positions[block]:
            raise AggregationError(f"{prefix}:raw_position_mismatch")
        seen.add((block, position))
    if len(seen) != MAX_BLOCKS:
        raise AggregationError(f"{prefix}:measurement_raw_samples_missing")
    if any(probe_raw[index]["value"] != probes[index] for index in range(repetitions)):
        raise AggregationError(f"{prefix}:probe_timing_calibration_mismatch")
    calibration_rows = [sample for sample in raw if isinstance(sample, Mapping) and sample.get("phase") != "measurement"]
    declared_raw_warmups = [
        {key: sample[key] for key in ("phase", "sample_index", "value", "unit")}
        for sample in warmup_raw
    ]
    if declared_raw_warmups != list(declared_samples) or [sample["value"] for sample in warmup_raw] != list(durations):
        raise AggregationError(f"{prefix}:warmup_raw_projection_invalid")
    if arm["calibration_samples"] != calibration_rows or len(calibration_seen) != count + len(calibration):
        raise AggregationError(f"{prefix}:calibration_projection_invalid")
    statistics = _require_mapping(arm["statistics"], f"{prefix}.statistics")
    if set(statistics) != {"count", "median_ns", "mad_ns", "iqr_ns", "min_ns", "max_ns"} or statistics.get("count") != MAX_BLOCKS:
        raise AggregationError(f"{prefix}:statistics_contract_invalid")
    for field in ("median_ns", "min_ns", "max_ns"):
        _finite(statistics[field], f"{prefix}.statistics.{field}", positive=True)
    for field in ("mad_ns", "iqr_ns"):
        _finite(statistics[field], f"{prefix}.statistics.{field}", nonnegative=True)
    expected_statistics = {
        "median_ns": _median([pair[0] for pair in result]),
        "mad_ns": _mad([pair[0] for pair in result]),
        "iqr_ns": _iqr([pair[0] for pair in result]),
        "min_ns": min(pair[0] for pair in result),
        "max_ns": max(pair[0] for pair in result),
    }
    if any(statistics[field] != value for field, value in expected_statistics.items()):
        raise AggregationError(f"{prefix}:statistics_not_reconstructed_from_blocks")
    return result, {"warmup_count": count, "repetitions": repetitions, "raw_measurements": len(seen)}


def _validate_supervisor_rss(evidence: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    value = evidence.get("rss_peak_bytes")
    reason = evidence.get("rss_missing_reason")
    if value is not None and reason is not None:
        raise AggregationError(f"{prefix}:rss_value_and_missing_reason_both_set")
    if value is None:
        if not isinstance(reason, str) or not reason or not missing_reason_allowed(reason):
            raise AggregationError(f"{prefix}:rss_missing_reason_required")
        return {"value": None, "missing_reason": reason}
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MEMORY_MAX_INT:
        raise AggregationError(f"{prefix}:rss_value_invalid")
    return {"value": value, "missing_reason": None}


def _validate_adapter_contract(value: Any) -> dict[str, Any]:
    contract = _require_mapping(value, "result.evidence.adapter_contract")
    if set(contract) != {"common_result_ready", "reason", "mapping"}:
        raise AggregationError("session:adapter_contract_keys_invalid")
    if type(contract["common_result_ready"]) is not bool or contract["common_result_ready"] is not False:
        raise AggregationError("session:adapter_contract_common_result_ready_invalid")
    if (
        not isinstance(contract["reason"], str)
        or contract["reason"] != _ADAPTER_CONTRACT["reason"]
        or len(contract["reason"]) > 256
    ):
        raise AggregationError("session:adapter_contract_reason_invalid")
    mapping = _require_mapping(contract["mapping"], "result.evidence.adapter_contract.mapping")
    if set(mapping) != set(_ADAPTER_CONTRACT["mapping"]) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in mapping.items()
    ):
        raise AggregationError("session:adapter_contract_mapping_invalid")
    if dict(mapping) != _ADAPTER_CONTRACT["mapping"]:
        raise AggregationError("session:adapter_contract_mapping_invalid")
    return {
        "common_result_ready": False,
        "reason": contract["reason"],
        "mapping": dict(mapping),
    }


def _validate_correctness(
    domain: Mapping[str, Any], prefix: str, *, manifest: Mapping[str, Any], fixture: Mapping[str, Any],
) -> dict[str, Any]:
    correctness = _require_mapping(domain.get("correctness"), f"{prefix}.correctness")
    if set(correctness) != {"cases", "passed", "performance", "sign_invariant"}:
        raise AggregationError(f"{prefix}:correctness_root_contract_invalid")
    cases = correctness["cases"]
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)) or len(cases) != len(_CORRECTNESS_NAMES):
        raise AggregationError(f"{prefix}:correctness_case_count_invalid")
    names: list[str] = []
    for index, case in enumerate(cases):
        item = _require_mapping(case, f"{prefix}.cases[{index}]")
        name = item.get("name")
        if not isinstance(name, str) or name in names:
            raise AggregationError(f"{prefix}:correctness_case_name_invalid")
        names.append(name)
        expected_keys = SIGN_INVARIANT_KEYS if name == "sign_invariant" else CORRECTNESS_FULL_KEYS
        if set(item) != expected_keys or item.get("passed") is not True:
            raise AggregationError(f"{prefix}:correctness_case_contract_invalid")
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AggregationError(f"{prefix}:correctness_seed_invalid")
        if name != "sign_invariant":
            shape = item["shape"]
            if not isinstance(shape, Sequence) or len(shape) != 4 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
                raise AggregationError(f"{prefix}:correctness_shape_invalid")
            if item["layout"] != "C-contiguous" or not isinstance(item["dtype"], str) or not isinstance(item["zero_rhs"], bool):
                raise AggregationError(f"{prefix}:correctness_metadata_invalid")
            caps = _require_mapping(item["hard_caps"], f"{prefix}.hard_caps")
            if caps != CORRECTNESS_HARD_CAPS:
                raise AggregationError(f"{prefix}:correctness_caps_invalid")
            try:
                if name == "performance_fixture":
                    workload = manifest["workload"]
                    validate_performance_case(
                        item,
                        {
                            **fixture,
                            "a_shape": workload["a_shape"],
                            "b_shape": workload["b_shape"],
                            "dtype": workload["dtype"],
                            "layout": workload["layout"],
                        },
                    )
                else:
                    validate_fixed_case(item)
            except CorrectnessContractError as exc:
                raise AggregationError(f"{prefix}:{exc}") from exc
        else:
            try:
                validate_sign_invariant_case(
                    item,
                    manifest["seeds"]["fixture"],
                    fixture_digest(fixture["a_sha256"], fixture["b_sha256"], manifest["seeds"]["fixture"]),
                )
            except CorrectnessContractError as exc:
                raise AggregationError(f"{prefix}:{exc}") from exc
            continue
        metrics = _require_mapping(item["metrics"], f"{prefix}.metrics")
        if set(metrics) != _CORRECTNESS_METRIC_KEYS:
            raise AggregationError(f"{prefix}:correctness_metrics_contract_invalid")
        for metric_name, metric_value in metrics.items():
            if metric_name == "rel_q99_abs_oracle_ge_1":
                detail = _require_mapping(metric_value, f"{prefix}.{metric_name}")
                if set(detail) != {"value", "missing_reason"}:
                    raise AggregationError(f"{prefix}:correctness_quantile_contract_invalid")
                if detail["value"] is None:
                    if detail["missing_reason"] != "no_oracle_elements_abs_ge_1":
                        raise AggregationError(f"{prefix}:correctness_quantile_missing_reason_invalid")
                else:
                    _finite(detail["value"], f"{prefix}.{metric_name}.value", nonnegative=True)
                    if detail["missing_reason"] is not None:
                        raise AggregationError(f"{prefix}:correctness_quantile_xor_invalid")
            else:
                _finite(metric_value, f"{prefix}.{metric_name}", nonnegative=True)
    if tuple(names) != _CORRECTNESS_NAMES or correctness["passed"] is not True:
        raise AggregationError(f"{prefix}:correctness_cases_incomplete")
    by_name = {case["name"]: case for case in cases}
    for key, expected_name in (("performance", "performance_fixture"), ("sign_invariant", "sign_invariant")):
        linked = _require_mapping(correctness[key], f"{prefix}.{key}")
        if linked.get("name") != expected_name or linked.get("passed") is not True:
            raise AggregationError(f"{prefix}:correctness_link_invalid")
        if canonical_json_bytes(linked) != canonical_json_bytes(by_name[expected_name]):
            raise AggregationError(f"{prefix}:correctness_link_not_exact_case")
        if expected_name == "sign_invariant" and linked.get("seed") != manifest["seeds"]["fixture"]:
            raise AggregationError(f"{prefix}:sign_invariant_seed_not_bound")
    return {"case_names": list(names), "passed": True}


def _validate_session(envelope: Any, expected_set: str) -> dict[str, Any]:
    session = _require_mapping(envelope, "session")
    if set(session) != _SESSION_KEYS:
        raise AggregationError("session:unknown_or_missing_keys")
    manifest = validate_manifest(_require_mapping(session["manifest"], "manifest"))
    if manifest["mode"] != "aa_gpu" or manifest["process"]["set"] != expected_set:
        raise AggregationError("session:wrong_mode_or_process_set")
    index = manifest["process"]["index"]
    if index not in range(MAX_SESSIONS):
        raise AggregationError("session:process_index_invalid")
    seed_table = AA_SESSION_SEEDS
    if manifest["seeds"] != {
        "fixture": seed_table[f"{expected_set}_fixture"] + index,
        "order": seed_table[f"{expected_set}_order"] + index,
        "bootstrap_seed": AA_BOOTSTRAP_SEEDS[expected_set],
    }:
        raise AggregationError("session:registered_seed_mismatch")
    closed = close_manifest(manifest)
    result = validate_result(_require_mapping(session["result"], "result"), manifest=closed)
    if (result["status"], result["classification"], result["action"], result["error"]) != (
        "completed", "measurement_complete", "baseline_fallback", None
    ):
        raise AggregationError("session:common_result_not_measurement_complete")
    evidence = _require_mapping(result["evidence"], "result.evidence")
    if set(evidence) - _RESULT_EVIDENCE_KEYS or not {
        "benchmark_classification", "benchmark_action", "aggregation_required", "adapter_contract", "benchmark_evidence"
    }.issubset(evidence):
        raise AggregationError("session:result_evidence_contract_invalid")
    if (evidence["benchmark_classification"], evidence["benchmark_action"], evidence["aggregation_required"]) != (
        "measurement_complete", "aggregation_required", True
    ):
        raise AggregationError("session:domain_aggregation_required_missing")
    adapter_contract = _validate_adapter_contract(evidence["adapter_contract"])
    domain = _require_mapping(evidence["benchmark_evidence"], "benchmark_evidence")
    if set(domain) - _BENCHMARK_EVIDENCE_KEYS:
        raise AggregationError("session:unknown_benchmark_evidence_key")
    if domain.get("aggregation_required") is not True or domain.get("fresh_process_required") is not True or domain.get("cache_state") != "unknown":
        raise AggregationError("session:domain_gate_not_evaluable")
    for timing_name in ("compile_wrapper_setup_ns", "first_eval_compile_inclusive_ns", "total_elapsed_ns"):
        timing_value = domain.get(timing_name)
        if timing_value is not None:
            _finite(timing_value, f"benchmark_evidence.{timing_name}", positive=True)
    fixture = _require_mapping(domain.get("fixture"), "fixture")
    if set(fixture) != _FIXTURE_KEYS or fixture.get("fixture_seed") != manifest["seeds"]["fixture"]:
        raise AggregationError("session:fixture_contract_invalid")
    for key in _FIXTURE_KEYS - {"fixture_seed"}:
        if not isinstance(fixture[key], str) or not _SHA256_RE.fullmatch(fixture[key]):
            raise AggregationError(f"session:fixture.{key}_invalid")
    correctness = _validate_correctness(domain, "session", manifest=manifest, fixture=fixture)
    memory = _validate_memory(domain, "session")
    memory_limit = _validate_memory_limit(domain.get("memory_limit"), "session")
    supervisor_rss = _validate_supervisor_rss(evidence, "session.evidence")
    arms = _require_mapping(domain.get("arms"), "arms")
    if set(arms) != _ARM_KEYS:
        raise AggregationError("session:arms_contract_invalid")
    baseline, baseline_meta = _validate_arm(arms["baseline"], "session.baseline")
    candidate, candidate_meta = _validate_arm(arms["candidate"], "session.candidate")
    comparison = _require_mapping(domain.get("comparison"), "comparison")
    if set(comparison) != _COMPARISON_KEYS:
        raise AggregationError("session:comparison_root_contract_invalid")
    order = comparison.get("order")
    expected_order = _balanced_order(manifest["seeds"]["order"])
    if order != expected_order:
        raise AggregationError("session:order_evidence_mismatch")
    if comparison.get("aggregation_required") is not True or comparison.get("action") != "aggregation_required":
        raise AggregationError("session:comparison_not_aggregation_observation")
    if comparison.get("benchmark_classification") != "session_observation" or comparison.get("global_decision") is not None:
        raise AggregationError("session:comparison_domain_contract_invalid")
    if comparison.get("comparison_kind") != "aa_gpu_null_control" or comparison.get("aggregation_gate") != "aa_gate":
        raise AggregationError("session:comparison_kind_invalid")
    ratio_statistics = _require_mapping(comparison.get("ratio_statistics"), "comparison.ratio_statistics")
    if set(ratio_statistics) != _RATIO_STAT_KEYS:
        raise AggregationError("session:ratio_statistics_contract_invalid")
    if ratio_statistics["count"] != MAX_BLOCKS:
        raise AggregationError("session:ratio_statistics_count_invalid")
    for name, value in ratio_statistics.items():
        if name != "count":
            _finite(value, f"comparison.ratio_statistics.{name}", positive=name in {"median_ratio", "min_ratio", "max_ratio"}, nonnegative=name in {"mad_ratio", "iqr_ratio"})
    blocks = comparison.get("blocks")
    if not isinstance(blocks, Sequence) or len(blocks) != MAX_BLOCKS:
        raise AggregationError("session:comparison_block_count_invalid")
    pairs: list[tuple[float, float]] = []
    for index, block in enumerate(blocks):
        item = _require_mapping(block, f"comparison.blocks[{index}]")
        if set(item) != _COMPARISON_BLOCK_KEYS or item["block_index"] != index:
            raise AggregationError("session:comparison_block_contract_invalid")
        first, second = order[index], "candidate" if order[index] == "baseline" else "baseline"
        if (item["first"], item["second"]) != (first, second):
            raise AggregationError("session:comparison_order_invalid")
        baseline_ns = _finite(item["baseline_per_eval_ns"], "comparison.baseline", positive=True)
        candidate_ns = _finite(item["candidate_per_eval_ns"], "comparison.candidate", positive=True)
        ratio = _finite(item["ratio"], "comparison.ratio", positive=True)
        if baseline[index][0] != baseline_ns or candidate[index][0] != candidate_ns or ratio != candidate_ns / baseline_ns:
            raise AggregationError("session:comparison_ratio_or_arm_mismatch")
        pairs.append((baseline_ns, candidate_ns))
    ratios = [candidate_ns / baseline_ns for baseline_ns, candidate_ns in pairs]
    expected_ratio_statistics = {
        "count": MAX_BLOCKS,
        "median_ratio": _median(ratios),
        "mad_ratio": _mad(ratios),
        "iqr_ratio": _iqr(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }
    if any(ratio_statistics[field] != value for field, value in expected_ratio_statistics.items()):
        raise AggregationError("session:ratio_statistics_not_reconstructed_from_blocks")
    expected_raw = list(arms["baseline"]["raw_samples"]) + list(arms["candidate"]["raw_samples"])
    if domain.get("raw_samples") != expected_raw or comparison.get("raw_samples") != expected_raw:
        raise AggregationError("session:raw_sample_projection_mismatch")
    return {
        "manifest": manifest,
        "run_id": manifest["run_id"],
        "index": manifest["process"]["index"],
        "provenance": manifest["provenance"],
        "workload": manifest["workload"],
        "fixture": dict(fixture),
        "fixture_sha256": fixture["fixture_sha256"],
        "order_seed": manifest["seeds"]["order"],
        "fixture_seed": manifest["seeds"]["fixture"],
        "bootstrap_seed": manifest["seeds"]["bootstrap_seed"],
        "bootstrap_seed_manifest_bound": True,
        "seed_contract": "aa_gate",
        "pairs": pairs,
        "memory": memory,
        "memory_limit": memory_limit,
        "supervisor_rss": supervisor_rss,
        "correctness": correctness,
        "adapter_contract": dict(adapter_contract),
        "arm_meta": {"baseline": baseline_meta, "candidate": candidate_meta},
    }


def aggregate_h0_aa(
    characterization_sessions: Sequence[Mapping[str, Any]],
    confirmation_sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly three characterization and three confirmation A/A sessions."""

    gates: dict[str, Any] = {}
    failures: list[str] = []
    try:
        if not isinstance(characterization_sessions, Sequence) or isinstance(characterization_sessions, (str, bytes, bytearray)):
            raise AggregationError("characterization:not_sequence")
        if not isinstance(confirmation_sessions, Sequence) or isinstance(confirmation_sessions, (str, bytes, bytearray)):
            raise AggregationError("confirmation:not_sequence")
        if len(characterization_sessions) != MAX_SESSIONS or len(confirmation_sessions) != MAX_SESSIONS:
            raise AggregationError("sets:must_contain_exactly_three_sessions")
        _check_input_bounds({"characterization": characterization_sessions, "confirmation": confirmation_sessions})
        if len(canonical_json_bytes({"characterization": characterization_sessions, "confirmation": confirmation_sessions})) > MAX_INPUT_BYTES:
            raise AggregationError("input:exceeds_size_limit")
        try:
            char = [_validate_session(item, "characterization") for item in characterization_sessions]
        except (AggregationError, ManifestError, ProtocolError, StatisticsError, TypeError, ValueError) as exc:
            gates["characterization"] = {"status": "not_evaluable", "failures": [_bounded_text(str(exc))]}
            raise
        try:
            conf = [_validate_session(item, "confirmation") for item in confirmation_sessions]
        except (AggregationError, ManifestError, ProtocolError, StatisticsError, TypeError, ValueError) as exc:
            gates["confirmation"] = {"status": "not_evaluable", "failures": [_bounded_text(str(exc))]}
            raise
        all_sessions = char + conf
        run_ids = [session["run_id"] for session in all_sessions]
        if len(set(run_ids)) != len(run_ids):
            raise AggregationError("sessions:duplicate_run_id")
        for group_name, group in (("characterization", char), ("confirmation", conf)):
            indices = sorted(session["index"] for session in group)
            if indices != [0, 1, 2]:
                raise AggregationError(f"{group_name}:missing_or_duplicate_process")
        reference = all_sessions[0]
        for session in all_sessions[1:]:
            if canonical_json_bytes(session["provenance"]) != canonical_json_bytes(reference["provenance"]):
                raise AggregationError("sessions:provenance_mismatch")
            if canonical_json_bytes(session["workload"]) != canonical_json_bytes(reference["workload"]):
                raise AggregationError("sessions:workload_mismatch")
        char.sort(key=lambda session: session["index"])
        conf.sort(key=lambda session: session["index"])
        for set_name, sessions in (("characterization", char), ("confirmation", conf)):
            expected_bootstrap_seed = AA_BOOTSTRAP_SEEDS[set_name]
            if any(session["bootstrap_seed"] != expected_bootstrap_seed for session in sessions):
                raise AggregationError(f"{set_name}:manifest_bootstrap_seed_mismatch")
        gate = aa_gate([session["pairs"] for session in char], [session["pairs"] for session in conf])
        for set_name in ("characterization", "confirmation"):
            if gate["sets"][set_name]["bootstrap_seed"] != AA_BOOTSTRAP_SEEDS[set_name]:
                raise AggregationError(f"{set_name}:aa_gate_bootstrap_seed_mismatch")
        gates = {
            "characterization": {"sessions": _session_evidence(char), "gate": _gate_for_set(gate, "characterization")},
            "confirmation": {"sessions": _session_evidence(conf), "gate": _gate_for_set(gate, "confirmation")},
        }
        # aa_gate returns both sets; keep the exact set summaries and make the
        # overall decision explicit without introducing a second statistical rule.
        valid = bool(gate["h0_valid"])
        if not valid:
            failures.extend(gate["failures"])
        result = {
            "schema_version": 1,
            "classification": "h0_valid" if valid else "h0_invalid",
            "h0_valid": valid,
            "engineering_equivalence_gate": valid,
            "scientific_equivalence_claim": False,
            "promotion_gate_applicable": False,
            "seed_contract": "aa_gate",
            "bootstrap_seed_manifest_bound": True,
            "aggregation_contract_ready": True,
            "live_execution_authorized": False,
            "live_ready": False,
            "live_ready_reason": "live execution authorization is not granted by offline aggregation",
            "action": "baseline_reference",
            "next_step": "h0_measurement_system_validated" if valid else "repair_and_remeasure",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "prng": gate["prng"],
            "percentile_method": gate["percentile_method"],
            "gates": {
                "characterization": _gate_for_set(gate, "characterization"),
                "confirmation": _gate_for_set(gate, "confirmation"),
            },
            "session_contracts": {"characterization": _session_evidence(char), "confirmation": _session_evidence(conf)},
            "failures": failures,
        }
        return _bound_result(result)
    except (AggregationError, ManifestError, ProtocolError, StatisticsError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        reason = _bounded_text(str(exc)) or "aggregation:invalid_input"
        return _bound_result(_invalid([reason], gates=gates))


def _session_evidence(sessions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": session["run_id"],
            "process_index": session["index"],
            "fixture_seed": session["fixture_seed"],
            "fixture": session["fixture"],
            "order_seed": session["order_seed"],
            "bootstrap_seed": session["bootstrap_seed"],
            "bootstrap_seed_manifest_bound": session["bootstrap_seed_manifest_bound"],
            "seed_contract": session["seed_contract"],
            "fixture_sha256": session["fixture_sha256"],
            "memory": session["memory"],
            "memory_limit": session["memory_limit"],
            "supervisor_rss": session["supervisor_rss"],
            "correctness": session["correctness"],
            "adapter_contract": session["adapter_contract"],
            "arm_meta": session["arm_meta"],
        }
        for session in sessions
    ]


def _gate_for_set(gate: Mapping[str, Any], name: str) -> dict[str, Any]:
    failures = [failure for failure in gate["failures"] if failure.startswith(f"{name}.")]
    return {
        "set": name,
        "classification": "tie" if not failures else "h0_invalid",
        "h0_valid": not failures,
        "r_aa": gate["sets"][name]["r_aa"],
        "ci_lower": gate["sets"][name]["ci_lower"],
        "ci_upper": gate["sets"][name]["ci_upper"],
        "session_ratios": gate["sets"][name]["session_ratios"],
        "session_log_ratios": gate["sets"][name]["session_log_ratios"],
        "session_log_sd": gate["sets"][name]["session_log_sd"],
        "bootstrap_seed": gate["sets"][name]["bootstrap_seed"],
        "bootstrap_seed_manifest_bound": True,
        "seed_contract": "aa_gate",
        "bootstrap_replicates": gate["sets"][name]["bootstrap_replicates"],
        "prng": gate["sets"][name]["prng"],
        "percentile_method": gate["sets"][name]["percentile_method"],
        "engineering_band": list(ENGINEERING_EQUIVALENCE_BAND),
        "session_band": list(SESSION_RATIO_BAND),
        "failures": failures,
    }


__all__ = ["AggregationError", "aggregate_h0_aa"]
