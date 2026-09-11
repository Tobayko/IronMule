"""CPU policy-overhead and bounded MLX runtime validation measurements."""

from __future__ import annotations

import hashlib
import math
import resource
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from friday_evidence.budget import BudgetGuard
from friday_h1.constants import N_MATMULS

from .constants import (
    BATCHED_PLAN,
    GPU_MAX_RATIO,
    GPU_MEASUREMENT_BLOCKS,
    GPU_WARMUP_PAIRS,
    POLICY_ITERATIONS_PER_ARM,
    POLICY_MAX_INCREMENTAL_NS,
    POLICY_MAX_MEDIAN_NS,
    POLICY_MAX_P95_NS,
    POLICY_MEASUREMENT_BLOCKS,
    POLICY_WARMUP_BLOCKS,
    SHAPE,
)
from .executor import RuntimeController, execute_serial


class BenchmarkError(RuntimeError):
    """A bounded engineering validation cannot produce admissible evidence."""


class ValidationBackend(Protocol):
    def matmul(self, left: Any, right: Any) -> Any: ...

    def eval_many(self, values: Sequence[Any]) -> None: ...

    def synchronize(self) -> None: ...

    def to_host(self, value: Any) -> Any: ...

    def memory_snapshot(self) -> Mapping[str, int | None]: ...


@dataclass(frozen=True)
class _MetadataTensor:
    shape: tuple[int, int] = SHAPE
    dtype: str = "float16"


def _median(values: Sequence[int | float]) -> float:
    if not values:
        raise BenchmarkError("statistics require at least one value")
    result = float(statistics.median(values))
    if not math.isfinite(result):
        raise BenchmarkError("measurement is not finite")
    return result


def _mad(values: Sequence[int | float]) -> float:
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _percentile(values: Sequence[int | float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise BenchmarkError("percentile request is invalid")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure_loop(function: Callable[[], str], iterations: int, clock_ns: Callable[[], int]) -> int:
    started = clock_ns()
    sink = ""
    for _ in range(iterations):
        sink = function()
    elapsed = clock_ns() - started
    if sink not in {"serial", "batched"} or elapsed <= 0:
        raise BenchmarkError("policy timing loop did not execute")
    return elapsed


def benchmark_policy_overhead(
    controller: RuntimeController,
    *,
    warmup_blocks: int = POLICY_WARMUP_BLOCKS,
    measurement_blocks: int = POLICY_MEASUREMENT_BLOCKS,
    iterations: int = POLICY_ITERATIONS_PER_ARM,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Measure cached observe+select against a direct immutable plan lookup."""

    for name, value in (
        ("warmup_blocks", warmup_blocks),
        ("measurement_blocks", measurement_blocks),
        ("iterations", iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BenchmarkError(f"{name} must be a positive integer")
    left = _MetadataTensor()
    operands = tuple(_MetadataTensor() for _ in range(N_MATMULS))
    initial = controller.decide(left, operands)
    expected = initial.strategy

    def direct() -> str:
        return expected

    def policy() -> str:
        return controller.decide(left, operands).strategy

    for index in range(warmup_blocks):
        order = (direct, policy) if index % 2 == 0 else (policy, direct)
        for arm in order:
            _measure_loop(arm, iterations, clock_ns)

    blocks: list[dict[str, int | str]] = []
    for index in range(measurement_blocks):
        values: dict[str, int] = {}
        order = ("baseline", "policy") if index % 2 == 0 else ("policy", "baseline")
        for name in order:
            function = direct if name == "baseline" else policy
            values[name] = _measure_loop(function, iterations, clock_ns)
        blocks.append(
            {
                "block_index": index,
                "order": "ab" if index % 2 == 0 else "ba",
                "baseline_total_ns": values["baseline"],
                "policy_total_ns": values["policy"],
                "baseline_ns_per_call": values["baseline"] // iterations,
                "policy_ns_per_call": values["policy"] // iterations,
            }
        )
    baseline = [int(block["baseline_ns_per_call"]) for block in blocks]
    policy_values = [int(block["policy_ns_per_call"]) for block in blocks]
    increments = [
        candidate - control
        for candidate, control in zip(policy_values, baseline, strict=True)
    ]
    baseline_median = _median(baseline)
    policy_median = _median(policy_values)
    policy_p95 = _percentile(policy_values, 0.95)
    incremental = _median(increments)
    gate = (
        initial.strategy == "batched"
        and policy_median <= POLICY_MAX_MEDIAN_NS
        and policy_p95 <= POLICY_MAX_P95_NS
        and incremental <= POLICY_MAX_INCREMENTAL_NS
        and controller.circuit_reason is None
    )
    return {
        "design": {
            "warmup_blocks": warmup_blocks,
            "measurement_blocks": measurement_blocks,
            "iterations_per_arm": iterations,
            "balanced_order": True,
            "cached_evidence": True,
        },
        "policy": {
            "authorized": controller.evidence.authorized,
            "reason": initial.reason,
            "strategy": initial.strategy,
            "decision_record_id": controller.evidence.decision_record_id,
        },
        "metrics": {
            "baseline_median_ns": baseline_median,
            "baseline_mad_ns": _mad(baseline),
            "policy_median_ns": policy_median,
            "policy_mad_ns": _mad(policy_values),
            "policy_p95_ns": policy_p95,
            "incremental_median_ns": incremental,
            "gate_passed": gate,
        },
        "thresholds": {
            "policy_max_median_ns": POLICY_MAX_MEDIAN_NS,
            "policy_max_p95_ns": POLICY_MAX_P95_NS,
            "incremental_max_median_ns": POLICY_MAX_INCREMENTAL_NS,
        },
        "blocks": blocks,
    }


def _rss_peak_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _digest_outputs(
    values: Sequence[Any], backend: ValidationBackend, np_module: Any
) -> tuple[str, list[Any]]:
    digest = hashlib.sha256()
    host: list[Any] = []
    for value in values:
        array = np_module.asarray(backend.to_host(value))
        contiguous = np_module.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii", errors="strict"))
        digest.update(str(tuple(int(item) for item in contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
        host.append(contiguous)
    return digest.hexdigest(), host


def _time_plan(
    function: Callable[[], tuple[Any, ...]],
    *,
    guard: BudgetGuard,
    clock_ns: Callable[[], int],
) -> tuple[tuple[Any, ...], int]:
    started = clock_ns()
    outputs = function()
    elapsed = clock_ns() - started
    if elapsed <= 0:
        raise BenchmarkError("runtime plan duration must be positive")
    guard.record_gpu(elapsed / 1_000_000_000.0)
    return outputs, elapsed


def validate_prepared_runtime(
    controller: RuntimeController,
    *,
    backend: ValidationBackend,
    left: Any,
    operands: Sequence[Any],
    np_module: Any,
    power_source: str,
    guard: BudgetGuard | None = None,
    candidate_started: bool = False,
    warmup_pairs: int = GPU_WARMUP_PAIRS,
    measurement_blocks: int = GPU_MEASUREMENT_BLOCKS,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    process_clock_ns: Callable[[], int] = time.process_time_ns,
) -> dict[str, Any]:
    if len(operands) != N_MATMULS:
        raise BenchmarkError("prepared runtime workload has the wrong operand count")
    if (
        isinstance(warmup_pairs, bool)
        or not isinstance(warmup_pairs, int)
        or warmup_pairs <= 0
        or isinstance(measurement_blocks, bool)
        or not isinstance(measurement_blocks, int)
        or measurement_blocks <= 0
    ):
        raise BenchmarkError("runtime validation repetition counts must be positive")
    decision = controller.decide(left, operands)
    if decision.strategy != "batched" or decision.plan != BATCHED_PLAN:
        raise BenchmarkError(f"runtime policy did not authorize batching: {decision.reason}")
    active_guard = guard or BudgetGuard()
    if not candidate_started:
        active_guard.before_candidate()
    cpu_started = process_clock_ns()

    def baseline() -> tuple[Any, ...]:
        return execute_serial(backend, left, operands)

    def candidate() -> tuple[Any, ...]:
        result = controller.execute(backend, left, operands)
        if result.decision.plan != BATCHED_PLAN:
            raise BenchmarkError("runtime policy changed during validation")
        return result.outputs

    try:
        reference, _ = _time_plan(baseline, guard=active_guard, clock_ns=clock_ns)
        optimized, _ = _time_plan(candidate, guard=active_guard, clock_ns=clock_ns)
        reference_digest, reference_host = _digest_outputs(reference, backend, np_module)
        optimized_digest, optimized_host = _digest_outputs(optimized, backend, np_module)
        max_abs_error = max(
            float(
                np_module.max(
                    np_module.abs(
                        candidate_value.astype(np_module.float32)
                        - reference_value.astype(np_module.float32)
                    )
                )
            )
            for candidate_value, reference_value in zip(
                optimized_host, reference_host, strict=True
            )
        )
        byte_identical = reference_digest == optimized_digest and max_abs_error == 0.0
        if not byte_identical:
            raise BenchmarkError("runtime candidate is not byte-identical to serial")
        del reference, optimized, reference_host, optimized_host

        warmups: list[dict[str, int | str]] = []
        for index in range(warmup_pairs):
            order = "ab" if index % 2 == 0 else "ba"
            functions = (("a", baseline), ("b", candidate))
            if order == "ba":
                functions = tuple(reversed(functions))
            values: dict[str, int] = {}
            for name, function in functions:
                _, values[name] = _time_plan(function, guard=active_guard, clock_ns=clock_ns)
            warmups.append(
                {
                    "block_index": index,
                    "order": order,
                    "a_ns": values["a"],
                    "b_ns": values["b"],
                }
            )

        blocks: list[dict[str, int | str]] = []
        for index in range(measurement_blocks):
            order = "ab" if index % 2 == 0 else "ba"
            functions = (("a", baseline), ("b", candidate))
            if order == "ba":
                functions = tuple(reversed(functions))
            values = {}
            for name, function in functions:
                _, values[name] = _time_plan(function, guard=active_guard, clock_ns=clock_ns)
            blocks.append(
                {
                    "block_index": index,
                    "order": order,
                    "a_ns": values["a"],
                    "b_ns": values["b"],
                }
            )
        active_guard.finish_candidate()
    except Exception:
        # Preserve the original exception; the guard is accounting, not recovery.
        raise

    baseline_values = [int(block["a_ns"]) for block in blocks]
    candidate_values = [int(block["b_ns"]) for block in blocks]
    ratios = [
        candidate_ns / baseline_ns
        for candidate_ns, baseline_ns in zip(
            candidate_values, baseline_values, strict=True
        )
    ]
    ratio = _median(ratios)
    gate = ratio <= GPU_MAX_RATIO and byte_identical and controller.circuit_reason is None
    budget = active_guard.summary()
    return {
        "design": {
            "workload": "matmul-fp16-2048x2048-n8",
            "warmup_pairs": warmup_pairs,
            "measurement_blocks": measurement_blocks,
            "balanced_order": True,
            "baseline_plan": "serial_per_op_eval_and_sync",
            "candidate_plan": BATCHED_PLAN,
            "engineering_validation_only": True,
        },
        "policy": {
            "authorized": controller.evidence.authorized,
            "reason": decision.reason,
            "strategy": decision.strategy,
            "decision_record_id": controller.evidence.decision_record_id,
        },
        "correctness": {
            "byte_identical": byte_identical,
            "max_abs_error": max_abs_error,
            "reference_sha256": reference_digest,
            "candidate_sha256": optimized_digest,
        },
        "metrics": {
            "baseline_median_ns": _median(baseline_values),
            "baseline_mad_ns": _mad(baseline_values),
            "candidate_median_ns": _median(candidate_values),
            "candidate_mad_ns": _mad(candidate_values),
            "ratio": ratio,
            "effect_percent": (ratio - 1.0) * 100.0,
            "max_abs_error": max_abs_error,
            "byte_identical": byte_identical,
            "gate_passed": gate,
        },
        "thresholds": {"maximum_ratio": GPU_MAX_RATIO, "maximum_abs_error": 0.0},
        "warmups": warmups,
        "blocks": blocks,
        "resources": {
            "cpu_process_ns": process_clock_ns() - cpu_started,
            "rss_peak_bytes": _rss_peak_bytes(),
            **dict(backend.memory_snapshot()),
            "budget": budget,
        },
        "power_source": power_source,
    }


def run_mlx_validation(controller: RuntimeController) -> dict[str, Any]:
    """Load the sealed fixture only after evidence and AC-power release gates."""

    if not controller.evidence.authorized:
        raise BenchmarkError(
            "H1 evidence did not authorize runtime validation: "
            f"{controller.evidence.reason}"
        )
    from friday_h1.runner import MlxBackend, _load_real_workload, require_ac_power

    power_source = require_ac_power()
    guard = BudgetGuard()
    guard.before_candidate()
    backend = MlxBackend()
    left, operands, np_module = _load_real_workload(backend, guard=guard)
    return validate_prepared_runtime(
        controller,
        backend=backend,
        left=left,
        operands=operands,
        np_module=np_module,
        power_source=power_source,
        guard=guard,
        candidate_started=True,
    )


__all__ = [
    "BenchmarkError",
    "benchmark_policy_overhead",
    "run_mlx_validation",
    "validate_prepared_runtime",
]
