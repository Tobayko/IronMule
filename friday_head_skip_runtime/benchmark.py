"""CPU policy and guarded MLX qualification for the bounded runtime."""

from __future__ import annotations

import math
import resource
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from experiments.head_skip_formal import study
from friday_evidence.budget import BudgetGuard
from friday_evidence.registry import BudgetPolicy
from tools._bench import require_ac_power

from .constants import (
    GPU_MAX_EXTRA_PEAK_BYTES,
    GPU_MAX_RATIO,
    GPU_MEASUREMENT_ORDERS,
    OUTPUT_TOKENS,
    POLICY_ITERATIONS_PER_ARM,
    POLICY_MAX_INCREMENTAL_NS,
    POLICY_MAX_MEDIAN_NS,
    POLICY_MAX_P95_NS,
    POLICY_MEASUREMENT_BLOCKS,
    POLICY_WARMUP_BLOCKS,
)
from .executor import GenerationOutput, GenerationRequest, RuntimeController
from .mlx_backend import MlxGenerationBackend
from .policy import REGISTERED_SCOPE

_POLICY = BudgetPolicy(
    gpu_work_limit_s=120.0,
    continuous_gpu_limit_s=6.0,
    required_break_s=4.0,
    duty_window_s=60.0,
    duty_cycle_limit=0.15,
    wall_limit_s=1_200.0,
    candidate_cooldown_s=60.0,
)
_PACING_TARGET = 0.14


class BenchmarkError(RuntimeError):
    """A runtime qualification cannot produce admissible engineering evidence."""


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
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure_loop(function: Callable[[], str], iterations: int) -> int:
    started = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    ended = time.perf_counter_ns()
    duration = ended - started
    if duration <= 0:
        raise BenchmarkError("policy timing loop did not execute")
    return duration


def benchmark_policy_overhead(
    controller: RuntimeController,
    *,
    warmup_blocks: int = POLICY_WARMUP_BLOCKS,
    measurement_blocks: int = POLICY_MEASUREMENT_BLOCKS,
    iterations: int = POLICY_ITERATIONS_PER_ARM,
) -> dict[str, Any]:
    """Compare cached observe/select with direct immutable-plan lookup."""

    for name, value in (
        ("warmup_blocks", warmup_blocks),
        ("measurement_blocks", measurement_blocks),
        ("iterations", iterations),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BenchmarkError(f"{name} must be a positive integer")

    def direct() -> str:
        return "lm_head_last_position_of_final_prefill_block_only"

    def policy() -> str:
        return controller.decide_scope(REGISTERED_SCOPE).plan

    if not controller.evidence.authorized or policy() != direct():
        raise BenchmarkError("cached policy does not authorize the registered scope")
    for index in range(warmup_blocks):
        order = "ab" if index % 2 == 0 else "ba"
        for arm in order:
            _measure_loop(direct if arm == "a" else policy, iterations)
    blocks: list[dict[str, int | str]] = []
    for index in range(measurement_blocks):
        order = "ab" if index % 2 == 0 else "ba"
        values: dict[str, int] = {}
        for arm in order:
            values[arm] = _measure_loop(direct if arm == "a" else policy, iterations)
        blocks.append(
            {
                "block_index": index,
                "order": order,
                "direct_ns": values["a"],
                "policy_ns": values["b"],
            }
        )
    direct_each = [float(block["direct_ns"]) / iterations for block in blocks]
    policy_each = [float(block["policy_ns"]) / iterations for block in blocks]
    incremental = [candidate - baseline for baseline, candidate in zip(direct_each, policy_each)]
    policy_median = _median(policy_each)
    policy_p95 = _percentile(policy_each, 0.95)
    incremental_median = _median(incremental)
    gate = (
        policy_median <= POLICY_MAX_MEDIAN_NS
        and policy_p95 <= POLICY_MAX_P95_NS
        and incremental_median <= POLICY_MAX_INCREMENTAL_NS
        and controller.circuit_reason is None
        and policy() == direct()
    )
    return {
        "policy": asdict(controller.evidence),
        "thresholds": {
            "policy_max_median_ns": POLICY_MAX_MEDIAN_NS,
            "policy_max_p95_ns": POLICY_MAX_P95_NS,
            "policy_max_incremental_ns": POLICY_MAX_INCREMENTAL_NS,
        },
        "metrics": {
            "direct_median_ns": _median(direct_each),
            "policy_median_ns": policy_median,
            "policy_mad_ns": _mad(policy_each),
            "policy_p95_ns": policy_p95,
            "incremental_median_ns": incremental_median,
            "gate_passed": gate,
        },
        "blocks": blocks,
    }


def _pace(guard: BudgetGuard, seconds: float) -> None:
    if not math.isfinite(seconds) or seconds <= 0:
        raise BenchmarkError("GPU duration is invalid")
    guard.record_gpu(seconds)
    required = seconds * (1.0 - _PACING_TARGET) / _PACING_TARGET
    for _ in range(max(4, math.ceil(required / _POLICY.required_break_s))):
        guard.required_break()


def _rss_peak_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _swap_used() -> int | None:
    try:
        import psutil

        value = psutil.swap_memory().used
    except Exception:
        return None
    return value if type(value) is int and value >= 0 else None


def _run_arm(
    *,
    arm: str,
    controller: RuntimeController,
    backend: MlxGenerationBackend,
    request: GenerationRequest,
    token_ids: Sequence[int],
    guard: BudgetGuard,
) -> tuple[GenerationOutput, str]:
    backend.reset_peak_memory()
    if arm == "a":
        output = backend.generate_baseline(token_ids, request)
        plan = "full_lm_head_all_prefill_positions"
    elif arm == "b":
        result = controller.execute(backend, request)
        output = result.output
        plan = result.decision.plan
    else:
        raise BenchmarkError("unknown measurement arm")
    # The backend stops both durations before returning. Guard pacing starts only here.
    _pace(guard, output.total_ns / 1e9)
    return output, plan


def _output_record(output: GenerationOutput) -> dict[str, Any]:
    return {
        "token_ids": list(output.token_ids),
        "token_sha256": output.token_sha256,
        "prefill_ns": output.prefill_ns,
        "total_ns": output.total_ns,
        "prefill_blocks": output.prefill_blocks,
        "head_calls": output.head_calls,
        "memory": dict(output.memory),
    }


def run_mlx_validation(controller: RuntimeController) -> dict[str, Any]:
    """Run the single preregistered engineering qualification on local MLX."""

    power_source = require_ac_power()
    guard = BudgetGuard(_POLICY)
    guard.before_candidate()
    process_started = time.process_time_ns()
    swap_before = _swap_used()
    backend = MlxGenerationBackend.load_local()
    request = GenerationRequest(prompt_content=study.PROMPT_CONTENT, output_tokens=OUTPUT_TOKENS)
    token_ids = backend.encode_prompt(request.prompt_content)
    decision = controller.decide_scope(REGISTERED_SCOPE)
    if decision.strategy != "head_skip":
        raise BenchmarkError(f"runtime policy is unavailable: {decision.reason}")

    correctness_a, _ = _run_arm(
        arm="a",
        controller=controller,
        backend=backend,
        request=request,
        token_ids=token_ids,
        guard=guard,
    )
    correctness_b, correctness_plan = _run_arm(
        arm="b",
        controller=controller,
        backend=backend,
        request=request,
        token_ids=token_ids,
        guard=guard,
    )
    expected_blocks = math.ceil(len(token_ids) / backend.prefill_chunk)
    token_identical = correctness_a.token_ids == correctness_b.token_ids
    path_exact = (
        correctness_plan == "lm_head_last_position_of_final_prefill_block_only"
        and correctness_b.prefill_blocks == expected_blocks
        and correctness_b.head_calls == 1
        and correctness_a.prefill_blocks == expected_blocks
        and correctness_a.head_calls == expected_blocks
    )
    if not token_identical:
        raise BenchmarkError("correctness_failed_terminal")
    if not path_exact:
        raise BenchmarkError("registered candidate path was not exercised exactly")

    warmup: list[dict[str, Any]] = []
    warm_values: dict[str, GenerationOutput] = {}
    for arm in "ba":
        warm_values[arm], _ = _run_arm(
            arm=arm,
            controller=controller,
            backend=backend,
            request=request,
            token_ids=token_ids,
            guard=guard,
        )
    warmup.append(
        {
            "order": "ba",
            "a_prefill_ns": warm_values["a"].prefill_ns,
            "b_prefill_ns": warm_values["b"].prefill_ns,
        }
    )

    blocks: list[dict[str, Any]] = []
    for index, order in enumerate(GPU_MEASUREMENT_ORDERS):
        values: dict[str, GenerationOutput] = {}
        plans: dict[str, str] = {}
        for arm in order:
            values[arm], plans[arm] = _run_arm(
                arm=arm,
                controller=controller,
                backend=backend,
                request=request,
                token_ids=token_ids,
                guard=guard,
            )
        if values["a"].token_ids != values["b"].token_ids:
            raise BenchmarkError("correctness_failed_terminal")
        blocks.append(
            {
                "block_index": index,
                "order": order,
                "a": _output_record(values["a"]),
                "b": _output_record(values["b"]),
                "candidate_plan": plans["b"],
            }
        )
    guard.finish_candidate()
    a_values = [int(block["a"]["prefill_ns"]) for block in blocks]
    b_values = [int(block["b"]["prefill_ns"]) for block in blocks]
    ratios = [b / a for a, b in zip(a_values, b_values)]
    ratio = _median(ratios)
    a_peaks = [block["a"]["memory"].get("mlx_peak_memory_bytes") for block in blocks]
    b_peaks = [block["b"]["memory"].get("mlx_peak_memory_bytes") for block in blocks]
    valid_a_peaks = [value for value in a_peaks if type(value) is int]
    valid_b_peaks = [value for value in b_peaks if type(value) is int]
    peak_delta = (
        max(valid_b_peaks) - max(valid_a_peaks)
        if valid_a_peaks and valid_b_peaks
        else None
    )
    swap_after = _swap_used()
    swap_delta = (
        swap_after - swap_before
        if swap_before is not None and swap_after is not None
        else None
    )
    resource_gate = (
        peak_delta is not None
        and peak_delta <= GPU_MAX_EXTRA_PEAK_BYTES
        and swap_delta is not None
        and swap_delta <= 0
    )
    gate = (
        token_identical
        and path_exact
        and ratio <= GPU_MAX_RATIO
        and resource_gate
        and controller.circuit_reason is None
        and all(block["candidate_plan"] == decision.plan for block in blocks)
    )
    return {
        "policy": asdict(controller.evidence),
        "workload": {
            "prompt_tokens": len(token_ids),
            "output_tokens": OUTPUT_TOKENS,
            "prefill_chunk": backend.prefill_chunk,
            "power_source": power_source,
        },
        "correctness": {
            "token_identical": token_identical,
            "token_sha256": correctness_a.token_sha256,
            "candidate_path_exercised": path_exact,
        },
        "thresholds": {
            "max_ratio": GPU_MAX_RATIO,
            "max_extra_peak_bytes": GPU_MAX_EXTRA_PEAK_BYTES,
            "duty_cycle": 0.15,
        },
        "metrics": {
            "baseline_median_ns": _median(a_values),
            "candidate_median_ns": _median(b_values),
            "baseline_mad_ns": _mad(a_values),
            "candidate_mad_ns": _mad(b_values),
            "ratio": ratio,
            "effect_percent": 100.0 * (ratio - 1.0),
            "peak_memory_delta_bytes": peak_delta,
            "swap_delta_bytes": swap_delta,
            "byte_identical": token_identical,
            "gate_passed": gate,
        },
        "warmup": warmup,
        "blocks": blocks,
        "resources": {
            "cpu_process_ns": time.process_time_ns() - process_started,
            "rss_peak_bytes": _rss_peak_bytes(),
            "guard": guard.summary(),
        },
    }


__all__ = [
    "BenchmarkError",
    "benchmark_policy_overhead",
    "run_mlx_validation",
]
