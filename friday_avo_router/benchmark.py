"""Paired CPU overhead and real-tensor shadow validation."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from .constants import (
    COLD_LOAD_MAX_NS,
    ENFORCED_PLAN,
    POLICY_BLOCKS,
    POLICY_INCREMENTAL_MEDIAN_MAX_NS,
    POLICY_ITERATIONS,
    POLICY_MEDIAN_MAX_NS,
    POLICY_P95_MAX_NS,
    POLICY_WARMUP_BLOCKS,
    SERIAL_PLAN,
)
from .router import ShadowDecision, ShadowRouter


Clock = Callable[[], int]


@dataclass(frozen=True)
class _TensorMeta:
    shape: tuple[int, int]
    dtype: str = "float16"


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _mad(values: list[float]) -> float:
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _recommendation_key(decision: ShadowDecision) -> tuple[str, str, str]:
    return decision.route, decision.recommendation_strategy, decision.recommendation_plan


def _direct_key(router: ShadowRouter, left: Any, operands: tuple[Any, ...]) -> tuple[str, str, str]:
    route, decision = router.direct_decision(left, operands)
    if decision is None:
        return "serial", "serial", SERIAL_PLAN
    return route, decision.strategy, decision.plan


def _run_arm(
    operation: Callable[[int], tuple[str, str, str]],
    iterations: int,
    clock_ns: Clock,
) -> tuple[float, int]:
    checksum = 0
    started = clock_ns()
    for index in range(iterations):
        route, strategy, plan = operation(index)
        checksum += len(route) + len(strategy) + len(plan)
    elapsed = clock_ns() - started
    if elapsed < 0:
        raise RuntimeError("benchmark clock moved backwards")
    return elapsed / iterations, checksum


def benchmark_policy_overhead(
    router: ShadowRouter,
    *,
    cold_load_ns: int,
    warmup_blocks: int = POLICY_WARMUP_BLOCKS,
    blocks: int = POLICY_BLOCKS,
    iterations: int = POLICY_ITERATIONS,
    clock_ns: Clock = time.perf_counter_ns,
) -> dict[str, object]:
    if not router.ready:
        raise RuntimeError("both policies must authorize before benchmarking")
    if min(warmup_blocks, blocks, iterations) <= 0:
        raise ValueError("benchmark dimensions must be positive")

    left = _TensorMeta((2048, 2048))
    rhs = _TensorMeta((2048, 2048))
    n8 = tuple(rhs for _ in range(8))
    n10 = tuple(rhs for _ in range(10))

    def direct(index: int) -> tuple[str, str, str]:
        operands = n8 if index % 2 == 0 else n10
        return _direct_key(router, left, operands)

    def candidate(index: int) -> tuple[str, str, str]:
        operands = n8 if index % 2 == 0 else n10
        return _recommendation_key(router.decide(left, operands))

    agreement = all(direct(index) == candidate(index) for index in range(2))
    if not agreement:
        raise RuntimeError("router recommendation differs before timing")

    for block in range(warmup_blocks):
        order = (direct, candidate) if block % 2 == 0 else (candidate, direct)
        for operation in order:
            _run_arm(operation, iterations, clock_ns)

    baseline: list[float] = []
    routed: list[float] = []
    checksum_equal = True
    for block in range(blocks):
        if block % 2 == 0:
            direct_value, direct_sum = _run_arm(direct, iterations, clock_ns)
            routed_value, routed_sum = _run_arm(candidate, iterations, clock_ns)
        else:
            routed_value, routed_sum = _run_arm(candidate, iterations, clock_ns)
            direct_value, direct_sum = _run_arm(direct, iterations, clock_ns)
        baseline.append(direct_value)
        routed.append(routed_value)
        checksum_equal = checksum_equal and direct_sum == routed_sum

    increments = [candidate_ns - baseline_ns for baseline_ns, candidate_ns in zip(baseline, routed)]
    baseline_median = float(statistics.median(baseline))
    router_median = float(statistics.median(routed))
    router_p95 = _percentile(routed, 0.95)
    incremental_median = float(statistics.median(increments))
    gates = {
        "both_evidence_authorized": router.ready,
        "cold_load": cold_load_ns <= COLD_LOAD_MAX_NS,
        "decision_agreement": agreement and checksum_equal,
        "router_median": router_median <= POLICY_MEDIAN_MAX_NS,
        "router_p95": router_p95 <= POLICY_P95_MAX_NS,
        "incremental_median": incremental_median <= POLICY_INCREMENTAL_MEDIAN_MAX_NS,
    }
    return {
        "cold_load_ns": cold_load_ns,
        "warmup_blocks": warmup_blocks,
        "blocks": blocks,
        "iterations_per_arm": iterations,
        "baseline_median_ns": baseline_median,
        "baseline_mad_ns": _mad(baseline),
        "router_median_ns": router_median,
        "router_mad_ns": _mad(routed),
        "router_p95_ns": router_p95,
        "incremental_median_ns": incremental_median,
        "baseline_block_ns": baseline,
        "router_block_ns": routed,
        "gates": gates,
        "gate_passed": all(gates.values()),
    }


def validate_real_tensor_shadow(router: ShadowRouter) -> dict[str, object]:
    if not router.ready:
        raise RuntimeError("both policies must authorize before shadow validation")
    import mlx.core as mx

    mx.reset_peak_memory()
    left = mx.zeros((2048, 2048), dtype=mx.float16)
    rhs = mx.ones((2048, 2048), dtype=mx.float16)
    mx.eval(left, rhs)
    mx.synchronize()

    cases = {
        "n8_exact": (left, tuple(rhs for _ in range(8)), "n8", "batched"),
        "n10_exact": (left, tuple(rhs for _ in range(10)), "n10", "batched"),
        "count_9": (left, tuple(rhs for _ in range(9)), "serial", "serial"),
        "wrong_shape": (left[:1024, :], tuple(rhs for _ in range(8)), "serial", "serial"),
    }
    wrong_dtype = mx.zeros((2048, 2048), dtype=mx.float32)
    mx.eval(wrong_dtype)
    cases["wrong_dtype"] = (
        wrong_dtype,
        tuple(wrong_dtype for _ in range(10)),
        "serial",
        "serial",
    )

    results: dict[str, object] = {}
    agreement = True
    enforced = True
    expected = True
    for name, (case_left, operands, expected_route, expected_strategy) in cases.items():
        shadow = router.decide(case_left, operands)
        direct = _direct_key(router, case_left, operands)
        shadow_key = _recommendation_key(shadow)
        agreement = agreement and shadow_key == direct
        enforced = enforced and shadow.enforced_plan == ENFORCED_PLAN
        expected = expected and shadow.route == expected_route and shadow.recommendation_strategy == expected_strategy
        results[name] = shadow.as_dict()

    gates = {
        "both_evidence_authorized": router.ready,
        "direct_policy_agreement": agreement,
        "serial_shadow_only": enforced,
        "registered_and_negative_cases": expected,
        "no_matmul_executed": True,
    }
    return {
        "cases": results,
        "mlx_active_memory_bytes": int(mx.get_active_memory()),
        "mlx_cache_memory_bytes": int(mx.get_cache_memory()),
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "gates": gates,
        "gate_passed": all(gates.values()),
    }
