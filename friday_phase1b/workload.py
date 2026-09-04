"""Frozen fixtures, MLX baselines, correctness checks, and block timing."""

from __future__ import annotations

import gc
import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from .canonical import canonical_sha256
from .constants import (
    ABS_MAX_CAP,
    BASELINE_NAMES,
    NORMALIZED_L2_CAP,
    PAIR_ATOL,
    PAIR_RTOL,
    RELATIVE_ORACLE_FLOOR,
    REL_Q99_CAP,
)
from .kernel_source import EPSILON, HIDDEN_SIZE, ROWS


Arm = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True)
class Fixture:
    name: str
    seed: int | None
    x: Any
    residual: Any
    weight: Any
    oracle: Any
    digest: str


def _array_sha256(array: Any) -> str:
    little = array.astype("<f2", copy=False)
    return hashlib.sha256(little.tobytes(order="C")).hexdigest()


def make_fixture(np: Any, name: str, seed: int | None) -> Fixture:
    shape = (ROWS, HIDDEN_SIZE)
    if name == "zeros":
        if seed is not None:
            raise ValueError("zeros fixture has no seed")
        x = np.zeros(shape, dtype="<f2")
        residual = np.zeros(shape, dtype="<f2")
        weight = np.ones((HIDDEN_SIZE,), dtype="<f2")
    elif name == "constant":
        if seed is not None:
            raise ValueError("constant fixture has no seed")
        x = np.full(shape, 0.5, dtype="<f2")
        residual = np.full(shape, 0.25, dtype="<f2")
        weight = np.ones((HIDDEN_SIZE,), dtype="<f2")
    else:
        if seed is None:
            raise ValueError("random fixture requires a seed")
        generator = np.random.Generator(np.random.PCG64(seed))
        if name == "cancellation":
            x = generator.normal(0.0, 1.0, size=shape).astype("<f2")
            residual = (-x).astype("<f2")
            weight = generator.uniform(0.75, 1.25, size=HIDDEN_SIZE).astype("<f2")
        elif name in {"visible_normal", "performance"}:
            sigma = 0.5 if name == "visible_normal" else 0.75
            x = generator.normal(0.0, sigma, size=shape).astype("<f2")
            residual = generator.normal(0.0, sigma, size=shape).astype("<f2")
            weight = generator.uniform(0.75, 1.25, size=HIDDEN_SIZE).astype("<f2")
        elif name == "visible_bounded":
            x = generator.uniform(-4.0, 4.0, size=shape).astype("<f2")
            residual = generator.uniform(-4.0, 4.0, size=shape).astype("<f2")
            weight = generator.uniform(0.5, 1.5, size=HIDDEN_SIZE).astype("<f2")
        elif name == "holdout_normal":
            x = generator.normal(0.0, 1.25, size=shape).astype("<f2")
            residual = generator.normal(0.0, 1.25, size=shape).astype("<f2")
            weight = generator.uniform(0.5, 1.5, size=HIDDEN_SIZE).astype("<f2")
        else:
            raise ValueError("fixture name is not registered")
    z = (x.astype(np.float32) + residual.astype(np.float32)).astype("<f2")
    z64 = z.astype(np.float64)
    mean_square = np.sum(z64 * z64, axis=-1, keepdims=True, dtype=np.float64) / HIDDEN_SIZE
    oracle = (
        z64
        * np.reciprocal(np.sqrt(mean_square + EPSILON))
        * weight.astype(np.float64)[None, :]
    ).astype("<f2")
    components = {
        "name": name,
        "seed": seed,
        "shape": [ROWS, HIDDEN_SIZE],
        "dtype": "float16",
        "x_sha256": _array_sha256(x),
        "residual_sha256": _array_sha256(residual),
        "weight_sha256": _array_sha256(weight),
        "oracle_sha256": _array_sha256(oracle),
    }
    return Fixture(name, seed, x, residual, weight, oracle, canonical_sha256(components))


def to_mlx(mx: Any, fixture: Fixture) -> tuple[Any, Any, Any]:
    if not all(
        bool(value.flags.c_contiguous)
        for value in (fixture.x, fixture.residual, fixture.weight, fixture.oracle)
    ):
        raise ValueError("registered host fixture is not C-contiguous")
    x = mx.array(fixture.x, dtype=mx.float16)
    residual = mx.array(fixture.residual, dtype=mx.float16)
    weight = mx.array(fixture.weight, dtype=mx.float16)
    mx.eval(x, residual, weight)
    mx.synchronize()
    return x, residual, weight


def _transparent(mx: Any) -> Arm:
    def call(x: Any, residual: Any, weight: Any) -> Any:
        z = (x.astype(mx.float32) + residual.astype(mx.float32)).astype(mx.float16)
        z32 = z.astype(mx.float32)
        mean_square = mx.mean(z32 * z32, axis=-1, keepdims=True)
        return (z32 * mx.rsqrt(mean_square + EPSILON) * weight.astype(mx.float32)).astype(
            mx.float16
        )

    return call


def _fast(mx: Any) -> Arm:
    def call(x: Any, residual: Any, weight: Any) -> Any:
        return mx.fast.rms_norm(x + residual, weight, EPSILON)

    return call


def make_baseline(mx: Any, name: str) -> Arm:
    if name not in BASELINE_NAMES:
        raise ValueError("baseline name is not registered")
    if name == "eager_transparent":
        return _transparent(mx)
    if name == "compiled_transparent":
        return mx.compile(_transparent(mx), shapeless=False)
    if name == "fast_rms_norm":
        return _fast(mx)
    return mx.compile(_fast(mx), shapeless=False)


def make_all_baselines(mx: Any) -> dict[str, Arm]:
    return {name: make_baseline(mx, name) for name in BASELINE_NAMES}


def evaluate(mx: Any, np: Any, arm: Arm, inputs: tuple[Any, Any, Any]) -> Any:
    output = arm(*inputs)
    mx.eval(output)
    mx.synchronize()
    return np.array(output, dtype="<f2", copy=True)


def _q99(np: Any, values: Any) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 0.99, method="linear"))


def accuracy_metrics(np: Any, output: Any, oracle: Any, *, exact_zero: bool) -> dict[str, Any]:
    if output.shape != (ROWS, HIDDEN_SIZE) or output.dtype != np.dtype("<f2"):
        raise ValueError("output shape or dtype differs")
    finite = bool(np.isfinite(output).all())
    difference = np.abs(output.astype(np.float64) - oracle.astype(np.float64))
    abs_max = float(np.max(difference))
    oracle64 = oracle.astype(np.float64)
    mask = np.abs(oracle64) >= RELATIVE_ORACLE_FLOOR
    relative = difference[mask] / np.abs(oracle64[mask]) if bool(mask.any()) else np.empty(0)
    rel_q99 = _q99(np, relative)
    denominator = max(float(np.linalg.norm(oracle64.ravel())), float(np.finfo(np.float64).tiny))
    normalized_l2 = float(np.linalg.norm(difference.ravel()) / denominator)
    exact_zero_passed = (
        bool(np.all(output.view(np.uint16) == 0)) if exact_zero else True
    )
    gates = {
        "finite": finite,
        "abs_max": abs_max <= ABS_MAX_CAP,
        "rel_q99": rel_q99 <= REL_Q99_CAP,
        "normalized_l2": normalized_l2 <= NORMALIZED_L2_CAP,
        "exact_zero": exact_zero_passed,
    }
    return {
        "finite": finite,
        "abs_max": abs_max,
        "rel_q99": rel_q99,
        "normalized_l2": normalized_l2,
        "exact_zero": exact_zero_passed,
        "gates": gates,
        "passed": all(gates.values()),
    }


def pair_metrics(np: Any, candidate: Any, baseline: Any) -> dict[str, Any]:
    difference = np.abs(candidate.astype(np.float64) - baseline.astype(np.float64))
    threshold = PAIR_ATOL + PAIR_RTOL * np.abs(baseline.astype(np.float64))
    passed = bool(np.all(difference <= threshold))
    return {
        "atol": PAIR_ATOL,
        "rtol": PAIR_RTOL,
        "max_abs": float(np.max(difference)),
        "passed": passed,
    }


def warmup(mx: Any, arm: Arm, inputs: tuple[Any, Any, Any], count: int) -> None:
    for _ in range(count):
        output = arm(*inputs)
        mx.eval(output)
        mx.synchronize()


def measure_arms(
    mx: Any,
    arms: dict[str, Arm],
    inputs: tuple[Any, Any, Any],
    *,
    blocks: int,
    operations: int,
    order_seed: int,
) -> dict[str, Any]:
    if not arms or blocks <= 0 or operations <= 0:
        raise ValueError("timing geometry is invalid")
    names = tuple(arms)
    generator = random.Random(order_seed)
    samples = {name: [] for name in names}
    base = list(names)
    generator.shuffle(base)
    orders = [
        base[offset:] + base[:offset]
        for offset in (block % len(base) for block in range(blocks))
    ]
    generator.shuffle(orders)
    for order in orders:
        for name in order:
            mx.synchronize()
            started = time.perf_counter_ns()
            for _operation in range(operations):
                output = arms[name](*inputs)
                mx.eval(output)
                mx.synchronize()
            elapsed = time.perf_counter_ns() - started
            if elapsed <= 0:
                raise RuntimeError("non-positive timing block")
            samples[name].append(elapsed / operations)
    return {"samples_ns": samples, "orders": orders}


def memory_probe(
    mx: Any,
    arm: Arm,
    inputs: tuple[Any, Any, Any],
    *,
    operations: int,
) -> dict[str, int]:
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    output = None
    for _ in range(operations):
        output = arm(*inputs)
        mx.eval(output)
        mx.synchronize()
    result = {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    del output
    gc.collect()
    return result
