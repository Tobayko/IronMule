"""Lazy H0 matmul benchmark engine with an injectable, deterministic backend.

The production backend is created only inside :func:`run_mlx_benchmark` and imports
NumPy/MLX lazily.  Offline tests inject a backend implementing this narrow adapter:

``from_host(array)``, ``matmul(a, b)``, ``eval(output)``, ``synchronize()``,
``to_host(output)`` and, for the compile arm, ``compile(function, shapeless=False)``.
Optional ``memory_metrics()`` and ``set_memory_limit(bytes)`` are best-effort
telemetry hooks.  The engine never executes a custom kernel or accepts source/code
from a manifest.

The returned object is strict JSON-compatible.  ``benchmark_classification`` retains
H0 domain values such as ``tie`` and ``baseline_reference``.  The existing common
worker-result protocol has no such values; ``adapter_contract`` therefore states the
required worker mapping instead of silently turning a null-control result into a
promotion claim.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json_bytes
from .correctness_contract import (
    CORRECTNESS_CASES,
    CORRECTNESS_HARD_CAPS,
    CorrectnessContractError,
    PERFORMANCE_CASE_NAME,
    fixture_digest,
    memory_api_error_reason,
    MEMORY_MAX_INT,
    memory_name_allowed,
    normalize_memory_missing_reason,
    trusted_performance_fixture_identity,
    validate_fixed_case,
    validate_performance_case,
    validate_sign_invariant_case,
)
from .manifest import ManifestError, manifest_hash, validate_manifest
from .protocol import PRODUCTION_JSON_DEPTH


H0_BATCH_MIN_NS = 50_000_000
H0_BATCH_MAX_NS = 200_000_000
H0_MAX_REPETITIONS = 4096
H0_INITIAL_WARMUPS = 8
H0_MAX_WARMUPS = 16
H0_WARMUP_TOLERANCE = 0.05
H0_BLOCKS = 30
H0_FIRST_EVAL_LIMIT_NS = 10_000_000_000
H0_SYNC_LIMIT_NS = 5_000_000_000
H0_TOTAL_LIMIT_NS = 120_000_000_000
H0_MEMORY_LIMIT_BYTES = 1 << 30
H0_TINY = 1e-30
H0_MAX_RESULT_BYTES = 1 << 20
H0_MAX_MEMORY_ENTRIES = 128
H0_CACHE_STATE = "unknown"

# This is the single registry for benchmark-domain failures.  The worker imports
# this value instead of maintaining a second, drifting allowlist at the protocol
# boundary.
REGISTERED_BENCHMARK_ERROR_CODES = frozenset(
    {
        "non_finite_measurement",
        "non_finite_result",
        "non_json_result",
        "fixture_metadata_invalid",
        "invalid_fixture_seed",
        "invalid_fixture_shape",
        "backend_contract",
        "compile_contract",
        "runtime_unavailable",
        "clock_contract",
        "evaluation_timeout_observed",
        "synchronization_timeout_observed",
        "empty_measurements",
        "warmup_unstable",
        "repetition_window_unreachable",
        "order_contract",
        "correctness_shape",
        "correctness_nonfinite",
        "invalid_ratio",
        "correctness_failed",
        "correctness_contract",
        "mode_not_benchmarkable",
        "total_timeout_observed",
        "manifest_invalid",
        "backend_error",
        "result_too_large",
    }
)


def _registered_failure_code(code: Any) -> str:
    """Return a closed failure code for the benchmark-domain boundary."""

    if isinstance(code, str) and code in REGISTERED_BENCHMARK_ERROR_CODES:
        return code
    return "backend_error"

_JSON_SAFE_MAX_NODES = 50_000
_JSON_SAFE_MAX_SEQUENCE = 10_000
_JSON_SAFE_MAX_STRING = 64 * 1024
_JSON_SAFE_MAX_INT = (1 << 63) - 1


def _adapter_contract() -> dict[str, Any]:
    """Return a fresh registered adapter contract for each domain result."""

    return {
        "common_result_ready": False,
        "reason": "single-process measurements require aggregation before any global decision",
        "mapping": {
            "runtime_unavailable": "invalid/baseline_fallback",
            "invalid*": "invalid/baseline_fallback",
            "measurement_complete": "aggregation_required",
            "baseline_reference": "not_run",
        },
    }

_CORRECTNESS_CASES = CORRECTNESS_CASES


class BenchmarkError(RuntimeError):
    """A fail-closed benchmark condition with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.evidence = dict(evidence or {})


class RuntimeUnavailable(BenchmarkError):
    """Raised only when the requested MLX backend cannot be created."""


_MAX_DIAGNOSTIC_INT = (1 << 63) - 1


def _positive_diagnostic_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > _MAX_DIAGNOSTIC_INT:
        return None
    return value


def _failure_diagnostic(code: Any, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Project internal error evidence into the closed invalid-result envelope."""

    code = _registered_failure_code(code)
    details: dict[str, Any] = {}
    source = evidence if isinstance(evidence, Mapping) else {}
    if code in {"evaluation_timeout_observed", "synchronization_timeout_observed", "total_timeout_observed"}:
        field = {
            "evaluation_timeout_observed": "evaluation_ns",
            "synchronization_timeout_observed": "synchronize_ns",
            "total_timeout_observed": "total_ns",
        }[code]
        value = _positive_diagnostic_int(source.get(field))
        if value is not None:
            details[field] = value
    elif code == "warmup_unstable":
        block_values = source.get("warmup_block_per_eval_ns")
        blocks = source.get("warmup_blocks")
        if isinstance(block_values, (list, tuple)) and isinstance(blocks, (list, tuple)):
            if len(block_values) == H0_MAX_WARMUPS and len(blocks) == H0_MAX_WARMUPS:
                values = [_positive_diagnostic_int(value) for value in block_values]
                copied_blocks: list[dict[str, int]] = []
                valid = all(value is not None for value in values)
                for index, block in enumerate(blocks):
                    if not isinstance(block, Mapping):
                        valid = False
                        break
                    expected_keys = {
                        "block_index", "evaluations", "block_ns", "per_eval_ns",
                        "median_eval_ns", "min_eval_ns", "max_eval_ns",
                    }
                    if set(block) != expected_keys:
                        valid = False
                        break
                    block_index = block.get("block_index")
                    evaluations = block.get("evaluations")
                    block_ns = block.get("block_ns")
                    per_eval_ns = block.get("per_eval_ns")
                    median_eval_ns = block.get("median_eval_ns")
                    min_eval_ns = block.get("min_eval_ns")
                    max_eval_ns = block.get("max_eval_ns")
                    if (
                        isinstance(block_index, bool) or not isinstance(block_index, int) or block_index != index
                        or _positive_diagnostic_int(evaluations) is None or evaluations > H0_MAX_REPETITIONS
                        or _positive_diagnostic_int(block_ns) is None or block_ns < H0_BATCH_MIN_NS
                        or _positive_diagnostic_int(per_eval_ns) is None
                        or any(
                            isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > _MAX_DIAGNOSTIC_INT
                            for item in (median_eval_ns, min_eval_ns, max_eval_ns)
                        )
                        or not (min_eval_ns <= median_eval_ns <= max_eval_ns)
                        or per_eval_ns != max(1, int(round(block_ns / evaluations)))
                        or values[index] != per_eval_ns
                    ):
                        valid = False
                        break
                    copied_blocks.append({key: int(block[key]) for key in expected_keys})
                if valid:
                    details = {
                        "warmup_block_per_eval_ns": [int(value) for value in values],
                        "warmup_blocks": copied_blocks,
                    }
        if not details:
            warmups = source.get("warmups_ns")
            if isinstance(warmups, (list, tuple)) and len(warmups) == H0_MAX_WARMUPS:
                values = [_positive_diagnostic_int(value) for value in warmups]
                if all(value is not None for value in values):
                    details["warmups_ns"] = [int(value) for value in values]
    elif code == "repetition_window_unreachable":
        repetitions = _positive_diagnostic_int(source.get("repetitions"))
        batch_ns = _positive_diagnostic_int(source.get("batch_ns"))
        if repetitions is not None and repetitions <= 4096 and repetitions & (repetitions - 1) == 0 and batch_ns is not None:
            details = {"repetitions": repetitions, "batch_ns": batch_ns}
    elif code == "correctness_failed":
        correctness = source.get("correctness")
        if isinstance(correctness, Mapping):
            try:
                copied = _json_safe(correctness)
            except (BenchmarkError, RecursionError, TypeError, ValueError):
                copied = {}
            if isinstance(copied, dict):
                details = {"correctness": copied}
    elif code == "result_too_large":
        details = {"truncated": True, "missing_reason": "result_limit"}
    schema_version = 2 if code == "warmup_unstable" and set(details) == {"warmup_block_per_eval_ns", "warmup_blocks"} else 1
    return {"schema_version": schema_version, "code": code, "details": details}


@dataclass(frozen=True)
class _Fixture:
    a: Any
    b: Any
    fixture_seed: int
    a_sha256: str
    b_sha256: str
    metadata_sha256: str
    fixture_sha256: str


@dataclass(frozen=True)
class _Timed:
    duration_ns: int
    evaluation_ns: int
    synchronize_ns: int


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise BenchmarkError("non_finite_measurement", f"{name} is not finite")
    return float(value)


def _json_safe(value: Any) -> Any:
    """Convert scalar-like values under deterministic JSON safety budgets."""

    active: set[int] = set()
    nodes = 0

    def visit(current: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > _JSON_SAFE_MAX_NODES:
            raise BenchmarkError("result_too_large", "result exceeds the JSON node budget")
        if depth > PRODUCTION_JSON_DEPTH:
            raise BenchmarkError("result_too_large", "result exceeds the JSON depth budget")
        if current is None or isinstance(current, bool):
            return current
        if isinstance(current, int):
            if not -_JSON_SAFE_MAX_INT <= current <= _JSON_SAFE_MAX_INT:
                raise BenchmarkError("result_too_large", "result integer exceeds signed 64-bit bounds")
            return current
        if isinstance(current, float):
            if not math.isfinite(current):
                raise BenchmarkError("non_finite_result", "result contains a non-finite float")
            return current
        if isinstance(current, str):
            if len(canonical_json_bytes(current)) > _JSON_SAFE_MAX_STRING:
                raise BenchmarkError("result_too_large", "result string exceeds the bounded length")
            return current
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in active:
                raise BenchmarkError("non_json_result", "result contains a cycle")
            active.add(identity)
            try:
                if len(current) > _JSON_SAFE_MAX_SEQUENCE:
                    raise BenchmarkError("result_too_large", "result object exceeds the key budget")
                result: dict[str, Any] = {}
                for key, child in current.items():
                    text_key = str(key)
                    if len(canonical_json_bytes(text_key)) > _JSON_SAFE_MAX_STRING:
                        raise BenchmarkError("result_too_large", "result key exceeds the bounded length")
                    result[text_key] = visit(child, depth + 1)
                return result
            finally:
                active.remove(identity)
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            identity = id(current)
            if identity in active:
                raise BenchmarkError("non_json_result", "result contains a cycle")
            active.add(identity)
            try:
                if len(current) > _JSON_SAFE_MAX_SEQUENCE:
                    raise BenchmarkError("result_too_large", "result sequence exceeds the item budget")
                return [visit(child, depth + 1) for child in current]
            finally:
                active.remove(identity)
        item = getattr(current, "item", None)
        if callable(item):
            identity = id(current)
            if identity in active:
                raise BenchmarkError("non_json_result", "result contains a cycle")
            active.add(identity)
            try:
                return visit(item(), depth + 1)
            finally:
                active.remove(identity)
        raise BenchmarkError("non_json_result", f"unsupported result value type {type(current).__name__}")

    return visit(value, 0)


def _bounded_text(value: Any, maximum: int = 1024) -> str:
    text = str(value)
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def _tag_calibration(samples: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [{**sample, "arm": arm, "position": "calibration"} for sample in samples]


def _bounded_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return finite JSON evidence while enforcing the one MiB result contract."""

    try:
        safe = _json_safe(result)
    except BenchmarkError as exc:
        if exc.code != "result_too_large":
            raise
        safe = {
            "schema_version": 1,
            "run_id": _bounded_text(result.get("run_id", ""), 128),
            "mode": _bounded_text(result.get("mode", ""), 128),
            "manifest_sha256": result.get("manifest_sha256"),
            "status": "invalid",
            "classification": "invalid",
            "benchmark_classification": "invalid",
            "action": "baseline_fallback",
            "error": {"code": "result_too_large", "message": "result exceeded one MiB"},
            "evidence": {"failure_diagnostic": _failure_diagnostic("result_too_large")},
            "adapter_contract": _adapter_contract(),
        }
    encoded = json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) <= H0_MAX_RESULT_BYTES:
        return safe
    compact = {
        "schema_version": safe.get("schema_version", 1),
        "run_id": _bounded_text(safe.get("run_id", ""), 128),
        "mode": _bounded_text(safe.get("mode", ""), 128),
        "manifest_sha256": safe.get("manifest_sha256"),
        "status": "invalid",
        "classification": "invalid",
        "benchmark_classification": "invalid",
        "action": "baseline_fallback",
        "error": {"code": "result_too_large", "message": "result exceeded one MiB"},
        "evidence": {"failure_diagnostic": _failure_diagnostic("result_too_large")},
        # Keep the oversize fallback a valid benchmark-domain envelope.  The
        # worker must be able to preserve this diagnostic instead of turning a
        # bounded result into a second protocol error.
        "adapter_contract": _adapter_contract(),
    }
    return compact


def _canonical_metadata(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("fixture_metadata_invalid", str(exc)) from exc


def _little_endian_fp16(np_module: Any, value: Any) -> Any:
    array = np_module.asarray(value, dtype="<f2", order="C")
    if not array.flags.c_contiguous:
        array = np_module.ascontiguousarray(array, dtype="<f2")
    return array


def _hash_array(array: Any) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _generate_fixture(np_module: Any, fixture_seed: int, *, shape: int = 2048) -> _Fixture:
    """Generate PCG64 FP32 host values, then convert the exact inputs to little-endian FP16."""

    if isinstance(fixture_seed, bool) or not isinstance(fixture_seed, int) or fixture_seed < 0:
        raise BenchmarkError("invalid_fixture_seed", "fixture seed must be an unsigned integer")
    if isinstance(shape, bool) or not isinstance(shape, int) or shape <= 0:
        raise BenchmarkError("invalid_fixture_shape", "fixture shape must be positive")
    generator = np_module.random.Generator(np_module.random.PCG64(fixture_seed))
    a32 = generator.uniform(-1.0, 1.0, size=(shape, shape)).astype(np_module.float32)
    b32 = generator.uniform(-1.0, 1.0, size=(shape, shape)).astype(np_module.float32)
    a = _little_endian_fp16(np_module, a32)
    b = _little_endian_fp16(np_module, b32)
    metadata = {
        "dtype": "float16",
        "endianness": "little",
        "generator": "PCG64",
        "distribution": "uniform[-1,1)",
        "shape": [shape, shape],
        "fixture_seed": fixture_seed,
    }
    metadata_bytes = _canonical_metadata(metadata)
    a_digest = _hash_array(a)
    b_digest = _hash_array(b)
    fixture_digest = hashlib.sha256(
        b"friday_h0_fixture_v1\0" + a.tobytes(order="C") + b.tobytes(order="C") + metadata_bytes
    ).hexdigest()
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    if shape == 2048:
        try:
            trusted = trusted_performance_fixture_identity(
                a_shape=a.shape, b_shape=b.shape, dtype=str(a.dtype), layout="C-contiguous", fixture_seed=fixture_seed,
            )
        except CorrectnessContractError as exc:
            raise BenchmarkError("correctness_contract", str(exc)) from exc
        if (a_digest, b_digest, metadata_digest, fixture_digest) != (
            trusted["a_sha256"], trusted["b_sha256"], trusted["metadata_sha256"], trusted["fixture_sha256"]
        ):
            raise BenchmarkError("correctness_contract", "generated performance fixture is not the trusted identity")
    return _Fixture(a, b, fixture_seed, a_digest, b_digest, metadata_digest, fixture_digest)


class _BackendAdapter:
    """Adapter with strict method names, keeping the engine independent of MLX."""

    def __init__(self, backend: Any, np_module: Any) -> None:
        self.backend = backend
        self.np = np_module

    def from_host(self, value: Any) -> Any:
        method = getattr(self.backend, "from_host", None)
        if callable(method):
            return method(value)
        method = getattr(self.backend, "array", None)
        if callable(method):
            return method(value)
        raise BenchmarkError("backend_contract", "backend lacks from_host/array")

    def matmul(self, a: Any, b: Any) -> Any:
        method = getattr(self.backend, "matmul", None)
        if not callable(method):
            raise BenchmarkError("backend_contract", "backend lacks matmul")
        return method(a, b)

    def eval(self, value: Any) -> Any:
        method = getattr(self.backend, "eval", None)
        if not callable(method):
            raise BenchmarkError("backend_contract", "backend lacks eval")
        return method(value)

    def synchronize(self) -> None:
        method = getattr(self.backend, "synchronize", None)
        if not callable(method):
            raise BenchmarkError("backend_contract", "backend lacks synchronize")
        method()

    def to_host(self, value: Any) -> Any:
        method = getattr(self.backend, "to_host", None)
        if callable(method):
            return method(value)
        return self.np.asarray(value)

    def compile(self, function: Callable[..., Any]) -> Callable[..., Any]:
        method = getattr(self.backend, "compile", None)
        if not callable(method):
            raise BenchmarkError("backend_contract", "backend lacks compile for compile_comparison")
        try:
            return method(function, shapeless=False)
        except TypeError as exc:
            raise BenchmarkError("compile_contract", "compile must accept shapeless=False") from exc

    def set_memory_limit(self, limit_bytes: int) -> dict[str, Any]:
        method = getattr(self.backend, "set_memory_limit", None)
        if not callable(method):
            return {"attempted": False, "hard_limit": False, "applied": False, "missing_reason": "api_unavailable"}
        try:
            returned = method(limit_bytes)
        except Exception as exc:  # best effort by contract; never claim enforcement
            return {
                "attempted": True,
                "hard_limit": False,
                "applied": False,
                "missing_reason": memory_api_error_reason(exc),
            }
        # MLX's public API returns the previous limit as a signed integer.  Do
        # not accept bool (an int subclass), None, or a truthy arbitrary value.
        if type(returned) is not int or returned < 0 or returned > (1 << 63) - 1:
            return {"attempted": True, "hard_limit": False, "applied": False, "missing_reason": "invalid_source_value"}
        return {"attempted": True, "hard_limit": False, "applied": True, "missing_reason": None}

    def memory_metrics(self) -> list[dict[str, Any]]:
        custom = getattr(self.backend, "memory_metrics", None)
        if callable(custom):
            try:
                values = custom()
            except Exception as exc:  # telemetry is non-fatal and explicit
                return [{"name": "custom", "api": "memory_metrics", "unit": "bytes", "value": None, "missing_reason": memory_api_error_reason(exc)}]
            return _normalize_memory_values(values)
        candidates = (
            ("mlx_active_memory", "get_active_memory"),
            ("mlx_peak_memory", "get_peak_memory"),
            ("mlx_cache_memory", "get_cache_memory"),
            ("rss", "get_rss_memory"),
        )
        result = []
        for name, api_name in candidates:
            method = getattr(self.backend, api_name, None)
            if not callable(method):
                result.append({"name": name, "api": api_name, "unit": "bytes", "value": None, "missing_reason": "api_unavailable"})
                continue
            try:
                raw = method()
                if type(raw) is not int or raw < 0 or raw > MEMORY_MAX_INT:
                    raise ValueError("invalid_source_value")
                number = raw
            except Exception as exc:
                reason = "invalid_source_value" if str(exc) == "invalid_source_value" else memory_api_error_reason(exc)
                result.append({"name": name, "api": api_name, "unit": "bytes", "value": None, "missing_reason": reason})
            else:
                result.append({"name": name, "api": api_name, "unit": "bytes", "value": number, "missing_reason": None})
        return result


def _normalize_memory_values(values: Any) -> list[dict[str, Any]]:
    if isinstance(values, Mapping):
        values = [dict(name=str(key), api=str(key), unit="bytes", value=value, missing_reason=None) for key, value in values.items()]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return [{"name": "custom", "api": "memory_metrics", "unit": "bytes", "value": None, "missing_reason": "invalid_api_payload"}]
    result = []
    for entry in list(values)[:H0_MAX_MEMORY_ENTRIES]:
        if not isinstance(entry, Mapping):
            result.append({"name": "custom", "api": "memory_metrics", "unit": "bytes", "value": None, "missing_reason": "invalid_api_payload"})
            continue
        name = str(entry.get("name", entry.get("api", "custom")))
        if not memory_name_allowed(name):
            result.append({"name": "custom", "api": "memory_metrics", "unit": "bytes", "value": None, "missing_reason": "invalid_api_payload"})
            continue
        api = str(entry.get("api", name))
        unit = str(entry.get("unit", "bytes"))
        value = entry.get("value")
        reason = entry.get("missing_reason")
        if value is not None:
            try:
                if type(value) is not int or value < 0 or value > MEMORY_MAX_INT:
                    raise ValueError("invalid_source_value")
                reason = None
            except (TypeError, ValueError):
                value, reason = None, "invalid_source_value"
        elif not isinstance(reason, str) or not reason:
            reason = "not_recorded"
        else:
            reason = normalize_memory_missing_reason(reason)
        result.append({"name": name, "api": api, "unit": unit, "value": value, "missing_reason": reason})
    if len(values) > H0_MAX_MEMORY_ENTRIES:
        result.append({"name": "memory_metrics", "api": "memory_metrics", "unit": "bytes", "value": None, "missing_reason": "entry_limit"})
    return result


class _MLXBackend:
    def __init__(self, np_module: Any) -> None:
        try:
            mlx = importlib.import_module("mlx.core")
        except Exception as exc:
            raise RuntimeUnavailable("runtime_unavailable", f"MLX import unavailable: {type(exc).__name__}") from exc
        self.mx = mlx
        self.np = np_module

    def from_host(self, value: Any) -> Any:
        return self.mx.array(value)

    def matmul(self, a: Any, b: Any) -> Any:
        return self.mx.matmul(a, b)

    def eval(self, value: Any) -> Any:
        self.mx.eval(value)
        return value

    def synchronize(self) -> None:
        self.mx.synchronize()

    def to_host(self, value: Any) -> Any:
        return self.np.asarray(value)

    def compile(self, function: Callable[..., Any], *, shapeless: bool = False) -> Callable[..., Any]:
        return self.mx.compile(function, shapeless=shapeless)

    def set_memory_limit(self, limit_bytes: int) -> Any:
        return self.mx.set_memory_limit(limit_bytes)

    def get_active_memory(self) -> Any:
        return self.mx.get_active_memory()

    def get_peak_memory(self) -> Any:
        return self.mx.get_peak_memory()

    def get_cache_memory(self) -> Any:
        return self.mx.get_cache_memory()


def _make_backend(factory: Any, manifest: Mapping[str, Any], np_module: Any) -> _BackendAdapter:
    if factory is None:
        return _BackendAdapter(_MLXBackend(np_module), np_module)
    backend = factory
    if callable(factory):
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            backend = factory(manifest, np_module)
        else:
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_varargs = any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
            if has_varargs or len(positional) >= 2:
                backend = factory(manifest, np_module)
            elif len(positional) == 1:
                backend = factory(manifest)
            else:
                backend = factory()
    return _BackendAdapter(backend, np_module)


def _load_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except Exception as exc:
        raise RuntimeUnavailable("runtime_unavailable", f"NumPy import unavailable: {type(exc).__name__}") from exc


def _clock(clock_ns: Callable[[], int] | None) -> Callable[[], int]:
    value = clock_ns or time.perf_counter_ns
    if not callable(value):
        raise BenchmarkError("clock_contract", "clock_ns must be callable")
    return value


def _nonnegative_delta(start: Any, end: Any, name: str) -> int:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise BenchmarkError("clock_contract", f"{name} clock must return integers")
    delta = end - start
    if delta < 0:
        raise BenchmarkError("clock_contract", f"{name} clock moved backwards")
    return delta


def _measure_once(backend: _BackendAdapter, function: Callable[[], Any], clock: Callable[[], int]) -> _Timed:
    start = clock()
    output = function()
    backend_output = backend.eval(output)
    evaluated = clock()
    backend.synchronize()
    finished = clock()
    evaluation_ns = _nonnegative_delta(start, evaluated, "evaluation")
    synchronize_ns = _nonnegative_delta(evaluated, finished, "synchronization")
    duration_ns = _nonnegative_delta(start, finished, "measurement")
    if evaluation_ns > H0_FIRST_EVAL_LIMIT_NS:
        raise BenchmarkError("evaluation_timeout_observed", "single evaluation exceeded 10 seconds", evidence={"evaluation_ns": evaluation_ns})
    if synchronize_ns > H0_SYNC_LIMIT_NS:
        raise BenchmarkError("synchronization_timeout_observed", "synchronization exceeded 5 seconds", evidence={"synchronize_ns": synchronize_ns})
    del backend_output
    return _Timed(duration_ns, evaluation_ns, synchronize_ns)


def _median(values: Sequence[int | float]) -> float:
    if not values:
        raise BenchmarkError("empty_measurements", "measurement sequence is empty")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mad(values: Sequence[int | float]) -> float:
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _iqr(values: Sequence[int | float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BenchmarkError("empty_measurements", "measurement sequence is empty")
    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return quantile(0.75) - quantile(0.25)


def _stable_last_five(values: Sequence[int | float]) -> bool:
    if len(values) < 5:
        return False
    center = _median(values[-5:])
    if center == 0:
        return all(float(value) == 0.0 for value in values[-5:])
    return all(0.95 * center <= float(value) <= 1.05 * center for value in values[-5:])


def _per_eval_ns(block_ns: int, evaluations: int) -> int:
    if evaluations <= 0:
        raise BenchmarkError("empty_measurements", "block has no evaluations")
    return max(1, int(round(block_ns / evaluations)))


def _warmup_block(
    backend: _BackendAdapter,
    function: Callable[[], Any],
    clock: Callable[[], int],
    block_index: int,
) -> dict[str, int]:
    """Run one time-based warmup block with the production batch statistic."""

    started = clock()
    timings: list[_Timed] = []
    while True:
        timings.append(_measure_once(backend, function, clock))
        elapsed = _nonnegative_delta(started, clock(), "warmup_block")
        if elapsed >= H0_BATCH_MIN_NS:
            break
        if len(timings) >= H0_MAX_REPETITIONS:
            raise BenchmarkError(
                "repetition_window_unreachable",
                "4096 warmup evaluations remain below the 50 ms minimum",
                evidence={"repetitions": H0_MAX_REPETITIONS, "batch_ns": max(1, elapsed)},
            )
    block_ns = elapsed
    duration_values = [timing.duration_ns for timing in timings]
    return {
        "block_index": block_index,
        "evaluations": len(timings),
        "block_ns": block_ns,
        "per_eval_ns": _per_eval_ns(block_ns, len(timings)),
        "median_eval_ns": max(0, int(round(_median(duration_values)))),
        "min_eval_ns": min(duration_values),
        "max_eval_ns": max(duration_values),
    }


def _warmups(backend: _BackendAdapter, function: Callable[[], Any], clock: Callable[[], int]) -> dict[str, Any]:
    blocks: list[dict[str, int]] = []
    samples: list[dict[str, int | str]] = []
    while len(blocks) < H0_INITIAL_WARMUPS or not _stable_last_five([block["per_eval_ns"] for block in blocks]):
        block = _warmup_block(backend, function, clock, len(blocks))
        blocks.append(block)
        # The closed success schema stores one bounded gate sample and one
        # bounded summary per block; it never retains per-evaluation objects.
        samples.append({"phase": "warmup", "sample_index": len(samples), "value": block["per_eval_ns"], "unit": "ns"})
        if len(blocks) >= H0_MAX_WARMUPS:
            break
    gate_values = [block["per_eval_ns"] for block in blocks]
    stable = _stable_last_five(gate_values)
    if not stable:
        raise BenchmarkError(
            "warmup_unstable",
            "last five warmup block gate values are not stable after 16 blocks",
            evidence={"warmup_block_per_eval_ns": gate_values, "warmup_blocks": blocks},
        )
    return {
        "count": len(blocks),
        "durations_ns": gate_values,
        "stable": stable,
        "median_ns": _median(gate_values[-5:]),
        "samples": samples,
        "blocks": blocks,
    }


def _batch(backend: _BackendAdapter, function: Callable[[], Any], repetitions: int, clock: Callable[[], int]) -> tuple[int, list[_Timed]]:
    start = clock()
    timings = [_measure_once(backend, function, clock) for _ in range(repetitions)]
    finished = clock()
    elapsed = _nonnegative_delta(start, finished, "batch")
    return elapsed, timings


def choose_repetitions(backend: _BackendAdapter, function: Callable[[], Any], clock_ns: Callable[[], int]) -> dict[str, Any]:
    """Select the smallest power-of-two batch in the registered 50--200 ms window."""

    repetitions = 1
    while repetitions <= H0_MAX_REPETITIONS:
        elapsed, timings = _batch(backend, function, repetitions, clock_ns)
        if H0_BATCH_MIN_NS <= elapsed <= H0_BATCH_MAX_NS:
            return {
                "repetitions": repetitions,
                "batch_ns": elapsed,
                "probe_timings": [timing.duration_ns for timing in timings],
                "calibration_samples": [
                    {"phase": "repetition_probe", "repetitions": repetitions, "sample_index": index, "value": timing.duration_ns, "unit": "ns"}
                    for index, timing in enumerate(timings)
                ],
            }
        if elapsed > H0_BATCH_MAX_NS:
            raise BenchmarkError("repetition_window_unreachable", "one repetition already exceeds the 50-200 ms target", evidence={"repetitions": repetitions, "batch_ns": elapsed})
        repetitions *= 2
    raise BenchmarkError("repetition_window_unreachable", "4096 repetitions remain below the 50-200 ms target")


def _balanced_order(seed: int, count: int = H0_BLOCKS) -> list[str]:
    if count <= 0 or count % 2:
        raise BenchmarkError("order_contract", "paired block count must be positive and even")
    values = ["baseline"] * (count // 2) + ["candidate"] * (count // 2)
    random.Random(seed).shuffle(values)
    return values


def _arm_statistics(values: Sequence[int | float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median_ns": _median(values),
        "mad_ns": _mad(values),
        "iqr_ns": _iqr(values),
        "min_ns": min(values),
        "max_ns": max(values),
    }


def _run_arm(backend: _BackendAdapter, function: Callable[[], Any], clock: Callable[[], int], *, first_eval_ns: int | None = None) -> dict[str, Any]:
    warmup = _warmups(backend, function, clock)
    repetition = choose_repetitions(backend, function, clock)
    batches: list[dict[str, Any]] = []
    per_eval: list[float] = []
    calibration_samples = _tag_calibration(list(warmup["samples"]) + list(repetition["calibration_samples"]), "baseline")
    raw_samples = list(calibration_samples)
    for block_index in range(H0_BLOCKS):
        elapsed, timings = _batch(backend, function, repetition["repetitions"], clock)
        per_eval_ns = elapsed / repetition["repetitions"]
        per_eval.append(per_eval_ns)
        raw_samples.append({"phase": "measurement", "sample_kind": "timing_batch", "sample_index": block_index, "block_index": block_index, "arm": "baseline", "position": "single", "repetitions": repetition["repetitions"], "value": elapsed, "unit": "ns"})
        batches.append(
            {
                "block_index": block_index,
                "batch_ns": elapsed,
                "per_eval_ns": per_eval_ns,
                "repetitions": repetition["repetitions"],
                "position": "single",
                "evaluation_ns": [timing.evaluation_ns for timing in timings],
                "synchronize_ns": [timing.synchronize_ns for timing in timings],
            }
        )
    result = {
        "warmup": warmup,
        "repetitions": repetition,
        "batches": batches,
        "statistics": _arm_statistics(per_eval),
        "calibration_samples": calibration_samples,
        "raw_samples": raw_samples,
    }
    if first_eval_ns is not None:
        result["first_eval_compile_inclusive_ns"] = first_eval_ns
    return result


def _run_paired_arms(
    backend: _BackendAdapter,
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    clock: Callable[[], int],
    order: Sequence[str],
    *,
    candidate_first_eval_ns: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Warm/calibrate each arm, then execute every pair in the registered order."""

    prepared = {}
    for name, function in (("baseline", baseline), ("candidate", candidate)):
        warmup = _warmups(backend, function, clock)
        repetition = choose_repetitions(backend, function, clock)
        prepared[name] = {
            "warmup": warmup,
            "repetitions": repetition,
            "batches": [],
            "per_eval": [],
            "raw_samples": _tag_calibration(list(warmup["samples"]) + list(repetition["calibration_samples"]), name),
            "memory": _memory_snapshot(backend, clock, phase="after_calibration", arm=name),
        }
    for block_index, first in enumerate(order):
        if first not in {"baseline", "candidate"}:
            raise BenchmarkError("order_contract", "paired order contains an unknown arm")
        second = "candidate" if first == "baseline" else "baseline"
        for position, arm in (("first", first), ("second", second)):
            elapsed, timings = _batch(backend, {"baseline": baseline, "candidate": candidate}[arm], prepared[arm]["repetitions"]["repetitions"], clock)
            per_eval_ns = elapsed / prepared[arm]["repetitions"]["repetitions"]
            prepared[arm]["per_eval"].append(per_eval_ns)
            prepared[arm]["batches"].append({
                "block_index": block_index,
                "batch_ns": elapsed,
                "per_eval_ns": per_eval_ns,
                "repetitions": prepared[arm]["repetitions"]["repetitions"],
                "position": position,
                "evaluation_ns": [timing.evaluation_ns for timing in timings],
                "synchronize_ns": [timing.synchronize_ns for timing in timings],
            })
            prepared[arm]["raw_samples"].append({
                "phase": "measurement", "sample_kind": "timing_batch", "sample_index": block_index,
                "block_index": block_index, "arm": arm, "position": position,
                "repetitions": prepared[arm]["repetitions"]["repetitions"], "value": elapsed, "unit": "ns",
            })
    result = {}
    for name, state in prepared.items():
        result[name] = {
            "warmup": state["warmup"],
            "repetitions": state["repetitions"],
            "batches": state["batches"],
            "statistics": _arm_statistics(state["per_eval"]),
            "calibration_samples": list(state["raw_samples"][: state["warmup"]["count"] + len(state["repetitions"]["calibration_samples"])]),
            "raw_samples": state["raw_samples"],
        }
    if candidate_first_eval_ns is not None:
        result["candidate"]["first_eval_compile_inclusive_ns"] = candidate_first_eval_ns
    return result


def _oracle(np_module: Any, a: Any, b: Any) -> Any:
    return np_module.matmul(np_module.asarray(a, dtype=np_module.float64), np_module.asarray(b, dtype=np_module.float64))


def _error_metrics(np_module: Any, observed: Any, reference: Any) -> dict[str, Any]:
    observed64 = np_module.asarray(observed, dtype=np_module.float64)
    reference64 = np_module.asarray(reference, dtype=np_module.float64)
    if observed64.shape != reference64.shape:
        raise BenchmarkError("correctness_shape", "backend output shape differs from oracle")
    if not np_module.all(np_module.isfinite(observed64)) or not np_module.all(np_module.isfinite(reference64)):
        raise BenchmarkError("correctness_nonfinite", "backend or oracle contains NaN/Inf")
    absolute = np_module.abs(observed64 - reference64)
    relative = absolute / np_module.maximum(np_module.abs(reference64), H0_TINY)
    flat_abs = absolute.reshape(-1)
    flat_rel = relative.reshape(-1)
    oracle_norm = float(np_module.linalg.norm(reference64.reshape(-1)))
    diff_norm = float(np_module.linalg.norm((observed64 - reference64).reshape(-1)))
    normalized_l2 = diff_norm / max(oracle_norm, H0_TINY)
    oracle_inf = float(np_module.max(np_module.abs(reference64))) if reference64.size else 0.0
    scaled_inf = float(np_module.max(absolute)) / max(oracle_inf, H0_TINY)
    large = np_module.abs(reference64).reshape(-1) >= 1.0
    rel_q99_large: dict[str, Any]
    if np_module.any(large):
        rel_q99_large = {"value": float(np_module.quantile(relative.reshape(-1)[large], 0.99)), "missing_reason": None}
    else:
        rel_q99_large = {"value": None, "missing_reason": "no_oracle_elements_abs_ge_1"}
    metrics = {
        "abs_q50": float(np_module.quantile(flat_abs, 0.50)),
        "abs_q95": float(np_module.quantile(flat_abs, 0.95)),
        "abs_q99": float(np_module.quantile(flat_abs, 0.99)),
        "abs_max": float(np_module.max(flat_abs)),
        "rel_q50": float(np_module.quantile(flat_rel, 0.50)),
        "rel_q95": float(np_module.quantile(flat_rel, 0.95)),
        "rel_q99": float(np_module.quantile(flat_rel, 0.99)),
        "rel_max": float(np_module.max(flat_rel)),
        "rel_q99_abs_oracle_ge_1": rel_q99_large,
        "normalized_l2": normalized_l2,
        "scaled_normalized_inf": scaled_inf,
    }
    return metrics


def _correctness_case(backend: _BackendAdapter, np_module: Any, *, name: str, a: Any, b: Any, seed: int, zero_rhs: bool = False) -> dict[str, Any]:
    reference = _oracle(np_module, a, b)
    observed = _run_backend_matmul(backend, a, b)
    metrics = _error_metrics(np_module, observed, reference)
    rel_q99 = metrics["rel_q99_abs_oracle_ge_1"]["value"]
    passed = metrics["abs_max"] <= 1.0 and (rel_q99 is None or rel_q99 <= 0.05) and metrics["normalized_l2"] <= 0.01
    if zero_rhs and not np_module.array_equal(observed, np_module.zeros_like(observed)):
        passed = False
    return {
        "name": name,
        "shape": [int(a.shape[0]), int(a.shape[1]), int(b.shape[0]), int(b.shape[1])],
        "dtype": str(a.dtype),
        "layout": "C-contiguous" if a.flags.c_contiguous and b.flags.c_contiguous else "invalid",
        "seed": seed,
        "a_sha256": _hash_array(a),
        "b_sha256": _hash_array(b),
        "fixture_digest": fixture_digest(_hash_array(a), _hash_array(b), seed),
        "zero_rhs": zero_rhs,
        "metrics": metrics,
        "passed": bool(passed),
        "hard_caps": {"abs_max": 1.0, "rel_q99_abs_oracle_ge_1": 0.05, "normalized_l2": 0.01},
    }


def _run_backend_matmul(backend: _BackendAdapter, a: Any, b: Any) -> Any:
    device_a = backend.from_host(a)
    device_b = backend.from_host(b)
    output = backend.matmul(device_a, device_b)
    materialized = backend.eval(output)
    backend.synchronize()
    return backend.to_host(output if materialized is None else materialized)


def _generate_case(np_module: Any, shape_a: tuple[int, int], shape_b: tuple[int, int], seed: int, low: float, high: float, zero_rhs: bool) -> tuple[Any, Any]:
    generator = np_module.random.Generator(np_module.random.PCG64(seed))
    a = _little_endian_fp16(np_module, generator.uniform(low, high, size=shape_a).astype(np_module.float32))
    if zero_rhs:
        b = np_module.zeros(shape=shape_b, dtype="<f2", order="C")
    else:
        b = _little_endian_fp16(np_module, generator.uniform(low, high, size=shape_b).astype(np_module.float32))
    return a, b


def _correctness_suite(backend: _BackendAdapter, np_module: Any, fixture: _Fixture) -> dict[str, Any]:
    cases = []
    for name, shape_a, shape_b, seed, low, high, zero_rhs in _CORRECTNESS_CASES:
        a, b = _generate_case(np_module, shape_a, shape_b, seed, low, high, zero_rhs)
        case = _correctness_case(backend, np_module, name=name, a=a, b=b, seed=seed, zero_rhs=zero_rhs)
        try:
            validate_fixed_case(case)
        except CorrectnessContractError as exc:
            raise BenchmarkError("correctness_contract", str(exc)) from exc
        cases.append(case)
    performance = _correctness_case(backend, np_module, name="performance_fixture", a=fixture.a, b=fixture.b, seed=fixture.fixture_seed)
    try:
        validate_performance_case(
            performance,
            {
                "a_shape": [int(fixture.a.shape[0]), int(fixture.a.shape[1])],
                "b_shape": [int(fixture.b.shape[0]), int(fixture.b.shape[1])],
                "dtype": str(fixture.a.dtype),
                "layout": "C-contiguous",
                "fixture_seed": fixture.fixture_seed,
                "a_sha256": fixture.a_sha256,
                "b_sha256": fixture.b_sha256,
                "metadata_sha256": fixture.metadata_sha256,
                "fixture_sha256": fixture.fixture_sha256,
            },
        )
    except CorrectnessContractError as exc:
        raise BenchmarkError("correctness_contract", str(exc)) from exc
    sign_a = _run_backend_matmul(backend, -fixture.a, fixture.b)
    sign_reference = -_oracle(np_module, fixture.a, fixture.b)
    sign_metrics = _error_metrics(np_module, sign_a, sign_reference)
    sign_rel_q99 = sign_metrics["rel_q99_abs_oracle_ge_1"]["value"]
    sign_passed = sign_metrics["abs_max"] <= 1.0 and (sign_rel_q99 is None or sign_rel_q99 <= 0.05) and sign_metrics["normalized_l2"] <= 0.01
    sign_invariant = {
        "name": "sign_invariant",
        "seed": fixture.fixture_seed,
        "fixture_digest": fixture_digest(fixture.a_sha256, fixture.b_sha256, fixture.fixture_seed),
        "reference": PERFORMANCE_CASE_NAME,
        "relation": "negate_left_operand",
        "passed": bool(sign_passed),
    }
    try:
        validate_sign_invariant_case(sign_invariant, fixture.fixture_seed, performance["fixture_digest"])
    except CorrectnessContractError as exc:
        raise BenchmarkError("correctness_contract", str(exc)) from exc
    cases.extend([performance, sign_invariant])
    return {"cases": cases, "passed": all(case["passed"] for case in cases), "performance": performance, "sign_invariant": cases[-1]}


def _compile_function(backend: _BackendAdapter, a: Any, b: Any, clock: Callable[[], int]) -> tuple[Callable[[], Any], int, int]:
    device_a = backend.from_host(a)
    device_b = backend.from_host(b)
    setup_start = clock()
    compiled = backend.compile(lambda left, right: backend.matmul(left, right))
    setup_end = clock()
    setup_ns = _nonnegative_delta(setup_start, setup_end, "compile_setup")
    first_output = compiled(device_a, device_b)
    first_eval = _measure_existing_output(backend, first_output, clock)
    return lambda: compiled(device_a, device_b), first_eval.duration_ns if first_eval is not None else setup_ns, setup_ns


def _measure_existing_output(backend: _BackendAdapter, output: Any, clock: Callable[[], int]) -> _Timed:
    start = clock()
    materialized = backend.eval(output)
    evaluated = clock()
    backend.synchronize()
    finished = clock()
    evaluation_ns = _nonnegative_delta(start, evaluated, "evaluation")
    synchronize_ns = _nonnegative_delta(evaluated, finished, "synchronization")
    duration_ns = _nonnegative_delta(start, finished, "measurement")
    if evaluation_ns > H0_FIRST_EVAL_LIMIT_NS:
        raise BenchmarkError("evaluation_timeout_observed", "first evaluation exceeded 10 seconds", evidence={"evaluation_ns": evaluation_ns})
    if synchronize_ns > H0_SYNC_LIMIT_NS:
        raise BenchmarkError("synchronization_timeout_observed", "first synchronization exceeded 5 seconds", evidence={"synchronize_ns": synchronize_ns})
    del materialized
    return _Timed(duration_ns, evaluation_ns, synchronize_ns)


def _comparison_result(arms: Mapping[str, Any], order: Sequence[str], *, aa: bool) -> dict[str, Any]:
    baseline = arms["baseline"]
    candidate = arms["candidate"]
    blocks = []
    raw_samples = []
    ratios = []
    for index, first in enumerate(order):
        second = "candidate" if first == "baseline" else "baseline"
        first_batch = baseline["batches"][index] if first == "baseline" else candidate["batches"][index]
        second_batch = baseline["batches"][index] if second == "baseline" else candidate["batches"][index]
        baseline_ns = baseline["batches"][index]["per_eval_ns"]
        candidate_ns = candidate["batches"][index]["per_eval_ns"]
        ratio = candidate_ns / baseline_ns if baseline_ns > 0 else None
        if ratio is None or not math.isfinite(ratio):
            raise BenchmarkError("invalid_ratio", "paired ratio is not finite")
        ratios.append(ratio)
        blocks.append({"block_index": index, "first": first, "second": second, "baseline_per_eval_ns": baseline_ns, "candidate_per_eval_ns": candidate_ns, "ratio": ratio})
    raw_samples.extend(baseline.get("raw_samples", []))
    raw_samples.extend(candidate.get("raw_samples", []))
    ratio_stats = {
        "count": len(ratios),
        "median_ratio": _median(ratios),
        "mad_ratio": _mad(ratios),
        "iqr_ratio": _iqr(ratios),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
    }
    return {
        "order": list(order),
        "blocks": blocks,
        "raw_samples": raw_samples,
        "ratio_statistics": ratio_stats,
        "benchmark_classification": "session_observation",
        "action": "aggregation_required",
        "aggregation_required": True,
        "aggregation_gate": "aa_gate" if aa else "phase1_promotion_gate",
        "global_decision": None,
    }


def _single_result(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_samples": list(arm.get("raw_samples", [])),
        "benchmark_classification": "baseline_reference",
        "action": "not_run",
        "aggregation_required": False,
        "cache_state": H0_CACHE_STATE,
    }


def _memory_gate(metrics: Sequence[Mapping[str, Any]]) -> str:
    values = {str(metric.get("name")): metric.get("value") for metric in metrics}
    if values.get("mlx_peak_memory") is None or values.get("rss") is None:
        return "not_evaluable_missing_required_metric"
    return "aggregation_required"


def _memory_snapshot(backend: _BackendAdapter, clock: Callable[[], int], *, phase: str, arm: str) -> list[dict[str, Any]]:
    measured_at = clock()
    values = backend.memory_metrics()
    return [
        {
            **metric,
            "measurement_phase": phase,
            "arm": arm,
            "measured_at_ns": measured_at,
            "reset_state": "not_reset_or_api_unavailable",
        }
        for metric in values
    ]


def _result_base(manifest: Mapping[str, Any], digest: str | None, *, status: str, classification: str, action: str, error: Mapping[str, Any] | None, evidence: Mapping[str, Any]) -> dict[str, Any]:
    error_value = dict(error) if error is not None else None
    result_evidence = (
        {"failure_diagnostic": _failure_diagnostic(error_value.get("code", ""), evidence)}
        if status == "invalid" and error_value is not None
        else dict(evidence)
    )
    return {
        "schema_version": 1,
        "run_id": manifest.get("run_id", ""),
        "mode": manifest.get("mode", ""),
        "manifest_sha256": digest,
        "status": status,
        "classification": classification,
        "benchmark_classification": classification,
        "action": action,
        "error": error_value,
        "evidence": result_evidence,
        "adapter_contract": _adapter_contract(),
    }


def run_mlx_benchmark(manifest: dict[str, Any], *, backend_factory: Any = None, clock_ns: Callable[[], int] | None = None) -> dict[str, Any]:
    """Run the fixed H0 harness, or return structured fail-closed evidence.

    ``backend_factory`` may be a backend object or a callable accepting zero args,
    ``manifest``, or ``(manifest, numpy_module)``.  The returned evidence is ready
    for Storage raw-sample insertion; a future worker adapter must validate its
    domain classification before calling the common-result protocol.
    """

    raw_manifest = manifest if isinstance(manifest, Mapping) else {}
    clock = _clock(clock_ns)
    started = clock()
    digest: str | None = None
    try:
        validated = validate_manifest(manifest)
        digest = manifest_hash(validated)
        np_module = _load_numpy()
        fixture = _generate_fixture(np_module, validated["seeds"]["fixture"])
        backend = _make_backend(backend_factory, validated, np_module)
        memory_limit = backend.set_memory_limit(H0_MEMORY_LIMIT_BYTES)
        memory_before = _memory_snapshot(backend, clock, phase="before_correctness", arm="all")
        correctness = _correctness_suite(backend, np_module, fixture)
        if not correctness["passed"]:
            raise BenchmarkError("correctness_failed", "correctness matrix or hard caps failed", evidence={"correctness": correctness})
        device_a = backend.from_host(fixture.a)
        device_b = backend.from_host(fixture.b)
        eager = lambda: backend.matmul(device_a, device_b)
        first_eval_ns: int | None = None
        compile_setup_ns: int | None = None
        if validated["mode"] == "eager_baseline":
            arms = {"baseline": _run_arm(backend, eager, clock)}
            arms["baseline"]["memory"] = _memory_snapshot(backend, clock, phase="after_measurement", arm="baseline")
            comparison = _single_result(arms["baseline"])
            comparison["aggregation_required"] = False
            comparison["cache_state"] = H0_CACHE_STATE
        else:
            if validated["mode"] == "aa_gpu":
                left = lambda: backend.matmul(device_a, device_b)
                right = lambda: backend.matmul(device_a, device_b)
                order = _balanced_order(validated["seeds"]["order"])
                arms = _run_paired_arms(backend, left, right, clock, order)
                comparison_kind = "aa_gpu_null_control"
            elif validated["mode"] == "compile_comparison":
                compiled, first_eval_ns, compile_setup_ns = _compile_function(backend, fixture.a, fixture.b, clock)
                order = _balanced_order(validated["seeds"]["order"])
                arms = _run_paired_arms(backend, eager, compiled, clock, order, candidate_first_eval_ns=first_eval_ns)
                comparison_kind = "eager_vs_compiled"
            else:
                raise BenchmarkError("mode_not_benchmarkable", f"mode {validated['mode']} is not a benchmark arm")
            comparison = _comparison_result(arms, order, aa=validated["mode"] == "aa_gpu")
            comparison["comparison_kind"] = comparison_kind
        arm_memory = [metric for arm in arms.values() for metric in arm.get("memory", [])]
        memory = memory_before + arm_memory + _memory_snapshot(backend, clock, phase="after_timing", arm="all")
        finished = clock()
        total_ns = _nonnegative_delta(started, finished, "total")
        if total_ns > H0_TOTAL_LIMIT_NS:
            raise BenchmarkError("total_timeout_observed", "process exceeded 120 seconds after returning", evidence={"total_ns": total_ns})
        evidence = {
            "fixture": {"fixture_seed": fixture.fixture_seed, "a_sha256": fixture.a_sha256, "b_sha256": fixture.b_sha256, "metadata_sha256": fixture.metadata_sha256, "fixture_sha256": fixture.fixture_sha256},
            "correctness": correctness,
            "memory": memory,
            "memory_limit": memory_limit,
            "memory_gate": _memory_gate(memory),
            "cache_state": H0_CACHE_STATE,
            "fresh_process_required": True,
            "aggregation_required": bool(comparison.get("aggregation_required", False)),
            "compile_wrapper_setup_ns": compile_setup_ns,
            "first_eval_compile_inclusive_ns": first_eval_ns,
            "total_elapsed_ns": total_ns,
            "arms": arms,
            "comparison": comparison,
            "raw_samples": comparison["raw_samples"],
        }
        domain = "baseline_reference" if validated["mode"] == "eager_baseline" else "measurement_complete"
        action = "not_run" if validated["mode"] == "eager_baseline" else "aggregation_required"
        return _bounded_result(_result_base(validated, digest, status="completed", classification=domain, action=action, error=None, evidence=evidence))
    except RuntimeUnavailable as exc:
        code = _registered_failure_code(exc.code)
        message = str(exc) if code == exc.code else "benchmark failure code is not registered"
        return _bounded_result(_result_base(raw_manifest, digest, status="invalid", classification="runtime_unavailable", action="baseline_fallback", error={"code": code, "message": message}, evidence=exc.evidence))
    except (BenchmarkError, ManifestError, TypeError, ValueError) as exc:
        code = _registered_failure_code(exc.code) if isinstance(exc, BenchmarkError) else "manifest_invalid"
        message = str(exc) if not isinstance(exc, BenchmarkError) or code == exc.code else "benchmark failure code is not registered"
        evidence = exc.evidence if isinstance(exc, BenchmarkError) else {}
        return _bounded_result(_result_base(raw_manifest, digest, status="invalid", classification="invalid: correctness" if code == "correctness_failed" else "invalid", action="baseline_fallback", error={"code": code, "message": message}, evidence=evidence))
    except Exception as exc:  # fail-closed boundary for backend/library faults
        return _bounded_result(_result_base(raw_manifest, digest, status="invalid", classification="invalid", action="baseline_fallback", error={"code": "backend_error", "message": f"{type(exc).__name__}: {exc}"}, evidence={}))


__all__ = [
    "BenchmarkError",
    "RuntimeUnavailable",
    "choose_repetitions",
    "run_mlx_benchmark",
]
