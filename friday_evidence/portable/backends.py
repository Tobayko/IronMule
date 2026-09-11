"""Native-only accelerator adapters used by the portable DATA1 runner.

This module deliberately imports no accelerator framework at import time.  The
runner imports it only from its short-lived worker, after the parent has
validated the sealed experiment specification.  None of these adapters has a
CPU fallback: an unavailable or wrong accelerator is a failed measurement,
not a different measurement.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
import platform
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np


class BackendError(RuntimeError):
    """The requested native backend cannot make the requested measurement."""


class UnsupportedCandidate(BackendError):
    """A semantically distinct candidate cannot be formed for this shape."""


class NativeBackend(ABC):
    """Minimal accelerator surface shared by the real-device adapters."""

    name: str

    @abstractmethod
    def prepare_inputs(self, left: np.ndarray, right: np.ndarray) -> tuple[Any, Any]:
        """Copy immutable host operands to this backend's actual device."""

    @abstractmethod
    def native(self, left: Any, right: Any) -> Any:
        """Execute the unmodified native ``A @ B`` reference."""

    @abstractmethod
    def compile_candidate(self, candidate: str, rows: int | None) -> Callable[[Any, Any], Any]:
        """Return a compiled candidate, or raise :class:`UnsupportedCandidate`."""

    @abstractmethod
    def synchronize(self) -> None:
        """Make preceding device work complete before a timing boundary."""

    @abstractmethod
    def to_host(self, value: Any) -> np.ndarray:
        """Return a float32 host representation for exactness inspection."""

    @abstractmethod
    def fingerprint(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return stable hardware and framework/environment identity only."""

    @abstractmethod
    def resources(self) -> dict[str, Any]:
        """Return a live resource sample, or explicit unknown availability."""

    def wait(self, value: Any) -> None:
        """Wait for this specific result, not merely an unrelated stream item."""
        del value
        self.synchronize()

    def _timed(self, operation: Callable[[], Any]) -> tuple[Any, float]:
        self.synchronize()
        started = time.perf_counter()
        value = operation()
        self.wait(value)
        return value, time.perf_counter() - started

    def pair(
        self,
        left: Any,
        right: Any,
        candidate: Callable[[Any, Any], Any],
        order: str,
    ) -> tuple[float, float, bool, float]:
        """Run and byte-check a balanced native/candidate pair.

        Exactness is intentionally checked here, beside the real tensors, for
        every warmup and every recorded sample.  Timing excludes host copies
        used for the correctness comparison, which are reported separately.
        """
        if order not in {"AB", "BA"}:
            raise ValueError("pair order must be AB or BA")
        if order == "AB":
            baseline, baseline_seconds = self._timed(lambda: self.native(left, right))
            trial, candidate_seconds = self._timed(lambda: candidate(left, right))
        else:
            trial, candidate_seconds = self._timed(lambda: candidate(left, right))
            baseline, baseline_seconds = self._timed(lambda: self.native(left, right))
        check_started = time.perf_counter()
        reference = np.ascontiguousarray(self.to_host(baseline), dtype=np.float32)
        observed = np.ascontiguousarray(self.to_host(trial), dtype=np.float32)
        correctness_seconds = time.perf_counter() - check_started
        finite = bool(np.isfinite(reference).all() and np.isfinite(observed).all())
        exact = finite and reference.shape == observed.shape and reference.tobytes() == observed.tobytes()
        return baseline_seconds, candidate_seconds, exact, correctness_seconds


class MLXBackend(NativeBackend):
    name = "mlx"

    def __init__(self) -> None:
        try:
            import mlx.core as mx
        except Exception as exc:  # Import failure is evidence, never a fallback.
            raise BackendError("MLX is unavailable") from exc
        self.mx = mx
        try:
            mx.set_default_device(mx.gpu)
        except Exception as exc:
            raise BackendError("MLX GPU cannot be selected") from exc
        try:
            raw_info = mx.device_info()
        except Exception as exc:
            raise BackendError("MLX device information is unavailable") from exc
        self._hardware = _stable_mlx_hardware(raw_info)
        try:
            self._mlx_version = package_version("mlx")
        except PackageNotFoundError as exc:
            raise BackendError("MLX package version is unavailable") from exc

    def prepare_inputs(self, left: np.ndarray, right: np.ndarray) -> tuple[Any, Any]:
        return self.mx.array(left), self.mx.array(right)

    def native(self, left: Any, right: Any) -> Any:
        return self.mx.matmul(left, right)

    def compile_candidate(self, candidate: str, rows: int | None) -> Callable[[Any, Any], Any]:
        if candidate == "native":
            return self.native
        if candidate not in {"compiled", "rows128", "rows512"}:
            raise UnsupportedCandidate(f"unknown MLX candidate: {candidate}")
        compiled = self.mx.compile(lambda left, right: self.mx.matmul(left, right))
        if rows is None:
            return compiled
        return self._row_partition(compiled, rows)

    def _row_partition(self, compiled: Callable[[Any, Any], Any], rows: int) -> Callable[[Any, Any], Any]:
        def execute(left: Any, right: Any) -> Any:
            count = int(left.shape[0])
            if count <= rows:
                raise UnsupportedCandidate(f"rows{rows} has no distinct partition for {count} rows")
            return self.mx.concatenate([compiled(left[start : start + rows], right) for start in range(0, count, rows)], axis=0)

        return execute

    def synchronize(self) -> None:
        self.mx.eval(self.mx.array(0))
        self.mx.synchronize()

    def wait(self, value: Any) -> None:
        self.mx.eval(value)
        self.mx.synchronize()

    def to_host(self, value: Any) -> np.ndarray:
        self.mx.eval(value)
        return np.asarray(value)

    def fingerprint(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            dict(self._hardware),
            {"framework": "mlx", "framework_version": self._mlx_version},
        )

    def resources(self) -> dict[str, Any]:
        try:
            active = int(self.mx.get_active_memory())
            peak = int(self.mx.get_peak_memory())
            cache = int(self.mx.get_cache_memory())
        except Exception:
            return {"availability": "unknown", "reason": "mlx_resource_api_unavailable"}
        if min(active, peak, cache) < 0:
            return {"availability": "unknown", "reason": "mlx_resource_values_invalid"}
        return {
            "availability": "available", "active_memory_bytes": active,
            "peak_memory_bytes": peak, "cache_memory_bytes": cache,
            "total_memory_bytes": self._hardware["memory"]["total_memory_bytes"],
        }


class CUDABackend(NativeBackend):
    name = "cuda"

    def __init__(self) -> None:
        try:
            import torch
        except Exception as exc:
            raise BackendError("PyTorch is unavailable") from exc
        if not torch.cuda.is_available():
            raise BackendError("CUDA is unavailable; CPU fallback is prohibited")
        self.torch = torch
        self.device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(self.device)
        self.device_name = str(properties.name)
        if "t4" not in self.device_name.lower():
            raise BackendError(f"CUDA device must be NVIDIA T4, found {self.device_name}")
        # DATA1 needs FP32 semantics, not silently-selected tensor-float-32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def prepare_inputs(self, left: np.ndarray, right: np.ndarray) -> tuple[Any, Any]:
        return (
            self.torch.from_numpy(left).to(self.device, dtype=self.torch.float32),
            self.torch.from_numpy(right).to(self.device, dtype=self.torch.float32),
        )

    def native(self, left: Any, right: Any) -> Any:
        return left @ right

    def compile_candidate(self, candidate: str, rows: int | None) -> Callable[[Any, Any], Any]:
        if candidate == "native":
            return self.native
        compiler = getattr(self.torch, "compile", None)
        if not callable(compiler):
            raise UnsupportedCandidate("torch.compile is unavailable")
        compiled = compiler(lambda left, right: left @ right, fullgraph=True)
        if rows is None:
            return compiled
        return self._row_partition(compiled, rows)

    def _row_partition(self, compiled: Callable[[Any, Any], Any], rows: int) -> Callable[[Any, Any], Any]:
        def execute(left: Any, right: Any) -> Any:
            count = int(left.shape[0])
            if count <= rows:
                raise UnsupportedCandidate(f"rows{rows} has no distinct partition for {count} rows")
            return self.torch.cat([compiled(left[start : start + rows], right) for start in range(0, count, rows)], dim=0)

        return execute

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def to_host(self, value: Any) -> np.ndarray:
        return value.detach().to("cpu", dtype=self.torch.float32).contiguous().numpy()

    def fingerprint(self) -> tuple[dict[str, Any], dict[str, Any]]:
        properties = self.torch.cuda.get_device_properties(self.device)
        driver = "unknown"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=False, capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                driver = result.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
        return (
            {
                "backend": self.name,
                "accelerator": self.device_name,
                "device_capability": list(self.torch.cuda.get_device_capability(self.device)),
                "driver": driver,
                "memory": {"total_memory_bytes": int(properties.total_memory)},
            },
            {
                "framework": "torch", "framework_version": str(self.torch.__version__),
                "cuda_runtime": str(self.torch.version.cuda), "tf32_matmul": False,
            },
        )

    def resources(self) -> dict[str, Any]:
        try:
            properties = self.torch.cuda.get_device_properties(self.device)
            allocated = int(self.torch.cuda.memory_allocated(self.device))
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
        except Exception:
            return {"availability": "unknown", "reason": "cuda_resource_api_unavailable"}
        if min(allocated, peak) < 0 or int(properties.total_memory) <= 0:
            return {"availability": "unknown", "reason": "cuda_resource_values_invalid"}
        return {
            "availability": "available", "allocated_memory_bytes": allocated,
            "peak_allocated_memory_bytes": peak,
            "total_memory_bytes": int(properties.total_memory),
        }


class TPUBackend(NativeBackend):
    name = "tpu"

    def __init__(self) -> None:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            raise BackendError("JAX is unavailable") from exc
        devices = [device for device in jax.devices() if device.platform == "tpu"]
        if not devices:
            raise BackendError("JAX TPU is unavailable; CPU fallback is prohibited")
        self.jax = jax
        self.jnp = jnp
        self.device = devices[0]
        self._tpu_devices = devices

    def prepare_inputs(self, left: np.ndarray, right: np.ndarray) -> tuple[Any, Any]:
        return self.jax.device_put(left, self.device), self.jax.device_put(right, self.device)

    def native(self, left: Any, right: Any) -> Any:
        with self.jax.default_matmul_precision("float32"):
            return self.jnp.matmul(left, right, precision=self.jax.lax.Precision.HIGHEST)

    def compile_candidate(self, candidate: str, rows: int | None) -> Callable[[Any, Any], Any]:
        if candidate == "native":
            return self.native

        def matmul(left: Any, right: Any) -> Any:
            with self.jax.default_matmul_precision("float32"):
                return self.jnp.matmul(left, right, precision=self.jax.lax.Precision.HIGHEST)

        compiled = self.jax.jit(matmul, device=self.device)
        if rows is None:
            return compiled
        return self._row_partition(compiled, rows)

    def _row_partition(self, compiled: Callable[[Any, Any], Any], rows: int) -> Callable[[Any, Any], Any]:
        def execute(left: Any, right: Any) -> Any:
            count = int(left.shape[0])
            if count <= rows:
                raise UnsupportedCandidate(f"rows{rows} has no distinct partition for {count} rows")
            return self.jnp.concatenate([compiled(left[start : start + rows], right) for start in range(0, count, rows)], axis=0)

        return execute

    def synchronize(self) -> None:
        # A scalar device_get is a real completion barrier for queued JAX work.
        self.jax.device_get(self.jnp.asarray(0, dtype=self.jnp.int32).block_until_ready())

    def wait(self, value: Any) -> None:
        value.block_until_ready()

    def to_host(self, value: Any) -> np.ndarray:
        return np.asarray(self.jax.device_get(value.block_until_ready()), dtype=np.float32)

    def fingerprint(self) -> tuple[dict[str, Any], dict[str, Any]]:
        details: dict[str, Any] = {
            "backend": self.name, "accelerator": str(getattr(self.device, "device_kind", self.device)),
            "platform": self.device.platform, "topology": {
                "device_count": len(self._tpu_devices),
                "process_count": len({getattr(device, "process_index", 0) for device in self._tpu_devices}),
            },
        }
        client = getattr(self.device, "client", None)
        platform_version = getattr(client, "platform_version", None)
        if isinstance(platform_version, str) and platform_version:
            details["driver"] = platform_version
        return (
            details,
            {
                "framework": "jax", "framework_version": str(self.jax.__version__),
                "jaxlib_version": _jaxlib_version(), "matmul_precision": "float32_highest",
            },
        )

    def resources(self) -> dict[str, Any]:
        stats = getattr(self.device, "memory_stats", None)
        if not callable(stats):
            return {"availability": "unknown", "reason": "tpu_resource_api_unavailable"}
        try:
            raw = stats()
        except Exception:
            return {"availability": "unknown", "reason": "tpu_resource_api_unavailable"}
        if not isinstance(raw, dict):
            return {"availability": "unknown", "reason": "tpu_resource_values_invalid"}
        total = _first_positive_int(raw, "bytes_limit", "total_memory_bytes", "total_bytes", "memory_limit")
        active = _first_nonnegative_int(raw, "bytes_in_use", "allocated_bytes", "bytes_used")
        peak = _first_nonnegative_int(raw, "peak_bytes_in_use", "peak_allocated_bytes", "peak_bytes")
        if total is None or active is None or peak is None:
            return {"availability": "unknown", "reason": "tpu_resource_values_unavailable"}
        return {
            "availability": "available", "active_memory_bytes": active,
            "peak_memory_bytes": peak, "total_memory_bytes": total,
        }


def _jaxlib_version() -> str:
    try:
        import jaxlib
        return str(jaxlib.__version__)
    except Exception:
        return "unknown"


def _first_positive_int(values: dict[Any, Any], *keys: str) -> int | None:
    for key in keys:
        value = values.get(key)
        if type(value) is int and value > 0:
            return value
    return None


def _first_nonnegative_int(values: dict[Any, Any], *keys: str) -> int | None:
    for key in keys:
        value = values.get(key)
        if type(value) is int and value >= 0:
            return value
    return None


def _stable_mlx_hardware(raw_info: Any) -> dict[str, Any]:
    """Build a frozen MLX identity without allocation/counter state."""
    if not isinstance(raw_info, dict):
        raise BackendError("MLX device information is invalid")
    chip = next((raw_info.get(key) for key in ("chip_name", "device_name", "name", "model") if isinstance(raw_info.get(key), str) and raw_info.get(key).strip()), None)
    total = _first_positive_int(raw_info, "total_memory_bytes", "memory_size", "memory_bytes", "recommended_max_memory")
    if chip is None or total is None:
        raise BackendError("MLX concrete chip identity or total memory is unavailable")
    return {
        "backend": "mlx", "accelerator": chip.strip(), "driver": platform.release(),
        "memory": {"total_memory_bytes": total},
    }


def make_backend(name: str) -> NativeBackend:
    """Instantiate a real-device adapter without any fallback behavior."""
    adapters: dict[str, type[NativeBackend]] = {"mlx": MLXBackend, "cuda": CUDABackend, "tpu": TPUBackend}
    try:
        return adapters[name]()
    except KeyError as exc:
        raise BackendError(f"unknown backend: {name}") from exc


def candidate_rows(candidate: str) -> int | None:
    """Map the sealed candidate identifiers to their row partition, if any."""
    return {"native": None, "compiled": None, "rows128": 128, "rows512": 512}.get(candidate)


__all__ = [
    "BackendError", "CUDABackend", "MLXBackend", "NativeBackend", "TPUBackend",
    "UnsupportedCandidate", "candidate_rows", "make_backend",
]
