"""Bounded process-memory and swap guard for local calibration workers."""

from __future__ import annotations

import math
import os
import subprocess
import time
from typing import Any, Mapping

from .readiness import _swap_used


SWAP_DELTA_LIMIT_BYTES = 256 * 1024 * 1024
RSS_LIMIT_FRACTION = 0.60
POLL_INTERVAL_SECONDS = 0.25
_MAX_BYTES = 2**63 - 1
_READY_KEYS = frozenset({
    "startup_wall_seconds", "process_peak_rss_bytes", "mlx_active_bytes",
    "mlx_peak_bytes", "mlx_cache_bytes", "recommended_working_set_bytes",
})


class MemoryGuardError(RuntimeError):
    """A memory or swap safety condition cannot be proven."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _rss_bytes(pid: int) -> int:
    """Read one owned PID's RSS from the real macOS process table."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "pid=,uid=,rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MemoryGuardError("rss_probe_failed") from exc
    if result.returncode != 0:
        raise MemoryGuardError("rss_probe_failed")
    rows = [line.split() for line in result.stdout.splitlines() if line.split()]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise MemoryGuardError("rss_probe_invalid")
    try:
        row_pid, owner, rss_kib = (int(value, 10) for value in rows[0])
    except (TypeError, ValueError):
        raise MemoryGuardError("rss_probe_invalid") from None
    if row_pid != pid or owner != os.geteuid() or rss_kib < 0:
        raise MemoryGuardError("rss_probe_invalid")
    return rss_kib * 1024


def _finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value)) and value >= 0
    except (OverflowError, ValueError):
        return False


class LoadMemoryGuard:
    """Poll a worker's RSS and swap growth against fixed safety ceilings."""

    def __init__(self, *, swap_baseline_bytes: int, memory_total_bytes: int, on_sample=None):
        if type(swap_baseline_bytes) is not int or not 0 <= swap_baseline_bytes <= _MAX_BYTES:
            raise ValueError("swap_baseline_bytes must be a non-negative integer")
        if type(memory_total_bytes) is not int or not 0 < memory_total_bytes <= _MAX_BYTES:
            raise ValueError("memory_total_bytes must be a positive integer")
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable or None")
        self.swap_baseline_bytes = swap_baseline_bytes
        self.memory_total_bytes = memory_total_bytes
        self.on_sample = on_sample
        self.samples: list[dict[str, Any]] = []
        self._last_poll_monotonic = 0.0
        self._last_sample: dict[str, Any] | None = None
        self._bound_pid: int | None = None
        self._failure_code: str | None = None

    def _record(self, sample: dict[str, Any]) -> None:
        self.samples.append(dict(sample))
        if self.on_sample is not None:
            self.on_sample(dict(sample))

    def _fail(self, code: str, sample: dict[str, Any]) -> None:
        self._failure_code = code
        if code not in sample["errors"]:
            sample["errors"].append(code)
        self._record(sample)
        raise MemoryGuardError(code)

    def __call__(self, pid: int, force: bool = False) -> dict[str, Any]:
        if type(pid) is not int or pid <= 0:
            raise MemoryGuardError("pid_invalid")
        if type(force) is not bool:
            raise MemoryGuardError("force_invalid")
        if self._bound_pid is None:
            self._bound_pid = pid
        elif self._bound_pid != pid:
            raise MemoryGuardError("pid_mismatch")
        if self._failure_code is not None:
            raise MemoryGuardError(self._failure_code)
        now = time.monotonic()
        if not force and self._last_sample is not None and now - self._last_poll_monotonic < POLL_INTERVAL_SECONDS:
            return dict(self._last_sample)
        started = time.time_ns()
        started_monotonic = time.monotonic_ns()
        errors: list[str] = []
        rss: int | None = None
        swap: int | None = None
        try:
            rss = _rss_bytes(pid)
        except MemoryGuardError as exc:
            errors.append(exc.code)
        try:
            swap, error = _swap_used()
            if error is not None:
                errors.append(error)
            if type(swap) is not int or not 0 <= swap <= _MAX_BYTES:
                swap = None
                if "swap_probe_invalid" not in errors:
                    errors.append("swap_probe_invalid")
        except Exception:
            errors.append("swap_probe_failed")
        finished = time.time_ns()
        sample: dict[str, Any] = {
            "started_unix_ns": started,
            "finished_unix_ns": finished,
            "started_monotonic_ns": started_monotonic,
            "finished_monotonic_ns": time.monotonic_ns(),
            "pid": pid,
            "rss_bytes": rss,
            "swap_used_bytes": swap,
            "swap_delta_bytes": None if swap is None else swap - self.swap_baseline_bytes,
            "errors": sorted(set(errors)),
        }
        self._last_poll_monotonic = now
        self._last_sample = dict(sample)
        if errors:
            self._fail(errors[0], sample)
        assert rss is not None and swap is not None
        rss_limit = int(self.memory_total_bytes * RSS_LIMIT_FRACTION)
        swap_delta = swap - self.swap_baseline_bytes
        sample["rss_limit_bytes"] = rss_limit
        sample["swap_delta_bytes"] = swap_delta
        if swap_delta > SWAP_DELTA_LIMIT_BYTES:
            self._fail("swap_delta_exceeded", sample)
        if rss > rss_limit:
            self._fail("rss_limit_exceeded", sample)
        self._record(sample)
        self._last_sample = dict(sample)
        return dict(sample)

    def validate_ready(self, ready: Mapping[str, Any]) -> dict[str, Any]:
        if self._failure_code is not None:
            raise MemoryGuardError(self._failure_code)
        if not isinstance(ready, Mapping) or not _READY_KEYS.issubset(ready):
            self._failure_code = "ready_telemetry_invalid"
            raise MemoryGuardError(self._failure_code)
        startup = ready["startup_wall_seconds"]
        if not _finite_nonnegative(startup) or startup <= 0 or startup > 86400:
            self._failure_code = "startup_wall_invalid"
            raise MemoryGuardError(self._failure_code)
        for key in (
            "process_peak_rss_bytes", "mlx_active_bytes", "mlx_peak_bytes",
            "mlx_cache_bytes", "recommended_working_set_bytes",
        ):
            value = ready[key]
            if type(value) is not int or not 0 <= value <= _MAX_BYTES:
                self._failure_code = f"{key}_invalid"
                raise MemoryGuardError(self._failure_code)
        rss = ready["process_peak_rss_bytes"]
        recommended = ready["recommended_working_set_bytes"]
        if rss <= 0:
            self._failure_code = "process_peak_rss_bytes_invalid"
            raise MemoryGuardError(self._failure_code)
        if recommended <= 0:
            self._failure_code = "recommended_working_set_bytes_invalid"
            raise MemoryGuardError(self._failure_code)
        rss_limit = int(self.memory_total_bytes * RSS_LIMIT_FRACTION)
        if rss > rss_limit:
            self._failure_code = "rss_peak_limit_exceeded"
            raise MemoryGuardError(self._failure_code)
        if ready["mlx_active_bytes"] > ready["mlx_peak_bytes"]:
            self._failure_code = "mlx_active_exceeds_peak"
            raise MemoryGuardError(self._failure_code)
        if ready["mlx_peak_bytes"] > rss_limit:
            self._failure_code = "mlx_peak_limit_exceeded"
            raise MemoryGuardError(self._failure_code)
        if ready["mlx_peak_bytes"] > recommended:
            self._failure_code = "mlx_peak_working_set_exceeded"
            raise MemoryGuardError(self._failure_code)
        return {
            **{key: ready[key] for key in _READY_KEYS},
            "rss_limit_bytes": rss_limit,
            "swap_delta_limit_bytes": SWAP_DELTA_LIMIT_BYTES,
        }


__all__ = [
    "LoadMemoryGuard", "MemoryGuardError", "POLL_INTERVAL_SECONDS",
    "RSS_LIMIT_FRACTION", "SWAP_DELTA_LIMIT_BYTES",
]
