"""Darwin process-memory telemetry through the stable proc_pid_rusage V4 ABI."""

from __future__ import annotations

import ctypes
import platform
import time
from functools import lru_cache
from typing import Any

RUSAGE_INFO_V4 = 4
RUSAGE_INFO_V4_SIZE = 296
_MAX_PID = 2**31 - 1


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


if ctypes.sizeof(_RusageInfoV4) != RUSAGE_INFO_V4_SIZE:
    raise RuntimeError("Darwin rusage_info_v4 ABI size mismatch")


class ProcessMemoryError(RuntimeError):
    """A process-memory sample could not be obtained safely."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@lru_cache(maxsize=1)
def _libproc() -> Any:
    if platform.system() != "Darwin":
        raise ProcessMemoryError("unsupported_platform")
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        function = library.proc_pid_rusage
    except (AttributeError, OSError) as exc:
        raise ProcessMemoryError("libproc_unavailable") from exc
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def _validate_uint64(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ProcessMemoryError(f"{field}_invalid")
    return value


def sample_process_memory(pid: int) -> dict[str, int]:
    """Return one real V4 process-memory sample for an owned Darwin PID."""
    if type(pid) is not int or not 1 <= pid <= _MAX_PID:
        raise ProcessMemoryError("pid_invalid")
    started = time.monotonic_ns()
    function = _libproc()
    usage = _RusageInfoV4()
    try:
        result = function(pid, RUSAGE_INFO_V4, ctypes.byref(usage))
    except (OSError, ctypes.ArgumentError) as exc:
        raise ProcessMemoryError("proc_pid_rusage_failed") from exc
    if result != 0:
        raise ProcessMemoryError("proc_pid_rusage_failed")
    try:
        observed = time.monotonic_ns()
        values = {
            "resident_bytes": usage.ri_resident_size,
            "physical_footprint_bytes": usage.ri_phys_footprint,
            "lifetime_peak_footprint_bytes": usage.ri_lifetime_max_phys_footprint,
            "interval_peak_footprint_bytes": usage.ri_interval_max_phys_footprint,
            "wired_bytes": usage.ri_wired_size,
            "process_start_abstime": usage.ri_proc_start_abstime,
            "observed_monotonic_ns": observed,
            "duration_ns": observed - started,
        }
    except (ValueError, TypeError) as exc:
        raise ProcessMemoryError("proc_pid_rusage_decode_failed") from exc
    return {key: _validate_uint64(value, key) for key, value in values.items()}


__all__ = ["RUSAGE_INFO_V4", "RUSAGE_INFO_V4_SIZE", "ProcessMemoryError", "sample_process_memory"]
