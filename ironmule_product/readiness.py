"""Portable, read-only host readiness probes for the local product.

The probe is deliberately independent of MLX and model loading.  It records a
small numeric snapshot and typed error codes; command output, process names and
filesystem paths never leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
import json
import math
import os
import platform
import re
import subprocess
import time
from typing import Any, Mapping


_LOADAVG_FIELDS = ("load_1m", "load_5m", "load_15m")
_SWAP_USED_RE = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)", re.IGNORECASE)
_FREE_PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_SWAP_MULTIPLIERS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_POWER_SOURCES = frozenset(("ac", "battery", "unknown"))
_HARDWARE_OUTPUT_LIMIT = 256 * 1024
_HARDWARE_CACHE: dict[str, Any] | None = None


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _foundation_state() -> tuple[bool | None, int | None, str | None]:
    """Read public NSProcessInfo flags through the Objective-C runtime."""

    if platform.system() != "Darwin":
        return None, None, "foundation_unsupported_platform"
    try:
        objc_name = ctypes.util.find_library("objc")
        foundation_name = ctypes.util.find_library("Foundation")
        if not objc_name or not foundation_name:
            return None, None, "foundation_unavailable"
        ctypes.CDLL(foundation_name)
        objc = ctypes.CDLL(objc_name)
        objc_get_class = objc.objc_getClass
        objc_get_class.argtypes = [ctypes.c_char_p]
        objc_get_class.restype = ctypes.c_void_p
        sel_register_name = objc.sel_registerName
        sel_register_name.argtypes = [ctypes.c_char_p]
        sel_register_name.restype = ctypes.c_void_p
        msg_send = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
        process_info = msg_send(objc_get_class(b"NSProcessInfo"), sel_register_name(b"processInfo"))
        if not process_info:
            return None, None, "foundation_process_info_unavailable"
        responds_to_selector = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        responds_selector = sel_register_name(b"respondsToSelector:")
        low_selector = sel_register_name(b"isLowPowerModeEnabled")
        thermal_selector = sel_register_name(b"thermalState")
        if not responds_to_selector(process_info, responds_selector, low_selector):
            return None, None, "foundation_low_power_selector_unavailable"
        if not responds_to_selector(process_info, responds_selector, thermal_selector):
            return None, None, "foundation_thermal_selector_unavailable"
        low_power_send = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p))
        thermal_send = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p))
        low_power = bool(low_power_send(process_info, low_selector))
        thermal = int(thermal_send(process_info, thermal_selector))
        if thermal not in (0, 1, 2, 3):
            return low_power, None, "foundation_invalid_thermal_state"
        return low_power, thermal, None
    except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError):
        return None, None, "foundation_probe_failed"


def _command(args: list[str], error_code: str, *, timeout: float = 3.0) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None, error_code
    if result.returncode != 0:
        return None, error_code
    return result.stdout, None


def _hardware_command(args: list[str], error_code: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(args, capture_output=True, text=False, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None, error_code
    if result.returncode != 0:
        return None, error_code
    if len(result.stdout) > _HARDWARE_OUTPUT_LIMIT:
        return None, "hardware_output_too_large"
    try:
        return result.stdout.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "hardware_output_invalid_utf8"


def _safe_technical_string(value: Any, *, limit: int = 256) -> str | None:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        return None
    return value


def hardware_identity() -> dict[str, Any]:
    """Return cached stable chip/GPU identity metadata, queried once per process."""

    global _HARDWARE_CACHE
    if _HARDWARE_CACHE is not None:
        return {"chip_name": _HARDWARE_CACHE["chip_name"], "gpu_devices": [dict(item) for item in _HARDWARE_CACHE["gpu_devices"]], "hardware_errors": list(_HARDWARE_CACHE["hardware_errors"])}
    errors: list[str] = []
    supported = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    if not supported:
        _HARDWARE_CACHE = {"chip_name": None, "gpu_devices": [], "hardware_errors": ["hardware_unsupported_platform"]}
        return {"chip_name": None, "gpu_devices": [], "hardware_errors": ["hardware_unsupported_platform"]}

    chip_output, error = _hardware_command(["sysctl", "-n", "machdep.cpu.brand_string"], "chip_name_probe_failed")
    chip_name = _safe_technical_string(chip_output.strip() if chip_output is not None else None)
    if chip_name is None:
        errors.append(error or "chip_name_invalid")

    display_output, error = _hardware_command(["system_profiler", "SPDisplaysDataType", "-json"], "gpu_probe_failed")
    devices: list[dict[str, Any]] = []
    if error or display_output is None:
        errors.append(error or "gpu_probe_failed")
    else:
        try:
            payload = json.loads(display_output)
        except (TypeError, ValueError):
            payload = None
            errors.append("gpu_output_invalid_json")
        rows = payload.get("SPDisplaysDataType") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            errors.append("gpu_devices_unavailable")
        else:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                # Only the GPU model field is admitted; ``_name`` is also used
                # for attached displays in this payload and is never trusted.
                model = _safe_technical_string(row.get("sppci_model"))
                raw_cores = row.get("sppci_cores")
                try:
                    cores = int(raw_cores) if isinstance(raw_cores, (str, int)) and not isinstance(raw_cores, bool) else None
                except (TypeError, ValueError):
                    cores = None
                if cores is not None and cores <= 0:
                    cores = None
                metal = _safe_technical_string(row.get("spdisplays_metal") or row.get("spdisplays_mtlgpufamilysupport"), limit=128)
                devices.append({"model": model, "cores": cores, "metal_support": metal})
            if not devices:
                errors.append("gpu_devices_unavailable")
            if any(device["model"] is None for device in devices):
                errors.append("gpu_model_unavailable")
            if any(device["cores"] is None for device in devices):
                errors.append("gpu_core_count_unavailable")

    _HARDWARE_CACHE = {"chip_name": chip_name, "gpu_devices": devices, "hardware_errors": sorted(set(errors))}
    return {"chip_name": chip_name, "gpu_devices": [dict(item) for item in devices], "hardware_errors": sorted(set(errors))}


def _power_source() -> tuple[str, str | None]:
    output, error = _command(["pmset", "-g", "ps"], "power_probe_failed")
    if error:
        return "unknown", error
    assert output is not None
    lowered = output.lower()
    if "ac power" in lowered or "wall power" in lowered:
        return "ac", None
    if "battery power" in lowered:
        return "battery", None
    return "unknown", "power_source_unknown"


def _sysctl_value(name: str, error_code: str) -> tuple[str | None, str | None]:
    return _command(["sysctl", "-n", name], error_code)


def _memory_total() -> tuple[int | None, str | None]:
    output, error = _sysctl_value("hw.memsize", "memory_total_probe_failed")
    if error or output is None:
        return None, error
    value = output.strip()
    if not value.isdigit():
        return None, "memory_total_invalid"
    parsed = int(value, 10)
    return (parsed, None) if parsed > 0 else (None, "memory_total_invalid")


def _cpu_count() -> tuple[int | None, str | None]:
    output, error = _sysctl_value("hw.logicalcpu", "cpu_count_probe_failed")
    if error or output is None:
        return None, error
    value = output.strip()
    if not value.isdigit() or int(value, 10) <= 0:
        return None, "cpu_count_invalid"
    return int(value, 10), None


def _swap_used() -> tuple[int | None, str | None]:
    output, error = _sysctl_value("vm.swapusage", "swap_probe_failed")
    if error or output is None:
        return None, error
    match = _SWAP_USED_RE.search(output)
    if match is None:
        return None, "swap_used_invalid"
    amount = float(match.group(1))
    if not math.isfinite(amount) or amount < 0:
        return None, "swap_used_invalid"
    return int(amount * _SWAP_MULTIPLIERS[match.group(2).upper()]), None


def _memory_free_percent() -> tuple[float | None, str | None]:
    output, error = _command(["memory_pressure", "-Q"], "memory_free_probe_failed")
    if error or output is None:
        return None, error
    match = _FREE_PERCENT_RE.search(output)
    if match is None:
        return None, "memory_free_invalid"
    value = float(match.group(1))
    if not math.isfinite(value) or not 0 <= value <= 100:
        return None, "memory_free_invalid"
    return value, None


def probe() -> dict[str, Any]:
    """Capture a bounded read-only readiness snapshot."""

    started = time.monotonic()
    observed_unix_ns = time.time_ns()
    observed_monotonic = started
    errors: list[str] = []
    supported = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}

    power_source, error = _power_source()
    if error:
        errors.append(error)
    low_power_mode, thermal_state, error = _foundation_state()
    if error:
        errors.append(error)
    cpu_count, error = _cpu_count()
    if error:
        errors.append(error)
    memory_total_bytes, error = _memory_total()
    if error:
        errors.append(error)
    swap_used_bytes, error = _swap_used()
    if error:
        errors.append(error)
    memory_free_percent, error = _memory_free_percent()
    if error:
        errors.append(error)
    try:
        load_values = os.getloadavg()
    except (AttributeError, OSError):
        load_values = ()
        errors.append("load_probe_failed")
    load_1m = float(load_values[0]) if len(load_values) >= 1 and _finite_number(load_values[0]) else None
    if load_1m is None and "load_probe_failed" not in errors:
        errors.append("load_invalid")
    if cpu_count is not None and load_1m is not None:
        load_ratio = load_1m / cpu_count
    else:
        load_ratio = None
        errors.append("load_ratio_unavailable")

    return {
        "observed_unix_ns": observed_unix_ns,
        "observed_monotonic": observed_monotonic,
        "duration_seconds": max(0.0, time.monotonic() - started),
        "platform_supported": supported,
        "power_source": power_source if supported else "unknown",
        "low_power_mode": low_power_mode if supported else None,
        "thermal_state": thermal_state if supported else None,
        "cpu_count": cpu_count,
        "load_1m": load_1m,
        "load_ratio": load_ratio,
        "memory_total_bytes": memory_total_bytes,
        "swap_used_bytes": swap_used_bytes,
        "memory_free_percent": memory_free_percent,
        "errors": sorted(set(errors)),
    }


@dataclass(frozen=True)
class ReadinessPolicy:
    max_load_ratio: float = 0.8
    allowed_thermal_states: tuple[int, ...] = (0, 1)
    min_free_percent: float = 10.0
    stable_samples: int = 3
    sample_interval_s: float = 5.0
    max_gap_s: float = 15.0
    max_sample_age_s: float = 10.0

    def __post_init__(self) -> None:
        if not _finite_number(self.max_load_ratio) or self.max_load_ratio < 0:
            raise ValueError("max_load_ratio must be finite and non-negative")
        if type(self.stable_samples) is not int or self.stable_samples < 1:
            raise ValueError("stable_samples must be a positive integer")
        for name in ("min_free_percent", "sample_interval_s", "max_gap_s", "max_sample_age_s"):
            value = getattr(self, name)
            if not _finite_number(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.sample_interval_s <= 0 or self.max_sample_age_s <= 0 or self.max_gap_s <= 0:
            raise ValueError("sample interval, maximum gap, and sample age must be positive")
        if not self.allowed_thermal_states or any(type(state) is not int or state not in (0, 1, 2, 3) for state in self.allowed_thermal_states):
            raise ValueError("allowed_thermal_states must contain valid thermal states")
        if self.max_gap_s < self.sample_interval_s:
            raise ValueError("max_gap_s must be at least sample_interval_s")
        if self.min_free_percent > 100:
            raise ValueError("min_free_percent must be at most 100")


def _invalid_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    required = {
        "observed_unix_ns", "observed_monotonic", "duration_seconds", "platform_supported", "power_source", "low_power_mode", "thermal_state",
        "cpu_count", "load_1m", "load_ratio", "memory_total_bytes", "swap_used_bytes", "memory_free_percent", "errors",
    }
    for field in sorted(required - set(snapshot)):
        reasons.append(f"missing_{field}")
    unix_ns = snapshot.get("observed_unix_ns")
    if type(unix_ns) is not int or unix_ns <= 0:
        reasons.append("invalid_observed_unix_ns")
    monotonic = snapshot.get("observed_monotonic")
    if not _finite_number(monotonic):
        reasons.append("invalid_observed_monotonic")
    duration = snapshot.get("duration_seconds")
    if not _finite_number(duration) or float(duration) < 0:
        reasons.append("invalid_duration_seconds")
    if type(snapshot.get("platform_supported")) is not bool:
        reasons.append("invalid_platform_supported")
    power_source = snapshot.get("power_source")
    if not isinstance(power_source, str) or power_source not in _POWER_SOURCES:
        reasons.append("invalid_power_source")
    if snapshot.get("low_power_mode") is not None and type(snapshot.get("low_power_mode")) is not bool:
        reasons.append("invalid_low_power_mode")
    thermal = snapshot.get("thermal_state")
    if thermal is not None and (type(thermal) is not int or thermal not in (0, 1, 2, 3)):
        reasons.append("invalid_thermal_state")
    for field in ("cpu_count", "memory_total_bytes"):
        value = snapshot.get(field)
        if type(value) is not int or value <= 0:
            reasons.append(f"invalid_{field}")
    swap = snapshot.get("swap_used_bytes")
    if type(swap) is not int or swap < 0:
        reasons.append("invalid_swap_used_bytes")
    for field in ("load_1m", "load_ratio", "memory_free_percent"):
        value = snapshot.get(field)
        if not _finite_number(value):
            reasons.append(f"invalid_{field}")
    if _finite_number(snapshot.get("load_1m")) and float(snapshot["load_1m"]) < 0:
        reasons.append("invalid_load_1m")
    if _finite_number(snapshot.get("load_ratio")) and float(snapshot["load_ratio"]) < 0:
        reasons.append("invalid_load_ratio")
    if _finite_number(snapshot.get("load_1m")) and type(snapshot.get("cpu_count")) is int and snapshot["cpu_count"] > 0 and _finite_number(snapshot.get("load_ratio")):
        expected_ratio = float(snapshot["load_1m"]) / snapshot["cpu_count"]
        if not math.isclose(float(snapshot["load_ratio"]), expected_ratio, rel_tol=1e-6, abs_tol=1e-9):
            reasons.append("load_ratio_inconsistent")
    if _finite_number(snapshot.get("memory_free_percent")) and not 0 <= float(snapshot["memory_free_percent"]) <= 100:
        reasons.append("invalid_memory_free_percent")
    return sorted(set(reasons))


def evaluate(snapshot: Mapping[str, Any], policy: ReadinessPolicy, now_monotonic: float | None = None) -> dict[str, Any]:
    """Evaluate one snapshot without probing or touching the operating system."""

    if not isinstance(snapshot, Mapping):
        return {"eligible": False, "reasons": ["invalid_snapshot"]}
    now = time.monotonic() if now_monotonic is None else now_monotonic
    reasons = _invalid_snapshot(snapshot)
    if not _finite_number(now):
        reasons.append("invalid_now_monotonic")
    observed = snapshot.get("observed_monotonic")
    if _finite_number(now) and _finite_number(observed):
        age = float(now) - float(observed)
        if age < 0:
            reasons.append("timestamp_in_future")
        elif age > policy.max_sample_age_s:
            reasons.append("sample_too_old")
    if snapshot.get("platform_supported") is not True:
        reasons.append("platform_unsupported")
    if snapshot.get("power_source") != "ac":
        reasons.append("power_not_ac")
    if snapshot.get("low_power_mode") is not False:
        reasons.append("low_power_mode_unavailable_or_enabled")
    if snapshot.get("thermal_state") not in policy.allowed_thermal_states:
        reasons.append("thermal_state_not_allowed")
    if _finite_number(snapshot.get("load_ratio")) and float(snapshot["load_ratio"]) > policy.max_load_ratio:
        reasons.append("load_ratio_too_high")
    if _finite_number(snapshot.get("memory_free_percent")) and float(snapshot["memory_free_percent"]) < policy.min_free_percent:
        reasons.append("memory_free_percent_too_low")
    errors = snapshot.get("errors")
    if not isinstance(errors, list) or any(not isinstance(error, str) or not error for error in errors):
        reasons.append("invalid_errors")
    elif errors:
        reasons.append("probe_errors_present")
    return {"eligible": not reasons, "reasons": sorted(set(reasons))}


class ReadinessWindow:
    """Require spaced, fresh eligible samples before declaring stability."""

    def __init__(self, policy: ReadinessPolicy | None = None):
        self.policy = policy or ReadinessPolicy()
        self._last_observed: float | None = None
        self._consecutive = 0

    def observe(self, snapshot: Mapping[str, Any], now_monotonic: float | None = None) -> dict[str, Any]:
        decision = evaluate(snapshot, self.policy, now_monotonic)
        observed = snapshot.get("observed_monotonic") if isinstance(snapshot, Mapping) else None
        if not decision["eligible"] or not _finite_number(observed):
            self._last_observed = None
            self._consecutive = 0
            return {**decision, "stable": False, "consecutive": 0}
        timestamp = float(observed)
        if self._last_observed is None:
            self._last_observed = timestamp
            self._consecutive = 1
        else:
            gap = timestamp - self._last_observed
            if gap == 0:
                return {**decision, "stable": False, "consecutive": self._consecutive, "reasons": sorted(set(decision["reasons"] + ["duplicate_sample"]))}
            if gap < 0:
                self._last_observed = None
                self._consecutive = 0
                return {"eligible": False, "reasons": ["out_of_order_sample"], "stable": False, "consecutive": 0}
            if gap < self.policy.sample_interval_s:
                return {**decision, "stable": False, "consecutive": self._consecutive, "reasons": sorted(set(decision["reasons"] + ["sample_interval_too_short"]))}
            if gap > self.policy.max_gap_s:
                self._consecutive = 1
                self._last_observed = timestamp
                return {**decision, "stable": False, "consecutive": 1, "reasons": sorted(set(decision["reasons"] + ["sampling_gap_too_large"]))}
            self._consecutive += 1
            self._last_observed = timestamp
        return {**decision, "stable": self._consecutive >= self.policy.stable_samples, "consecutive": self._consecutive}


__all__ = ["ReadinessPolicy", "ReadinessWindow", "evaluate", "hardware_identity", "probe"]
