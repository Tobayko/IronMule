"""Hardware and OS environment tuning for Apple Silicon M1 Max.

Optimizes the operating system execution environment without altering model weights:
1. Thread QoS Pinning: Sets Mach thread QoS to QOS_CLASS_USER_INTERACTIVE (0x21),
   ensuring the kernel scheduler pins inference loops to Firestorm Performance Cores (P-Cores).
2. Metal Cache Sizing: Configures MLX Metal allocation cache to retain intermediate buffers,
   eliminating Mach VM allocation churn.
3. Wired Memory Clamping: Sets physical wired memory limit up to 70% of Unified Memory (24 GB),
   preventing the macOS dynamic_pager from compressing or swapping active model pages.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any

import mlx.core as mx


def set_performance_qos() -> bool:
    """Set current thread QoS class to QOS_CLASS_USER_INTERACTIVE (0x21) on macOS."""
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(None)
        # QOS_CLASS_USER_INTERACTIVE = 0x21, relative_priority = 0
        res = libc.pthread_set_qos_class_self_np(0x21, 0)
        return res == 0
    except Exception:
        return False


def tune_runtime_environment(uma_gb: float = 34.0) -> dict[str, Any]:
    """Apply all hardware, OS, and Metal environment tunings."""
    qos_ok = set_performance_qos()

    # Metal allocation cache: 50% of UMA to avoid buffer reallocation syscalls
    cache_bytes = int(0.5 * uma_gb * (1024**3))
    # Wired memory limit: 70% of UMA to prevent swapping
    wired_bytes = int(0.7 * uma_gb * (1024**3))

    try:
        set_c = getattr(mx, "set_cache_limit", None) or getattr(getattr(mx, "metal", None), "set_cache_limit", None)
        if set_c:
            set_c(cache_bytes)
        set_w = getattr(mx, "set_wired_limit", None) or getattr(getattr(mx, "metal", None), "set_wired_limit", None)
        if set_w:
            set_w(wired_bytes)
        dev_fn = getattr(mx, "device_info", None) or getattr(getattr(mx, "metal", None), "device_info", None)
        if dev_fn:
            metal_info = dev_fn()
    except Exception as exc:
        metal_info = {"error": str(exc)}

    return {
        "qos_interactive": qos_ok,
        "metal_cache_limit_gb": cache_bytes / (1024**3),
        "metal_wired_limit_gb": wired_bytes / (1024**3),
        "device_name": metal_info.get("device_name", "Apple Silicon"),
        "max_recommended_working_set_gb": metal_info.get("max_recommended_working_set_size", 0) / (1024**3),
    }


__all__ = ["set_performance_qos", "tune_runtime_environment"]
