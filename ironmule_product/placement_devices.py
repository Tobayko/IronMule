"""GGUF workers whose measured backend and free memory can accommodate this model."""
from __future__ import annotations

import os

from .device_pool import cuda_devices


def candidates(weight_bytes, draft_bytes=0):
    from ironmule.ggml_engine import ggml_devices, gpu_memory_bytes, memory_bytes

    # KV allocation is measured more precisely by the normal per-worker quality/width gates.
    needed = weight_bytes + draft_bytes + max(256 << 20, (weight_bytes + draft_bytes) // 8)
    cpu = memory_bytes()[0] >= needed
    rows = ggml_devices() if os.environ.get("IRONMULE_GGML_GPU_LAYERS") != "0" else ()
    devices = []
    if rows and all(row["provider"] == "cuda" for row in rows):
        for device in cuda_devices():
            memory = gpu_memory_bytes(device)
            if memory and memory[0] >= needed:
                devices.append(device)
    # A single engine can split layers over visible cards when individual replicas do not fit.
    if rows and not devices and sum(r["free_memory"] for r in rows) >= needed:
        devices.append("local")
    if cpu:
        devices.append("cpu")
    return devices


def worker_environment(device):
    if device == "cpu":
        return {"CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                "IRONMULE_GGML_GPU_LAYERS": "0", "IRONMULE_KG_LIB": "", "IRONMULE_KG5": "0"}
    if device == "local":
        return {}
    return {"CUDA_VISIBLE_DEVICES": device}
