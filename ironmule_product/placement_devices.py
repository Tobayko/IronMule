"""GGUF workers whose measured backend and free memory can accommodate this model."""
from __future__ import annotations

import os
import re
from pathlib import Path

from .device_pool import cuda_devices
from .errors import InvalidRequest


def rocm_bus(row):
    """Normalize HIP's PCI address for both worker verification and the Linux driver sensor."""
    address = str(row.get("pci", "")).lower()
    match = re.fullmatch(r"(?:0000)?([a-f0-9]{4}:[a-f0-9]{2}:[a-f0-9]{2}\.[0-7])", address)
    return match[1] if row["provider"] == "rocm" and match else None


def rocm_selectors(rows):
    """Map the parent's registered physical cards to HIP selectors within its inherited ROCr allocation."""
    mask = os.environ.get("HIP_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    indices = [value.strip() for value in mask.split(",")] if mask is not None else None
    selectors = {}
    for row in rows:
        bus = rocm_bus(row)
        ordinal = re.fullmatch(r"(?:ROCm|CUDA)([0-9]+)", row.get("name", ""))
        if not bus or not ordinal or bus in selectors:
            return {}
        index = int(ordinal[1])
        if indices is not None and (index >= len(indices) or not indices[index].isdecimal()):
            return {}
        selectors[bus] = indices[index] if indices is not None else str(index)
    return selectors if len(set(selectors.values())) == len(selectors) else {}


def verify_rocm_worker(bus):
    from ironmule_devices import ggml_devices

    rows = ggml_devices()
    if len(rows) != 1 or rocm_bus(rows[0]) != bus:
        raise InvalidRequest(f"expected one ROCm worker on PCI device {bus}")


def accelerators_quiet(devices, *, pci_root=Path("/sys/bus/pci/devices")):
    """Return the NVIDIA sensors still to check, or None while a GPU is busy or unobservable."""
    from ironmule_devices import ggml_devices

    cuda = [d for d in devices if d not in ("cpu", "local") and not d.startswith("rocm:")]
    buses = [d.removeprefix("rocm:") for d in devices if d.startswith("rocm:")]
    if "local" in devices:
        rows = ggml_devices()
        providers = {row["provider"] for row in rows}
        if providers == {"cuda"}:
            cuda = cuda_devices()
            if not cuda:
                return None
        elif providers == {"rocm"}:
            buses.extend(rocm_bus(row) for row in rows)
        elif providers:
            return None
    try:
        if any(not bus or not 0 <= int((pci_root / bus / "gpu_busy_percent").read_text()) <= 25 for bus in buses):
            return None
    except (OSError, ValueError):
        return None
    return cuda


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
    elif rows and all(row["provider"] == "rocm" for row in rows):
        selectors = rocm_selectors(rows)
        devices.extend("rocm:" + bus for row in rows
                       if (bus := rocm_bus(row)) in selectors and row["free_memory"] >= needed)
    # A single engine can split layers over visible cards when individual replicas do not fit.
    aggregate = rows[:1] if os.environ.get("IRONMULE_GGML_SPLIT") == "none" else rows
    if rows and not devices and sum(r["free_memory"] for r in aggregate) >= needed:
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
    if device.startswith("rocm:"):
        from ironmule_devices import ggml_devices
        selector = rocm_selectors(ggml_devices()).get(device.removeprefix("rocm:"))
        if selector is None:
            raise InvalidRequest(f"ROCm worker {device} is no longer visible")
        return {"HIP_VISIBLE_DEVICES": selector, "IRONMULE_KG_LIB": "", "IRONMULE_KG5": "0"}
    return {"CUDA_VISIBLE_DEVICES": device}
