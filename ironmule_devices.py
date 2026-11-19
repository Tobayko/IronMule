"""Dependency-light host accelerator inventory and pinned GGML device discovery."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import re
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any


def bindings() -> Any:
    """`llama_cpp`, from an install or from `IRONMULE_GGML_BINDINGS` (an unpacked `llama-cpp-python` source)."""
    path = os.environ.get("IRONMULE_GGML_BINDINGS")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import diskcache  # noqa: F401 - the bindings import it for a cache class this engine never uses
    except ImportError:
        sys.modules.setdefault("diskcache", types.SimpleNamespace(Cache=object))
    import llama_cpp

    return llama_cpp


VISIBLE = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
           "GPU_DEVICE_ORDINAL", "GGML_VK_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK")
# PCI IDs published by Google's tpu-info (tpu_info/device.py). VFIO may own a
# cloud TPU even when no TPU-specific kernel driver is installed in the guest.
TPU_MODELS = {"0x005e": "v4", "0x0063": "v5e", "0x0062": "v5p", "0x006f": "v6e", "0x0076": "v7x"}
PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def host_accelerators(sysfs: Path = Path("/sys")) -> list[dict]:
    """Passive observations. A PCI device or a TPU allocation hint does not prove runtime support."""
    devices = []
    for path in sorted((sysfs / "bus/pci/devices").glob("*")):
        vendor = _read(path / "vendor")
        if vendor not in PCI_VENDORS and vendor not in ("0x1ae0", "0x1ac1"):
            continue
        device = _read(path / "device")
        version = TPU_MODELS.get(device) if vendor == "0x1ae0" else None
        if vendor == "0x1ae0" and device == "0x0027":
            version = {"0x004e": "v2", "0x004f": "v3"}.get(_read(path / "subsystem_device"))
        if version:
            kind = "tpu"
        elif vendor == "0x1ac1" and device == "0x089a":
            kind = "edge-tpu"
        elif vendor in PCI_VENDORS and _read(path / "class")[:4] in ("0x03", "0x12"):
            kind = PCI_VENDORS[vendor]
        else:
            continue
        driver = (path / "driver").resolve().name if (path / "driver").is_symlink() else ""
        devices.append({"vendor": kind, "pci": path.name, "device": device,
                        "driver": driver, "driver_version": _read(path / "driver/module/version"),
                        "unique_id": _read(path / "unique_id"), "source": "pci"})
        if version:
            devices[-1]["model"] = version
    if (os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
            or any(os.environ.get(k) for k in ("TPU_NAME", "TPU_ACCELERATOR_TYPE", "COLAB_TPU_ADDR", "TPU_WORKER_ID"))):
        devices.append({"vendor": "tpu", "source": "allocation-environment"})
    return devices


@lru_cache(maxsize=1)
def _nvidia_identity():
    # The registry's CUDA0 is a visible index and may be remapped across workers;
    # bus IDs and UUIDs keep physical boards and MIG allocation masks distinct.
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,uuid,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=3, check=True)
        return sorted(result.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        return []


def runtime_binding(devices: tuple[dict, ...]) -> dict:
    stable = [{k: v for k, v in row.items() if k != "free_memory"} for row in devices]
    host = host_accelerators()
    nvidia = any(row["provider"] == "cuda" for row in stable) or any(row["vendor"] == "nvidia" for row in host)
    host_id = _read(Path("/etc/machine-id")) if devices else ""
    return {"host_id_sha256": hashlib.sha256(host_id.encode()).hexdigest() if host_id else None,
            "devices": stable, "host": host, "nvidia_devices": _nvidia_identity() if nvidia else [],
            "visibility": {k: os.environ.get(k) for k in VISIBLE},
            "nvidia_driver": _read(Path("/proc/driver/nvidia/version")),
            "ggml_gpu_layers": os.environ.get("IRONMULE_GGML_GPU_LAYERS"),
            "ggml_split": os.environ.get("IRONMULE_GGML_SPLIT", "layer")}


def _function(lib, name, returns, *params):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = list(params), returns
    return fn


def _driver_details(lib, provider: str, name: str) -> dict:
    """Query only the API of the device GGML registered. Both APIs use visible ordinals."""
    if provider not in ("cuda", "rocm"):
        return {}
    api = "cuda" if provider == "cuda" else "hip"
    details = {}
    for key, suffix in (("driver", "DriverGetVersion"), ("runtime", "RuntimeGetVersion")):
        if hasattr(lib, api + suffix):
            version = C.c_int()
            fn = _function(lib, api + suffix, C.c_int, C.POINTER(C.c_int))
            if fn(C.byref(version)) == 0:
                details[key] = version.value
    index = re.fullmatch(r"(?:CUDA|ROCm)([0-9]+)", name)
    if index and hasattr(lib, api + "DeviceGetPCIBusId"):
        address = C.create_string_buffer(32)
        pci = _function(lib, api + "DeviceGetPCIBusId", C.c_int, C.c_char_p, C.c_int, C.c_int)
        if pci(address, len(address), int(index.group(1))) == 0:
            details["pci"] = address.value.decode(errors="replace")
    if provider == "rocm" and index and hasattr(lib, "hipDeviceGetUuid"):
        uuid = (C.c_ubyte * 16)()
        get_uuid = _function(lib, "hipDeviceGetUuid", C.c_int, C.c_void_p, C.c_int)
        if get_uuid(uuid, int(index.group(1))) == 0 and any(uuid):
            details["uuid"] = bytes(uuid).hex()
    return details


def _devices(L):
    # Use the handle of the loaded llama.cpp. A second, independently loaded
    # registry may enumerate different devices from those that will serve.
    L.llama_backend_init()
    lib = L.llama_cpp._lib
    p, s, size = C.c_void_p, C.c_char_p, C.c_size_t
    count = _function(lib, "ggml_backend_dev_count", size)
    get = _function(lib, "ggml_backend_dev_get", p, size)
    kind = _function(lib, "ggml_backend_dev_type", C.c_int, p)
    reg = _function(lib, "ggml_backend_dev_backend_reg", p, p)
    provider = _function(lib, "ggml_backend_reg_name", s, p)
    name = _function(lib, "ggml_backend_dev_name", s, p)
    description = _function(lib, "ggml_backend_dev_description", s, p)
    memory = _function(lib, "ggml_backend_dev_memory", None, p, C.POINTER(size), C.POINTER(size))
    for index in range(count()):
        dev = get(index)
        device_type = kind(dev)
        if device_type not in (0, 1):  # pinned enum: CPU, GPU, auxiliary accelerator
            continue
        free, total = size(), size()
        memory(dev, C.byref(free), C.byref(total))
        backend = provider(reg(dev)).decode(errors="replace").lower()
        label = name(dev).decode(errors="replace")
        row = {"provider": backend, "name": label, "description": description(dev).decode(errors="replace"),
               "type": ("cpu", "gpu")[device_type], "total_memory": total.value, "free_memory": free.value}
        if device_type == 1:
            row.update(_driver_details(lib, backend, label))
        yield dev, row


def ggml_devices(L: Any | None = None) -> tuple[dict, ...]:
    """GPU devices registered by the actual backend. RPC support alone does not qualify a GPU."""
    try:
        return tuple(row for _, row in _devices(L or bindings()) if row["type"] == "gpu")
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
        return ()


class _Init(C.Structure):
    _fields_ = [("size", C.c_size_t), ("buffer", C.c_void_p), ("no_alloc", C.c_bool)]


def verify_ggml(provider: str) -> list[dict]:
    """Run a tiny F32 graph directly on each matching registered device, in a doctor child process."""
    L = bindings()
    devices = [(dev, row) for dev, row in _devices(L) if row["provider"] == provider
               and row["type"] == ("cpu" if provider == "cpu" else "gpu")]
    if not devices:
        raise RuntimeError(f"no registered GGML {provider} device")
    lib = L.llama_cpp._lib
    p, size = C.c_void_p, C.c_size_t
    init_backend = _function(lib, "ggml_backend_dev_init", p, p, C.c_char_p)
    init_ctx = _function(lib, "ggml_init", p, _Init)
    tensor = _function(lib, "ggml_new_tensor_1d", p, p, C.c_int, C.c_int64)
    add = _function(lib, "ggml_add", p, p, p, p)
    graph = _function(lib, "ggml_new_graph_custom", p, p, size, C.c_bool)
    expand = _function(lib, "ggml_build_forward_expand", None, p, p)
    supported = _function(lib, "ggml_backend_dev_supports_op", C.c_bool, p, p)
    allocate = _function(lib, "ggml_backend_alloc_ctx_tensors", p, p, p)
    write = _function(lib, "ggml_backend_tensor_set", None, p, p, size, size)
    read = _function(lib, "ggml_backend_tensor_get", None, p, p, size, size)
    compute = _function(lib, "ggml_backend_graph_compute", C.c_int, p, p)
    for dev, row in devices:
        backend = init_backend(dev, None)
        if not backend:
            raise RuntimeError(f"{row['name']} could not initialize")
        ctx = buf = None
        try:
            ctx = init_ctx(_Init(65536, None, True))
            if not ctx:
                raise RuntimeError("GGML context allocation failed")
            left, right = tensor(ctx, 0, 4), tensor(ctx, 0, 4)  # pinned enum GGML_TYPE_F32
            result = add(ctx, left, right)
            if not supported(dev, result):
                raise RuntimeError(f"{row['name']} lacks F32 add")
            plan = graph(ctx, 16, False)
            expand(plan, result)
            buf = allocate(ctx, backend)
            if not buf:
                raise RuntimeError(f"allocation failed on {row['name']}")
            vec = C.c_float * 4
            write(left, vec(1, 2, -3, 4), 0, C.sizeof(vec))
            write(right, vec(4, 1, 2, -7), 0, C.sizeof(vec))
            if compute(backend, plan) != 0:
                raise RuntimeError(f"graph execution failed on {row['name']}")
            observed = vec()
            read(result, observed, 0, C.sizeof(vec))
            if list(observed) != [5, 3, -1, -3]:
                raise RuntimeError(f"incorrect GPU/CPU sum on {row['name']}")
        finally:
            if buf:
                _function(lib, "ggml_backend_buffer_free", None, p)(buf)
            if ctx:
                _function(lib, "ggml_free", None, p)(ctx)
            _function(lib, "ggml_backend_free", None, p)(backend)
    return [row for _, row in devices]
