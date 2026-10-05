"""Hardware self-discovery: static fingerprint plus measured device behaviour.

The existing fingerprint keys the tuned profile store. The microbenchmarks
describe achieved quantized GEMV rates and amortized submitted scalar-operation
cost, including host submission and synchronization. They neither measure raw
DRAM peak bandwidth nor establish device kernel counts. Diagnostic knowledge
can inform experiments; only comparative execution evidence can qualify a path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import tempfile
import time
from functools import lru_cache
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from friday_evidence import canonical, statistics
from friday_evidence.canonical import canonical_json_bytes, canonical_sha256
from friday_evidence.statistics import summarise
from friday_evidence.portable import supervisor
from friday_evidence.portable.supervisor import SupervisorError, _lease

def _store() -> Path:
    """Where tuned profiles and fingerprints live.

    Override with `IRONMULE_HOME` to keep the store on a different volume, or to
    give two checkouts separate tuning results on one machine. `IRONMULE_HOME` is also
    the product's state root, which must be private, so the store is created 0700 (DATA3-B:
    a 0755 store made `setup` refuse the directory after `tune` or `benchmark`).
    """
    explicit = os.environ.get("IRONMULE_HOME")
    if explicit:
        return Path(explicit)
    return Path.home() / ".ironmule"


STORE = _store()


#: The one string-valued key `static_facts` reads; every other key is an integer.
_STRING_SYSCTLS = frozenset({"machdep.cpu.brand_string"})


def _sysctl(name: str) -> str | None:
    """One value as `sysctl -n NAME` prints it, read through `sysctlbyname` (P1).

    Reading the kernel directly starts no process, so the fingerprint no longer shells out
    to `sysctl(8)` at all; the values, and with them every stored fingerprint, are the same.
    """
    if platform.system() != "Darwin":
        return None
    try:
        key, size = name.encode(), ctypes.c_size_t(0)
        if _libc().sysctlbyname(key, None, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0 or not size.value:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if _libc().sysctlbyname(key, buffer, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
    except (OSError, AttributeError, ValueError, TypeError):
        return None
    raw = buffer.raw[:size.value]
    if name in _STRING_SYSCTLS:
        value = raw.rstrip(b"\0").decode("utf-8", "replace").strip()
    elif size.value in (4, 8):
        value = str(int.from_bytes(raw, "little", signed=True))
    else:
        return None
    return value or None


class _XswUsage(ctypes.Structure):
    """<sys/sysctl.h>: three 64-bit byte counts, then page size and an encrypted flag."""

    _fields_ = [("xsu_total", ctypes.c_uint64), ("xsu_avail", ctypes.c_uint64),
                ("xsu_used", ctypes.c_uint64), ("xsu_pagesize", ctypes.c_uint32),
                ("xsu_encrypted", ctypes.c_int32)]


@lru_cache(maxsize=1)
def _libc():
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def swap_used_bytes() -> int | None:
    """Swap in use right now, or None if the machine will not say.

    This reads `vm.swapusage` through `sysctlbyname` rather than shelling out to
    `sysctl(8)`, and the reason is measured: `B55` found the two subprocess probes it
    replaced costing 17.4 ms of a 21 ms wrapper overhead, by itself enough to make a
    routed single request 3.7% slower than naming its mode by hand. A telemetry field
    that changes the number it reports on is not telemetry.

    Swap is the failure mode a router has to be able to show: a route that is faster
    while paging is not faster, it is borrowing.
    """
    try:
        usage = _XswUsage()
        size = ctypes.c_size_t(ctypes.sizeof(usage))
        if _libc().sysctlbyname(b"vm.swapusage", ctypes.byref(usage),
                                ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        return int(usage.xsu_used)
    except (OSError, AttributeError, ValueError, TypeError):
        return None


class _VMStatistics64(ctypes.Structure):
    """`<mach/vm_statistics.h>`: the numbers `vm_stat(1)` prints, without running it."""

    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64), ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4
#: `kern.memorystatus_vm_pressure_level`, as macOS defines it.
MEMORY_PRESSURE_NORMAL = 1
MEMORY_PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}


def vm_counters() -> dict[str, int] | None:
    """Every VM figure a resource gate needs, in one Mach call and no subprocess.

    `B65` is why this exists natively. A gate that shells out to `vm_stat(1)` cannot run
    inside a guarded child, and one that samples only between blocks cannot say what
    happened during them. The counters here are the same ones `vm_stat` prints: the
    monotone ones agree exactly, and `free_count` and `wire_count` are instantaneous and
    move between any two reads by construction.
    """
    try:
        stats = _VMStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32))
        if _libc().host_statistics64(_libc().mach_host_self(), _HOST_VM_INFO64,
                                     ctypes.byref(stats), ctypes.byref(count)) != 0:
            return None
        return {name: int(getattr(stats, name)) for name, _type in _VMStatistics64._fields_}
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def memory_pressure_level() -> int | None:
    """macOS's own verdict on memory pressure: 1 normal, 2 warn, 4 critical.

    `B65` measured why this matters. `vm_stat`'s `Swapouts` is a system-wide monotone
    counter that also moves while a machine compacts a swap file it already holds, so a
    gate built on it blocks on the machine's housekeeping rather than on the run. This is
    the system saying whether memory is actually under pressure. `None` means the machine
    would not say, and a gate must treat that as a block, not as a pass.
    """
    try:
        value = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if _libc().sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(value),
                                ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        return int(value.value)
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def installed_memory_bytes() -> int | None:
    """Installed physical memory, or None if the machine will not say.

    `hw.memsize` read through `sysctlbyname`, the same way `swap_used_bytes` reads
    `vm.swapusage`. `static_facts()` also reports this number, but it reaches it through
    `_sysctl()`, which shells out — and `B60` measured what that costs where it matters:
    the Q3f child guard blocks `subprocess.Popen`, so an `Engine` built with
    `wired_fraction > 0` inside a confirmation child raised `GuardViolation` and the child
    exited with status 1, on every model and every machine. The knob could therefore never
    enter a profile. Only the one caller in that path needs the number; `static_facts()`
    runs in the parent and is left exactly as it was.
    """
    if platform.system() != "Darwin":
        # Linux (MLX CUDA): sysconf, still no subprocess for the Q3f guard.
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) or None
        except (OSError, ValueError, AttributeError):
            return None
    try:
        value = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if _libc().sysctlbyname(b"hw.memsize", ctypes.byref(value),
                                ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        return int(value.value) or None
    except (OSError, AttributeError, ValueError, TypeError):
        return None


@lru_cache(maxsize=1)
def _gpu_cores() -> int | None:
    """Apple GPU core count. system_profiler is slow, so this is cached with the rest."""
    try:
        out = subprocess.run(
            ["system_profiler", "-json", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=30,
        )
        for item in json.loads(out.stdout).get("SPDisplaysDataType", []):
            cores = item.get("sppci_cores") or item.get("spdisplays_ndrvs_cores")
            if cores:
                return int(str(cores).split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return None


def static_facts() -> dict[str, Any]:
    """Everything knowable without touching the GPU. Cheap enough to call always."""
    facts: dict[str, Any] = {
        "machine": platform.machine(),
        "system": platform.system(),
        "os_release": platform.release(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "cpu_logical": int(_sysctl("hw.logicalcpu") or 0),
        "cpu_performance": int(_sysctl("hw.perflevel0.logicalcpu") or 0),
        "cpu_efficiency": int(_sysctl("hw.perflevel1.logicalcpu") or 0),
        "memory_bytes": int(_sysctl("hw.memsize") or 0),
        "gpu_cores": _gpu_cores(),
        "python": platform.python_version(),
    }
    try:
        import mlx.core as mx  # noqa: PLC0415 - optional at fingerprint time
        facts["mlx"] = getattr(mx, "__version__", None) or _mlx_version()
        cuda = getattr(mx, "cuda", None)
        facts["gpu_available"] = bool(mx.metal.is_available()
                                      or (cuda is not None and cuda.is_available()))
    except ImportError:
        facts["mlx"] = None
        facts["gpu_available"] = False
    if facts["system"] != "Darwin":
        facts.update(_linux_facts(facts["gpu_available"]))
    return facts


# PORT1 (Kaggle T4): for devices its table does not list, MLX's CUDA backend commits a CUDA
# graph every 20 ops or 100 MB. At 400/4000 IronMule's best 1B arm ran at 0.77x, tokens
# identical (`experiments/kaggle_compat/results/port1-perf-00bb9890`). The MB limit is left
# at MLX's 100: with 4000 a 4B tune child and a 12B float32 load ran out of memory on the
# 16 GB T4 (`port1-run5-632b904f`), while decode's gain comes from fewer, larger op graphs.
CUDA_GRAPH_DEFAULTS = {"MLX_MAX_OPS_PER_BUFFER": "400"}
# Families whose output depends on CUDA graphs on a T4: Qwen 3.5's stock decode gave different
# tokens across processes with graphs and one digest without, at the same speed (ledger PERF1
# run 13 and BACKLOG1, PERF1-T). Keyed by config.json's `model_type`.
CUDA_GRAPHS_OFF_MODEL_TYPES = frozenset({"qwen3_5"})


def apply_cuda_graph_defaults(model_type: str | None = None) -> dict[str, str]:
    """Size CUDA graph commits for pre-Ampere GPUs unless the caller already chose.

    Linux only: the same variables size Metal command buffers, and the Apple path stays
    unchanged. Only compute capability below 8 (Volta/Turing) is touched, which is where
    MLX uses its generic default and where the evidence was measured; A100/H100-class
    devices keep MLX's own per-device values. A `model_type` in `CUDA_GRAPHS_OFF_MODEL_TYPES`
    also switches CUDA graphs off. MLX reads the variables once, at the first GPU operation,
    so callers run this before loading a model; after that it changes nothing MLX reads.
    Returns what it set.
    """
    wanted = dict(CUDA_GRAPH_DEFAULTS)
    if model_type in CUDA_GRAPHS_OFF_MODEL_TYPES:
        wanted["MLX_USE_CUDA_GRAPHS"] = "0"
    if platform.system() != "Linux" or all(name in os.environ for name in wanted):
        return {}
    try:
        import mlx.core as mx  # noqa: PLC0415

        if not mx.cuda.is_available() or int(mx.device_info().get("compute_capability_major", 99)) >= 8:
            return {}
    except Exception:  # noqa: BLE001 - no CUDA device, nothing to size
        return {}
    applied = {name: value for name, value in wanted.items() if name not in os.environ}
    os.environ.update(applied)
    return applied


def _linux_facts(gpu_available: bool) -> dict[str, Any]:
    """The fingerprint fields sysctl fills on a Mac, for Linux hosts running MLX CUDA.

    The Darwin path never calls this, so Mac fingerprints and their profiles are unchanged.
    Without it every Linux host shares one fingerprint and a profile tuned on one GPU
    would be applied on another. The GPU name and memory go into `chip` because the
    fingerprint's key set is fixed.
    """
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    if gpu_available:
        try:
            import mlx.core as mx  # noqa: PLC0415

            info = mx.device_info()
            if info.get("device_name"):
                gpu = f"{info['device_name']} {int(info.get('total_memory', 0)) // 2**20} MiB"
        except Exception:  # noqa: BLE001 - a missing device name leaves the field empty
            gpu = None
    return {"chip": " + ".join(part for part in (cpu, gpu) if part) or None,
            "cpu_logical": os.cpu_count() or 0,
            "memory_bytes": installed_memory_bytes() or 0}


def _mlx_version() -> str | None:
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version("mlx")
    except Exception:
        return None


def fingerprint(facts: dict[str, Any] | None = None) -> str:
    """Stable id for "this hardware plus this MLX". Changes -> retune."""
    facts = static_facts() if facts is None else facts
    keys = ("machine", "chip", "cpu_logical", "memory_bytes", "gpu_cores", "mlx")
    payload = json.dumps({k: facts.get(k) for k in keys}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


PROBE_VERSION = 4
PROBE_SCHEMA = "ironmule.hardware_probe.v4"
DEFAULT_PROBE_MAX_AGE_S = 7 * 86400
_MAX_PROBE_BYTES = 128 * 1024
_MAX_REPEATS = 32
_MAX_TIMING_S = 86400
_TIMING_NAMES = ("scalar_chain_s", "gemv_small_s", "gemv_large_s")
_PROTOCOL = {
    "schema": "ironmule.hardware_microbenchmarks.v1",
    "warmups": 2, "repeats": 5, "scalar_submitted_ops": 512,
    "scalar_dtype": "float32", "gemv_dtype": "bfloat16",
    "in_features": 2560, "out_features": [1024, 262144],
    "quantization_bits": 4, "quantization_group_size": 64,
    "target_bytes": [512 * 2**20, 1024 * 2**20],
    "chain_min": 2, "chain_max": 48,
    "clock": "host_submit_eval_synchronize",
    "random_seed": 20261001, "random_keys": "seed + out_features + index; input index 48",
}


class HardwareProbeUnavailable(RuntimeError):
    """A usable diagnostic cache is absent and measurement was disabled."""

    def __init__(self, reasons: list[str]):
        self.reasons = tuple(reasons)
        super().__init__("hardware probe unavailable: " + ", ".join(reasons))


def _repeats(value: int) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_REPEATS:
        raise ValueError(f"repeats must be an integer in [1, {_MAX_REPEATS}]")
    return value


def _positive(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except (OverflowError, ValueError):
        return False


def _gemv_bytes(out_features: int, target_bytes: int) -> int:
    per = out_features * 2560 * 4 // 8 + out_features * (2560 // 64) * 4
    chain = max(2, min(48, target_bytes // per))
    return chain * per


def _measured_from_samples(samples: dict[str, list[float]]) -> dict[str, Any]:
    summaries = {name: summarise(values) for name, values in samples.items()}
    small = _gemv_bytes(1024, 512 * 2**20) / summaries["gemv_small_s"]["median"] / 1e9
    large = _gemv_bytes(262144, 1024 * 2**20) / summaries["gemv_large_s"]["median"] / 1e9
    return {
        "probe_version": PROBE_VERSION,
        "dispatch_us": summaries["scalar_chain_s"]["median"] / 512 * 1e6,
        "gemv_gbps_small": small, "gemv_gbps_large": large,
        "gemv_size_sensitivity": large / small,
        "raw_timing_samples": samples, "timing_summary": summaries,
        "repeats": len(samples["scalar_chain_s"]),
    }


def _time(fn, repeats: int, *, raw_samples: list[float] | None = None) -> float:
    _repeats(repeats)
    import mlx.core as mx
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        elapsed = time.perf_counter() - started
        if not _positive(elapsed):
            raise ValueError("hardware timing must be finite and positive")
        samples.append(elapsed)
    if raw_samples is not None:
        raw_samples.extend(samples)
    return _median(samples)


def _gemv_gbps(out_features: int, target_bytes: int, repeats: int, *,
               raw_samples: list[float] | None = None) -> float:
    """Achieved bandwidth of a 4-bit GEMV at one weight-matrix size.

    Calls are chained inside a single eval so launch cost is amortised, and enough
    distinct weight buffers are allocated that the system level cache cannot make
    a small matrix look fast. Measured on M1 Max: 1.4 MB reaches ~104 GB/s while
    360 MB reaches ~324 GB/s, so this is a real hardware characteristic, not noise.
    """
    import mlx.core as mx

    in_features, group, bits = 2560, 64, 4
    per = out_features * in_features * bits // 8 + out_features * (in_features // group) * 4
    chain = max(2, min(48, target_bytes // per))
    weights = []
    for index in range(chain):
        key = mx.random.key(_PROTOCOL["random_seed"] + out_features + index)
        dense = mx.random.normal((out_features, in_features), key=key).astype(mx.bfloat16)
        weights.append(mx.quantize(dense, group_size=group, bits=bits))
        del dense
    x = mx.random.normal((1, in_features), key=mx.random.key(
        _PROTOCOL["random_seed"] + out_features + 48)).astype(mx.bfloat16)
    mx.eval(x, *[t for q in weights for t in q])

    def chained():
        return mx.sum(mx.stack([
            mx.quantized_matmul(x, *q, transpose=True, group_size=group, bits=bits).sum()
            for q in weights]))

    elapsed = _time(chained, repeats, raw_samples=raw_samples)
    weights.clear()  # drop the buffers before clearing the cache; `chained` is not called again
    x = None
    mx.clear_cache()
    return chain * per / elapsed / 1e9


def measure(repeats: int = 5) -> dict[str, Any]:
    """Bounded GPU microbenchmarks that describe how this machine executes decode.

    Deliberately *not* a plain streaming-read benchmark: `mx.sum` over a large
    buffer reports 175 GB/s on a machine whose matmuls reach 324 GB/s, so the
    reduction limits it rather than the memory system. What decode is actually made
    of is 4-bit GEMV, so that is what gets measured.
    """
    _repeats(repeats)
    import mlx.core as mx

    apply_cuda_graph_defaults()  # the first GPU operation of `tune` happens here
    samples = {name: [] for name in _TIMING_NAMES}

    # Amortized cost of dependent scalar submissions, including host and waits.
    tiny = mx.zeros((1,), dtype=mx.float32)
    mx.eval(tiny)
    launches = 512

    def chain_tiny():
        value = tiny
        for _ in range(launches):
            value = value + 1.0
        return value

    _time(chain_tiny, repeats, raw_samples=samples["scalar_chain_s"])

    # Achieved GEMV bandwidth at a per-layer matrix size and at an output-head size.
    _gemv_gbps(1024, 512 * 2**20, repeats, raw_samples=samples["gemv_small_s"])
    _gemv_gbps(262144, 1024 * 2**20, repeats, raw_samples=samples["gemv_large_s"])

    mx.clear_cache()
    # These are achieved workload rates, not a raw DRAM bandwidth measurement.
    # dispatch_us includes host graph submission and completion synchronization;
    # it does not establish the number of device kernels that actually executed.
    return _measured_from_samples(samples)


def stable_device_info(raw_info: Mapping[str, Any]) -> dict[str, Any]:
    """Project actual MLX device metadata onto stable identity fields only.

    Free memory and allocation/utilization counters are observations, not device
    identity. Unknown fields are excluded until their stability is established.
    This shared projection performs no inventory or device operations.
    """
    if not isinstance(raw_info, Mapping):
        raise ValueError("MLX device information must be a mapping")
    texts = {"device_name", "architecture", "pci_bus_id", "uuid", "chip_name", "name", "model"}
    positive = {"total_memory", "memory_size", "total_memory_bytes", "memory_bytes",
                "max_buffer_length", "max_recommended_working_set_size", "recommended_max_memory",
                "resource_limit"}
    nonnegative = {"compute_capability_major", "compute_capability_minor"}
    result = {}
    for name in sorted(texts | positive | nonnegative):
        if name not in raw_info:
            continue
        value = raw_info[name]
        if value is not None:
            if name in texts:
                if not isinstance(value, str) or not 1 <= len(value) <= 256 or not value.isprintable():
                    raise ValueError(f"invalid stable MLX device field: {name}")
            elif type(value) is not int or not (1 if name in positive else 0) <= value <= 2**63 - 1:
                raise ValueError(f"invalid stable MLX device field: {name}")
        result[name] = value
    if (not any(result.get(name) for name in ("device_name", "chip_name", "name", "model"))
            or not any(result.get(name) for name in ("total_memory", "memory_size", "total_memory_bytes", "memory_bytes"))):
        raise ValueError("MLX device name or physical memory identity is unavailable")
    return result


def device_identity() -> dict[str, Any]:
    """Stable identity of the device MLX computes on; the CPU itself where no GPU is available.

    MLX's CPU backend reports no device name or memory, which made the learned runtime refuse to
    start on a CPU-only machine (CPU1).
    """
    import mlx.core as mx
    try:
        return stable_device_info(mx.device_info())
    except ValueError:
        cuda = getattr(mx, "cuda", None)
        if mx.metal.is_available() or (cuda is not None and cuda.is_available()):
            raise
        facts = static_facts()
        return stable_device_info({"device_name": f"cpu: {facts.get('chip') or facts['machine']}",
                                   "memory_size": facts["memory_bytes"]})


def _backend_binding(facts: dict[str, Any], *, observation: dict[str, Any] | None = None) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        mlx_lm = version("mlx-lm")
    except PackageNotFoundError:
        mlx_lm = None
    result = {
        "kind": "unavailable", "mlx": _mlx_version(), "mlx_lm": mlx_lm,
        "device_info": None,
        "graph_environment": {name: os.environ.get(name) for name in
                              ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")},
    }
    if observation is not None:
        observation.update(observed_unix_ns=time.time_ns(), free_memory_bytes=None,
                           availability="unavailable", errors=["free_memory_unavailable"])
    if facts.get("gpu_available") is not True:
        return result
    try:
        import mlx.core as mx

        result["mlx"] = getattr(mx, "__version__", None) or result["mlx"]
        cuda = getattr(mx, "cuda", None)
        kind = ("cuda" if cuda is not None and cuda.is_available()
                else "metal" if mx.metal.is_available() else "unavailable")
        info = mx.device_info()
        if kind != "unavailable" and isinstance(info, dict) and info:
            result["device_info"] = stable_device_info(info)
            result["kind"] = kind
            if observation is not None and "free_memory" in info:
                free = info["free_memory"]
                if type(free) is int and 0 <= free <= 2**63 - 1:
                    observation.update(free_memory_bytes=free, availability="available", errors=[])
                else:
                    observation["errors"] = ["free_memory_invalid"]
    except Exception:  # backend metadata unavailable; no performance fact is invented
        pass
    return result


def _probe_binding(facts: dict[str, Any], *, observation: dict[str, Any] | None = None) -> dict[str, Any]:
    sources = {
        "ironmule/hw.py": Path(__file__),
        "friday_evidence/canonical.py": Path(canonical.__file__),
        "friday_evidence/statistics.py": Path(statistics.__file__),
        "friday_evidence/portable/supervisor.py": Path(supervisor.__file__),
    }
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name, path in sources.items()}
    return json.loads(canonical_json_bytes({
        "schema": "ironmule.hardware_probe_binding.v2",
        "fingerprint": fingerprint(facts), "static": facts,
        "backend": (_backend_binding(facts) if observation is None
                    else _backend_binding(facts, observation=observation)), "protocol": _PROTOCOL,
        "source_files": hashes, "source_sha256": canonical_sha256(hashes),
    }))


def _cache_age(value: float) -> float:
    if not _positive(value):
        raise ValueError("max_age_s must be finite and positive")
    return float(value)


def _validate_probe(record: Any, expected: dict[str, Any], *, max_age_s: float,
                    now_unix_ns: int) -> tuple[bool, list[str]]:
    keys = {"schema", "fingerprint", "static", "binding", "observed_unix_ns",
            "finished_unix_ns", "availability", "errors", "measured", "sha256", "device_observations"}
    if not isinstance(record, dict) or set(record) != keys or record.get("schema") != PROBE_SCHEMA:
        return False, ["probe_schema_invalid"]
    reasons = []
    try:
        body = {key: value for key, value in record.items() if key != "sha256"}
        if record["sha256"] != canonical_sha256(body):
            reasons.append("probe_digest_mismatch")
        if (canonical_json_bytes(record["binding"]) != canonical_json_bytes(expected)
                or canonical_json_bytes(record["static"]) != canonical_json_bytes(expected["static"])
                or record["fingerprint"] != expected["fingerprint"]):
            reasons.append("probe_binding_mismatch")
    except (TypeError, ValueError, OverflowError):
        reasons.append("probe_data_invalid")
    observed, finished = record["observed_unix_ns"], record["finished_unix_ns"]
    if (type(observed) is not int or type(finished) is not int
            or not 0 <= observed <= finished <= now_unix_ns):
        reasons.append("probe_timestamp_invalid")
    elif (now_unix_ns - finished) / 1e9 > max_age_s:
        reasons.append("probe_stale")
    errors = record["errors"]
    if not isinstance(errors, list) or any(not isinstance(item, str) or not item for item in errors):
        reasons.append("probe_errors_invalid")
    observations = record["device_observations"]
    observation_keys = {"observed_unix_ns", "free_memory_bytes", "availability", "errors"}
    if not isinstance(observations, dict) or set(observations) != {"before", "after"}:
        reasons.append("probe_device_observation_invalid")
    else:
        for item in observations.values():
            if (not isinstance(item, dict) or set(item) != observation_keys
                    or type(item["observed_unix_ns"]) is not int
                    or not 0 <= item["observed_unix_ns"] <= now_unix_ns
                    or not isinstance(item["errors"], list)
                    or any(not isinstance(error, str) or not error for error in item["errors"])
                    or not isinstance(item["availability"], str)
                    or item["availability"] not in {"available", "unavailable"}):
                reasons.append("probe_device_observation_invalid")
                continue
            if item["availability"] == "available":
                free = item["free_memory_bytes"]
                if type(free) is not int or not 0 <= free <= 2**63 - 1 or item["errors"]:
                    reasons.append("probe_device_observation_invalid")
            elif item["free_memory_bytes"] is not None or not item["errors"]:
                reasons.append("probe_device_observation_invalid")
    available = record["availability"]
    measured = record["measured"]
    if available == "unavailable":
        if measured != {} or not errors:
            reasons.append("probe_unavailable_invalid")
    elif available == "available":
        facts = expected["static"]
        if (errors or facts.get("gpu_available") is not True
                or not facts.get("chip") or not _positive(facts.get("memory_bytes"))
                or not expected["backend"]["mlx"] or expected["backend"]["kind"] == "unavailable"):
            reasons.append("probe_available_invalid")
        measured_keys = {"probe_version", "dispatch_us", "gemv_gbps_small", "gemv_gbps_large",
                         "gemv_size_sensitivity", "raw_timing_samples", "timing_summary", "repeats"}
        if not isinstance(measured, dict) or set(measured) != measured_keys:
            reasons.append("probe_measurement_shape_invalid")
        else:
            raw = measured["raw_timing_samples"]
            repeats = measured["repeats"]
            if (type(measured["probe_version"]) is not int or measured["probe_version"] != PROBE_VERSION
                    or type(repeats) is not int or repeats != _PROTOCOL["repeats"]
                    or not 1 <= repeats <= _MAX_REPEATS or not isinstance(raw, dict)
                    or set(raw) != set(_TIMING_NAMES)
                    or any(not isinstance(raw[name], list) or len(raw[name]) != repeats
                           or any(not _positive(value) or value > _MAX_TIMING_S for value in raw[name])
                           for name in _TIMING_NAMES)):
                reasons.append("probe_timing_samples_invalid")
            else:
                derived = _measured_from_samples(raw)
                if canonical_json_bytes(measured) != canonical_json_bytes(derived):
                    reasons.append("probe_summary_mismatch")
                if any(not _positive(measured[name]) for name in
                       ("dispatch_us", "gemv_gbps_small", "gemv_gbps_large", "gemv_size_sensitivity")):
                    reasons.append("probe_measurements_invalid")
    else:
        reasons.append("probe_availability_invalid")
    return not reasons, sorted(set(reasons))


def validate_probe(record: Any, facts: dict[str, Any] | None = None, *,
                   max_age_s: float = DEFAULT_PROBE_MAX_AGE_S,
                   now_unix_ns: int | None = None) -> tuple[bool, list[str]]:
    """Validate diagnostic evidence against current facts, without GPU workloads.

    Supplying already captured ``facts`` avoids another OS inventory. Backend
    metadata and source hashes are read only. A valid ``unavailable`` record is
    still unavailable; validation never authorizes an execution profile.
    """
    age = _cache_age(max_age_s)
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if type(now) is not int or now < 0:
        return False, ["probe_now_invalid"]
    try:
        expected = _probe_binding(static_facts() if facts is None else facts)
        return _validate_probe(record, expected, max_age_s=age, now_unix_ns=now)
    except (OSError, TypeError, ValueError, OverflowError):
        return False, ["probe_binding_unavailable"]


def _read_probe(path: Path) -> Any:
    # A replaced/symlinked or public cache is not trusted input. The old cache
    # can be refreshed at the same path without following its symlink target.
    before = path.stat(follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
            or before.st_mode & 0o077 or before.st_size > _MAX_PROBE_BYTES):
        raise ValueError("hardware cache is not a bounded private regular file")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise ValueError("hardware cache changed while opening")
        data = stream.read(_MAX_PROBE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(data) != before.st_size or (opened.st_size, opened.st_mtime_ns) !=
            (after.st_size, after.st_mtime_ns)):
        raise ValueError("hardware cache changed while reading")
    current = path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("hardware cache was replaced while reading")
    return json.loads(data)


def _write_probe(path: Path, record: dict[str, Any]) -> None:
    content = canonical_json_bytes(record)
    if len(content) > _MAX_PROBE_BYTES:
        raise ValueError("hardware record exceeds cache budget")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".hw-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cached_probe(path: Path, binding: dict[str, Any], age: float) -> tuple[Any, list[str]]:
    try:
        cached = _read_probe(path)
        valid, reasons = _validate_probe(cached, binding, max_age_s=age, now_unix_ns=time.time_ns())
        return cached if valid else None, reasons
    except (OSError, ValueError, TypeError, OverflowError):
        return None, ["probe_cache_unreadable"]


def probe(force: bool = False, *, cache_dir: Path | None = None,
          max_age_s: float = DEFAULT_PROBE_MAX_AGE_S,
          allow_measure: bool = True) -> dict[str, Any]:
    """Reuse or refresh the existing fingerprint-keyed diagnostic hardware cache.

    The stable binding is separate from volatile timing samples and timestamps.
    Calls that refresh execute the established GPU workloads; run them at an
    owned idle/startup boundary. ``allow_measure=False`` reads only a usable
    cache and otherwise raises ``HardwareProbeUnavailable`` without replacing
    it. This record is not performance qualification.
    """
    if type(force) is not bool:
        raise ValueError("force must be a boolean")
    if type(allow_measure) is not bool:
        raise ValueError("allow_measure must be a boolean")
    age = _cache_age(max_age_s)
    requested = time.time_ns()
    facts = static_facts()
    if allow_measure and facts.get("gpu_available") is True:
        apply_cuda_graph_defaults()
    try:
        before = {}
        binding = _probe_binding(facts, observation=before)
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        raise HardwareProbeUnavailable(["probe_binding_unavailable"]) from exc
    path = Path(cache_dir if cache_dir is not None else STORE) / f"hw-{fingerprint(facts)}.json"
    reasons = ["probe_refresh_requested"]
    if not force:
        cached, reasons = _cached_probe(path, binding, age)
        if cached is not None:
            return cached
    if not allow_measure:
        raise HardwareProbeUnavailable(reasons)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with _lease(path.with_name("." + path.stem + ".lock")):
            # A different caller may have refreshed the cache while this caller
            # was discovering its binding. Concurrent force requests share that
            # new measurement, while a later explicit force still refreshes.
            cached, _ = _cached_probe(path, binding, age)
            if cached is not None and (not force or cached["observed_unix_ns"] >= requested):
                return cached
            observed = time.time_ns()
            available = (facts.get("gpu_available") is True and bool(facts.get("chip"))
                         and _positive(facts.get("memory_bytes")) and bool(binding["backend"]["mlx"])
                         and binding["backend"]["kind"] != "unavailable")
            record = {"schema": PROBE_SCHEMA, "fingerprint": binding["fingerprint"],
                      "static": binding["static"], "binding": binding,
                      "observed_unix_ns": observed, "finished_unix_ns": observed,
                      "availability": "available" if available else "unavailable",
                      "errors": [] if available else ["hardware_or_backend_unavailable"],
                      "device_observations": {"before": before, "after": {}},
                      "measured": measure() if available else {}}
            if _probe_binding(facts, observation=record["device_observations"]["after"]) != binding:
                raise RuntimeError("hardware probe binding changed during measurement")
            record["finished_unix_ns"] = time.time_ns()
            record["sha256"] = canonical_sha256(record)
            valid, reasons = _validate_probe(record, binding, max_age_s=age, now_unix_ns=time.time_ns())
            if not valid:
                raise ValueError("invalid hardware measurement: " + ", ".join(reasons))
            _write_probe(path, record)
            return record
    except SupervisorError as exc:
        reason = "probe_refresh_busy" if str(exc) == "data1_lock_busy" else "probe_lease_unavailable"
        raise HardwareProbeUnavailable([reason]) from exc


def _self_check() -> None:
    facts = static_facts()
    assert facts["machine"], "machine must be known"
    assert fingerprint(facts) == fingerprint(facts), "fingerprint must be stable"
    other = dict(facts, memory_bytes=facts["memory_bytes"] + 1)
    assert fingerprint(other) != fingerprint(facts), "fingerprint must react to hardware"
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([4.0, 1.0, 2.0, 3.0]) == 2.5
    print("hw self-check ok:", fingerprint(facts), facts["chip"], facts["gpu_cores"], "gpu cores")


if __name__ == "__main__":
    _self_check()
