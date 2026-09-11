"""Hardware self-discovery: static fingerprint plus measured device behaviour.

The fingerprint keys the tuned profile store, so a machine that has never been
seen before tunes itself once and then reuses the result. The three
microbenchmarks exist to *predict* good starting knobs on unseen hardware
(bandwidth -> how memory bound decode is, dispatch -> whether kernel fusion and
readback batching pay off, wired -> how much of the model can stay resident).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

def _store() -> Path:
    """Where tuned profiles and fingerprints live.

    Override with `IRONMULE_HOME` to keep the store on a different volume, or to
    give two checkouts separate tuning results on one machine.
    """
    explicit = os.environ.get("IRONMULE_HOME")
    if explicit:
        return Path(explicit)
    return Path.home() / ".ironmule"


STORE = _store()


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
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
        facts["gpu_available"] = mx.metal.is_available()
    except ImportError:
        facts["mlx"] = None
        facts["gpu_available"] = False
    return facts


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


PROBE_VERSION = 2   # bump when measure() changes shape, so cached records refresh


def _time(fn, repeats: int) -> float:
    import mlx.core as mx
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return _median(samples)


def _gemv_gbps(out_features: int, target_bytes: int, repeats: int) -> float:
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
    for _ in range(chain):
        dense = mx.random.normal((out_features, in_features)).astype(mx.bfloat16)
        weights.append(mx.quantize(dense, group_size=group, bits=bits))
        del dense
    x = mx.random.normal((1, in_features)).astype(mx.bfloat16)
    mx.eval(x, *[t for q in weights for t in q])

    def chained():
        return mx.sum(mx.stack([
            mx.quantized_matmul(x, *q, transpose=True, group_size=group, bits=bits).sum()
            for q in weights]))

    elapsed = _time(chained, repeats)
    del weights, x
    mx.clear_cache()
    return chain * per / elapsed / 1e9


def measure(repeats: int = 5) -> dict[str, float]:
    """Bounded GPU microbenchmarks that describe how this machine executes decode.

    Deliberately *not* a plain streaming-read benchmark: `mx.sum` over a large
    buffer reports 175 GB/s on a machine whose matmuls reach 324 GB/s, so the
    reduction limits it rather than the memory system. What decode is actually made
    of is 4-bit GEMV, so that is what gets measured.
    """
    import mlx.core as mx

    results: dict[str, float] = {"probe_version": PROBE_VERSION}

    # Kernel dispatch cost: many dependent, trivially sized kernels.
    tiny = mx.zeros((1,), dtype=mx.float32)
    mx.eval(tiny)
    launches = 512

    def chain_tiny():
        value = tiny
        for _ in range(launches):
            value = value + 1.0
        return value

    results["dispatch_us"] = _time(chain_tiny, repeats) / launches * 1e6

    # Achieved GEMV bandwidth at a per-layer matrix size and at an output-head size.
    results["gemv_gbps_small"] = _gemv_gbps(1024, 512 * 2**20, repeats)
    results["gemv_gbps_large"] = _gemv_gbps(262144, 1024 * 2**20, repeats)
    # >1 means small matrices are penalised, so merging projections may pay here.
    results["gemv_size_sensitivity"] = results["gemv_gbps_large"] / results["gemv_gbps_small"]

    mx.clear_cache()
    return results


def probe(force: bool = False) -> dict[str, Any]:
    """Full hardware record, cached per fingerprint under IRONMULE_HOME."""
    facts = static_facts()
    ident = fingerprint(facts)
    path = STORE / f"hw-{ident}.json"
    if path.is_file() and not force:
        try:
            cached = json.loads(path.read_text())
            if cached.get("measured", {}).get("probe_version") == PROBE_VERSION:
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    record: dict[str, Any] = {"fingerprint": ident, "static": facts, "measured": {}}
    if facts.get("gpu_available"):
        record["measured"] = measure()
    STORE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


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
