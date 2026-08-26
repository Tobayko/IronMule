"""Hardware self-discovery: static fingerprint plus measured device behaviour.

The fingerprint keys the tuned profile store, so a machine that has never been
seen before tunes itself once and then reuses the result. The three
microbenchmarks exist to *predict* good starting knobs on unseen hardware
(bandwidth -> how memory bound decode is, dispatch -> whether kernel fusion and
readback batching pay off, wired -> how much of the model can stay resident).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

def _store() -> Path:
    """Where tuned profiles and fingerprints live.

    The project was called Claude Forge before it was called IronMule. An existing
    `~/.claude_forge` is adopted rather than orphaned, so a machine that already
    paid for a tuning run keeps it.
    """
    explicit = os.environ.get("IRONMULE_HOME")
    if explicit:
        return Path(explicit)
    new = Path.home() / ".ironmule"
    legacy = Path.home() / ".claude_forge"
    if not new.exists() and legacy.is_dir():
        return legacy
    return new


STORE = _store()


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


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
