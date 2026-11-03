"""AG3: what the agent knows about the machine's architecture, and how close its measured costs come to it.

A decode step reads every weight once, so a memory-bound step takes at least `weight bytes / memory bandwidth`
(the roofline floor). The agent's world model measures the real step (`step_s`). Their ratio is the share of the
card's memory bandwidth a decode step uses. That shows where speed is left, and that headroom lies in kernels,
not in scheduling.

The profile comes from the device's name (`nvidia-smi` or the Apple chip name) and a table of published peak
memory bandwidths. A device the table does not know gets no floor, never a guessed one. Nothing here changes a
decision or an answer.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any, Callable

# Published peak memory bandwidth in GB/s, matched against the device name in order (first match wins). For an
# Apple chip sold in two bandwidth bins, the higher bin is listed, so the floor is a lower bound either way.
BANDWIDTH_GB_S: tuple[tuple[str, float], ...] = (
    (r"\bT4\b", 320.0),
    (r"\bL4\b", 300.0),
    (r"\bA10G?\b", 600.0),
    (r"A100.*80\s?GB.*PCIe|A100.*PCIE.*80", 1935.0),
    (r"A100.*80\s?GB", 2039.0),
    (r"A100", 1555.0),
    (r"H100.*PCIe", 2000.0),
    (r"H100", 3350.0),
    (r"L40S", 864.0),
    (r"V100", 900.0),
    (r"P100", 732.0),
    (r"RTX 4090", 1008.0),
    (r"RTX 4080", 716.8),
    (r"RTX 3090", 936.2),
    (r"RTX 3080", 760.3),
    (r"RTX A6000", 768.0),
    (r"M1 Ultra", 800.0), (r"M1 Max", 400.0), (r"M1 Pro", 200.0), (r"\bM1\b", 68.25),
    (r"M2 Ultra", 800.0), (r"M2 Max", 400.0), (r"M2 Pro", 200.0), (r"\bM2\b", 100.0),
    (r"M3 Ultra", 819.0), (r"M3 Max", 400.0), (r"M3 Pro", 150.0), (r"\bM3\b", 100.0),
    (r"M4 Max", 546.0), (r"M4 Pro", 273.0), (r"\bM4\b", 120.0),
)
# Plans whose decode reads the checkpoint's own weight bytes; float32 and dequantize widen the weights in memory.
CHECKPOINT_PLANS = (None, "native")


def bandwidth_gb_s(name: str | None) -> float | None:
    if not name:
        return None
    for pattern, value in BANDWIDTH_GB_S:
        if re.search(pattern, name):
            return value
    return None


def weight_bytes(snapshot: str | Path | None) -> int | None:
    """The checkpoint's weight bytes: every `*.safetensors` file in its snapshot."""
    if snapshot is None:
        return None
    try:
        files = list(Path(snapshot).glob("*.safetensors"))
        total = sum(path.stat().st_size for path in files)
    except OSError:
        return None
    return total or None


def device_name(device: str | None, run: Callable[..., Any] = subprocess.run) -> str | None:
    """The name of one CUDA device (index or UUID) as `nvidia-smi` reports it; else None."""
    try:
        listed = run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    rows = [[part.strip() for part in line.split(",")] for line in listed.stdout.splitlines()]
    rows = [row for row in rows if len(row) == 3]
    if device is None:
        return rows[0][2] if rows else None
    return next((row[2] for row in rows if device in (row[0], row[1])), None)


def profile(name: str | None, weights: int | str | Path | None, compute_dtype: str | None) -> dict[str, Any]:
    """The architecture facts the agent reports: the device, its peak memory bandwidth, the weights a decode step
    reads (bytes, or the snapshot to count them in), and the floor that puts on a step (None where any of them is
    unknown, with the reason)."""
    bandwidth = bandwidth_gb_s(name)
    if not isinstance(weights, int) or isinstance(weights, bool) or weights <= 0:
        weights = weight_bytes(weights if isinstance(weights, (str, Path)) else None)
    reason = None
    if bandwidth is None:
        reason = "the device's memory bandwidth is not in the table"
    elif weights is None:
        reason = "the checkpoint's weight bytes could not be read"
    elif compute_dtype not in CHECKPOINT_PLANS:
        reason = f"the {compute_dtype} plan widens the weights in memory"
    floor = None if reason else weights / (bandwidth * 1e9)
    return {"device": name, "memory_bandwidth_gb_s": bandwidth,
            "weights_gb": None if weights is None else round(weights / 1e9, 3),
            "decode_floor_ms": None if floor is None else round(floor * 1000, 3), "reason": reason}


def utilisation(hardware: dict[str, Any] | None, step_s: float | None,
                per_session_s: float | None = None) -> dict[str, Any] | None:
    """The measured decode step against the floor: the share of memory bandwidth it uses and the factor a
    bandwidth-bound kernel could still gain at most. With the cost of a further session in a round, also how much
    of a whole step that session adds (`session_cost_share`): near 1 means every session reads the weights again
    (grouped batch-1), near 0 that a round shares them (true batching). None without a floor or a measured step."""
    if not hardware or hardware.get("decode_floor_ms") is None or not step_s or step_s <= 0:
        return None
    floor = hardware["decode_floor_ms"] / 1000
    share = min(1.0, floor / step_s)
    out = {"decode_ms": round(step_s * 1000, 3), "bandwidth_share": round(share, 3),
           "kernel_headroom_x": round(1 / share, 2)}
    if per_session_s is not None and per_session_s >= 0:
        out["session_cost_share"] = round(per_session_s / step_s, 3)
    return out
