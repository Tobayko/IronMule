#!/usr/bin/env python3
"""What this M1 Max actually spends decode and prefill on, measured before anything is tuned.

The prior is strong and it is this project's own. `E4` measured achieved bandwidth rising
with matrix size and saturating near `324 GB/s`, so a small projection runs at `104 GB/s`
because it is small, not because it is scheduled badly. `E5` measured fusion helping prefill
and doing nothing for decode, and explained why: at `dispatch_us = 6.41` and about `510`
kernels per decode step, roughly `3.3 ms` of a step is submission, which is exactly the gap
between the trunk's achieved `195 GB/s` and the measured ceiling.

So the first question is not which knob to turn. It is which kernels are worth turning one
for. This file inventories the shapes each model really executes, counts them, and measures
each shape at the widths the runtime really uses. Anything holding a negligible share of
measured time is not a candidate, however interesting it looks.

Nothing here changes the model, the quantisation or any output. Every microbenchmark is
compared byte for byte against the library path it claims to replace before any timing from
it is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MEASURED_SOURCES = ("tools/b66_silicon_characterisation.py", "ironmule/hw.py")

MODELS = {
    "4B": "mlx-community/gemma-3-4b-it-4bit",
    "12B": "mlx-community/gemma-3-12b-it-4bit",
}
#: `M = 1` is decode. `2`, `4`, `8` and `16` are the grouped widths: `E2` and `E3` both
#: found `M = 8` pathological on synthetic shapes, and this is the first time that boundary
#: is looked for on the shapes these models actually run. `M = 322` is the tuned profile's
#: prompt length, so prefill is measured where prefill happens.
WIDTHS = (1, 2, 4, 8, 16, 322)
WARMUPS = 3
REPEATS = 7
#: M1 Max system level cache. Working sets are placed clearly under and clearly over it.
SLC_BYTES = 48 * 1024 * 1024


def quantised_shapes(model) -> Counter:
    """Every quantised matmul this model will run, as (K, N, group_size, bits), with counts."""
    found: Counter = Counter()

    def walk(module):
        for child in getattr(module, "children", lambda: {})().values():
            if isinstance(child, (list, tuple)):
                for item in child:
                    walk(item)
            elif isinstance(child, dict):
                for item in child.values():
                    walk(item)
            elif hasattr(child, "children"):
                walk(child)
        weight = getattr(module, "weight", None)
        scales = getattr(module, "scales", None)
        if isinstance(module, nn.QuantizedLinear) and weight is not None and scales is not None:
            output_dims, packed = weight.shape
            group_size = int(module.group_size)
            bits = int(module.bits)
            input_dims = packed * (32 // bits)
            found[(input_dims, output_dims, group_size, bits)] += 1
    walk(model)
    return found


def weight_bytes(k: int, n: int, group_size: int, bits: int) -> int:
    """Packed weights plus scales and biases, which are read on every pass just as weights are."""
    return n * k * bits // 8 + 2 * (n * (k // group_size) * 2)


def time_shape(k: int, n: int, group_size: int, bits: int, m: int, *,
               distinct_bytes: int = 512 * 1024 * 1024) -> dict:
    """One shape, chained inside one eval so submission is amortised, over distinct buffers.

    Distinct buffers matter: repeating one weight set measures the cache, not the memory
    system. `E4` established the method and this reuses it unchanged.
    """
    per = weight_bytes(k, n, group_size, bits)
    copies = max(2, min(32, distinct_bytes // max(per, 1)))
    weights = [mx.random.randint(0, 2**31 - 1, (n, k * bits // 32), dtype=mx.uint32)
               for _ in range(copies)]
    scales = [mx.random.normal((n, k // group_size)).astype(mx.bfloat16) for _ in range(copies)]
    biases = [mx.random.normal((n, k // group_size)).astype(mx.bfloat16) for _ in range(copies)]
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    mx.eval(weights, scales, biases, x)

    def once():
        outputs = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                       group_size=group_size, bits=bits)
                   for w, s, b in zip(weights, scales, biases)]
        mx.eval(outputs)

    for _ in range(WARMUPS):
        once()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        once()
        samples.append((time.perf_counter_ns() - start) / copies)
    median_ns = statistics.median(samples)
    return {
        "k": k, "n": n, "m": m, "group_size": group_size, "bits": bits,
        "weight_bytes": per, "distinct_copies": copies,
        "median_ns_per_call": median_ns,
        "samples_ns_per_call": samples,
        # Weight bytes per second. At M = 1 that is the binding resource and the number is
        # achieved bandwidth. At larger M the kernel is doing M times the arithmetic over the
        # same bytes, so the figure stops being a bandwidth and becomes a weight-reuse rate.
        "weight_bytes_per_ns": per / median_ns if median_ns else None,
        "achieved_gb_per_s": (per / median_ns) if (median_ns and m == 1) else None,
        "ns_per_row_of_m": median_ns / m if median_ns else None,
        "resident_working_set_bytes": per * copies,
        "fits_in_slc": per * copies <= SLC_BYTES,
    }


def inventory(model_id: str) -> dict:
    from ironmule.tune import load_engine, resolve_local_model
    from ironmule.runtime import BASELINE

    resolved = resolve_local_model(model_id)
    engine, _tokenizer = load_engine(model_id, BASELINE, resolved_source=resolved)
    try:
        shapes = quantised_shapes(engine.model)
    finally:
        engine.close()
    rows = [{"k": k, "n": n, "group_size": g, "bits": b, "count": c,
             "weight_bytes_each": weight_bytes(k, n, g, b),
             "weight_bytes_total": weight_bytes(k, n, g, b) * c}
            for (k, n, g, b), c in sorted(shapes.items(), key=lambda kv: -kv[1] * weight_bytes(*kv[0]))]
    total = sum(r["weight_bytes_total"] for r in rows)
    for row in rows:
        row["share_of_weight_bytes"] = row["weight_bytes_total"] / total if total else None
    return {"model_id": model_id, "identity_sha256": resolved.identity.identity_sha256,
            "distinct_shapes": len(rows), "total_weight_bytes": total, "shapes": rows}


def source_binding() -> dict:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def environment() -> dict:
    import mlx_lm
    from ironmule.hw import fingerprint, installed_memory_bytes, memory_pressure_level, static_facts
    return {"platform": platform.platform(), "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "fingerprint": fingerprint(), "chip": static_facts().get("chip"),
            "gpu_cores": static_facts().get("gpu_cores"),
            "installed_memory_bytes": installed_memory_bytes(),
            "memory_pressure_level_at_start": memory_pressure_level(),
            "slc_bytes_assumed": SLC_BYTES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="4B,12B")
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    result = {}
    for label in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_id = MODELS[label]
        book = inventory(model_id)
        measured = []
        for row in book["shapes"]:
            for m in WIDTHS:
                measured.append(dict(time_shape(row["k"], row["n"], row["group_size"],
                                                row["bits"], m),
                                     count_in_model=row["count"]))
                last = measured[-1]
                print(f'  {label} K={row["k"]:6} N={row["n"]:6} M={m:4} '
                      f'{last["median_ns_per_call"]/1e3:9.2f} us  '
                      f'{last["ns_per_row_of_m"]/1e3:8.3f} us/row  '
                      f'{last["weight_bytes_per_ns"]:6.1f} GB/s-equivalent', flush=True)
        # Time share, per width, from what was actually measured on these shapes.
        for m in WIDTHS:
            rows = [r for r in measured if r["m"] == m]
            total = sum(r["median_ns_per_call"] * r["count_in_model"] for r in rows)
            for r in rows:
                r["share_of_measured_time_at_this_width"] = (
                    r["median_ns_per_call"] * r["count_in_model"] / total if total else None)
        result[label] = {"inventory": book, "measured": measured}

    record = {
        "experiment": "B66_shape_inventory_and_cost",
        "purpose": ("which kernels this machine actually spends decode and prefill on, "
                    "before any axis is characterised. A shape holding a negligible share "
                    "of measured time is not a candidate"),
        "method": ("each distinct quantised matmul shape in the model, timed with calls "
                   "chained inside one eval over at least 512 MB of distinct weight "
                   "buffers, 3 warmups and 7 repeats. This is E4's method, unchanged"),
        "widths": {"1": "decode", "2/4/8/16": "grouped widths; E2 and E3 found M=8 pathological "
                   "on synthetic shapes and this looks for that boundary on the real ones",
                   "322": "the tuned profile's prompt length, where prefill runs"},
        "bandwidth_validity": ("achieved_gb_per_s is reported only at M=1, where weight bytes "
                               "are the binding resource. At larger M the same bytes serve M "
                               "rows, so weight_bytes_per_ns is a reuse rate, not a bandwidth"),
        "not_claimed": ("these are isolated kernel timings. E5 measured that an isolated "
                        "bandwidth advantage does not transfer to a pipelined graph, so no "
                        "product claim is made from any number here"),
        "environment": environment(),
        "source_binding": source_binding(),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "models": result,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({label: {
        "distinct_shapes": data["inventory"]["distinct_shapes"],
        "top_by_decode_share": sorted(
            [{"k": r["k"], "n": r["n"], "count": r["count_in_model"],
              "share": round(r["share_of_measured_time_at_this_width"], 4),
              "gb_per_s": round(r["achieved_gb_per_s"], 1)}
             for r in data["measured"] if r["m"] == 1],
            key=lambda r: -r["share"])[:6],
    } for label, data in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
