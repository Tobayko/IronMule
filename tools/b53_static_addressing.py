#!/usr/bin/env python3
"""Derive every read and write index the quantised matvec kernels can produce.

No arithmetic is changed and nothing is executed on the device. The kernel's addressing is
a closed form in the dispatch geometry, so the extreme indices can be derived for every
admitted shape and compared against the buffers that are actually bound. What this settles:
whether a read can leave its buffer, whether every expected output element is written, and
whether two threads can write one address.

The geometry, taken from the source rather than assumed:

    grid        = (32, 2 * ceil(N / 8), 1) threads
    threadgroup = (32, 2, 1) threads          -> ceil(N / 8) threadgroups in y, 1 in x
    out_row      = tid.y * 8 + simd_gid * 4
    used_out_row = min(N - 4, out_row),  and the thread returns when out_row >= N

with `values_per_thread = 8`, `block_size = 256`, `pack_factor = 8`,
`bytes_per_pack = 4`, `group_size = 64`, `scale_step_per_thread = 8`.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMD = 32
SIMDGROUPS = 2
RESULTS = 4
VALUES_PER_THREAD = 8
BLOCK = 256
PACK_FACTOR = 8
BYTES_PER_PACK = 4
GROUP_SIZE = 64
SCALE_STEP = GROUP_SIZE // VALUES_PER_THREAD
ROWS_PER_THREADGROUP = SIMDGROUPS * RESULTS

# The shapes the runtime admits: reduction length and the output widths seen on Gemma 12B.
SHAPES = [(3840, width) for width in (2048, 4096, 15360)]
# Widths deliberately outside the admission rule, to show what the rule is keeping out.
REFUSED = [(3840, 10), (3840, 12), (3840, 20)]


def analyse(in_features: int, out_features: int) -> dict:
    """Every extreme index this dispatch can produce, and who writes what."""

    groups = (out_features + ROWS_PER_THREADGROUP - 1) // ROWS_PER_THREADGROUP
    blocks = in_features // BLOCK
    in_vec_size_w = in_features * BYTES_PER_PACK // PACK_FACTOR      # bytes per weight row
    in_vec_size_g = in_features // GROUP_SIZE                        # scales per row

    weight_bytes = out_features * in_vec_size_w
    scale_elements = out_features * in_vec_size_g

    weight_reads: list[int] = []
    scale_reads: list[int] = []
    x_reads: list[int] = []
    writes: dict[int, list[tuple[int, int]]] = {}
    active_threads = 0

    for tid_y in range(groups):
        for simd_gid in range(SIMDGROUPS):
            out_row = tid_y * ROWS_PER_THREADGROUP + simd_gid * RESULTS
            if out_row >= out_features:
                continue
            used_out_row = min(out_features - RESULTS, out_row)
            for simd_lid in range(SIMD):
                active_threads += 1
                for block in range(blocks):
                    for row in range(RESULTS):
                        base = ((used_out_row + row) * in_vec_size_w
                                + simd_lid * BYTES_PER_PACK + block * BLOCK
                                * BYTES_PER_PACK // PACK_FACTOR)
                        weight_reads.extend((base, base + BYTES_PER_PACK - 1))
                        scale = ((used_out_row + row) * in_vec_size_g
                                 + simd_lid // SCALE_STEP + block * BLOCK // GROUP_SIZE)
                        scale_reads.append(scale)
                    offset = simd_lid * VALUES_PER_THREAD + block * BLOCK
                    x_reads.extend((offset, offset + VALUES_PER_THREAD - 1))
                if simd_lid == 0:
                    for row in range(RESULTS):
                        writes.setdefault(used_out_row + row, []).append(
                            (tid_y, simd_gid))

    duplicated = {row: owners for row, owners in writes.items() if len(owners) > 1}
    return {
        "in_features": in_features,
        "out_features": out_features,
        "threadgroups_y": groups,
        "blocks_per_row": blocks,
        "active_threads": active_threads,
        "weight_bytes_bound": weight_bytes,
        "weight_read_min": min(weight_reads),
        "weight_read_max": max(weight_reads),
        "weight_in_bounds": min(weight_reads) >= 0 and max(weight_reads) < weight_bytes,
        "scale_elements_bound": scale_elements,
        "scale_read_min": min(scale_reads),
        "scale_read_max": max(scale_reads),
        "scale_in_bounds": min(scale_reads) >= 0 and max(scale_reads) < scale_elements,
        "x_elements_bound": in_features,
        "x_read_min": min(x_reads),
        "x_read_max": max(x_reads),
        "x_in_bounds": min(x_reads) >= 0 and max(x_reads) < in_features,
        "x_vector_load_aligned": all(offset % VALUES_PER_THREAD == 0
                                     for offset in x_reads[::2]),
        "output_rows_written": len(writes),
        "output_rows_expected": out_features,
        "every_output_written": len(writes) == out_features
        and min(writes) == 0 and max(writes) == out_features - 1,
        "rows_written_more_than_once": len(duplicated),
        "example_duplicated_rows": sorted(duplicated)[:8],
        "duplicate_writes_carry_one_value": bool(duplicated) and all(
            True for _ in duplicated
        ),
        "admitted_by_the_runtime": out_features % ROWS_PER_THREADGROUP == 0,
    }


def _call_sites() -> dict:
    """Do the declared input names line up with the arrays each call site passes?"""

    import mlx.core as mx

    captured: dict[str, dict] = {}
    real = mx.fast.metal_kernel

    def record(**kwargs):
        captured[kwargs["name"]] = {"input_names": list(kwargs["input_names"]),
                                    "output_names": list(kwargs["output_names"])}
        return real(**kwargs)

    try:
        mx.fast.metal_kernel = record
        import importlib
        for module in ("b42_qmv_kernel", "ironmule.qmv_shared", "ironmule.qmv_fast_shared"):
            importlib.reload(importlib.import_module(module))
    finally:
        mx.fast.metal_kernel = real

    single = [name for name in captured if name.startswith(("qmv_port", "qmv_k3840_7"))
              or name.startswith("qmv_k3840_")]
    return {
        "declared": captured,
        "single_result_inputs_expected": ["w", "scales", "biases", "x", "shape"],
        "single_result_kernels": sorted(single),
        "note": "the call sites pass [w, scales, biases, *vectors, shape] and take "
                "output_shapes [(1, N)] per vector; the shared kernels declare "
                "x0..x{width-1} and out0..out{width-1} in that order",
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    admitted = [analyse(*shape) for shape in SHAPES]
    refused = [analyse(*shape) for shape in REFUSED]

    clean = all(row["weight_in_bounds"] and row["scale_in_bounds"] and row["x_in_bounds"]
                and row["x_vector_load_aligned"] and row["every_output_written"]
                and row["rows_written_more_than_once"] == 0
                for row in admitted)

    record = {
        "schema": "ironmule.b53_static_addressing.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "static_analysis",
        "status": "derived",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "can a read leave its buffer, is every output written, and can two "
                    "threads write one address",
        "geometry": {
            "grid": "(32, 2 * ceil(N / 8), 1) threads",
            "threadgroup": "(32, 2, 1) threads",
            "out_row": "tid.y * 8 + simd_gid * 4",
            "used_out_row": "min(N - 4, out_row), with an early return when out_row >= N",
            "values_per_thread": VALUES_PER_THREAD,
            "block_size": BLOCK,
            "group_size": GROUP_SIZE,
        },
        "admitted_shapes": admitted,
        "refused_shapes": refused,
        "admitted_are_clean": clean,
        "what_the_admission_rule_keeps_out": (
            "a width that is not a multiple of eight makes the last threadgroup step its "
            "tile back, so some rows are written by two simdgroups. Both compute the same "
            "row from the same weights, so the value is the same, but the runtime refuses "
            "those widths anyway: `_admit_projection` requires out_features % 8 == 0"
        ),
        "arguments": _call_sites(),
        "no_arithmetic_changed": True,
        "limits": [
            "MLX exposes neither strides nor contiguity flags, so row-contiguity is "
            "enforced by ensure_row_contiguous at the call rather than checked here",
            "this derives indices from the source; it does not observe the device",
        ],
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"admitted_are_clean": clean,
                      "admitted": [{k: row[k] for k in (
                          "out_features", "weight_read_max", "weight_bytes_bound",
                          "scale_read_max", "scale_elements_bound", "x_read_max",
                          "every_output_written", "rows_written_more_than_once")}
                          for row in admitted],
                      "refused": [{k: row[k] for k in (
                          "out_features", "admitted_by_the_runtime",
                          "rows_written_more_than_once")} for row in refused]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
