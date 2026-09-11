#!/usr/bin/env python3
"""Why one Gemma runs the fast quantised matrix-vector kernel and another does not.

`B24S` found that 4B spends 99.6% of its GPU time in `affine_qmv_fast`, while 1B spends
98.6% in plain `affine_qmv` and 12B splits 74/26. The models differ in exactly one way
that matters here: whether the reduction dimension is a multiple of 512.

    1B   hidden 1152 (no), intermediate 6912 (no)
    4B   hidden 2560 (yes), intermediate 10240 (yes)
    12B  hidden 3840 (no), intermediate 15360 (yes)

This measures the price of that difference on the real device: same shape family, K
swept across the boundary, bandwidth rather than latency so the comparison is fair.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZE = 64
BITS = 4


def _bytes_read(rows: int, columns: int) -> int:
    """Packed weights plus one scale and one bias per group, all read once per call."""

    groups = rows * (columns // GROUP_SIZE)
    return rows * columns * BITS // 8 + groups * 2 * 2


def _time_call(weights, scales, biases, vectors, repeats: int, warmup: int) -> float:
    """One synchronisation costs ~240 us, far more than the kernel, so time a whole chain.

    Each call takes a different vector: identical calls would be one node in the graph.
    """

    def chain() -> list:
        return [
            mx.quantized_matmul(
                vector, weights, scales, biases,
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            for vector in vectors
        ]

    for _ in range(warmup):
        mx.eval(chain())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        mx.eval(chain())
        samples.append((time.perf_counter_ns() - started) / len(vectors))
    return median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rows", type=int, default=4096, help="output dimension N")
    parser.add_argument(
        "--columns",
        type=int,
        nargs="+",
        default=[1152, 1280, 1536, 2048, 2560, 3072, 3584, 3840, 4096, 6912, 7168, 10240, 15360],
    )
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--chain", type=int, default=64,
                        help="calls per synchronisation; one sync costs far more than one kernel")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=5,
                        help="interleaved repetitions; direction alternates per block")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    # Interleaved blocks, alternating direction: the machine drifts by tens of percent
    # over a run, and a sequential sweep would charge that drift to whichever shape came
    # last. Every shape is measured once per block, in both orders.
    samples: dict[int, list[float]] = {columns: [] for columns in args.columns}
    for block in range(args.blocks):
        order = args.columns if block % 2 == 0 else list(reversed(args.columns))
        for columns in order:
            dense = mx.random.normal((args.rows, columns)).astype(mx.float16)
            weights, scales, biases = mx.quantize(dense, group_size=GROUP_SIZE, bits=BITS)
            vectors = [
                mx.random.normal((1, columns)).astype(mx.float16) for _ in range(args.chain)
            ]
            mx.eval(weights, scales, biases, *vectors)
            samples[columns].append(
                _time_call(weights, scales, biases, vectors, args.repeats, args.warmup)
            )
            del dense, weights, scales, biases, vectors
            mx.clear_cache()

    results = []
    for columns in args.columns:
        block_times = samples[columns]
        elapsed = median(block_times)
        read = _bytes_read(args.rows, columns)
        results.append(
            {
                "rows": args.rows,
                "columns": columns,
                "columns_mod_512": columns % 512,
                "multiple_of_512": columns % 512 == 0,
                "bytes_read": read,
                "median_ns": elapsed,
                "min_ns": min(block_times),
                "max_ns": max(block_times),
                "blocks": len(block_times),
                "achieved_bytes_per_s": read / (elapsed / 1e9),
                "achieved_bytes_per_s_best": read / (min(block_times) / 1e9),
            }
        )
        spread = (max(block_times) - min(block_times)) / elapsed
        print(
            f"  K={columns:>6} {'aligned  ' if columns % 512 == 0 else 'unaligned'}"
            f" {elapsed/1e3:8.1f} us  {read/(elapsed/1e9)/1e9:7.1f} GB/s"
            f"  best {read/(min(block_times)/1e9)/1e9:7.1f}  spread {spread*100:5.1f}%"
        )

    aligned = [row["achieved_bytes_per_s"] for row in results if row["multiple_of_512"]]
    unaligned = [row["achieved_bytes_per_s"] for row in results if not row["multiple_of_512"]]
    record = {
        "schema": "ironmule.qmv_shape_bench.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "microbenchmark",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "what does a reduction dimension off the 512 boundary cost the quantised matrix-vector kernel",
        "method": (
            "mx.quantized_matmul, 4 bit, group size 64, transpose=True, batch 1; median of "
            f"{args.repeats} calls after {args.warmup} warmups; bandwidth from packed weight "
            "bytes plus scales and biases"
        ),
        "limits": [
            "a microbenchmark on synthetic weights, not a model step",
            "bandwidth assumes every weight byte is read exactly once",
            "kernel selection is inferred from the boundary, not read from MLX source",
        ],
        "rows": args.rows,
        "aligned_bandwidth_median": median(aligned) if aligned else None,
        "unaligned_bandwidth_median": median(unaligned) if unaligned else None,
        "aligned_over_unaligned": (median(aligned) / median(unaligned))
        if aligned and unaligned
        else None,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key in ("aligned_bandwidth_median", "unaligned_bandwidth_median",
                                 "aligned_over_unaligned")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
