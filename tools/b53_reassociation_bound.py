#!/usr/bin/env python3
"""How far apart can two correct summation orders of one Gemma projection row be?

The B53 record's central inference is that the two arms cannot have read the same bytes,
because their outputs differ by several units in the last place of a bfloat16 result. That
inference needs a number, not an argument. This measures the spread of the same dot
product across many float32 summation orders, converts it into bfloat16 units in the last
place, and prints it beside the recorded disagreement.
"""

from __future__ import annotations

import argparse
import json
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The values pytest printed for the one observed failure, decoded as bfloat16.
RECORDED = {
    "2048": {"observed": [86.0, -45.75], "expected": [82.5, -40.0]},
    "4096": {"observed": [-85.5, 14.375], "expected": [-82.0]},
    "15360": {"observed": [27.5, 131.0], "expected": [27.625, 126.5]},
}


def _bfloat16_ulp(value: float) -> float:
    """The gap between neighbouring bfloat16 numbers around `value`."""

    bits = struct.unpack("<I", struct.pack("<f", abs(value)))[0]
    lower = struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]
    upper = struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF0000) + 0x00010000))[0]
    return upper - lower


def _ulps_apart(a: float, b: float) -> float:
    gap = _bfloat16_ulp(max(abs(a), abs(b)) or 1.0)
    return abs(a - b) / gap if gap else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--orders", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Real quantised Gemma-shaped data, dequantised so every order sums the same terms.
    dense = mx.random.normal((args.rows, 3840)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    dequantised = mx.dequantize(weight, scales, biases, group_size=64, bits=4)
    x = mx.random.normal((1, 3840)).astype(mx.bfloat16)
    mx.eval(dequantised, x)

    rows = np.array(dequantised.astype(mx.float32), copy=True)
    vector = np.array(x.astype(mx.float32), copy=True).reshape(-1)
    generator = np.random.default_rng(20260910)

    worst_ulps = 0.0
    worst_relative = 0.0
    for row in rows:
        terms = row * vector
        exact = float(np.sum(terms.astype(np.float64)))
        results = []
        for _ in range(args.orders):
            order = generator.permutation(terms.size)
            results.append(float(np.sum(terms[order], dtype=np.float32)))
        spread = max(results) - min(results)
        worst_ulps = max(worst_ulps, _ulps_apart(max(results), min(results)))
        if exact:
            worst_relative = max(worst_relative, spread / abs(exact))

    recorded = {}
    for width, pair in RECORDED.items():
        pairs = list(zip(pair["observed"], pair["expected"]))
        recorded[width] = [
            {"observed": a, "expected": b, "ulps_apart": round(_ulps_apart(a, b), 3),
             "relative": abs(a - b) / max(abs(b), 1e-30)}
            for a, b in pairs
        ]

    largest_recorded = max(entry["ulps_apart"]
                           for entries in recorded.values() for entry in entries)
    record = {
        "schema": "ironmule.b53_reassociation_bound.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "analysis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "can a different but correct summation order explain the recorded "
                    "disagreement",
        "method": f"{args.rows} real dequantised Gemma projection rows, {args.orders} "
                  "random summation orders each, float32 accumulation, spread expressed "
                  "in bfloat16 units in the last place of the result",
        "reassociation_spread_worst_ulps": worst_ulps,
        "reassociation_spread_worst_relative": worst_relative,
        "recorded_disagreement": recorded,
        "largest_recorded_ulps": largest_recorded,
        "answer": ("no" if largest_recorded > max(worst_ulps, 1.0) * 2 else "not excluded"),
        "reading": (
            "reassociation moves the float32 sum by well under one bfloat16 unit in the "
            "last place, so it can flip at most the final rounding. A disagreement of "
            "several units cannot come from summation order, which leaves the two arms "
            "having read different bytes"
        ),
        "mlx_version": mx.__version__,
        "numpy_version": np.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"reassociation_worst_ulps": worst_ulps,
                      "reassociation_worst_relative": worst_relative,
                      "largest_recorded_ulps": largest_recorded,
                      "answer": record["answer"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
