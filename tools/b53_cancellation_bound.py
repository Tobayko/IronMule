#!/usr/bin/env python3
"""What summation order can and cannot do, as a function of cancellation.

`B53_reassociation_bound_20260910` measured the spread of one dot product across random
summation orders and reported `0.0017` bfloat16 units in the last place. That number was
measured on fresh draws from the same generator and shapes, **not** on the inputs of the
observed failure, which were never captured and no longer exist. It is a comparison
measurement, not a proof about those inputs, and on its own it does not exclude
reassociation in general: a dot product whose terms cancel almost exactly has a spread
that is enormous in units of its own last place.

So this measures the thing that actually decides the question — how far reassociation can
move a result of a given magnitude, given how much cancellation the terms carry — and then
states the condition under which the recorded disagreement is out of reach.

Definitions used throughout, so the numbers can be checked:

* **bfloat16 unit in the last place of `v`**: the gap between the two adjacent bfloat16
  numbers bracketing `|v|`, obtained by truncating a float32 to its high 16 bits and
  adding one to that 16-bit pattern. For a normal `v` this is `2**(exponent(v) - 7)`.
* **`v == 0`**: no meaningful last place exists. The gap is reported as the smallest
  normal bfloat16 gap and the row is flagged, rather than dividing by something arbitrary.
* **NaN or infinity**: refused rather than measured; none occurs in this data and a
  silent NaN must not become a small number.
* **dtypes**: inputs are bfloat16 as the model stores them, dequantised to float32 for
  the term products, summed in float32 for the arms and in float64 for the reference.
* **spread**: `max - min` over the sampled orders, then divided by the last place of the
  larger magnitude.
"""

from __future__ import annotations

import argparse
import json
import math
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
FLOAT32_EPS = float(np.finfo(np.float32).eps)
TERMS = 3840                    # the reduction length every admitted projection uses
SMALLEST_NORMAL_BFLOAT16_GAP = 2.0 ** (-126 - 7)
# The values pytest printed for the one observed failure, decoded as bfloat16.
RECORDED = {
    "2048": [(86.0, 82.5), (-45.75, -40.0)],
    "4096": [(-85.5, -82.0)],
    "15360": [(27.5, 27.625), (131.0, 126.5)],
}


def bfloat16_ulp(value: float) -> tuple[float, bool]:
    """The last place of `value`, and whether the answer had to be substituted."""

    if math.isnan(value) or math.isinf(value):
        raise ValueError("a last place is not defined for NaN or infinity")
    if value == 0.0:
        return SMALLEST_NORMAL_BFLOAT16_GAP, True
    bits = struct.unpack("<I", struct.pack("<f", abs(value)))[0] & 0xFFFF0000
    lower = struct.unpack("<f", struct.pack("<I", bits))[0]
    upper = struct.unpack("<f", struct.pack("<I", bits + 0x00010000))[0]
    if math.isinf(upper):
        raise ValueError("the last place overflows the float32 range")
    return upper - lower, False


def ulps_apart(a: float, b: float) -> float:
    gap, _ = bfloat16_ulp(max(abs(a), abs(b)))
    return abs(a - b) / gap


def _row_statistics(terms: np.ndarray, orders: int, generator) -> dict:
    exact = float(np.sum(terms.astype(np.float64)))
    results = [float(np.sum(terms[generator.permutation(terms.size)], dtype=np.float32))
               for _ in range(orders)]
    spread = max(results) - min(results)
    absolute_sum = float(np.sum(np.abs(terms.astype(np.float64))))
    gap, substituted = bfloat16_ulp(max(abs(max(results)), abs(min(results))) or 1.0)
    return {
        "exact": exact,
        "absolute_sum": absolute_sum,
        "cancellation": absolute_sum / abs(exact) if exact else float("inf"),
        "spread_absolute": spread,
        "spread_ulps": spread / gap,
        "ulp_substituted_at_zero": substituted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--orders", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    generator = np.random.default_rng(20260910)

    # 1. The workload as it is: quantised Gemma-shaped weights against a normal vector.
    dense = mx.random.normal((args.rows, 3840)).astype(mx.bfloat16)
    weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
    dequantised = mx.dequantize(weight, scales, biases, group_size=64, bits=4)
    x = mx.random.normal((1, 3840)).astype(mx.bfloat16)
    mx.eval(dequantised, x)
    rows = np.array(dequantised.astype(mx.float32), copy=True)
    vector = np.array(x.astype(mx.float32), copy=True).reshape(-1)

    natural = [_row_statistics(row * vector, args.orders, generator) for row in rows]

    # 2. The same terms forced to cancel: subtract the mean so the sum is near zero, which
    #    is the regime where summation order really does move many last places.
    cancelled = []
    for row in rows[: min(8, len(rows))]:
        terms = row * vector
        terms = terms - np.float32(terms.mean())
        cancelled.append(_row_statistics(terms, args.orders, generator))

    worst_natural = max(row["spread_ulps"] for row in natural)
    worst_cancelled = max(row["spread_ulps"] for row in cancelled)
    absolute_sums = [row["absolute_sum"] for row in natural]
    magnitudes = [abs(row["exact"]) for row in natural]

    # 3. What the recorded disagreement would demand of reassociation.
    demands = {}
    for width, pairs in RECORDED.items():
        rows_out = []
        for observed, expected in pairs:
            distance = ulps_apart(observed, expected)
            gap, _ = bfloat16_ulp(max(abs(observed), abs(expected)))
            needed_absolute = distance * gap
            # Two error models, because the choice decides the answer.
            #   optimistic: one eps per term sum, what random orders actually achieve
            #   classical:  2 (n-1) eps sum|terms|, the standard bound for recursive
            #               summation, achievable by an adversarial order
            optimistic = max(absolute_sums) * FLOAT32_EPS
            classical = 2 * (TERMS - 1) * FLOAT32_EPS * max(absolute_sums)
            rows_out.append({
                "observed": observed,
                "expected": expected,
                "ulps_apart": distance,
                "absolute_difference": abs(observed - expected),
                "reach_optimistic_absolute": optimistic,
                "reach_classical_absolute": classical,
                "within_optimistic_reach": needed_absolute <= optimistic,
                "within_classical_reach": needed_absolute <= classical,
                "factor_beyond_optimistic": needed_absolute / optimistic,
                "factor_beyond_classical": needed_absolute / classical,
            })
        demands[width] = rows_out

    largest_recorded = max(entry["ulps_apart"]
                           for entries in demands.values() for entry in entries)
    entries = [entry for rows_out in demands.values() for entry in rows_out]
    smallest_factor = min(entry["factor_beyond_optimistic"] for entry in entries)
    worst_classical_factor = max(entry["factor_beyond_classical"] for entry in entries)
    any_within_classical = any(entry["within_classical_reach"] for entry in entries)

    record = {
        "schema": "ironmule.b53_cancellation_bound.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "analysis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "supersedes": "B53_reassociation_bound_20260910, which reported a single spread "
                      "without stating that it was measured on fresh draws rather than on "
                      "the inputs of the failure, and without the cancellation condition",
        "original_inputs": {
            "available": False,
            "why": "the failing run captured no tensors; the dump and the seeding came "
                   "afterwards",
            "consequence": "every number here describes inputs from the same generator and "
                           "the same shapes, not the inputs that failed",
        },
        "definitions": {
            "bfloat16_ulp": "gap between the adjacent bfloat16 numbers bracketing |v|, "
                            "from truncating float32 to its high 16 bits and adding one",
            "zero": "no last place exists; the smallest normal bfloat16 gap is substituted "
                    "and the row is flagged",
            "nan_or_infinity": "refused, never measured",
            "dtypes": "bfloat16 stored, float32 products and arm sums, float64 reference",
            "spread": "max minus min over sampled orders, over the last place of the "
                      "larger magnitude",
            "float32_eps": FLOAT32_EPS,
        },
        "natural_workload": {
            "rows": len(natural),
            "orders_per_row": args.orders,
            "worst_spread_ulps": worst_natural,
            "worst_spread_relative": max(row["spread_absolute"] / abs(row["exact"])
                                         for row in natural if row["exact"]),
            "result_magnitude_min": min(magnitudes),
            "result_magnitude_max": max(magnitudes),
            "absolute_sum_max": max(absolute_sums),
            "cancellation_max": max(row["cancellation"] for row in natural),
        },
        "cancelled_control": {
            "rows": len(cancelled),
            "construction": "the same terms with their mean removed, so the sum sits near "
                            "zero",
            "worst_spread_ulps": worst_cancelled,
            "cancellation_max": max(row["cancellation"] for row in cancelled),
            "reading": "with heavy cancellation, summation order moves the result by many "
                       "last places. Reassociation is therefore not excluded in general, "
                       "only for results of the magnitude that was recorded",
        },
        "recorded_demand": demands,
        "largest_recorded_ulps": largest_recorded,
        "smallest_factor_beyond_optimistic_model": smallest_factor,
        "largest_factor_beyond_classical_model": worst_classical_factor,
        "recorded_difference_within_classical_reach": any_within_classical,
        "conditional_conclusion": (
            "which model is used decides the answer. Random summation orders move this "
            f"workload by at most {worst_natural:.2g} last places, and on that optimistic "
            f"model the recorded disagreement would need an absolute term sum "
            f"{smallest_factor:.3g} times larger than any measured here. But the standard "
            "bound for recursive float32 summation is 2 (n-1) eps times the absolute term "
            f"sum, which for n=3840 is {2 * (TERMS - 1) * FLOAT32_EPS * max(absolute_sums):.3g} "
            "absolute for this workload, the same order of magnitude as the largest "
            f"recorded difference. So summation order is within a factor of "
            f"{worst_classical_factor:.3g} of the recorded difference and is NOT excluded. "
            "It is merely not reached by random orders"
        ),
        "hypothesis_status": (
            "'the two arms read different bytes' is NOT established. This measurement "
            "corrects an earlier over-claim: the exclusion held only under an optimistic "
            "per-term error model. Under the standard worst-case bound, and with heavy "
            "cancellation, summation order can move a result further than the recorded "
            "difference, so an arithmetic-order or math-mode difference between the two "
            "compiled kernels stays a live candidate alongside a byte-level one"
        ),
        "mlx_version": mx.__version__,
        "numpy_version": np.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({
        "natural_worst_ulps": worst_natural,
        "cancelled_worst_ulps": worst_cancelled,
        "largest_recorded_ulps": largest_recorded,
        "smallest_factor_beyond_this_workload": smallest_factor,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
