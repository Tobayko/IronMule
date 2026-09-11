#!/usr/bin/env python3
"""Separate how the data is made from how it is used, and compare the two flows.

Configuration A is the call path as the failing test had it: inputs drawn from MLX's
global random state and used straight away, with one `eval` at the end and no extra
boundaries. Configuration B is the diagnostic flow: an explicit local PRNG key, inputs
materialised once and their bytes digested, every buffer held for the whole comparison,
and each arm evaluated on its own.

B passing neither repairs nor explains A. The point of running both is that a difference
in A alone would locate the fault in the flow, and a difference in both would locate it in
the arithmetic.

Three arms in each configuration, and none of them is assumed correct:

    library against library      does the reference reproduce itself
    transcription against library
    specialised kernel against library

On any difference the inputs, scales, biases, both outputs, the kernel identities and the
first differing byte are written out before anything else happens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b42_qmv_kernel import K3840, PORT, run  # noqa: E402
from b52_automatic_selection import write_once  # noqa: E402

DUMPS = PROJECT_ROOT / ".friday-data" / "b53-dumps"
WIDTHS = (2048, 4096, 15360)
IN_FEATURES = 3840


def _digest(array) -> str:
    return hashlib.sha256(bytes(memoryview(array))).hexdigest()


def _library(weight, scales, biases, x):
    return mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                               group_size=64, bits=4)


def _first_difference(got: bytes, expected: bytes) -> dict:
    for index, (a, b) in enumerate(zip(got, expected)):
        if a != b:
            return {"index": index, "got": a, "expected": b}
    return {"index": min(len(got), len(expected)), "got": None, "expected": None}


def _dump(configuration: str, arm: str, out_features: int, arrays: dict,
          difference: dict) -> str:
    DUMPS.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.time_ns()}-{configuration}-{arm}-{out_features}"
    mx.savez(str(DUMPS / f"{stamp}.npz"), **arrays)
    meta = {"configuration": configuration, "arm": arm, "out_features": out_features,
            "in_features": IN_FEATURES, "first_difference": difference,
            "mlx": mx.__version__,
            "input_digests": {name: _digest(array) for name, array in arrays.items()}}
    (DUMPS / f"{stamp}.json").write_text(json.dumps(meta, indent=2, sort_keys=True),
                                         encoding="utf-8")
    return stamp


def configuration_a(iterations: int) -> dict:
    """The original flow: global random state, no extra boundaries."""

    findings = {}
    for out_features in WIDTHS:
        mismatches = []
        for _ in range(iterations):
            dense = mx.random.normal((out_features, IN_FEATURES)).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
            x = mx.random.normal((1, IN_FEATURES)).astype(mx.bfloat16)
            first = _library(weight, scales, biases, x)
            second = _library(weight, scales, biases, x)
            ported = run(PORT, weight, scales, biases, x, out_features, IN_FEATURES)
            special = run(K3840, weight, scales, biases, x, out_features, IN_FEATURES)
            mx.eval(first, second, ported, special)
            expected = bytes(memoryview(first))
            for arm, produced in (("library_against_library", bytes(memoryview(second))),
                                  ("port_against_library", bytes(memoryview(ported))),
                                  ("k3840_against_library", bytes(memoryview(special)))):
                if produced == expected:
                    continue
                stamp = _dump("A", arm, out_features,
                              {"weight": weight, "scales": scales, "biases": biases,
                               "x": x, "library_first": first, "library_second": second,
                               "ported": ported, "special": special},
                              _first_difference(produced, expected))
                mismatches.append({"arm": arm, "dump": stamp})
        findings[str(out_features)] = {"comparisons": iterations * 3,
                                       "mismatches": mismatches}
    return findings


def configuration_b(iterations: int, seed: int) -> dict:
    """The diagnostic flow: explicit key, materialised bytes, buffers held."""

    findings = {}
    key = mx.random.key(seed)
    for out_features in WIDTHS:
        mismatches = []
        mutations = []
        held = []
        for iteration in range(iterations):
            key, weight_key, vector_key = mx.random.split(key, 3)
            dense = mx.random.normal((out_features, IN_FEATURES),
                                     key=weight_key).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
            x = mx.random.normal((1, IN_FEATURES), key=vector_key).astype(mx.bfloat16)
            mx.eval(weight, scales, biases, x)
            before = {"weight": _digest(weight), "scales": _digest(scales),
                      "biases": _digest(biases), "x": _digest(x)}
            held.append((weight, scales, biases, x))

            first = _library(weight, scales, biases, x)
            mx.eval(first)
            second = _library(weight, scales, biases, x)
            mx.eval(second)
            ported = run(PORT, weight, scales, biases, x, out_features, IN_FEATURES)
            mx.eval(ported)
            special = run(K3840, weight, scales, biases, x, out_features, IN_FEATURES)
            mx.eval(special)

            after = {"weight": _digest(weight), "scales": _digest(scales),
                     "biases": _digest(biases), "x": _digest(x)}
            if after != before:
                mutations.append({"iteration": iteration,
                                  "changed": [name for name in before
                                              if before[name] != after[name]]})
            expected = bytes(memoryview(first))
            for arm, produced in (("library_against_library", bytes(memoryview(second))),
                                  ("port_against_library", bytes(memoryview(ported))),
                                  ("k3840_against_library", bytes(memoryview(special)))):
                if produced == expected:
                    continue
                stamp = _dump("B", arm, out_features,
                              {"weight": weight, "scales": scales, "biases": biases,
                               "x": x, "library_first": first, "library_second": second,
                               "ported": ported, "special": special},
                              _first_difference(produced, expected))
                mismatches.append({"arm": arm, "dump": stamp, "iteration": iteration})
        findings[str(out_features)] = {"comparisons": iterations * 3,
                                       "mismatches": mismatches,
                                       "input_mutations": mutations,
                                       "buffers_held": len(held)}
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    a = configuration_a(args.iterations)
    b = configuration_b(args.iterations, args.seed)

    a_clean = not any(row["mismatches"] for row in a.values())
    b_clean = not any(row["mismatches"] for row in b.values())
    mutated = any(row["input_mutations"] for row in b.values())

    record = {
        "schema": "ironmule.b53_flow_comparison.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the difference live in how the data is made and held, or in the "
                    "arithmetic",
        "configuration_a": {"description": "global random state, no extra eval or "
                                          "synchronisation boundaries, one eval at the end",
                            "per_width": a, "clean": a_clean},
        "configuration_b": {"description": "explicit local PRNG key, inputs materialised "
                                          "and digested, every buffer held, one eval per arm",
                            "per_width": b, "clean": b_clean,
                            "inputs_mutated_by_any_call": mutated},
        "arms": ["library_against_library", "port_against_library",
                 "k3840_against_library"],
        "arm_is_not_truth": "a difference names a disagreement between two arms, not a "
                            "culprit",
        "b_passing_proves_nothing_about_a": True,
        "streams": {
            "default_device": str(mx.default_device()),
            "note": "every call here runs on the default stream; no code in this path "
                    "selects another one",
        },
        "verdict": ("NO DIFFERENCE IN EITHER FLOW" if a_clean and b_clean
                    else "DIFFERENCE IN A ONLY" if b_clean
                    else "DIFFERENCE IN B ONLY" if a_clean
                    else "DIFFERENCE IN BOTH"),
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": record["verdict"], "a_clean": a_clean,
                      "b_clean": b_clean, "inputs_mutated": mutated,
                      "comparisons_per_configuration":
                          sum(row["comparisons"] for row in a.values())},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
