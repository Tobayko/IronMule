#!/usr/bin/env python3
"""Qualify and time the `K=3840` kernels on the real Gemma 12B weights.

Two gates, in order. First bit identity: every captured activation is pushed through
every captured projection, and the output bytes must equal `mx.quantized_matmul`'s. A
single differing byte locks that variant out of the timing.

Then speed, on the same real matrices. One round walks every projection once, so the
weights read per round are far past any cache and the same matrix is never hammered.
An A/A arm runs the reference against itself to price the measurement noise before any
claim is made about the candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b42_qmv_kernel import COMPILE_OPTIONS, K3840, PORT, run  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_K = 3840


def _digest(array: mx.array) -> str:
    return hashlib.sha256(bytes(memoryview(array))).hexdigest()


def _resolve(root, path: str):
    node = root
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _reference(module, x):
    return mx.quantized_matmul(
        x, module.weight, module.scales, module.biases,
        transpose=True, group_size=module.group_size, bits=module.bits,
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--rounds", type=int, default=3, help="rounds timed per block")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.capture / "manifest.json").read_text())
    model, _ = load(args.model)
    language = getattr(model, "language_model", model)

    # Bind every captured activation to its module and prove both are the recorded bytes.
    cases, mismatches = [], []
    for entry in manifest["activations"]:
        path = entry["module_path"]
        module = _resolve(language, path)
        x = mx.load(str(args.capture / entry["file"]))
        mx.eval(x)
        weight_entry = next(w for w in manifest["weights"] if w["module_path"] == path)
        checks = {
            "activation": _digest(x) == entry["sha256"],
            "weight": _digest(module.weight) == weight_entry["weight_sha256"],
            "scales": _digest(module.scales) == weight_entry["scales_sha256"],
            "biases": _digest(module.biases) == weight_entry["biases_sha256"],
        }
        if not all(checks.values()):
            mismatches.append({"module_path": path, "checks": checks})
            continue
        cases.append({
            "path": path,
            "module": module,
            "x": x,
            "out_features": int(module.weight.shape[0]),
            "bytes": int(module.weight.nbytes + module.scales.nbytes + module.biases.nbytes),
        })
    if mismatches:
        print(json.dumps({"state": "capture_mismatch", "mismatches": mismatches}, indent=2))
        return 1

    # Gate one: bit identity on real weights and real activations.
    identity = []
    for case in cases:
        module, x = case["module"], case["x"]
        reference = _reference(module, x)
        ported = run(PORT, module.weight, module.scales, module.biases, x,
                     case["out_features"], TARGET_K)
        special = run(K3840, module.weight, module.scales, module.biases, x,
                      case["out_features"], TARGET_K)
        mx.eval(reference, ported, special)
        reference_bytes = bytes(memoryview(reference))
        identity.append({
            "module_path": case["path"],
            "out_features": case["out_features"],
            "port_bit_identical": bytes(memoryview(ported)) == reference_bytes,
            "k3840_bit_identical": bytes(memoryview(special)) == reference_bytes,
        })
    port_clean = all(row["port_bit_identical"] for row in identity)
    k3840_clean = all(row["k3840_bit_identical"] for row in identity)
    print(json.dumps({"port_bit_identical": port_clean,
                      "k3840_bit_identical": k3840_clean,
                      "cases": len(identity)}, sort_keys=True))

    arms = {"reference_a": [], "reference_b": [], "port": [], "k3840": []}
    round_bytes = sum(case["bytes"] for case in cases)
    if k3840_clean and port_clean:
        def sweep(which: str):
            outputs = []
            for case in cases:
                module, x = case["module"], case["x"]
                if which.startswith("reference"):
                    outputs.append(_reference(module, x))
                else:
                    kernel = PORT if which == "port" else K3840
                    outputs.append(run(kernel, module.weight, module.scales, module.biases,
                                       x, case["out_features"], TARGET_K))
            mx.eval(outputs)
            return outputs

        for name in arms:  # compile and warm every arm outside the timed region
            for _ in range(2):
                sweep(name)

        order = list(arms)
        for block in range(args.blocks):
            rotation = order[block % len(order):] + order[: block % len(order)]
            for name in rotation:
                began = time.perf_counter_ns()
                for _ in range(args.rounds):
                    sweep(name)
                arms[name].append((time.perf_counter_ns() - began) / args.rounds)

    medians = {name: median(values) if values else None for name, values in arms.items()}

    def paired(numerator: str, denominator: str) -> dict:
        """Block-paired ratios with a percentile bootstrap, the way the ledger reports."""

        if not arms[numerator] or not arms[denominator]:
            return {}
        ratios = [n / d for n, d in zip(arms[numerator], arms[denominator])]
        rng = random.Random(20260909)
        draws = sorted(
            median([ratios[rng.randrange(len(ratios))] for _ in ratios])
            for _ in range(10000)
        )
        return {
            "median": median(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "blocks": len(ratios),
            "below_one": sum(1 for value in ratios if value < 1.0),
            "ci95_low": draws[int(0.025 * len(draws))],
            "ci95_high": draws[int(0.975 * len(draws)) - 1],
        }

    comparisons = {
        "aa_reference_b_over_a": paired("reference_b", "reference_a"),
        "port_over_reference": paired("port", "reference_a"),
        "k3840_over_reference": paired("k3840", "reference_a"),
        "k3840_over_port": paired("k3840", "port"),
    }
    aa_ratio = (medians["reference_b"] / medians["reference_a"]) if medians["reference_a"] else None
    record = {
        "schema": "ironmule.qmv_k3840_kernel.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "kernel_qualification",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "is a K=3840-specialised quantised matvec kernel bit-identical and faster",
        "model": args.model,
        "capture_manifest": str(args.capture / "manifest.json"),
        "capture_experiment": manifest["experiment_id"],
        "cases": len(cases),
        "bytes_per_round": round_bytes,
        "rounds_per_timing": args.rounds,
        "blocks": args.blocks,
        "compile_options": COMPILE_OPTIONS,
        "bit_identity": identity,
        "port_bit_identical": port_clean,
        "k3840_bit_identical": k3840_clean,
        "arm_block_ns": arms,
        "arm_median_ns": medians,
        "aa_ratio": aa_ratio,
        "paired": comparisons,
        "mlx_version": mx.__version__,
        "limits": [
            "one machine, one MLX build, bfloat16 4-bit group size 64",
            "a sweep is 21 real projections of one model, not a decode step",
            "the A/A arm prices the harness noise; a claim must clear it",
        ],
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    summary = {key: value for key, value in record.items()
               if key in ("cases", "bytes_per_round", "port_bit_identical",
                          "k3840_bit_identical", "arm_median_ns", "paired")}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
