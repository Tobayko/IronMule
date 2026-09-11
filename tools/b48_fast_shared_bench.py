#!/usr/bin/env python3
"""Is one weight sweep for two requests worth anything on the aligned projections?

Same design as `B45K`, pointed at `qmv_fast` shapes instead. The reduction length is a
parameter now, because `K=4096` and `K=15360` are separate shape families.

Four arms, all doing the same work: two independent projections of two different real
activations through the same real weights.

* `library_x2` is the path in service: two `mx.quantized_matmul` calls, enqueued
  asynchronously and synchronised once, so the comparison is against already grouped
  calls rather than against serial execution.
* `library_x2_aa` is that arm against itself, pricing the harness.
* `mine_separate_x2` is two calls of the same kernel at width 1. It separates the
  kernel from the sharing: whatever it wins is not the shared load.
* `mine_shared_x2` is one call at width 2, one weight sweep for both requests.

Preregistered: width 2 only, width 4 is not attempted unless width 2 clears its gate.
Success needs `mine_shared_x2 / library_x2` at or below 0.90. Bit identity is checked
outside the timed region and any difference locks the variant.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b48_fast_shared_kernel import COMPILE_OPTIONS, run_fast  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE = 0.95


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def _resolve(root, path: str):
    node = root
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--target-k", type=int, required=True)
    parser.add_argument("--capture-a", type=Path, required=True)
    parser.add_argument("--capture-b", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    first = json.loads((args.capture_a / "manifest.json").read_text())
    second = json.loads((args.capture_b / "manifest.json").read_text())
    model, _ = load(args.model)
    language = getattr(model, "language_model", model)

    by_path_b = {entry["module_path"]: entry for entry in second["activations"]}
    cases, skipped_same = [], []
    for entry in first["activations"]:
        path = entry["module_path"]
        other = by_path_b.get(path)
        if other is None:
            continue
        if other["sha256"] == entry["sha256"]:
            skipped_same.append(path)  # the brief forbids using one vector twice
            continue
        module = _resolve(language, path)
        x0 = mx.load(str(args.capture_a / entry["file"]))
        x1 = mx.load(str(args.capture_b / other["file"]))
        mx.eval(x0, x1)
        cases.append({
            "path": path, "module": module, "x0": x0, "x1": x1,
            "out_features": int(module.weight.shape[0]),
            "bytes": int(module.weight.nbytes + module.scales.nbytes + module.biases.nbytes),
        })

    def library(case):
        module = case["module"]
        return [
            mx.quantized_matmul(vector, module.weight, module.scales, module.biases,
                                transpose=True, group_size=module.group_size,
                                bits=module.bits)
            for vector in (case["x0"], case["x1"])
        ]

    def mine_separate(case):
        module = case["module"]
        return [
            run_fast(module.weight, module.scales, module.biases, [vector],
                     case["out_features"], args.target_k)[0]
            for vector in (case["x0"], case["x1"])
        ]

    def mine_shared(case):
        module = case["module"]
        return run_fast(module.weight, module.scales, module.biases,
                        [case["x0"], case["x1"]], case["out_features"], args.target_k)

    variants = {
        "library_x2": library,
        "library_x2_aa": library,
        "mine_separate_x2": mine_separate,
        "mine_shared_x2": mine_shared,
    }

    # Gate one, outside every timed region.
    identity = []
    for case in cases:
        expected = [bytes(memoryview(value)) for value in library(case)]
        mx.eval(expected)
        rows = {}
        for name in ("mine_separate_x2", "mine_shared_x2"):
            got = variants[name](case)
            mx.eval(got)
            rows[name] = [bytes(memoryview(value)) for value in got] == expected
        identity.append({"module_path": case["path"], **rows})
    clean = all(all(v for k, v in row.items() if k != "module_path") for row in identity)
    print(json.dumps({"cases": len(cases), "skipped_identical_vectors": len(skipped_same),
                      "bit_identical": clean}, sort_keys=True))
    if not clean:
        args.out.write_text(json.dumps({"state": "LOCKED", "identity": identity}, indent=2))
        return 1

    def sweep(name):
        outputs = []
        for case in cases:
            outputs.extend(variants[name](case))
        mx.eval(outputs)

    for name in variants:
        for _ in range(2):
            sweep(name)

    before = _vm()
    arms = {name: [] for name in variants}
    order = list(variants)
    for block in range(args.blocks):
        rotation = order[block % len(order):] + order[: block % len(order)]
        for name in rotation:
            began = time.perf_counter_ns()
            for _ in range(args.rounds):
                sweep(name)
            arms[name].append((time.perf_counter_ns() - began) / args.rounds)
    after = _vm()
    paging = {key: after.get(key, 0) - value for key, value in before.items()}

    def interval(numerator, denominator):
        ratios = [n / d for n, d in zip(arms[numerator], arms[denominator])]
        rng = random.Random(20260909)
        draws = sorted(median([ratios[rng.randrange(len(ratios))] for _ in ratios])
                       for _ in range(10000))
        return {
            "median": median(ratios), "min": min(ratios), "max": max(ratios),
            "ci95_low": draws[int(0.025 * len(draws))],
            "ci95_high": draws[int(0.975 * len(draws)) - 1],
            "blocks_below_gate": sum(1 for value in ratios if value <= GATE),
        }

    shared = interval("mine_shared_x2", "library_x2")
    aa = interval("library_x2_aa", "library_x2")
    separate = interval("mine_separate_x2", "library_x2")
    sharing_only = interval("mine_shared_x2", "mine_separate_x2")
    aa_passes = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
    meets_gate = shared["ci95_high"] <= GATE
    decision = ("KERNEL GO" if meets_gate and aa_passes and paging.get("Swapouts", 0) == 0
                else ("BLOCKED" if paging.get("Swapouts", 0) else "KERNEL NO-GO"))

    record = {
        "schema": "ironmule.fast_shared_weight_kernel.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "kernel_qualification",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does one weight sweep for two requests help on an aligned reduction length",
        "target_k": args.target_k,
        "decision": decision,
        "model": args.model,
        "gate": GATE,
        "preregistered": {
            "widths": [2], "target_k": args.target_k,
            "width_4_rule": "attempted only if width 2 clears the gate",
            "blocks": args.blocks,
            "rounds_per_block": args.rounds,
            "arms": list(variants),
            "order": "rotated so every arm meets every position",
            "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
            "success_rule": "mine_shared_x2 / library_x2 upper bound at or below 0.90",
            "abort_rule": "swapouts during the run make it not evaluable",
            "budget": "one run, no extension",
        },
        "cases": len(cases),
        "skipped_identical_vectors": skipped_same,
        "bytes_per_sweep": sum(case["bytes"] for case in cases),
        "bit_identity": identity,
        "bit_identical": clean,
        "arm_median_ns": {name: median(values) for name, values in arms.items()},
        "arm_block_ns": arms,
        "shared_over_library": shared,
        "separate_over_library": separate,
        "sharing_effect_shared_over_separate": sharing_only,
        "aa_null_control": aa,
        "aa_null_control_passes": aa_passes,
        "meets_gate": meets_gate,
        "paging_during_run": paging,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "compile_options": COMPILE_OPTIONS,
        "limits": [
            "logical bytes per second are not measured DRAM traffic",
            "a sweep is 18 real projections, not a decode step",
            "one machine, one MLX build, bfloat16 4-bit group size 64",
        ],
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key in ("decision", "meets_gate", "shared_over_library",
                                 "separate_over_library",
                                 "sharing_effect_shared_over_separate",
                                 "aa_null_control", "aa_null_control_passes",
                                 "arm_median_ns", "paging_during_run")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
