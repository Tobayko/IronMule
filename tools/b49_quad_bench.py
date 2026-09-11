#!/usr/bin/env python3
"""Is one weight sweep for four requests better than two sweeps for two?

The comparison that matters is `quad` against `two_pairs`: both share, and the only
difference is whether four activations ride one load or two rides each. `library_x4` is
the unshared baseline, four asynchronous calls synchronised once, and an A/A arm prices
the harness.

Four different real decode activations per projection, from four documented model runs.
Real weights well past the cache, rotated so the same matrix is never hammered.
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
from b49_quad_kernel import COMPILE_OPTIONS, TARGET_K, run_shared  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE = 0.95
KEYS = ("a", "b", "c", "d")


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
    parser.add_argument("--captures", type=Path, required=True,
                        help="directory holding b49_cap_a .. b49_cap_d")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifests = {k: json.loads((args.captures / f"b49_cap_{k}" / "manifest.json").read_text())
                 for k in KEYS}
    model, _ = load(args.model)
    language = getattr(model, "language_model", model)

    by_path = {k: {e["module_path"]: e for e in manifests[k]["activations"]} for k in KEYS}
    common = set.intersection(*(set(by_path[k]) for k in KEYS))
    cases, skipped = [], []
    for path in sorted(common):
        digests = {by_path[k][path]["sha256"] for k in KEYS}
        if len(digests) != len(KEYS):
            skipped.append(path)  # the brief forbids reusing one vector
            continue
        module = _resolve(language, path)
        vectors = [mx.load(str(args.captures / f"b49_cap_{k}" / by_path[k][path]["file"]))
                   for k in KEYS]
        mx.eval(*vectors)
        cases.append({"path": path, "module": module, "vectors": vectors,
                      "out_features": int(module.weight.shape[0]),
                      "bytes": int(module.weight.nbytes + module.scales.nbytes
                                   + module.biases.nbytes)})

    def library(case):
        m = case["module"]
        return [mx.quantized_matmul(v, m.weight, m.scales, m.biases, transpose=True,
                                    group_size=m.group_size, bits=m.bits)
                for v in case["vectors"]]

    def two_pairs(case):
        m = case["module"]
        out = []
        for start in (0, 2):
            out.extend(run_shared(m.weight, m.scales, m.biases,
                                  case["vectors"][start:start + 2], case["out_features"]))
        return out

    def quad(case):
        m = case["module"]
        return run_shared(m.weight, m.scales, m.biases, case["vectors"], case["out_features"])

    variants = {"library_x4": library, "library_x4_aa": library,
                "two_pairs": two_pairs, "quad": quad}

    identity = []
    for case in cases:
        expected = [bytes(memoryview(v)) for v in library(case)]
        mx.eval(expected)
        row = {"module_path": case["path"]}
        for name in ("two_pairs", "quad"):
            got = variants[name](case)
            mx.eval(got)
            row[name] = [bytes(memoryview(v)) for v in got] == expected
        identity.append(row)
    clean = all(row[name] for row in identity for name in ("two_pairs", "quad"))
    print(json.dumps({"cases": len(cases), "skipped_shared_vectors": len(skipped),
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
        return {"median": median(ratios), "min": min(ratios), "max": max(ratios),
                "ci95_low": draws[int(0.025 * len(draws))],
                "ci95_high": draws[int(0.975 * len(draws)) - 1],
                "blocks_under_gate": sum(1 for v in ratios if v < GATE)}

    quad_over_pairs = interval("quad", "two_pairs")
    quad_over_library = interval("quad", "library_x4")
    pairs_over_library = interval("two_pairs", "library_x4")
    aa = interval("library_x4_aa", "library_x4")
    aa_passes = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
    helps = quad_over_pairs["ci95_high"] < 1.0

    record = {
        "schema": "ironmule.quad_shared_kernel.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "kernel_qualification",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does one sweep for four beat two sweeps for two, at K=3840",
        "decision": ("KERNEL GO" if helps and aa_passes and paging.get("Swapouts", 0) == 0
                     else ("BLOCKED" if paging.get("Swapouts", 0) else "KERNEL NO-GO")),
        "model": args.model,
        "target_k": TARGET_K,
        "cases": len(cases),
        "skipped_shared_vectors": skipped,
        "bytes_per_sweep": sum(c["bytes"] for c in cases),
        "bit_identity": identity,
        "bit_identical": clean,
        "arm_median_ns": {name: median(values) for name, values in arms.items()},
        "arm_block_ns": arms,
        "quad_over_two_pairs": quad_over_pairs,
        "quad_over_library": quad_over_library,
        "two_pairs_over_library": pairs_over_library,
        "aa_null_control": aa,
        "aa_null_control_passes": aa_passes,
        "quad_helps_over_pairs": helps,
        "paging_during_run": paging,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "compile_options": COMPILE_OPTIONS,
        "limits": [
            "a sweep is 18 real projections, not a decode step",
            "register pressure is not read from a counter; Metal exposes none here",
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
                      if key in ("decision", "quad_over_two_pairs", "quad_over_library",
                                 "two_pairs_over_library", "aa_null_control_passes",
                                 "arm_median_ns", "paging_during_run")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
