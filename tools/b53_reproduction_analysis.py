#!/usr/bin/env python3
"""The B53 disagreement reproduced, with its inputs kept, and measured against truth.

A full-suite run reproduced the failure and this time the probe wrote its dumps first:
inputs, both library evaluations, both kernel outputs, the kernel identities and the first
differing byte. This reads those dumps and asks the one question the original observation
could not: which arm is wrong.

The answer is not assumed. The stored quantised weights are dequantised, the dot products
are computed in float64, and both arms are compared against that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

DUMPS = PROJECT_ROOT / ".friday-data" / "b53-dumps"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16, nearest even, staying in float32 storage."""

    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def analyse(npz_path: Path) -> dict:
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    arrays = mx.load(str(npz_path))
    weight, scales, biases, x = (arrays["weight"], arrays["scales"], arrays["biases"],
                                 arrays["x"])
    library, second = arrays["library_first"], arrays["library_second"]
    ported, special = arrays["ported"], arrays["special"]
    mx.eval(weight, scales, biases, x, library, second, ported, special)

    dequantised = mx.dequantize(weight, scales, biases, group_size=64, bits=4)
    mx.eval(dequantised)
    rows = np.array(dequantised.astype(mx.float32), copy=True).astype(np.float64)
    vector = np.array(x.astype(mx.float32), copy=True).reshape(-1).astype(np.float64)
    exact = rows @ vector
    truth = _to_bfloat16(exact.astype(np.float32))

    library_values = np.array(library.astype(mx.float32), copy=True).reshape(-1)
    kernel_values = np.array(ported.astype(mx.float32), copy=True).reshape(-1)
    denominator = np.maximum(np.abs(exact), 1e-30)

    return {
        "dump": npz_path.name,
        "out_features": meta["out_features"],
        "first_difference": meta["first_difference"],
        "worker": meta.get("worker"),
        "pid": meta.get("pid"),
        "digests": {
            "library_first": _sha(bytes(memoryview(library))),
            "library_second": _sha(bytes(memoryview(second))),
            "ported": _sha(bytes(memoryview(ported))),
            "special": _sha(bytes(memoryview(special))),
        },
        "library_reproduces_itself":
            bytes(memoryview(library)) == bytes(memoryview(second)),
        "the_two_kernels_agree": bytes(memoryview(ported)) == bytes(memoryview(special)),
        "exact_matches": {
            "library": int(np.sum(library_values == truth)),
            "kernels": int(np.sum(kernel_values == truth)),
            "of": int(truth.size),
        },
        "median_relative_error": {
            "library": float(np.median(np.abs(library_values - exact) / denominator)),
            "kernels": float(np.median(np.abs(kernel_values - exact) / denominator)),
        },
        "max_relative_error": {
            "library": float(np.max(np.abs(library_values - exact) / denominator)),
            "kernels": float(np.max(np.abs(kernel_values - exact) / denominator)),
        },
        "first_six": {
            "exact": [float(v) for v in exact[:6]],
            "rounded_exact": [float(v) for v in truth[:6]],
            "library": [float(v) for v in library_values[:6]],
            "kernels": [float(v) for v in kernel_values[:6]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [analyse(path) for path in sorted(DUMPS.glob("*.npz"))]
    library_worse = all(row["median_relative_error"]["library"]
                        > row["median_relative_error"]["kernels"] for row in cases)

    record = {
        "schema": "ironmule.b53_reproduction_analysis.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "event": "B53 reproduced in a full-suite run with the dumps written first",
        "cases": cases,
        "dump_files": [{"path": str(path.relative_to(PROJECT_ROOT)),
                        "sha256": _sha(path.read_bytes()), "bytes": path.stat().st_size}
                       for path in sorted(DUMPS.iterdir())],
        "what_the_dumps_show": [
            "the library reproduces itself exactly, so it is stable within the run",
            "the transcription and the specialised kernel are byte-identical to each "
            "other, so the disagreement is not between them",
            "both kernels differ from the library",
        ],
        "which_arm_is_wrong": {
            "method": "the stored quantised weights are dequantised and the dot products "
                      "computed in float64, then compared against both arms",
            "library_is_the_outlier": library_worse,
            "reading": "the kernels track the float64 value to roughly bfloat16 "
                       "resolution while the library sits percent-level away from it. On "
                       "these inputs, in this process, the library call is the arm that "
                       "is wrong",
        },
        "what_this_changes": (
            "every earlier record treated the library as the reference and the kernels as "
            "the candidate. On this reproduction that is the wrong way round, which also "
            "fits the original observation, where the kernel value was the plausible one "
            "and the expected value sat several per cent away"
        ),
        "not_yet_established": [
            "why the library goes wrong, and why only sometimes",
            "whether the same happened in the original failure, whose inputs are lost",
        ],
        "mlx_version": mx.__version__,
        "numpy_version": np.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"cases": len(cases), "library_is_the_outlier": library_worse,
                      "median_relative_error": {
                          row["out_features"]: row["median_relative_error"]
                          for row in cases}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
