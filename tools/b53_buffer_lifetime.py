#!/usr/bin/env python3
"""Does the kernel's uninitialised output ever keep a value it did not write?

The recorded B53 disagreement is far too large for a reordered float32 sum, so the two
arms cannot have read the same bytes. `tools/b42_qmv_kernel.py` states that the output is
deliberately not pre-initialised. This drives the allocator towards handing the kernel a
recently freed buffer whose contents are known, and then looks for those contents in the
result. A single sentinel in an output proves the mechanism; nothing else here does.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b42_qmv_kernel import K3840, PORT, run  # noqa: E402
from b52_automatic_selection import write_once  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = 12345.0            # exactly representable in bfloat16: 0x4640 -> 12352.0
WIDTHS = (2048, 4096, 15360)


def _sentinel_bits() -> int:
    """The bfloat16 bit pattern a sentinel-filled buffer actually holds."""

    as_bf16 = mx.array([SENTINEL], dtype=mx.bfloat16)
    mx.eval(as_bf16)
    return struct.unpack("<H", bytes(memoryview(as_bf16)))[0]


def _poison(out_features: int, count: int) -> None:
    """Fill and free `count` buffers of the output shape, so the pool holds them."""

    scratch = [mx.full((1, out_features), SENTINEL, dtype=mx.bfloat16)
               for _ in range(count)]
    mx.eval(scratch)
    del scratch
    gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--buffers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pattern = _sentinel_bits()
    findings: dict = {}
    mismatches: list[dict] = []
    sentinel_hits: list[dict] = []

    for out_features in WIDTHS:
        checked = 0
        for iteration in range(args.iterations):
            dense = mx.random.normal((out_features, 3840)).astype(mx.bfloat16)
            weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
            x = mx.random.normal((1, 3840)).astype(mx.bfloat16)
            mx.eval(weight, scales, biases, x)

            _poison(out_features, args.buffers)

            reference = mx.quantized_matmul(x, weight, scales, biases, transpose=True,
                                            group_size=64, bits=4)
            ported = run(PORT, weight, scales, biases, x, out_features, 3840)
            special = run(K3840, weight, scales, biases, x, out_features, 3840)
            mx.eval(reference, ported, special)

            expected = bytes(memoryview(reference))
            for arm, produced in (("port", bytes(memoryview(ported))),
                                  ("k3840", bytes(memoryview(special)))):
                checked += 1
                words = struct.unpack(f"<{len(produced) // 2}H", produced)
                hits = [index for index, word in enumerate(words) if word == pattern]
                if hits:
                    sentinel_hits.append({"arm": arm, "out_features": out_features,
                                          "iteration": iteration,
                                          "positions": hits[:16], "count": len(hits)})
                if produced != expected:
                    first = next(i for i, (a, b) in enumerate(zip(produced, expected))
                                 if a != b)
                    mismatches.append({"arm": arm, "out_features": out_features,
                                       "iteration": iteration, "first_index": first,
                                       "got": produced[first], "expected": expected[first]})
            if sentinel_hits or mismatches:
                break
        findings[str(out_features)] = {"comparisons": checked}
        if sentinel_hits or mismatches:
            break

    record = {
        "schema": "ironmule.b53_buffer_lifetime.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "diagnosis",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the kernel's uninitialised output ever retain a value it did "
                    "not write",
        "sentinel_value": SENTINEL,
        "sentinel_bfloat16_bits": pattern,
        "buffers_poisoned_per_iteration": args.buffers,
        "iterations_per_width": args.iterations,
        "per_width": findings,
        "sentinel_hits": sentinel_hits,
        "mismatches": mismatches,
        "mechanism_shown": bool(sentinel_hits),
        "verdict": ("MECHANISM SHOWN" if sentinel_hits
                    else "MISMATCH WITHOUT SENTINEL" if mismatches
                    else "MECHANISM NOT SHOWN"),
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"verdict": record["verdict"], "per_width": findings,
                      "sentinel_hits": len(sentinel_hits),
                      "mismatches": len(mismatches)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
