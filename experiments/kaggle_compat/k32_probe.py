"""PERF1-L: is PERF1's `k32` kernel bit-identical to MLX's own float32 quantised matvec?

Usage: PERF1_ARITH=free|pinned python k32_probe.py OUT.json

The float32 plan computes the bf16 checkpoint in float32 (`set_dtype`), so its matvecs take
float32 activations, scales and biases. `k32` is `perf1.py`'s row kernel reading those as
float32. Only a bit-identical kernel may speed the qualified plan up without a new gate.
Synthetic 4-bit, group-64 weights at the decode shapes of Mistral Small 3.2 24B (the plan's
one `doctor` recommendation) and Qwen 3 8B, 1 to 8 rows: `perf1.qmv` against
`mx.quantized_matmul`, bit equality and maximum difference.
"""
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf1  # noqa: E402

SHAPES = {"mistral-24b": [(5120, 4096), (5120, 1024), (4096, 5120), (5120, 32768), (32768, 5120)],
          "qwen3-8b": [(4096, 4096), (4096, 1024), (4096, 12288), (12288, 4096)]}
ROWS = (1, 2, 4, 8)

mx.random.seed(20260925)
results = []
for model, shapes in SHAPES.items():
    for k, n in shapes:
        w, scales, biases = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
        scales, biases = scales.astype(mx.float32), biases.astype(mx.float32)
        for rows in ROWS:
            x = mx.random.normal((1, rows, k)).astype(mx.bfloat16).astype(mx.float32)
            ours = perf1.qmv(x, w, scales, biases)
            theirs = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=64, bits=4)
            diff = mx.abs(ours - theirs)
            mx.eval(ours, theirs, diff)
            results.append({"model": model, "k": k, "n": n, "rows": rows,
                            "bit_equal": bool(mx.array_equal(ours, theirs).item()),
                            "max_abs_diff": diff.max().item(),
                            "max_rel_diff": (diff / (mx.abs(theirs) + 1e-30)).max().item()})
            print(results[-1], flush=True)
with open(sys.argv[1], "w") as stream:
    json.dump({"schema": "ironmule.perf1l-probe.v1", "arith": os.environ.get("PERF1_ARITH", "free"),
               "device": str(mx.default_device()), "results": results, "performance_claim": False},
              stream, indent=1)
