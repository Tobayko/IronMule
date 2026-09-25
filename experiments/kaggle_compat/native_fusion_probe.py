"""PERF1-N: does fusing projections change the `native` plan's outputs, and on which path?

Usage: python native_fusion_probe.py OUT.json

`fuse_projections` concatenates the quantised q/k/v and gate/up weights along the output
dimension, and `native` is installed after fusion, so a fused module runs one matmul with a
wider N. Synthetic 4-bit, group-64 weights at Qwen 3 14B's shapes (hidden 5120; q 5120,
k and v 1024; gate and up 17408 each), bf16 activations: per row count, IronMule's
`cuda_native.matmul` on the fused weight against the concatenation of the separate parts.
Up to MAX_ROWS rows run IronMule's kernel, more rows the float16 GEMM.
"""
import json
import sys

import mlx.core as mx

from ironmule import cuda_native

HIDDEN = 5120
PARTS = {"qkv": (5120, 1024, 1024), "gate_up": (17408, 17408)}
ROWS = (1, 2, 4, 8, 9, 16, 64, 512)

mx.random.seed(20260925)
results = []
for name, outputs in PARTS.items():
    parts = []
    for n in outputs:
        w, scales, biases = mx.quantize((mx.random.normal((n, HIDDEN)) * 0.02).astype(mx.bfloat16),
                                        group_size=64, bits=4)
        parts.append((w, scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)))
    fused = tuple(mx.concatenate([part[i] for part in parts], axis=0) for i in range(3))
    for rows in ROWS:
        x = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
        separate = mx.concatenate([cuda_native.matmul(x, *part) for part in parts], axis=-1)
        together = cuda_native.matmul(x, *fused)
        diff = mx.abs(separate.astype(mx.float32) - together.astype(mx.float32))
        mx.eval(separate, together, diff)
        results.append({"projection": name, "rows": rows,
                        "path": "kernel" if rows <= cuda_native.MAX_ROWS else "float16_gemm",
                        "bit_equal": bool(mx.array_equal(separate, together).item()),
                        "max_abs_diff": diff.max().item(),
                        "differing_share": (diff > 0).astype(mx.float32).mean().item()})
        print(results[-1], flush=True)
with open(sys.argv[1], "w") as stream:
    json.dump({"schema": "ironmule.perf1n-probe.v1", "device": str(mx.default_device()), "results": results,
               "performance_claim": False}, stream, indent=1)
