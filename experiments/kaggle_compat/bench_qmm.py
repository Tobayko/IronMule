"""PORT1 diagnostic: 4-bit quantized matvec cost by activation dtype on this device.

Gemma 3 4B shapes (hidden 2560, MLP 10240) by default, group size 64; `QMM_SHAPES` takes a
JSON list of `[out_features, in_features]` pairs instead, which is how PORT2 puts llama 3.1
8B's shapes next to Qwen 3 8B's. No performance claim: it locates where decode time goes,
it does not measure IronMule.
"""
import json
import os
import statistics as st
import time

import mlx.core as mx

SHAPES = json.loads(os.environ.get("QMM_SHAPES", "[[10240, 2560], [2560, 10240], [2560, 2560]]"))
rows = []
for dtype in (mx.bfloat16, mx.float16, mx.float32):
    for out_features, in_features in SHAPES:
        dense = mx.random.normal((out_features, in_features)).astype(dtype)
        q, s, b = mx.quantize(dense, group_size=64, bits=4)
        x = mx.random.normal((1, 1, in_features)).astype(dtype)
        mx.eval(q, s, b, x)

        def call():
            return sum(mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4).sum()
                       for _ in range(64))

        for _ in range(3):
            mx.eval(call())
        samples = []
        for _ in range(7):
            started = time.perf_counter()
            mx.eval(call())
            mx.synchronize()
            samples.append((time.perf_counter() - started) / 64 * 1e3)
        rows.append({"dtype": str(dtype), "shape": [out_features, in_features], "ms_per_call": st.median(samples)})
        print(rows[-1], flush=True)
print(json.dumps({"schema": "ironmule.port1-qmm-dtype.v1", "device": str(mx.default_device()),
                  "shapes": SHAPES, "rows": rows}))
