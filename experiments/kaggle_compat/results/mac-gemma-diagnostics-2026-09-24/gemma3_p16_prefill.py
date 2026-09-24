# Diagnostic only (Mac, Metal): does the native plan's float16 prefill overflow on Gemma 3 12B?
# Multi-row 4-bit matmuls are replaced by cuda_native.matmul's prefill arithmetic
# (float16 input, float16-dequantised weight, float16 GEMM, cast back); NLL on WikiText-2 chunks
# through the prefill path against the bf16 checkpoint as loaded.
import json
import math
import sys

import mlx.core as mx
from mlx_lm import load

CHUNKS, TOKENS = int(sys.argv[1]), 512
text = open(sys.argv[2]).read()  # WikiText-2 raw test as one text file (run: a local scratch copy)
model, tok = load("mlx-community/gemma-3-12b-it-4bit")
ids = tok.encode(text)
_orig = mx.quantized_matmul
stats = {"calls": 0, "x_max": 0.0, "y_max": 0.0, "nonfinite": 0}


def p16(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode="affine", **kw):
    if x.size <= 8 * x.shape[-1] or biases is None or bits != 4 or scales.dtype != mx.bfloat16:
        return _orig(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, mode=mode, **kw)
    dense = mx.dequantize(w, scales.astype(mx.float16), biases.astype(mx.float16), group_size=group_size, bits=bits)
    y16 = mx.matmul(x.astype(mx.float16), dense.T)
    mx.eval(y16)
    stats["calls"] += 1
    stats["x_max"] = max(stats["x_max"], float(mx.max(mx.abs(x.astype(mx.float32)))))
    finite = mx.isfinite(y16)
    stats["nonfinite"] += int((~finite).sum())
    stats["y_max"] = max(stats["y_max"], float(mx.max(mx.where(finite, mx.abs(y16), 0).astype(mx.float32))))
    return y16.astype(x.dtype)


def nll():
    out = []
    for c in range(CHUNKS):
        chunk = mx.array([tok.bos_token_id] + ids[1 + c * TOKENS:1 + (c + 1) * TOKENS])[None]
        logits = model(chunk[:, :-1]).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out.append(float(-mx.take_along_axis(lp, chunk[:, 1:, None], axis=-1).mean()))
    return out


def p32(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode="affine", **kw):
    if x.size <= 8 * x.shape[-1] or biases is None or bits != 4 or scales.dtype != mx.bfloat16:
        return _orig(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, mode=mode, **kw)
    dense = mx.dequantize(w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=group_size, bits=bits)
    return mx.matmul(x.astype(mx.float32), dense.T).astype(x.dtype)


stock = nll()
mx.quantized_matmul = p16
cand = nll()
mx.quantized_matmul = p32
ref32 = nll()
report = {"stock_nll": stock, "p16_nll": cand, "p32_nll": ref32,
          "p16_vs_p32_abs": [abs(a - b) for a, b in zip(cand, ref32)],
          "stock_vs_p32_abs": [abs(a - b) for a, b in zip(stock, ref32)], **stats, "fp16_max": 65504}
print(json.dumps(report, indent=1))
