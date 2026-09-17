"""PORT2-B: which op makes `float32` cost llama 3.1 what it pays every other family?

Usage: python dtype_ops.py MODEL_ID REVISION OUT.json

Llama 3.1 8B runs at 1.48-1.54 of stock under `compute_dtype="float32"` while Gemma 3,
Qwen 3, Qwen 3.5, gpt-oss and Mistral 3 all land between 0.28 and 0.55. The 4-bit matvec
diagnostic already excluded the obvious cause: at llama's own projection shapes
`float32/bfloat16` is 0.577, and at Qwen 3 8B's — whose attention shapes are identical —
0.580. Both predict a gain.

Two differences survive that exclusion. llama 3.1 builds a `Llama3RoPE`, which hands
`mx.fast.rope` its own `freqs` array instead of a base, and llama's `lm_head` is untied over
128256 rows where Qwen 3 ties its embedding. This times exactly those two, on the real
loaded model, per dtype. Run it on llama and on Qwen 3 8B; the control is what makes an
answer an answer.
"""
import json
import statistics as st
import sys
import time

import mlx.core as mx

import ironmule
from ironmule.runtime import _project, _text, _trunk
from ironmule.tune import load_engine

MODEL_ID, REVISION, OUT = sys.argv[1:4]
REPEATS = 64
SAMPLES = 7


def timed(call):
    for _ in range(3):
        mx.eval(call())
    samples = []
    for _ in range(SAMPLES):
        started = time.perf_counter()
        mx.eval(call())
        mx.synchronize()
        samples.append((time.perf_counter() - started) / REPEATS * 1e3)
    return round(st.median(samples), 5)


report = {"schema": "ironmule.port2-dtype-ops.v1", "model_id": MODEL_ID, "revision": REVISION,
          "device": str(mx.default_device()), "dtypes": {}, "performance_claim": False}
for dtype in (None, "float32", "float16"):
    row = {}
    try:
        engine, _ = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION,
                                compute_dtype=dtype)
        trunk, text = _trunk(engine.model), _text(engine.model)
        attention = trunk.layers[0].self_attn
        rope = getattr(attention, "rope", None)
        row["rope_class"] = type(rope).__name__ if rope is not None else None
        row["rope_has_own_freqs"] = getattr(rope, "_freqs", None) is not None
        row["tied_embeddings"] = bool(getattr(text, "tie_word_embeddings", False))
        row["hidden"] = int(trunk.embed_tokens.weight.shape[-1]) if hasattr(
            trunk.embed_tokens, "weight") else None

        heads = attention.n_heads
        # `scale ** -2` rounds to 127 for a 128-wide head, which `mx.fast.rope` refuses.
        # Read the width off a real projection instead of reconstructing it from a float.
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            projection = getattr(attention, "q_proj", None) or attention.qkv_proj
            out_features = (projection.scales.shape[0] if hasattr(projection, "scales")
                            else projection.weight.shape[0])
            head_dim = out_features // heads
        row["head_dim"] = int(head_dim)
        queries = mx.zeros((1, heads, 1, int(head_dim)), dtype=getattr(mx, dtype) if dtype else None)
        hidden = mx.zeros((1, 1, trunk.norm.weight.shape[-1]),
                          dtype=getattr(mx, dtype) if dtype else None)
        mx.eval(queries, hidden)

        if rope is not None:
            row["rope_ms"] = timed(lambda: sum(rope(queries, offset=i).sum()
                                               for i in range(REPEATS)))
        row["lm_head_ms"] = timed(lambda: sum(_project(engine.model, hidden).sum()
                                              for _ in range(REPEATS)))
        row["norm_ms"] = timed(lambda: sum(trunk.norm(hidden).sum() for _ in range(REPEATS)))
        row["peak_memory_bytes"] = int(mx.get_peak_memory())
    except Exception as exc:  # noqa: BLE001 — an unsupported dtype is a result
        row["error"] = f"{type(exc).__name__}: {exc}"
    report["dtypes"][str(dtype)] = row
    print(str(dtype), json.dumps(row, default=str), flush=True)
    mx.clear_cache()

base = report["dtypes"].get("None", {})
for name, row in report["dtypes"].items():
    if name == "None" or "error" in row:
        continue
    row["vs_bf16"] = {key: round(row[key] / base[key], 4)
                      for key in ("rope_ms", "lm_head_ms", "norm_ms")
                      if key in row and base.get(key)}
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1, default=str)
print(json.dumps(report["dtypes"], indent=1, default=str))
