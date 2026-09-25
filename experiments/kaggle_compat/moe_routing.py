"""OSS1: does a mixture-of-experts model's expert choice change with the compute dtype?

Usage: python moe_routing.py MODEL_ID REVISION TEXT_FILE OUT.json DTYPE CHUNK [CHUNK ...]

DTYPE is `bf16` (the checkpoint's own dtype) or an opt-in compute plan (`float32`,
`float16`). Chunks are sliced exactly as `quality.py` slices them (one encode of the whole
text, 512 tokens plus one target each), so a chunk index names the same tokens as in the
quality gate. Per chunk: the next-token NLL, and for every layer and position the sorted
top-k experts the router chose and the router's margin between the k-th and the (k+1)-th
logit. Written for mlx-lm's `gpt_oss`, whose router calls the module-level `mlx_topk`.
"""
import json
import sys
import time

import mlx.core as mx
from mlx_lm.models import gpt_oss

import ironmule
from ironmule.tune import load_engine

MODEL_ID, REVISION, TEXT, OUT, DTYPE = sys.argv[1:6]
CHUNKS = [int(value) for value in sys.argv[6:]]
CHUNK_TOKENS = 512

started = time.time()
engine, tokenizer = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION,
                                compute_dtype=None if DTYPE == "bf16" else DTYPE)
recorded = []
original_topk = gpt_oss.mlx_topk


def recording_topk(a, k, axis=-1):
    values, indices = original_topk(a, k, axis=axis)
    ordered = mx.sort(a.astype(mx.float32), axis=axis)
    recorded.append((mx.sort(indices, axis=axis), ordered[..., -k] - ordered[..., -k - 1]))
    return values, indices


gpt_oss.mlx_topk = recording_topk
ids = tokenizer.encode(open(TEXT, encoding="utf-8").read())
rows = []
for index in CHUNKS:
    chunk = ids[index * CHUNK_TOKENS: (index + 1) * CHUNK_TOKENS + 1]
    inputs, targets = mx.array([chunk[:-1]]), mx.array(chunk[1:])
    recorded.clear()
    logits = engine.model(inputs)[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.mean(lp[mx.arange(CHUNK_TOKENS), targets])
    mx.eval(nll, *[array for pair in recorded for array in pair])
    rows.append({
        "chunk": index, "nll": nll.item(),
        "experts": [indices[0].tolist() for indices, _ in recorded],
        "margins": [[round(v, 5) for v in margin[0].tolist()] for _, margin in recorded],
    })
    print(index, DTYPE, round(rows[-1]["nll"], 4), len(recorded), "layers", flush=True)
    del logits, lp
    mx.clear_cache()

bos = getattr(tokenizer, "bos_token_id", None)
with open(OUT, "w") as stream:
    json.dump({"schema": "ironmule.oss1-routing.v1", "model_id": MODEL_ID, "revision": REVISION,
               "dtype": DTYPE, "chunk_tokens": CHUNK_TOKENS, "bos_token_id": bos,
               "text_starts_with_bos": bool(ids) and ids[0] == bos, "rows": rows,
               "device": str(mx.default_device()), "seconds": time.time() - started,
               "performance_claim": False}, stream)
print(json.dumps({"dtype": DTYPE, "chunks": len(rows), "seconds": round(time.time() - started, 1)}))
