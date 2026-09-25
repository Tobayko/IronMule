"""PERF1-K: does computing requests as one batch change what the model predicts?

Usage: python batch_gate.py MODEL_ID REVISION PLAN TEXT_FILE OUT_DIR KEY
       PLAN: bf16 | float32 | native (the product's compute_dtype)

One process, one model loaded through `load_engine` as the product loads it. 16 chunks of
512 tokens of TEXT_FILE, sliced and given BOS as `perf1.py nll` does; the mean next-token NLL
of each chunk one chunk at a time (batch 1) and eight equal-length chunks at a time (batch 8),
on the prefill path (one forward over the chunk) and the decode path (teacher-forced, one
token at a time through the cache). Equal lengths need no padding, so the pair isolates the
batch's arithmetic. Writes gate-KEY-single-PATH.json, gate-KEY-batch8-PATH.json and
gate-summary-KEY.json, the pair judged by `gate_summary.judge` (bound 1.005).
"""
import json
import math
import os
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import ironmule
from ironmule.tune import load_engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_summary import judge  # noqa: E402

MODEL_ID, REVISION, PLAN, TEXT, OUT, KEY = sys.argv[1:7]
CHUNKS, TOKENS, WIDTH = 16, 512, 8

engine, tokenizer = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION,
                                compute_dtype=None if PLAN == "bf16" else PLAN)
model = engine.model
with open(TEXT, encoding="utf-8") as stream:
    ids = tokenizer.encode(stream.read())
stride = (len(ids) - TOKENS - 1) // CHUNKS
bos = getattr(tokenizer, "bos_token_id", None)
chunks = []
for index in range(CHUNKS):
    seq = ids[index * stride:index * stride + TOKENS + 1]
    if bos is not None and seq[0] != bos:
        seq = [bos] + seq[:-1]
    chunks.append(seq)


def prefill(group):
    x, y = mx.array([seq[:-1] for seq in group]), mx.array([seq[1:] for seq in group])
    logits = model(x).astype(mx.float32)
    per = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
    return [[float(v) for v in row] for row in per.tolist()]


def decode(group):
    cache = make_prompt_cache(model)
    x = mx.array(group)
    values = [[] for _ in group]
    for t in range(TOKENS):
        logits = model(x[:, t:t + 1], cache=cache)[:, -1].astype(mx.float32)
        value = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, x[:, t + 1:t + 2], axis=-1)[:, 0]
        for row, v in zip(values, value.tolist()):
            row.append(float(v))
    return values


summary = {}
for path_mode, run in (("prefill", prefill), ("decode", decode)):
    results = {}
    for arm, width in (("single", 1), ("batch8", WIDTH)):
        started = time.time()
        per_chunk = []
        for start in range(0, CHUNKS, width):
            per_chunk.extend(run(chunks[start:start + width]))
            mx.clear_cache()
        nonfinite = sum(not math.isfinite(v) for row in per_chunk for v in row)
        results[arm] = {"arm": arm, "plan": PLAN, "path_mode": path_mode, "batch": width, "chunks": CHUNKS,
                        "chunk_tokens": TOKENS, "bos": bos, "chunk_nll": [sum(row) / len(row) for row in per_chunk],
                        "nonfinite": nonfinite, "seconds": round(time.time() - started, 1),
                        "model_id": MODEL_ID, "revision": REVISION, "performance_claim": False}
        with open(os.path.join(OUT, f"gate-{KEY}-{arm}-{path_mode}.json"), "w") as stream:
            json.dump(results[arm], stream, indent=1)
        print(path_mode, arm, round(sum(results[arm]["chunk_nll"]) / CHUNKS, 5), results[arm]["seconds"], flush=True)
    summary[path_mode] = judge(results["single"], results["batch8"])
with open(os.path.join(OUT, f"gate-summary-{KEY}.json"), "w") as stream:
    json.dump(summary, stream, indent=1)
print(json.dumps({path: {k: v for k, v in row.items() if k in ("ratio", "interval", "verdict")}
                  for path, row in summary.items()}))
