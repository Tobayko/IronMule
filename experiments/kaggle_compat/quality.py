"""PORT1-C quality: the same bf16 checkpoint computed in bf16 and in float32.

Usage: python quality.py MODEL_ID REVISION TEXT_FILE OUT.json [CHUNKS] [CHUNK_TOKENS]

Both precisions are loaded side by side and fed identical token chunks from a fixed text
(WikiText-2 raw test). Per chunk: next-token NLL of each, KL(bf16 || float32) and top-1
agreement. Perplexity ratio float32/bf16 gets a bootstrap interval over chunks. A precision
plan passes quality when the upper bound of that ratio stays below 1.005.
"""
import json
import math
import os
import random
import statistics as st
import sys
import time

import mlx.core as mx

import ironmule
from ironmule.tune import load_engine

MODEL_ID, REVISION, TEXT, OUT = sys.argv[1:5]
CHUNKS = int(sys.argv[5]) if len(sys.argv) > 5 else 16
CHUNK_TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 512

started = time.time()
# QUALITY_ONLY loads a single precision, because a checkpoint above about 6 GB does not fit
# twice on a 15 GB card. `float32` yields per-chunk NLL to compare against another device's
# float32 reference; `bf16` is the other half of that pair, so two processes on the same
# card can be compared where one process cannot hold both (PORT2: gpt-oss 20B, Mistral 24B).
# In single-precision mode `bf16` means the checkpoint's own dtype and anything else names
# an opt-in compute plan (`float32`, `float16`), so two processes form one comparable pair.
ONLY = os.environ.get("QUALITY_ONLY")
if ONLY and ONLY != "bf16":
    single, tokenizer = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION,
                                    compute_dtype=ONLY)
    bf16, fp32 = None, single
elif ONLY == "bf16":
    single, tokenizer = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION)
    bf16, fp32 = single, None
else:
    bf16, tokenizer = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION)
    fp32, _ = load_engine(MODEL_ID, ironmule.BASELINE, revision=REVISION,
                          compute_dtype="float32")
ids = tokenizer.encode(open(TEXT, encoding="utf-8").read())
rows = []
for index in range(CHUNKS):
    chunk = ids[index * CHUNK_TOKENS: (index + 1) * CHUNK_TOKENS + 1]
    if len(chunk) < CHUNK_TOKENS + 1:
        break
    inputs, targets = mx.array([chunk[:-1]]), mx.array(chunk[1:])
    positions = mx.arange(CHUNK_TOKENS)
    if ONLY == "bf16":
        logits = bf16.model(inputs)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.mean(lp[positions, targets])
        mx.eval(nll)
        rows.append({"nll_bf16": nll.item()})
        print(index, rows[-1], flush=True)
        del lp
        mx.clear_cache()
        continue
    logits32 = fp32.model(inputs)[0].astype(mx.float32)
    lp32 = logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)
    nll32 = -mx.mean(lp32[positions, targets])
    if ONLY:
        mx.eval(nll32)
        rows.append({f"nll_{ONLY}": nll32.item()})
    else:
        logits16 = bf16.model(inputs)[0].astype(mx.float32)
        lp16 = logits16 - mx.logsumexp(logits16, axis=-1, keepdims=True)
        nll16 = -mx.mean(lp16[positions, targets])
        kl = mx.mean(mx.sum(mx.exp(lp16) * (lp16 - lp32), axis=-1))
        top1 = mx.mean(mx.argmax(lp16, axis=-1) == mx.argmax(lp32, axis=-1))
        mx.eval(nll16, nll32, kl, top1)
        rows.append({"nll_bf16": nll16.item(), "nll_fp32": nll32.item(), "kl": kl.item(), "top1": top1.item()})
        del lp16
    print(index, rows[-1], flush=True)
    del lp32
    mx.clear_cache()

if ONLY:
    key = f"nll_{ONLY}"
    report = {"schema": "ironmule.port1-quality.v1", "model_id": MODEL_ID, "revision": REVISION,
              "text": TEXT, "chunks": len(rows), "chunk_tokens": CHUNK_TOKENS, "rows": rows, "only": ONLY,
              f"ppl_{ONLY}": math.exp(st.mean(r[key] for r in rows)),
              "device": str(mx.default_device()), "seconds": time.time() - started}
    with open(OUT, "w") as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    raise SystemExit(0)

rng = random.Random(20260915)


def ppl_ratio(sample):
    return math.exp(st.mean(r["nll_fp32"] for r in sample) - st.mean(r["nll_bf16"] for r in sample))


boot = sorted(ppl_ratio([rng.choice(rows) for _ in rows]) for _ in range(10000))
report = {"schema": "ironmule.port1-quality.v1", "model_id": MODEL_ID, "revision": REVISION,
          "text": TEXT, "chunks": len(rows), "chunk_tokens": CHUNK_TOKENS, "rows": rows,
          "ppl_bf16": math.exp(st.mean(r["nll_bf16"] for r in rows)),
          "ppl_fp32": math.exp(st.mean(r["nll_fp32"] for r in rows)),
          "ppl_ratio_fp32_over_bf16": ppl_ratio(rows),
          "ppl_ratio_ci": [boot[250], boot[9750]],
          "mean_kl_bf16_fp32": st.mean(r["kl"] for r in rows),
          "top1_agreement": st.mean(r["top1"] for r in rows),
          "quality_gate_upper_below_1_005": boot[9750] < 1.005,
          "device": str(mx.default_device()), "seconds": time.time() - started}
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
