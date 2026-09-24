# Diagnostic only (Mac, Metal): Gemma perplexity on WikiText-2 raw test, 16 x 512, bf16, with
# chunks cut as quality.py and perf1.py nll cut them (one encode; BOS only where the encode put
# it) and with the model's BOS on every chunk. Run inline on 2026-09-24 for
# mlx-community/gemma-3-4b-it-4bit and mlx-community/gemma-4-e2b-it-4bit (latest revisions).
# Usage: python gemma_bos_perplexity.py MODEL_ID WIKITEXT_TXT
import math
import sys

import mlx.core as mx
from mlx_lm import load

text = open(sys.argv[2]).read()
model, tok = load(sys.argv[1])
ids = tok.encode(text)
print("first id is bos:", ids[0] == tok.bos_token_id)


def ppl(bos):
    out = []
    for i in range(16):
        chunk = ids[i * 512:(i + 1) * 512 + 1]
        if bos and chunk[0] != tok.bos_token_id:
            chunk = [tok.bos_token_id] + chunk[:-1]
        logits = model(mx.array([chunk[:-1]])).astype(mx.float32)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        out.append(float(-mx.take_along_axis(lp, mx.array(chunk[1:])[:, None], axis=-1).mean()))
    return math.exp(sum(out) / len(out)), [round(v, 2) for v in out[:6]]


print("quality.py chunking (no BOS after chunk 0):", ppl(False))
print("BOS on every chunk:", ppl(True))
