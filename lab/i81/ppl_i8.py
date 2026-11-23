# Perplexity on the gate's 16 chunks and the benchmark's prefill time, for one IRONMULE_I8 setting (read at load)
import json, os, sys, time
import torch
from ironmule.ggml_engine import GgmlEngine
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos, gate_lead
from ironmule_product import compare
G = '/models/Qwen3-32B-Q4_K_M.gguf'
tok = GgufTokenizer(G)
ids, bos = _text_ids(tok), gate_bos(tok)
e = GgmlEngine(G, seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
nll = []
for i in range(CHUNKS):
    chunk = ids[i * CHUNK_TOKENS:(i + 1) * CHUNK_TOKENS + 1]
    nll.append(e.chunk_nll(chunk, gate_lead(chunk, bos)))
pre = {}
for width in (32, 8):
    toks = sorted((e.tokenize(p) for p in compare.performance_prompts(0, width, max(width, 8))), key=len)
    best = 9e9
    for rep in range(3):
        e._clear(); torch.cuda.synchronize(); t = time.perf_counter()
        e._decode([(toks[s][j], j, s, j == len(toks[s]) - 1) for s in range(len(toks)) for j in range(len(toks[s]))])
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t)
    pre[width] = round(best, 3)
print("PPL " + json.dumps({"i8": os.environ.get("IRONMULE_I8", "0"), "max_k": os.environ.get("IRONMULE_I8_MAX_K", "0"),
                           "nll": nll, "prefill_s": pre}), flush=True)
e.close()
