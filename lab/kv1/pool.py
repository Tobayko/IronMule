"""KV1 pool probe: one shared KV pool (llama.cpp `kv_unified`) of `cells` cells for W sequences against W' split
sequences of 2,048 cells, flash attention, Qwen3-4B Q8_0 on this CPU. Full-load decode steps at a given context per
sequence (prompt `ctx` tokens), median of 3 x 16 steps. Usage: pool.py GGUF OUT.json"""
import json, statistics, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
gguf, out = sys.argv[1], sys.argv[2]
res = []
def run(seqs, seq_ctx, unified, ctxs, steps=16):
    e = GgmlEngine(gguf, seqs=seqs, seq_ctx=seq_ctx, speculate=False, flash_attn=True, kv_unified=unified)
    base = e.tokenize(CHECK_TEXT)
    for ctx in ctxs:
        prompt = (base * (ctx // len(base) + 1))[:ctx]
        times = []
        for _ in range(3):
            e._clear()
            rows = [(t, j, s, j == ctx - 1) for s in range(seqs) for j, t in enumerate(prompt)]
            for st in range(0, len(rows), e.seq_ctx):
                e._decode(rows[st:st + e.seq_ctx])
            t0 = time.perf_counter()
            for k in range(steps):
                e._decode([(prompt[k % ctx], ctx + k, s, True) for s in range(seqs)])
            times.append(time.perf_counter() - t0)
        step = statistics.median(times) / steps
        row = {"seqs": seqs, "cells": seqs * seq_ctx, "unified": unified, "ctx": ctx, "step_s": step, "tok_s": seqs / step}
        print(json.dumps(row), flush=True); res.append(row)
    e.close()
run(16, 2048, False, (64, 512))
run(32, 1024, True, (64, 512))   # 32 sequences sharing the 16 x 2,048 cells
run(32, 1024, False, (64, 512))  # the split layout at the same memory (32 x 1,024), for the attention cost
open(out, "w").write(json.dumps(res, indent=1))
