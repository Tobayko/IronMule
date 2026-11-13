"""KV1 attention probe: full-load decode step of W sequences at context C, flash attention on and off, Qwen3-4B Q8_0
on this CPU; median of 3 x 8 steps. Usage: attn.py GGUF OUT.json"""
import json, os, statistics, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
gguf, out = sys.argv[1], sys.argv[2]
res = []
for fa in [x == "1" for x in os.environ.get("ATTN_FA", "1,0").split(",")]:
    e = GgmlEngine(gguf, seqs=16, seq_ctx=2048, speculate=False, flash_attn=fa)
    base = e.tokenize(CHECK_TEXT)
    for w, ctx in ((1, 64), (1, 1024), (16, 64), (16, 256), (16, 1024)):
        prompt = (base * (ctx // len(base) + 1))[:ctx]
        e._clear()
        for s in range(w):
            for st in range(0, ctx, 2048):
                e._decode([(t, j, s, j == ctx - 1) for j, t in enumerate(prompt)][st:st + 2048])
        times = []
        for r in range(3):
            t0 = time.perf_counter()
            for k in range(8):
                e._decode([(prompt[k % ctx], ctx + r * 8 + k, s, True) for s in range(w)])
            times.append(time.perf_counter() - t0)
        row = {"fa1": os.environ.get("FA1", "1"), "fa": fa, "seqs": w, "ctx": ctx, "step_s": statistics.median(times) / 8}
        print(json.dumps(row), flush=True); res.append(row)
    e.close()
open(out, "w").write(json.dumps(res, indent=1))
