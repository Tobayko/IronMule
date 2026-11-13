"""Shared prefix: full-load decode steps of N sequences that each hold a P-token shared prefix and T own tokens;
split = every sequence its own copy (llama.cpp's default, FA1 per row); unified = one pool, the prefix cells shared
by llama_memory_seq_cp (FA1u). Median of 3 x 8 steps. Usage: share_speed.py GGUF N P T"""
import json, statistics, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
gguf, n, P, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for mode in ("split", "unified"):
    e = GgmlEngine(gguf, seqs=n, seq_ctx=1024, speculate=False, flash_attn=True, kv_unified=(mode == "unified"))
    base = e.tokenize(CHECK_TEXT * 200)
    prefix = base[:P]
    e._clear()
    t0 = time.perf_counter()
    if mode == "unified":
        for st in range(0, P, 512):
            e._decode([(t, j, 0, False) for j, t in enumerate(prefix)][st:st + 512])
        for s in range(1, n):
            e.L.llama_memory_seq_cp(e.memory, 0, s, 0, P)
    else:
        for s in range(n):
            e._decode([(t, j, s, False) for j, t in enumerate(prefix)])
    rows = [(base[P + (7 * s + j) % 500], P + j, s, False) for s in range(n) for j in range(T)]
    for st in range(0, len(rows), 1024):
        e._decode(rows[st:st + 1024])
    setup = time.perf_counter() - t0
    times = []
    for r in range(3):
        t1 = time.perf_counter()
        for k in range(8):
            e._decode([(base[k], P + T + r * 8 + k, s, True) for s in range(n)])
        times.append(time.perf_counter() - t1)
    step = statistics.median(times) / 8
    print(json.dumps({"mode": mode, "n": n, "prefix": P, "own": T, "step_s": round(step, 4), "tok_s": round(n / step, 1),
                      "setup_s": round(setup, 1)}), flush=True)
    e.close()
