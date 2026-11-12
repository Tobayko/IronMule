"""E1 breakdown: per-op compute and barrier-wait time inside llama_decode, via the ROW1-instrumented ggml copy."""
import os, sys, time, ctypes
sys.path.insert(0, "/home/user/IronMule")
import ironmule.ggml_engine as ge
gguf = sys.argv[1]
ge.GgmlEngine._warm_up = lambda self: None
eng = ge.GgmlEngine(gguf, speculate=False, seqs=32, seq_ctx=256, max_rows=64, kv_f32=os.environ.get("ROW1_KVF32") == "1", flash_attn=os.environ.get("ROW1_FA") == "1")
L = eng.L
cpu = ctypes.CDLL(os.path.join(os.environ["LLAMA_CPP_LIB_PATH"], "libggml-cpu.so"))
prompt = eng.tokenize("The quick brown fox jumps over the lazy dog. " * 4)[:32]
P = len(prompt)
def dec(rows):
    b = eng.batch; b.n_tokens = len(rows)
    for i, (t, pos, seq) in enumerate(rows):
        b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, pos, 1, 1
        b.seq_id[i][0] = seq
    t0 = time.perf_counter(); assert L.llama_decode(eng.ctx, b) == 0; return time.perf_counter() - t0
for s in range(32):
    dec([(t, j, s) for j, t in enumerate(prompt)])
N = 8
for R in [1, 8, 32]:
    for s in range(32): L.llama_memory_seq_rm(eng.memory, s, P, -1)
    dec([(prompt[0], P, s) for s in range(R)])
    cpu.row1_reset(); tot = 0
    for k in range(N):
        tot += dec([(prompt[k % P], P + 1 + k, s) for s in range(R)])
    path = f"{sys.argv[2]}_R{R}.txt"; cpu.row1_dump(path.encode())
    print(f"R={R} mean call {tot / N * 1e3:.1f} ms -> {path}")
    rows = [l.split() for l in open(path)]
    rows = [(n, int(c), float(a) / N / eng.threads * 1e3, float(w) / N / eng.threads * 1e3) for n, c, a, w in rows]
    rows.sort(key=lambda r: -(r[2] + r[3]))
    sc = sum(r[2] for r in rows); sw = sum(r[3] for r in rows)
    print(f"  per-thread avg: compute {sc:.1f} ms, barrier wait {sw:.1f} ms")
    for n, c, a, w in rows[:8]:
        print(f"  {n:14s} nodes/call {c // N:4d} compute {a:6.1f} ms wait {w:5.1f} ms")
