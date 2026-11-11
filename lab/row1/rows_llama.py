"""E1: llama.cpp decode-call time against rows. R sequences x 1 position, and 1 sequence x R positions."""
import os, sys, time, statistics, json
sys.path.insert(0, "/home/user/IronMule")
import ironmule.ggml_engine as ge

gguf = sys.argv[1]
mmap = os.environ.get("ROW1_MMAP", "1") == "1"
RS = [int(x) for x in os.environ.get("ROW1_RS", "1,2,4,8,16,32").split(",")]
REPS = int(os.environ.get("ROW1_REPS", "9"))
ge.GgmlEngine._warm_up = lambda self: None
if not mmap:
    L = ge.bindings()
    orig = L.llama_model_default_params
    def p():
        q = orig(); q.use_mmap = False; return q
    L.llama_model_default_params = p
eng = ge.GgmlEngine(gguf, speculate=False, seqs=32, seq_ctx=256, max_rows=64)
L = eng.L
prompt = eng.tokenize("The quick brown fox jumps over the lazy dog. " * 4)[:32]
P = len(prompt)

def dec(rows):
    b = eng.batch; b.n_tokens = len(rows)
    for i, (t, pos, seq) in enumerate(rows):
        b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, pos, 1, 1
        b.seq_id[i][0] = seq
    t0 = time.perf_counter()
    assert L.llama_decode(eng.ctx, b) == 0
    return time.perf_counter() - t0

# prefill all 32 sequences with the same prompt
for s in range(32):
    dec([(t, j, s) for j, t in enumerate(prompt)])
out = {"gguf": os.path.basename(gguf), "mmap": mmap, "seqs": {}, "window": {}}
pos = P
for R in RS:  # warm each shape once, then REPS
    ts = []
    for k in range(REPS + 1):
        dt = dec([(prompt[(pos) % P], pos, s) for s in range(R)]); pos += 1
        if k: ts.append(dt)
    out["seqs"][R] = statistics.median(ts) * 1e3
    # rows already decoded in sequences >= R stay behind; positions keep rising for all seqs we touched only
    for s in range(R, 32):
        pass
    # keep every sequence's positions in step: advance the untouched ones is not needed (each seq checks its own)
    print(f"seqs R={R}: {out['seqs'][R]:.1f} ms", flush=True)
    # after this R, sequences R..31 lag behind; next R uses them at `pos`, which llama.cpp rejects -> realign
    for s in range(R, 32):
        L.llama_memory_seq_rm(eng.memory, s, P, -1)
    if R != RS[-1]:
        # realign: clear all to the prompt and restart positions
        for s in range(32):
            L.llama_memory_seq_rm(eng.memory, s, P, -1)
        pos = P
mem = eng.memory
for s in range(32):
    L.llama_memory_seq_rm(mem, s, P, -1)
for R in RS:
    ts = []
    for k in range(REPS + 1):
        dt = dec([(prompt[j % P], P + j, 0) for j in range(R)])
        L.llama_memory_seq_rm(mem, 0, P, -1)
        if k: ts.append(dt)
    out["window"][R] = statistics.median(ts) * 1e3
    print(f"window R={R}: {out['window'][R]:.1f} ms", flush=True)
with open(f"/proc/{os.getpid()}/smaps_rollup") as f:
    out["AnonHugePages_kB"] = [int(l.split()[1]) for l in f if l.startswith("AnonHugePages")][0]
print(json.dumps(out))
