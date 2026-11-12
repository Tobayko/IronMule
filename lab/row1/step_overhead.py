"""Round 5 part 5: the pure-Python cost of GgmlEngine.run per step, llama.cpp replaced by a no-op stub (decode
costs about 0), random logits over Qwen3's 151,936-token vocabulary. Usage: [IRONMULE_GGML_PROMPT_BUDGET=B] python step_overhead.py"""
import ctypes, sys, time, types, random
import numpy as np
sys.path.insert(0, "/home/user/IronMule")
from ironmule import ggml_engine as ge
from ironmule.speculation import StepPolicy

V = 151936
class Batch:
    def __init__(self, n):
        self.token = (ctypes.c_int32 * n)(); self.pos = (ctypes.c_int32 * n)(); self.n_seq_id = (ctypes.c_int32 * n)()
        self.logits = (ctypes.c_int8 * n)(); self.seq_id = [(ctypes.c_int32 * 1)() for _ in range(n)]; self.n_tokens = 0
rng = np.random.default_rng(0)
LOG = rng.standard_normal((2048, V)).astype(np.float32)
PTRS = [LOG[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for i in range(2048)]
calls = {"decode": 0, "rows": 0}
L = types.SimpleNamespace(
    llama_decode=lambda ctx, b: (calls.__setitem__("decode", calls["decode"] + 1), calls.__setitem__("rows", calls["rows"] + b.n_tokens), 0)[2],
    llama_get_logits_ith=lambda ctx, i: PTRS[i],
    llama_memory_seq_rm=lambda m, s, a, b: True,
    llama_vocab_is_eog=lambda v, t: False)
def make(seqs, speculate=True):
    e = ge.GgmlEngine.__new__(ge.GgmlEngine)
    e.L, e.np, e.ctx, e.vocab, e.n_vocab, e.memory = L, np, None, None, V, None
    e.seqs, e.seq_ctx, e.threads = seqs, 1024, 4
    e.batch = Batch(1024); e.policy = StepPolicy(16) if speculate else None
    e._free = list(range(seqs)); import threading; e._lock = threading.Lock(); e.steps = 0
    return e
for n in (1, 8, 16, 32):
    for spec in (True, False):
        e = make(32, spec)
        random.seed(n)
        jobs = [ge.Job([random.randrange(1000) for _ in range(40)], 64, emit=lambda t: None) for _ in range(n)]
        calls.update(decode=0, rows=0)
        t = time.perf_counter(); e.run(jobs); dt = time.perf_counter() - t
        print(f"N={n:2d} spec={spec!s:5} steps={e.steps:3d} rows={calls['rows']:5d} python ms/step={1000*dt/e.steps:6.2f}")
# argmax alone
t = time.perf_counter()
for i in range(32): int(LOG[i].argmax())
print(f"argmax x32: {1000*(time.perf_counter()-t):.2f} ms")
