"""ST1 step 1: what the engine's per-step Python costs as the context grows, the old scans against the new indexes.
Prompt lookup per step (draft() against NgramIndex), the pool's longest-prefix search per new request (a scan of every
held sequence against the trie) and the per-step context copy (ids + tokens) the engine no longer makes."""
import random, sys, time
sys.path.insert(0, __import__("os").path.abspath(__file__ + "/../../.."))
from ironmule.speculation import NgramIndex, draft
from ironmule import ggml_engine as ge


def per_call(fn, n):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6


rng = random.Random(0)
print("context  draft-scan-us  index-us  concat-us")
for n_ctx in (512, 2048, 8192, 32768):
    ctx = [rng.randrange(32000) for _ in range(n_ctx)]  # few repeats: the scan's worst case
    idx = NgramIndex(ctx)
    idx.draft(7)
    print(n_ctx, round(per_call(lambda: draft(ctx, 7), 20), 1), round(per_call(lambda: idx.draft(7), 2000), 2),
          round(per_call(lambda: ctx[:n_ctx // 2] + ctx[n_ctx // 2:], 200), 1))

import types
L = types.SimpleNamespace(llama_memory_seq_rm=lambda *a: True, llama_memory_seq_cp=lambda *a: None)
print("sequences x tokens  scan-us  trie-us")
for seqs, length in ((8, 2048), (32, 2048), (64, 4096)):
    pool = ge.SharedPool(types.SimpleNamespace(L=L, memory=None, prefix_reuse=True), 10**7, seqs + 2, seqs + 2, 2)
    system = [rng.randrange(32000) for _ in range(length // 2)]
    for s in range(seqs):
        ids = system + [rng.randrange(32000) for _ in range(length // 2)]
        seq, _ = pool.place(ids, 8)
        pool.sync(seq, ids, 8)
    query = system + [rng.randrange(32000) for _ in range(64)]

    def scan():
        best, match = None, 0
        for seq, held in pool.tokens.items():
            n = ge.GgmlEngine._common(held, query)
            if n > match:
                best, match = seq, n
        return best, match

    def scan_loop():  # the token-by-token loop the scan used before
        best, match = None, 0
        for seq, held in pool.tokens.items():
            i, m = 0, min(len(held), len(query))
            while i < m and held[i] == query[i]:
                i += 1
            if i > match:
                best, match = seq, i
        return best, match

    assert scan() == scan_loop() == pool.longest(query)
    print(seqs, length, round(per_call(scan_loop, 20), 1), round(per_call(scan, 50), 1),
          round(per_call(lambda: pool.longest(query), 200), 1))
