# SD1 probe: draft acceptance (teacher forcing, k=1) of Qwen3-0.6B for Qwen3-32B on the benchmark prompts, and step costs
import os, statistics as st, time
import torch
from ironmule.ggml_engine import GgmlEngine
from ironmule_product import compare
TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
T = GgmlEngine('/models/Qwen3-32B-Q4_K_M.gguf', seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
D = GgmlEngine('/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf', seqs=32, seq_ctx=2048, flash_attn=True, speculate=False, kernel_mask=False)
def timed(e, lens, per, n=12):
    """Median ms of n consecutive steps with `per` rows per sequence at advancing positions (as decoding runs)."""
    ts, pos = [], dict(lens)
    for _ in range(n):
        rows = [(8, pos[s] + j, s, True) for s in sorted(pos) for j in range(per)]
        torch.cuda.synchronize(); t = time.perf_counter(); e._decode(rows); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        for s in pos: pos[s] += per
    return round(1e3 * st.median(ts[3:]), 2)
for width, seed in ((32, 0), (32, 1), (8, 0)):
    prompts = compare.performance_prompts(seed, width, max(width, 8))
    ids = sorted((T.tokenize(TPL.format(p)) for p in prompts), key=len)
    n = len(ids)
    T._clear()
    first = T._decode([(ids[s][j], j, s, j == len(ids[s]) - 1) for s in range(n) for j in range(len(ids[s]))])
    ans = [[t] for t in first]
    live = [not T.eog(t) for t in first]
    for k in range(31):
        rows = [(ans[s][-1], len(ids[s]) + len(ans[s]) - 1, s, True) for s in range(n)]
        out = T._decode(rows)
        for s in range(n):
            if live[s]:
                ans[s].append(out[s]); live[s] = not T.eog(out[s])
            else:
                ans[s].append(ans[s][-1])
    # step costs after the answers: 1 and 2 rows per sequence, advancing (12 steps each)
    at = {s: len(ids[s]) + 31 for s in range(n)}
    c1 = timed(T, at, 1)
    at = {s: p + 12 for s, p in at.items()}
    c2 = timed(T, at, 2)
    at = {s: p + 24 for s, p in at.items()}
    os.environ["IRONMULE_W4_SPLIT64"] = "1"
    c2s = timed(T, at, 2)
    os.environ.pop("IRONMULE_W4_SPLIT64", None)
    # draft model: teacher-forced predictions over each answer
    D._clear()
    hit = tot = 0
    rows, want = [], []
    for s in range(n):
        seq = ids[s] + ans[s]
        stop = len(ids[s]) + (ans[s].index(next(t for t in ans[s] if T.eog(t))) if any(T.eog(t) for t in ans[s]) else 31)
        for j in range(len(seq) - 1):
            flag = len(ids[s]) - 1 <= j < stop
            rows.append((seq[j], j, s, flag))
            if flag:
                want.append(seq[j + 1])
    pred = []
    for half in (range(0, n // 2), range(n // 2, n)):
        pred += D._decode([r for r in rows if r[2] in half])
    for p_, w_ in zip(pred, want):
        hit += int(p_ == w_); tot += 1
    d1 = timed(D, {s: len(ids[s]) + 31 for s in range(n)}, 1)
    lens = sorted(set(len(i) for i in ids))
    print(f"SD width {width} seed {seed}: prompt lengths {lens[:6]}{'...' if len(lens) > 6 else ''} ({len(lens)} distinct); "
          f"draft acceptance k=1 {hit}/{tot} = {hit / tot:.3f}; target step {n} rows {c1} ms, {2 * n} rows MMQ {c2} ms, "
          f"{2 * n} rows W4R split {c2s} ms; draft step {n} rows {d1} ms", flush=True)
T.close(); D.close()
