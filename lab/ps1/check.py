"""PS1 (2026-11-25): a packed prefill over KV streams with unequal prompt and kept-prefix lengths, then greedy decode
steps; writes every output row's logits and the decoded tokens, so that two processes (IRONMULE_PS1=0 / 1) can be
compared. CPU or GPU build; flash attention on (PS1 needs an untransposed V cache).

  IRONMULE_PS1=1 python lab/ps1/check.py GGUF OUT.npz [--steps 8]
"""
import argparse
import time

import numpy as np

from ironmule.ggml_engine import GgmlEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gguf")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seqs", type=int, default=8)
    a = p.parse_args()
    e = GgmlEngine(a.gguf, seqs=a.seqs, seq_ctx=2048, speculate=False, flash_attn=True)
    text = " ".join(f"Site {1000 + j} measured {20 + j % 60} units, next {j * 7 % 13}." for j in range(400))
    ids = e.tokenize(text)
    lengths = ([33, 25, 33, 140, 133, 140, 33, 120] * 4)[:a.seqs]
    keeps = ([3, 11, 0, 3, 12, 0, 5, 7, 1, 9, 2, 6, 4, 10, 8, 0] * 2)[:a.seqs]
    prompts = [ids[k * 13:k * 13 + n] for k, n in enumerate(lengths)]
    e._clear()
    for s, (pr, kp) in enumerate(zip(prompts, keeps)):  # the kept prefixes, as an earlier request left them
        if kp:
            e._decode([(t, j, s, False) for j, t in enumerate(pr[:kp])])
    rows = [(t, j, s, j == len(pr) - 1) for s, (pr, kp) in enumerate(zip(prompts, keeps)) for j, t in enumerate(pr) if j >= kp]
    L = e.L
    b = e._batch_for(len(rows))
    for i, (t, pos, s, lg) in enumerate(rows):
        b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, pos, 1, int(lg)
        b.seq_id[i][0] = s
    t0 = time.perf_counter()
    assert L.llama_decode(e.ctx, b) == 0
    dt = time.perf_counter() - t0
    outs = [i for i, r in enumerate(rows) if r[3]]
    logits = np.stack([np.ctypeslib.as_array(L.llama_get_logits_ith(e.ctx, i), shape=(e.n_vocab,)).copy() for i in outs])
    toks = [list(logits.argmax(1))]
    pos = [len(pr) for pr in prompts]
    cur = toks[0]
    for _ in range(a.steps):
        picks = e._decode([(int(cur[s]), pos[s], s, True) for s in range(len(prompts))])
        pos = [x + 1 for x in pos]
        cur = picks
        toks.append(list(picks))
    np.savez(a.out, logits=logits, tokens=np.array(toks), prefill_s=dt, rows=len(rows))
    print("rows", len(rows), "prefill_s", round(dt, 3), "tokens", np.array(toks).T.tolist())
    e.close()


if __name__ == "__main__":
    main()
