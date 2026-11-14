"""DM1 probe: speculation with a small draft model where verification rows are nearly free (RL-MATH: 2.4 ms a row
against a 99 ms step on this CPU). Target Qwen3-4B Q8_0, draft Qwen3-0.6B Q8_0, one request at a time, greedy, 128
tokens, 8 prompts. Plain decoding against draft-k-then-verify for several k: tokens/s, how many drafts were accepted
by position, where the time went, and whether the tokens are the plain run's.
Usage: probe.py TARGET.gguf DRAFT.gguf OUT.json [k,k,...]"""
import json, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine
target_path, draft_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "3,5,7").split(",")]
LIMIT = 128
PROMPTS = ["Write a short story about a lighthouse keeper.", "Explain how a refrigerator keeps food cold.",
           "Write a Python function that checks whether a string is a palindrome, with a docstring.",
           "List five facts about the planet Mars.", "What causes the seasons on Earth? Answer in a paragraph.",
           "Write a JSON object describing a book with title, author, year and three tags.",
           "Summarize the plot of Romeo and Juliet in a few sentences.", "Give step by step instructions to make tea."]
T = GgmlEngine(target_path, seqs=1, seq_ctx=2048, speculate=False, flash_attn=True)
D = GgmlEngine(draft_path, seqs=1, seq_ctx=2048, speculate=False, flash_attn=True)
def chat(text):
    return T.tokenize(f"<|im_start|>user\n{text} /no_think<|im_end|>\n<|im_start|>assistant\n")
def plain(ids):
    T._clear(); toks = [T._prefill(ids, 0)]; t0 = time.perf_counter()
    while len(toks) < LIMIT and not T.eog(toks[-1]):
        toks.append(T._decode([(toks[-1], len(ids) + len(toks) - 1, 0, True)])[0])
    return toks, time.perf_counter() - t0
def spec(ids, k):
    T._clear(); D._clear()
    toks = [T._prefill(ids, 0)]; dheld = 0; stats = {"draft_s": 0.0, "verify_s": 0.0, "steps": 0, "accepted": [0] * (k + 1)}
    t0 = time.perf_counter()
    while len(toks) < LIMIT and not T.eog(toks[-1]):
        ctx = ids + toks
        kk = min(k, LIMIT - len(toks) - 1)
        td = time.perf_counter(); drafts = []
        if kk > 0:
            feed = ctx[dheld:]
            nxt = D._decode([(t, dheld + j, 0, j == len(feed) - 1) for j, t in enumerate(feed)])[0]
            dheld = len(ctx); drafts.append(nxt)
            while len(drafts) < kk and not D.eog(drafts[-1]):
                nxt = D._decode([(drafts[-1], dheld, 0, True)])[0]; dheld += 1; drafts.append(nxt)
        stats["draft_s"] += time.perf_counter() - td
        tv = time.perf_counter(); pos = len(ctx) - 1
        picks = T._decode([(t, pos + j, 0, True) for j, t in enumerate([toks[-1]] + drafts)])
        stats["verify_s"] += time.perf_counter() - tv
        a = 0
        while a < len(drafts) and drafts[a] == picks[a]: a += 1
        new = drafts[:a] + [picks[a]]
        stats["accepted"][a] += 1; stats["steps"] += 1
        if a < len(drafts):
            T.L.llama_memory_seq_rm(T.memory, 0, pos + 1 + a, -1)
        keep = len(ctx) + min(a, max(0, len(drafts) - 1))  # the draft's cells that still match the text
        if dheld > keep:
            D.L.llama_memory_seq_rm(D.memory, 0, keep, -1); dheld = keep
        for t in new:
            toks.append(t)
            if T.eog(t) or len(toks) >= LIMIT: break
    stats["s"] = time.perf_counter() - t0
    return toks, stats
import numpy as np
class Stop:
    """The optimal myopic stopping rule: one more draft costs c_d + c_r and adds q * p expected tokens, where q is the
    chance that every draft so far is accepted and p the chance the next one is; draft on while that marginal rate
    beats the step's rate so far, (1 + E) / T. p comes from the draft's own top-1 probability through a calibration
    learned online (acceptance by confidence bin); the costs are running means of this machine's steps."""
    BINS = 10
    def __init__(self):
        self.tried = [2.0] * self.BINS; self.taken = [2.0 * (i + 0.5) / self.BINS for i in range(self.BINS)]
        self.tried[-1] = 20.0; self.taken[-1] = 16.0  # most drafts are confident ones: the mean starts near 0.7
        self.c_d, self.c_v, self.c_r = 0.015, 0.085, 0.0025
    def p(self, conf):
        b = min(self.BINS - 1, int(conf * self.BINS)); return self.taken[b] / self.tried[b]
    def mean(self):
        return sum(self.taken) / sum(self.tried)
    def more(self, q, expected, k):
        """Compute one more draft? Its cost is paid before its confidence is known, so its chance is the mean."""
        return q * self.mean() / (self.c_d + self.c_r) > (1.0 + expected) / (self.c_v + k * (self.c_d + self.c_r))
    def learn(self, confs, accepted):
        for i, conf in enumerate(confs[:accepted + 1]):  # every draft that was reached
            b = min(self.BINS - 1, int(conf * self.BINS)); self.tried[b] += 1; self.taken[b] += i < accepted
def dstep(rows):
    b = D.batch; b.n_tokens = len(rows)
    for i, (t, p, s_, l) in enumerate(rows):
        b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, p, 1, int(l); b.seq_id[i][0] = s_
    assert D.L.llama_decode(D.ctx, b) == 0
    row = np.ctypeslib.as_array(D.L.llama_get_logits_ith(D.ctx, len(rows) - 1), shape=(D.n_vocab,))
    top = int(row.argmax()); m = float(row[top])
    return top, float(1.0 / np.exp(row - m).sum())
def spec_opt(ids, stop, kmax=12):
    T._clear(); D._clear()
    toks = [T._prefill(ids, 0)]; dheld = 0; st = {"draft_s": 0.0, "verify_s": 0.0, "steps": 0, "drafted": 0, "accepted": 0}
    t0 = time.perf_counter()
    while len(toks) < LIMIT and not T.eog(toks[-1]):
        ctx = ids + toks; room = min(kmax, LIMIT - len(toks) - 1)
        td = time.perf_counter(); drafts, confs, q, expected = [], [], 1.0, 0.0
        if room > 0:
            feed = ctx[dheld:]
            nxt, conf = dstep([(t, dheld + j, 0, j == len(feed) - 1) for j, t in enumerate(feed)]); dheld = len(ctx)
            while True:
                drafts.append(nxt); confs.append(conf); q *= stop.p(conf); expected += q  # computed: a row is cheap
                if len(drafts) >= room or D.eog(nxt) or not stop.more(q, expected, len(drafts)): break
                nxt, conf = dstep([(drafts[-1], dheld, 0, True)]); dheld += 1
        dt = time.perf_counter() - td
        tv = time.perf_counter(); pos = len(ctx) - 1
        picks = T._decode([(t, pos + j, 0, True) for j, t in enumerate([toks[-1]] + drafts)])
        vt = time.perf_counter() - tv
        a = 0
        while a < len(drafts) and drafts[a] == picks[a]: a += 1
        stop.learn(confs, a)
        calls = max(1, dheld - len(ctx) + 1)
        stop.c_d += 0.05 * (dt / calls - stop.c_d)
        stop.c_v += 0.05 * (vt - stop.c_r * len(drafts) - stop.c_v)
        st["draft_s"] += dt; st["verify_s"] += vt; st["steps"] += 1; st["drafted"] += len(drafts); st["accepted"] += a
        if a < len(drafts):
            T.L.llama_memory_seq_rm(T.memory, 0, pos + 1 + a, -1)
        keep = len(ctx) + min(a, dheld - len(ctx))
        if dheld > keep:
            D.L.llama_memory_seq_rm(D.memory, 0, keep, -1); dheld = keep
        for t in drafts[:a] + [picks[a]]:
            toks.append(t)
            if T.eog(t) or len(toks) >= LIMIT: break
    st["s"] = time.perf_counter() - t0
    return toks, st
res = {"prompts": len(PROMPTS), "limit": LIMIT, "runs": {}}
ids_all = [chat(p) for p in PROMPTS]
plain(ids_all[0]); spec(ids_all[0], 3)  # warm both models
ref, tot_t, tot_s = [], 0, 0.0
for ids in ids_all:
    toks, s = plain(ids); ref.append(toks); tot_t += len(toks) - 1; tot_s += s
res["plain_tok_s"] = tot_t / tot_s
print("plain", round(res["plain_tok_s"], 2), "tok/s", flush=True)
for k in (KS if "--no-fixed" not in sys.argv else []):
    tt, ts, same, dr, ve, acc, steps = 0, 0.0, 0, 0.0, 0.0, [0] * (k + 1), 0
    for ids, r in zip(ids_all, ref):
        toks, st = spec(ids, k); tt += len(toks) - 1; ts += st["s"]; same += toks == r
        dr += st["draft_s"]; ve += st["verify_s"]; steps += st["steps"]; acc = [x + y for x, y in zip(acc, st["accepted"])]
    row = {"tok_s": tt / ts, "speedup": tt / ts / res["plain_tok_s"], "identical": same, "draft_share": dr / ts,
           "tokens_per_step": tt / steps, "accepted_hist": acc, "draft_ms_per_step": 1000 * dr / steps,
           "verify_ms_per_step": 1000 * ve / steps}
    res["runs"][str(k)] = row
    print("k", k, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
stop = Stop()
for rnd in range(2):  # the second round runs with the calibration the first learned
    tt, ts, same, dr, steps, drafted, accepted = 0, 0.0, 0, 0.0, 0, 0, 0
    for ids, r in zip(ids_all, ref):
        toks, st = spec_opt(ids, stop); tt += len(toks) - 1; ts += st["s"]; same += toks == r
        dr += st["draft_s"]; steps += st["steps"]; drafted += st["drafted"]; accepted += st["accepted"]
    row = {"tok_s": tt / ts, "speedup": tt / ts / res["plain_tok_s"], "identical": same, "draft_share": dr / ts,
           "tokens_per_step": tt / steps, "drafts_per_step": drafted / steps, "acceptance": accepted / max(1, drafted),
           "calibration": [round(a / b, 2) for a, b in zip(stop.taken, stop.tried)],
           "costs_ms": [round(1000 * x, 2) for x in (stop.c_d, stop.c_v, stop.c_r)]}
    res["runs"][f"opt{rnd}"] = row
    print("opt", rnd, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items()}), flush=True)
open(out, "w").write(json.dumps(res, indent=1))
