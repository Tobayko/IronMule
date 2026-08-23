"""Sagt die Laenge einer Uebereinstimmung voraus, ob ihre Fortsetzung akzeptiert wird?
Erst das Signal messen, dann eine Policy darauf bauen -- nicht umgekehrt."""
import sys, json, statistics
from collections import defaultdict
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from friday_hardware import HardwareProfile, accepted_prefix
from mlx_lm import load
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

MIN_N, MAX_EXTEND, DRAFT, TOKENS = 3, 40, 4, 96

def match_with_length(tokens, min_n, max_extend, draft):
    """Juengste Uebereinstimmung, plus wie weit sie sich rueckwaerts fortsetzt."""
    if len(tokens) <= min_n: return 0, []
    needle = tokens[-min_n:]
    for start in range(len(tokens) - min_n - 1, -1, -1):
        if tokens[start:start+min_n] != needle: continue
        # rueckwaerts verlaengern, solange es haelt
        length = min_n
        while (length < max_extend and start - (length - min_n) - 1 >= 0
               and tokens[start - (length - min_n) - 1] == tokens[-length - 1]):
            length += 1
        return length, list(tokens[start+min_n : start+min_n+draft])
    return 0, []

require_ac_power(); g = BudgetGuard()
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

prof = HardwareProfile.load(Path("profiles/m1max_gemma-3-4b-4bit-g64.json"))
snap = resolve_local_model_snapshot(prof.model_id)
m, tok = load(str(snap.path)); sampler = make_sampler(temp=0.0)

buckets = defaultdict(lambda: [0, 0])   # Trefferlaenge -> [gedraftet, akzeptiert]
for f in sorted(Path("experiments/prompt_lookup/real").glob("*.txt")):
    txt = tok.apply_chat_template([{"role":"user","content":f.read_text()}], add_generation_prompt=True)
    ids = list(txt if isinstance(txt,list) else tok.encode(txt))
    cache = make_prompt_cache(m); ctx = list(ids)
    t0 = __import__("time").perf_counter()
    lg = m(mx.array([ctx]), cache=cache)
    y = int(sampler(lg[:,-1,:].astype(mx.float32))[0]); mx.eval(y); ctx.append(y)
    gen = [y]
    while len(gen) < TOKENS:
        mlen, drafted = (0, [])
        if can_trim_prompt_cache(cache):
            mlen, drafted = match_with_length(ctx, MIN_N, MAX_EXTEND, DRAFT)
        window = [ctx[-1]] + drafted
        lg = m(mx.array([window]), cache=cache)
        picks = sampler(lg[0].astype(mx.float32)).tolist(); mx.eval(lg)
        keep = accepted_prefix(drafted, picks[:-1]) if drafted else 0
        if drafted:
            b = buckets[min(mlen, 30)]
            b[0] += len(drafted); b[1] += keep
        surplus = len(window) - (keep+1)
        if surplus > 0: trim_prompt_cache(cache, surplus)
        for t in picks[:keep+1]:
            ctx.append(int(t)); gen.append(int(t))
            if len(gen) >= TOKENS: break
    charge(__import__("time").perf_counter() - t0)

print(f"{'Trefferlaenge':>14}{'gedraftet':>11}{'akzeptiert':>12}{'Akzeptanz':>11}")
rows=[]
for L in sorted(buckets):
    d, a = buckets[L]
    if d: rows.append((L, d, a, a/d)); print(f"{L:>14}{d:>11}{a:>12}{a/d:>11.3f}")
# grob gruppiert
print()
for lo, hi, lbl in ((3,4,"3-4"), (5,8,"5-8"), (9,15,"9-15"), (16,40,"16+")):
    d = sum(buckets[L][0] for L in buckets if lo <= L <= hi)
    a = sum(buckets[L][1] for L in buckets if lo <= L <= hi)
    if d: print(f"  Treffer {lbl:>5}: {a}/{d} = {a/d:.3f}")
