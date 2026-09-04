"""Gebuendelter Host-Readback: alle N Schritte pruefen statt jeden.

Zyklus 6 mass 15,3 % Ersparnis bei ganz entfallendem Readback -- ein Arm, der nicht
anhalten kann. Dies ist die abrufbare Form.
Vorregistrierung: experiments/sync/PREREGISTRATION_BATCHED.md
"""
import sys, json, time, statistics
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

STEPS, REPS, CHUNK = 128, 2, 256
INTERVALS = (1, 2, 4, 8, 16, 32)

require_ac_power()
guard = BudgetGuard()
_debt = 0.0
def charge(sec):
    global _debt
    guard.record_gpu(sec); _debt += sec * (1 - 0.15) / 0.15
    while _debt >= 4.0:
        guard.required_break(); _debt -= 4.0

snap = resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
model, tok = load(str(snap.path))
sampler = make_sampler(temp=0.0)
stops = frozenset(int(t) for t in tok.eos_token_ids)

FILLER = ("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ") * 40
txt = tok.apply_chat_template(
    [{"role": "user", "content": FILLER + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
IDS = list(txt if isinstance(txt, list) else tok.encode(txt))

def decode(interval):
    cache = make_prompt_cache(model)
    logits = None
    for s in range(0, len(IDS), CHUNK):
        at = time.perf_counter()
        logits = model(mx.array([IDS[s:s + CHUNK]]), cache=cache)
        mx.eval(logits); mx.synchronize(); charge(time.perf_counter() - at)
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(y)
    at = time.perf_counter()
    cols, pending, out = [y], [], []
    for i in range(STEPS):
        logits = model(y, cache=cache)
        y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
        mx.async_eval(y)
        cols.append(y); pending.append(y)
        if (i + 1) % interval == 0:
            # Gebuendelter Readback: eine Synchronisation je Intervall, dann pruefen.
            mx.eval(pending[-1])
            vals = [int(p[0, 0]) for p in pending]
            out.extend(vals)
            pending = []
            if any(v in stops for v in vals):
                break
    if pending:
        mx.eval(pending[-1]); out.extend(int(p[0, 0]) for p in pending)
    mx.synchronize()
    worked = time.perf_counter() - at
    charge(worked)
    return [int(cols[0][0, 0])] + out, worked

decode(1)  # Aufwaermen, verworfen

per, toks = {n: [] for n in INTERVALS}, {}
for _ in range(REPS):
    for n in INTERVALS:
        out, w = decode(n)
        per[n].append(w / len(out))
        toks.setdefault(n, out)

ref = toks[1]
ms = {n: round(statistics.median(v) * 1000, 4) for n, v in per.items()}
base = ms[1]
readback_ms = 14.3671 - 12.1683      # Zyklus 6, gemessen
res = {"candidate_id": "batched-readback-20260824-01", "formal_claim": False,
       "steps": STEPS, "reps": REPS,
       "ms_per_token": ms,
       "ratio_vs_interval_1": {n: round(v / base, 4) for n, v in ms.items()},
       "saving_pct": {n: round((1 - v / base) * 100, 2) for n, v in ms.items()},
       "identical_to_interval_1": {n: toks[n] == ref for n in INTERVALS},
       "tokens_produced": {n: len(toks[n]) for n in INTERVALS},
       "derived_breakeven_length": {
           n: (None if n == 1 else
               round(((n - 1) / 2 * base) / ((1 - 1 / n) * readback_ms), 1))
           for n in INTERVALS},
       "derivation_note": ("Break-even ist BERECHNET, nicht gemessen: erwarteter "
                           "Ueberlauf (N-1)/2 Token zur vollen Schrittzeit gegen "
                           "gesparte Readbacks (Zyklus 6: 2,199 ms je Schritt)"),
       "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
Path("experiments/sync/batched_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "budget"}, indent=2))
