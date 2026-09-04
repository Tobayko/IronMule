"""Kostet die logsumexp-Normalisierung vor einem greedy-Argmax etwas Messbares?

generate.py rechnet sie unbedingt; argmax ist gegenueber dem Abzug einer Konstante
invariant. Vorregistrierung: experiments/logsumexp/PREREGISTRATION.md
"""
import sys, json, time, statistics
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

STEPS, REPS, CHUNK = 128, 3, 256

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

FILLER = ("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ") * 40
txt = tok.apply_chat_template(
    [{"role": "user", "content": FILLER + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
IDS = list(txt if isinstance(txt, list) else tok.encode(txt))

def decode(normalise):
    cache = make_prompt_cache(model)
    logits = None
    for s in range(0, len(IDS), CHUNK):
        at = time.perf_counter()
        logits = model(mx.array([IDS[s:s + CHUNK]]), cache=cache)
        mx.eval(logits); mx.synchronize(); charge(time.perf_counter() - at)

    def pick(lg):
        row = lg[:, -1, :]
        if normalise:
            row = row - mx.logsumexp(row, keepdims=True)   # wie generate.py
        return mx.argmax(row, axis=-1)[:, None]

    y = pick(logits); mx.eval(y)
    at = time.perf_counter()
    out = [int(y[0, 0])]
    for _ in range(STEPS):
        logits = model(y, cache=cache)
        y = pick(logits)
        mx.eval(y)
        out.append(int(y[0, 0]))
    mx.synchronize()
    # Dauer VOR dem Verbuchen festhalten: charge() schlaeft die Guard-Pausen, und
    # danach zu messen zaehlt sie der Rechnung zu. Ein erster Lauf meldete so 107 ms
    # je Token statt der tatsaechlichen 12.
    worked = time.perf_counter() - at
    charge(worked)
    return out, worked

# Normalisierung isoliert, ohne Modellaufruf
row = mx.random.normal((1, 262208)).astype(mx.bfloat16); mx.eval(row)
for _ in range(5):
    mx.eval(row - mx.logsumexp(row, keepdims=True))
mx.synchronize()
iso = []
at = time.perf_counter()
for _ in range(200):
    t = time.perf_counter_ns()
    mx.eval(row - mx.logsumexp(row, keepdims=True)); mx.synchronize()
    iso.append(time.perf_counter_ns() - t)
charge(time.perf_counter() - at)
iso_ms = statistics.median(iso) / 1e6

decode(True)   # Aufwaermen

per, toks = {True: [], False: []}, {}
for _ in range(REPS):
    for norm in (True, False):
        out, w = decode(norm)
        per[norm].append(w / len(out))
        toks.setdefault(norm, out)

ms = {("normalised" if k else "raw"): round(statistics.median(v) * 1000, 4)
      for k, v in per.items()}
base = ms["normalised"]
res = {"candidate_id": "logsumexp-skip-20260824-01", "formal_claim": False,
       "steps": STEPS, "reps": REPS,
       "ms_per_token": ms,
       "saving_pct": round((1 - ms["raw"] / base) * 100, 3),
       "identical": toks[True] == toks[False],
       "isolated_logsumexp_ms": round(iso_ms, 4),
       "isolated_share_of_step_pct": round(iso_ms / base * 100, 3),
       "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
Path("experiments/logsumexp/results.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "budget"}, indent=2))
