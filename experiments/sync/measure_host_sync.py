"""Was kostet der Host-Readback je Decode-Schritt?

Der heutige Pfad liest jedes Token zum Host (mlx_lm/generate.py:466, y.item()) fuer
Stop-Token-Pruefung und Streaming. Gerechnet wird dabei dasselbe; verschoben wird nur
der Zeitpunkt des Lesens. Vorregistrierung: experiments/sync/PREREGISTRATION.md
"""
import sys, json, time, statistics
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
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
sampler = make_sampler(temp=0.0)
stops = frozenset(int(t) for t in tok.eos_token_ids)

FILLER = ("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ") * 40
txt = tok.apply_chat_template(
    [{"role": "user", "content": FILLER + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
IDS = list(txt if isinstance(txt, list) else tok.encode(txt))

def prefill():
    cache = make_prompt_cache(model)
    logits = None
    for s in range(0, len(IDS), CHUNK):
        at = time.perf_counter()
        logits = model(mx.array([IDS[s:s + CHUNK]]), cache=cache)
        mx.eval(logits); mx.synchronize(); charge(time.perf_counter() - at)
    return cache, logits

def decode(mode):
    """Nur die Decode-Phase wird gemessen; der Prefill zaehlt nicht mit."""
    cache, logits = prefill()
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(y)
    at = time.perf_counter()
    out = []
    if mode == "deferred":
        cols = [y]
        for _ in range(STEPS):
            logits = model(y, cache=cache)
            y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
            mx.async_eval(y)           # kein Host-Readback, nur Fortschritt anstossen
            cols.append(y)
        mx.eval(cols[-1]); mx.synchronize()
        out = [int(v) for v in mx.concatenate(cols, axis=1)[0].tolist()]
    else:
        out = [int(y[0, 0])]
        for _ in range(STEPS):
            logits = model(y, cache=cache)
            y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
            mx.eval(y)
            t = int(y[0, 0])           # Host-Readback
            out.append(t)
            if mode == "eos_check" and t in stops:
                break
        mx.synchronize()
    worked = time.perf_counter() - at
    charge(worked)
    return out, worked

# Aufwaermen, Ergebnis verworfen
decode("readback")

per = {m: [] for m in ("readback", "deferred", "eos_check")}
toks = {}
for _ in range(REPS):
    for mode in ("readback", "deferred", "eos_check"):
        out, w = decode(mode)
        per[mode].append(w / len(out))
        toks.setdefault(mode, out)

ref = toks["readback"]
res = {"candidate_id": "host-sync-20260824-01", "formal_claim": False,
       "steps": STEPS, "reps": REPS, "prompt_tokens": len(IDS),
       "ms_per_token": {m: round(statistics.median(v) * 1000, 4) for m, v in per.items()},
       "tokens_produced": {m: len(t) for m, t in toks.items()},
       "identical_to_readback": {m: toks[m][:len(ref)] == ref[:len(toks[m])]
                                 for m in toks},
       "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
base = res["ms_per_token"]["readback"]
res["ratio_vs_readback"] = {m: round(v / base, 4) for m, v in res["ms_per_token"].items()}
res["saving_deferred_pct"] = round((1 - res["ms_per_token"]["deferred"] / base) * 100, 2)
Path("experiments/sync/results.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "budget"}, indent=2))
