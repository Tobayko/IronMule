"""Liefert ein warmer Prozess dieselben Token wie ein frischer?

Nach vier Zyklen, in denen dreimal eine formabhaengige Numerik die Ausgabe still
veraenderte, ist das keine rhetorische Frage: ein persistenter Prozess traegt
Speicherlage, Allokatorzustand und aufgebaute Kernel ueber Anfragen hinweg.

Modus "cold": ein Prompt, dann Ende. Modus "warm": P Q P R P in einem Prozess.
Vorregistrierung: experiments/persistent/PREREGISTRATION.md
"""
import sys, json, time, resource
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

GEN, CHUNK = 32, 256
MODE = sys.argv[1] if len(sys.argv) > 1 else "warm"

require_ac_power()
guard = BudgetGuard()
_debt = 0.0
def charge(sec):
    global _debt
    guard.record_gpu(sec); _debt += sec * (1 - 0.15) / 0.15
    while _debt >= 4.0:
        guard.required_break(); _debt -= 4.0

t_start = time.perf_counter()
snap = resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
model, tok = load(str(snap.path))
load_done = time.perf_counter()
sampler = make_sampler(temp=0.0)

FILLER = ("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ") * 40
PROMPTS = {
    "P": FILLER + "\n\nWhy is false sharing slow?",
    "Q": FILLER + "\n\nWhat does a TLB miss cost?",
    "R": FILLER + "\n\nWhen does store forwarding fail?",
}

def ids_for(key):
    txt = tok.apply_chat_template([{"role": "user", "content": PROMPTS[key]}],
                                  add_generation_prompt=True)
    return list(txt if isinstance(txt, list) else tok.encode(txt))

def answer(key):
    """Frischer KV-Cache je Anfrage. Praefix-Wiederverwendung ist hier bewusst aus."""
    ids = ids_for(key)
    cache = make_prompt_cache(model)
    at = time.perf_counter()
    logits = None
    for s in range(0, len(ids), CHUNK):
        logits = model(mx.array([ids[s:s + CHUNK]]), cache=cache)
        mx.eval(logits); mx.synchronize()
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]; mx.eval(y)
    ttft = time.perf_counter() - at
    out = [int(y[0, 0])]
    for _ in range(GEN - 1):
        logits = model(y, cache=cache)
        y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]; mx.eval(y)
        out.append(int(y[0, 0]))
    mx.synchronize()
    total = time.perf_counter() - at
    charge(total)
    return {"key": key, "tokens": out, "ttft_s": round(ttft, 4),
            "total_s": round(total, 4),
            "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

sequence = ["P"] if MODE == "cold" else ["P", "Q", "P", "R", "P"]
results = [answer(k) for k in sequence]

print(json.dumps({
    "mode": MODE,
    "startup_s": round(load_done - t_start, 4),
    "prompt_tokens": len(ids_for("P")),
    "results": results,
    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
    "budget": {k: v for k, v in guard.summary().items() if "limit" not in k},
}))
