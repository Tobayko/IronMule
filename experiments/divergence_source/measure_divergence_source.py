"""Wo entsteht die Zerteilungs-Divergenz? Schichtweiser Vergleich der KV-Caches.

Zyklus 2 zeigte die Divergenz, Zyklus 4 ihre Wirkung. Keiner nannte die Ursache.
Vorregistrierung: experiments/divergence_source/PREREGISTRATION.md
"""
import sys, json, time
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

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

FILLER = ("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ") * 30
txt = tok.apply_chat_template(
    [{"role": "user", "content": FILLER + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
IDS = list(txt if isinstance(txt, list) else tok.encode(txt))
print("Prompt:", len(IDS), "Token")

def prefill(splits):
    cache = make_prompt_cache(model)
    pos, logits = 0, None
    for n in splits:
        piece = IDS[pos:pos + n]; pos += n
        if not piece:
            continue
        at = time.perf_counter()
        logits = model(mx.array([piece]), cache=cache)
        mx.eval(logits); mx.synchronize(); charge(time.perf_counter() - at)
    return cache, logits

N = len(IDS)
cache_a, logits_a = prefill([N])
cache_b, logits_b = prefill([512, N - 512])

def stats(a, b):
    """Maximaler absoluter Unterschied und Groessenordnung, auf dem gueltigen Teil."""
    if a is None or b is None:
        return None
    n = min(a.shape[2], b.shape[2])
    x, y = a[..., :n, :].astype(mx.float32), b[..., :n, :].astype(mx.float32)
    d = mx.abs(x - y)
    scale = mx.maximum(mx.abs(x).max(), mx.array(1e-9))
    mx.eval(d, scale)
    return {"max_abs": float(d.max().item()),
            "mean_abs": float(d.mean().item()),
            "max_rel": float((d.max() / scale).item())}

layers = []
for i, (ca, cb) in enumerate(zip(cache_a, cache_b)):
    k = stats(getattr(ca, "keys", None), getattr(cb, "keys", None))
    v = stats(getattr(ca, "values", None), getattr(cb, "values", None))
    layers.append({"layer": i, "type": type(ca).__name__, "keys": k, "values": v})

la = logits_a[0, -1, :].astype(mx.float32)
lb = logits_b[0, -1, :].astype(mx.float32)
mx.eval(la, lb)
top_a = mx.argsort(-la)[:2].tolist()
gap_a = float((la[top_a[0]] - la[top_a[1]]).item())
logit_diff = float(mx.abs(la - lb).max().item())
first = next((l["layer"] for l in layers
              if (l["keys"] and l["keys"]["max_abs"] > 0)
              or (l["values"] and l["values"]["max_abs"] > 0)), None)

res = {"candidate_id": "divergence-source-20260824-01", "formal_claim": False,
       "prompt_tokens": N, "splits_compared": [[N], [512, N - 512]],
       "first_layer_with_difference": first,
       "layers": layers,
       "final_logit_max_abs_diff": logit_diff,
       "top1_top2_logit_gap": gap_a,
       "difference_can_flip_choice": logit_diff > gap_a,
       "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
Path("experiments/divergence_source/results.json").write_text(json.dumps(res, indent=2))
print("erste abweichende Schicht:", first)
print(f"{'Schicht':>8}{'Typ':>18}{'keys max':>12}{'values max':>12}")
for l in layers[:6] + layers[-3:]:
    k = l["keys"]["max_abs"] if l["keys"] else None
    v = l["values"]["max_abs"] if l["values"] else None
    print(f"{l['layer']:>8}{l['type']:>18}{k if k is None else round(k,6):>12}"
          f"{v if v is None else round(v,6):>12}")
print(f"\nLogit-Unterschied letzte Position: {logit_diff:.6f}")
print(f"Abstand Top-1 zu Top-2:           {gap_a:.6f}")
print(f"kann die Wahl kippen:             {logit_diff > gap_a}")
