"""KV-Cache-Reallokation im laufenden Decode lokalisieren.

Misst je Decodeschritt Zeit UND die Form jedes der 34 Caches. Eine Formaenderung
ist eine beobachtete Reallokation -- vorhergesagte Positionen werden erst danach
verglichen. Liefert nebenbei die vom Auftrag verlangten p50/p95/p99 der
Inter-Token-Latenz, die der Baseline bisher fehlten.

Vorregistrierung: experiments/kv_realloc/PREREGISTRATION.md
"""
import sys, json, math, time, statistics
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

STEPS, REPS, CHUNK = 48, 8, 256
PROMPT_RANGE = (757, 767)          # vorregistriert, faellt sonst geschlossen aus
BANDWIDTH_B_S = 358.4e9            # Geraetemodell, Zyklus 4
BYTES_PER_POS = 4 * 256 * 2        # n_kv_heads * head_dim * bfloat16

if "--execute" not in sys.argv:
    print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
    raise SystemExit(78)

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

UNIT = ("You are a careful engineering assistant working in a Python repository. "
        "Follow the existing style and explain your reasoning briefly. ")
txt = tok.apply_chat_template(
    [{"role": "user", "content": UNIT * 34 + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
IDS = list(txt if isinstance(txt, list) else tok.encode(txt))
if not PROMPT_RANGE[0] <= len(IDS) <= PROMPT_RANGE[1]:
    raise SystemExit(f"refused: prompt is {len(IDS)} tokens, registered {PROMPT_RANGE}")


def run():
    """Ein Prefill plus STEPS Decodeschritte, je Schritt Zeit und Cacheformen."""
    cache = make_prompt_cache(model)
    logits = None
    for s in range(0, len(IDS), CHUNK):
        at = time.perf_counter()
        logits = model(mx.array([IDS[s:s + CHUNK]]), cache=cache)
        mx.eval(logits); mx.synchronize()
        charge(time.perf_counter() - at)
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(y)

    prev = [c.keys.shape[2] for c in cache]          # .shape synchronisiert nicht
    after_prefill = list(prev)
    out, itl, events = [int(y[0, 0])], [], []
    for _ in range(STEPS):
        at = time.perf_counter()
        logits = model(y, cache=cache)
        y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
        mx.eval(y); mx.synchronize()
        itl.append(time.perf_counter() - at)          # vor charge(), sonst Ruhezeit drin
        now = [c.keys.shape[2] for c in cache]
        events.append([j for j, (a, b) in enumerate(zip(prev, now)) if a != b])
        prev = now
        out.append(int(y[0, 0]))
        if out[-1] in stops:
            break
    charge(sum(itl))
    return out, itl, events, after_prefill, prev


def pct(values, q):
    """Nearest-rank Perzentil. math.ceil, weil round() hier um eins danebenlag."""
    s = sorted(values)
    rank = math.ceil(q / 100 * len(s) - 1e-9)
    return s[min(len(s) - 1, max(0, rank - 1))]


run()  # Aufwaermen, verworfen
runs = [run() for _ in range(REPS)]

tokens = [r[0] for r in runs]
identical = all(t == tokens[0] for t in tokens)
events0 = runs[0][2]
consistent = all(r[2] == events0 for r in runs)
realloc_steps = [i for i, e in enumerate(events0) if e]

by_step = list(zip(*[r[1] for r in runs]))            # je Schrittindex REPS Werte
med = [statistics.median(v) * 1000 for v in by_step]
spread = [(max(v) - min(v)) * 1000 for v in by_step]
plain = [med[i] for i in range(len(med)) if i not in realloc_steps]
baseline = statistics.median(plain)

layers_global = [j for j in range(34) if j % 6 == 5]
observed = {}
for i in realloc_steps:
    g = sum(1 for j in events0[i] if j in layers_global)
    observed[i + 1] = {"layers": len(events0[i]), "global_layers": g,
                       "rotating_layers": len(events0[i]) - g,
                       "excess_ms": round(med[i] - baseline, 4),
                       "width_before": runs[0][3][events0[i][0]] if i == 0 else None}

predicted = {}
for name, step, n_layers, width in (("rotating", 1, 29, len(IDS) + 256),
                                    ("global", 768 - len(IDS) + 1, 5, 1024)):
    traffic = n_layers * 2 * (2 * width + 256) * BYTES_PER_POS
    predicted[step] = {"class": name, "layers": n_layers,
                       "traffic_mb": round(traffic / 1e6, 1),
                       "expected_ms": round(traffic / BANDWIDTH_B_S * 1e3, 4)}

excess_total = sum(med[i] - baseline for i in realloc_steps)
decode_total = sum(med)
flat = [v * 1000 for r in runs for v in r[1]]

res = {
    "candidate_id": "kv-cache-realloc-20260824-01", "formal_claim": False,
    "prompt_tokens": len(IDS), "steps": STEPS, "reps": REPS, "chunk": CHUNK,
    "cache_widths_after_prefill": {"rotating": runs[0][3][0], "global": runs[0][3][5]},
    "cache_widths_after_decode": {"rotating": runs[0][4][0], "global": runs[0][4][5]},
    "realloc_steps_observed": [i + 1 for i in realloc_steps],
    "realloc_steps_predicted": sorted(predicted),
    "prediction_matched": sorted(predicted) == [i + 1 for i in realloc_steps],
    "events_consistent_across_reps": consistent,
    "observed": observed, "predicted": predicted,
    "baseline_ms_per_step": round(baseline, 4),
    "excess_at_realloc_ms": round(excess_total, 4),
    "excess_share_of_decode_pct": round(excess_total / decode_total * 100, 4),
    "noise_band_ms": {"median_step_spread": round(statistics.median(spread), 4),
                      "max_step_spread": round(max(spread), 4)},
    "itl_ms": {"p50": round(pct(flat, 50), 4), "p95": round(pct(flat, 95), 4),
               "p99": round(pct(flat, 99), 4),
               "min": round(min(flat), 4), "max": round(max(flat), 4)},
    "median_ms_per_step": [round(v, 4) for v in med],
    "token_identity": identical,
    "tokens_produced": len(tokens[0]),
    "budget": {k: v for k, v in guard.summary().items() if "limit" not in k},
}
Path("experiments/kv_realloc/results.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items()
                  if k not in ("budget", "median_ms_per_step")}, indent=2))
