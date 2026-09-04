"""Was kostet der LM-Head auf Prefill-Positionen, die niemand liest?

gemma3_text.Model.__call__ wendet lm_head auf alle Positionen an; beim Prefill wird
genau eine Zeile gelesen. Vorregistrierung: experiments/head_skip/PREREGISTRATION.md
"""
import sys, json, time, statistics
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

GEN, REPS = 32, 2
CHUNKS = (128, 256, 512)

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
inner = model.language_model if hasattr(model, "language_model") else model
body, head = inner.model, inner.lm_head
sampler = make_sampler(temp=0.0)

FILLER = ("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ") * 40
txt = tok.apply_chat_template(
    [{"role": "user", "content": FILLER + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
IDS = list(txt if isinstance(txt, list) else tok.encode(txt))

def prefill(chunk, skip_head):
    """skip_head: Head nur auf der letzten Position des letzten Blocks."""
    cache = make_prompt_cache(model)
    at = time.perf_counter()
    logits = None
    for s in range(0, len(IDS), chunk):
        piece = mx.array([IDS[s:s + chunk]])
        if skip_head:
            h = body(piece, cache=cache)
            last = s + chunk >= len(IDS)
            logits = head(h[:, -1:, :]) if last else None
            mx.eval(logits if last else h)
        else:
            logits = model(piece, cache=cache)
            mx.eval(logits)
        mx.synchronize()
    took = time.perf_counter() - at
    charge(took)
    return cache, logits, took

def generate(chunk, skip_head):
    cache, logits, pre = prefill(chunk, skip_head)
    at = time.perf_counter()
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]; mx.eval(y)
    out = [int(y[0, 0])]
    for _ in range(GEN - 1):
        lg = model(y, cache=cache)
        y = sampler(lg[:, -1, :].astype(mx.float32))[:, None]; mx.eval(y)
        out.append(int(y[0, 0]))
    mx.synchronize(); charge(time.perf_counter() - at)
    return out, pre

generate(256, False)  # Aufwaermen

rows, toks = [], {}
for chunk in CHUNKS:
    times = {False: [], True: []}
    for _ in range(REPS):
        for skip in (False, True):
            out, pre = generate(chunk, skip)
            times[skip].append(pre)
            toks.setdefault((chunk, skip), out)
    full, lean = statistics.median(times[False]), statistics.median(times[True])
    rows.append({"chunk": chunk,
                 "prefill_full_head_s": round(full, 4),
                 "prefill_head_last_only_s": round(lean, 4),
                 "head_share_of_prefill": round(1 - lean / full, 4),
                 "identical": toks[(chunk, True)] == toks[(chunk, False)]})
    print(rows[-1], flush=True)

res = {"candidate_id": "prefill-head-skip-20260824-01", "formal_claim": False,
       "prompt_tokens": len(IDS), "generate_tokens": GEN, "reps": REPS,
       "rows": rows,
       "identical_across_all": all(r["identical"] for r in rows),
       "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
Path("experiments/head_skip/results.json").write_text(json.dumps(res, indent=2))
print("\n identisch ueberall:", res["identical_across_all"])
