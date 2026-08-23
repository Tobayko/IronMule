"""Gibt es eine Prefill-Blockgroesse, die tokenidentisch zum Einzelblock bleibt?

Zyklus 1 zeigte, dass dieselben Token, anders gestueckelt, andere Ausgaben erzeugen
koennen. Vier Kandidaten der Liste haengen daran, weil sie alle die Blockstruktur
veraendern. Vorregistrierung: experiments/chunk_identity/PREREGISTRATION.md
"""
import sys, json, time
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

GEN = 16
CHUNKS = (64, 128, 256, 512, 1024)
TARGETS = (300, 680, 1200, 2000)

require_ac_power()
guard = BudgetGuard()

# Ruhezeit wird aufsummiert statt je Block aufgerundet. Bei Blockgroesse 64 und
# 2000 Token sind das 32 Bloecke; eine 4-s-Mindestpause je Block ergaebe 128 s Ruhe
# fuer eine einzige Zelle und liesse die Blockgroesse selbst zum Zeitfresser werden.
# Die Schuld bleibt erhalten, nur ihre Abtragung wird gebuendelt.
_debt = 0.0
def charge(seconds):
    global _debt
    guard.record_gpu(seconds)
    _debt += seconds * (1 - 0.15) / 0.15
    while _debt >= 4.0:
        guard.required_break()
        _debt -= 4.0

snap = resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
model, tok = load(str(snap.path))
sampler = make_sampler(temp=0.0)

def prompt_of(target):
    """Prompt nahe `target` Token, aus wiederholtem Fliesstext."""
    unit = ("You are a careful engineering assistant working in a Python repository. "
            "Follow the existing style and explain your reasoning briefly. ")
    reps = max(1, target // 22)
    body = unit * reps + "\n\nWhy is false sharing slow?"
    txt = tok.apply_chat_template([{"role": "user", "content": body}],
                                  add_generation_prompt=True)
    return list(txt if isinstance(txt, list) else tok.encode(txt))

def run(ids, chunk):
    """Prefill in Bloecken von `chunk` (oder ein Block), dann `GEN` Token greedy."""
    cache = make_prompt_cache(model)
    size = len(ids) if chunk is None else chunk
    logits = None
    for start in range(0, len(ids), size):
        piece = ids[start:start + size]
        at = time.perf_counter()
        logits = model(mx.array([piece]), cache=cache)
        mx.eval(logits); mx.synchronize()
        charge(time.perf_counter() - at)
    at = time.perf_counter()
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(y)
    out = [int(y[0, 0])]
    for _ in range(GEN - 1):
        logits = model(y, cache=cache)
        y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
        mx.eval(y)
        out.append(int(y[0, 0]))
    mx.synchronize(); charge(time.perf_counter() - at)
    blocks = 1 if chunk is None else -(-len(ids) // size)
    return out, blocks

rows = []
for target in TARGETS:
    ids = prompt_of(target)
    reference, _ = run(ids, None)
    for chunk in CHUNKS:
        if chunk >= len(ids):
            rows.append({"prompt_tokens": len(ids), "chunk": chunk, "blocks": 1,
                         "identical": True, "first_diff": None,
                         "note": "Block groesser als Prompt, entspricht Einzelblock"})
            print(rows[-1], flush=True); continue
        out, blocks = run(ids, chunk)
        diff = next((i for i, (a, b) in enumerate(zip(reference, out)) if a != b), None)
        rows.append({"prompt_tokens": len(ids), "chunk": chunk, "blocks": blocks,
                     "identical": out == reference, "first_diff": diff})
        print(rows[-1], flush=True)

# Welche Blockgroesse haelt ueber ALLE geprueften Laengen?
holds = {c: all(r["identical"] for r in rows
                if r["chunk"] == c and not r.get("note")) for c in CHUNKS}
tested = {c: sum(1 for r in rows if r["chunk"] == c and not r.get("note")) for c in CHUNKS}
result = {"candidate_id": "chunk-identity-20260824-01", "formal_claim": False,
          "generate_tokens": GEN, "rows": rows,
          "holds_across_all_lengths": {str(c): holds[c] for c in CHUNKS},
          "lengths_actually_tested": {str(c): tested[c] for c in CHUNKS},
          "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
          "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
Path("experiments/chunk_identity/results.json").write_text(json.dumps(result, indent=2))
print("\nhaelt ueber alle geprueften Laengen:", result["holds_across_all_lengths"])
print("davon tatsaechlich geprueft:", result["lengths_actually_tested"])
