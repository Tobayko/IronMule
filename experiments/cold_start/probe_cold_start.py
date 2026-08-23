"""Ein Kaltstart, in seine Anteile zerlegt. Laeuft als eigener Prozess und beendet
sich; der Elternprozess ruft ihn mehrfach auf. Gibt eine JSON-Zeile auf stdout.

Vorregistrierung: experiments/cold_start/PREREGISTRATION.md
"""
import time
_t_entry = time.perf_counter()          # erste ausfuehrbare Zeile
import json, os, resource, sys

_t_stdlib = time.perf_counter()
import mlx.core as mx
_t_mlx = time.perf_counter()
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
_t_mlxlm = time.perf_counter()

sys.path.insert(0, "tools")
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power

require_ac_power()
guard = BudgetGuard()
_debt = 0.0
def charge(sec):
    global _debt
    guard.record_gpu(sec); _debt += sec * (1 - 0.15) / 0.15
    while _debt >= 4.0:
        guard.required_break(); _debt -= 4.0

_t0 = time.perf_counter()
snap = resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
_t_resolve = time.perf_counter()
model, tok = load(str(snap.path))
_t_load = time.perf_counter()

sampler = make_sampler(temp=0.0)
unit = ("You are a careful engineering assistant working in a Python repository. "
        "Follow the existing style and explain your reasoning briefly. ")
txt = tok.apply_chat_template(
    [{"role": "user", "content": unit * 40 + "\n\nWhy is false sharing slow?"}],
    add_generation_prompt=True)
ids = list(txt if isinstance(txt, list) else tok.encode(txt))

# Warm-up: ein einzelnes Token, damit Allokation und Kernelaufbau hier anfallen
# und nicht spaeter dem Prefill zugerechnet werden.
from mlx_lm.models.cache import make_prompt_cache
c = make_prompt_cache(model)
a = time.perf_counter()
o = model(mx.array([[ids[0]]]), cache=c); mx.eval(o); mx.synchronize()
_warm1 = time.perf_counter() - a; charge(_warm1)

a = time.perf_counter()
o = model(mx.array([[ids[1]]]), cache=c); mx.eval(o); mx.synchronize()
_warm2 = time.perf_counter() - a; charge(_warm2)

# Prefill des vollen Prompts bis zum ersten Token, frischer Cache
c2 = make_prompt_cache(model)
a = time.perf_counter()
for s in range(0, len(ids), 256):
    o = model(mx.array([ids[s:s+256]]), cache=c2); mx.eval(o); mx.synchronize()
y = sampler(o[:, -1, :].astype(mx.float32)); mx.eval(y)
_prefill = time.perf_counter() - a; charge(_prefill)

# Interpreterstart: Differenz zwischen Elternstempel (argv) und erster Zeile hier.
parent_spawn = float(sys.argv[1]) if len(sys.argv) > 1 else None
print(json.dumps({
    "interpreter_start_s": None if parent_spawn is None
                            else round(time.time() - parent_spawn - (time.perf_counter() - _t_entry), 4),
    "stdlib_import_s": round(_t_stdlib - _t_entry, 4),
    "import_mlx_s": round(_t_mlx - _t_stdlib, 4),
    "import_mlx_lm_s": round(_t_mlxlm - _t_mlx, 4),
    "snapshot_resolve_s": round(_t_resolve - _t0, 4),
    "model_load_s": round(_t_load - _t_resolve, 4),
    "warmup_first_forward_s": round(_warm1, 4),
    "second_forward_s": round(_warm2, 4),
    "prefill_to_first_token_s": round(_prefill, 4),
    "prompt_tokens": len(ids),
    "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
}))
