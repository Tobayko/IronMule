"""Prefill is the other half of inference and none of this work has touched it.
A coding agent sends long prompts and takes short answers, so its clock is
time-to-first-token, not tokens per second. Prefill is also a different shape from
batched decode -- one long row instead of many short ones -- so the width policy
derived for decode says nothing about it."""
import sys, time, json, statistics
sys.path.insert(0,"tools")
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
m,tok=load(str(snap.path))

def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

def prefill(total, chunk):
    """Fill `total` positions in blocks of `chunk`; report seconds and positions/s."""
    # Sum kernel time only. Timing the whole loop charges the guard's verified
    # sleeps to the model, which reported 13.6 ms per position where the real
    # figure is under 2 -- the rests are seven times the work by design.
    cache=make_prompt_cache(m)
    done=0; gpu=0.0
    while done < total:
        n=min(chunk, total-done)
        piece=mx.array([[7]*n])
        at=time.perf_counter()
        out=m(piece, cache=cache); mx.eval(out); mx.synchronize()
        w=time.perf_counter()-at
        gpu+=w; charge(w)
        done+=n
    return gpu

TOTAL=2048
out={}
for chunk in (256,512,1024,2048):
    best=prefill(TOTAL,chunk)
    out[chunk]={"seconds":round(best,3),"positions_per_second":round(TOTAL/best,1),
                "ms_per_position":round(best/TOTAL*1000,4)}
    print(chunk, out[chunk], flush=True)
out["_peak_gb"]=round(mx.get_peak_memory()/1e9,3)
out["_budget"]={k:v for k,v in g.summary().items() if "limit" not in k}
open("/private/tmp/claude-501/-Users-tobiasburandt-Project-Friday/96bb49e3-a7ef-4bd1-a68c-e23ff5b77ec6/scratchpad/prefill.json","w").write(json.dumps(out,indent=2))
