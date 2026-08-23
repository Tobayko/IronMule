"""Is the 23.8% dispatch saving real at constant shapes, or was all of it the stale
trace? A fixed-window cache would hold shapes constant, which is the precondition a
growing cache violates. Freezing the offset reproduces that shape profile exactly --
each step writes to the same slot and reads back the same extent -- so compilation
becomes legitimate and the output can be checked against the eager path."""
import sys, time, statistics, json
sys.path.insert(0,"tools")
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

CTX = 256

def run(mid, reps=15):
    snap=resolve_local_model_snapshot(mid); m,_=load(str(snap.path))
    body=(m.language_model if hasattr(m,"language_model") else m).model
    layers=len(body.layers)
    gb=sum(p.size*p.dtype.size for _,p in tree_flatten(body.parameters()))/1e9
    cache=make_prompt_cache(body)
    mx.eval(body(mx.array([[1]*CTX]),cache=cache)); mx.synchronize()
    frozen=[c.offset for c in cache]
    step=mx.array([[7]])

    def freeze():
        for c,o in zip(cache,frozen): c.offset=o

    def eager(x):
        freeze(); return body(x,cache=cache)
    # Declare the cache tensors as captured state. Without this MLX refuses the
    # compile outright, which is the correct behaviour and better than the silent
    # wrong answer the growing-cache version produced.
    state=[]
    for c in cache: state += [c.keys, c.values]
    compiled_inner = mx.compile(lambda x: body(x, cache=cache),
                                inputs=state, outputs=state)
    def compiled(x):
        freeze(); return compiled_inner(x)

    # correctness before timing
    freeze(); a=body(step,cache=cache); mx.eval(a)
    for _ in range(3): mx.eval(compiled(step))
    b=compiled(step); mx.eval(b); mx.synchronize()
    same = bool(mx.all(a==b).item()); maxerr=float(mx.max(mx.abs(a-b)).item())

    out={}
    for name,fn in (("eager",eager),("compiled",compiled)):
        for _ in range(5): mx.eval(fn(step))
        mx.synchronize()
        s=[]; t0=time.perf_counter()
        for _ in range(reps):
            t=time.perf_counter_ns(); mx.eval(fn(step)); mx.synchronize()
            s.append(time.perf_counter_ns()-t)
        charge(time.perf_counter()-t0)
        out[name]=round(statistics.median(s)/1e6,4)
    del m; mx.clear_cache()
    return {"layers":layers,"weight_gb":round(gb,6),"identisch":same,
            "max_abs_error":maxerr, **out,
            "delta_pct":round((out["compiled"]/out["eager"]-1)*100,2)}

res={}
for mid,lbl in (("mlx-community/gemma-3-4b-it-4bit","4b"),
                ("mlx-community/gemma-3-1b-it-4bit","1b")):
    res[lbl]=run(mid); print(lbl, res[lbl], flush=True)

from measure_device_model import fit_device_model
for mode in ("eager","compiled"):
    pts=[{"layers":res[k]["layers"],"weight_gb":res[k]["weight_gb"],"ms":res[k][mode]} for k in ("4b","1b")]
    res[f"fit_{mode}"]=fit_device_model(pts)
open(f"{sys.argv[1]}/fixedshape.json","w").write(json.dumps(res,indent=2,default=float))
