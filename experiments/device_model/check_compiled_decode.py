import sys, time
sys.path.insert(0,"tools")
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
require_ac_power(); g=BudgetGuard()
snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
m,tok=load(str(snap.path)); sampler=make_sampler(temp=0.0)
txt=tok.apply_chat_template([{"role":"user","content":
    "Explain in three sentences how a CPU cache line works."}],add_generation_prompt=True)
ids=list(txt if isinstance(txt,list) else tok.encode(txt))

def run(compiled, n=48):
    cache=make_prompt_cache(m)
    y=mx.array([ids])
    lg=m(y,cache=cache); mx.eval(lg); mx.synchronize()
    y=sampler(lg[:,-1,:].astype(mx.float32))[:,None]; mx.eval(y)
    step_fn = mx.compile(lambda x: m(x, cache=cache)) if compiled else (lambda x: m(x, cache=cache))
    outs=[int(y[0,0])]
    t0=time.perf_counter()
    for _ in range(n):
        lg=step_fn(y)
        y=sampler(lg[:,-1,:].astype(mx.float32))[:,None]
        mx.eval(y); outs.append(int(y[0,0]))
    mx.synchronize()
    w=time.perf_counter()-t0; g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()
    return outs, w/n*1000

a,ta = run(False); b,tb = run(True)
print("Token identisch:", a==b, f"({len(a)} Token)")
print(f"eager    {ta:.3f} ms/Token")
print(f"compiled {tb:.3f} ms/Token   -> {(1-tb/ta)*100:+.1f}%")
if a!=b:
    diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    print("erste Abweichung bei Position", diff[0] if diff else "?")
print("Text gleich:", tok.decode(a)==tok.decode(b))
