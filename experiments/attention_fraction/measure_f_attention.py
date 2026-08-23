"""f_attention: the share of a decode step that attention actually owns.

Amdahl bounds every attention-kernel project by this number, so it is measured
before any kernel is written. Attention is the only part of a decode step whose
cost grows with KV length; FFN and projections do not care. So the slope of step
time against context length isolates attention without needing a profiler."""
import sys, time, json, statistics
sys.path.insert(0,"tools")
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
m,tok=load(str(snap.path))
print("n_layers", len(m.layers) if hasattr(m,'layers') else '?')
a=m.layers[0].self_attn if hasattr(m,'layers') else None
if a is not None:
    print("n_heads",a.n_heads,"n_kv_heads",a.n_kv_heads,"head_dim",a.head_dim)
    print("sliding pattern:", [bool(m.layers[i].self_attn.is_sliding) for i in range(min(12,len(m.layers)))])

CHUNK=1024; TARGET=16384
cache=make_prompt_cache(m)
ctx_len=0
out={}
def decode_ms(cache, n=10):
    step=mx.array([[7]])
    for _ in range(2):
        o=m(step,cache=cache); mx.eval(o)
    mx.synchronize()
    s=[]
    t0=time.perf_counter()
    for _ in range(n):
        t=time.perf_counter_ns()
        o=m(step,cache=cache); mx.eval(o); mx.synchronize()
        s.append(time.perf_counter_ns()-t)
    w=time.perf_counter()-t0; g.record_gpu(w)
    for _ in range(int(-(-(w*4)//4))): g.required_break()
    return statistics.median(s)/1e6

out[0]=decode_ms(cache)
while ctx_len < TARGET:
    chunk=mx.array([[7]*CHUNK])
    t=time.perf_counter()
    o=m(chunk,cache=cache); mx.eval(o); mx.synchronize()
    w=time.perf_counter()-t; g.record_gpu(w)
    for _ in range(int(-(-(w*4)//4))): g.required_break()
    ctx_len+=CHUNK
    if ctx_len in (1024,2048,4096,8192,16384):
        out[ctx_len]=decode_ms(cache)
        print(f"ctx={ctx_len:>6} step={out[ctx_len]:.3f} ms  peak={mx.get_peak_memory()/1e9:.2f} GB")

base=out[0]
res={"decode_ms_by_ctx":{k:round(v,4) for k,v in out.items()},
     "f_attention":{k:round((v-base)/v,4) for k,v in out.items() if k},
     "fixed_ms":round(base,4),
     "peak_gb":round(mx.get_peak_memory()/1e9,3),
     "budget":{k:v for k,v in g.summary().items() if "limit" not in k}}
print(json.dumps(res,indent=2))
