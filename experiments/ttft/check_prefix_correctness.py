"""Liefert der wiederverwendete Praefix-Cache dieselben Token wie ein frischer?

Gemma 3 haelt 29 der 34 Layer in einem rotierenden Cache mit Fenster 1024. Ein
Praefix, das darueber hinausgeht, ist im Cache nur noch teilweise vorhanden -- die
Wiederverwendung waere dann NICHT aequivalent. Diese Grenze wird gesucht, nicht
angenommen."""
import sys, json, time
sys.path.insert(0,"tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
CHUNK, GEN = 512, 24
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
m,tok=load(str(snap.path)); sampler=make_sampler(temp=0.0)

def prefill(ids, cache):
    for i in range(0, len(ids), CHUNK):
        t=time.perf_counter()
        o=m(mx.array([ids[i:i+CHUNK]]), cache=cache); mx.eval(o); mx.synchronize()
        charge(time.perf_counter()-t)
    return o

def gen(ids, cache=None, reuse=0, n=GEN):
    """Prefill und Decode werden getrennt verbucht. Beides als einen Lastblock zu
    zaehlen meldet eine kontinuierliche Last, die es nie gab, und reisst bei langen
    Praefixen die 6-s-Grenze auf Arbeit, die tatsaechlich gestueckelt lief."""
    c = cache if cache is not None else make_prompt_cache(m)
    rest = ids[reuse:]
    lg = prefill(rest, c)                      # verbucht sich selbst je Block
    t=time.perf_counter()
    y = sampler(lg[:,-1,:].astype(mx.float32))[:,None]; mx.eval(y)
    out=[int(y[0,0])]
    for _ in range(n-1):
        lg = m(y, cache=c)
        y = sampler(lg[:,-1,:].astype(mx.float32))[:,None]; mx.eval(y)
        out.append(int(y[0,0]))
    mx.synchronize(); charge(time.perf_counter()-t)
    return out

rows=[]
for reps in (30, 60, 120, 200):
    P=("You are a careful engineering assistant working in a Python repository. "
       "Follow the existing style and explain your reasoning briefly. ")*reps
    A=[{"role":"user","content":P+"\n\nWhat does a cache line do?"}]
    B=[{"role":"user","content":P+"\n\nWhy is false sharing slow?"}]
    ia,ib=[list(x if isinstance(x,list) else tok.encode(x)) for x in
           (tok.apply_chat_template(A,add_generation_prompt=True),
            tok.apply_chat_template(B,add_generation_prompt=True))]
    common=0
    for x,y in zip(ia,ib):
        if x!=y: break
        common+=1
    fresh = gen(ib)                                   # voller Prefill
    c=make_prompt_cache(m); prefill(ib[:common], c)   # Praefix vorrechnen
    reused = gen(ib, cache=c, reuse=common)           # nur den Rest
    rows.append({"prefix_tokens":common,"prompt_tokens":len(ib),
                 "identical":fresh==reused,
                 "first_diff":next((i for i,(x,y) in enumerate(zip(fresh,reused)) if x!=y), None)})
    print(rows[-1], flush=True)
Path("experiments/ttft/prefix_correctness.json").write_text(
    json.dumps({"rows":rows,"window_note":"Gemma 3 4B: 29/34 Layer rotierend, Fenster 1024",
                "budget":{k:v for k,v in g.summary().items() if "limit" not in k}},indent=2))
