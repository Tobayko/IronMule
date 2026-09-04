"""Aendert allein die Zerteilung des Prefills die erzeugten Token?
Kein Praefix-Cache im Spiel -- nur dieselben Token, anders gestueckelt."""
import sys, json, time
sys.path.insert(0,"tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
require_ac_power(); g=BudgetGuard()
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()
snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
m,tok=load(str(snap.path)); sampler=make_sampler(temp=0.0)

P=("You are a careful engineering assistant working in a Python repository. "
   "Follow the existing style and explain your reasoning briefly. ")*30
txt=tok.apply_chat_template([{"role":"user","content":P+"\n\nWhy is false sharing slow?"}],
                            add_generation_prompt=True)
ids=list(txt if isinstance(txt,list) else tok.encode(txt))
N=len(ids); print("Prompt:", N, "Token")

def run(splits, n=24):
    c=make_prompt_cache(m); pos=0
    for s in splits:
        piece=ids[pos:pos+s]; pos+=s
        if not piece: continue
        t=time.perf_counter(); o=m(mx.array([piece]),cache=c); mx.eval(o); mx.synchronize()
        charge(time.perf_counter()-t)
    t=time.perf_counter()
    y=sampler(o[:,-1,:].astype(mx.float32))[:,None]; mx.eval(y); out=[int(y[0,0])]
    for _ in range(n-1):
        o=m(y,cache=c); y=sampler(o[:,-1,:].astype(mx.float32))[:,None]; mx.eval(y)
        out.append(int(y[0,0]))
    mx.synchronize(); charge(time.perf_counter()-t)
    return out

cases={"ein Block":[N], "512+Rest":[512,N-512], "Praefix+11":[N-11,11],
       "256er":[256]*(N//256)+[N%256], "128er":[128]*(N//128)+[N%128]}
ref=None; rows=[]
for lbl,sp in cases.items():
    o=run(sp)
    if ref is None: ref=o
    d=next((i for i,(x,y) in enumerate(zip(ref,o)) if x!=y), None)
    rows.append({"zerteilung":lbl,"bloecke":len([x for x in sp if x]),
                 "identisch_zu_ein_block":o==ref,"erste_abweichung":d})
    print(rows[-1], flush=True)
Path("experiments/ttft/prefill_chunking.json").write_text(json.dumps(
    {"prompt_tokens":N,"rows":rows,
     "budget":{k:v for k,v in g.summary().items() if "limit" not in k}},indent=2))
