"""Haelt Blockgroesse 256 wirklich, oder waren vier Laengen zu wenig?
Bei einer Ausfallrate von rund 29 % je Zelle ist 4/4 unauffaellig."""
import sys, json, time
sys.path.insert(0,"tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
GEN=16; CHUNKS=(256,1024)
TARGETS=(450, 900, 1500, 2500, 3000, 3500)
require_ac_power(); guard=BudgetGuard(); _debt=0.0
def charge(sec):
    global _debt
    guard.record_gpu(sec); _debt += sec*(1-0.15)/0.15
    while _debt>=4.0: guard.required_break(); _debt-=4.0
snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
model,tok=load(str(snap.path)); sampler=make_sampler(temp=0.0)
def prompt_of(t):
    unit=("You are a careful engineering assistant working in a Python repository. "
          "Follow the existing style and explain your reasoning briefly. ")
    body=unit*max(1,t//22)+"\n\nWhy is false sharing slow?"
    x=tok.apply_chat_template([{"role":"user","content":body}],add_generation_prompt=True)
    return list(x if isinstance(x,list) else tok.encode(x))
def run(ids, chunk):
    c=make_prompt_cache(model); size=len(ids) if chunk is None else chunk; lg=None
    for s in range(0,len(ids),size):
        at=time.perf_counter(); lg=model(mx.array([ids[s:s+size]]),cache=c)
        mx.eval(lg); mx.synchronize(); charge(time.perf_counter()-at)
    at=time.perf_counter()
    y=sampler(lg[:,-1,:].astype(mx.float32))[:,None]; mx.eval(y); out=[int(y[0,0])]
    for _ in range(GEN-1):
        lg=model(y,cache=c); y=sampler(lg[:,-1,:].astype(mx.float32))[:,None]
        mx.eval(y); out.append(int(y[0,0]))
    mx.synchronize(); charge(time.perf_counter()-at); return out
rows=[]
for t in TARGETS:
    ids=prompt_of(t); ref=run(ids,None)
    for ch in CHUNKS:
        if ch>=len(ids):
            rows.append({"prompt_tokens":len(ids),"chunk":ch,"identical":True,
                         "note":"=Einzelblock"}); print(rows[-1],flush=True); continue
        o=run(ids,ch)
        d=next((i for i,(a,b) in enumerate(zip(ref,o)) if a!=b),None)
        rows.append({"prompt_tokens":len(ids),"chunk":ch,"blocks":-(-len(ids)//ch),
                     "identical":o==ref,"first_diff":d})
        print(rows[-1],flush=True)
res={"candidate_id":"chunk-identity-20260824-01","phase":"confirmation","formal_claim":False,
     "rows":rows,"budget":{k:v for k,v in guard.summary().items() if "limit" not in k}}
for ch in CHUNKS:
    real=[r for r in rows if r["chunk"]==ch and not r.get("note")]
    res[f"chunk_{ch}_holds"]={"tested":len(real),"identical":sum(r["identical"] for r in real)}
Path("experiments/chunk_identity/confirmation.json").write_text(json.dumps(res,indent=2))
print("\n", {k:v for k,v in res.items() if k.startswith("chunk_")})
