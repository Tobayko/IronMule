"""TTFT nach Klassen getrennt. Ein gemeinsamer Schaetzer ueber cold_process,
warm_uncached und warm_prefix_hit waere bedeutungslos -- die drei unterscheiden
sich um Groessenordnungen und haben verschiedene Ursachen."""
import sys, time, json, statistics
sys.path.insert(0,"tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
t_load=time.perf_counter(); m,tok=load(str(snap.path)); load_s=time.perf_counter()-t_load
sampler=make_sampler(temp=0.0)

PREFIX=("You are a careful engineering assistant working in a Python repository. "
        "Follow the existing style. Explain your reasoning briefly. ")*40
QUESTIONS=["What does a cache line do?","Why is false sharing slow?",
           "When is a TLB miss expensive?","What is store forwarding?"]

def ids_for(q):
    txt=tok.apply_chat_template([{"role":"system","content":PREFIX},
                                 {"role":"user","content":q}],add_generation_prompt=True)
    return list(txt if isinstance(txt,list) else tok.encode(txt))

def common_prefix(seqs):
    """Gemma 3 hat keine eigene System-Rolle -- der Systemtext wird in den User-Turn
    gemischt. Ein wiederverwendbares Praefix ist deshalb das gemeinsame Token-Praefix
    der fertig gerenderten Prompts, nicht eine 'System-Message'."""
    n=0
    for col in zip(*seqs):
        if len(set(col))!=1: break
        n+=1
    return n

def ttft(ids, cache=None, reuse=0):
    """Zeit bis zum ersten Token. reuse = bereits im Cache liegende Praefixlaenge."""
    c = cache if cache is not None else make_prompt_cache(m)
    rest = ids[reuse:]
    t0=time.perf_counter()
    lg = m(mx.array([rest]), cache=c)
    first = sampler(lg[:,-1,:].astype(mx.float32))
    mx.eval(first); mx.synchronize()
    d=time.perf_counter()-t0
    return d, int(first[0]), c

all_ids=[ids_for(q) for q in QUESTIONS]
full=all_ids[0]
common=common_prefix(all_ids)
print(f"Prompt {len(full)} Token, gemeinsames Systempraefix {common} Token")

# Aufwaermen
d,_,_ = ttft(full); charge(d)

warm_unc=[]
for _ in range(3):
    d,tk,_=ttft(full); charge(d); warm_unc.append(d)

# Praefix einmal vorrechnen, dann je Frage nur den Rest
hits=[]
for qi in all_ids:
    c=make_prompt_cache(m)
    t=time.perf_counter(); o=m(mx.array([full[:common]]),cache=c); mx.eval(o); mx.synchronize()
    charge(time.perf_counter()-t)
    d,tk,_=ttft(qi, cache=c, reuse=common); charge(d)
    hits.append((len(qi)-common, d))

res={"model_load_s":round(load_s,4),
     "prompt_tokens":len(full),"shared_prefix_tokens":common,
     "warm_uncached_ttft_ms":round(statistics.median(warm_unc)*1000,2),
     "warm_prefix_hit":[{"rest_tokens":r,"ttft_ms":round(d*1000,2)} for r,d in hits],
     "warm_prefix_hit_median_ms":round(statistics.median(d for _,d in hits)*1000,2),
     "peak_gb":round(mx.get_peak_memory()/1e9,3),
     "budget":{k:v for k,v in g.summary().items() if "limit" not in k}}
res["prefix_hit_speedup"]=round(res["warm_uncached_ttft_ms"]/res["warm_prefix_hit_median_ms"],2)
Path("experiments/ttft/ttft_classes.json").write_text(json.dumps(res,indent=2))
print(json.dumps({k:v for k,v in res.items() if k!="budget"},indent=2))
