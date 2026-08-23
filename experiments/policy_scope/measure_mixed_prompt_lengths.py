"""Every batch measurement so far replicated one prompt. Real batches do not.
Unequal prompt lengths force padding, and padding is work the GPU does for nothing.
If that eats the 5.35x, the headline number describes a workload nobody runs."""
import sys, time, json, statistics, random
sys.path.insert(0,"tools")
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
snap=resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
m,tok=load(str(snap.path)); sampler=make_sampler(temp=0.0)
B=16; MEAN=64

FILLER=("A cache line is the unit a processor moves between memory and cache. "
        "Threads that write different variables sharing one line contend anyway. ")

def prompt_of(target):
    """Chat-templated prompt of roughly `target` tokens."""
    body=FILLER*max(1,target//18)
    txt=tok.apply_chat_template([{"role":"user","content":body+" Summarise in one sentence."}],
                                add_generation_prompt=True)
    ids=list(txt if isinstance(txt,list) else tok.encode(txt))
    return ids[:target] if len(ids)>target else ids

def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

def steady(prompts, label):
    """Slope between two token counts; cancels prefill and setup."""
    batch_generate(m,tok,prompts,max_tokens=2,sampler=sampler)
    ts={}
    for mt in (4,16):
        best=None
        for _ in range(2):
            t=time.perf_counter()
            batch_generate(m,tok,prompts,max_tokens=mt,sampler=sampler)
            w=time.perf_counter()-t; charge(w)
            best=w if best is None else min(best,w)
        ts[mt]=best
    per=(ts[16]-ts[4])/12
    lens=[len(p) for p in prompts]
    return {"label":label,"ms_per_step":round(per*1000,2),"tok_s":round(B/per,1),
            "prefill_s":round(ts[4]-per*4,3),
            "prompt_len_min":min(lens),"prompt_len_max":max(lens),
            "prompt_len_mean":round(sum(lens)/len(lens),1),
            "padded_positions":max(lens)*B,"useful_positions":sum(lens),
            "padding_waste":round(1-sum(lens)/(max(lens)*B),4)}

rng=random.Random(20260823)
uniform=[prompt_of(MEAN) for _ in range(B)]
# Same mean length, spread over a realistic range.
mixed=[prompt_of(rng.randint(MEAN//4, MEAN*7//4)) for _ in range(B)]

out=[steady(uniform,"identisch"), steady(mixed,"gemischt")]
base=out[0]["tok_s"]
for r in out: r["vs_identisch"]=round(r["tok_s"]/base,3)
print(json.dumps({"arms":out,"peak_gb":round(mx.get_peak_memory()/1e9,2),
                  "budget":{k:v for k,v in g.summary().items() if "limit" not in k}},indent=2))
