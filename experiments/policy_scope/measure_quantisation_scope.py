"""Is the width cliff a property of the quantisation config or of the kernel?

If the cliff moves with group size or bit width, the policy describes one build of
one model and has no business in a router. If it stays put, it is a kernel tiling
boundary and generalises. Tested on the matmul directly, in a dependency chain,
because that is where the cliff lives -- not on a whole model, which would confound
it with attention and norms."""
import sys, time, json, statistics
sys.path.insert(0,"tools")
from _bench import BudgetGuard, require_ac_power
import mlx.core as mx

require_ac_power(); g=BudgetGuard()
H, I, LAYERS = 2560, 10240, 8          # 4B FFN shape, shortened chain
WIDTHS=(4,5,6,8,12,32,48,64)

def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

def chain_ms(M, uq, us_, ub, dq, ds_, db, group, bits, reps=6):
    x0=mx.random.normal((1,M,H)).astype(mx.float16); mx.eval(x0)
    def once():
        x=x0
        for _ in range(LAYERS):
            h=mx.quantized_matmul(x,uq,us_,ub,transpose=True,group_size=group,bits=bits)
            x=mx.quantized_matmul(h,dq,ds_,db,transpose=True,group_size=group,bits=bits)
        mx.eval(x); mx.synchronize()
    once()
    s=[]; t0=time.perf_counter()
    for _ in range(reps):
        t=time.perf_counter_ns(); once(); s.append(time.perf_counter_ns()-t)
    charge(time.perf_counter()-t0)
    return statistics.median(s)/1e6

out={}
for bits in (4,8):
    for group in (32,64,128):
        wu=mx.random.normal((I,H)).astype(mx.float16)
        wd=mx.random.normal((H,I)).astype(mx.float16)
        uq,us_,ub=mx.quantize(wu,group_size=group,bits=bits)
        dq,ds_,db=mx.quantize(wd,group_size=group,bits=bits)
        mx.eval(uq,us_,ub,dq,ds_,db)
        row={}
        for M in WIDTHS:
            ms=chain_ms(M,uq,us_,ub,dq,ds_,db,group,bits)
            row[M]={"ms":round(ms,3),"ms_per_pos":round(ms/M,4)}
        # regression = a wider width whose per-position cost is worse than a narrower one
        best_so_far=float("inf"); regs=[]
        for M in WIDTHS:
            pp=row[M]["ms_per_pos"]
            if pp > best_so_far*1.05: regs.append(M)
            best_so_far=min(best_so_far,pp)
        out[f"{bits}bit_g{group}"]={"widths":row,"regressions":regs,
            "best_width":min(WIDTHS,key=lambda M:row[M]["ms_per_pos"])}
        print(f"{bits}bit g{group}: Regressionen {regs}  beste Breite {out[f'{bits}bit_g{group}']['best_width']}", flush=True)
        del wu,wd,uq,us_,ub,dq,ds_,db; mx.clear_cache()
out["_budget"]={k:v for k,v in g.summary().items() if "limit" not in k}
open(f"{sys.argv[1]}/quant.json","w").write(json.dumps(out,indent=2))
