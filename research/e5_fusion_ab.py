"""E5: is projection fusion a real decode win? Paired A/B, 6 fresh processes."""
from ironmule import ab, bench
from ironmule.runtime import Knobs

BASE = dict(compiled_fixed_cache=True, fused_argmax=True, head_skip_prefill=True)
res = ab.run({"no_fusion": Knobs(**BASE), "fusion": Knobs(**BASE, fuse_projections=True)},
             processes=6, repeats=7, warmup=2)
print(ab.report(res))
print("raw:", bench.record("E5-projection-fusion-ab", res))
