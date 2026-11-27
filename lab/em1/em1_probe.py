"""EM1 probe: the same engine waves with the token embedding on the GPU (IRONMULE_GGML_EM1=1) or on the
CPU (=0, llama.cpp's placement). Prints one JSON line: whether EM1 engaged, wave wall times, the median 32-row decode
call and the prefill call from the engine trace, and the tokens (compared across the two runs by the caller).

  IRONMULE_GGML_EM1=1 python lab/em1/em1_probe.py MODEL.gguf OUT.json
"""
import json
import os
import statistics as st
import sys
import tempfile
import time

trace = tempfile.mktemp(suffix=".trace")
os.environ["IRONMULE_GGML_TRACE"] = trace

from ironmule.ggml_engine import EM1, GgmlEngine, Job  # noqa: E402
from ironmule_product import compare  # noqa: E402

TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
gguf, out = sys.argv[1], sys.argv[2]
e = GgmlEngine(gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
walls, tokens = [], {}
for width in (8, 32):
    for seed in (0, 1, 2):
        prompts = compare.performance_prompts(seed, width, max(width, 8))
        jobs = [Job(e.tokenize(TPL.format(p)), 32) for p in prompts]
        open(trace, "a").write(f"# wave {width} {seed}\n")
        t = time.perf_counter()
        e.run(jobs)
        walls.append((width, seed, round(time.perf_counter() - t, 4)))
        tokens[f"{width}-{seed}"] = [j.tokens for j in jobs]
calls = [tuple(float(x) for x in line.split()[:3]) for line in open(trace) if line.strip() and not line.startswith("#")]
dec32 = [c[1] for c in calls if int(c[2]) == 32]
dec8 = [c[1] for c in calls if int(c[2]) == 8]
pre = [c[1] for c in calls if int(c[2]) > 64]
res = {"em1_flag": EM1, "em1_engaged": e._em1 is not None, "walls": walls,
       "decode32_ms": round(1e3 * st.median(dec32), 3) if dec32 else None,
       "decode8_ms": round(1e3 * st.median(dec8), 3) if dec8 else None,
       "prefill_ms": [round(1e3 * p, 1) for p in pre]}
print("EM1 " + json.dumps(res), flush=True)
json.dump({"result": res, "tokens": tokens}, open(out, "w"))
e.close()
