# SD1 A/B at the engine: waves of the benchmark prompts (32-token answers), wall time and the answers
import json, os, sys, time
import torch
from ironmule.ggml_engine import GgmlEngine, Job
from ironmule_product import compare
TPL = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
draft = '/models/Qwen_Qwen3-0.6B-Q4_K_M.gguf' if os.environ.get("IRONMULE_GGML_SD", "0") != "0" else None
e = GgmlEngine('/models/Qwen3-32B-Q4_K_M.gguf', seqs=32, seq_ctx=2048, flash_attn=True, speculate=True, draft_gguf=draft)
print("WAVE drafter", type(e.drafter).__name__ if e.drafter else None, "refused", e.draft_refused, flush=True)
out = {}
for width in [int(w) for w in sys.argv[1].split(',')]:
    for seed in (0, 1, 2):
        prompts = compare.performance_prompts(seed, width, max(width, 8))
        jobs = [Job(e.tokenize(TPL.format(p)), 32) for p in prompts]
        torch.cuda.synchronize(); t = time.perf_counter()
        e.run(jobs)
        torch.cuda.synchronize(); wall = time.perf_counter() - t
        toks = sum(len(j.tokens) for j in jobs)
        out[f"{width}-{seed}"] = [j.tokens for j in jobs]
        print(f"WAVE width {width} seed {seed}: {wall:.3f} s, {toks} tokens, {toks / wall:.1f} tok/s"
              + (f", drafter {e.drafter.summary()}" if e.drafter else ""), flush=True)
json.dump(out, open(sys.argv[2], "w"))
e.close()
