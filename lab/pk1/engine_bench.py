"""PK1 engine benchmark: the deployment comparison's workload straight into IronMule's GGML engine (no HTTP).

Prompts are `compare.performance_prompts` rendered with Qwen3's chat template (thinking off), 32 new tokens, greedy.
For each concurrency N the engine gets N requests at once (8 for N=1 and N=8, as the HTTP harness sends 8 requests
there; at N=1 they run one after another). Prints JSON lines: wall seconds, output tok/s, first-token p50/p95, and a
digest of all generated tokens so arms can be compared token for token.

  python lab/pk1/engine_bench.py --gguf /models/Qwen3-32B-Q4_K_M.gguf --widths 1,8,32 --repeats 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time

from ironmule.ggml_engine import GgmlEngine, Job
from ironmule_product.compare import performance_prompts


def render(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def wave(engine, prompts, width, limit):
    ids = [engine.tokenize(render(p)) for p in prompts]
    start = time.monotonic_ns()
    jobs = []
    if width == 1:
        for x in ids:
            job = Job(list(x), limit)
            engine.run([job])
            jobs.append(job)
    else:
        jobs = [Job(list(x), limit) for x in ids]
        engine.run(jobs)
    wall = (time.monotonic_ns() - start) / 1e9
    if width == 1:  # first token of a sequential request counts from its own start, as a client sees it
        firsts = []
        t = start
        for job in jobs:
            firsts.append((job.first_ns - t) / 1e6)
            t = job.finished_ns
    else:
        firsts = [(job.first_ns - start) / 1e6 for job in jobs]
    firsts.sort()
    tokens = sum(len(j.tokens) for j in jobs)
    digest = hashlib.sha256(json.dumps([j.tokens for j in jobs]).encode()).hexdigest()[:16]
    return {"width": width, "requests": len(jobs), "prompt_tokens": sum(map(len, ids)), "tokens": tokens,
            "wall_s": round(wall, 4), "tok_s": round(tokens / wall, 2),
            "ttft_p50_ms": round(statistics.median(firsts), 1),
            "ttft_p95_ms": round(firsts[max(0, -(-95 * len(firsts) // 100) - 1)], 1), "digest": digest}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--widths", default="1,8,32")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--seqs", type=int, default=32)
    p.add_argument("--seq-ctx", type=int, default=2048)
    p.add_argument("--label", default=os.environ.get("PK1_LABEL", "arm"))
    a = p.parse_args()
    started = time.monotonic()
    engine = GgmlEngine(a.gguf, seqs=a.seqs, seq_ctx=a.seq_ctx, flash_attn=True,
                        speculate=os.environ.get("PK1_SPEC", "1") != "0",
                        draft_gguf=os.environ.get("PK1_DRAFT") or None)
    steps = []
    engine.observer = lambda rows, prompt, kv, seconds: steps.append((rows, prompt, seconds))
    print(json.dumps({"label": a.label, "load_s": round(time.monotonic() - started, 1),
                      "settings": getattr(engine, "settings", {}), "gpu_layers": engine.gpu_layers}), flush=True)
    widths = [int(w) for w in a.widths.split(",")]
    # warm-up wave at the widest width, not measured
    engine.run([Job(engine.tokenize(render(x)), 4) for x in performance_prompts(7, 99, max(widths))])
    for rep in range(a.repeats):
        for width in widths:
            prompts = performance_prompts(1000 + rep * 1009 + width, width, max(width, 8))
            steps.clear()
            row = wave(engine, prompts, width, a.limit)
            decode = [s for s in steps if s[1] == 0]
            prefill = [s for s in steps if s[1] > 0]
            row.update(label=a.label, repeat=rep, decode_steps=len(decode),
                       decode_ms=round(1000 * sum(s[2] for s in decode) / max(1, len(decode)), 2),
                       decode_rows=round(sum(s[0] for s in decode) / max(1, len(decode)), 1),
                       prefill_calls=len(prefill), prefill_s=round(sum(s[2] for s in prefill), 3))
            print(json.dumps(row), flush=True)
    print(json.dumps({"label": a.label, "summary": {k: v for k, v in engine.summary().items()
                                                    if k in ("pack", "prefill_batch", "kernels", "pilot", "draft_model")}}), flush=True)
    engine.close()


if __name__ == "__main__":
    main()
