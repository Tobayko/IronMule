"""PK1 HTTP benchmark of one running OpenAI-compatible server with the deployment comparison's workload.

Uses ironmule_product.compare (the same streaming client, prompts and metrics as production_compare.py), so numbers are
comparable to the paired harness. The server must already listen on --url.

  python lab/pk1/http_bench.py --url http://127.0.0.1:8001 --engine vllm --model compare --widths 1,8,32 --repeats 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

from ironmule.factual_gate import MAX_TOKENS, cases, correct
from ironmule_product import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--engine", required=True, choices=("vllm", "ironmule", "llama_cpp"))
    p.add_argument("--model", default="compare")
    p.add_argument("--widths", default="1,8,32")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--quality", action="store_true", help="also grade the factual cases at each width")
    p.add_argument("--label", default="")
    a = p.parse_args()
    server = {"base_url": a.url, "model_id": a.model}
    deadline = time.monotonic() + 3600
    compare.stream(a.engine, server, compare.WARMUP, 8, deadline)
    for rep in range(a.repeats):
        for width in [int(w) for w in a.widths.split(",")]:
            seed = a.seed + rep * 1009 + width
            prompts = compare.performance_prompts(seed, width, max(width, 8))
            rows, elapsed = compare.batch(a.engine, server, prompts, width, a.max_tokens, deadline)
            m = compare.performance_metrics(rows, elapsed)
            out = {"label": a.label, "engine": a.engine, "repeat": rep, "width": width,
                   "tok_s": round(m["tokens_per_second"], 2), "wall_s": round(m["wall_s"], 3),
                   "ttft_p50_ms": round(m["ttft_median_ms"], 1), "ttft_p95_ms": round(m["ttft_p95_ms"], 1),
                   "tokens": m["completion_tokens"], "prompt_tokens": sum(m["performance_prompt_tokens"]),
                   "texts": hashlib.sha256(json.dumps([r["text"] for r in rows]).encode()).hexdigest()[:16]}
            if a.quality:
                qs = cases(seed)
                qrows, _ = compare.batch(a.engine, server, [c.prompt for c in qs], width, MAX_TOKENS, deadline)
                out["quality"] = f"{sum(correct(c, r['text']) for c, r in zip(qs, qrows))}/{len(qs)}"
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
