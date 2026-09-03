#!/usr/bin/env python3
"""Empirical measurement of TTFT reduction with PrefixCache on Apple Silicon M1 Max.

Measures Gemma 4B and Gemma 12B across 4 distinct queries with a shared prefix
(350-400 tokens), validating:
1. Miss-TTFT (unoptimized baseline) vs Hit-TTFT (stateful prefix cache).
2. Exact token identity (100% match gate).
3. Wall-time and latency speedup factors.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
if str(IRONMULE) not in sys.path:
    sys.path.insert(0, str(IRONMULE))

from friday_serve.ironmule_backend import IronMuleBackend
from ironmule.hw import static_facts

MODELS = [
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-3-12b-it-4bit",
]

SHARED_CONTEXT = (
    "You are Friday, an autonomous AI operating system designed to optimize large language model inference on Apple Silicon hardware. "
    "The Apple Silicon M1 Max processor features a high-performance system-on-chip (SoC) architecture manufactured on a 5-nanometer process. "
    "It includes 10 CPU cores comprising 8 performance cores and 2 high-efficiency cores, alongside up to 32 GPU execution cores. "
    "A standout characteristic of the M1 Max is its 512-bit wide unified memory bus, providing an aggregate memory bandwidth of up to 400 GB/s. "
    "Because the CPU, GPU, and Neural Engine share a unified memory pool, host-to-device PCI Express data transfers are completely eliminated. "
    "In modern autoregressive LLM inference pipelines, the computation consists of two primary phases: prompt prefill and token generation. "
    "During the prefill phase, the model processes the entire prompt sequence in parallel, making this phase compute-bound on the GPU ALUs. "
    "In contrast, during the autoregressive decode phase, tokens are produced one by one, making memory bandwidth the limiting operational factor. "
    "To achieve maximum inference throughput, stateful prefix caching stores the computed Key-Value (KV) cache for invariant prompt prefixes. "
    "Subsequent requests sharing the same prefix can bypass prefill computation for those tokens, dramatically reducing Time to First Token (TTFT). "
    "Speculative decoding further accelerates decode throughput by verifying multiple draft tokens in a single forward pass. "
    "By combining fused projections, compiled graph execution, asynchronous readbacks, and prefix caching, Friday maximizes token generation efficiency. "
    "Always provide clear, rigorous, and direct technical answers to user questions based on this architectural specification."
)

QUERIES = [
    "Question: State the memory bus width of the M1 Max chip.",
    "Question: Why is autoregressive decode typically memory bandwidth bound?",
    "Question: Describe the primary benefit of unified memory for LLM execution.",
    "Question: Explain how stateful prefix caching accelerates time to first token.",
]

MAX_TOKENS = 16
N_REPETITIONS = 3


def benchmark_model(model_id: str) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Benchmarking Model: {model_id}")
    print(f"=======================================================")

    backend = IronMuleBackend.load(model_id)

    # Encode queries
    full_prompts = [SHARED_CONTEXT + "\n\n" + q for q in QUERIES]
    encoded_queries = [backend.encode(p) for p in full_prompts]

    # Find exact shared prefix
    min_len = min(len(eq) for eq in encoded_queries)
    common_len = 0
    for i in range(min_len):
        if all(eq[i] == encoded_queries[0][i] for eq in encoded_queries):
            common_len += 1
        else:
            break

    prefix_ids = encoded_queries[0][:common_len]
    print(f"Prefix token count: {len(prefix_ids)}")
    for i, eq in enumerate(encoded_queries):
        print(f"  Query {i+1} prompt tokens: {len(eq)} (Tail: {len(eq) - len(prefix_ids)} tokens)")

    # 1. Warmup (2 passes on baseline)
    print("\nWarming up baseline...")
    backend.set_prefix_cache(None)
    for _ in range(2):
        backend.generate(encoded_queries[0], max_tokens=MAX_TOKENS, knobs={})

    # 2. Baseline Measurements (No Prefix Cache)
    print("Measuring Baseline (No Cache, 3 repetitions per query)...")
    baseline_results: list[dict[str, Any]] = []
    for i, (q, eq) in enumerate(zip(QUERIES, encoded_queries)):
        ttft_runs = []
        total_runs = []
        tokens_baseline = None
        text_baseline = None

        for rep in range(N_REPETITIONS):
            t0 = time.perf_counter_ns()
            res = backend.generate(eq, max_tokens=MAX_TOKENS, knobs={})
            t1 = time.perf_counter_ns()
            ttft_ms = res["prefill_ns"] / 1e6
            total_ms = res["total_ns"] / 1e6
            ttft_runs.append(ttft_ms)
            total_runs.append(total_ms)
            if tokens_baseline is None:
                tokens_baseline = res["logical_tokens"]
                text_baseline = res.get("text")

        med_ttft = statistics.median(ttft_runs)
        med_total = statistics.median(total_runs)
        print(f"  Q{i+1} Baseline Median TTFT: {med_ttft:.2f} ms | Total: {med_total:.2f} ms")
        baseline_results.append({
            "query_index": i + 1,
            "query": q,
            "prompt_tokens": len(eq),
            "median_ttft_ms": round(med_ttft, 2),
            "median_total_ms": round(med_total, 2),
            "ttft_runs_ms": [round(x, 2) for x in ttft_runs],
            "tokens": tokens_baseline,
            "text": text_baseline,
        })

    # 3. Warmup and Populate Prefix Cache
    print("\nInitializing PrefixCache with shared prefix...")
    backend.set_prefix_cache(prefix_ids)

    # Initial Population (Cache Miss on Q1)
    t0 = time.perf_counter_ns()
    pop_res = backend.generate(encoded_queries[0], max_tokens=MAX_TOKENS, knobs={})
    t1 = time.perf_counter_ns()
    pop_ttft_ms = pop_res["prefill_ns"] / 1e6
    pop_total_ms = pop_res["total_ns"] / 1e6
    print(f"  Initial Cache Miss / Population (Q1): TTFT: {pop_ttft_ms:.2f} ms | Hits: {pop_res['prefix_cache_hits']}")

    # 4. Measure Cache Hits across all queries (including Q1 repeat)
    print("Measuring Stateful Cache Hits (3 repetitions per query)...")
    cache_results: list[dict[str, Any]] = []
    for i, (q, eq) in enumerate(zip(QUERIES, encoded_queries)):
        ttft_runs = []
        total_runs = []
        hits_runs = []
        tokens_cache = None
        text_cache = None

        for rep in range(N_REPETITIONS):
            t0 = time.perf_counter_ns()
            res = backend.generate(eq, max_tokens=MAX_TOKENS, knobs={})
            t1 = time.perf_counter_ns()
            ttft_ms = res["prefill_ns"] / 1e6
            total_ms = res["total_ns"] / 1e6
            ttft_runs.append(ttft_ms)
            total_runs.append(total_ms)
            hits_runs.append(res["prefix_cache_hits"])
            if tokens_cache is None:
                tokens_cache = res["logical_tokens"]
                text_cache = res.get("text")

        med_ttft = statistics.median(ttft_runs)
        med_total = statistics.median(total_runs)
        print(f"  Q{i+1} Cache Hit Median TTFT: {med_ttft:.2f} ms | Total: {med_total:.2f} ms | Hits: {hits_runs[-1]}")
        cache_results.append({
            "query_index": i + 1,
            "query": q,
            "prompt_tokens": len(eq),
            "tail_tokens": len(eq) - len(prefix_ids),
            "median_ttft_ms": round(med_ttft, 2),
            "median_total_ms": round(med_total, 2),
            "ttft_runs_ms": [round(x, 2) for x in ttft_runs],
            "prefix_cache_hits": hits_runs[-1],
            "tokens": tokens_cache,
            "text": text_cache,
        })

    # 5. Analysis & Verification
    print("\n--- Evaluation & Verification ---")
    query_comparisons: list[dict[str, Any]] = []
    all_tokens_match = True

    for i in range(len(QUERIES)):
        base = baseline_results[i]
        c = cache_results[i]
        token_match = (base["tokens"] == c["tokens"])
        if not token_match:
            all_tokens_match = False

        ttft_reduction_pct = (1.0 - (c["median_ttft_ms"] / base["median_ttft_ms"])) * 100.0
        ttft_speedup = base["median_ttft_ms"] / c["median_ttft_ms"] if c["median_ttft_ms"] > 0 else float("inf")

        print(f"Query {i+1}:")
        print(f"  Baseline TTFT: {base['median_ttft_ms']} ms -> Cache Hit TTFT: {c['median_ttft_ms']} ms")
        print(f"  TTFT Reduction: {ttft_reduction_pct:.2f}% (Speedup: {ttft_speedup:.2f}x)")
        print(f"  Token Identity: {'100% MATCH' if token_match else 'MISMATCH'}")

        query_comparisons.append({
            "query_index": i + 1,
            "query": QUERIES[i],
            "baseline_ttft_ms": base["median_ttft_ms"],
            "cache_hit_ttft_ms": c["median_ttft_ms"],
            "ttft_reduction_pct": round(ttft_reduction_pct, 2),
            "ttft_speedup": round(ttft_speedup, 2),
            "token_match": token_match,
            "tokens_baseline": base["tokens"],
            "tokens_cache": c["tokens"],
        })

    # Aggregate
    agg_base_ttft = statistics.median(b["median_ttft_ms"] for b in baseline_results)
    agg_hit_ttft = statistics.median(c["median_ttft_ms"] for c in cache_results)
    agg_reduction_pct = (1.0 - (agg_hit_ttft / agg_base_ttft)) * 100.0
    agg_speedup = agg_base_ttft / agg_hit_ttft

    print(f"\nModel {model_id} Summary:")
    print(f"  Overall Baseline Median TTFT: {agg_base_ttft:.2f} ms")
    print(f"  Overall Cache Hit Median TTFT: {agg_hit_ttft:.2f} ms")
    print(f"  Overall TTFT Reduction: {agg_reduction_pct:.2f}%")
    print(f"  Overall Speedup: {agg_speedup:.2f}x")
    print(f"  100% Token Identity Gate: {'PASSED' if all_tokens_match else 'FAILED'}")

    return {
        "model_id": model_id,
        "model_revision": backend.model_revision,
        "prefix_token_count": len(prefix_ids),
        "initial_miss_ttft_ms": round(pop_ttft_ms, 2),
        "aggregate": {
            "baseline_median_ttft_ms": round(agg_base_ttft, 2),
            "cache_hit_median_ttft_ms": round(agg_hit_ttft, 2),
            "ttft_reduction_pct": round(agg_reduction_pct, 2),
            "ttft_speedup": round(agg_speedup, 2),
            "token_identity_passed": all_tokens_match,
        },
        "query_comparisons": query_comparisons,
    }


def main() -> None:
    facts = static_facts()
    print("=" * 60)
    print(f"FRIDAY EMPIRICAL PREFIX CACHE BENCHMARK")
    print(f"Hardware: {facts.get('chip')} | GPU Cores: {facts.get('gpu_cores')} | Memory: {facts.get('memory_bytes', 0)//(1024**3)} GB")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    results: dict[str, Any] = {
        "benchmark": "prefix_cache_empirical",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": facts,
        "models": {},
    }

    for model_id in MODELS:
        model_res = benchmark_model(model_id)
        results["models"][model_id] = model_res

    # Output file
    output_path = PROJECT_ROOT / "experiments" / "model_benchmark" / "prefix_cache_empirical_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Benchmark completed successfully!")
    print(f"Raw results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
