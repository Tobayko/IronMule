#!/usr/bin/env python3
"""Empirical multi-model benchmark across Gemma 1B, 4B, and 12B on Apple Silicon.

Measures TTFT, Decode TPS, Total Latency, and verifies exact Token Identity
across Baseline vs. Dispatched Knobs vs. Combined Knobs.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from friday_serve.ironmule_backend import IronMuleBackend
import mlx.core as mx

MODELS = {
    "1B": "mlx-community/gemma-3-1b-it-4bit",
    "4B": "mlx-community/gemma-3-4b-it-4bit",
    "12B": "mlx-community/gemma-3-12b-it-4bit",
}

CONFIGS = {
    "baseline": {},
    "dispatched": {"head_skip_prefill": True, "compiled_fixed_cache": True},
    "combined_r8": {"head_skip_prefill": True, "compiled_fixed_cache": True, "readback_every": 8},
}

PROMPTS = {
    "short_q": "Explain in two sentences why Apple Silicon unified memory improves LLM throughput.",
    "code_task": "Write a Python function to compute the moving average of a list using a sliding window.",
    "reasoning": "A store offers a 20% discount on a $50 item. After the discount, a 10% sales tax is applied. What is the final price? Show steps.",
}

OUTPUT_TOKENS = 48
WARMUP_RUNS = 1
PAIRS = 3


def benchmark_model(model_tag: str, model_id: str) -> dict:
    print(f"\n=======================================================")
    print(f"LOADING MODEL {model_tag}: {model_id}")
    print(f"=======================================================")
    backend = IronMuleBackend.load(model_id)
    results = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n--- Benchmark Prompt: {prompt_name} ({model_tag}) ---")
        token_ids = backend.encode(prompt_text)
        print(f"Prompt length: {len(token_ids)} tokens, target output: {OUTPUT_TOKENS} tokens")

        prompt_results = {}
        tokens_by_config = {}

        # Warmup all engines
        for cfg_name, knobs in CONFIGS.items():
            _ = backend.generate(token_ids, max_tokens=OUTPUT_TOKENS, knobs=knobs)
            mx.eval()
            mx.synchronize()

        # Measurement iterations
        for cfg_name, knobs in CONFIGS.items():
            ttft_list = []
            decode_tps_list = []
            total_time_list = []
            last_tokens = None

            for i in range(PAIRS):
                mx.eval()
                mx.synchronize()
                out = backend.generate(token_ids, max_tokens=OUTPUT_TOKENS, knobs=knobs)
                ttft_ms = out["prefill_ns"] / 1e6
                decode_s = out["decode_ns"] / 1e9
                n_tokens = len(out["logical_tokens"])
                tps = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0.0
                total_s = (out["prefill_ns"] + out["decode_ns"]) / 1e9

                ttft_list.append(ttft_ms)
                decode_tps_list.append(tps)
                total_time_list.append(total_s)
                last_tokens = list(out["logical_tokens"])

            tokens_by_config[cfg_name] = last_tokens
            prompt_results[cfg_name] = {
                "ttft_ms_median": statistics.median(ttft_list),
                "decode_tps_median": statistics.median(decode_tps_list),
                "total_time_s_median": statistics.median(total_time_list),
                "tokens_generated": len(last_tokens),
            }

        # Check Token Identity against baseline
        base_tokens = tokens_by_config["baseline"]
        for cfg_name in ("dispatched", "combined_r8"):
            is_identical = tokens_by_config[cfg_name] == base_tokens
            prompt_results[cfg_name]["token_identical_to_baseline"] = is_identical
            if not is_identical:
                print(f"WARNING: Token mismatch in {cfg_name} for {prompt_name}!")

        # Compute speedups
        base_total = prompt_results["baseline"]["total_time_s_median"]
        base_ttft = prompt_results["baseline"]["ttft_ms_median"]
        base_tps = prompt_results["baseline"]["decode_tps_median"]

        for cfg_name in ("dispatched", "combined_r8"):
            cand_total = prompt_results[cfg_name]["total_time_s_median"]
            cand_ttft = prompt_results[cfg_name]["ttft_ms_median"]
            cand_tps = prompt_results[cfg_name]["decode_tps_median"]

            speedup_total = (1.0 - cand_total / base_total) * 100.0
            ttft_speedup = (1.0 - cand_ttft / base_ttft) * 100.0
            tps_gain = (cand_tps / base_tps - 1.0) * 100.0

            prompt_results[cfg_name]["end_to_end_speedup_percent"] = speedup_total
            prompt_results[cfg_name]["ttft_speedup_percent"] = ttft_speedup
            prompt_results[cfg_name]["tps_gain_percent"] = tps_gain

            print(
                f"[{cfg_name.upper()}] Total: {cand_total*1000:.1f}ms (Speedup: {speedup_total:+.2f}%), "
                f"TTFT: {cand_ttft:.1f}ms ({ttft_speedup:+.2f}%), "
                f"TPS: {cand_tps:.1f} tok/s ({tps_gain:+.2f}%), "
                f"Identical: {prompt_results[cfg_name]['token_identical_to_baseline']}"
            )

        results[prompt_name] = prompt_results

    del backend
    gc.collect()
    mx.metal.clear_cache()
    return results


def main():
    all_results = {}
    for tag, model_id in MODELS.items():
        all_results[tag] = benchmark_model(tag, model_id)

    out_file = PROJECT_ROOT / "experiments" / "model_benchmark" / "gemma_family_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved complete benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
