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
from _bench import harness_preconditions
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
REPS = 6


def benchmark_model(model_tag: str, model_id: str, guard) -> dict:
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
        samples = {cfg: {"ttft": [], "tps": [], "total": []} for cfg in CONFIGS}
        tokens_seen = {cfg: [] for cfg in CONFIGS}
        cfg_names = list(CONFIGS)

        # Warmup all engines
        for knobs in CONFIGS.values():
            backend.generate(token_ids, max_tokens=OUTPUT_TOKENS, knobs=knobs)
            mx.synchronize()

        # Interleaved measurement: every rep runs all configs, order rotated so
        # warm-up drift cannot systematically favour one config.
        for rep in range(REPS):
            order = cfg_names if rep % 2 == 0 else list(reversed(cfg_names))
            for cfg_name in order:
                mx.synchronize()
                out = backend.generate(token_ids, max_tokens=OUTPUT_TOKENS, knobs=CONFIGS[cfg_name])
                guard.record_gpu((out["prefill_ns"] + out["decode_ns"]) / 1e9)
                n = len(out["logical_tokens"])
                decode_s = out["decode_ns"] / 1e9
                samples[cfg_name]["ttft"].append(out["prefill_ns"] / 1e6)
                samples[cfg_name]["tps"].append((n - 1) / decode_s if decode_s > 0 and n > 1 else 0.0)
                samples[cfg_name]["total"].append((out["prefill_ns"] + out["decode_ns"]) / 1e9)
                tokens_seen[cfg_name].append(list(out["logical_tokens"]))
            guard.required_break()

        for cfg_name in CONFIGS:
            prompt_results[cfg_name] = {
                "ttft_ms_median": statistics.median(samples[cfg_name]["ttft"]),
                "decode_tps_median": statistics.median(samples[cfg_name]["tps"]),
                "total_time_s_median": statistics.median(samples[cfg_name]["total"]),
                "tokens_generated": len(tokens_seen[cfg_name][0]),
            }

        # Token identity: every rep of a candidate must match that rep's baseline.
        for cfg_name in ("dispatched", "combined_r8"):
            is_identical = all(
                cand == base
                for cand, base in zip(tokens_seen[cfg_name], tokens_seen["baseline"])
            )
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
    guard = harness_preconditions()
    all_results = {}
    for tag, model_id in MODELS.items():
        all_results[tag] = benchmark_model(tag, model_id, guard)

    out_file = PROJECT_ROOT / "experiments" / "model_benchmark" / "gemma_family_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved complete benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
