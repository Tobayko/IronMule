#!/usr/bin/env python3
"""Comprehensive benchmark for LONG TASKS on Apple Silicon (M1 Max).

Evaluates:
- Long Prompt Contexts (~380 tokens)
- Long Generation Lengths (128 and 256 tokens)
Across:
- TTFT (Time To First Token) in ms
- Decode Throughput (tokens/s)
- Total Wall Latency (s)
- Effective Memory Bandwidth Utilization (GB/s & % of 400 GB/s peak)
- Peak Memory / RSS
- 100% Token Identity verification

Configurations tested:
- Baseline: Standard MLX eager execution (Knobs default)
- Core: compiled_fixed_cache + head_skip_prefill (r=1)
- Core + R8: compiled_fixed_cache + head_skip_prefill (r=8)
- Core + R16: compiled_fixed_cache + head_skip_prefill (r=16)
"""

from __future__ import annotations

import gc
import json
import statistics as st
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(IRONMULE))

import mlx.core as mx
from tools._bench import harness_preconditions, resolve_local_model_snapshot
from mlx_lm import load
from mlx.utils import tree_flatten

from ironmule.runtime import Knobs, Engine
from ironmule.tune import _eos_ids

# Long document prompt (~380 tokens)
LONG_PROMPT_DOC = (
    "Apple Silicon architecture integrates the CPU, GPU, Neural Engine, and Secure Enclave onto a single "
    "System on a Chip (SoC) using unified memory architecture (UMA). In traditional PC architectures, the CPU and GPU "
    "maintain separate physical memory pools and communicate over a PCI Express (PCIe) bus. This discrete design introduces "
    "significant latency and serialization overhead, as data must be explicitly copied across the PCIe bus before each computation. "
    "In contrast, Apple's unified memory allows all processing engines on the SoC to access a single, high-bandwidth memory pool "
    "without data duplication. For large language model (LLM) inference, which is fundamentally bounded by memory bandwidth during "
    "autoregressive token generation, unified memory provides a massive architectural advantage. "
    "During autoregressive decoding, every single generated token requires reading the entire model weight matrices from memory into "
    "the execution units. On an Apple M1 Max chip, the unified memory bus provides up to 400 gigabytes per second of theoretical "
    "bandwidth across a 512-bit wide memory interface. When evaluating large Transformer models with billions of parameters, "
    "the total latency of the decoding phase is almost completely dictated by how quickly these weights can be streamed from LPDDR5 "
    "memory into the GPU's registers and execution pipelines. "
    "Furthermore, Metal Performance Shaders and Apple's MLX machine learning framework are designed specifically to exploit this unified "
    "memory hierarchy. By compiling compute graphs using just-in-time Metal compilation, MLX avoids redundant kernel dispatch overhead "
    "and intermediate buffer allocations. When combined with pre-allocated constant-shape Key-Value (KV) caches and efficient "
    "slicing of logit projections during prefill, Apple Silicon achieves unprecedented efficiency for on-device artificial intelligence. "
    "Analyze the structural bottlenecks of Transformer inference on Apple Silicon, and discuss how memory bandwidth, prefill logit slicing, "
    "and asynchronous readback bundling interact to determine end-to-end token latency."
)

OUTPUT_TOKENS_LIST = [128, 256]


def get_model_size_gb(model) -> float:
    flat = tree_flatten(model.parameters())
    tot_bytes = sum(arr.nbytes for _, arr in flat)
    return tot_bytes / 1e9


def benchmark_model_long_tasks(model_id: str, tag: str, guard) -> dict:
    print(f"\n=======================================================")
    print(f"=== LONG TASK BENCHMARK: {tag} ({model_id}) ===")
    print(f"=======================================================")

    snapshot = resolve_local_model_snapshot(model_id)
    model, tokenizer = load(str(snapshot.path))
    model_gb = get_model_size_gb(model)
    eos_ids = _eos_ids(tokenizer)
    print(f"Model weight size: {model_gb:.2f} GB")

    prompt_ids = tokenizer.encode(LONG_PROMPT_DOC)
    prompt_len = len(prompt_ids)
    print(f"Long Prompt Tokens: {prompt_len} tokens")

    configs = [
        ("Baseline (Unoptimized, r=1)", Knobs()),
        ("Core (Compiled + HeadSkip, r=1)", Knobs(compiled_fixed_cache=True, head_skip_prefill=True, readback_every=1)),
        ("Core + Readback Bundled (r=8)", Knobs(compiled_fixed_cache=True, head_skip_prefill=True, readback_every=8)),
        ("Core + Deep Bundling (r=16)", Knobs(compiled_fixed_cache=True, head_skip_prefill=True, readback_every=16)),
    ]

    results_by_length = {}

    for max_tokens in OUTPUT_TOKENS_LIST:
        print(f"\n--- Testing Long Output: {max_tokens} tokens (Prompt: {prompt_len}) ---")
        length_results = {}

        engines = {name: Engine(model, tokenizer, knobs) for name, knobs in configs}
        names = [name for name, _ in configs]
        base_name = names[0]

        # Warmup all engines
        for eng in engines.values():
            eng.generate(prompt_ids[:16], 8, eos_ids)
        mx.synchronize()

        samples = {name: {"ttft": [], "tps": [], "wall": [], "peak": []} for name in names}
        tokens_seen = {name: [] for name in names}

        for rep in range(3):
            order = names if rep % 2 == 0 else list(reversed(names))
            for cfg_name in order:
                mx.reset_peak_memory()
                mx.synchronize()
                res = engines[cfg_name].generate(prompt_ids, max_tokens, eos_ids)
                guard.record_gpu(res["total_ns"] / 1e9)
                tok_count = len(res["physical_tokens"])
                decode_s = res["decode_ns"] / 1e9
                samples[cfg_name]["ttft"].append(res["prefill_ns"] / 1e6)
                samples[cfg_name]["tps"].append((tok_count - 1) / max(decode_s, 1e-6))
                samples[cfg_name]["wall"].append(res["total_ns"] / 1e9)
                samples[cfg_name]["peak"].append(mx.get_peak_memory() / (1024 * 1024))
                tokens_seen[cfg_name].append(res["physical_tokens"])
            guard.required_break()

        for cfg_name in names:
            match = all(
                cand == base
                for cand, base in zip(tokens_seen[cfg_name], tokens_seen[base_name])
            )
            length_results[cfg_name] = {
                "ttft_ms": st.median(samples[cfg_name]["ttft"]),
                "decode_tps": st.median(samples[cfg_name]["tps"]),
                "wall_s": st.median(samples[cfg_name]["wall"]),
                "peak_mem_mb": st.median(samples[cfg_name]["peak"]),
                "tokens_identical": match,
            }
            r = length_results[cfg_name]
            print(
                f"[{cfg_name}]: TTFT {r['ttft_ms']:.2f} ms | Decode {r['decode_tps']:.2f} tok/s | "
                f"Wall {r['wall_s']:.3f} s | Peak RAM {r['peak_mem_mb']:.0f} MB | Identical: {match}"
            )

        # Print Gains against baseline
        base = length_results["Baseline (Unoptimized, r=1)"]
        print(f"\n   >>> Summary of Gains vs Baseline ({max_tokens} tokens) <<<")
        for cfg_name in ["Core (Compiled + HeadSkip, r=1)", "Core + Readback Bundled (r=8)", "Core + Deep Bundling (r=16)"]:
            cand = length_results[cfg_name]
            ttft_gain = (1.0 - cand["ttft_ms"] / base["ttft_ms"]) * 100.0
            tps_gain = (cand["decode_tps"] / base["decode_tps"] - 1.0) * 100.0
            wall_gain = (1.0 - cand["wall_s"] / base["wall_s"]) * 100.0
            print(
                f"   {cfg_name}:\n"
                f"      TTFT Reduction:    {ttft_gain:+.2f}%\n"
                f"      Decode TPS Gain:   {tps_gain:+.2f}%\n"
                f"      Wall Time Saved:   {wall_gain:+.2f}%\n"
                f"      Tokens Identical:  {cand['tokens_identical']}"
            )

        results_by_length[f"{max_tokens}_tokens"] = length_results

    del model, tokenizer, engines
    gc.collect()
    mx.clear_cache()
    mx.synchronize()

    return {
        "model_id": model_id,
        "tag": tag,
        "model_gb": model_gb,
        "prompt_tokens": prompt_len,
        "lengths": results_by_length,
    }


MODELS = {
    "Gemma_4B": ("mlx-community/gemma-3-4b-it-4bit", "Gemma 4B"),
    "Gemma_12B": ("mlx-community/gemma-3-12b-it-4bit", "Gemma 12B"),
}


def main():
    # continuous_gpu_limit lifted for this study by user decision 2026-09-03
    # (12B/256-token runs exceed 6 s); duty cycle and wall limit stay.
    #
    # One model per invocation: `benchmark_long_tasks.py Gemma_12B`. The process
    # exits between models so only one LLM is ever resident in unified memory.
    guard = harness_preconditions(allow_long_gpu=True)
    wanted = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)

    out_path = PROJECT_ROOT / "experiments" / "model_benchmark" / "long_tasks_benchmark_results.json"
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for key in wanted:
        model_id, tag = MODELS[key]
        all_results[key] = benchmark_model_long_tasks(model_id, tag, guard)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\n[saved] {tag} -> {out_path}")


if __name__ == "__main__":
    main()
