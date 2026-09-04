#!/usr/bin/env python3
"""Benchmark Long Tasks under ThroughputMode W=4 on Apple Silicon (M1 Max).

Evaluates:
- 4 Concurrent Long Requests
- Prompt Context: 379 tokens
- Output: 128 tokens per request (Total 512 generated tokens)
Across:
- Arm A: Baseline Interactive (Knobs default, InteractiveMode)
- Arm C: Core Interactive (Core Knobs, InteractiveMode)
- Arm D: Core Throughput W=4 (Core Knobs, ThroughputMode(max_width=4))
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
from tools._bench import enforce_offline, resolve_local_model_snapshot
from mlx_lm import load

from ironmule.runtime import Knobs, Engine, BASELINE
from ironmule.service import Runtime, InteractiveMode, ThroughputMode, Request, StrictOneShotPlan

LONG_PROMPT_PREFIX = (
    "Apple Silicon architecture integrates the CPU, GPU, Neural Engine, and Secure Enclave onto a single "
    "System on a Chip (SoC) using unified memory architecture (UMA). In traditional PC architectures, the CPU and GPU "
    "maintain separate physical memory pools and communicate over a PCIe bus. This discrete design introduces "
    "significant latency and serialization overhead, as data must be explicitly copied across the bus. "
    "In contrast, Apple unified memory allows all processing engines on the SoC to access a single, high-bandwidth memory pool "
    "without data duplication. For large language model inference, which is fundamentally bounded by memory bandwidth during "
    "autoregressive token generation, unified memory provides a massive architectural advantage. "
)

QUESTIONS = [
    LONG_PROMPT_PREFIX + "Analyze how memory bandwidth bounds the autoregressive decoding phase.",
    LONG_PROMPT_PREFIX + "Discuss how prefill logit slicing eliminates redundant matrix multiplies.",
    LONG_PROMPT_PREFIX + "Explain why constant-shape KV caching is required for Metal graph compilation.",
    LONG_PROMPT_PREFIX + "Evaluate how asynchronous readback bundling minimizes CPU-GPU sync stalls.",
]
MAX_TOKENS = 128


def run_long_throughput_benchmark(model_id: str, tag: str, model_gb: float) -> dict:
    print(f"\n=======================================================")
    print(f"=== LONG TASK CONCURRENT THROUGHPUT: {tag} ===")
    print(f"=======================================================")

    snapshot = resolve_local_model_snapshot(model_id)
    model, tokenizer = load(str(snapshot.path))

    knobs_opt = Knobs(
        compiled_fixed_cache=True,
        head_skip_prefill=True,
    )

    engine_a = Engine(model, tokenizer, BASELINE)
    engine_c = Engine(model, tokenizer, knobs_opt)
    engine_d = Engine(model, tokenizer, knobs_opt)

    rt_a = Runtime(engine_a, tokenizer, mode=InteractiveMode())
    rt_c = Runtime(engine_c, tokenizer, mode=InteractiveMode())
    rt_d = Runtime(engine_d, tokenizer, mode=ThroughputMode(max_width=4))

    def make_reqs():
        return [
            Request(prompt_ids=tokenizer.encode(q), max_tokens=MAX_TOKENS, plan=StrictOneShotPlan())
            for q in QUESTIONS
        ]

    # Warmup
    print("Warming up runtimes...")
    _ = rt_a.serve(make_reqs()[:2])
    _ = rt_c.serve(make_reqs()[:2])
    _ = rt_d.serve(make_reqs()[:2])

    arms = [
        ("Arm A (Baseline Interactive)", rt_a),
        ("Arm C (Core Interactive)", rt_c),
        ("Arm D (Core Throughput W=4)", rt_d),
    ]

    measurements = {}
    tokens_by_arm = {}

    for label, rt in arms:
        walls = []
        rates = []
        tokens = []

        for rep in range(3):
            mx.reset_peak_memory()
            mx.eval()
            mx.synchronize()
            t0 = time.perf_counter()
            res = rt.serve(make_reqs())
            mx.eval()
            mx.synchronize()
            t1 = time.perf_counter()

            wall_s = t1 - t0
            tot_tokens = sum(len(r.tokens) for r in res)
            rate = tot_tokens / wall_s
            walls.append(wall_s)
            rates.append(rate)
            if rep == 0:
                tokens = [r.tokens for r in res]

        med_wall = st.median(walls)
        med_rate = st.median(rates)
        measurements[label] = {
            "wall_s": med_wall,
            "rate_tok_s": med_rate,
            "peak_mb": mx.get_peak_memory() / (1024 * 1024),
        }
        tokens_by_arm[label] = tokens
        eff_bw = (model_gb * med_rate)
        print(f"[{label}]: Wall={med_wall:.3f}s | Rate={med_rate:.2f} tok/s | Eff. BW={eff_bw:.1f} GB/s")

    base_wall = measurements["Arm A (Baseline Interactive)"]["wall_s"]
    base_rate = measurements["Arm A (Baseline Interactive)"]["rate_tok_s"]

    # Verify Token Identity
    tok_a = tokens_by_arm["Arm A (Baseline Interactive)"]
    tok_d = tokens_by_arm["Arm D (Core Throughput W=4)"]
    matches = sum(1 for a, d in zip(tok_a, tok_d) if a == d)
    print(f"\nToken Identity Match (Arm D vs Arm A): {matches}/{len(QUESTIONS)} ({matches/len(QUESTIONS)*100:.1f}%)")

    print("\n--- Speedup Summary ---")
    for label in ["Arm C (Core Interactive)", "Arm D (Core Throughput W=4)"]:
        wall = measurements[label]["wall_s"]
        rate = measurements[label]["rate_tok_s"]
        wall_red = (1.0 - wall / base_wall) * 100.0
        rate_gain = (rate / base_rate - 1.0) * 100.0
        print(f"{label}:")
        print(f"  Wall Time Reduction: {wall_red:+.2f}% (Ratio: {wall/base_wall:.4f})")
        print(f"  Throughput Gain:     {rate_gain:+.2f}% (Rate Ratio: {rate/base_rate:.4f})")

    del model, tokenizer
    mx.eval()
    mx.synchronize()

    return {
        "model_id": model_id,
        "measurements": measurements,
        "identity_matches": matches,
    }


def main():
    enforce_offline()
    results = {}
    results["Gemma_4B"] = run_long_throughput_benchmark(
        "mlx-community/gemma-3-4b-it-4bit", "Gemma 4B IT 4-bit", 2.56
    )
    results["Gemma_12B"] = run_long_throughput_benchmark(
        "mlx-community/gemma-3-12b-it-4bit", "Gemma 12B IT 4-bit", 7.19
    )

    out_file = PROJECT_ROOT / "experiments" / "model_benchmark" / "long_tasks_concurrent_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
