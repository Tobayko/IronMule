#!/usr/bin/env python3
"""Test whether ThroughputMode + Core Knobs achieves >20% throughput gain on Gemma 12B.

Compares:
- Arm A: Baseline Interactive (Knobs default, InteractiveMode)
- Arm C: Optimized Interactive (Core Knobs, InteractiveMode)
- Arm D: Optimized Throughput (Core Knobs, ThroughputMode(max_width=4))

Measures total wall time, total tokens generated, and aggregate token throughput (tok/s).
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

MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
PROMPTS = [
    "Explain why Apple Silicon unified memory reduces memory copy overhead in AI workloads.",
    "Write a Python function to compute the moving average of a list using a sliding window.",
    "Solve this step-by-step: A shop offers 20% discount on $60, then 10% tax. What is final cost?",
    "Summarize why speculative decoding requires exact token verification in greedy mode.",
]
MAX_TOKENS = 48
WARMUP = 1
REPEATS = 3


def main():
    enforce_offline()
    print("=== LOADING GEMMA 12B MODEL ===")
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    print("Model loaded successfully.")

    # Knobs configurations
    knobs_baseline = BASELINE
    knobs_optimized = Knobs(
        compiled_fixed_cache=True,
        head_skip_prefill=True,
        fuse_projections=True,
        readback_every=8,
    )

    print("\nInitializing Runtimes...")
    engine_a = Engine(model, tokenizer, knobs_baseline)
    engine_opt = Engine(model, tokenizer, knobs_optimized)

    rt_a = Runtime(engine_a, tokenizer, mode=InteractiveMode())
    rt_c = Runtime(engine_opt, tokenizer, mode=InteractiveMode())
    rt_d = Runtime(engine_opt, tokenizer, mode=ThroughputMode(max_width=4))

    # Build requests
    def make_requests():
        return [
            Request(prompt_ids=tokenizer.encode(p), max_tokens=MAX_TOKENS, plan=StrictOneShotPlan())
            for p in PROMPTS
        ]

    # Warmup
    print("Warming up all runtimes...")
    _ = rt_a.serve(make_requests())
    _ = rt_c.serve(make_requests())
    _ = rt_d.serve(make_requests())
    mx.eval()
    mx.synchronize()

    results = {}
    arms = [
        ("Arm A (Baseline Interactive)", rt_a),
        ("Arm C (Optimized Interactive)", rt_c),
        ("Arm D (Optimized Throughput W=4)", rt_d),
    ]

    print("\n=== MEASURING REAL PERFORMANCE ON 4 CONCURRENT REQUESTS ===")

    for label, rt in arms:
        wall_times = []
        rates = []
        all_tokens = []

        for rep in range(REPEATS):
            mx.eval()
            mx.synchronize()
            t0 = time.perf_counter()
            reqs = make_requests()
            res = rt.serve(reqs)
            t1 = time.perf_counter()

            wall_s = t1 - t0
            tot_tokens = sum(len(r.tokens) for r in res)
            rate = tot_tokens / wall_s

            wall_times.append(wall_s)
            rates.append(rate)
            if rep == REPEATS - 1:
                all_tokens = [r.tokens for r in res]

        med_wall = st.median(wall_times)
        med_rate = st.median(rates)
        results[label] = {
            "wall_s_median": med_wall,
            "rate_tok_s_median": med_rate,
            "tokens": all_tokens,
        }
        print(f"[{label}]: Wall={med_wall:.3f}s, Aggregate Rate={med_rate:.2f} tok/s")

    # Comparisons against Arm A
    base_wall = results["Arm A (Baseline Interactive)"]["wall_s_median"]
    base_rate = results["Arm A (Baseline Interactive)"]["rate_tok_s_median"]

    print("\n=== COMPARISON & SPEEDUP SUMMARY ===")
    for label in ["Arm C (Optimized Interactive)", "Arm D (Optimized Throughput W=4)"]:
        wall = results[label]["wall_s_median"]
        rate = results[label]["rate_tok_s_median"]
        wall_red = (1.0 - wall / base_wall) * 100.0
        rate_gain = (rate / base_rate - 1.0) * 100.0
        print(f"{label}:")
        print(f"  Wall Time Reduction: {wall_red:+.2f}% (Ratio: {wall/base_wall:.4f})")
        print(f"  Throughput Gain:     {rate_gain:+.2f}% (Rate Ratio: {rate/base_rate:.4f})")

    out_file = PROJECT_ROOT / "experiments" / "model_benchmark" / "gemma_12b_throughput_verification.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "model_id": MODEL_ID,
        "base_wall_s": base_wall,
        "base_rate_tok_s": base_rate,
        "results": {
            k: {
                "wall_s": v["wall_s_median"],
                "rate_tok_s": v["rate_tok_s_median"],
            }
            for k, v in results.items()
        },
    }, indent=2))
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
