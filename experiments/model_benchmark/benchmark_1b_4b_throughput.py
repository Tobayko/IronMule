#!/usr/bin/env python3
"""Empirical test of ThroughputMode + Core Knobs on Gemma 1B and Gemma 4B.

Tests:
- Arm A: Baseline Interactive (Knobs default, InteractiveMode)
- Arm C: Optimized Interactive (Core Knobs, InteractiveMode)
- Arm D: Core Throughput W=4 (Core Knobs, ThroughputMode(max_width=4))

Verifies:
- Wall time reduction
- Token throughput (tok/s) gain
- Exact token identity (100% match) across all questions
"""

from __future__ import annotations

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

QUESTIONS = [
    "What does unified memory avoid?",
    "How are quantised weights stored?",
    "Why must a timer synchronise?",
    "What bounds decoding?",
    "What does MLX do lazily?",
    "Which two units share the memory?",
]
MAX_TOKENS = 48
WARMUPS = 2
REPEATS = 5


def run_benchmark_for_model(model_id: str, tag: str) -> dict:
    print(f"\n=======================================================")
    print(f"=== BENCHMARKING {tag} ({model_id}) ===")
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
    print("Running warmups...")
    for _ in range(WARMUPS):
        _ = rt_a.serve(make_reqs()[:2])
        _ = rt_c.serve(make_reqs()[:2])
        _ = rt_d.serve(make_reqs()[:2])
        mx.eval()
        mx.synchronize()

    arms = [
        ("Arm A (Baseline Interactive)", rt_a),
        ("Arm C (Core Interactive)", rt_c),
        ("Arm D (Core Throughput W=4)", rt_d),
    ]

    measurements = {}
    tokens_by_arm = {}

    for label, rt in arms:
        wall_times = []
        rates = []
        arm_tokens = []

        for rep in range(REPEATS):
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

            wall_times.append(wall_s)
            rates.append(rate)
            if rep == REPEATS - 1:
                arm_tokens = [r.tokens for r in res]

        med_wall = st.median(wall_times)
        med_rate = st.median(rates)
        measurements[label] = {
            "wall_s_median": med_wall,
            "rate_tok_s_median": med_rate,
            "wall_times": wall_times,
            "rates": rates,
        }
        tokens_by_arm[label] = arm_tokens
        print(f"[{label}]: Wall={med_wall:.4f}s, Aggregate Rate={med_rate:.2f} tok/s")

    # Verify Token Identity
    print("\n--- Verifying Token Identity (Arm A vs Arm C vs Arm D) ---")
    tokens_a = tokens_by_arm["Arm A (Baseline Interactive)"]
    tokens_c = tokens_by_arm["Arm C (Core Interactive)"]
    tokens_d = tokens_by_arm["Arm D (Core Throughput W=4)"]

    c_matches = sum(1 for a, c in zip(tokens_a, tokens_c) if a == c)
    d_matches = sum(1 for a, d in zip(tokens_a, tokens_d) if a == d)

    print(f"Arm C vs Arm A Match: {c_matches}/{len(QUESTIONS)} ({c_matches/len(QUESTIONS)*100:.1f}%)")
    print(f"Arm D vs Arm A Match: {d_matches}/{len(QUESTIONS)} ({d_matches/len(QUESTIONS)*100:.1f}%)")

    base_wall = measurements["Arm A (Baseline Interactive)"]["wall_s_median"]
    base_rate = measurements["Arm A (Baseline Interactive)"]["rate_tok_s_median"]

    summary = {
        "model_id": model_id,
        "tag": tag,
        "base_wall_s": base_wall,
        "base_rate_tok_s": base_rate,
        "arms": measurements,
        "comparisons": {},
        "token_identity": {
            "arm_c_matches": c_matches,
            "arm_d_matches": d_matches,
            "total_questions": len(QUESTIONS),
        },
    }

    print("\n--- Summary of Gains vs Baseline ---")
    for label in ["Arm C (Core Interactive)", "Arm D (Core Throughput W=4)"]:
        wall = measurements[label]["wall_s_median"]
        rate = measurements[label]["rate_tok_s_median"]
        wall_red = (1.0 - wall / base_wall) * 100.0
        rate_gain = (rate / base_rate - 1.0) * 100.0
        summary["comparisons"][label] = {
            "wall_reduction_pct": wall_red,
            "wall_ratio": wall / base_wall,
            "rate_gain_pct": rate_gain,
            "rate_ratio": rate / base_rate,
        }
        print(f"{label}:")
        print(f"  Wall Time Reduction: {wall_red:+.2f}% (Ratio: {wall/base_wall:.4f})")
        print(f"  Throughput Gain:     {rate_gain:+.2f}% (Rate Ratio: {rate/base_rate:.4f})")

    # Clean up memory
    del model, tokenizer, engine_a, engine_c, engine_d, rt_a, rt_c, rt_d
    mx.eval()
    mx.synchronize()

    return summary


def main():
    enforce_offline()
    results = {}

    # 1. Gemma 1B
    results["Gemma_1B"] = run_benchmark_for_model(
        model_id="mlx-community/gemma-3-1b-it-4bit",
        tag="Gemma 1B IT 4-bit",
    )

    # 2. Gemma 4B
    results["Gemma_4B"] = run_benchmark_for_model(
        model_id="mlx-community/gemma-3-4b-it-4bit",
        tag="Gemma 4B IT 4-bit",
    )

    out_file = PROJECT_ROOT / "experiments" / "model_benchmark" / "gemma_1b_4b_throughput_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nAll results saved to: {out_file}")


if __name__ == "__main__":
    main()
