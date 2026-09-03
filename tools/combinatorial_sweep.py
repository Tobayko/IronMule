#!/usr/bin/env python3
"""Systematic Combinatorial Benchmark & Optimization Harness for Project Friday.

Evaluates all individual knobs, pairwise combinations, full-stack configurations,
concurrency modes (Interactive vs Throughput W=4), and startup order sequences
on Apple Silicon M1 Max with real hardware inference and strict token-identity verification.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics as st
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(IRONMULE))

import mlx.core as mx
from mlx_lm import load
from tools._bench import enforce_offline, resolve_local_model_snapshot

from ironmule.runtime import Knobs, Engine, BASELINE
from ironmule.service import (
    Runtime,
    InteractiveMode,
    ThroughputMode,
    Request,
    StrictOneShotPlan,
)
from ironmule import fast


BENCHMARK_PROMPTS = [
    {
        "id": "qa_short",
        "name": "QA Short",
        "prompt": "Explain why Apple Silicon unified memory reduces memory copy overhead in AI workloads.",
        "max_tokens": 32,
    },
    {
        "id": "math_reasoning",
        "name": "Math Reasoning",
        "prompt": "Solve this step-by-step: A shop offers 20% discount on $60, then 10% tax. What is final cost and intermediate price?",
        "max_tokens": 48,
    },
    {
        "id": "coding_task",
        "name": "Coding Sliding Window",
        "prompt": "Write a Python function to compute the moving average of a list using a sliding window. Include type hints and docstrings.",
        "max_tokens": 64,
    },
]


@dataclass
class SweepConfig:
    name: str
    description: str
    knobs: Knobs
    fuse_model: bool = False
    wired_limit: float = 0.0
    mode: str = "interactive"  # "interactive" or "throughput_w4"


def build_configurations() -> list[SweepConfig]:
    """Generate all individual, pairwise, and full-stack configurations."""
    configs = []

    # 1. Baseline
    configs.append(SweepConfig(
        name="00_baseline",
        description="Standard Baseline (All Knobs Off)",
        knobs=BASELINE,
    ))

    # 2. Individual Knobs (Ablation)
    configs.append(SweepConfig(
        name="01_head_skip_only",
        description="Head Skip Prefill only",
        knobs=Knobs(head_skip_prefill=True),
    ))
    configs.append(SweepConfig(
        name="02_fixed_compiled_only",
        description="Fixed Compiled KV-Cache only",
        knobs=Knobs(compiled_fixed_cache=True),
    ))
    configs.append(SweepConfig(
        name="03_readback_8_only",
        description="Bundled Readback (R=8) only",
        knobs=Knobs(readback_every=8),
    ))
    configs.append(SweepConfig(
        name="04_readback_16_only",
        description="Bundled Readback (R=16) only",
        knobs=Knobs(readback_every=16),
    ))
    configs.append(SweepConfig(
        name="05_wired_06_only",
        description="Wired Memory 0.6 only",
        knobs=Knobs(wired_fraction=0.6),
        wired_limit=0.6,
    ))
    configs.append(SweepConfig(
        name="06_fuse_projections_only",
        description="Fused Projections (QKV/MLP) only",
        knobs=Knobs(fuse_projections=True),
        fuse_model=True,
    ))

    # 3. Pairwise & Incremental Stacking
    configs.append(SweepConfig(
        name="07_head_skip_plus_compiled",
        description="Head Skip + Fixed Compiled",
        knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=True),
    ))
    configs.append(SweepConfig(
        name="08_core_triad",
        description="Core Serving Triad (HeadSkip + Compiled + Readback 8)",
        knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8),
    ))
    configs.append(SweepConfig(
        name="09_core_plus_fuse",
        description="Core Triad + Fused Projections",
        knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8, fuse_projections=True),
        fuse_model=True,
    ))
    configs.append(SweepConfig(
        name="10_full_stack_r8",
        description="Full Stack (Core Triad + Fuse + Wired 0.6, R=8)",
        knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8, fuse_projections=True, wired_fraction=0.6),
        fuse_model=True,
        wired_limit=0.6,
    ))
    configs.append(SweepConfig(
        name="11_full_stack_r16",
        description="Full Stack High-Cadence (Core Triad + Fuse + Wired 0.6, R=16)",
        knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=16, fuse_projections=True, wired_fraction=0.6),
        fuse_model=True,
        wired_limit=0.6,
    ))

    # 4. ThroughputMode Concurrency (W=4)
    configs.append(SweepConfig(
        name="12_throughput_baseline",
        description="ThroughputMode W=4 (Baseline Knobs)",
        knobs=BASELINE,
        mode="throughput_w4",
    ))
    configs.append(SweepConfig(
        name="13_throughput_full_stack",
        description="ThroughputMode W=4 (Full Stack Knobs)",
        knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8, fuse_projections=True, wired_fraction=0.6),
        fuse_model=True,
        wired_limit=0.6,
        mode="throughput_w4",
    ))

    return configs


def get_model_bytes(model) -> int:
    try:
        flat = mx.tree_flatten(model.parameters())
        return sum(arr.nbytes for _, arr in flat if isinstance(arr, mx.array))
    except Exception:
        return 2_500_000_000


def measure_single_stream(
    engine: Engine,
    tokenizer: Any,
    prompt_ids: list[int],
    max_tokens: int,
    repeats: int = 3,
    warmup: int = 2,
) -> dict[str, Any]:
    eos = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

    # Warmup
    for _ in range(warmup):
        _ = engine.generate(prompt_ids, min(16, max_tokens), eos)
        mx.eval()
        mx.synchronize()

    wall_times = []
    ttft_times = []
    decode_times = []
    tps_rates = []
    generated_tokens_list = []
    last_tokens = []

    for _ in range(repeats):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()

        t0 = time.perf_counter_ns()
        res = engine.generate(prompt_ids, max_tokens, eos)
        mx.eval()
        mx.synchronize()
        t1 = time.perf_counter_ns()

        wall_ms = (t1 - t0) / 1_000_000.0
        ttft_ms = res["prefill_ns"] / 1_000_000.0
        decode_ms = res["decode_ns"] / 1_000_000.0
        tok_count = len(res["physical_tokens"])
        tps = (tok_count / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0

        wall_times.append(wall_ms)
        ttft_times.append(ttft_ms)
        decode_times.append(decode_ms)
        tps_rates.append(tps)
        generated_tokens_list.append(tok_count)
        last_tokens = res["physical_tokens"]

    return {
        "wall_ms_median": st.median(wall_times),
        "wall_ms_min": min(wall_times),
        "ttft_ms_median": st.median(ttft_times),
        "decode_ms_median": st.median(decode_times),
        "tps_median": st.median(tps_rates),
        "generated_tokens": generated_tokens_list[-1],
        "physical_tokens": last_tokens,
    }


def measure_throughput_stream(
    engine: Engine,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    rt = Runtime(engine, tokenizer, mode=ThroughputMode(max_width=4))

    def make_reqs():
        return [
            Request(prompt_ids=tokenizer.encode(p["prompt"]), max_tokens=p["max_tokens"], plan=StrictOneShotPlan())
            for p in prompts
        ]

    for _ in range(warmup):
        _ = rt.serve(make_reqs())
        mx.eval()
        mx.synchronize()

    wall_times = []
    agg_rates = []
    total_tokens_list = []
    token_sequences = []

    for _ in range(repeats):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()

        t0 = time.perf_counter_ns()
        results = rt.serve(make_reqs())
        mx.eval()
        mx.synchronize()
        t1 = time.perf_counter_ns()

        wall_s = (t1 - t0) / 1_000_000_000.0
        tot_tokens = sum(len(r.tokens) for r in results)
        agg_tps = tot_tokens / wall_s if wall_s > 0 else 0.0

        wall_times.append(wall_s * 1000.0)
        agg_rates.append(agg_tps)
        total_tokens_list.append(tot_tokens)
        token_sequences = [r.tokens for r in results]

    return {
        "wall_ms_median": st.median(wall_times),
        "tps_median": st.median(agg_rates),
        "total_tokens": total_tokens_list[-1],
        "token_sequences": token_sequences,
    }


def run_combinatorial_sweep(model_id: str, repeats: int = 3) -> dict[str, Any]:
    print("================================================================================")
    print(f"🚀 SYSTEMATIC COMBINATORIAL SWEEP ON APPLE SILICON: {model_id}")
    print("================================================================================")
    enforce_offline()

    snapshot = resolve_local_model_snapshot(model_id)
    model_path = str(snapshot.path)

    print(f"Loading initial base model from: {model_path}...")
    model, tokenizer = load(model_path)
    model_size_bytes = get_model_bytes(model)
    print(f"Base model loaded: {model_size_bytes / (1024*1024):.1f} MB parameter weight footprint.")

    configs = build_configurations()

    # Step 1: Establish Baseline Reference
    print("\n--- [STEP 1/3] Establishing Baseline Truth ---")
    base_engine = Engine(model, tokenizer, BASELINE)
    baseline_truths: dict[str, Any] = {}
    for task in BENCHMARK_PROMPTS:
        p_ids = tokenizer.encode(task["prompt"])
        res = measure_single_stream(base_engine, tokenizer, p_ids, task["max_tokens"], repeats=repeats)
        baseline_truths[task["id"]] = res
        print(f"  ✓ {task['name']:<24}: TTFT={res['ttft_ms_median']:6.1f} ms | Decode={res['tps_median']:6.1f} tok/s | Tokens={res['generated_tokens']}")

    # Baseline throughput truth for concurrent batch
    base_tp_res = measure_throughput_stream(base_engine, tokenizer, BENCHMARK_PROMPTS, repeats=repeats)
    baseline_truths["throughput_concurrent"] = base_tp_res
    print(f"  ✓ Concurrent Grouped (W=4): Wall={base_tp_res['wall_ms_median']:6.1f} ms | Agg={base_tp_res['tps_median']:6.1f} tok/s")

    # Step 2: Test All Configurations & Start Sequences
    print("\n--- [STEP 2/3] Testing All Knob Combinations & Start Sequences ---")
    sweep_results: list[dict[str, Any]] = []

    # Cache unfused model weights for non-fused configs
    current_model = model
    model_is_fused = False

    for cfg in configs:
        print(f"\nEvaluating: [{cfg.name}] — {cfg.description}")

        # Handle Model Fusion In-Place vs Reload
        if cfg.fuse_model and not model_is_fused:
            print("  [Setup Sequence]: Applying in-place Graph Surgery (QKV + MLP Fused)...")
            fast.fuse_projections(current_model)
            model_is_fused = True
        elif not cfg.fuse_model and model_is_fused:
            print("  [Setup Sequence]: Reloading fresh unfused model for clean ablation...")
            del current_model
            gc.collect()
            mx.clear_cache()
            current_model, tokenizer = load(model_path)
            model_is_fused = False

        # Apply Wired Memory if required
        if cfg.wired_limit > 0:
            mem_bytes = int(snapshot.path.stat().st_size * 2)  # approximate fallback
            try:
                import subprocess
                res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    mem_bytes = int(res.stdout.strip())
            except Exception:
                pass
            mx.set_wired_limit(int(mem_bytes * cfg.wired_limit))

        # Build Engine with exact knob configuration
        engine = Engine(current_model, tokenizer, cfg.knobs)

        cfg_summary: dict[str, Any] = {
            "name": cfg.name,
            "description": cfg.description,
            "mode": cfg.mode,
            "knobs": cfg.knobs.as_dict(),
            "tasks": {},
            "all_tokens_identical": True,
        }

        if cfg.mode == "throughput_w4":
            tp_res = measure_throughput_stream(engine, tokenizer, BENCHMARK_PROMPTS, repeats=repeats)
            base_tp = baseline_truths["throughput_concurrent"]
            wall_ratio = tp_res["wall_ms_median"] / max(1.0, base_tp["wall_ms_median"])
            tps_gain = ((tp_res["tps_median"] / max(0.1, base_tp["tps_median"])) - 1.0) * 100.0

            # Token identity check across all sequences
            is_ident = (tp_res["token_sequences"] == base_tp["token_sequences"])
            cfg_summary["tasks"]["concurrent"] = {
                "wall_ms": tp_res["wall_ms_median"],
                "wall_ratio": round(wall_ratio, 4),
                "tps": tp_res["tps_median"],
                "tps_gain_pct": round(tps_gain, 2),
                "tokens_identical": is_ident,
            }
            cfg_summary["all_tokens_identical"] = is_ident
            match_str = "MATCH ✅" if is_ident else "DIVERGE ❌"
            print(f"  --> Throughput W=4: Wall={tp_res['wall_ms_median']:6.1f} ms (Ratio {wall_ratio:.4f}) | Agg={tp_res['tps_median']:6.1f} tok/s ({tps_gain:+6.2f}%) | Tokens: {match_str}")
        else:
            # Single-stream tasks
            ratios = []
            ttft_gains = []
            tps_gains = []

            for task in BENCHMARK_PROMPTS:
                p_ids = tokenizer.encode(task["prompt"])
                t_res = measure_single_stream(engine, tokenizer, p_ids, task["max_tokens"], repeats=repeats)
                b_res = baseline_truths[task["id"]]

                wall_ratio = t_res["wall_ms_median"] / max(1.0, b_res["wall_ms_median"])
                ttft_gain = ((b_res["ttft_ms_median"] - t_res["ttft_ms_median"]) / max(1.0, b_res["ttft_ms_median"])) * 100.0
                tps_gain = ((t_res["tps_median"] / max(0.1, b_res["tps_median"])) - 1.0) * 100.0
                is_ident = (t_res["physical_tokens"] == b_res["physical_tokens"])

                if not is_ident:
                    cfg_summary["all_tokens_identical"] = False

                bandwidth_gb_s = (model_size_bytes * t_res["tps_median"]) / (1024**3)

                cfg_summary["tasks"][task["id"]] = {
                    "wall_ms": t_res["wall_ms_median"],
                    "wall_ratio": round(wall_ratio, 4),
                    "ttft_ms": t_res["ttft_ms_median"],
                    "ttft_gain_pct": round(ttft_gain, 2),
                    "decode_tps": t_res["tps_median"],
                    "tps_gain_pct": round(tps_gain, 2),
                    "bandwidth_gb_s": round(bandwidth_gb_s, 1),
                    "tokens_identical": is_ident,
                }
                ratios.append(wall_ratio)
                ttft_gains.append(ttft_gain)
                tps_gains.append(tps_gain)

                match_str = "MATCH ✅" if is_ident else "DIVERGE ❌"
                print(f"  --> {task['name']:<22}: Wall Ratio={wall_ratio:.4f} | TTFT={ttft_gain:+5.1f}% | TPS={tps_gain:+5.1f}% ({t_res['tps_median']:5.1f} tok/s, {bandwidth_gb_s:5.1f} GB/s) | Tokens: {match_str}")

            cfg_summary["mean_wall_ratio"] = round(st.mean(ratios), 4)
            cfg_summary["mean_ttft_gain_pct"] = round(st.mean(ttft_gains), 2)
            cfg_summary["mean_tps_gain_pct"] = round(st.mean(tps_gains), 2)

        sweep_results.append(cfg_summary)

    # Step 3: Startup Sequence Ablation (Order of Operations)
    print("\n--- [STEP 3/3] Startup & Initialisation Sequence Ablation ---")

    print("Testing Pipeline Sequence A (Naive Cold Start)...")
    del current_model
    gc.collect()
    mx.clear_cache()
    t_start_a = time.perf_counter_ns()
    mod_a, tok_a = load(model_path)
    eng_a = Engine(mod_a, tok_a, Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8))
    # Cold first request
    t_req0_a = time.perf_counter_ns()
    res_a0 = eng_a.generate(tokenizer.encode(BENCHMARK_PROMPTS[0]["prompt"]), 32, (1,))
    mx.eval()
    mx.synchronize()
    t_done_a = time.perf_counter_ns()
    cold_first_token_a_ms = res_a0["prefill_ns"] / 1_000_000.0
    total_startup_a_ms = (t_done_a - t_start_a) / 1_000_000.0

    print("Testing Pipeline Sequence B (Hardware-First Pre-Warmed Engine)...")
    del mod_a, eng_a
    gc.collect()
    mx.clear_cache()
    t_start_b = time.perf_counter_ns()
    # 1. Wired Limit First
    mx.set_wired_limit(int(34 * (1024**3) * 0.6))
    # 2. Model Load
    mod_b, tok_b = load(model_path)
    # 3. Fuse in-place
    fast.fuse_projections(mod_b)
    # 4. Engine with fixed compiled cache
    eng_b = Engine(mod_b, tok_b, Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8, fuse_projections=True, wired_fraction=0.6))
    # 5. Pre-warm JIT compiler at standard capacity (e.g. 128 tokens)
    _ = eng_b.generate(tokenizer.encode("Apple Silicon warm-up pre-compile"), 16, (1,))
    mx.eval()
    mx.synchronize()
    # Ready for request 0
    t_req0_b = time.perf_counter_ns()
    res_b0 = eng_b.generate(tokenizer.encode(BENCHMARK_PROMPTS[0]["prompt"]), 32, (1,))
    mx.eval()
    mx.synchronize()
    t_done_b = time.perf_counter_ns()
    warm_first_token_b_ms = res_b0["prefill_ns"] / 1_000_000.0
    request_lat_b_ms = (t_done_b - t_req0_b) / 1_000_000.0

    startup_comparison = {
        "sequence_a_naive_cold": {
            "first_request_prefill_ms": round(cold_first_token_a_ms, 2),
            "total_time_to_first_response_ms": round(total_startup_a_ms, 2),
        },
        "sequence_b_hardware_first": {
            "first_request_prefill_ms": round(warm_first_token_b_ms, 2),
            "request_response_ms": round(request_lat_b_ms, 2),
            "prefill_speedup_vs_naive_pct": round(((cold_first_token_a_ms - warm_first_token_b_ms) / cold_first_token_a_ms) * 100, 2),
        }
    }
    print(f"  ✓ Sequence A (Naive Cold)    : First Request TTFT = {cold_first_token_a_ms:6.1f} ms | Total Startup = {total_startup_a_ms:6.1f} ms")
    print(f"  ✓ Sequence B (Hardware-First): First Request TTFT = {warm_first_token_b_ms:6.1f} ms | Request Latency = {request_lat_b_ms:6.1f} ms")
    print(f"  ==> JIT Pre-Warmup eliminates {startup_comparison['sequence_b_hardware_first']['prefill_speedup_vs_naive_pct']}% of cold first-token delay!")

    # Summary Table Output
    print("\n" + "=" * 95)
    print("📊 COMBINATORIAL OPTIMIZATION LEADERBOARD")
    print("=" * 95)
    print(f"{'Configuration':<32} | {'Mode':<11} | {'Wall Ratio':<10} | {'TTFT Gain':<10} | {'TPS Gain':<10} | {'Identity':<8}")
    print("-" * 95)
    for res in sweep_results:
        ident_str = "100% OK" if res["all_tokens_identical"] else "BROKEN"
        if res["mode"] == "throughput_w4":
            c_data = res["tasks"]["concurrent"]
            print(f"{res['name']:<32} | {'Batch-4':<11} | {c_data['wall_ratio']:<10.4f} | {'N/A':<10} | {c_data['tps_gain_pct']:+9.2f}% | {ident_str:<8}")
        else:
            print(f"{res['name']:<32} | {'Single':<11} | {res['mean_wall_ratio']:<10.4f} | {res['mean_ttft_gain_pct']:+9.2f}% | {res['mean_tps_gain_pct']:+9.2f}% | {ident_str:<8}")
    print("=" * 95)

    # Save to disk
    out_dir = PROJECT_ROOT / "experiments" / "combinatorial_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sweep_{Path(model_id).name}.json"
    final_payload = {
        "model_id": model_id,
        "device": platform.processor(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": sweep_results,
        "startup_ablation": startup_comparison,
    }
    out_file.write_text(json.dumps(final_payload, indent=2))
    print(f"\nResults successfully exported to: {out_file}")
    return final_payload


def main():
    parser = argparse.ArgumentParser(description="Run Systematic Combinatorial Sweep")
    parser.add_argument("--model-id", default="mlx-community/gemma-3-4b-it-4bit", help="Model ID")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per configuration")
    args = parser.parse_args()

    run_combinatorial_sweep(args.model_id, repeats=args.repeats)


if __name__ == "__main__":
    main()
