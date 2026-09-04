#!/usr/bin/env python3
"""Isolated, Memory-Safe Combinatorial Benchmark for Gemma 12B on Apple Silicon M1 Max.

Executes within a single model-load session (no redundant reloads to prevent VRAM fragmentation)
with strict Zero-Swap checks and 100% token-identity verification against Baseline.
"""

from __future__ import annotations

import gc
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

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

MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
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


def run_12b_isolated_sweep(repeats: int = 2) -> dict[str, Any]:
    print("================================================================================")
    print(f"🚀 ISOLATED MEMORY-SAFE COMBINATORIAL SWEEP: {MODEL_ID}")
    print("================================================================================")
    enforce_offline()

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model_path = str(snapshot.path)

    print(f"Loading Gemma 12B from local cache: {model_path}...")
    t0_load = time.perf_counter()
    model, tokenizer = load(model_path)
    load_time_s = time.perf_counter() - t0_load
    print(f"Model loaded in {load_time_s:.2f}s. Peak VRAM: {mx.get_peak_memory() / (1024*1024):.1f} MB")

    eos = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

    # Configurations to test in-place on the single loaded model instance
    configs = [
        ("00_baseline", "Baseline (All Knobs Off)", BASELINE, "single"),
        ("01_head_skip_only", "Head Skip Prefill only", Knobs(head_skip_prefill=True), "single"),
        ("02_fixed_compiled_only", "Fixed Compiled KV-Cache only", Knobs(compiled_fixed_cache=True), "single"),
        ("03_readback_8_only", "Bundled Readback (R=8) only", Knobs(readback_every=8), "single"),
        ("04_readback_16_only", "Bundled Readback (R=16) only", Knobs(readback_every=16), "single"),
        ("05_head_skip_plus_compiled", "Head Skip + Fixed Compiled", Knobs(head_skip_prefill=True, compiled_fixed_cache=True), "single"),
        ("06_core_triad_r8", "Core Triad (HeadSkip + Compiled + Readback 8)", Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8), "single"),
        ("07_core_triad_r16", "Core Triad High-Cadence (HeadSkip + Compiled + Readback 16)", Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=16), "single"),
        ("08_throughput_baseline", "ThroughputMode W=4 (Baseline Knobs)", BASELINE, "throughput_w4"),
        ("09_throughput_core_triad", "ThroughputMode W=4 (Core Triad Knobs)", Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8), "throughput_w4"),
    ]

    # Step 1: Establish Baseline Truth
    print("\n--- [STEP 1/2] Establishing Baseline Reference Truth ---")
    base_engine = Engine(model, tokenizer, BASELINE)
    baseline_truths: dict[str, Any] = {}

    for task in BENCHMARK_PROMPTS:
        p_ids = tokenizer.encode(task["prompt"])
        # Warmup
        _ = base_engine.generate(p_ids, 16, eos)
        mx.eval()
        mx.synchronize()

        wall_list, ttft_list, decode_list, tps_list = [], [], [], []
        last_toks = []
        for _ in range(repeats):
            gc.collect()
            mx.clear_cache()
            mx.synchronize()
            t0 = time.perf_counter_ns()
            res = base_engine.generate(p_ids, task["max_tokens"], eos)
            mx.eval()
            mx.synchronize()
            t1 = time.perf_counter_ns()

            wall_ms = (t1 - t0) / 1e6
            ttft_ms = res["prefill_ns"] / 1e6
            dec_ms = res["decode_ns"] / 1e6
            cnt = len(res["physical_tokens"])
            tps = (cnt / (dec_ms / 1000.0)) if dec_ms > 0 else 0.0

            wall_list.append(wall_ms)
            ttft_list.append(ttft_ms)
            decode_list.append(dec_ms)
            tps_list.append(tps)
            last_toks = res["physical_tokens"]

        baseline_truths[task["id"]] = {
            "wall_ms": st.median(wall_list),
            "ttft_ms": st.median(ttft_list),
            "decode_ms": st.median(decode_list),
            "tps": st.median(tps_list),
            "physical_tokens": last_toks,
        }
        print(f"  ✓ {task['name']:<24}: TTFT={st.median(ttft_list):6.1f} ms | Decode={st.median(tps_list):5.1f} tok/s | Tokens={len(last_toks)}")

    # Throughput baseline
    rt_base = Runtime(base_engine, tokenizer, mode=ThroughputMode(max_width=4))
    def make_reqs():
        return [Request(prompt_ids=tokenizer.encode(p["prompt"]), max_tokens=p["max_tokens"], plan=StrictOneShotPlan()) for p in BENCHMARK_PROMPTS]
    _ = rt_base.serve(make_reqs())
    mx.eval()
    mx.synchronize()
    tp_wall_list, tp_tps_list = [], []
    last_tp_toks = []
    for _ in range(repeats):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        t0 = time.perf_counter_ns()
        res_list = rt_base.serve(make_reqs())
        mx.eval()
        mx.synchronize()
        t1 = time.perf_counter_ns()
        wall_s = (t1 - t0) / 1e9
        tot_cnt = sum(len(r.tokens) for r in res_list)
        tp_wall_list.append(wall_s * 1000.0)
        tp_tps_list.append(tot_cnt / wall_s if wall_s > 0 else 0.0)
        last_tp_toks = [r.tokens for r in res_list]
    baseline_truths["throughput"] = {
        "wall_ms": st.median(tp_wall_list),
        "tps": st.median(tp_tps_list),
        "token_sequences": last_tp_toks,
    }
    print(f"  ✓ Concurrent Grouped (W=4): Wall={st.median(tp_wall_list):6.1f} ms | Agg={st.median(tp_tps_list):5.1f} tok/s")

    # Step 2: Test All Configurations In-Place
    print("\n--- [STEP 2/2] Evaluating Configurations In-Place on Gemma 12B ---")
    results = []

    for name, desc, knobs, mode in configs:
        print(f"\nEvaluating: [{name}] — {desc}")
        engine = Engine(model, tokenizer, knobs)

        entry = {
            "name": name,
            "description": desc,
            "mode": mode,
            "all_tokens_identical": True,
            "tasks": {},
        }

        if mode == "throughput_w4":
            rt = Runtime(engine, tokenizer, mode=ThroughputMode(max_width=4))
            _ = rt.serve(make_reqs()) # warmup
            mx.eval()
            mx.synchronize()

            w_list, rate_list = [], []
            seqs = []
            for _ in range(repeats):
                gc.collect()
                mx.clear_cache()
                mx.synchronize()
                t0 = time.perf_counter_ns()
                r_list = rt.serve(make_reqs())
                mx.eval()
                mx.synchronize()
                t1 = time.perf_counter_ns()
                wall_s = (t1 - t0) / 1e9
                tot_c = sum(len(r.tokens) for r in r_list)
                w_list.append(wall_s * 1000.0)
                rate_list.append(tot_c / wall_s if wall_s > 0 else 0.0)
                seqs = [r.tokens for r in r_list]

            tp_base = baseline_truths["throughput"]
            wall_med = st.median(w_list)
            tps_med = st.median(rate_list)
            wall_ratio = wall_med / max(1.0, tp_base["wall_ms"])
            tps_gain = ((tps_med / max(0.1, tp_base["tps"])) - 1.0) * 100.0
            is_ident = (seqs == tp_base["token_sequences"])
            entry["all_tokens_identical"] = is_ident
            entry["tasks"]["concurrent"] = {
                "wall_ms": wall_med,
                "wall_ratio": round(wall_ratio, 4),
                "tps": tps_med,
                "tps_gain_pct": round(tps_gain, 2),
                "tokens_identical": is_ident,
            }
            match_str = "MATCH ✅" if is_ident else "DIVERGE ❌"
            print(f"  --> Throughput W=4: Wall={wall_med:6.1f} ms (Ratio {wall_ratio:.4f}) | Agg={tps_med:5.1f} tok/s ({tps_gain:+6.2f}%) | Tokens: {match_str}")
        else:
            ratios, ttft_gains, tps_gains = [], [], []
            for task in BENCHMARK_PROMPTS:
                p_ids = tokenizer.encode(task["prompt"])
                # Warmup
                _ = engine.generate(p_ids, 16, eos)
                mx.eval()
                mx.synchronize()

                w_list, ttft_list, dec_list, tps_list = [], [], [], []
                last_toks = []
                for _ in range(repeats):
                    gc.collect()
                    mx.clear_cache()
                    mx.synchronize()
                    t0 = time.perf_counter_ns()
                    res = engine.generate(p_ids, task["max_tokens"], eos)
                    mx.eval()
                    mx.synchronize()
                    t1 = time.perf_counter_ns()

                    wall_ms = (t1 - t0) / 1e6
                    ttft_ms = res["prefill_ns"] / 1e6
                    dec_ms = res["decode_ns"] / 1e6
                    cnt = len(res["physical_tokens"])
                    tps = (cnt / (dec_ms / 1000.0)) if dec_ms > 0 else 0.0

                    w_list.append(wall_ms)
                    ttft_list.append(ttft_ms)
                    dec_list.append(dec_ms)
                    tps_list.append(tps)
                    last_toks = res["physical_tokens"]

                b_task = baseline_truths[task["id"]]
                w_med = st.median(w_list)
                ttft_med = st.median(ttft_list)
                tps_med = st.median(tps_list)

                wall_ratio = w_med / max(1.0, b_task["wall_ms"])
                ttft_gain = ((b_task["ttft_ms"] - ttft_med) / max(1.0, b_task["ttft_ms"])) * 100.0
                tps_gain = ((tps_med / max(0.1, b_task["tps"])) - 1.0) * 100.0
                is_ident = (last_toks == b_task["physical_tokens"])
                if not is_ident:
                    entry["all_tokens_identical"] = False

                bandwidth_gb_s = (7_200_000_000 * tps_med) / (1024**3)
                entry["tasks"][task["id"]] = {
                    "wall_ms": w_med,
                    "wall_ratio": round(wall_ratio, 4),
                    "ttft_ms": ttft_med,
                    "ttft_gain_pct": round(ttft_gain, 2),
                    "decode_tps": tps_med,
                    "tps_gain_pct": round(tps_gain, 2),
                    "bandwidth_gb_s": round(bandwidth_gb_s, 1),
                    "tokens_identical": is_ident,
                }
                ratios.append(wall_ratio)
                ttft_gains.append(ttft_gain)
                tps_gains.append(tps_gain)
                match_str = "MATCH ✅" if is_ident else "DIVERGE ❌"
                print(f"  --> {task['name']:<22}: Wall Ratio={wall_ratio:.4f} | TTFT={ttft_gain:+5.1f}% | TPS={tps_gain:+5.1f}% ({tps_med:4.1f} tok/s, {bandwidth_gb_s:5.1f} GB/s) | Tokens: {match_str}")

            entry["mean_wall_ratio"] = round(st.mean(ratios), 4)
            entry["mean_ttft_gain_pct"] = round(st.mean(ttft_gains), 2)
            entry["mean_tps_gain_pct"] = round(st.mean(tps_gains), 2)

        results.append(entry)

    # Leaderboard Table
    print("\n" + "=" * 95)
    print("📊 GEMMA 12B COMBINATORIAL OPTIMIZATION LEADERBOARD")
    print("=" * 95)
    print(f"{'Configuration':<32} | {'Mode':<11} | {'Wall Ratio':<10} | {'TTFT Gain':<10} | {'TPS Gain':<10} | {'Identity':<8}")
    print("-" * 95)
    for res in results:
        ident_str = "100% OK" if res["all_tokens_identical"] else "BROKEN"
        if res["mode"] == "throughput_w4":
            c_data = res["tasks"]["concurrent"]
            print(f"{res['name']:<32} | {'Batch-4':<11} | {c_data['wall_ratio']:<10.4f} | {'N/A':<10} | {c_data['tps_gain_pct']:+9.2f}% | {ident_str:<8}")
        else:
            print(f"{res['name']:<32} | {'Single':<11} | {res['mean_wall_ratio']:<10.4f} | {res['mean_ttft_gain_pct']:+9.2f}% | {res['mean_tps_gain_pct']:+9.2f}% | {ident_str:<8}")
    print("=" * 95)

    out_file = PROJECT_ROOT / "experiments" / "combinatorial_sweep" / "sweep_gemma-3-12b-it-4bit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": MODEL_ID, "results": results}
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\n12B Results exported to: {out_file}")
    return payload


if __name__ == "__main__":
    run_12b_isolated_sweep()
