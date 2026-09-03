#!/usr/bin/env python3
"""Benchmark Draft-Model Speculative Decoding on Apple Silicon M1 Max.

Model Pairing:
- Draft:  mlx-community/gemma-3-1b-it-4bit (~0.8 GB)
- Target: mlx-community/gemma-3-12b-it-4bit (~7.2 GB)

Evaluates:
1. Baseline Greedy 12B decode speed (tok/s).
2. Speculative decoding with K in {1, 2, 3}.
3. Acceptance rate and wall speedup.
4. Terminal Gate: 100% Exact Token Identity against Greedy 12B!
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(IRONMULE))

import mlx.core as mx
from mlx_lm import load
from _bench import enforce_offline, resolve_local_model_snapshot
from ironmule.runtime import Knobs, Engine, BASELINE
from friday_serve.model_speculation import SpeculativeDraftEngine

TARGET_MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
DRAFT_MODEL_ID = "mlx-community/gemma-3-1b-it-4bit"

TEST_PROMPTS = [
    {
        "name": "QA Concept",
        "prompt": "Explain why Apple Silicon unified memory reduces memory copy overhead in AI workloads.",
        "max_tokens": 32,
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function to compute the moving average of a list using a sliding window.",
        "max_tokens": 32,
    },
    {
        "name": "Math Reasoning",
        "prompt": "Solve this step-by-step: A shop offers 20% discount on $60, then 10% tax. What is final cost?",
        "max_tokens": 32,
    },
]


def main():
    enforce_offline()
    print("================================================================================")
    print("🚀 REAL HARDWARE BENCHMARK: DRAFT-MODEL SPECULATIVE DECODING (1B -> 12B)")
    print("================================================================================")

    # 1. Load Models into UMA
    print(f"Loading Target Model: {TARGET_MODEL_ID}...")
    t_target = resolve_local_model_snapshot(TARGET_MODEL_ID)
    target_model, tokenizer = load(str(t_target.path))

    print(f"Loading Draft Model:  {DRAFT_MODEL_ID}...")
    t_draft = resolve_local_model_snapshot(DRAFT_MODEL_ID)
    draft_model, _ = load(str(t_draft.path))

    print(f"Both models loaded into Unified Memory. Total VRAM: {mx.get_peak_memory() / (1024**2):.1f} MB\n")

    eos_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))
    target_engine = Engine(target_model, tokenizer, Knobs(head_skip_prefill=True, compiled_fixed_cache=False, readback_every=1))

    # 2. Run Greedy 12B Baseline Reference
    print("--- [STEP 1/2] Measuring Greedy 12B Baseline Truth ---")
    baseline_results = {}
    for task in TEST_PROMPTS:
        p_ids = tokenizer.encode(task["prompt"])
        # Warmup
        _ = target_engine.generate(p_ids, 8, eos_ids)
        mx.eval()
        mx.synchronize()

        t0 = time.perf_counter_ns()
        res = target_engine.generate(p_ids, task["max_tokens"], eos_ids)
        mx.eval()
        mx.synchronize()
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        dec_ms = res["decode_ns"] / 1e6
        cnt = len(res["physical_tokens"])
        tps = (cnt / (dec_ms / 1000.0)) if dec_ms > 0 else 0.0

        baseline_results[task["name"]] = {
            "tokens": res["physical_tokens"],
            "wall_ms": wall_ms,
            "decode_ms": dec_ms,
            "tps": tps,
        }
        print(f"  ✓ {task['name']:<18}: Wall={wall_ms:6.1f} ms | Decode={tps:4.1f} tok/s | Tokens: {cnt}")

    # 3. Run Speculative Decoding with K in {1, 2, 3}
    for k_val in (1, 2, 3):
        print(f"\n--- [STEP 2/2] Evaluating Speculative Engine (Lookahead K={k_val}) ---")
        spec_engine = SpeculativeDraftEngine(
            target_model, draft_model, tokenizer,
            target_knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=False, readback_every=1),
            draft_knobs=Knobs(head_skip_prefill=True, compiled_fixed_cache=False, readback_every=1),
            k=k_val,
        )

        all_ident = True
        for task in TEST_PROMPTS:
            p_ids = tokenizer.encode(task["prompt"])
            # Warmup
            _ = spec_engine.generate(p_ids, 8)
            mx.eval()
            mx.synchronize()

            t0 = time.perf_counter_ns()
            res = spec_engine.generate(p_ids, task["max_tokens"])
            mx.eval()
            mx.synchronize()
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            dec_ms = res["decode_ns"] / 1e6
            cnt = len(res["logical_tokens"])
            tps = (cnt / (dec_ms / 1000.0)) if dec_ms > 0 else 0.0

            base = baseline_results[task["name"]]
            base_tokens = base["tokens"][:cnt]
            is_match = (res["logical_tokens"] == base_tokens)
            if not is_match:
                all_ident = False

            tps_gain = ((tps / max(0.1, base["tps"])) - 1.0) * 100.0
            wall_ratio = wall_ms / max(1.0, base["wall_ms"])
            m_str = "MATCH ✅" if is_match else "DIFF ❌"

            print(f"  --> {task['name']:<16}: Wall Ratio={wall_ratio:.4f} | TPS={tps:4.1f} tok/s ({tps_gain:+5.1f}%) | Acc={res['acceptance_rate']*100:.1f}% | Tokens: {m_str}")

        print(f"  ==> Lookahead K={k_val} Overall Token Identity: {'100% OK ✅' if all_ident else 'FAILED ❌'}")


if __name__ == "__main__":
    main()
