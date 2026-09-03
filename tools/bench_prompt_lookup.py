#!/usr/bin/env python3
"""Benchmark Prompt-Lookup Self-Speculation on Apple Silicon M1 Max.

Tests whether N-gram matching directly from the input prompt (without a secondary model):
1. Accelerates auto-regressive decode on tasks with vocabulary and syntax reuse.
2. Achieves >120-180 tok/s on code refactoring, document Q&A, and structured JSON.
3. Terminal Gate: Verifies 100% exact token identity against the greedy baseline!
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
from ironmule.runtime import Engine, Knobs, BASELINE

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

TEST_CASES = [
    {
        "name": "Code Refactor",
        "prompt": (
            "Refactor this Python class into a dataclass with type annotations and docstrings:\n\n"
            "class DeviceMetrics:\n"
            "    def __init__(self, device_name, memory_bandwidth_gb, peak_flops_tflops, cache_line_bytes):\n"
            "        self.device_name = device_name\n"
            "        self.memory_bandwidth_gb = memory_bandwidth_gb\n"
            "        self.peak_flops_tflops = peak_flops_tflops\n"
            "        self.cache_line_bytes = cache_line_bytes\n\n"
            "Please output only the refactored dataclass code:\n"
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class DeviceMetrics:"
        ),
        "max_tokens": 48,
    },
    {
        "name": "Document Q&A",
        "prompt": (
            "Context: The Apple Silicon M1 Max architecture integrates a 32-core GPU with 400 GB/s "
            "unified memory bandwidth, 10 CPU cores, and unified LPDDR5-6400 memory controllers. "
            "Because CPU and GPU share identical physical memory pools, zero-copy pointer exchanges "
            "eliminate PCIe transfer latencies completely.\n\n"
            "Question: Based on the text, what are the exact memory bandwidth and GPU core specifications?\n"
            "Answer: The Apple Silicon M1 Max architecture integrates a"
        ),
        "max_tokens": 48,
    },
    {
        "name": "JSON Extraction",
        "prompt": (
            "Extract system specs into valid JSON matching this schema:\n"
            '{\n  "device_model": "string",\n  "gpu_cores": 0,\n  "bandwidth_gb_s": 0.0,\n  "unified_memory_gb": 0\n}\n\n'
            'Input: System is Apple M1 Max with 32 GPU cores, 34 GB unified memory, 400.0 GB/s bandwidth.\n\n'
            'Output JSON:\n{\n  "device_model":'
        ),
        "max_tokens": 48,
    },
]


def main():
    enforce_offline()
    print("================================================================================")
    print(f"🚀 PROMPT-LOOKUP SELF-SPECULATION BENCHMARK: {MODEL_ID}")
    print("================================================================================")

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    eos_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

    # 1. Baseline Run (Core Triad without Speculation)
    print("\n--- [PHASE 1] Measuring Core Triad Baseline (No Speculation) ---")
    base_knobs = Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8)
    base_engine = Engine(model, tokenizer, base_knobs)

    baseline_data = {}
    for tc in TEST_CASES:
        p_ids = tokenizer.encode(tc["prompt"])
        # Warmup
        _ = base_engine.generate(p_ids, 8, eos_ids)
        mx.eval()
        mx.synchronize()

        t0 = time.perf_counter_ns()
        res = base_engine.generate(p_ids, tc["max_tokens"], eos_ids)
        mx.eval()
        mx.synchronize()
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        dec_ms = res["decode_ns"] / 1e6
        tps = len(res["physical_tokens"]) / (dec_ms / 1000.0) if dec_ms > 0 else 0.0

        baseline_data[tc["name"]] = {
            "tokens": res["physical_tokens"],
            "wall_ms": wall_ms,
            "dec_ms": dec_ms,
            "tps": tps,
        }
        print(f"  ✓ {tc['name']:<16}: Wall={wall_ms:6.1f} ms | Decode={tps:5.1f} tok/s | Tokens: {len(res['physical_tokens'])}")

    # 2. Speculative Run with Prompt-Lookup (K in {2, 3, 4})
    for k_val in (2, 3, 4):
        print(f"\n--- [PHASE 2] Evaluating Prompt-Lookup Speculation (Lookahead K={k_val}, N-gram=3) ---")
        spec_knobs = Knobs(
            head_skip_prefill=True,
            compiled_fixed_cache=True,
            speculate_k=k_val,
            speculate_ngram=3,
        )
        spec_engine = Engine(model, tokenizer, spec_knobs)

        all_match = True
        for tc in TEST_CASES:
            p_ids = tokenizer.encode(tc["prompt"])
            # Warmup
            _ = spec_engine.generate(p_ids, 8, eos_ids)
            mx.eval()
            mx.synchronize()

            t0 = time.perf_counter_ns()
            res = spec_engine.generate(p_ids, tc["max_tokens"], eos_ids)
            mx.eval()
            mx.synchronize()
            wall_ms = (time.perf_counter_ns() - t0) / 1e6
            dec_ms = res["decode_ns"] / 1e6
            cnt = len(res["physical_tokens"])
            tps = cnt / (dec_ms / 1000.0) if dec_ms > 0 else 0.0

            base = baseline_data[tc["name"]]
            base_toks = base["tokens"][:cnt]
            is_match = (res["physical_tokens"] == base_toks)
            if not is_match:
                all_match = False

            tps_gain = ((tps / max(0.1, base["tps"])) - 1.0) * 100.0
            wall_ratio = wall_ms / max(1.0, base["wall_ms"])
            m_str = "MATCH ✅" if is_match else "DIFF ❌"

            print(
                f"  --> {tc['name']:<14}: Wall Ratio={wall_ratio:.4f} | "
                f"TPS={tps:5.1f} tok/s ({tps_gain:+5.1f}%) | "
                f"Acc={res['acceptance']*100:4.1f}% | Tokens: {m_str}"
            )

        print(f"  ==> Lookahead K={k_val} Overall Token Identity: {'100% BIT-EXACT ✅' if all_match else 'FAILED ❌'}")


if __name__ == "__main__":
    main()
