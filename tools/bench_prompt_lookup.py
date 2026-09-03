#!/usr/bin/env python3
"""Benchmark Prompt-Lookup Self-Speculation on Apple Silicon M1 Max.

Tests whether N-gram matching directly from the input prompt (without a secondary model):
1. Accelerates auto-regressive decode on tasks with vocabulary and syntax reuse.
2. Achieves >120-180 tok/s on code refactoring, document Q&A, and structured JSON.
3. Terminal Gate: Verifies 100% exact token identity against the greedy baseline!
"""

from __future__ import annotations

import statistics
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
from _bench import harness_preconditions, resolve_local_model_snapshot
from ironmule.runtime import Engine, Knobs

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


def _wall_ms(engine, p_ids, max_tokens, eos_ids, guard):
    t0 = time.perf_counter_ns()
    res = engine.generate(p_ids, max_tokens, eos_ids)
    mx.eval()
    mx.synchronize()
    ns = time.perf_counter_ns() - t0
    guard.record_gpu(ns / 1e9)
    return ns / 1e6, res


def main():
    guard = harness_preconditions()
    print("================================================================================")
    print(f"🚀 PROMPT-LOOKUP SELF-SPECULATION BENCHMARK: {MODEL_ID}")
    print("================================================================================")

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    eos_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

    # Delivery-path knob set. The speculative arm differs by speculate_k ONLY —
    # readback_every stays 8 so any delta is attributable to speculation.
    base_knobs = Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8)
    base_engine = Engine(model, tokenizer, base_knobs)

    for k_val in (2, 3, 4):
        print(f"\n--- Prompt-Lookup Speculation K={k_val}, N-gram=3 (paired vs baseline) ---")
        spec_knobs = Knobs(
            head_skip_prefill=True,
            compiled_fixed_cache=True,
            readback_every=8,
            speculate_k=k_val,
            speculate_ngram=3,
        )
        spec_engine = Engine(model, tokenizer, spec_knobs)

        all_match = True
        for tc in TEST_CASES:
            p_ids = tokenizer.encode(tc["prompt"])
            base_engine.generate(p_ids, 8, eos_ids); mx.synchronize()
            spec_engine.generate(p_ids, 8, eos_ids); mx.synchronize()

            ratios, acc = [], 0.0
            base_tokens = spec_tokens = None
            for rep in range(6):
                if rep % 2 == 0:
                    b_ms, b_res = _wall_ms(base_engine, p_ids, tc["max_tokens"], eos_ids, guard)
                    s_ms, s_res = _wall_ms(spec_engine, p_ids, tc["max_tokens"], eos_ids, guard)
                else:
                    s_ms, s_res = _wall_ms(spec_engine, p_ids, tc["max_tokens"], eos_ids, guard)
                    b_ms, b_res = _wall_ms(base_engine, p_ids, tc["max_tokens"], eos_ids, guard)
                ratios.append(s_ms / b_ms)
                acc = s_res.get("acceptance", 0.0)
                base_tokens, spec_tokens = b_res["physical_tokens"], s_res["physical_tokens"]

            # full-sequence identity, equal length required
            is_match = base_tokens == spec_tokens
            all_match &= is_match
            r = statistics.median(ratios)
            print(
                f"  --> {tc['name']:<14}: wall ratio (median of 6) {r:.4f} "
                f"({(1 - r) * 100:+.1f}%) | acc {acc * 100:4.1f}% | "
                f"identity {'MATCH' if is_match else 'DIFF'} "
                f"(base {len(base_tokens)} tok, spec {len(spec_tokens)} tok)"
            )
            guard.required_break()

        print(f"  ==> K={k_val} token identity: {'exact on every case' if all_match else 'BROKEN'}")


if __name__ == "__main__":
    main()
