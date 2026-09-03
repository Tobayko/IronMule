#!/usr/bin/env python3
"""Benchmark Radix-Tree Global Prefix Caching on Apple Silicon M1 Max.

Measures TTFT and Prefill Latency across independent user requests
sharing a common System Prompt prefix (e.g., standard agent instructions).
"""

from __future__ import annotations

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
from ironmule.runtime import Engine, Knobs
from friday_serve.radix_cache import RadixCache

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

SYSTEM_INSTRUCTION = (
    "You are Project Friday, an autonomous Apple Silicon systems runtime agent developed by Google DeepMind. "
    "You operate strictly under empirical evidence, ensuring 100% mathematical token identity, zero mocks, "
    "and hardware-aware optimization across Unified Memory architectures, Metal shaders, and neural runtimes. "
    "Always analyze performance frontiers using Roofline modeling, bandwidth bounds, and empirical benchmarks."
)

USER_QUERIES = [
    "What is the theoretical peak FP16 compute of the Apple M1 Max?",
    "Explain how unified memory eliminates host-to-device PCIe transfer bottlenecks.",
    "Why does 4-bit integer dequantization bottleneck SIMD ALU execution units?",
    "Calculate the memory bandwidth required to decode 100 tokens per second on a 4B model.",
]


def main():
    enforce_offline()
    print("================================================================================")
    print("🌲 BENCHMARK: RADIX-TREE GLOBAL PREFIX CACHING (M1 Max)")
    print("================================================================================")

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    knobs = Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8)
    engine = Engine(model, tokenizer, knobs)

    radix = RadixCache(max_tokens=8192)

    # Encode system prefix
    sys_ids = tokenizer.encode(SYSTEM_INSTRUCTION)
    print(f"Shared System Prompt Length: {len(sys_ids)} tokens\n")

    # Warmup
    _ = engine.generate(sys_ids[:32], 4, (1,))
    mx.eval()
    mx.synchronize()

    # --- Phase 1: Pre-cache the System Prompt into Radix Tree ---
    print("1. Ingesting shared system prompt into Radix-Tree KV-Cache...")
    t0 = time.perf_counter_ns()
    cap = engine._capacity(len(sys_ids), 32)
    state, tok = engine._prefill(sys_ids, cap)
    from ironmule.runtime import _leaves
    mx.eval(*[leaf for leaf in _leaves(state)])
    mx.synchronize()
    ingest_ms = (time.perf_counter_ns() - t0) / 1e6
    radix.insert(sys_ids, state["layers"])
    print(f"   ✓ System prompt cached in {ingest_ms:.1f} ms. Node registered in Radix Trie.\n")

    # --- Phase 2: Process independent queries sharing the prefix ---
    print(f"{'Request':<20} | {'Mode':<18} | {'TTFT':<10} | {'Tokens':<8} | {'Speedup':<10}")
    print("-" * 75)

    cold_ttfts = []
    cached_ttfts = []

    for i, q in enumerate(USER_QUERIES):
        full_prompt = f"{SYSTEM_INSTRUCTION}\n\nUser Question: {q}\nAnswer:"
        prompt_ids = tokenizer.encode(full_prompt)

        # A. Cold run without Radix Cache
        t0 = time.perf_counter_ns()
        cap_cold = engine._capacity(len(prompt_ids), 16)
        state_cold, _ = engine._prefill(prompt_ids, cap_cold)
        mx.eval(*[leaf for leaf in _leaves(state_cold)])
        mx.synchronize()
        cold_ttft_ms = (time.perf_counter_ns() - t0) / 1e6
        cold_ttfts.append(cold_ttft_ms)

        # B. Warm run with Radix Cache
        t0 = time.perf_counter_ns()
        match_len, cached_state, _ = radix.match_prefix(prompt_ids)
        assert match_len == len(sys_ids)

        suffix = prompt_ids[match_len:]
        cap_warm = engine._capacity(len(prompt_ids), 16)
        
        # Build state from cached layers at offset match_len
        warm_state = {"position": {"offset": mx.array(match_len, dtype=mx.int32)}, "layers": cached_state}
        state_warm, hidden = engine._feed(warm_state, suffix, cap_warm)
        
        from ironmule.runtime import _project, _leaves
        logits = _project(engine.model, hidden[:, -1:, :] if engine.knobs.head_skip_prefill else hidden)
        token_warm = mx.argmax(logits[:, -1, :], axis=-1).reshape((1, 1))
        mx.eval(token_warm, *_leaves(state_warm))
        mx.synchronize()
        cached_ttft_ms = (time.perf_counter_ns() - t0) / 1e6
        cached_ttfts.append(cached_ttft_ms)

        speedup = cold_ttft_ms / max(0.1, cached_ttft_ms)
        print(
            f"Query #{i+1:<14} | "
            f"Cold: {cold_ttft_ms:5.1f} ms   | "
            f"Radix: {cached_ttft_ms:4.1f} ms | "
            f"{len(prompt_ids):<8} | "
            f"{speedup:5.1f}x faster"
        )

    avg_cold = sum(cold_ttfts) / len(cold_ttfts)
    avg_cached = sum(cached_ttfts) / len(cached_ttfts)
    overall_speedup = avg_cold / max(0.1, avg_cached)

    print("\n" + "=" * 75)
    print(f"🚀 RADIX CACHE ERGEBNIS: {overall_speedup:.1f}x durchschnittliche TTFT-Beschleunigung!")
    print(f"   Kalter Prefill: {avg_cold:.1f} ms -> Radix-Treffer: {avg_cached:.1f} ms")
    print(f"   Radix Trie Stats: {radix.hits} Hits, {radix.tokens_saved} Tokens eingespart.")
    print("================================================================================")


if __name__ == "__main__":
    main()
