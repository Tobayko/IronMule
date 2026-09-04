#!/usr/bin/env python3
"""Benchmark Multi-Turn Conversation & System Prompt Prefix Caching on Apple Silicon M1 Max.

Measures TTFT (Time To First Token) across multi-turn chat interactions:
- Turn 1: Long System Prompt (400 tokens) + User Question 1 (20 tokens)
- Turn 2: Conversation Context (450 tokens) + User Follow-Up Question 2 (20 tokens)

Compares:
- Arm A: Uncached Baseline (recomputes the entire 470 tokens from scratch)
- Arm B: Stateful Prefix Cache (reuses the 450 tokens from Turn 1, prefills only 20 tokens)

Verifies:
1. Exact TTFT reduction in milliseconds and percentage.
2. 100% Token Identity of Turn 2 answer between uncached and cached.
"""

from __future__ import annotations

import gc
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
from ironmule.runtime import Knobs, Engine, PrefixCache

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

SYSTEM_PROMPT = """You are Project Friday, an expert hardware-aware AI systems engineer specialized in Apple Silicon,
Metal Performance Shaders, unified memory architectures, Roofline modeling, kernel dispatch fusion, and low-latency
LLM inference. You always provide highly technical, mathematically rigorous, and concrete answers.
Here is your architecture guide:
1. Apple Silicon M1 Max has 32 GPU cores, 400 GB/s Unified Memory bandwidth, and 10 CPU cores.
2. Memory copy overhead between CPU and GPU is zero due to unified memory addressing.
3. Metal shaders compile via MLX JIT to native Apple GPU machine instructions.
4. KV-cache management must avoid re-allocation during autoregressive decoding.
5. In causal self-attention, key and value projections are cached per layer across all heads.
6. Prefill is compute-bound for long prompts, whereas single-sequence decode is memory-bandwidth-bound.
7. Batching increases decode arithmetic intensity by reusing model weights across multiple queries.
8. Speculative decoding verifies draft tokens in a single parallel verification forward pass.
9. Prefix caching allows bit-exact reuse of static prompt prefixes across conversation turns.
10. All benchmarks must use paired measurements with median and spread reporting.
"""

USER_TURN_1 = "Explain the difference in arithmetic intensity between the prefill phase and the autoregressive decode phase."
USER_TURN_2 = "Given that decode is memory-bound, calculate the theoretical maximum decode tokens per second for a 4-billion parameter 4-bit model on 400 GB/s bandwidth."


def main():
    enforce_offline()
    print("================================================================================")
    print(f"🚀 MULTI-TURN PREFIX CACHING BENCHMARK: {MODEL_ID}")
    print("================================================================================")

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    eos_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

    # Encode multi-turn conversation
    chat_turn1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TURN_1},
    ]
    prompt1_str = tokenizer.apply_chat_template(chat_turn1, add_generation_prompt=True, tokenize=False)
    prompt1_ids = tokenizer.encode(prompt1_str)

    # Encode system prompt alone for prefix boundary
    sys_prompt_str = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}], add_generation_prompt=False, tokenize=False
    )
    sys_prefix_ids = tokenizer.encode(sys_prompt_str)

    print(f"System Prompt Length:    {len(sys_prefix_ids)} tokens")
    print(f"Turn 1 Total Prompt Len: {len(prompt1_ids)} tokens")

    knobs = Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8)
    engine_uncached = Engine(model, tokenizer, knobs)

    # Turn 1 Generation (produce assistant response)
    print("\n--- [TURN 1] Generating Initial Assistant Response ---")
    res1 = engine_uncached.generate(prompt1_ids, 48, eos_ids)
    mx.eval()
    mx.synchronize()
    assistant_turn1_text = tokenizer.decode([t for t in res1["logical_tokens"] if t not in eos_ids])
    print(f"  ✓ Turn 1 Generated {len(res1['logical_tokens'])} tokens (TTFT: {res1['prefill_ns']/1e6:.1f} ms)")

    # Build Turn 2 Prompt (Conversation history + User Question 2)
    chat_turn2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TURN_1},
        {"role": "assistant", "content": assistant_turn1_text},
        {"role": "user", "content": USER_TURN_2},
    ]
    prompt2_str = tokenizer.apply_chat_template(chat_turn2, add_generation_prompt=True, tokenize=False)
    prompt2_ids = tokenizer.encode(prompt2_str)

    # Reusable Prefix is everything up to the new User message
    prefix_chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TURN_1},
        {"role": "assistant", "content": assistant_turn1_text},
    ]
    prefix_history_str = tokenizer.apply_chat_template(prefix_chat_history, add_generation_prompt=False, tokenize=False)
    history_prefix_ids = tokenizer.encode(prefix_history_str)

    print(f"\nTurn 2 Total Prompt Len: {len(prompt2_ids)} tokens")
    print(f"Turn 2 Reusable Prefix:  {len(history_prefix_ids)} tokens ({len(history_prefix_ids)/len(prompt2_ids)*100:.1f}% of prompt)")
    print(f"Turn 2 New Delta Tokens: {len(prompt2_ids) - len(history_prefix_ids)} tokens")

    # -------------------------------------------------------------------------
    # Arm A: Turn 2 Uncached Baseline
    # -------------------------------------------------------------------------
    print("\n--- [ARM A] Turn 2 Uncached (Prefills entire prompt from scratch) ---")
    uncached_ttfts = []
    uncached_tokens = []
    for _ in range(3):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        res_a = engine_uncached.generate(prompt2_ids, 32, eos_ids)
        mx.eval()
        mx.synchronize()
        ttft_ms = res_a["prefill_ns"] / 1e6
        uncached_ttfts.append(ttft_ms)
        uncached_tokens = res_a["logical_tokens"]

    med_ttft_a = sorted(uncached_ttfts)[len(uncached_ttfts)//2]
    print(f"  ✓ Uncached TTFT: {med_ttft_a:.2f} ms | First Token Latency")

    # -------------------------------------------------------------------------
    # Arm B: Turn 2 With Prefix Cache
    # -------------------------------------------------------------------------
    print("\n--- [ARM B] Turn 2 With Prefix Cache (Reuses Conversation History) ---")
    engine_cached = Engine(model, tokenizer, knobs)
    # Configure prefix cache with conversation history
    prefix_cache = PrefixCache(history_prefix_ids)
    engine_cached.prefix_cache = prefix_cache

    # Prime cache with Turn 1 / history
    _ = engine_cached.generate(history_prefix_ids, 1, eos_ids)
    mx.eval()
    mx.synchronize()
    print(f"  ✓ History Prefix cached into Unified Memory ({len(history_prefix_ids)} tokens)")

    cached_ttfts = []
    cached_tokens = []
    for _ in range(3):
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        res_b = engine_cached.generate(prompt2_ids, 32, eos_ids)
        mx.eval()
        mx.synchronize()
        ttft_ms = res_b["prefill_ns"] / 1e6
        cached_ttfts.append(ttft_ms)
        cached_tokens = res_b["logical_tokens"]

    med_ttft_b = sorted(cached_ttfts)[len(cached_ttfts)//2]
    print(f"  ✓ Prefix-Cached TTFT: {med_ttft_b:.2f} ms (Prefix Cache Hits: {prefix_cache.hits})")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("📊 PREFIX CACHING VERIFICATION & SPEEDUP SUMMARY")
    print("================================================================================")
    ttft_speedup = ((med_ttft_a - med_ttft_b) / med_ttft_a) * 100.0
    factor = med_ttft_a / max(0.1, med_ttft_b)

    print(f"Uncached Turn 2 TTFT:       {med_ttft_a:6.2f} ms")
    print(f"Prefix-Cached Turn 2 TTFT:   {med_ttft_b:6.2f} ms")
    print(f"TTFT Reduction:             {ttft_speedup:+6.2f}% ({factor:.1f}x faster TTFT!)")

    token_match = (uncached_tokens == cached_tokens)
    print(f"Turn 2 Output Token Match:  {'100% IDENTICAL ✅' if token_match else 'DIFF ❌'}")
    print("================================================================================")


if __name__ == "__main__":
    main()
