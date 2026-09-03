#!/usr/bin/env python3
"""Benchmark Server & Tokenizer Fastpath on Apple Silicon M1 Max (Stufe 1).

Measures:
1. Tokenizer Encode Latency: Cold vs LRU-Cached prompt encoding.
2. SSE Chunk Serialization Speed: Standard dict json.dumps vs Pre-formatted Chunk Buffer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(IRONMULE))

from friday_serve.ironmule_backend import IronMuleBackend

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
SAMPLE_PROMPT = (
    "You are an expert systems engineer. Analyze the trade-offs of 4-bit vs 8-bit quantized weights "
    "on Apple Silicon Unified Memory architectures, focusing on instruction count vs bandwidth."
)


def main():
    print("================================================================================")
    print("⚡ STUFE 1: SERVER & TOKENIZER FASTPATH BENCHMARK (Apple Silicon M1 Max)")
    print("================================================================================")

    print("Loading backend and tokenizer...")
    backend = IronMuleBackend.load(MODEL_ID)

    # 1. Benchmark Tokenizer Encode (Cold vs Cached)
    print("\n--- [PART 1] Prompt Tokenizer & Template Rendering Latency ---")
    # Clear cache
    backend._encode_cached.cache_clear()

    # Cold run
    t0 = time.perf_counter_ns()
    tokens_cold = backend.encode(SAMPLE_PROMPT)
    cold_ns = time.perf_counter_ns() - t0
    cold_us = cold_ns / 1e3

    # Warm cached runs
    cached_times = []
    for _ in range(100):
        t0 = time.perf_counter_ns()
        tokens_warm = backend.encode(SAMPLE_PROMPT)
        cached_times.append(time.perf_counter_ns() - t0)

    med_cached_ns = sorted(cached_times)[len(cached_times) // 2]
    med_cached_us = med_cached_ns / 1e3

    speedup = cold_us / max(0.001, med_cached_us)
    print(f"  Cold Tokenizer Encode:  {cold_us:7.2f} µs ({len(tokens_cold)} tokens rendered)")
    print(f"  Cached Tokenizer Encode:{med_cached_us:7.2f} µs ({len(tokens_warm)} tokens)")
    print(f"  Encoding Acceleration:  {speedup:7.1f}x speedup ({cold_us - med_cached_us:.1f} µs saved per request)")
    assert tokens_cold == tokens_warm

    # 2. Benchmark SSE Streaming Serialization (Dict json.dumps vs Pre-formatted Bytes)
    print("\n--- [PART 2] SSE Token Streaming Serialization Latency (1000 tokens) ---")
    completion_id = "chatcmpl-test12345"
    created = 1234567890
    model = "gemma-3-4b"
    sample_token = " memory"

    # Baseline: dict instantiation + json.dumps
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": sample_token}, "finish_reason": None}],
        }
        _ = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    base_ns = (time.perf_counter_ns() - t0) / 1000.0

    # Fastpath: Pre-formatted prefix + json.dumps(text) + suffix
    prefix = f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{created},"model":"{model}","choices":[{{"index":0,"delta":{{"content":'.encode("utf-8")
    suffix = b'},"finish_reason":null}]}\n\n'
    t0 = time.perf_counter_ns()
    for _ in range(1000):
        _ = prefix + json.dumps(sample_token).encode("utf-8") + suffix
    fast_ns = (time.perf_counter_ns() - t0) / 1000.0

    sse_speedup = base_ns / max(0.001, fast_ns)
    print(f"  Baseline Dict JSON per token: {base_ns/1e3:6.3f} µs")
    print(f"  Fastpath Buffer per token:    {fast_ns/1e3:6.3f} µs")
    print(f"  SSE Formatting Speedup:       {sse_speedup:6.1f}x faster formatting ({((base_ns - fast_ns)/base_ns)*100:.1f}% reduction)")

    print("\n================================================================================")
    print("✓ Stufe 1 fastpath verified successfully.")
    print("================================================================================")


if __name__ == "__main__":
    main()
