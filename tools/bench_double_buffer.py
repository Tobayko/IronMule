#!/usr/bin/env python3
"""Benchmark Double-Buffered Metal Execution vs Synchronous Decode Loop.

Evaluates whether pipelining the Metal command buffer dispatch ahead of Python
host processing (token decoding, SSE serialization, socket simulation)
eliminates host bubbles and improves overall token throughput on M1 Max.
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

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
PROMPT = "Explain the fundamental principles of asynchronous GPU command queues on modern unified architectures."


def simulate_network_host_work():
    # Simulate ~0.5ms of Python serialization, JSON encoding, and network socket write
    time.sleep(0.0005)


def bench_sync_loop(engine, prompt_ids, max_tokens, n_reps=10):
    capacity = engine._capacity(len(prompt_ids), max_tokens)
    times = []

    for _ in range(n_reps):
        state, token = engine._prefill(prompt_ids, capacity)
        body = engine._body(capacity, 1)
        curr = token
        t0 = time.perf_counter_ns()

        for _ in range(max_tokens - 1):
            out = body(curr, state)
            picks = engine._picks(out)
            curr, state = picks[:, -1:], out[1]
            mx.eval(curr)
            mx.synchronize()
            _ = int(curr.item())
            simulate_network_host_work()

        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        times.append(elapsed_ms)

    return sorted(times)[len(times) // 2]


def bench_pipelined_loop(engine, prompt_ids, max_tokens, n_reps=10):
    capacity = engine._capacity(len(prompt_ids), max_tokens)
    times = []

    for _ in range(n_reps):
        state, token = engine._prefill(prompt_ids, capacity)
        body = engine._body(capacity, 1)
        curr = token
        t0 = time.perf_counter_ns()

        # Prefill first step ahead of loop
        out = body(curr, state)
        picks = engine._picks(out)
        next_curr, next_state = picks[:, -1:], out[1]
        mx.async_eval(next_curr)

        for _ in range(max_tokens - 1):
            # Evaluate current ready token
            mx.eval(next_curr)
            val = int(next_curr.item())
            curr, state = next_curr, next_state

            # Launch NEXT step on GPU BEFORE doing host work!
            out = body(curr, state)
            picks = engine._picks(out)
            next_curr, next_state = picks[:, -1:], out[1]
            mx.async_eval(next_curr)

            # While GPU is computing NEXT token, CPU executes host work!
            simulate_network_host_work()

        mx.eval(next_curr)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        times.append(elapsed_ms)

    return sorted(times)[len(times) // 2]


def main():
    enforce_offline()
    print("================================================================================")
    print("⚡ BENCHMARK: DOUBLE-BUFFERED ASYNCHRONOUS PIPELINE (M1 Max)")
    print("================================================================================")

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    knobs = Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=1)
    engine = Engine(model, tokenizer, knobs)

    p_ids = tokenizer.encode(PROMPT)
    max_toks = 64

    print(f"Model: {MODEL_ID}")
    print(f"Simulating real streaming I/O overhead: 0.5 ms per token\n")

    # Warmup
    _ = bench_sync_loop(engine, p_ids, 8, n_reps=2)
    _ = bench_pipelined_loop(engine, p_ids, 8, n_reps=2)

    print("Measuring Synchronous Loop (Standard Sequential)...")
    sync_ms = bench_sync_loop(engine, p_ids, max_toks, n_reps=5)
    sync_tps = (max_toks - 1) / (sync_ms / 1000.0)
    print(f"  Synchronous Decode: {sync_ms:6.1f} ms | {sync_tps:5.1f} tok/s")

    print("\nMeasuring Double-Buffered Pipelined Loop (Overlapped GPU/Host)...")
    pipe_ms = bench_pipelined_loop(engine, p_ids, max_toks, n_reps=5)
    pipe_tps = (max_toks - 1) / (pipe_ms / 1000.0)
    print(f"  Pipelined Decode:   {pipe_ms:6.1f} ms | {pipe_tps:5.1f} tok/s")

    gain = ((pipe_tps / sync_tps) - 1.0) * 100.0
    print("\n--------------------------------------------------------------------------------")
    print(f"🚀 Pipelining Speedup: {gain:+5.1f}% throughput ({sync_ms - pipe_ms:.1f} ms eliminated)")
    print("================================================================================")


if __name__ == "__main__":
    main()
