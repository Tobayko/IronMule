#!/usr/bin/env python3
"""Double-buffered Metal dispatch vs. synchronous decode, paired.

Question: does launching the decode of step t+1 on the GPU before doing the
per-token host work for step t (detokenise + JSON-encode the SSE chunk + write
it to a buffer) hide the host cost behind GPU compute on M1 Max?

The host work here is the real work a streaming server does per readback, not a
``time.sleep``. Both arms run the delivery-path knob set (``readback_every=8``)
and are interleaved AB/BA so warm-up drift cannot favour one.
"""

from __future__ import annotations

import io
import json
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
PROMPT = "Explain the fundamental principles of asynchronous GPU command queues on modern unified architectures."


def host_work(tokenizer, tok: int, sink: io.BytesIO) -> None:
    """Exactly what the SSE server does per emitted token."""
    text = tokenizer.decode([tok])
    chunk = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    )
    sink.write(f"data: {chunk}\n\n".encode("utf-8"))


def bench_sync(engine, tokenizer, prompt_ids, max_tokens) -> float:
    capacity = engine._capacity(len(prompt_ids), max_tokens)
    state, token = engine._prefill(prompt_ids, capacity)
    body = engine._body(capacity, 1)
    curr = token
    sink = io.BytesIO()
    t0 = time.perf_counter_ns()
    for _ in range(max_tokens - 1):
        out = body(curr, state)
        picks = engine._picks(out)
        curr, state = picks[:, -1:], out[1]
        mx.eval(curr)
        host_work(tokenizer, int(curr.item()), sink)
    mx.synchronize()
    return (time.perf_counter_ns() - t0) / 1e6


def bench_pipelined(engine, tokenizer, prompt_ids, max_tokens) -> float:
    capacity = engine._capacity(len(prompt_ids), max_tokens)
    state, token = engine._prefill(prompt_ids, capacity)
    body = engine._body(capacity, 1)
    curr = token
    sink = io.BytesIO()
    t0 = time.perf_counter_ns()

    out = body(curr, state)
    picks = engine._picks(out)
    next_curr, next_state = picks[:, -1:], out[1]
    mx.async_eval(next_curr)

    for _ in range(max_tokens - 1):
        mx.eval(next_curr)
        val = int(next_curr.item())
        curr, state = next_curr, next_state
        # launch the next GPU step before spending CPU on host work
        out = body(curr, state)
        picks = engine._picks(out)
        next_curr, next_state = picks[:, -1:], out[1]
        mx.async_eval(next_curr)
        host_work(tokenizer, val, sink)

    mx.eval(next_curr)
    mx.synchronize()
    return (time.perf_counter_ns() - t0) / 1e6


def main():
    guard = harness_preconditions()
    print("=" * 80)
    print("⚡ DOUBLE-BUFFERED PIPELINE vs SYNCHRONOUS DECODE (paired, M1 Max)")
    print("=" * 80)

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    knobs = Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8)
    engine = Engine(model, tokenizer, knobs)
    p_ids = tokenizer.encode(PROMPT)
    max_toks = 64

    # warm up both paths
    bench_sync(engine, tokenizer, p_ids, 8)
    bench_pipelined(engine, tokenizer, p_ids, 8)

    sync_ms, pipe_ms = [], []
    for i in range(8):
        if i % 2 == 0:
            sync_ms.append(bench_sync(engine, tokenizer, p_ids, max_toks))
            guard.record_gpu(sync_ms[-1] / 1e3)
            pipe_ms.append(bench_pipelined(engine, tokenizer, p_ids, max_toks))
            guard.record_gpu(pipe_ms[-1] / 1e3)
        else:
            pipe_ms.append(bench_pipelined(engine, tokenizer, p_ids, max_toks))
            guard.record_gpu(pipe_ms[-1] / 1e3)
            sync_ms.append(bench_sync(engine, tokenizer, p_ids, max_toks))
            guard.record_gpu(sync_ms[-1] / 1e3)

    s, p = statistics.median(sync_ms), statistics.median(pipe_ms)
    ratio = p / s
    print(f"  Synchronous:  {s:6.1f} ms (median of {len(sync_ms)})")
    print(f"  Pipelined:    {p:6.1f} ms (median of {len(pipe_ms)})")
    print(f"  Ratio p/s:    {ratio:.4f}   ({(1 - ratio) * 100:+.2f}% wall)")
    print("=" * 80)


if __name__ == "__main__":
    main()
