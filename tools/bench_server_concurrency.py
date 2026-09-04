#!/usr/bin/env python3
"""Real-Hardware Concurrency Benchmark: Dynamic Server Batching on Apple Silicon M1 Max.

Measures real HTTP/SSE throughput with mlx-community/gemma-3-4b-it-4bit:
- Arm 1: Sequential Execution (4 requests sent one-by-one, concurrency=1)
- Arm 2: Continuous Dynamic Batching (4 requests sent concurrently, concurrency=4)

Evaluates total wall time, aggregate tokens/second, and verifies 100% token identity!
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from _bench import enforce_offline
from friday_serve.cli import load_profile
from friday_serve.http_server import create_server
from friday_serve.ironmule_backend import IronMuleBackend
from friday_serve.server import Server

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
PROMPTS = [
    "Explain why Apple Silicon unified memory reduces memory copy overhead in AI workloads.",
    "Write a Python function to compute the moving average of a list using a sliding window.",
    "Solve this step-by-step: A shop offers 20% discount on $60, then 10% tax. What is final cost?",
    "Summarize why speculative decoding requires exact token verification in greedy mode.",
]
MAX_TOKENS = 32


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def send_request(port: int, prompt: str, max_tokens: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_s = time.perf_counter() - t0
    choice = body["choices"][0]
    content = choice["message"]["content"]
    usage = body.get("usage", {})
    gen_tokens = usage.get("completion_tokens", 0)
    return {
        "content": content,
        "tokens": gen_tokens,
        "wall_s": wall_s,
    }


def main():
    enforce_offline()
    print("================================================================================")
    print(f"🚀 REAL HARDWARE CONCURRENCY BENCHMARK: DYNAMIC BATCHING ({MODEL_ID})")
    print("================================================================================")

    db_path = PROJECT_ROOT / ".friday-data" / "device-profile.sqlite3"
    profile = load_profile(db_path)
    print(f"Loaded certified device profile: {getattr(profile, 'profile_id', 'unknown')}")

    print("Loading model into Unified Memory...")
    backend = IronMuleBackend.load(MODEL_ID)
    server_core = Server(backend, profile)
    print("Model loaded successfully.\n")

    # -------------------------------------------------------------------------
    # Arm 1: Sequential Execution (concurrency=1)
    # -------------------------------------------------------------------------
    print("--- [ARM 1] Measuring Sequential Single-Flight Execution ---")
    port_seq = get_free_port()
    srv_seq = create_server(server_core, host="127.0.0.1", port=port_seq, max_concurrency=1)
    t_seq_th = threading.Thread(target=srv_seq.serve_forever, daemon=True)
    t_seq_th.start()
    time.sleep(0.2)

    # Warmup
    _ = send_request(port_seq, "Warmup ping", 8)

    t0_seq = time.perf_counter()
    seq_results = []
    for prompt in PROMPTS:
        res = send_request(port_seq, prompt, MAX_TOKENS)
        seq_results.append(res)
    total_seq_s = time.perf_counter() - t0_seq

    total_seq_toks = sum(r["tokens"] for r in seq_results)
    seq_tps = total_seq_toks / total_seq_s
    print(f"  ✓ Sequential Total Wall: {total_seq_s * 1000:.1f} ms | Tokens: {total_seq_toks} | Agg TPS: {seq_tps:.2f} tok/s")
    srv_seq.shutdown()
    srv_seq.server_close()

    # -------------------------------------------------------------------------
    # Arm 2: Concurrent Dynamic Batching (concurrency=4)
    # -------------------------------------------------------------------------
    print("\n--- [ARM 2] Measuring Concurrent Dynamic Batching (W=4) ---")
    port_batch = get_free_port()
    srv_batch = create_server(server_core, host="127.0.0.1", port=port_batch, max_concurrency=4)
    t_batch_th = threading.Thread(target=srv_batch.serve_forever, daemon=True)
    t_batch_th.start()
    time.sleep(0.2)

    # Warmup
    _ = send_request(port_batch, "Warmup ping", 8)

    t0_batch = time.perf_counter()
    batch_results = [None] * len(PROMPTS)
    threads = []

    def client_worker(idx: int, p: str):
        batch_results[idx] = send_request(port_batch, p, MAX_TOKENS)

    for idx, p in enumerate(PROMPTS):
        t = threading.Thread(target=client_worker, args=(idx, p))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_batch_s = time.perf_counter() - t0_batch
    total_batch_toks = sum(r["tokens"] for r in batch_results)
    batch_tps = total_batch_toks / total_batch_s
    print(f"  ✓ Concurrent Total Wall: {total_batch_s * 1000:.1f} ms | Tokens: {total_batch_toks} | Agg TPS: {batch_tps:.2f} tok/s")
    srv_batch.shutdown()
    srv_batch.server_close()

    # -------------------------------------------------------------------------
    # Evaluation & Verification
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("📊 DYNAMIC BATCHING COMPARISON & TOKEN VERIFICATION")
    print("================================================================================")
    speedup = ((batch_tps / seq_tps) - 1.0) * 100.0
    wall_reduction = ((total_seq_s - total_batch_s) / total_seq_s) * 100.0

    print(f"Sequential Duration: {total_seq_s * 1000:7.1f} ms | {seq_tps:5.1f} tok/s")
    print(f"Batch-4 Duration:    {total_batch_s * 1000:7.1f} ms | {batch_tps:5.1f} tok/s")
    print(f"Wall-Time Reduction: {wall_reduction:+6.2f}%")
    print(f"Aggregate TPS Gain:  {speedup:+6.2f}%\n")

    all_match = True
    for i in range(len(PROMPTS)):
        match = (seq_results[i]["content"] == batch_results[i]["content"])
        if not match:
            all_match = False
        m_str = "MATCH ✅" if match else "DIFF ❌"
        print(f"Request {i+1} [{m_str}]: Seq {seq_results[i]['tokens']} toks vs Batch {batch_results[i]['tokens']} toks")

    print("--------------------------------------------------------------------------------")
    print(f"Terminal Result: 100% Token Identity = {all_match}")
    print("================================================================================")


if __name__ == "__main__":
    main()
