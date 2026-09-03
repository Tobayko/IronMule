#!/usr/bin/env python3
"""Live Real-Hardware Demonstration of Friday & IronMule Cockpit on Gemma 12B (Apple M1 Max)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PORT = 8996
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"

SHARED_SYSTEM_PREFIX = (
    "You are a principal Apple Silicon systems architect specializing in "
    "GPU unified memory architecture, cache line contention, and autoregressive LLM decoding. "
    "Be technically precise and concise.\n\n"
)


def wait_for_server(timeout_s: float = 40.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def stream_chat(prompt: str, max_tokens: int = 64) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    tokens = []
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    tokens.append(delta)
    wall_s = time.perf_counter() - t0
    print(f"\n[12B Engine: Generated {len(tokens)} chunks in {wall_s:.2f}s]")
    return "".join(tokens)


def print_cockpit():
    with urllib.request.urlopen(f"{BASE_URL}/dashboard") as resp:
        print(resp.read().decode("utf-8"))


def main():
    print("================================================================================")
    print("🐎 IRONMULE — REAL HARDWARE LIVE DEMO WITH GEMMA 12B (7.19 GB)")
    print("================================================================================")

    # Launch server with Gemma 12B
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "friday.py"),
        "serve",
        "--model",
        MODEL_ID,
        "--port",
        str(PORT),
    ]
    print(f"Loading Gemma 12B into Unified Memory on port {PORT}...")
    server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        if not wait_for_server(35.0):
            print("❌ Server failed to start.")
            out, err = server_proc.communicate()
            print("STDOUT:", out.decode("utf-8", errors="ignore"))
            print("STDERR:", err.decode("utf-8", errors="ignore"))
            return 1
        print("✅ Gemma 12B Server is online and ready in Unified Memory!\n")

        # ----------------------------------------------------------------------
        # TEST 1: Cold Prefill on 12B (64 Tokens)
        # ----------------------------------------------------------------------
        print("="*80)
        print("TEST 1: Cold Start Inference on Gemma 12B (64 Tokens)")
        print("="*80)
        q1 = (
            "Explain how Apple Silicon 512-bit wide memory bus achieves 400 GB/s unified "
            "bandwidth and why this accelerates large language model autoregressive generation."
        )
        print("Streaming: ", end="", flush=True)
        stream_chat(q1, max_tokens=64)
        print("\n--- IRONMULE COCKPIT AFTER TEST 1 (12B COLD PREFILL) ---")
        print_cockpit()

        # ----------------------------------------------------------------------
        # TEST 2: Long-Task Sustained Generation (128 Tokens)
        # ----------------------------------------------------------------------
        print("\n" + "="*80)
        print("TEST 2: Sustained 128-Token Generation on Gemma 12B (Bandwidth Stress)")
        print("="*80)
        q2 = (
            "Compare the latency characteristics of Gemma 12B versus Gemma 4B on Apple Silicon. "
            "Discuss memory footprint (7.2 GB vs 2.6 GB), KV cache head dimension (256), and roofline limits."
        )
        print("Streaming: ", end="", flush=True)
        stream_chat(q2, max_tokens=128)
        print("\n--- IRONMULE COCKPIT AFTER TEST 2 (12B SUSTAINED GENERATION) ---")
        print_cockpit()

        print("\n" + "="*80)
        print("🎉 GEMMA 12B REAL HARDWARE LIVE DEMO COMPLETED SUCCESSFULLY!")
        print("================================================================================")

    finally:
        print("\nStopping Gemma 12B server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Gemma 12B server stopped cleanly.")


if __name__ == "__main__":
    main()
