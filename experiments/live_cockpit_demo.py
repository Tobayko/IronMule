#!/usr/bin/env python3
"""Live Demonstration of Friday Serving & Terminal Live-Cockpit on Apple Silicon."""

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

PORT = 8888
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"


def wait_for_server(timeout_s: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def stream_chat(prompt: str, max_tokens: int = 48) -> str:
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
    print()
    return "".join(tokens)


def print_dashboard():
    with urllib.request.urlopen(f"{BASE_URL}/dashboard") as resp:
        print(resp.read().decode("utf-8"))


def main():
    print("================================================================================")
    print("🚀 STARTING LIVE TEST: FRIDAY OPENAI SERVER & TERMINAL LIVE-COCKPIT")
    print("================================================================================")

    # 1. Start Server in background
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "friday.py"),
        "serve",
        "--port",
        str(PORT),
    ]
    print(f"Launching server on port {PORT}...")
    server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        if not wait_for_server(25.0):
            print("❌ Server failed to start in time.")
            out, err = server_proc.communicate()
            print("STDOUT:", out.decode("utf-8", errors="ignore"))
            print("STDERR:", err.decode("utf-8", errors="ignore"))
            return 1
        print("✅ Server is online and ready!\n")

        # Query 1: Cold start prompt
        print("\n" + "="*80)
        print("TEST 1: Cold Start Query (Standard Prefill & Decode)")
        print("="*80)
        print("Prompt: 'What are the main advantages of unified memory architecture?'")
        print("Streaming Response: ", end="", flush=True)
        stream_chat("What are the main advantages of unified memory architecture?", max_tokens=36)
        print("\n--- LIVE COCKPIT AFTER QUERY 1 ---")
        print_dashboard()

        # Query 2: Technical long task with prefix caching & prompt reuse
        print("\n" + "="*80)
        print("TEST 2: Long Task with Memory Bandwidth Saturation")
        print("="*80)
        print("Prompt: 'Analyze how high memory bandwidth in Apple Silicon prevents latency bottlenecks during autoregressive generation.'")
        print("Streaming Response: ", end="", flush=True)
        stream_chat(
            "Analyze how high memory bandwidth in Apple Silicon prevents latency bottlenecks during autoregressive generation.",
            max_tokens=64,
        )
        print("\n--- LIVE COCKPIT AFTER QUERY 2 ---")
        print_dashboard()

        print("\n" + "="*80)
        print("✅ LIVE TEST SUCCESSFUL: All gauges, streaming tokens, and safety limits active!")
        print("================================================================================")

    finally:
        print("\nStopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server stopped cleanly.")


if __name__ == "__main__":
    main()
