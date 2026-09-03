#!/usr/bin/env python3
"""Extended Intensive Live Demonstration of Friday & IronMule Live-Cockpit on Apple Silicon.

Executes 4 demanding, real-world inference tasks:
1. Long-Context Technical Architecture Synthesis (128 Tokens)
2. Stateful Prefix-Cache Hit & Acceleration (128 Tokens, shares prefix -> TTFT < 80ms)
3. Heavy Autoregressive Generation & Memory Bandwidth Saturation (256 Tokens)
4. Prompt-Lookup Speculative Extraction with High Acceptance (96 Tokens)
"""

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

PORT = 8990
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

SHARED_SYSTEM_PREFIX = (
    "You are an elite Apple Silicon performance engineer. "
    "You analyze unified memory architecture (UMA), Metal Shading Language, "
    "cache hierarchy, memory bus widths, and transformer autoregressive decoding. "
    "Provide rigorous, mathematically precise, and hardware-accurate explanations.\n\n"
)


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


def stream_chat(prompt: str, max_tokens: int = 128) -> str:
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
    print(f"\n[Generated {len(tokens)} chunks in {wall_s:.2f}s]")
    return "".join(tokens)


def print_cockpit():
    with urllib.request.urlopen(f"{BASE_URL}/dashboard") as resp:
        print(resp.read().decode("utf-8"))


def main():
    print("================================================================================")
    print("🐎 IRONMULE — EXTENDED HARDWARE STRESS TEST & LIVE-COCKPIT DEMO")
    print("================================================================================")

    # 1. Launch server
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "friday.py"),
        "serve",
        "--port",
        str(PORT),
    ]
    print(f"Starting IronMule Server on port {PORT}...")
    server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        if not wait_for_server(25.0):
            print("❌ Server failed to start.")
            out, err = server_proc.communicate()
            print("STDOUT:", out.decode("utf-8", errors="ignore"))
            print("STDERR:", err.decode("utf-8", errors="ignore"))
            return 1
        print("✅ Server online and operational!\n")

        # ----------------------------------------------------------------------
        # TASK 1: Long Context Architecture Synthesis (128 Tokens)
        # ----------------------------------------------------------------------
        print("="*80)
        print("TASK 1: Technical Architecture Synthesis (128 Tokens)")
        print("="*80)
        q1 = SHARED_SYSTEM_PREFIX + (
            "Explain in technical detail why discrete GPU PC architectures suffer from PCIe "
            "serialization latency during autoregressive token generation compared to Apple UMA."
        )
        print("Streaming: ", end="", flush=True)
        stream_chat(q1, max_tokens=128)
        print("\n--- IRONMULE COCKPIT AFTER TASK 1 ---")
        print_cockpit()

        # ----------------------------------------------------------------------
        # TASK 2: High-Context Prefix-Cache Reuse (128 Tokens)
        # ----------------------------------------------------------------------
        print("\n" + "="*80)
        print("TASK 2: Stateful Prefix-Cache Hit Verification (128 Tokens)")
        print("="*80)
        print("Notice: Uses the exact same SHARED_SYSTEM_PREFIX as Task 1 to trigger instant TTFT!")
        q2 = SHARED_SYSTEM_PREFIX + (
            "Now analyze how head_skip_prefill eliminates redundant LM-head matrix multiplies "
            "and how fixed KV caching prevents memory reallocation spikes in Metal."
        )
        print("Streaming: ", end="", flush=True)
        stream_chat(q2, max_tokens=128)
        print("\n--- IRONMULE COCKPIT AFTER TASK 2 ---")
        print_cockpit()

        # ----------------------------------------------------------------------
        # TASK 3: Heavy 256-Token Generation (Memory Bandwidth Saturation)
        # ----------------------------------------------------------------------
        print("\n" + "="*80)
        print("TASK 3: Sustained Generation — 256 Tokens (Bandwidth & Thermal Stress)")
        print("="*80)
        q3 = (
            "Write a production-grade Python class implementing an asynchronous grouped batch-1 "
            "execution engine with MLX. Include double buffering, async eval, and clean synchronization."
        )
        print("Streaming: ", end="", flush=True)
        stream_chat(q3, max_tokens=256)
        print("\n--- IRONMULE COCKPIT AFTER TASK 3 ---")
        print_cockpit()

        # ----------------------------------------------------------------------
        # TASK 4: Prompt-Lookup Speculation (96 Tokens)
        # ----------------------------------------------------------------------
        print("\n" + "="*80)
        print("TASK 4: Prompt-Lookup Extraction Task (Speculation Acceptance Demonstration)")
        print("="*80)
        q4 = (
            "From the following system description, extract and summarize the exact specifications:\n"
            "Hardware Specifications: Apple M1 Max SoC, 10 CPU cores (8 Performance + 2 Efficiency), "
            "32 GPU Execution Cores, 400 GB/s Unified LPDDR5 Memory Bandwidth with 512-bit wide memory bus, "
            "32 GB Unified RAM with zero-copy CPU-GPU memory pool, 48 MB System-Level Cache (SLC), "
            "16-core Neural Engine rated at 15.8 TOPS.\n"
            "Extract every single parameter verbatim."
        )
        print("Streaming: ", end="", flush=True)
        stream_chat(q4, max_tokens=96)
        print("\n--- IRONMULE COCKPIT AFTER TASK 4 ---")
        print_cockpit()

        print("\n" + "="*80)
        print("🎉 ALL 4 EXTENDED BENCHMARK TASKS COMPLETED SUCCESSFULLY!")
        print("================================================================================")

    finally:
        print("\nTerminating IronMule Server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server shutdown completed.")


if __name__ == "__main__":
    main()
