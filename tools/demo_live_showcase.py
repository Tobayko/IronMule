#!/usr/bin/env python3
"""Interactive Terminal Showcase: Live Model Streaming & Cockpit Dashboard.

Demonstrates:
1. Live startup of Friday Server on Apple Silicon M1 Max with Dual-Model Co-Residency.
2. Live streaming generation from Gemma 1B (Ultra-fast tier).
3. Live streaming generation from Gemma 4B (Reasoning tier).
4. Fetching and displaying the live Terminal Cockpit Dashboard (ANSI Box-Drawing,
   UMA Bandwidth Gauge, TTFT Tachometer, and Ring-Buffer Stream History).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = 8099
BASE_URL = f"http://127.0.0.1:{PORT}"


def stream_query(model_name: str, prompt: str, max_tokens: int = 32):
    print(f"\n💬 Request -> Model: \033[1;36m{model_name}\033[0m")
    print(f"   Prompt: \"\033[38;5;245m{prompt}\033[0m\"")
    print("   Response: \"", end="", flush=True)

    req_data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(req_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_t = None
    toks = []

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            l = line.decode("utf-8").strip()
            if l == "data: [DONE]":
                break
            if l.startswith("data: "):
                chunk = json.loads(l[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    if first_t is None:
                        first_t = time.perf_counter()
                    toks.append(delta)
                    print(f"\033[1;32m{delta}\033[0m", end="", flush=True)

    print("\"")
    ttft_ms = (first_t - t0) * 1000.0 if first_t else 0.0
    dur_sec = (time.perf_counter() - first_t) if first_t else 0.0
    tps = max_tokens / dur_sec if dur_sec > 0 else 0.0
    print(f"   ⚡ TTFT: \033[1;33m{ttft_ms:.1f} ms\033[0m | Decode: \033[1;32m{tps:.1f} tok/s\033[0m | Tokens: {max_tokens}")


def main():
    print("\033[1;35m" + "=" * 80 + "\033[0m")
    print("\033[1;35m🌟 PROJECT FRIDAY: LIVE HARDWARE SHOWCASE & TERMINAL DASHBOARD\033[0m")
    print("\033[1;35m" + "=" * 80 + "\033[0m")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print("\n1. Starting Friday Server with Dual-Model Co-Residency on port 8099...")
    cmd = [
        sys.executable,
        "tools/friday.py",
        "serve",
        "--port",
        str(PORT),
        "--no-interactive",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    ready = False
    deadline = time.time() + 35.0
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            clean = line.strip()
            print("   \033[38;5;244m[SERVER]\033[0m", clean)
            if "Friday Server running on" in clean:
                ready = True
                break
        if proc.poll() is not None:
            break

    if not ready:
        print("❌ Server failed to start!")
        proc.kill()
        sys.exit(1)

    try:
        # 2. Query Available Models
        with urllib.request.urlopen(f"{BASE_URL}/v1/models") as resp:
            catalog = json.loads(resp.read().decode("utf-8"))
            print(f"\n✓ Models in Unified Memory: {[m['id'] for m in catalog.get('data', [])]}")

        # 3. Stream from Gemma 1B
        stream_query("gemma-1b", "Why is Apple Silicon Unified Memory fast?", max_tokens=28)

        # 4. Stream from Gemma 4B
        stream_query("gemma-4b", "Explain how memory bandwidth dictates autoregressive LLM decode speed.", max_tokens=32)

        # 5. Fetch and display the Live Terminal Cockpit Dashboard
        print("\n" + "\033[1;34m" + "=" * 80 + "\033[0m")
        print("\033[1;34m📊 LIVE TERMINAL COCKPIT DASHBOARD (Rendered live from http://127.0.0.1:8099/dashboard)\033[0m")
        print("\033[1;34m" + "=" * 80 + "\033[0m\n")

        with urllib.request.urlopen(f"{BASE_URL}/dashboard") as resp:
            dashboard_text = resp.read().decode("utf-8")
            print(dashboard_text)

    finally:
        print("\nShutting down server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("✓ Server cleanly terminated.")


if __name__ == "__main__":
    main()
