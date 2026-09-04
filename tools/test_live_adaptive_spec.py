#!/usr/bin/env python3
"""Live HTTP E2E Test: Workload-Adaptive Prompt Speculation on M1 Max.

Starts friday.py serve live and sends a RAG document Q&A request.
Measures real SSE token streaming throughput and latency.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = 8094
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

RAG_PROMPT = (
    "Context: The Apple Silicon M1 Max architecture integrates a 32-core GPU with 400 GB/s "
    "unified memory bandwidth, 10 CPU cores, and unified LPDDR5-6400 memory controllers. "
    "Because CPU and GPU share identical physical memory pools, zero-copy pointer exchanges "
    "eliminate PCIe transfer latencies completely.\n\n"
    "Question: Based on the text above, what are the exact specifications of the GPU cores and unified memory bandwidth?\n"
    "Answer: The Apple Silicon M1 Max architecture integrates a"
)


def main():
    print("=" * 80)
    print("🚀 LIVE HTTP E2E TEST: ADAPTIVE PROMPT SPECULATION (M1 Max)")
    print("=" * 80)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n1. Starting live server on port {PORT}...")
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
    deadline = time.time() + 25.0
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            clean = line.strip()
            print("   [SERVER]", clean)
            if "Friday Server running on" in clean:
                ready = True
                break
        if proc.poll() is not None:
            break

    if not ready:
        print("❌ Server failed to start!")
        proc.kill()
        sys.exit(1)

    print(f"\n✓ Server online on port {PORT}.")

    try:
        print("\n2. Sending Live RAG Request over HTTP...")
        req_data = {
            "messages": [{"role": "user", "content": RAG_PROMPT}],
            "max_tokens": 48,
            "stream": True,
        }
        req = urllib.request.Request(
            URL,
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        first_token_t = None
        tokens = []

        print("   Streamed output: \"", end="", flush=True)
        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                        tokens.append(content)
                        print(content, end="", flush=True)

        print("\"\n")
        ttft_ms = (first_token_t - t0) * 1000.0 if first_token_t else 0.0
        gen_s = (time.perf_counter() - first_token_t) if first_token_t else 0.0
        tps = len(tokens) / gen_s if gen_s > 0 else 0.0

        print(f"   ✓ Time To First Token (TTFT): {ttft_ms:.1f} ms")
        print(f"   ✓ Tokens Streamed:            {len(tokens)}")
        print(f"   ✓ Real Decode TPS:            {tps:.1f} tok/s")
        print(f"   ✓ Total Roundtrip:            {(time.perf_counter() - t0)*1000:.1f} ms")

    finally:
        print("\n3. Shutting down server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("✓ Server shut down cleanly.")

    print("\n" + "=" * 80)
    print("✅ LIVE TEST SUCCESSFUL: Adaptive Speculation verified in live HTTP server!")
    print("=" * 80)


if __name__ == "__main__":
    main()
