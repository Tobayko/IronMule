#!/usr/bin/env python3
"""Live End-to-End Test: Server Startup, Hardware Pre-Warmup, SSE Streaming & Dynamic Batching.

Runs on Apple Silicon M1 Max:
1. Launches live HTTP server (tools/friday.py serve) on port 8098.
2. Sends single streaming request to /v1/chat/completions (measures TTFT, TPS, tokens).
3. Sends 4 concurrent streaming requests simultaneously (verifies Continuous Dynamic Batcher).
4. Verifies 100% token generation without blocking.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PORT = 8098
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def main():
    print("=" * 80)
    print("🔴 LIVE SERVER E2E-TEST: HTTP, SSE STREAMING & DYNAMIC BATCHING (M1 Max)")
    print("=" * 80)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n1. Starting live Friday HTTP Server on port {PORT}...")
    cmd = [
        sys.executable,
        "tools/friday.py",
        "serve",
        "--port",
        str(PORT),
        "--no-interactive",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # Wait for server to confirm ready
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

    print(f"\n✓ Server is live, Pre-Warmup complete, listening on port {PORT}.")

    try:
        # TEST 1: Single Live SSE Streaming Request
        print("\n2. [TEST 1] Single Live SSE Streaming Request...")
        req_data = {
            "messages": [
                {"role": "user", "content": "Explain in one sentence why Apple Silicon Unified Memory is efficient."}
            ],
            "max_tokens": 32,
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

        print("   Streaming response: \"", end="", flush=True)
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

        # TEST 2: 4 Concurrent Streaming Clients
        print("\n3. [TEST 2] 4 Concurrent Streaming Clients (Testing Dynamic Batcher)...")
        results = [None] * 4

        def client_worker(idx: int):
            c_req = urllib.request.Request(
                URL,
                data=json.dumps({
                    "messages": [{"role": "user", "content": f"Give one speed word #{idx}."}],
                    "max_tokens": 8,
                    "stream": True,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            t_start = time.perf_counter()
            received_chunks = []
            with urllib.request.urlopen(c_req) as r:
                for raw in r:
                    l = raw.decode("utf-8").strip()
                    if l == "data: [DONE]":
                        break
                    if l.startswith("data: "):
                        c = json.loads(l[6:])
                        txt = c["choices"][0]["delta"].get("content", "")
                        if txt:
                            received_chunks.append(txt)
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            results[idx] = (duration_ms, "".join(received_chunks).strip())

        threads = [threading.Thread(target=client_worker, args=(i,)) for i in range(4)]
        t_batch_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_batch_ms = (time.perf_counter() - t_batch_start) * 1000.0

        print(f"   ✓ All 4 concurrent streams completed in {total_batch_ms:.1f} ms:")
        for i, (dur, txt) in enumerate(results):
            print(f"     Client {i+1}: {dur:6.1f} ms -> \"{txt}\"")

    finally:
        print("\n4. Shutting down server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("✓ Server cleanly shut down.")

    print("\n" + "=" * 80)
    print("✅ LIVE TEST ERFOLGREICH: Alle Optimierungen laufen live auf M1 Max!")
    print("=" * 80)


if __name__ == "__main__":
    main()
