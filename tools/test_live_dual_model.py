#!/usr/bin/env python3
"""Live HTTP E2E Test: Dual-Model Zero-Cold-Start Co-Residency (1B + 4B in RAM).

Tests:
1. Server startup with pre-warmed Gemma 4B and Gemma 1B in Unified Memory.
2. Query /v1/models to verify multi-model catalog.
3. Live SSE streaming from Gemma 1B (>150 tok/s).
4. Live SSE streaming from Gemma 4B (~80-110 tok/s).
5. Simultaneous concurrent requests to BOTH models in Unified Memory!
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PORT = 8096
BASE_URL = f"http://127.0.0.1:{PORT}"


def main():
    print("=" * 80)
    print("🔥 LIVE HTTP E2E TEST: DUAL-MODEL ZERO-COLD-START CO-RESIDENCY (M1 Max)")
    print("=" * 80)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n1. Starting live Friday Server on port {PORT}...")
    cmd = [
        sys.executable,
        "tools/friday.py",
        "serve",
        "--port",
        str(PORT),
        "--no-interactive",
        "--dual-model",  # this test is explicitly about co-residency
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    ready = False
    deadline = time.time() + 35.0
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
        # 1. Query /v1/models
        print("\n2. [CHECK 1] Querying /v1/models...")
        with urllib.request.urlopen(f"{BASE_URL}/v1/models") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_ids = [m["id"] for m in data.get("data", [])]
            print(f"   ✓ Available Co-Resident Models in RAM: {model_ids}")
        assert any("1b" in m.lower() for m in model_ids), f"1B not in catalog: {model_ids}"
        assert any("4b" in m.lower() for m in model_ids), f"4B not in catalog: {model_ids}"

        # 2. Query Gemma 1B (Ultra-fast low-latency tier)
        print("\n3. [CHECK 2] Streaming from Gemma 1B (model='gemma-1b')...")
        req_1b = {
            "model": "gemma-1b",
            "messages": [{"role": "user", "content": "Name three benefits of unified memory."}],
            "max_tokens": 32,
            "stream": True,
        }
        r = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=json.dumps(req_1b).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        first_t = None
        toks_1b = []
        print("   Stream 1B: \"", end="", flush=True)
        total_tokens_1b = 0
        with urllib.request.urlopen(r) as resp:
            for line in resp:
                l = line.decode("utf-8").strip()
                if l == "data: [DONE]":
                    break
                if l.startswith("data: "):
                    c = json.loads(l[6:])
                    txt = c["choices"][0]["delta"].get("content", "")
                    if txt:
                        if first_t is None:
                            first_t = time.perf_counter()
                        toks_1b.append(txt)
                        total_tokens_1b += 1
                        print(txt, end="", flush=True)
        print("\"\n")
        ttft_1b = (first_t - t0) * 1000.0 if first_t else 0.0
        dur_1b = (time.perf_counter() - first_t) if first_t else 0.0
        tps_1b = total_tokens_1b / dur_1b if dur_1b > 0 else 0.0
        print(f"   ✓ 1B TTFT: {ttft_1b:.1f} ms | Decode TPS: {tps_1b:.1f} tok/s | Tokens: {total_tokens_1b}")
        assert total_tokens_1b > 0, "1B produced no tokens"

        # 3. Query Gemma 4B (High-quality reasoning tier)
        print("\n4. [CHECK 3] Streaming from Gemma 4B (model='gemma-4b')...")
        req_4b = {
            "model": "gemma-4b",
            "messages": [{"role": "user", "content": "Name three benefits of unified memory."}],
            "max_tokens": 32,
            "stream": True,
        }
        r = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=json.dumps(req_4b).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        first_t = None
        toks_4b = []
        print("   Stream 4B: \"", end="", flush=True)
        with urllib.request.urlopen(r) as resp:
            for line in resp:
                l = line.decode("utf-8").strip()
                if l == "data: [DONE]":
                    break
                if l.startswith("data: "):
                    c = json.loads(l[6:])
                    txt = c["choices"][0]["delta"].get("content", "")
                    if txt:
                        if first_t is None:
                            first_t = time.perf_counter()
                        toks_4b.append(txt)
                        print(txt, end="", flush=True)
        print("\"\n")
        ttft_4b = (first_t - t0) * 1000.0 if first_t else 0.0
        dur_4b = (time.perf_counter() - first_t) if first_t else 0.0
        tps_4b = len(toks_4b) / dur_4b if dur_4b > 0 else 0.0
        print(f"   ✓ 4B TTFT: {ttft_4b:.1f} ms | Decode TPS: {tps_4b:.1f} tok/s | Tokens: {len(toks_4b)}")
        assert len(toks_4b) > 0, "4B produced no tokens"

        # 4. Simultaneous Concurrent Multi-Model Execution
        print("\n5. [CHECK 4] Concurrent Multi-Model Query: Client 1 -> 1B, Client 2 -> 4B simultaneously...")
        results = {}

        def fetch(m_name):
            req_data = {
                "model": m_name,
                "messages": [{"role": "user", "content": "One word for speed."}],
                "max_tokens": 8,
                "stream": True,
            }
            req_obj = urllib.request.Request(
                f"{BASE_URL}/v1/chat/completions",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            t_s = time.perf_counter()
            received = []
            with urllib.request.urlopen(req_obj) as resp_obj:
                for ln in resp_obj:
                    ln_str = ln.decode("utf-8").strip()
                    if ln_str == "data: [DONE]":
                        break
                    if ln_str.startswith("data: "):
                        chunk_obj = json.loads(ln_str[6:])
                        wrd = chunk_obj["choices"][0]["delta"].get("content", "")
                        if wrd:
                            received.append(wrd)
            results[m_name] = ((time.perf_counter() - t_s) * 1000.0, "".join(received).strip())

        t_threads = [
            threading.Thread(target=fetch, args=("gemma-1b",)),
            threading.Thread(target=fetch, args=("gemma-4b",)),
        ]
        t_all_start = time.perf_counter()
        for th in t_threads:
            th.start()
        for th in t_threads:
            th.join()
        total_multi_ms = (time.perf_counter() - t_all_start) * 1000.0

        print(f"   ✓ Both models responded concurrently in {total_multi_ms:.1f} ms total:")
        for m_name, (lat, txt) in results.items():
            print(f"     Model '{m_name}': {lat:6.1f} ms -> \"{txt}\"")
        assert set(results) == {"gemma-1b", "gemma-4b"}, f"missing a model response: {results}"
        assert all(txt for _, txt in results.values()), f"a model returned empty text: {results}"

    finally:
        print("\n6. Shutting down server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("✓ Server cleanly shut down.")

    print("\n" + "=" * 80)
    print("✅ DUAL-MODEL CO-RESIDENCY TEST PASSED: catalog + both models streamed live.")
    print("================================================================================")


if __name__ == "__main__":
    main()
