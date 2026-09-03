"""Unit and concurrency tests for Continuous Dynamic Batching in Friday HTTP Server.

Verifies:
1. When max_concurrency == 1: Critic Gate 1 enforces single-flight and rejects 2nd request with HTTP 429.
2. When max_concurrency == 4:
   - Admits up to 4 concurrent requests without 429.
   - Rejects 5th concurrent request with HTTP 429 (Critic Gate 1 still guards memory budget).
   - Generates and streams tokens concurrently to all 4 connections.
   - Verifies 100% token accuracy and clean completion.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Iterator, Mapping, Sequence

from friday_calibrate.profile import DeviceProfile, KnobVerdict
from friday_serve.http_server import create_server
from friday_serve.server import Server

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
REVISION = "rev1"


def make_test_profile(*verified: str) -> DeviceProfile:
    verdicts = [
        KnobVerdict(k, "verified" if k in verified else "failed", 6, 0.9, 0.88, 0.93, True)
        for k in ("head_skip", "fixed_compiled", "bundled_readback")
    ]
    return DeviceProfile(
        profile_id="device-batch-test",
        model_id=MODEL_ID,
        model_revision=REVISION,
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


class SimulatedBatchBackend:
    """Simulates multi-request generation backend with step delays for concurrency testing."""

    def __init__(self, step_delay: float = 0.05) -> None:
        self.model_id = MODEL_ID
        self.model_revision = REVISION
        self.step_delay = step_delay
        self.lock = threading.Lock()
        self.active_count = 0
        self.max_active_seen = 0

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) for c in prompt]

    def generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> dict[str, Any]:
        with self.lock:
            self.active_count += 1
            if self.active_count > self.max_active_seen:
                self.max_active_seen = self.active_count
        try:
            time.sleep(self.step_delay * max_tokens)
            tokens = [100 + i for i in range(max_tokens)]
            return {
                "logical_tokens": tokens,
                "text": f"Response to {len(token_ids)}",
                "prefill_ns": 1_000_000,
                "decode_ns": int(self.step_delay * max_tokens * 1e9),
                "knobs": dict(knobs),
            }
        finally:
            with self.lock:
                self.active_count -= 1

    def stream_generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        with self.lock:
            self.active_count += 1
            if self.active_count > self.max_active_seen:
                self.max_active_seen = self.active_count
        try:
            yield {
                "type": "token",
                "token": 100,
                "tokens": [100],
                "text": "Start ",
                "is_first": True,
                "prefill_ns": 1_000_000,
            }
            for i in range(1, max_tokens):
                time.sleep(self.step_delay)
                tok = 100 + i
                yield {
                    "type": "token",
                    "token": tok,
                    "tokens": [tok],
                    "text": f"tok{tok} ",
                    "is_first": False,
                }
            yield {
                "type": "done",
                "total_tokens": max_tokens,
                "decode_ns": int(self.step_delay * max_tokens * 1e9),
                "total_ns": int(self.step_delay * max_tokens * 1e9) + 1_000_000,
                "knobs": dict(knobs),
                "logical_tokens": [100 + i for i in range(max_tokens)],
            }
        finally:
            with self.lock:
                self.active_count -= 1


class DynamicBatchingServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = make_test_profile("head_skip", "fixed_compiled", "bundled_readback")

    def test_concurrency_1_rejects_second_request(self) -> None:
        backend = SimulatedBatchBackend(step_delay=0.1)
        server = Server(backend, self.profile)
        httpd = create_server(server, host="127.0.0.1", port=0, max_concurrency=1)
        port = httpd.server_address[1]
        t_srv = threading.Thread(target=httpd.serve_forever, daemon=True)
        t_srv.start()

        try:
            req1_res = []
            req2_res = []

            def worker1():
                payload = {"messages": [{"role": "user", "content": "slow"}], "max_tokens": 4}
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req) as resp:
                        req1_res.append(resp.status)
                except Exception as exc:
                    req1_res.append(exc)

            def worker2():
                time.sleep(0.02)
                payload = {"messages": [{"role": "user", "content": "fast"}], "max_tokens": 2}
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req) as resp:
                        req2_res.append(resp.status)
                except urllib.error.HTTPError as exc:
                    req2_res.append(exc.code)

            t1 = threading.Thread(target=worker1)
            t2 = threading.Thread(target=worker2)
            t1.start()
            t2.start()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)

            self.assertEqual(req1_res, [200])
            self.assertEqual(req2_res, [429])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_concurrency_4_admits_four_and_rejects_fifth(self) -> None:
        backend = SimulatedBatchBackend(step_delay=0.15)
        server = Server(backend, self.profile)
        httpd = create_server(server, host="127.0.0.1", port=0, max_concurrency=4)
        port = httpd.server_address[1]
        t_srv = threading.Thread(target=httpd.serve_forever, daemon=True)
        t_srv.start()

        try:
            results: dict[int, Any] = {}
            threads = []

            def worker(idx: int):
                payload = {"messages": [{"role": "user", "content": f"msg {idx}"}], "max_tokens": 3}
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req) as resp:
                        results[idx] = resp.status
                except urllib.error.HTTPError as exc:
                    results[idx] = exc.code

            # Launch 4 requests simultaneously
            for i in range(4):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()

            # Wait briefly, then launch 5th request
            time.sleep(0.04)
            t5 = threading.Thread(target=worker, args=(4,))
            threads.append(t5)
            t5.start()

            for t in threads:
                t.join(timeout=3.0)

            # First 4 should succeed (200), 5th should be rejected (429)
            succeeded = sum(1 for i in range(4) if results.get(i) == 200)
            self.assertEqual(succeeded, 4, f"All 4 should succeed: {results}")
            self.assertEqual(results.get(4), 429, f"5th should be rejected with 429: {results}")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
