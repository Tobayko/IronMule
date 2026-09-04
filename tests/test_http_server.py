"""Comprehensive unit and integration tests for Friday HTTP/SSE Server.

Tests:
1. Health and Discovery Endpoints:
   - GET /health -> 200 OK {"status": "ok"}
   - GET /v1/models -> 200 OK with OpenAI model list
   - GET /status -> 200 OK with runtime explanation and knobs
2. Non-streaming Chat Completion:
   - POST /v1/chat/completions with stream=false -> 200 OK OpenAI JSON payload
3. Streaming Chat Completion:
   - POST /v1/chat/completions with stream=true -> 200 OK SSE stream
   - Verified chunk structure: deltas, stop finish_reason, and data: [DONE] termination
4. Critic Gate 1 - Concurrency Semaphore:
   - GPU inference is strictly limited to 1 concurrent request
   - Second overlapping request immediately receives HTTP 429 with 'Retry-After: 1'
5. Error Handling:
   - HTTP 400 on malformed JSON or empty messages
   - HTTP 404 on unknown paths
"""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Iterator, Mapping, Sequence

from friday_calibrate.profile import DeviceProfile, KnobVerdict
from friday_serve.http_server import create_server
from friday_serve.server import Generation, Server

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
REVISION = "rev1"


def make_profile(*verified: str) -> DeviceProfile:
    verdicts = [
        KnobVerdict(k, "verified" if k in verified else "failed", 6, 0.9, 0.88, 0.93, True)
        for k in ("head_skip", "fixed_compiled", "bundled_readback")
    ]
    return DeviceProfile(
        profile_id="device-http-test",
        model_id=MODEL_ID,
        model_revision=REVISION,
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


class FakeHTTPBackend:
    """Deterministic fake backend simulating IronMule inference for HTTP tests."""

    def __init__(self, *, delay: float = 0.0, block_event: threading.Event | None = None) -> None:
        self.model_id = MODEL_ID
        self.model_revision = REVISION
        self.delay = delay
        self.block_event = block_event
        self.calls: list[dict[str, Any]] = []

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) for c in prompt]

    def generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.calls.append({"token_ids": list(token_ids), "knobs": dict(knobs)})
        if self.delay > 0:
            time.sleep(self.delay)
        if self.block_event is not None:
            self.block_event.wait()
        tokens = list(range(100, 100 + max_tokens))
        return {
            "logical_tokens": tokens,
            "text": f"Response to {len(token_ids)} tokens",
            "prefill_ns": 500_000,
            "decode_ns": 1_000_000,
            "knobs": dict(knobs),
        }

    def stream_generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        self.calls.append({"token_ids": list(token_ids), "knobs": dict(knobs)})
        if self.delay > 0:
            time.sleep(self.delay)
        if self.block_event is not None:
            self.block_event.wait()

        # Emit first token
        yield {
            "type": "token",
            "token": 100,
            "tokens": [100],
            "text": "Hello",
            "is_first": True,
            "prefill_ns": 400_000,
            "prefix_cache_hits": 0,
        }

        # Emit subsequent tokens
        for i in range(1, max_tokens):
            yield {
                "type": "token",
                "token": 100 + i,
                "tokens": [100 + i],
                "text": f" chunk_{i}",
                "is_first": False,
            }

        # Emit final event
        yield {
            "type": "done",
            "total_tokens": max_tokens,
            "decode_ns": 800_000,
            "total_ns": 1_200_000,
            "knobs": dict(knobs),
            "prefix_cache_hits": 0,
            "logical_tokens": list(range(100, 100 + max_tokens)),
        }


class TestHTTPServer(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeHTTPBackend()
        self.profile = make_profile("head_skip")
        self.server_instance = Server(self.backend, self.profile)
        # Start server on ephemeral port (port=0)
        self.httpd = create_server(self.server_instance, host="127.0.0.1", port=0)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=2.0)

    def test_health_endpoint(self) -> None:
        """GET /health returns 200 OK status."""
        req = urllib.request.Request(f"{self.base_url}/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data, {"status": "ok"})

    def test_models_endpoint(self) -> None:
        """GET /v1/models returns OpenAI model list format."""
        req = urllib.request.Request(f"{self.base_url}/v1/models")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["object"], "list")
            self.assertTrue(len(data["data"]) >= 1)
            self.assertEqual(data["data"][0]["id"], MODEL_ID)

    def test_status_endpoint(self) -> None:
        """GET /status returns server explanation and model details."""
        req = urllib.request.Request(f"{self.base_url}/status")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "serving")
            self.assertEqual(data["model_id"], MODEL_ID)
            self.assertIn("explanation", data)

    def test_chat_completions_non_streaming(self) -> None:
        """POST /v1/chat/completions (stream=false) returns valid OpenAI completion."""
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "What is Friday?"}],
            "max_tokens": 5,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res["object"], "chat.completion")
            self.assertTrue(res["id"].startswith("chatcmpl-"))
            self.assertEqual(len(res["choices"]), 1)
            self.assertEqual(res["choices"][0]["message"]["role"], "assistant")
            self.assertIn("Response to", res["choices"][0]["message"]["content"])
            self.assertEqual(res["choices"][0]["finish_reason"], "stop")
            self.assertGreater(res["usage"]["total_tokens"], 0)

    def test_chat_completions_streaming(self) -> None:
        """POST /v1/chat/completions (stream=true) returns valid SSE stream ending with [DONE]."""
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": 4,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "text/event-stream")

            received_chunks: list[dict[str, Any]] = []
            saw_done = False

            while True:
                line = resp.readline().decode("utf-8")
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    saw_done = True
                    break
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    received_chunks.append(chunk)

            self.assertTrue(saw_done)
            self.assertTrue(len(received_chunks) >= 4)

            # First chunk content
            first_delta = received_chunks[0]["choices"][0]["delta"]
            self.assertEqual(first_delta["content"], "Hello")

            # Final finish chunk
            last_choice = received_chunks[-1]["choices"][0]
            self.assertEqual(last_choice["finish_reason"], "stop")

    def test_critic_gate_concurrency_semaphore(self) -> None:
        """Concurrent inference requests trigger HTTP 429 on overload (Gate 1)."""
        block_event = threading.Event()
        blocking_backend = FakeHTTPBackend(block_event=block_event)
        blocking_server = Server(blocking_backend, self.profile)

        httpd = create_server(blocking_server, host="127.0.0.1", port=0)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()

        try:
            req1_started = threading.Event()
            req1_status = []

            def worker1():
                payload = {
                    "messages": [{"role": "user", "content": "slow prompt"}],
                    "stream": False,
                    "max_tokens": 2,
                }
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                req1_started.set()
                try:
                    with urllib.request.urlopen(req) as resp:
                        req1_status.append(resp.status)
                except Exception as exc:
                    req1_status.append(exc)

            t1 = threading.Thread(target=worker1)
            t1.start()

            # Wait until request 1 is launched
            req1_started.wait()
            # Brief pause to ensure worker1 has acquired the concurrency semaphore
            time.sleep(0.05)

            # Request 2 should immediately hit HTTP 429
            payload2 = {
                "messages": [{"role": "user", "content": "second prompt"}],
                "stream": False,
                "max_tokens": 2,
            }
            req2 = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(payload2).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req2)

            self.assertEqual(caught.exception.code, 429)
            self.assertEqual(caught.exception.headers.get("Retry-After"), "1")
            err_body = json.loads(caught.exception.read().decode("utf-8"))
            self.assertEqual(err_body["error"]["code"], 429)
            self.assertEqual(err_body["error"]["type"], "concurrency_limit_error")

            # Release request 1
            block_event.set()
            t1.join(timeout=2.0)
            self.assertEqual(req1_status, [200])
        finally:
            block_event.set()
            httpd.shutdown()
            httpd.server_close()
            t.join(timeout=2.0)

    def test_bad_request_handling(self) -> None:
        """Invalid JSON or missing messages yields HTTP 400."""
        req_bad_json = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req_bad_json)
        self.assertEqual(caught.exception.code, 400)

        req_no_msg = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps({"max_tokens": 10}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req_no_msg)
        self.assertEqual(caught.exception.code, 400)

    def test_not_found_endpoint(self) -> None:
        """Unknown endpoints return HTTP 404."""
        req = urllib.request.Request(f"{self.base_url}/unknown_route")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req)
        self.assertEqual(caught.exception.code, 404)


class _TaggedBackend:
    """A fake backend that stamps its own model id into every chunk of text."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.model_revision = REVISION

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) for c in prompt]

    def stream_generate(self, token_ids, max_tokens, knobs):
        yield {"type": "token", "token": 1, "tokens": [1], "text": f"[{self.model_id}]",
               "is_first": True, "prefill_ns": 1, "prefix_cache_hits": 0}
        yield {"type": "done", "total_tokens": 1, "decode_ns": 1, "total_ns": 2,
               "knobs": dict(knobs), "prefix_cache_hits": 0, "logical_tokens": [1]}

    def generate(self, token_ids, max_tokens, knobs):
        return {"logical_tokens": [1], "text": f"[{self.model_id}]",
                "prefill_ns": 1, "decode_ns": 1, "knobs": dict(knobs)}


class DualModelRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.alt_id = "mlx-community/gemma-3-1b-it-4bit"
        main = _TaggedBackend(MODEL_ID)
        alt = _TaggedBackend(self.alt_id)
        server = Server(main, make_profile("head_skip"),
                        alternate_backends={self.alt_id: alt, "gemma-1b": alt})
        self.httpd = create_server(server, host="127.0.0.1", port=0)
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.t.join(timeout=2.0)

    def test_models_lists_both_and_no_bogus_entry(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/v1/models") as resp:
            ids = {m["id"] for m in json.loads(resp.read())["data"]}
        self.assertEqual(ids, {MODEL_ID, self.alt_id})
        self.assertNotIn("gemma-4b", ids)

    def test_streaming_request_for_the_alt_model_is_served_by_the_alt_backend(self) -> None:
        payload = {"model": "gemma-1b", "stream": True,
                   "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
        self.assertIn(f"[{self.alt_id}]", body)
        self.assertNotIn(f"[{MODEL_ID}]", body)
        self.assertIn('"model":"gemma-1b"', body.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
