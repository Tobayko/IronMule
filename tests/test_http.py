"""The OpenAI-compatible HTTP endpoint, exercised against a fake runtime.

No model and no MLX: these tests check the HTTP contract (routes, OpenAI JSON
shape, SSE framing, the one-at-a-time 429), not generation.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from ironmule.http import create_server
from ironmule.service import StreamEvent


class FakeRuntime:
    model_id = "fake/model-1"

    def __init__(self, *, delay: float = 0.0, text: str = "hello world") -> None:
        self.delay = delay
        self.text = text

    def encode_chat(self, messages):
        return [1] * sum(len(m["content"]) for m in messages)

    def _tokens(self):
        return list(range(2, 2 + len(self.text.split())))

    def generate(self, *, prompt_ids, max_tokens):
        if self.delay:
            time.sleep(self.delay)
        from ironmule.service import Result
        return Result(rid="r", tokens=self._tokens(), text=self.text,
                      stop_reason="eos", metrics={})

    def stream(self, *, prompt_ids, max_tokens):
        words = self.text.split()
        for i, word in enumerate(words):
            if self.delay:
                time.sleep(self.delay)
            last = i == len(words) - 1
            yield StreamEvent(text=(word if i == 0 else " " + word),
                              token=2 + i, index=i, done=last,
                              stop_reason="eos" if last else "")


@pytest.fixture
def server():
    srv = create_server(FakeRuntime(), host="127.0.0.1", port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def _post(base, payload):
    req = urllib.request.Request(f"{base}/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=8)


def test_health_and_models(server):
    base, _ = server
    with urllib.request.urlopen(f"{base}/health", timeout=8) as resp:
        assert json.loads(resp.read()) == {"status": "ok"}
    with urllib.request.urlopen(f"{base}/v1/models", timeout=8) as resp:
        data = json.loads(resp.read())
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "fake/model-1"


def test_non_streaming_completion_is_openai_shaped(server):
    base, _ = server
    with _post(base, {"model": "x", "messages": [{"role": "user", "content": "hi"}]}) as resp:
        body = json.loads(resp.read())
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello world"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 2
    assert body["model"] == "fake/model-1"


def test_streaming_completion_emits_sse_deltas_then_done(server):
    base, _ = server
    with _post(base, {"messages": [{"role": "user", "content": "hi"}], "stream": True}) as resp:
        lines = [ln.decode("utf-8").rstrip("\n") for ln in resp if ln.strip()]
    assert lines[-1] == "data: [DONE]"
    deltas = []
    for line in lines[:-1]:
        assert line.startswith("data: ")
        chunk = json.loads(line[6:])
        assert chunk["object"] == "chat.completion.chunk"
        piece = chunk["choices"][0]["delta"].get("content")
        if piece:
            deltas.append(piece)
    assert "".join(deltas) == "hello world"
    assert json.loads(lines[-2][6:])["choices"][0]["finish_reason"] == "stop"


def test_one_request_at_a_time_returns_429(server):
    base, srv = server
    srv.runtime.delay = 0.4
    results: list[int] = []

    def hit():
        try:
            with _post(base, {"messages": [{"role": "user", "content": "hi"}]}) as resp:
                results.append(resp.status)
        except urllib.error.HTTPError as exc:
            results.append(exc.code)

    first = threading.Thread(target=hit)
    first.start()
    time.sleep(0.1)
    hit()
    first.join(timeout=3)
    assert sorted(results) == [200, 429]


def test_bad_body_is_a_400(server):
    base, _ = server
    for payload in ({}, {"messages": []}, {"messages": [{"role": "user"}]}):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _post(base, payload)
        assert caught.value.code == 400
