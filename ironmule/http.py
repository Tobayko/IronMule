"""An OpenAI-compatible HTTP endpoint for a loaded :class:`ironmule.Runtime`.

Standard library only — no web framework, no extra dependency. It exposes the
runtime over ``POST /v1/chat/completions`` (streaming and non-streaming) and
``GET /v1/models`` so an OpenAI client, Cursor, or Open WebUI can talk to a
local model.

One model, one request at a time. This is the interactive (sequential) path:
the server holds a single permit and answers HTTP 429 while it is busy rather
than interleaving requests through one engine. Token output is identical to
``Runtime.generate`` / ``Runtime.stream`` — the HTTP layer adds no sampling and
no batching.

    from ironmule import Runtime
    from ironmule.http import serve

    serve(Runtime.load(), host="127.0.0.1", port=8000)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _messages_to_prompt_ids(runtime: Any, body: dict) -> list[int]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("each message needs a 'role' and a 'content'")
        clean.append({"role": str(message.get("role", "user")),
                      "content": str(message["content"])})
    return runtime.encode_chat(clean)


def _max_tokens(body: dict) -> int:
    raw = body.get("max_tokens")
    if raw is None:
        raw = body.get("max_completion_tokens")
    if raw is None:
        return 256
    value = int(raw)
    if value < 1:
        raise ValueError("max_tokens must be at least 1")
    return value


class _Handler(BaseHTTPRequestHandler):
    server_version = "IronMule"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------
    def log_message(self, *_args: Any) -> None:  # keep the terminal quiet
        pass

    @property
    def runtime(self) -> Any:
        return self.server.runtime  # type: ignore[attr-defined]

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._json(status, {"error": {"message": message, "type": kind, "code": status}})

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    # -- routes -----------------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._json(200, {"status": "ok"})
        elif path == "/v1/models":
            model_id = getattr(self.runtime, "model_id", "") or "local-model"
            self._json(200, {"object": "list", "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "ironmule"},
            ]})
        else:
            self._error(404, f"no route for GET {path}", "not_found")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._error(404, f"no route for POST {path}", "not_found")
            return
        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, f"invalid JSON body: {exc}")
            return
        try:
            prompt_ids = _messages_to_prompt_ids(self.runtime, body)
            max_tokens = _max_tokens(body)
        except ValueError as exc:
            self._error(400, str(exc))
            return

        permit = self.server.permit  # type: ignore[attr-defined]
        if not permit.acquire(blocking=False):
            self._error(429, "the model is busy with another request", "rate_limit_error")
            return
        try:
            if bool(body.get("stream", False)):
                self._stream(prompt_ids, max_tokens)
            else:
                self._complete(prompt_ids, max_tokens)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            permit.release()

    # -- generation -----------------------------------------------------------
    def _model_id(self) -> str:
        return getattr(self.runtime, "model_id", "") or "local-model"

    def _complete(self, prompt_ids: list[int], max_tokens: int) -> None:
        result = self.runtime.generate(prompt_ids=prompt_ids, max_tokens=max_tokens)
        finish = "stop" if result.stop_reason == "eos" else "length"
        self._json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._model_id(),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(result.tokens),
                "total_tokens": len(prompt_ids) + len(result.tokens),
            },
        })

    def _stream(self, prompt_ids: list[int], max_tokens: int) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model = self._model_id()

        # The stream ends when the connection closes: no Content-Length, so say so.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(delta: dict, finish: Any) -> None:
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        send({"role": "assistant", "content": ""}, None)
        finish_reason = "stop"
        for event in self.runtime.stream(prompt_ids=prompt_ids, max_tokens=max_tokens):
            if event.text:
                send({"content": event.text}, None)
            if event.done:
                finish_reason = "stop" if event.stop_reason == "eos" else "length"
        send({}, finish_reason)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class IronMuleHTTPServer(ThreadingHTTPServer):
    """A threading HTTP server bound to one loaded runtime."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], runtime: Any) -> None:
        self.runtime = runtime
        self.permit = threading.Semaphore(1)
        super().__init__(address, _Handler)


def create_server(runtime: Any, host: str = "127.0.0.1", port: int = 8000) -> IronMuleHTTPServer:
    return IronMuleHTTPServer((host, port), runtime)


def serve(runtime: Any, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the endpoint until interrupted."""
    server = create_server(runtime, host, port)
    model = getattr(runtime, "model_id", "") or "local-model"
    print(f"IronMule OpenAI endpoint: http://{host}:{port}/v1  (model: {model})")
    print("  POST /v1/chat/completions   GET /v1/models   GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


__all__ = ["IronMuleHTTPServer", "create_server", "serve"]
