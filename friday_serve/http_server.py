"""Lean OpenAI-compatible HTTP and SSE Server for Friday LLM Runtime.

Zero external dependencies: built strictly on the Python Standard Library
(http.server.ThreadingHTTPServer, json, threading, time, uuid).

Gates & Concurrency:
- Concurrency Semaphore (value 1) gates all GPU inference requests to prevent
  VRAM thrashing and memory swap (Critic Gate 1).
- Overload immediately returns HTTP 429 Too Many Requests with 'Retry-After: 1'.

Endpoints:
- POST /v1/chat/completions: Supports both stream: true (SSE) and stream: false (JSON).
- GET /v1/models: OpenAI model list for tooling compatibility (Cursor, OpenWebUI).
- GET /status: Friday runtime status, active knobs, and hardware explanation.
- GET /health: Simple liveness probe (HTTP 200).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence

from friday_runtime_core.controller import RuntimeExecutionError
from friday_serve.server import Server
from friday_serve.telemetry import TelemetryTracker, get_global_tracker
from friday_serve.terminal_dashboard import print_live_cockpit, render_cockpit


def _extract_prompt_and_tokens(server_instance: Server, body: dict[str, Any]) -> str | list[int]:
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise ValueError("Field 'messages' must be a non-empty list")

    backend = getattr(server_instance, "backend", None)
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            if isinstance(rendered, list):
                return [int(t) for t in rendered]
            elif isinstance(rendered, str) and hasattr(tokenizer, "encode"):
                return [int(t) for t in tokenizer.encode(rendered)]
        except Exception:
            pass

    # Single message extraction
    if len(messages) == 1 and isinstance(messages[0], dict):
        return str(messages[0].get("content", ""))

    # Fallback multi-turn formatting
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
    return "\n\n".join(parts)


class FridayRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing OpenAI v1 API and Friday telemetry."""

    server: FridayHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr request logging."""
        pass

    def _send_json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if path == "/v1/models":
            model_id = getattr(self.server.server_instance.backend, "model_id", "friday-model")
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "friday",
                        }
                    ],
                },
            )
            return

        if path == "/status":
            explained = {}
            if hasattr(self.server.server_instance, "explain"):
                try:
                    explained = self.server.server_instance.explain()
                except Exception:
                    explained = {}
            model_id = getattr(self.server.server_instance.backend, "model_id", "unknown")
            revision = getattr(self.server.server_instance.backend, "model_revision", "unknown")
            self._send_json(
                200,
                {
                    "status": "serving",
                    "model_id": model_id,
                    "model_revision": revision,
                    "explanation": explained,
                },
            )
            return

        if path == "/dashboard":
            tracker = self.server.telemetry_tracker
            rendered = render_cockpit(tracker, colored=True)
            data = rendered.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/telemetry":
            tracker = self.server.telemetry_tracker
            current = tracker.get_current()
            history = [m.as_dict() for m in tracker.get_history()]
            live = tracker.get_live().as_dict()
            self._send_json(
                200,
                {
                    "current": current.as_dict() if current else None,
                    "live": live,
                    "history": history,
                    "peak_bandwidth_gbs": tracker.peak_bandwidth_gbs,
                },
            )
            return

        if path == "/v1/models":
            main_id = getattr(self.server.server_instance.backend, "model_id", "friday-model")
            models = [
                {"id": main_id, "object": "model", "owned_by": "project-friday"},
                {"id": "gemma-4b", "object": "model", "owned_by": "project-friday"},
            ]
            alt = getattr(self.server.server_instance, "alternate_backends", {})
            for k, b in alt.items():
                m_id = getattr(b, "model_id", k)
                if not any(m["id"] == m_id for m in models):
                    models.append({"id": m_id, "object": "model", "owned_by": "project-friday"})
                if not any(m["id"] == k for m in models):
                    models.append({"id": k, "object": "model", "owned_by": "project-friday"})
            self._send_json(200, {"object": "list", "data": models})
            return

        self._send_json(404, {"error": {"message": f"Not found: {self.path}", "code": 404}})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"Not found: {self.path}", "code": 404}})
            return

        # Parse request body
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"Invalid JSON body: {exc}", "code": 400}})
            return

        try:
            prompt = _extract_prompt_and_tokens(self.server.server_instance, body)
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc), "code": 400}})
            return

        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 128)
        stream = bool(body.get("stream", False))

        target_model = body.get("model")
        is_alternate_model = bool(
            target_model
            and target_model != getattr(self.server.server_instance.backend, "model_id", "")
            and any(k in target_model.lower() for k in ("1b", "alternate"))
        )
        # Dynamic Batcher Path if enabled
        if self.server.batcher is not None and not is_alternate_model:
            try:
                if isinstance(prompt, str):
                    token_ids = self.server.server_instance.backend.encode(prompt)
                else:
                    token_ids = prompt

                from .dispatch import knobs_for
                knobs = knobs_for(self.server.server_instance.profile)

                session = self.server.batcher.submit(token_ids, max_tokens, knobs)
            except RuntimeError:
                self._send_json(
                    429,
                    {
                        "error": {
                            "message": f"Too Many Requests: Single-model GPU inference concurrency limit reached ({self.server.max_concurrency}).",
                            "type": "concurrency_limit_error",
                            "code": 429,
                        }
                    },
                    headers={"Retry-After": "1"},
                )
                return

            try:
                if stream:
                    self._handle_batcher_stream(session, body)
                else:
                    self._handle_batcher_non_stream(session, body)
            except (BrokenPipeError, ConnectionResetError):
                session.cancel()
            return

        # Fallback / Single-flight Concurrency Gate: non-blocking acquire
        acquired = self.server.concurrency_semaphore.acquire(blocking=False)
        if not acquired:
            self._send_json(
                429,
                {
                    "error": {
                        "message": f"Too Many Requests: Single-model GPU inference concurrency limit reached ({self.server.max_concurrency}).",
                        "type": "concurrency_limit_error",
                        "code": 429,
                    }
                },
                headers={"Retry-After": "1"},
            )
            return

        try:
            if stream:
                self._handle_stream(prompt, max_tokens, body)
            else:
                self._handle_non_stream(prompt, max_tokens, body)
        finally:
            self.server.concurrency_semaphore.release()

    def _handle_batcher_stream(self, session: Any, body: dict[str, Any]) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = getattr(self.server.server_instance.backend, "model_id", "friday-model")
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        init_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        self.wfile.write(f"data: {json.dumps(init_chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()

        chunk_prefix = f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{created},"model":"{model}","choices":[{{"index":0,"delta":{{"content":'.encode("utf-8")
        chunk_suffix = b'},"finish_reason":null}]}\n\n'

        for event in session.stream():
            etype = event.get("type")
            if etype == "token":
                text = event.get("text", "")
                if text:
                    self.wfile.write(chunk_prefix + json.dumps(text).encode("utf-8") + chunk_suffix)
                    self.wfile.flush()
            elif etype == "done":
                finish_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self.wfile.write(f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                break

    def _handle_batcher_non_stream(self, session: Any, body: dict[str, Any]) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = getattr(self.server.server_instance.backend, "model_id", "friday-model")
        created = int(time.time())

        collected_text = []
        logical_tokens = []
        prompt_len = len(session.prompt_ids)

        for event in session.stream():
            etype = event.get("type")
            if etype == "token":
                text = event.get("text", "")
                if text:
                    collected_text.append(text)
            elif etype == "done":
                logical_tokens = event.get("logical_tokens", [])
                break

        full_content = "".join(collected_text)
        completion_len = len(logical_tokens)

        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_len,
                "total_tokens": prompt_len + completion_len,
            },
        }
        self._send_json(200, response)

    def _handle_non_stream(
        self, prompt: str | Sequence[int], max_tokens: int, body: dict[str, Any]
    ) -> None:
        server_instance = self.server.server_instance
        model = body.get("model") or getattr(server_instance.backend, "model_id", "friday-model")
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        try:
            backend = server_instance.get_backend(model)
            if isinstance(prompt, str):
                gen = server_instance.generate(prompt, max_tokens=max_tokens, model=model)
                full_text = gen.text or ""
                num_tokens = len(gen.tokens)
                prompt_tokens = len(backend.encode(prompt))
            else:
                events = list(server_instance.stream_generate(prompt, max_tokens=max_tokens, model=model))
                text_parts = [e.get("text", "") for e in events if e.get("type") == "token"]
                full_text = "".join(text_parts)
                done_ev = events[-1] if events and events[-1].get("type") == "done" else {}
                num_tokens = int(done_ev.get("total_tokens", len(text_parts)))
                prompt_tokens = len(prompt)
        except RuntimeExecutionError as exc:
            self._send_json(500, {"error": {"message": str(exc), "code": 500}})
            return
        except Exception as exc:
            self._send_json(500, {"error": {"message": f"Inference failed: {exc}", "code": 500}})
            return

        if self.server.telemetry_tracker is not None:
            breaker = getattr(server_instance.controller, "circuit_reason", None) or "nominal"
            prefill_ns = getattr(gen, "prefill_ns", 0) if isinstance(prompt, str) else 0
            decode_ns = getattr(gen, "decode_ns", 0) if isinstance(prompt, str) else 0
            prefix_hits = getattr(gen, "prefix_cache_hits", 0) if isinstance(prompt, str) else 0
            action = getattr(gen, "plan", "baseline") if isinstance(prompt, str) else "baseline"
            self.server.telemetry_tracker.record_request(
                model_id=model,
                prefill_ns=prefill_ns,
                decode_ns=decode_ns,
                tokens_generated=num_tokens,
                prefix_cache_hits=prefix_hits,
                action=action,
                breaker_status=breaker,
            )
            if self.server.enable_dashboard and not getattr(self.server, "interactive_dashboard", False):
                print_live_cockpit(self.server.telemetry_tracker)

        self._send_json(
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": num_tokens,
                    "total_tokens": prompt_tokens + num_tokens,
                },
            },
        )
        self.close_connection = True

    def _handle_stream(
        self, prompt: str | Sequence[int], max_tokens: int, body: dict[str, Any]
    ) -> None:
        server_instance = self.server.server_instance
        model = getattr(server_instance.backend, "model_id", "friday-model")
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        first_prefill_ns = 0
        first_hits = 0
        token_count = 0

        backend = server_instance.get_backend(model)
        prompt_len = len(prompt) if isinstance(prompt, (list, tuple)) else len(backend.encode(prompt))
        if self.server.telemetry_tracker is not None:
            self.server.telemetry_tracker.start_request(
                model_id=model,
                prompt_tokens=prompt_len,
                max_tokens=max_tokens,
                action=getattr(server_instance.controller, "plan", "device_profile_dispatch"),
            )

        try:
            for event in server_instance.stream_generate(prompt, max_tokens=max_tokens, model=model):
                event_type = event.get("type")
                if event_type == "token":
                    token_count += 1
                    if event.get("is_first"):
                        first_prefill_ns = event.get("prefill_ns", 0)
                        first_hits = event.get("prefix_cache_hits", 0)
                        if self.server.telemetry_tracker is not None:
                            self.server.telemetry_tracker.update_first_token(
                                first_prefill_ns, first_hits > 0
                            )
                    else:
                        if self.server.telemetry_tracker is not None:
                            self.server.telemetry_tracker.update_tokens(token_count)
                    content = event.get("text", "")
                    if content:
                        chunk_prefix = f'data: {{"id":"{completion_id}","object":"chat.completion.chunk","created":{created},"model":"{model}","choices":[{{"index":0,"delta":{{"content":'.encode("utf-8")
                        chunk_suffix = b'},"finish_reason":null}]}\n\n'
                        self.wfile.write(chunk_prefix + json.dumps(content).encode("utf-8") + chunk_suffix)
                        self.wfile.flush()
                elif event_type == "done":
                    finish_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                    if self.server.telemetry_tracker is not None:
                        breaker = getattr(server_instance.controller, "circuit_reason", None) or "nominal"
                        self.server.telemetry_tracker.record_request(
                            model_id=model,
                            prefill_ns=first_prefill_ns,
                            decode_ns=event.get("decode_ns", 0),
                            tokens_generated=event.get("total_tokens", token_count),
                            prefix_cache_hits=event.get("prefix_cache_hits", first_hits),
                            action=event.get("plan", "device_profile_dispatch"),
                            breaker_status=breaker,
                        )
                        if self.server.enable_dashboard and not getattr(self.server, "interactive_dashboard", False):
                            print_live_cockpit(self.server.telemetry_tracker)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            err_chunk = {
                "error": {
                    "message": str(exc),
                    "code": 500,
                }
            }
            try:
                self.wfile.write(f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        finally:
            self.close_connection = True


class FridayHTTPServer(ThreadingHTTPServer):
    """Threading HTTP Server holding Friday runtime instance, telemetry tracker, and concurrency semaphore."""

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        server_instance: Server,
        telemetry_tracker: TelemetryTracker | None = None,
        enable_dashboard: bool = False,
        interactive_dashboard: bool = False,
        max_concurrency: int = 1,
    ) -> None:
        self.server_instance = server_instance
        self.telemetry_tracker = telemetry_tracker if telemetry_tracker is not None else get_global_tracker()
        self.enable_dashboard = enable_dashboard
        self.interactive_dashboard = interactive_dashboard
        self.max_concurrency = max(1, int(max_concurrency))
        self.concurrency_semaphore = threading.Semaphore(self.max_concurrency)
        self.batcher = None
        if self.max_concurrency > 1 and hasattr(server_instance.backend, "_engine"):
            from .batcher import ContinuousBatcher
            self.batcher = ContinuousBatcher(server_instance.backend, max_concurrency=self.max_concurrency)

        self.allow_reuse_address = True
        self.daemon_threads = True
        self.stop_event = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        super().__init__(server_address, RequestHandlerClass)

        if self.telemetry_tracker is not None:
            self.telemetry_tracker.set_server_info(server_address[0], server_address[1])

        if interactive_dashboard:
            self.start_interactive_monitor()

    def start_interactive_monitor(self, refresh_hz: float = 10.0) -> None:
        """Launch background interactive terminal monitor loop."""
        from .terminal_dashboard import run_interactive_monitor

        self.interactive_dashboard = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(
            target=run_interactive_monitor,
            args=(self.telemetry_tracker, self.stop_event, refresh_hz, True),
            daemon=True,
            name="IronMuleLiveCockpit",
        )
        self.monitor_thread.start()

    def stop_interactive_monitor(self) -> None:
        """Signal interactive monitor thread to stop cleanly and restore terminal."""
        self.stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def server_close(self) -> None:
        if self.batcher is not None:
            self.batcher.stop()
        self.stop_interactive_monitor()
        super().server_close()


def create_server(
    server_instance: Server,
    host: str = "127.0.0.1",
    port: int = 8080,
    telemetry_tracker: TelemetryTracker | None = None,
    enable_dashboard: bool = False,
    interactive_dashboard: bool = False,
    max_concurrency: int = 1,
) -> FridayHTTPServer:
    """Create and return a configured FridayHTTPServer."""
    return FridayHTTPServer(
        (host, port),
        FridayRequestHandler,
        server_instance,
        telemetry_tracker=telemetry_tracker,
        enable_dashboard=enable_dashboard,
        interactive_dashboard=interactive_dashboard,
        max_concurrency=max_concurrency,
    )


__all__ = ["FridayHTTPServer", "FridayRequestHandler", "create_server"]
