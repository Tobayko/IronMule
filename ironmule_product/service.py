"""Bounded product scheduling around a persistent, isolated model backend.

Only counters and timings are retained by the service. Prompt and response text
exist solely in the in-flight request and are never written to product state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Iterator

from .errors import (BackendUnavailable, InvalidRequest, StateError,
                     Overloaded, ProductError, RequestCancelled, RequestTimeout)
from .types import GenerationRequest, ModelSpec


class StopFilter:
    """Suppress stop strings, including strings spanning stream chunks."""

    def __init__(self, stops: tuple[str, ...]):
        self.stops = stops
        self.pending = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        self.pending += text
        matches = [self.pending.find(stop) for stop in self.stops if stop in self.pending]
        if matches:
            visible = self.pending[:min(matches)]
            self.pending = ""
            self.stopped = True
            return visible
        hold = 0
        for stop in self.stops:
            for size in range(min(len(stop) - 1, len(self.pending)), hold, -1):
                if self.pending.endswith(stop[:size]):
                    hold = size
                    break
        if hold:
            visible, self.pending = self.pending[:-hold], self.pending[-hold:]
        else:
            visible, self.pending = self.pending, ""
        return visible

    def finish(self) -> str:
        visible, self.pending = self.pending, ""
        return "" if self.stopped else visible


@dataclass
class _Session:
    request: GenerationRequest
    deadline: float | None
    submitted_ns: int = field(default_factory=time.monotonic_ns)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=32))
    cancelled: threading.Event = field(default_factory=threading.Event)
    backend_cancel: threading.Event = field(default_factory=threading.Event)


class ProductService:
    """One loaded reference model with bounded FIFO admission.

    This is the portable reference path. Optimized grouping is not claimed to
    exist here or enabled without a separately qualified configuration.
    """

    def __init__(self, store, *, backend=None, spec: ModelSpec | None = None):
        self.store = store
        self.settings = store.settings()
        self.backend = backend
        self.spec = spec
        self._pending: queue.Queue = queue.Queue(maxsize=self.settings["max_pending"])
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._active: str | None = None
        self._completed = self._failed = self._cancelled = 0
        self._thread = threading.Thread(target=self._run, name="ironmule-scheduler", daemon=True)
        self._thread.start()

    def health(self) -> dict[str, Any]:
        try:
            optimization = self.store.optimization_status()
        except StateError:
            # Corrupt optimizer diagnostics disable optimization, not a healthy
            # stock model. Model-registration admission remains independently
            # validated for every actual request.
            optimization = {"stage": "unavailable", "engine_started": False,
                            "activation_allowed": False, "error_code": "optimization_status_invalid"}
        with self._lock:
            return {"service": "ironmule", "ready": bool(not self._closed.is_set() and self.backend is not None and self.backend.ready),
                    "mode": self.settings["mode"], "execution": "exact",
                    "backend": ("mlx_lm_reference" if getattr(self.backend, "execution_variant", "reference") == "reference"
                                else getattr(self.backend, "execution_variant", "reference")),
                    "loaded_model": self.spec.model_id if self.spec else None,
                    "queued_requests": self._pending.qsize(), "active_requests": int(self._active is not None),
                    "completed_requests": self._completed, "failed_requests": self._failed,
                    "cancelled_requests": self._cancelled,
                    "optimization": optimization}

    def models(self) -> list[dict[str, Any]]:
        return [{"id": spec.model_id, "object": "model", "owned_by": "local",
                 "revision": spec.revision, "loaded": self.spec == spec and self.health()["ready"]}
                for spec in self.store.models()]

    def stream(self, request: GenerationRequest) -> Iterator[dict[str, Any]]:
        # Validate direct Python callers as well as the HTTP boundary. No
        # request id or execution switch comes from a remote payload.
        payload = request.as_dict()
        payload.pop("request_id", None)
        GenerationRequest.from_payload(payload, exact=True)
        with self._lock:
            if self._closed.is_set():
                raise BackendUnavailable("service is closed")
            registered = self.store.model(request.model)
            if self.spec is None or self.backend is None or not self.backend.ready:
                raise BackendUnavailable("no model worker is ready")
            if registered != self.spec:
                raise BackendUnavailable("requested model is registered but not loaded by this service")
            if request.request_id in self._sessions:
                raise InvalidRequest("duplicate request id")
            timeout = self.settings["request_timeout_s"]
            session = _Session(
                request,
                None if timeout is None else time.monotonic() + timeout,
            )
            try:
                self._pending.put_nowait(session)
            except queue.Full as exc:
                raise Overloaded("request queue is full; retry later") from exc
            self._sessions[request.request_id] = session
        return self._events(session)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(request_id)
            if session is None:
                return False
            session.cancelled.set()
            session.backend_cancel.set()
            return True

    def _events(self, session: _Session) -> Iterator[dict[str, Any]]:
        try:
            while True:
                if session.cancelled.is_set() or self._closed.is_set():
                    raise RequestCancelled("request cancelled")
                remaining = None if session.deadline is None else session.deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestTimeout("request deadline exceeded")
                try:
                    event = session.events.get(timeout=0.1 if remaining is None else min(remaining, 0.1))
                except queue.Empty:
                    continue
                if isinstance(event, ProductError):
                    raise event
                yield event
                if event.get("type") == "done":
                    return
        finally:
            self.cancel(session.request.request_id)

    def _emit(self, session: _Session, event: Any) -> bool:
        while not self._closed.is_set() and not session.cancelled.is_set():
            remaining = None if session.deadline is None else session.deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                session.backend_cancel.set()
                return False
            try:
                session.events.put(event, timeout=0.1 if remaining is None else min(remaining, 0.1))
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                session = self._pending.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if session.cancelled.is_set():
                    with self._lock:
                        self._cancelled += 1
                    continue
                with self._lock:
                    self._active = session.request.request_id
                self._perform(session)
            finally:
                with self._lock:
                    self._sessions.pop(session.request.request_id, None)
                    self._active = None
                self._pending.task_done()

    def _drain_cancelled_stream(self, stream: Iterator[dict[str, Any]]) -> bool:
        """Consume a cancelled backend request without delivering its content.

        The backend's cancellation event has already been set.  Reaching its
        terminal event keeps the persistent protocol synchronized; any drain
        failure retires the backend instead of claiming warm recovery.
        """
        try:
            for event in stream:
                if event.get("type") == "done":
                    return True
        except Exception:
            return False
        return False

    def _finish_cancelled(
        self,
        session: _Session,
        stream: Iterator[dict[str, Any]] | None,
        *,
        terminal_seen: bool,
    ) -> None:
        session.backend_cancel.set()
        drained = terminal_seen
        if stream is not None and not drained:
            drained = self._drain_cancelled_stream(stream)
        if not drained and self.backend is not None:
            # A partial or failed drain leaves protocol synchronization
            # unknown. Retire this worker; never restart or retry here.
            try:
                self.backend.close()
            except Exception:
                pass
        self._emit(session, RequestCancelled("request cancelled"))
        with self._lock:
            self._cancelled += 1

    def _perform(self, session: _Session) -> None:
        started = time.monotonic_ns()
        stop_filter = StopFilter(session.request.stop)
        stream = None
        done = None
        generated = 0
        prompt_tokens = 0
        try:
            remaining = None if session.deadline is None else session.deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise RequestTimeout("request expired in queue")
            if self.backend is None or not self.backend.ready:
                raise BackendUnavailable("model worker is unavailable")
            stream = self.backend.stream(session.request, cancel=session.backend_cancel, timeout=remaining)
            for event in stream:
                if session.cancelled.is_set():
                    if event.get("type") == "done":
                        done = dict(event)
                    raise RequestCancelled("request cancelled")
                kind = event.get("type")
                if kind == "token":
                    if done is not None:
                        raise BackendUnavailable("worker emitted tokens after completion")
                    generated += 1
                    if generated > session.request.max_tokens:
                        raise BackendUnavailable("worker exceeded the token limit")
                    prompt_tokens = event.get("prompt_tokens", prompt_tokens)
                    text = stop_filter.feed(event.get("text", ""))
                    if text and not self._emit(session, {**event, "text": text}):
                        raise RequestCancelled("response delivery interrupted")
                    if stop_filter.stopped:
                        session.backend_cancel.set()
                elif kind == "done":
                    if done is not None:
                        raise BackendUnavailable("worker emitted duplicate completion")
                    done = dict(event)
                elif kind == "error":
                    raise BackendUnavailable("model worker failed")
                else:
                    raise BackendUnavailable("worker emitted an invalid event")
            if done is None:
                raise BackendUnavailable("worker stream ended without completion")
            if session.cancelled.is_set():
                raise RequestCancelled("request cancelled")
            tail = stop_filter.finish()
            if tail and not self._emit(session, {"type": "token", "text": tail, "token_id": None}):
                raise RequestCancelled("response delivery interrupted")
            finish = "stop" if stop_filter.stopped else done.get("finish_reason")
            if finish not in ("stop", "length"):
                raise RequestCancelled("model generation cancelled")
            if not self._emit(session, {**done, "type": "done", "finish_reason": finish,
                                       "prompt_tokens": done.get("prompt_tokens", prompt_tokens),
                                       "completion_tokens": done.get("completion_tokens", generated),
                                       "queue_ns": started - session.submitted_ns,
                                       "service_ns": time.monotonic_ns() - session.submitted_ns}):
                raise RequestCancelled("response delivery interrupted")
            with self._lock:
                self._completed += 1
        except RequestCancelled:
            self._finish_cancelled(session, stream, terminal_seen=done is not None)
        except ProductError as exc:
            if session.cancelled.is_set():
                self._finish_cancelled(session, stream, terminal_seen=done is not None)
            else:
                self._emit(session, exc)
                with self._lock:
                    self._failed += 1
        except Exception:
            if session.cancelled.is_set():
                self._finish_cancelled(session, stream, terminal_seen=done is not None)
            else:
                # Library exceptions can include prompt text. Never reflect
                # their bodies or persist them as metadata.
                self._emit(session, BackendUnavailable("model worker failed"))
                with self._lock:
                    self._failed += 1
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            for session in self._sessions.values():
                session.cancelled.set()
                session.backend_cancel.set()
        if self.backend is not None:
            self.backend.close()
        self._thread.join(timeout=2.0)
        with self._lock:
            self._sessions.clear()
        while True:
            try:
                self._pending.get_nowait()
                self._pending.task_done()
            except queue.Empty:
                break
