"""Process-isolated stock ``mlx_lm`` backend for the local product API.

The parent process intentionally imports no MLX or model code.  A worker owns
one loaded model and communicates through a bounded JSON-lines protocol.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from .errors import BackendUnavailable, InvalidRequest, RequestTimeout
from .types import GenerationRequest, ModelSpec


PROTOCOL_VERSION = 1
MAX_PROTOCOL_LINE = 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024
DEFAULT_TIMEOUT = 120.0
_CANCEL_POLL_SECONDS = 0.05
_DEFAULT_DEADLINE = object()
WORKER_VARIANTS = frozenset(("reference", "bounded_prefetch", "prefix_reuse", "current_engine"))
ENGINE_CONFIGURATIONS = frozenset(("current_profile", "baseline_interactive", "core_interactive",
                                   "baseline_throughput", "core_throughput"))
CONTEXT_LIMIT = 8192


class _StartupGuardFailure(Exception):
    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _bounded_json_line(value: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BackendUnavailable("backend protocol encoding failed") from exc
    if len(encoded) >= MAX_PROTOCOL_LINE:
        raise BackendUnavailable("backend protocol message is too large")
    return encoded + b"\n"


class MLXWorkerClient:
    """A single persistent, GPU-only stock ``mlx_lm`` worker."""

    def __init__(self, spec: ModelSpec, *, startup_timeout: float | None = DEFAULT_TIMEOUT,
                 execution_variant: str = "reference", prefix_cache_max_entries: int = 4,
                 prefix_cache_max_bytes: int = 1024**3,
                 trace_prompt_identity: bool = False,
                 engine_configuration: str = "current_profile") -> None:
        if not isinstance(spec, ModelSpec):
            raise TypeError("spec must be ModelSpec")
        if startup_timeout is not None and (
            isinstance(startup_timeout, bool)
            or not isinstance(startup_timeout, (int, float))
            or not math.isfinite(startup_timeout)
            or startup_timeout <= 0
        ):
            raise ValueError("startup_timeout must be None or finite and positive")
        self.spec = spec
        if not isinstance(execution_variant, str) or execution_variant not in WORKER_VARIANTS:
            raise ValueError("unknown backend generation variant")
        if any(type(value) is not int or not 1 <= value <= 2**63 - 1
               for value in (prefix_cache_max_entries, prefix_cache_max_bytes)):
            raise ValueError("prefix cache capacity must be a positive bounded integer")
        self.execution_variant = execution_variant
        if not isinstance(engine_configuration, str) or engine_configuration not in ENGINE_CONFIGURATIONS:
            raise ValueError("unknown engine configuration")
        if execution_variant != "current_engine" and engine_configuration != "current_profile":
            raise ValueError("engine configuration requires current_engine")
        self.engine_configuration = engine_configuration
        if type(trace_prompt_identity) is not bool:
            raise ValueError("trace_prompt_identity must be a boolean")
        self.trace_prompt_identity = trace_prompt_identity
        self.prefix_cache_max_entries = prefix_cache_max_entries
        self.prefix_cache_max_bytes = prefix_cache_max_bytes
        self.last_generation_metadata: dict[str, Any] | None = None
        self.startup_timeout = None if startup_timeout is None else float(startup_timeout)
        self._process: subprocess.Popen[bytes] | None = None
        self._ready_payload: dict[str, Any] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buffer = bytearray()
        self._stderr_lock = threading.Lock()
        self._stdout_buffer = bytearray()
        self._lock = threading.RLock()
        self._stream_active = False
        self._usable = True

    @property
    def ready(self) -> bool:
        return self._ready_payload is not None and self._process is not None and self._process.poll() is None and self._usable

    def _drain_stderr(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                with self._stderr_lock:
                    remaining = MAX_STDERR_BYTES - len(self._stderr_buffer)
                    if remaining > 0:
                        self._stderr_buffer.extend(chunk[:remaining])
        except (OSError, ValueError):
            return

    @staticmethod
    def _frame_from_buffer(buffer: bytearray) -> bytes | None:
        newline = buffer.find(b"\n")
        if newline < 0:
            if len(buffer) > MAX_PROTOCOL_LINE:
                raise BackendUnavailable("backend protocol message is too large")
            return None
        if newline > MAX_PROTOCOL_LINE:
            raise BackendUnavailable("backend protocol message is too large")
        frame = bytes(buffer[:newline])
        del buffer[: newline + 1]
        return frame

    def _send(
        self,
        payload: dict[str, Any],
        deadline: float | None | object = _DEFAULT_DEADLINE,
    ) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise BackendUnavailable("backend worker is not running")
        encoded = _bounded_json_line(payload)
        fd = process.stdin.fileno()
        if deadline is _DEFAULT_DEADLINE:
            deadline = (
                None
                if self.startup_timeout is None
                else time.monotonic() + self.startup_timeout
            )
        selector = selectors.DefaultSelector()
        try:
            os.set_blocking(fd, False)
            offset = 0
            while offset < len(encoded):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestTimeout("backend worker command timed out")
                try:
                    written = os.write(fd, encoded[offset:])
                    if written <= 0:
                        raise BrokenPipeError
                    offset += written
                    continue
                except BlockingIOError:
                    selector.register(fd, selectors.EVENT_WRITE)
                    try:
                        wait = _CANCEL_POLL_SECONDS if remaining is None else min(remaining, _CANCEL_POLL_SECONDS)
                        if not selector.select(wait) and remaining is not None and time.monotonic() >= deadline:
                            raise RequestTimeout("backend worker command timed out")
                    finally:
                        selector.unregister(fd)
        except (BrokenPipeError, OSError, ValueError, RequestTimeout) as exc:
            self._mark_unusable()
            if isinstance(exc, RequestTimeout):
                raise
            raise BackendUnavailable("backend worker pipe failed") from exc
        finally:
            selector.close()

    def _read_event(
        self,
        deadline: float | None,
        *,
        cancel: threading.Event | None = None,
        on_cancel: Any = None,
    ) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise BackendUnavailable("backend worker is not running")
        fd = process.stdout.fileno()
        selector = selectors.DefaultSelector()
        try:
            os.set_blocking(fd, False)
            while True:
                # Buffered output does not suspend the caller's deadline or
                # cancellation. In particular, an eager producer must not
                # starve cancellation by keeping this buffer non-empty.
                if deadline is not None and time.monotonic() >= deadline:
                    raise RequestTimeout("backend worker timed out")
                if cancel is not None and cancel.is_set() and on_cancel is not None:
                    on_cancel()
                frame = self._frame_from_buffer(self._stdout_buffer)
                if frame is not None:
                    try:
                        value = json.loads(frame.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise BackendUnavailable("backend emitted invalid protocol data") from exc
                    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                        raise BackendUnavailable("backend emitted an invalid protocol event")
                    return value
                try:
                    while True:
                        capacity = MAX_PROTOCOL_LINE + 1 - len(self._stdout_buffer)
                        if capacity <= 0 and b"\n" not in self._stdout_buffer:
                            raise BackendUnavailable("backend protocol message is too large")
                        chunk = os.read(fd, min(65536, max(1, capacity)))
                        if not chunk:
                            if self._stdout_buffer:
                                raise BackendUnavailable("backend emitted an unterminated protocol frame")
                            raise BackendUnavailable("backend worker exited before completing the protocol")
                        self._stdout_buffer.extend(chunk)
                        if b"\n" in self._stdout_buffer or len(chunk) < 65536:
                            break
                except BlockingIOError:
                    pass
                frame = self._frame_from_buffer(self._stdout_buffer)
                if frame is not None:
                    try:
                        value = json.loads(frame.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise BackendUnavailable("backend emitted invalid protocol data") from exc
                    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
                        raise BackendUnavailable("backend emitted an invalid protocol event")
                    return value
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise RequestTimeout("backend worker timed out")
                if cancel is not None and cancel.is_set() and on_cancel is not None:
                    on_cancel()
                wait = (
                    _CANCEL_POLL_SECONDS
                    if remaining is None
                    else min(remaining, _CANCEL_POLL_SECONDS if cancel is not None else remaining)
                )
                selector.register(fd, selectors.EVENT_READ)
                try:
                    events = selector.select(wait)
                finally:
                    selector.unregister(fd)
                if not events:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RequestTimeout("backend worker timed out")
        finally:
            selector.close()

    @staticmethod
    def _validate_startup_telemetry(event: dict[str, Any]) -> None:
        telemetry_keys = {
            "startup_wall_seconds", "process_peak_rss_bytes", "mlx_active_bytes",
            "mlx_peak_bytes", "mlx_cache_bytes", "recommended_working_set_bytes",
        }
        if not telemetry_keys.intersection(event):
            return  # Legacy transport fixtures may omit telemetry entirely.
        if not telemetry_keys.issubset(event):
            raise BackendUnavailable("backend startup telemetry is incomplete")
        startup = event["startup_wall_seconds"]
        try:
            startup_float = float(startup)
        except (OverflowError, TypeError, ValueError):
            startup_float = math.inf
        if (isinstance(startup, bool) or not isinstance(startup, (int, float))
                or not math.isfinite(startup_float) or startup <= 0):
            raise BackendUnavailable("backend startup telemetry is invalid")
        byte_fields = {key: event[key] for key in telemetry_keys if key != "startup_wall_seconds"}
        if any(type(value) is not int or not 0 <= value <= 2**63 - 1 for value in byte_fields.values()):
            raise BackendUnavailable("backend startup telemetry is invalid")
        if byte_fields["process_peak_rss_bytes"] <= 0 or byte_fields["recommended_working_set_bytes"] <= 0:
            raise BackendUnavailable("backend startup telemetry is invalid")
        if event["mlx_peak_bytes"] < event["mlx_active_bytes"]:
            raise BackendUnavailable("backend startup telemetry is inconsistent")

    def start(self, *, startup_guard: Any = None) -> dict[str, Any]:
        if startup_guard is not None and not callable(startup_guard):
            raise TypeError("startup_guard must be callable or None")
        with self._lock:
            if self.ready:
                if startup_guard is not None:
                    raise ValueError("startup_guard cannot be applied to an already-ready worker")
                assert self._ready_payload is not None
                return dict(self._ready_payload)
            if not self._usable:
                raise BackendUnavailable("backend worker is unusable")
            if self._process is not None:
                self._mark_unusable()
                raise BackendUnavailable("backend worker did not become ready")
            worker = Path(__file__).with_name("worker.py")
            environment = os.environ.copy()
            environment.update({
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            })
            try:
                arguments = [sys.executable, "-I", "-u", str(worker), "--spec",
                             json.dumps(self.spec.as_dict(), separators=(",", ":"))]
                if self.execution_variant != "reference":
                    arguments += ["--execution-variant", self.execution_variant]
                if self.execution_variant == "prefix_reuse":
                    arguments += ["--prefix-cache-max-entries", str(self.prefix_cache_max_entries),
                                  "--prefix-cache-max-bytes", str(self.prefix_cache_max_bytes)]
                if self.execution_variant == "current_engine":
                    arguments += ["--engine-configuration", self.engine_configuration]
                self._process = subprocess.Popen(
                    arguments,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                )
            except (OSError, ValueError) as exc:
                self._mark_unusable()
                raise BackendUnavailable("backend worker could not start") from exc
            assert self._process.stderr is not None
            self._stdout_buffer.clear()
            self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(self._process.stderr,), daemon=True)
            self._stderr_thread.start()
            startup_deadline = (
                None
                if self.startup_timeout is None
                else time.monotonic() + self.startup_timeout
            )
            def call_startup_guard() -> None:
                try:
                    startup_guard(self._process.pid)
                except BaseException as exc:
                    raise _StartupGuardFailure(exc) from exc
            try:
                if startup_guard is None:
                    event = self._read_event(startup_deadline)
                else:
                    call_startup_guard()
                    while True:
                        poll_deadline = time.monotonic() + _CANCEL_POLL_SECONDS
                        if startup_deadline is not None:
                            poll_deadline = min(startup_deadline, poll_deadline)
                        try:
                            event = self._read_event(poll_deadline)
                        except RequestTimeout:
                            call_startup_guard()
                            if startup_deadline is not None and time.monotonic() >= startup_deadline:
                                raise
                            continue
                        call_startup_guard()
                        if startup_deadline is not None and time.monotonic() >= startup_deadline:
                            raise RequestTimeout("backend worker startup timed out")
                        break
            except (BackendUnavailable, RequestTimeout):
                self._mark_unusable()
                raise BackendUnavailable("stock MLX backend is unavailable")
            except _StartupGuardFailure as failure:
                try:
                    self._mark_unusable()
                except BaseException:
                    pass
                raise failure.original
            except BaseException:
                # A guard failure is caller-owned evidence (often a memory
                # limit); never replace it with a generic backend error.
                try:
                    self._mark_unusable()
                except BaseException:
                    pass
                raise
            if event.get("type") == "error":
                self._mark_unusable()
                raise BackendUnavailable("stock MLX backend is unavailable")
            if (
                event.get("type") != "ready"
                or event.get("protocol_version") != PROTOCOL_VERSION
                or event.get("device") != "gpu"
                or event.get("model_id") != self.spec.model_id
                or event.get("revision") != self.spec.revision
                or event.get("stop_handling") != "parent"
                or event.get("context_limit", CONTEXT_LIMIT) != CONTEXT_LIMIT
                or event.get("execution_variant", "reference") != self.execution_variant
            ):
                self._mark_unusable()
                raise BackendUnavailable("backend readiness event is invalid")
            try:
                self._validate_startup_telemetry(event)
            except BackendUnavailable:
                self._mark_unusable()
                raise
            self._ready_payload = event
            return dict(event)

    def _mark_unusable(self) -> None:
        self._usable = False
        process, self._process = self._process, None
        self._ready_payload = None
        self._stdout_buffer.clear()
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for stream in (process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    self._send({"type": "shutdown"}, time.monotonic() + 1.0)
                    process.wait(timeout=2.0)
                except (BackendUnavailable, RequestTimeout, subprocess.TimeoutExpired, OSError):
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2.0)
            self._mark_unusable()

    @staticmethod
    def _validate_prefix_metadata(value: Any) -> None:
        if not isinstance(value, dict) or set(value) != {"status", "commit_status", "trace", "stats"}:
            raise BackendUnavailable("backend prefix metadata is invalid")
        trace, stats = value["trace"], value["stats"]
        if not isinstance(value["status"], str) or value["status"] not in {"accepted", "discarded", "rejected", "cleared", "closed", "idle"}:
            raise BackendUnavailable("backend prefix metadata is invalid")
        if not isinstance(value["commit_status"], str) or not 1 <= len(value["commit_status"]) <= 128:
            raise BackendUnavailable("backend prefix metadata is invalid")
        trace_fields = {"status", "failure_type", "cache_hit", "cache_stored", "reused_tokens",
                        "fallback_reason", "restore_host_ns", "capture_host_ns", "commit_host_ns"}
        stat_fields = {"hits", "misses", "inserts", "skipped_oversize", "evictions", "entries", "bytes"}
        if not isinstance(trace, dict) or set(trace) != trace_fields or not isinstance(stats, dict) or set(stats) != stat_fields:
            raise BackendUnavailable("backend prefix metadata is invalid")
        if (any(type(trace[k]) is not bool for k in ("cache_hit", "cache_stored"))
                or any(not isinstance(trace[k], str) or not 1 <= len(trace[k]) <= 128
                       for k in ("status", "failure_type", "fallback_reason"))
                or any(type(trace[k]) is not int or not 0 <= trace[k] <= 2**63 - 1
                       for k in ("reused_tokens", "restore_host_ns", "capture_host_ns", "commit_host_ns"))
                or any(type(v) is not int or not 0 <= v <= 2**63 - 1 for v in stats.values())):
            raise BackendUnavailable("backend prefix metadata is invalid")

    def clear_prefix_cache(self) -> dict[str, Any]:
        """Reset an already-loaded opt-in session without loading a model."""
        with self._lock:
            if self.execution_variant != "prefix_reuse" or not self.ready:
                raise BackendUnavailable("prefix cache is not active")
            if self._stream_active:
                raise BackendUnavailable("cannot clear cache during an active request")
            request_id = "cache-clear-" + uuid.uuid4().hex
            try:
                self._send({"type": "clear_prefix_cache", "request_id": request_id}, None)
                event = self._read_event(None)
                self._validate_prefix_metadata(event.get("prefix_cache"))
                metadata = event["prefix_cache"]
                if (event.get("type") != "prefix_cache_cleared" or event.get("request_id") != request_id
                        or metadata["status"] != "cleared" or metadata["stats"]["entries"] != 0
                        or metadata["stats"]["bytes"] != 0 or self._stdout_buffer):
                    raise BackendUnavailable("prefix cache reset was not confirmed")
                return metadata
            except BaseException:
                self._mark_unusable()
                raise

    def stream(
        self,
        request: GenerationRequest,
        cancel: threading.Event | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
        *,
        variant: str | None = None,
        trace_forwards: bool = False,
    ) -> Iterator[dict[str, Any]]:
        if not isinstance(request, GenerationRequest):
            raise InvalidRequest("request must be GenerationRequest")
        if request.model != self.spec.model_id:
            raise InvalidRequest("model is not registered for this backend")
        variant = self.execution_variant if variant is None else variant
        if not isinstance(variant, str) or variant not in WORKER_VARIANTS:
            raise InvalidRequest("unknown backend generation variant")
        allowed = ({"current_engine"} if self.execution_variant == "current_engine" else
                   {"reference", "prefix_reuse"} if self.execution_variant == "prefix_reuse" else
                   {"reference", "bounded_prefetch"})
        if variant not in allowed:
            raise InvalidRequest("variant is not enabled for this worker")
        if type(trace_forwards) is not bool:
            raise InvalidRequest("trace_forwards must be a boolean")
        if not isinstance(request.request_id, str) or not request.request_id or len(request.request_id) > 256:
            raise InvalidRequest("request_id must be a bounded non-empty string")
        payload = request.as_dict()
        payload.pop("request_id", None)
        GenerationRequest.from_payload(payload, exact=True)
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be None or finite and positive")
        if cancel is not None and not isinstance(cancel, threading.Event):
            raise TypeError("cancel must be threading.Event or None")
        command = {"type": "generate", **request.as_dict(), "variant": variant,
                   "trace_forwards": trace_forwards, "trace_prompt_identity": self.trace_prompt_identity}
        try:
            _bounded_json_line(command)
        except BackendUnavailable as exc:
            # Encoding/size failures belong to the request, not the loaded
            # model. Reject before starting or retiring any worker.
            raise InvalidRequest("request exceeds the backend protocol boundary") from exc

        def events() -> Iterator[dict[str, Any]]:
            with self._lock:
                if self._stream_active:
                    raise BackendUnavailable("backend worker supports one active request")
                self._stream_active = True
                self.last_generation_metadata = None
            completed = False
            cancel_sent = False
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            token_count = 0
            prompt_tokens: int | None = None

            def request_cancel() -> None:
                nonlocal cancel_sent
                if not cancel_sent:
                    self._send({"type": "cancel", "request_id": request.request_id}, deadline)
                    cancel_sent = True
            try:
                if not self.ready:
                    self.start()
                if cancel is not None and cancel.is_set():
                    completed = True
                    yield {"type": "done", "request_id": request.request_id, "finish_reason": "cancelled",
                           "prompt_tokens": 0, "completion_tokens": 0, "variant": variant,
                           **({"model_forward_invocations": 0} if trace_forwards else {})}
                    return
                self._send(command, deadline)
                while True:
                    try:
                        event = self._read_event(deadline, cancel=cancel, on_cancel=request_cancel)
                    except RequestTimeout:
                        self._mark_unusable()
                        raise RequestTimeout("generation timed out")
                    except BackendUnavailable:
                        self._mark_unusable()
                        raise
                    if event.get("request_id") != request.request_id:
                        self._mark_unusable()
                        raise BackendUnavailable("backend response request id mismatch")
                    kind = event.get("type")
                    if kind == "token":
                        text = event.get("text", "")
                        token_id = event.get("token_id")
                        event_prompt = event.get("prompt_tokens")
                        event_completion = event.get("completion_tokens")
                        if (not isinstance(text, str) or type(token_id) is not int or token_id < 0
                                or type(event_prompt) is not int or event_prompt < 0
                                or type(event_completion) is not int or event_completion != token_count + 1
                                or event_completion > request.max_tokens
                                or (prompt_tokens is not None and event_prompt != prompt_tokens)):
                            self._mark_unusable()
                            raise BackendUnavailable("backend token event is invalid")
                        prompt_tokens = event_prompt if prompt_tokens is None else prompt_tokens
                        token_count += 1
                        yield event
                    elif kind == "done":
                        finish = event.get("finish_reason")
                        event_prompt = event.get("prompt_tokens")
                        event_completion = event.get("completion_tokens")
                        metrics = event.get("metrics", {})
                        forward_count = event.get("model_forward_invocations")
                        if (finish not in ("stop", "length", "cancelled")
                                or type(event_prompt) is not int or event_prompt < 0
                                or type(event_completion) is not int or not 0 <= event_completion <= request.max_tokens
                                or (prompt_tokens is not None and event_prompt != prompt_tokens)
                                or event_completion < token_count
                                or event_completion > token_count + 1
                                or not isinstance(metrics, dict)
                                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                                       or not math.isfinite(float(value))
                                       for value in metrics.values())
                                or (not cancel_sent and finish == "cancelled")):
                            self._mark_unusable()
                            raise BackendUnavailable("backend completion event is invalid")
                        prompt_tokens = event_prompt if prompt_tokens is None else prompt_tokens
                        if event.get("variant") != variant:
                            self._mark_unusable()
                            raise BackendUnavailable("backend completion variant mismatch")
                        if trace_forwards and (type(forward_count) is not int or forward_count < 0):
                            self._mark_unusable()
                            raise BackendUnavailable("backend forward trace is invalid")
                        if self._stdout_buffer:
                            self._mark_unusable()
                            raise BackendUnavailable("backend emitted data after completion")
                        if variant == "prefix_reuse":
                            try:
                                self._validate_prefix_metadata(event.get("prefix_cache"))
                            except BackendUnavailable:
                                self._mark_unusable()
                                raise
                            if (event["prefix_cache"]["stats"]["bytes"] > self.prefix_cache_max_bytes
                                    or event["prefix_cache"]["stats"]["entries"] > self.prefix_cache_max_entries):
                                self._mark_unusable()
                                raise BackendUnavailable("backend prefix cache accounting is invalid")
                        prompt_digest = event.get("prompt_ids_sha256")
                        if self.trace_prompt_identity and (not isinstance(prompt_digest, str)
                                or len(prompt_digest) != 64 or any(c not in "0123456789abcdef" for c in prompt_digest)):
                            self._mark_unusable()
                            raise BackendUnavailable("backend prompt identity is invalid")
                        self.last_generation_metadata = {
                            "variant": variant, "finish_reason": finish,
                            "prompt_ids_sha256": prompt_digest,
                            "prefix_cache": event.get("prefix_cache"),
                            "engine": event.get("engine"),
                        }
                        completed = True
                        yield event
                        return
                    elif kind == "error":
                        code = event.get("code")
                        if not isinstance(code, str) or not code:
                            self._mark_unusable()
                            raise BackendUnavailable("backend error event is invalid")
                        if code == "invalid_request":
                            # A rejected user request is not a transport or
                            # model failure; retain the warm worker.
                            completed = True
                            raise InvalidRequest("worker rejected the generation request")
                        if code == "variant_unavailable":
                            completed = True
                            raise BackendUnavailable("requested generation variant is unavailable")
                        self._mark_unusable()
                        raise BackendUnavailable("stock MLX backend failed")
                    else:
                        self._mark_unusable()
                        raise BackendUnavailable("backend emitted an unknown event")
            finally:
                with self._lock:
                    self._stream_active = False
                if not completed:
                    # Closing a generator early leaves unread protocol data in
                    # the pipe.  The persistent model is therefore discarded;
                    # this client never restarts it implicitly.
                    self._mark_unusable()

        return events()


__all__ = ["MLXWorkerClient", "MAX_PROTOCOL_LINE", "MAX_STDERR_BYTES", "PROTOCOL_VERSION"]
