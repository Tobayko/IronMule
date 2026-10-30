"""Process-isolated stock ``mlx_lm`` backend for the local product API.

The parent process intentionally imports no MLX or model code.  A worker owns
one loaded model and communicates through a bounded JSON-lines protocol.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from .errors import BackendUnavailable, InvalidRequest, ProductError, RequestTimeout
from .types import GenerationRequest, ModelSpec


PROTOCOL_VERSION = 2  # 2: a grouped request too long to run is rejected alone (`rejected` frame)
MAX_PROTOCOL_LINE = 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024
STDERR_TAIL_BYTES = 4096  # STAB1: what a worker exit record keeps of the worker's last stderr
DEFAULT_TIMEOUT = 120.0
DENSE_STARTUP_TIMEOUT = 600.0  # KB1: the dense plan compiles and probes every kernel shape before it is ready
_CANCEL_POLL_SECONDS = 0.05
_DEFAULT_DEADLINE = object()
STREAM_GROUP_REQUESTS = 8  # admitted together and prefilled at once; decode rotates at width 4
WORKER_VARIANTS = frozenset(("reference", "bounded_prefetch", "prefix_reuse", "current_engine", "automatic"))
ENGINE_CONFIGURATIONS = frozenset(("current_profile", "baseline_interactive", "core_interactive",
                                   "baseline_throughput", "core_throughput", "rust2", "dense_stacked"))
CONTEXT_LIMIT = 8192
# The worker rejects a grouped request alone when its prompt and output limit exceed the context limit or the
# engine's own bound (KB1: the dense plan's slot).
REJECTION_REASONS = frozenset({"context_limit", "engine_limit"})


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
    """A single persistent stock ``mlx_lm`` worker (GPU, else MLX's CPU device)."""

    def __init__(self, spec: ModelSpec, *, startup_timeout: float | None = DEFAULT_TIMEOUT,
                 execution_variant: str = "reference", prefix_cache_max_entries: int = 4,
                 prefix_cache_max_bytes: int = 1024**3,
                 trace_prompt_identity: bool = False,
                 engine_configuration: str = "current_profile",
                 selection_evidence: list[Any] | tuple[Any, ...] | None = None,
                 compute_dtype: str | None = None, chat_reuse: bool = False, thinking: bool = True,
                 cuda_device: str | None = None, engine_knobs: dict[str, Any] | None = None) -> None:
        if not isinstance(spec, ModelSpec):
            raise TypeError("spec must be ModelSpec")
        # HW1: the one CUDA device this worker may see (an index or UUID as CUDA_VISIBLE_DEVICES takes
        # it); None leaves the inherited environment alone, as before.
        if cuda_device is not None and (not isinstance(cuda_device, str)
                                        or re.fullmatch(r"[A-Za-z0-9-]{1,64}", cuda_device) is None):
            raise ValueError("cuda_device must be one CUDA device index or UUID")
        self.cuda_device = cuda_device
        if compute_dtype not in (None, "float32", "native", "dequantize"):
            raise ValueError("compute_dtype must be None, float32, native or dequantize")
        if compute_dtype is not None and execution_variant != "reference":
            raise ValueError("compute_dtype is available on the reference worker only")
        # Opt-in numeric plan (PORT1): changes output, reported in health.
        self.compute_dtype = compute_dtype
        if chat_reuse and execution_variant != "current_engine":
            raise ValueError("chat reuse is available on the current_engine worker only")
        self.chat_reuse = bool(chat_reuse)  # SERVE7: opt-in ChatTurnsPlan, changes output vs strict
        self.thinking = bool(thinking)  # CPU5: False renders the chat template with thinking off
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
        # KP1: class-A knobs for the tuned engine when no stored tune profile applies, as
        # {"knobs": {...}, "source": label}; they leave every answer unchanged.
        if engine_knobs is not None:
            from .knob_profile import class_a_knobs

            if (execution_variant != "current_engine" or engine_configuration != "current_profile"
                    or not isinstance(engine_knobs, dict) or set(engine_knobs) != {"knobs", "source"}
                    or class_a_knobs(engine_knobs["knobs"]) is None
                    or not isinstance(engine_knobs["source"], str)
                    or re.fullmatch(r"[A-Za-z0-9:._-]{1,64}", engine_knobs["source"]) is None):
                raise ValueError("engine_knobs must be class-A knobs with a source, for the tuned engine only")
            engine_knobs = json.loads(json.dumps(engine_knobs))
        self.engine_knobs = engine_knobs
        if type(trace_prompt_identity) is not bool:
            raise ValueError("trace_prompt_identity must be a boolean")
        self.trace_prompt_identity = trace_prompt_identity
        self.prefix_cache_max_entries = prefix_cache_max_entries
        self.prefix_cache_max_bytes = prefix_cache_max_bytes
        if selection_evidence is None:
            selection_evidence = []
        if not isinstance(selection_evidence, (list, tuple)) or len(selection_evidence) > 256:
            raise ValueError("selection_evidence must be a bounded list of mappings")
        try:
            evidence_json = json.dumps(list(selection_evidence), ensure_ascii=False, separators=(",", ":"),
                                       allow_nan=False)
            evidence_copy = json.loads(evidence_json)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("selection_evidence must contain bounded JSON records") from exc
        if (not isinstance(evidence_copy, list) or any(not isinstance(row, dict) for row in evidence_copy)
                or len(evidence_json.encode("utf-8")) >= MAX_PROTOCOL_LINE):
            raise ValueError("selection_evidence must contain bounded JSON records")
        self.selection_evidence = evidence_copy
        self.last_generation_metadata: dict[str, Any] | None = None
        self.last_batch_metadata: list[dict[str, Any]] | None = None
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
        # STAB1: why and how the worker process ended (reason, exit status, stderr tail); None while it runs.
        self.last_exit: dict[str, Any] | None = None
        self.last_memory: dict[str, int] | None = None  # MEM2: the worker's MLX counters at its last group end

    @property
    def ready(self) -> bool:
        return self._ready_payload is not None and self._process is not None and self._process.poll() is None and self._usable

    @property
    def device(self) -> str | None:
        """The device the ready worker reported (OPS1): gpu or cpu; None before readiness."""
        payload = self._ready_payload
        return None if payload is None else payload.get("device")

    @property
    def ready_payload(self) -> dict[str, Any] | None:
        """A copy of the worker's validated ready event; it holds no request data."""
        payload = self._ready_payload
        return None if payload is None else json.loads(json.dumps(payload))

    @property
    def batch_capacity(self) -> int:
        """Configured request capacity, not proof of a qualified GPU group width."""
        if (self.execution_variant == "current_engine"
                and self.engine_configuration in {"baseline_throughput", "core_throughput"}):
            return 32
        if self.execution_variant == "current_engine" and self.engine_configuration in ("current_profile",
                                                                                        "dense_stacked"):
            return STREAM_GROUP_REQUESTS  # SERVE4: simultaneous streams share one grouped engine session
        if (self.execution_variant == "automatic" and self.ready and self._ready_payload is not None
                and self._ready_payload.get("automatic_batch_qualified") is True):
            return 32
        return 1

    @property
    def reuses_slots(self) -> bool:
        """KB1: whether a member that ends frees room a newcomer may take while its group runs (the dense
        plan's slot pool); every other engine counts members over the group's life."""
        return self.execution_variant == "current_engine" and self.engine_configuration == "dense_stacked"

    @property
    def streams_batches(self) -> bool:
        """Whether grouped requests may stream (the engine-stream worker path, SERVE4)."""
        return self.execution_variant == "current_engine" and self.engine_configuration in ("current_profile",
                                                                                            "dense_stacked")

    def can_batch_requests(self, requests: Any) -> bool:
        """Check metadata admission to the opt-in transport without model execution."""
        least = 1 if self.streams_batches else 2  # SERVE8: a lone stream runs as a group others can join
        if not isinstance(requests, (list, tuple)) or not least <= len(requests) <= self.batch_capacity:
            return False
        ids: set[str] = set()
        commands = []
        try:
            for request in requests:
                if not isinstance(request, GenerationRequest) or request.model != self.spec.model_id:
                    return False
                if request.stream and not self.streams_batches:
                    return False
                if (not isinstance(request.request_id, str) or not request.request_id
                        or len(request.request_id) > 256 or request.request_id in ids):
                    return False
                ids.add(request.request_id)
                payload = request.as_dict()
                payload.pop("request_id", None)
                GenerationRequest.from_payload(payload, exact=True)
                commands.append(request.as_dict())
            _bounded_json_line({"type": "generate_batch", "batch_id": "batch-" + "b" * 32,
                                "requests": commands, "variant": self.execution_variant,
                                "trace_prompt_identity": self.trace_prompt_identity})
        except (BackendUnavailable, InvalidRequest, TypeError, ValueError):
            return False
        return True

    def _drain_stderr(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    return
                with self._stderr_lock:  # keep the newest bytes: a final error is what explains an exit
                    self._stderr_buffer.extend(chunk)
                    del self._stderr_buffer[:-MAX_STDERR_BYTES]
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
        on_tick: Any = None,
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
                if on_tick is not None:
                    on_tick()
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
                if on_tick is not None:
                    on_tick()
                wait = (
                    _CANCEL_POLL_SECONDS
                    if remaining is None
                    else min(remaining, _CANCEL_POLL_SECONDS if cancel is not None or on_tick is not None else remaining)
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
            if self.cuda_device is not None:
                environment["CUDA_VISIBLE_DEVICES"] = self.cuda_device
            try:
                arguments = [sys.executable, "-I", "-u", str(worker), "--spec",
                             json.dumps(self.spec.as_dict(), separators=(",", ":"))]
                if self.execution_variant != "reference":
                    arguments += ["--execution-variant", self.execution_variant]
                if self.compute_dtype is not None:
                    arguments += ["--compute-dtype", self.compute_dtype]
                if not self.thinking:
                    arguments += ["--no-thinking"]
                if self.execution_variant == "prefix_reuse":
                    arguments += ["--prefix-cache-max-entries", str(self.prefix_cache_max_entries),
                                  "--prefix-cache-max-bytes", str(self.prefix_cache_max_bytes)]
                if self.execution_variant == "current_engine":
                    arguments += ["--engine-configuration", self.engine_configuration]
                    if self.chat_reuse:
                        arguments += ["--chat-reuse"]
                    if self.engine_knobs is not None:
                        arguments += ["--engine-knobs", json.dumps(self.engine_knobs, sort_keys=True,
                                                                   separators=(",", ":"))]
                if self.execution_variant == "automatic":
                    arguments += ["--selection-evidence", json.dumps(
                        self.selection_evidence, ensure_ascii=False, separators=(",", ":"), allow_nan=False)]
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
                or event.get("device") not in ("gpu", "cpu")  # OPS1: the actual device, validated
                or event.get("model_id") != self.spec.model_id
                or event.get("revision") != self.spec.revision
                or event.get("stop_handling") != "parent"
                or event.get("context_limit", CONTEXT_LIMIT) != CONTEXT_LIMIT
                or event.get("execution_variant", "reference") != self.execution_variant
            ):
                self._mark_unusable()
                raise BackendUnavailable("backend readiness event is invalid")
            digest = event.get("sources_sha256")
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest)):
                self._mark_unusable()
                raise BackendUnavailable("backend source identity is invalid")
            if self.execution_variant == "automatic" and type(event.get("automatic_batch_qualified")) is not bool:
                self._mark_unusable()
                raise BackendUnavailable("backend automatic readiness is invalid")
            try:
                self._validate_startup_telemetry(event)
            except BackendUnavailable:
                self._mark_unusable()
                raise
            self._ready_payload = event
            return dict(event)

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr_buffer[-STDERR_TAIL_BYTES:]).decode("utf-8", "replace")

    def note_exit_reason(self, error: BaseException) -> None:
        """STAB1: name the failure that retired the worker when the retiring call did not."""
        if self.last_exit is not None and self.last_exit.get("reason") is None:
            self.last_exit["reason"] = _reason(error)

    def _mark_unusable(self, reason: str | None = None) -> None:
        self._usable = False
        process, self._process = self._process, None
        self._ready_payload = None
        self._stdout_buffer.clear()
        if process is None:
            return
        if reason is None and (current := sys.exc_info()[1]) is not None:
            reason = _reason(current)
        try:
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
        finally:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1.0)  # the worker's last words reach the buffer
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            self.last_exit = {"reason": reason, "pid": process.pid, "exit_status": process.poll(),
                              "reaped": process.returncode is not None, "stderr_tail": self.stderr_tail(),
                              "unix": time.time()}

    def close(self, reason: str = "closed") -> None:
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
            self._mark_unusable(reason)

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

    @staticmethod
    def _valid_selection_metadata(value: Any) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "candidate_id", "reason", "evidence_audit_id", "explored",
            "rejected_evidence_records", "actual_backend", "actual_group_width",
        }:
            return False
        return (isinstance(value["candidate_id"], str) and 1 <= len(value["candidate_id"]) <= 64
                and isinstance(value["reason"], str) and 1 <= len(value["reason"]) <= 128
                and (value["evidence_audit_id"] is None or isinstance(value["evidence_audit_id"], str)
                     and 1 <= len(value["evidence_audit_id"]) <= 256)
                and type(value["explored"]) is bool
                and type(value["rejected_evidence_records"]) is int
                and 0 <= value["rejected_evidence_records"] <= 256
                and isinstance(value["actual_backend"], str) and 1 <= len(value["actual_backend"]) <= 64
                and type(value["actual_group_width"]) is int and 1 <= value["actual_group_width"] <= 32)

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

    def complete_batch(
        self,
        requests: list[GenerationRequest] | tuple[GenerationRequest, ...],
        *,
        cancels: list[threading.Event | None] | tuple[threading.Event | None, ...] | None = None,
        timeout: float | None = None,
        on_event=None,
        joins=None,
    ) -> list[dict[str, Any]]:
        """Complete one real grouped engine session and return buffered events.

        ``on_event(request_id, event)`` also receives each validated token and done event as
        it arrives, so a streaming caller need not wait for the whole group (SERVE4)."""
        if not self.can_batch_requests(requests):
            raise InvalidRequest("requests are not eligible for grouped execution")
        request_list = list(requests)
        initial = len(request_list)
        if cancels is None:
            cancel_list: list[threading.Event | None] = [None] * len(request_list)
        elif (not isinstance(cancels, (list, tuple)) or len(cancels) != len(request_list)
              or any(value is not None and not isinstance(value, threading.Event) for value in cancels)):
            raise TypeError("cancels must match requests and contain threading.Event or None")
        else:
            cancel_list = list(cancels)
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be None or finite and positive")
        batch_id = "batch-" + uuid.uuid4().hex
        command = {"type": "generate_batch", "batch_id": batch_id,
                   "requests": [request.as_dict() for request in request_list],
                   "variant": self.execution_variant, "trace_prompt_identity": self.trace_prompt_identity}
        try:
            _bounded_json_line(command)
        except BackendUnavailable as exc:
            raise InvalidRequest("batch exceeds the backend protocol boundary") from exc

        with self._lock:
            if self._stream_active:
                raise BackendUnavailable("backend worker supports one active operation")
            self._stream_active = True
            self.last_generation_metadata = None
            self.last_batch_metadata = None
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            request_by_id = {request.request_id: request for request in request_list}
            events_by_id: dict[str, list[dict[str, Any]]] = {request.request_id: [] for request in request_list}
            token_counts = {request.request_id: 0 for request in request_list}
            prompt_counts: dict[str, int | None] = {request.request_id: None for request in request_list}
            terminals: dict[str, dict[str, Any]] = {}
            cancel_sent: set[str] = set()
            dispatched = False

            def poll_cancels() -> None:
                for request, cancel in zip(request_list, cancel_list):
                    if (cancel is not None and cancel.is_set() and request.request_id not in cancel_sent
                            and request.request_id not in terminals):  # an ended request has nothing to stop
                        self._send({"type": "cancel", "request_id": request.request_id}, deadline)
                        cancel_sent.add(request.request_id)
                send_joins()

            def send_joins() -> None:
                """SERVE8: requests the caller hands over while this batch runs join it."""
                fresh = list(joins()) if joins is not None and dispatched else []
                if not fresh:
                    return
                for request, cancel in fresh:
                    if (request.request_id in request_by_id
                            or not self.can_batch_requests([request_list[0], request])):
                        raise InvalidRequest("request is not eligible to join this batch")
                    request_list.append(request)
                    cancel_list.append(cancel)
                    request_by_id[request.request_id] = request
                    events_by_id[request.request_id] = []
                    token_counts[request.request_id] = 0
                    prompt_counts[request.request_id] = None
                self._send({"type": "join_batch", "batch_id": batch_id,
                            "requests": [request.as_dict() for request, _ in fresh]}, deadline)

            # The operation flag is protected by the lock, but inference is
            # not: close() must remain able to acquire the lock and terminate
            # this owned child while the worker is computing.
            self._lock.release()
            try:
                if not self.ready:
                    self.start()
                self._send(command, deadline)
                dispatched = True
                poll_cancels()
                while True:
                    try:
                        event = self._read_event(deadline, on_tick=poll_cancels)
                    except RequestTimeout:
                        self._mark_unusable()
                        raise RequestTimeout("batch generation timed out")
                    except BackendUnavailable:
                        self._mark_unusable()
                        raise
                    poll_cancels()
                    kind = event.get("type")
                    if kind == "batch_done":
                        ids = event.get("request_ids")
                        # A join sent while the worker was already ending the run arrives after it: the worker
                        # drops it, so it is missing here with nothing streamed, and goes back to the caller.
                        unserved = {r.request_id for r in request_list[initial:]
                                    if isinstance(ids, list) and r.request_id not in ids
                                    and not events_by_id[r.request_id] and r.request_id not in terminals}
                        if (event.get("batch_id") != batch_id
                                or ids != [r.request_id for r in request_list if r.request_id not in unserved]
                                or set(terminals) | unserved != set(request_by_id) or self._stdout_buffer):
                            raise BackendUnavailable("backend batch completion is invalid")
                        memory = event.get("memory")
                        self.last_memory = ({k: v for k, v in memory.items() if type(v) is int and v >= 0}
                                            if isinstance(memory, dict) else None)
                        rows = [{"request_id": request.request_id, "events": [], "metadata": None, "unserved": True}
                                if request.request_id in unserved else
                                {"request_id": request.request_id,
                                 "events": events_by_id[request.request_id],
                                 "metadata": terminals[request.request_id]}
                                for request in request_list]
                        self.last_batch_metadata = [dict(row["metadata"]) for row in rows if not row.get("unserved")]
                        return rows
                    if kind == "error":  # the worker's own report names the failure (STAB1: it was misread)
                        code = event.get("code")
                        raise BackendUnavailable("grouped engine generation failed"
                                                 + (f" ({code[:64]})" if isinstance(code, str) else ""))
                    request_id = event.get("request_id")
                    if event.get("batch_id") != batch_id or request_id not in request_by_id or request_id in terminals:
                        raise BackendUnavailable("backend batch response identity is invalid")
                    request = request_by_id[request_id]
                    if kind == "token":
                        text, token_id = event.get("text"), event.get("token_id")
                        prompt, completion = event.get("prompt_tokens"), event.get("completion_tokens")
                        if (not isinstance(text, str) or type(token_id) is not int or token_id < 0
                                or type(prompt) is not int or prompt < 0 or type(completion) is not int
                                or completion != token_counts[request_id] + 1 or completion > request.max_tokens
                                or (prompt_counts[request_id] is not None and prompt != prompt_counts[request_id])):
                            raise BackendUnavailable("backend batch token event is invalid")
                        prompt_counts[request_id] = prompt
                        token_counts[request_id] += 1
                        events_by_id[request_id].append(event)
                        if on_event is not None:
                            on_event(request_id, event)
                    elif kind == "done":
                        finish, prompt, completion = (event.get("finish_reason"), event.get("prompt_tokens"),
                                                       event.get("completion_tokens"))
                        engine = event.get("engine")
                        digest = event.get("prompt_ids_sha256")
                        if (finish not in {"stop", "length", "cancelled"} or type(prompt) is not int or prompt < 0
                                or type(completion) is not int or not token_counts[request_id] <= completion <= request.max_tokens
                                or (finish != "cancelled" and completion > token_counts[request_id] + 1)
                                or (prompt_counts[request_id] is not None and prompt != prompt_counts[request_id])
                                or event.get("variant") != self.execution_variant or not isinstance(engine, dict)
                                or (finish == "cancelled" and request_id not in cancel_sent)):
                            raise BackendUnavailable("backend batch completion event is invalid")
                        if self.trace_prompt_identity and (not isinstance(digest, str) or len(digest) != 64
                                or any(char not in "0123456789abcdef" for char in digest)):
                            raise BackendUnavailable("backend batch prompt identity is invalid")
                        events_by_id[request_id].append(event)
                        if on_event is not None:
                            on_event(request_id, event)
                        selection = event.get("selection")
                        if self.execution_variant == "automatic" and not self._valid_selection_metadata(selection):
                            raise BackendUnavailable("backend automatic selection metadata is invalid")
                        terminals[request_id] = {"variant": self.execution_variant, "finish_reason": finish,
                                                 "prompt_ids_sha256": digest, "engine": engine,
                                                 "selection": selection,
                                                 "delivery": "buffered_completion"}
                    elif kind == "rejected":
                        # Too long to run: the worker rejected this request alone, before any token, and the
                        # others go on. Its caller reports it as the single path reports the same request.
                        reason, limit = event.get("reason"), event.get("limit")
                        if (event.get("code") != "invalid_request" or reason not in REJECTION_REASONS
                                or type(limit) is not int or not 1 <= limit <= CONTEXT_LIMIT
                                or events_by_id[request_id]):
                            raise BackendUnavailable("backend batch rejection event is invalid")
                        events_by_id[request_id].append(event)
                        if on_event is not None:
                            on_event(request_id, event)
                        terminals[request_id] = {"variant": self.execution_variant, "finish_reason": None,
                                                 "rejected": reason, "limit": limit, "engine": None,
                                                 "delivery": "rejected"}
                    else:
                        raise BackendUnavailable("backend emitted an unknown batch event")
            except BaseException as exc:
                self.note_exit_reason(exc)
                self.last_generation_metadata = {
                    "batch_id": batch_id,
                    "partial": [{"request_id": request.request_id,
                                 "events": list(events_by_id[request.request_id]),
                                 "metadata": terminals.get(request.request_id)}
                                for request in request_list]
                }
                self.last_batch_metadata = [dict(terminals[request.request_id])
                                            for request in request_list if request.request_id in terminals]
                if dispatched:
                    self._mark_unusable()
                    self.note_exit_reason(exc)
                raise
            finally:
                self._lock.acquire()
                self._stream_active = False

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
                   {"automatic"} if self.execution_variant == "automatic" else
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
                        selection = event.get("selection")
                        if variant == "automatic" and not self._valid_selection_metadata(selection):
                            self._mark_unusable()
                            raise BackendUnavailable("backend automatic selection metadata is invalid")
                        self.last_generation_metadata = {
                            "variant": variant, "finish_reason": finish,
                            "prompt_ids_sha256": prompt_digest,
                            "prefix_cache": event.get("prefix_cache"),
                            "engine": event.get("engine"),
                            "selection": selection,
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
            except BaseException as exc:
                self.note_exit_reason(exc)
                raise
            finally:
                with self._lock:
                    self._stream_active = False
                if not completed:
                    # Closing a generator early leaves unread protocol data in
                    # the pipe.  The persistent model is therefore discarded;
                    # this client never restarts it implicitly.
                    self._mark_unusable()

        return events()


def _reason(error: BaseException) -> str:
    """A bounded failure name: product errors carry this module's own messages, others only their type."""
    return (f"{type(error).__name__}: {error}" if isinstance(error, ProductError) else type(error).__name__)[:200]


__all__ = ["MLXWorkerClient", "MAX_PROTOCOL_LINE", "MAX_STDERR_BYTES", "PROTOCOL_VERSION"]
