"""Bounded product scheduling around a persistent, isolated model backend.

Only counters and timings are retained by the service. Prompt and response text
exist solely in the in-flight request and are never written to product state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import queue
import threading
import time
from typing import Any, Iterator

from .errors import (BackendUnavailable, InvalidRequest, StateError,
                     Overloaded, ProductError, RequestCancelled, RequestTimeout)
from .serving_policy import ServingLog, ServingPolicy, load_policy

STABILITY_LOG = "stability-log.jsonl"  # STAB1: admission, groups, ends, worker exits; bounded like the serving log
MAX_RESTARTS, RESTART_WINDOW_S = 3, 600.0  # STAB1: a fourth worker loss within ten minutes stops recovery
RESTART_BACKOFF_S = (0.0, 5.0, 20.0)  # before the 1st, 2nd and 3rd attempt: device memory may free late
# What a replacement must report as the first worker did; `engine` holds the tuned profile the learner is bound to.
_IDENTITY_KEYS = ("model_id", "revision", "device", "execution_variant", "sources_sha256", "engine")


def _identity_of(backend: Any) -> dict[str, Any]:
    payload = getattr(backend, "ready_payload", None) or {}
    return {key: payload.get(key) for key in _IDENTITY_KEYS}
from .types import GenerationRequest, ModelSpec


def _set_replay_phase(state: dict, phase: str) -> bool:
    """Move a replay window to `phase` unless it is already being cancelled (RSI3-LIVE (f)).

    Cancellation ("drain") is final: a token or a group boundary arriving afterwards must not
    relabel the window, or a request that arrives next would be recorded in the wrong phase."""
    with state["lock"]:
        if phase == "drain":
            if state["cancel_ns"] is None:
                state["cancel_ns"] = time.monotonic_ns()
            state["phase"] = "drain"
            return True
        if state["cancel_ns"] is not None:
            return False
        state["phase"] = phase
        return True


def _replay_phase(state: dict) -> str:
    with state["lock"]:
        return state["phase"]


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
    terminal_seen: bool = False
    dispatched_ns: int | None = None
    first_ns: int | None = None  # RSI4's log: first token delivered
    width: int = 1               # streams in its group when it started (or joined)
    joined: bool = False         # SERVE8: joined a running group
    replay: dict | None = None   # RSI3-LIVE-P2: arrived during a replay window (window id, phase)
    first_after_replay: bool = False
    taken_ns: int | None = None  # when the scheduler took up its group (before any gather wait)
    end_ns: int | None = None    # RSI3-LIVE: when the scheduler retired it
    outcome: str | None = None   # RSI3-LIVE: ok, cancelled, else failed
    tokens: int = 0              # RSI3-LIVE: completion tokens delivered
    group: int | None = None     # STAB1: the scheduler group that served it
    error: str | None = None     # STAB1: the error code delivered to the caller, if any


class ProductService:
    """One loaded reference model with bounded FIFO admission.

    This is the portable reference path. Optimized grouping is not claimed to
    exist here or enabled without a separately qualified configuration.
    """

    policy: ServingPolicy = ServingPolicy()   # RSI4; __init__ loads the stored one
    serving_log: ServingLog | None = None
    learning: Any = None  # RSI3-LIVE: an attached live_learning.LiveLearning owns each group decision
    dispatch: Any = None  # LD1: an evidence-bound learned_dispatch.LearnedDispatch; None keeps FIFO order
    backend: Any = None
    _group_refused = False
    _gather_waited_ns = 0
    stability_log: ServingLog | None = None  # STAB1: bounded lifecycle events, no prompt text
    _groups = 0
    _in_flight: dict[str, Any] | None = None
    _progress_ns: int | None = None
    _exit_logged: Any = None
    # STAB1: builds an unstarted replacement worker; None keeps a lost worker lost (the pre-STAB1 contract).
    make_backend: Any = None
    _restarting = False
    _starting: Any = None
    _recovery_refused: str | None = None
    _restarts: list[float] = []  # replaced per instance before any append

    def __init__(self, store, *, backend=None, spec: ModelSpec | None = None):
        self.store = store
        self.settings = store.settings()
        self.backend = backend
        self.spec = spec
        self._pending: queue.Queue = queue.Queue(maxsize=self.settings["max_pending"])
        self._sessions: dict[str, _Session] = {}
        # A session removed from the stdlib queue has acquired one queue task.
        # Keep it here, rather than putting it back, so each acquisition has
        # exactly one matching task_done().
        self._carry: deque[_Session] = deque()
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._active: set[str] = set()
        self._completed = self._failed = self._cancelled = 0
        self._last_batch_size: int | None = None
        self._batch_count = 0
        # START1: work the scheduler may run while nothing is waiting, given a "preempted" check.
        # It runs on this thread, so no request can reach the backend while it does.
        self.idle_task = None
        self.idle_after_s = 30.0
        # RSI4: the scheduling policy (default = SERVE8) and the log `ironmule dream` learns it from.
        root = getattr(store, "root", None)
        self.policy = load_policy(root) if root is not None else ServingPolicy()
        self.serving_log = ServingLog(root) if root is not None else None
        self.stability_log = ServingLog(root, 4 << 20, STABILITY_LOG) if root is not None else None
        self._identity = _identity_of(backend)  # a replacement must match it
        self._restarts: list[float] = []
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
            result = {"service": "ironmule", "ready": bool(not self._closed.is_set() and self._backend_ready()),
                    "mode": self.settings["mode"],
                    "execution": ("numeric@dense" if getattr(self.backend, "engine_configuration", None) == "dense_stacked"
                                  else "exact" if getattr(self.backend, "compute_dtype", None) is None
                                  else f"exact@{self.backend.compute_dtype}"),
                    "backend": ("mlx_lm_reference" if getattr(self.backend, "execution_variant", "reference") == "reference"
                                else getattr(self.backend, "execution_variant", "reference")),
                    "loaded_model": self.spec.model_id if self.spec else None,
                    "queued_requests": self._pending.qsize() + len(self._carry), "active_requests": len(self._active),
                    "completed_requests": self._completed, "failed_requests": self._failed,
                    "cancelled_requests": self._cancelled,
                    "optimization": optimization}
            if self._last_batch_size is not None:
                result.update(last_batch_size=self._last_batch_size, batch_count=self._batch_count)
            result["device"] = getattr(self.backend, "device", None)  # OPS1: what the worker reported
            result.update(self._stability())
        if self.learning is not None:  # outside the service lock: admission never waits on the learner
            result["learning"] = self.learning.summary()
        if self.dispatch is not None:
            result["learned_dispatch"] = self.dispatch.status()
        policy = self.policy
        result["schedule"] = {"max_width": policy.max_width, "gather_ms": policy.gather_ms, "order": policy.order,
                              "source": policy.source}
        return result

    def accepting(self) -> bool:
        """HW1: whether `stream` would admit a request now (open, and a worker ready or being replaced)."""
        with self._lock:
            return not self._closed.is_set() and self._backend_ready()

    def outstanding(self) -> int:
        """HW1: admitted requests not yet retired, queued or in flight; what a device pool balances."""
        with self._lock:
            return len(self._sessions)

    def holds(self, request_id: str) -> bool:
        """HW1: whether this request id is admitted here and not yet retired."""
        with self._lock:
            return request_id in self._sessions

    def admitted_requests(self) -> list[GenerationRequest]:
        """LD1: the requests admitted here and not yet retired, which a learned placement reads."""
        with self._lock:
            return [session.request for session in self._sessions.values()]

    def note(self, event: str, **fields: Any) -> None:
        """STAB1: one lifecycle event: ids, counts, times and codes, never prompt or answer text. A worker
        exit adds the worker's last stderr (4 KiB), which can hold library output: the log is private
        state like the serving log, and published summaries leave that tail out."""
        log = self.stability_log
        if log is not None:
            log.append({"t": round(time.monotonic(), 4), "event": event, **fields})

    def _stability(self) -> dict[str, Any]:
        """STAB1: what tells a dead worker, a worker without progress, an incomplete group and blocked
        admission apart. Called under the service lock."""
        now = time.monotonic_ns()
        backend = self.backend
        exit_ = getattr(backend, "last_exit", None)
        process = getattr(backend, "_process", None)
        ready = bool(getattr(backend, "ready", False))
        worker = {"state": ("ready" if ready else "restarting" if self._restarting else
                            "refused" if self._recovery_refused else "lost" if exit_ is not None else "unavailable"),
                  "ready": ready, "pid": getattr(process, "pid", None), "restarts": len(self._restarts),
                  "scheduler_alive": self._thread.is_alive() if getattr(self, "_thread", None) else None,
                  "memory": getattr(backend, "last_memory", None),
                  "refused": self._recovery_refused,
                  "exit": None if exit_ is None else {k: exit_.get(k) for k in ("reason", "exit_status", "reaped")}}
        flight = self._in_flight
        if flight is not None:
            members = flight["members"]
            flight = {"group": flight["group"], "width": len(members),
                      "open": sum(1 for item in members if item.end_ns is None and item.outcome is None),
                      "age_s": (now - flight["started_ns"]) / 1e9,
                      "last_progress_age_s": None if self._progress_ns is None or self._progress_ns < flight["started_ns"]
                      else (now - self._progress_ns) / 1e9}
        oldest = [item.submitted_ns for item in list(self._carry)[:1]]
        with self._pending.mutex:
            oldest += [item.submitted_ns for item in list(self._pending.queue)[:1]]
        return {"worker": worker, "in_flight": flight, "groups_started": self._groups,
                "oldest_queued_s": (now - min(oldest)) / 1e9 if oldest else None}

    def _note_worker_exit(self) -> None:
        exit_ = getattr(self.backend, "last_exit", None)
        if exit_ is not None and exit_ is not self._exit_logged:
            self._exit_logged = exit_
            self.note("worker_exit", **exit_)

    def learning_status(self) -> dict[str, Any]:
        """RSI3-LIVE status for the API and chat page: serving, learning, candidate, strategy."""
        health = self.health()
        replay = getattr(self, "_replay_state", None)
        serving = {"ready": health["ready"], "device": health.get("device"), "backend": health["backend"],
                   "loaded_model": health["loaded_model"], "queued_requests": health["queued_requests"],
                   "replay_phase": None if replay is None else _replay_phase(replay)}
        if self.learning is None:
            return {"schema": "ironmule.live_learning.v1", "mode": "off", "eligible": False,
                    "reason": "learning is off; start the server with `--learning active`",
                    "strategy": {"active": "reference", "version": None, "reason": "learning is off"},
                    "candidate": None, "serving": serving}
        return {**self.learning.status(), "serving": serving}

    def learning_control(self, action: str, arm: str | None = None, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.learning is None:
            raise InvalidRequest("learning is off; start the server with `--learning active`")
        if not self.learning.eligible:
            raise InvalidRequest("learning is unavailable here: " + str(self.learning.reason))
        try:
            self.learning.control(action, arm=arm, evidence=evidence)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc
        return self.learning_status()

    def dispatch_status(self) -> dict[str, Any]:
        from .learned_dispatch import status_of
        return status_of(self.dispatch)

    def dispatch_control(self, action: str) -> dict[str, Any]:
        from .learned_dispatch import control
        try:
            return control(self.dispatch, action)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc

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
            if self.spec is None or not self._backend_ready():
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
            except queue.Full:
                if self.learning is not None:
                    self.learning.rejected(time.monotonic_ns())  # a rejection costs the policy (RSI3-LIVE)
                session = None
            if session is not None:
                self._sessions[request.request_id] = session
                queued = self._pending.qsize() + len(self._carry)
                replay = getattr(self, "_replay_state", None)
                if replay is not None:  # RSI3-LIVE-P2: real demand cancels replay the moment it is admitted
                    session.replay = {"window": replay["id"], "phase": _replay_phase(replay)}
                    self._replay_cancel()
                if self.learning is not None:
                    self.learning.arrived(request.request_id, session.submitted_ns, request)
        if session is None:  # logged outside the service lock
            self.note("reject", id=request.request_id, reason="queue_full")
            raise Overloaded("request queue is full; retry later")
        self.note("admit", id=request.request_id, queued=queued, deadline_s=self.settings["request_timeout_s"])
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
        if isinstance(event, ProductError):
            session.error = event.code
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

    def _backend_ready(self) -> bool:
        """Ready, or between workers while idle work runs (START1) or a lost worker is being replaced
        (STAB1): requests then wait in the queue under their own deadlines. Never without a scheduler."""
        thread = getattr(self, "_thread", None)
        if thread is not None and not thread.is_alive() and not self._closed.is_set():
            return False
        if getattr(self, "_idle_busy", False) or self._restarting:
            return True
        if self.backend is not None and self.backend.ready:
            return True
        return self._lost_worker()

    def _lost_worker(self) -> bool:
        """STAB1: the worker is gone and a replacement may be started."""
        return (self.make_backend is not None and self._recovery_refused is None and self.backend is not None
                and not getattr(self, "_idle_busy", False) and not self.backend.ready)

    def _recover(self) -> None:
        """STAB1: replace a lost worker, at most MAX_RESTARTS times within RESTART_WINDOW_S.

        Runs on the scheduler thread, so no group is in flight. The old worker must have been reaped
        (one that died between groups is retired here first), and so must every failed replacement,
        or recovery stops for good: two workers would share one device. A replacement must report the
        identity the service runs with, or it is closed and counts as a failed attempt: it would serve
        under qualifications it never earned. Requests that were in flight already received their
        error; none is retried."""
        old = self.backend
        if getattr(old, "last_exit", None) is None:  # died while idle: no call has retired it yet
            try:
                old.close("exited between groups")
            except Exception:  # noqa: BLE001 - the reap check below decides
                pass
        self._note_worker_exit()
        exit_ = getattr(old, "last_exit", None)
        reason = (exit_ or {}).get("reason") or "the worker was not ready"
        refused = None
        self._restarts = [t for t in self._restarts if time.monotonic() - t < RESTART_WINDOW_S]
        if exit_ is None or not exit_.get("reaped"):
            refused = "the old worker's exit is not confirmed"
        elif len(self._restarts) >= MAX_RESTARTS:
            refused = f"{MAX_RESTARTS} worker restarts within {RESTART_WINDOW_S:.0f} s"
        if refused is not None:
            self._recovery_refused = refused
            self.note("worker_restart", outcome="refused", reason=reason, refused=refused)
            return
        self._restarting = True
        outcome, fresh, pid = "failed", None, None
        started = time.monotonic()
        try:
            if self._closed.wait(RESTART_BACKOFF_S[min(len(self._restarts), len(RESTART_BACKOFF_S) - 1)]):
                outcome = "abandoned: service closed"
                return
            self._restarts.append(time.monotonic())
            with self._lock:
                if self._closed.is_set():
                    outcome = "abandoned: service closed"
                    return
                fresh = self._starting = self.make_backend()
            fresh.start()
            pid = getattr(getattr(fresh, "_process", None), "pid", None)
            identity = _identity_of(fresh)
            with self._lock:  # close() sees either the replacement as the backend or a closed service here
                if self._closed.is_set():
                    outcome = "abandoned: service closed"
                elif identity != self._identity:
                    outcome = "failed: a different identity (" + ", ".join(
                        key for key in _IDENTITY_KEYS if identity[key] != self._identity[key]) + ")"
                else:
                    self.backend, outcome = fresh, "ready"
        except Exception as exc:  # noqa: BLE001 - a failed start counts as an attempt
            outcome = "failed: " + type(exc).__name__
        finally:
            if fresh is not None and outcome != "ready":
                try:
                    fresh.close("replacement not used")
                except Exception:  # noqa: BLE001
                    pass
                left = getattr(fresh, "last_exit", None)
                if left is not None and not left.get("reaped"):
                    self._recovery_refused = "a failed replacement's exit is not confirmed"
            self._restarting, self._starting = False, None
            self.note("worker_restart", outcome=outcome, reason=reason, attempt=len(self._restarts),
                      s=time.monotonic() - started, pid=pid, refused=self._recovery_refused)

    def _backend_timeout(self, deadlines: list[float | None]) -> float | None:
        """STAB1: how long the backend may take for this work. Each request's own deadline cancels only
        that request (its caller gets RequestTimeout and cancels it); the worker is retired only when it
        has not finished a full request timeout after the last deadline, when it no longer answers
        cancellation. Before STAB1 the earliest member's deadline retired the worker for the group."""
        if not deadlines or any(deadline is None for deadline in deadlines):
            return None
        grace = (getattr(self, "settings", None) or {}).get("request_timeout_s") or 0.0
        return max(0.001, max(deadlines) - time.monotonic() + grace)

    def _shortest_first(self) -> Any:
        """Who decides whether a group starts shortest output limit first: an active learned dispatch (LD1),
        a policy that always does (RL6), or nobody (arrival order, None). Read once per decision."""
        dispatch = self.dispatch  # an operator may detach it while this runs
        if dispatch is not None and dispatch.active:
            return lambda requests: bool(dispatch.sjf(requests))
        if self.policy.order == "sjf":
            return lambda requests: True
        return None

    def _next_session(self) -> _Session | None:
        decide = self._shortest_first()
        if decide is not None:
            return self._next_sorted(decide)
        with self._lock:
            if self._carry:
                return self._carry.popleft()
        try:
            return self._pending.get(timeout=0.1)
        except queue.Empty:
            return None

    def _order_waiting(self, decide: Any) -> None:
        """Every waiting request into the carry, then shortest output limit first when `decide` says so for
        this group (LD1's policy, or RL6's fixed schedule), else arrival order. A request waiting longer
        than `SJF_AGE_S` starts by arrival. The caller holds `self._lock`."""
        from .learned_dispatch import SJF_AGE_S

        while True:
            try:
                self._carry.append(self._pending.get_nowait())
            except queue.Empty:
                break
        now = time.monotonic_ns()
        sjf = decide([session.request for session in self._carry])

        def key(session: _Session) -> tuple[bool, int, int]:
            if not sjf:
                return (False, 0, session.submitted_ns)
            aged = now - session.submitted_ns > SJF_AGE_S * 1e9
            return (not aged, 0 if aged else session.request.max_tokens, session.submitted_ns)

        waiting = sorted(self._carry, key=key)
        self._carry.clear()
        self._carry.extend(waiting)

    def _next_sorted(self, decide: Any) -> _Session | None:
        with self._lock:
            self._order_waiting(decide)
            if self._carry:
                return self._carry.popleft()
        try:
            first = self._pending.get(timeout=0.1)
        except queue.Empty:
            return None
        with self._lock:
            self._carry.append(first)
            self._order_waiting(decide)
            return self._carry.popleft()

    def _batch_capacity(self) -> int:
        capacity = getattr(self.backend, "batch_capacity", 1)
        capacity = min(capacity, self.policy.max_width) if type(capacity) is int else 1
        return capacity if 2 <= capacity <= 32 else 1

    def _group_for(self, first: _Session, width: int | None = None, gather_ms: int | None = None) -> list[_Session]:
        """Take only work already admitted, waiting at most the gather time for company.

        ``width``/``gather_ms`` are a learner's decision (RSI3-LIVE); without one, the policy's, or a
        learned dispatch's width (LD1)."""
        self._group_refused, self._gather_waited_ns = False, 0
        dispatch, decide = self.dispatch, self._shortest_first()
        if decide is not None:
            with self._lock:
                self._order_waiting(decide)
                if width is None and dispatch is not None and dispatch.active:
                    width = dispatch.width([first.request] + [session.request for session in self._carry])
        capacity = self._batch_capacity() if width is None else min(self._batch_capacity(), width)
        gather_ms = self.policy.gather_ms if gather_ms is None else gather_ms
        if capacity == 1:
            return [first]
        group = [first]
        with self._lock:
            while self._carry and len(group) < capacity:
                group.append(self._carry.popleft())
        while len(group) < capacity:
            try:
                group.append(self._pending.get_nowait())
            except queue.Empty:
                break
        if gather_ms > 0 and len(group) < capacity:  # RSI4: wait a little for company before the group starts
            started = time.monotonic_ns()
            until = time.monotonic() + gather_ms / 1000.0
            while len(group) < capacity and (left := until - time.monotonic()) > 0:
                try:
                    group.append(self._pending.get(timeout=left))
                except queue.Empty:
                    break
            self._gather_waited_ns = time.monotonic_ns() - started
        if len(group) < 2:
            return group
        checker = getattr(self.backend, "can_batch_requests", None)
        try:
            allowed = checker([item.request for item in group]) if callable(checker) else False
        except Exception:
            allowed = False
        if allowed:
            return group
        # The backend's metadata gate is authoritative.  Preserve every later
        # request in FIFO order without re-enqueueing an acquired task.
        self._group_refused = True
        with self._lock:
            self._carry.extendleft(reversed(group[1:]))
        return group[:1]

    def _streams_alone(self, session: _Session) -> bool:
        """SERVE8: on a backend that streams groups, one request still runs as a group, so a
        request arriving while it streams has a group to join."""
        if not getattr(self.backend, "streams_batches", False):
            return False
        try:
            return bool(self.backend.can_batch_requests([session.request]))
        except Exception:
            return False

    def _retire(self, session: _Session) -> None:
        if session.end_ns is None:
            session.end_ns = time.monotonic_ns()
        if session.outcome is None:
            session.outcome = "failed"
        if session.first_after_replay:
            try:
                self._record_preemption(session)
            except Exception:  # noqa: BLE001 - measurement never fails a request
                pass
        if self.learning is not None:
            try:
                self.learning.ended(session.request.request_id, session.end_ns)
            except Exception:  # noqa: BLE001 - a learner error must not stop the scheduler (STAB1)
                self._learning_fault()
        self.note("end", id=session.request.request_id, group=session.group, outcome=session.outcome,
                  error=session.error, tokens=session.tokens,
                  queue_s=None if session.dispatched_ns is None else (session.dispatched_ns - session.submitted_ns) / 1e9,
                  first_s=None if session.first_ns is None else (session.first_ns - session.submitted_ns) / 1e9,
                  total_s=(session.end_ns - session.submitted_ns) / 1e9)
        with self._lock:
            self._sessions.pop(session.request.request_id, None)
            self._active.discard(session.request.request_id)
        self._pending.task_done()

    def _queued(self) -> tuple[int, list[tuple[int, GenerationRequest]]]:
        """Waiting requests and the oldest eight as (submitted_ns, request): metadata for RSI3-LIVE."""
        with self._lock:
            oldest = list(self._carry)[:8]
            count = len(self._carry)
        with self._pending.mutex:
            count += len(self._pending.queue)
            oldest += list(self._pending.queue)[:8 - len(oldest)]
        return count, [(item.submitted_ns, item.request) for item in oldest]

    def _replay_window(self, trace: list[tuple[float, int, int]], choose: Any,
                       preempted: Any) -> dict[str, Any] | None:
        """RSI3-LIVE qualification: replay a metadata trace through the real backend.

        ``trace`` holds (arrival offset s, prompt characters, output limit); prompts are a fixed
        filler, never user text. ``choose(queue, waiting, rate, now_ns)`` gives each group's width
        and gather wait; groups run through this scheduler's own group execution. Runs on the
        scheduler thread while nothing waits. A real request cancels the window the moment it is
        admitted (RSI3-LIVE-P2): every replayed request stops at its next token boundary (or before
        its prefill), the window returns None, and that request's timeline is recorded. Request
        counters and the RSI4 log are left untouched. New replay work is refused until every
        replayed request of an earlier window has ended.
        """
        from .live_learning import filler

        if (self.backend is None or not self.backend.ready or self.spec is None
                or (self.learning is not None and self.learning.replay_blocked)):
            return None
        timeout = self.settings["request_timeout_s"] or 120
        deadline = time.monotonic() + timeout + (trace[-1][0] if trace else 0.0)
        sessions = [_Session(GenerationRequest(self.spec.model_id, (("user", filler(chars)),), max_tokens=tokens,
                                               request_id=f"replay-{index}-{time.monotonic_ns()}"), deadline)
                    for index, (_, chars, tokens) in enumerate(trace)]
        texts: dict[str, list[str]] = {item.request.request_id: [] for item in sessions}
        stop, aborted = threading.Event(), threading.Event()
        window = self._replay_windows = getattr(self, "_replay_windows", 0) + 1
        state = {"id": window, "phase": "prestart", "group": set(), "cancel_ns": None, "end_ns": None,
                 "lock": threading.Lock()}

        def cancel() -> None:
            _set_replay_phase(state, "drain")
            aborted.set()
            for item in sessions:
                item.cancelled.set()
                item.backend_cancel.set()

        def drain(item: _Session) -> None:
            while not (stop.is_set() and item.events.empty()):
                try:
                    event = item.events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if event is None:  # the window is over: leave at once, do not wait out a poll
                    return
                if isinstance(event, dict) and event.get("type") == "token":
                    texts[item.request.request_id].append(event.get("text") or "")
                    if item.request.request_id in state["group"] and _replay_phase(state) == "prefill":
                        _set_replay_phase(state, "decode")

        def watch() -> None:
            while not stop.wait(0.01):
                if preempted():
                    cancel()
                    return

        with self._lock:
            saved = (self._completed, self._failed, self._cancelled, self._last_batch_size, self._batch_count)
            self._replay_state, self._replay_cancel = state, cancel  # stream() cancels on admission
        log, self.serving_log = self.serving_log, None
        threads = [threading.Thread(target=drain, args=(item,), daemon=True) for item in sessions]
        threads.append(threading.Thread(target=watch, daemon=True))
        for thread in threads:
            thread.start()
        start = time.monotonic_ns()
        due = [start + int(offset * 1e9) for offset, _, _ in trace]
        waiting: list[_Session] = []
        admitted, gap, last = 0, None, None

        def admit() -> None:
            nonlocal admitted, gap, last
            now = time.monotonic_ns()
            while admitted < len(sessions) and due[admitted] <= now:
                item = sessions[admitted]
                item.submitted_ns = due[admitted]
                if last is not None:
                    step = max((due[admitted] - last) / 1e9, 1e-3)
                    gap = step if gap is None else 0.8 * gap + 0.2 * step
                last = due[admitted]
                waiting.append(item)
                admitted += 1

        try:
            if preempted():
                cancel()
            while (admitted < len(sessions) or waiting) and not aborted.is_set():
                admit()
                if not waiting:
                    aborted.wait(min(0.01, max(0.0, (due[admitted] - time.monotonic_ns()) / 1e9)))
                    continue
                now = time.monotonic_ns()
                idle_s = (now - last) / 1e9 if last is not None else 0.0
                rate = 0.0 if gap is None else 1.0 / max(gap, idle_s, 1e-3)  # the live estimate's formula
                width, gather_ms = choose(len(waiting), [(item.submitted_ns, item.request) for item in waiting[:8]],
                                          rate, now)
                width = max(1, min(width, self._batch_capacity()))
                until = time.monotonic_ns() + gather_ms * 1_000_000
                while (len(waiting) < width and admitted < len(sessions) and time.monotonic_ns() < until
                       and not aborted.is_set()):
                    aborted.wait(0.002)
                    admit()
                if aborted.is_set():
                    break
                group, waiting[:] = waiting[:width], waiting[width:]
                for item in group:
                    item.dispatched_ns = time.monotonic_ns()
                    item.width = len(group)
                state["group"] = {item.request.request_id for item in group}
                _set_replay_phase(state, "prefill")
                if len(group) > 1 or self._streams_alone(group[0]):
                    self._perform_batch(group)
                else:
                    self._perform(group[0])
                for item in group:
                    item.end_ns = item.end_ns or time.monotonic_ns()
                    item.outcome = item.outcome or "failed"
                _set_replay_phase(state, "prestart")
        finally:
            if aborted.is_set() or admitted < len(sessions) or waiting:
                cancel()
            stop.set()
            for item in sessions:
                try:
                    item.events.put_nowait(None)
                except queue.Full:
                    pass  # that drain thread ends on its next poll
            for thread in threads:
                thread.join(timeout=5)
            state["end_ns"] = time.monotonic_ns()
            with self._lock:
                self._replay_state = self._replay_cancel = None
                self._replay_log = {**{k: v for k, v in getattr(self, "_replay_log", {}).items() if k > window - 8},
                                    window: {"cancel_ns": state["cancel_ns"], "end_ns": state["end_ns"]}}
                (self._completed, self._failed, self._cancelled, self._last_batch_size, self._batch_count) = saved
            self.serving_log = log
            unsettled = [thread for thread in threads if thread.is_alive()]
            if unsettled and self.learning is not None:
                # Old model work is not provably over: no new replay until it is.
                self._replay_leftovers = unsettled
                self.learning.replay_blocked = "an earlier replay window has not ended yet"
        if aborted.is_set():
            self._replay_first_pending = window
            return None
        end = max(item.end_ns for item in sessions)
        rows = [{"ttft_s": None if item.first_ns is None else (item.first_ns - item.submitted_ns) / 1e9,
                 "itl_s": (None if item.first_ns is None or item.tokens < 2
                           else (item.end_ns - item.first_ns) / 1e9 / (item.tokens - 1)),
                 "e2e_s": (item.end_ns - item.submitted_ns) / 1e9, "tokens": item.tokens,
                 "outcome": item.outcome} for item in sessions]
        return {"rows": rows, "texts": ["".join(texts[item.request.request_id]) for item in sessions],
                "span_s": (end - start) / 1e9}

    def _record_preemption(self, session: _Session) -> None:
        """RSI3-LIVE (d): how long a real request that arrived during replay waited for it."""
        info = session.replay
        window = getattr(self, "_replay_log", {}).get(info["window"])
        if window is None or session.dispatched_ns is None or session.taken_ns is None or self.learning is None:
            return
        # Extra wait ends when the scheduler takes the request up. A gather wait after that is the
        # schedule's own choice, which it makes with or without replay; it is reported apart.
        arrive, taken, dispatch = session.submitted_ns, session.taken_ns, session.dispatched_ns
        cancel_ns = window["cancel_ns"] or arrive
        self.learning.preempted({
            "phase": info["phase"], "unix": time.time(),
            "extra_wait_s": max(0.0, (taken - arrive) / 1e9),
            "detect_s": max(0.0, (cancel_ns - arrive) / 1e9),
            "drain_s": max(0.0, (window["end_ns"] - max(cancel_ns, arrive)) / 1e9),
            "handoff_s": max(0.0, (taken - window["end_ns"]) / 1e9),
            "schedule_to_dispatch_s": max(0.0, (dispatch - taken) / 1e9),
            "first_after_dispatch_s": None if session.first_ns is None else (session.first_ns - dispatch) / 1e9,
            "outcome": session.outcome})

    def _learning_fault(self) -> None:
        """A learner error never stops serving: the reference schedule takes over, with the reason."""
        learning = self.learning
        if learning is not None:
            learning.reason = "the learner failed; the reference schedule serves until restart"

    def _run(self) -> None:
        idle_since = time.monotonic()
        while not self._closed.is_set():
            if self._lost_worker():
                self._recover()
                continue
            session = self._next_session()
            if session is None:
                if self.learning is not None:  # RSI3-LIVE: dream while nothing waits; demand preempts
                    leftovers = getattr(self, "_replay_leftovers", None)
                    if leftovers is not None and not any(thread.is_alive() for thread in leftovers):
                        self._replay_leftovers, self.learning.replay_blocked = None, None
                    try:
                        self.learning.idle(lambda: not self._pending.empty() or self._closed.is_set(),
                                           replay=self._replay_window)
                    except Exception:  # noqa: BLE001
                        self._learning_fault()
                if self.idle_task is not None and time.monotonic() - idle_since >= self.idle_after_s:
                    self._idle_busy = True
                    try:
                        self.idle_task(lambda: not self._pending.empty() or self._closed.is_set())
                    except Exception:  # noqa: BLE001 - idle work must never stop the scheduler
                        pass
                    finally:
                        self._idle_busy = False
                    if self.backend is not None and self.backend.ready:  # IdleTuner loaded a new profile
                        self._identity = _identity_of(self.backend)
                    idle_since = time.monotonic()
                continue
            idle_since = float("inf")
            taken = time.monotonic_ns()
            group = [session]
            decision = None
            if self.learning is not None:
                try:
                    waiting, oldest = self._queued()
                    # A request that just displaced replay is served, not explored with (RSI3-LIVE-P2).
                    decision = self.learning.decide(1 + waiting, [(session.submitted_ns, session.request)] + oldest[:7],
                                                    explore=session.replay is None)
                except Exception:  # noqa: BLE001
                    self._learning_fault()
            try:
                group = (self._group_for(session) if decision is None
                         else self._group_for(session, decision.width, decision.gather_ms))
                for item in group:
                    item.taken_ns = taken
                pending_replay = getattr(self, "_replay_first_pending", None)
                if pending_replay is not None:  # the first real group after a cancelled replay window
                    self._replay_first_pending = None
                    for item in group:
                        item.first_after_replay = bool(item.replay and item.replay["window"] == pending_replay)
                with self._lock:
                    self._active.update(item.request.request_id for item in group)
                    self._groups += 1
                    self._in_flight = {"group": self._groups, "members": group, "started_ns": time.monotonic_ns()}
                for item in group:
                    item.width = len(group)
                    item.group = self._groups
                self.note("group", group=self._groups, ids=[item.request.request_id for item in group],
                          width=len(group), pid=getattr(getattr(self.backend, "_process", None), "pid", None))
                if len(group) > 1 or self._streams_alone(group[0]):
                    self._perform_batch(group)
                else:
                    self._perform(group[0])
            finally:
                with self._lock:
                    flight, self._in_flight = self._in_flight, None
                if flight is not None:
                    self.note("group_end", group=flight["group"],
                              s=(time.monotonic_ns() - flight["started_ns"]) / 1e9,
                              memory=getattr(self.backend, "last_memory", None))
                self._note_worker_exit()
                for item in group:
                    self._retire(item)
                if decision is not None:
                    try:
                        waiting, oldest = self._queued()
                        self.learning.complete(group, refused=self._group_refused,
                                               gather_waited_ns=self._gather_waited_ns,
                                               queue_after=waiting, next_waiting=oldest)
                    except Exception:  # noqa: BLE001
                        self._learning_fault()
                idle_since = time.monotonic()

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
        drained = terminal_seen or stream is None  # without a stream nothing reached the worker
        if stream is not None and not drained:
            drained = self._drain_cancelled_stream(stream)
        if not drained and self.backend is not None:
            # A partial or failed drain leaves protocol synchronization
            # unknown. Retire this worker; the scheduler may replace it (STAB1), never retry here.
            self.note("worker_retire", reason="cancelled stream not drained", id=session.request.request_id)
            try:
                self.backend.close()
            except Exception:
                pass
        self._emit(session, RequestCancelled("request cancelled"))
        session.outcome = "cancelled"
        with self._lock:
            self._cancelled += 1

    def _consume_events(self, session: _Session, events: Iterator[dict[str, Any]]) -> None:
        """Validate and deliver one request's frames from either backend path."""
        dispatched = session.dispatched_ns if session.dispatched_ns is not None else time.monotonic_ns()
        stop_filter = StopFilter(session.request.stop)
        done = None
        generated = 0
        prompt_tokens = 0
        for event in events:
            self._progress_ns = time.monotonic_ns()  # STAB1: the backend's last sign of life
            if not isinstance(event, dict):
                raise BackendUnavailable("worker emitted an invalid event")
            if session.cancelled.is_set():
                if event.get("type") == "done":
                    done = dict(event)
                    session.terminal_seen = True
                raise RequestCancelled("request cancelled")
            kind = event.get("type")
            if kind == "token":
                if done is not None:
                    raise BackendUnavailable("worker emitted tokens after completion")
                generated += 1
                if generated > session.request.max_tokens:
                    raise BackendUnavailable("worker exceeded the token limit")
                prompt_tokens = event.get("prompt_tokens", prompt_tokens)
                if session.first_ns is None:
                    session.first_ns = time.monotonic_ns()
                text = stop_filter.feed(event.get("text", ""))
                if text and not self._emit(session, {**event, "text": text}):
                    raise RequestCancelled("response delivery interrupted")
                if stop_filter.stopped:
                    session.backend_cancel.set()
            elif kind == "done":
                if done is not None:
                    raise BackendUnavailable("worker emitted duplicate completion")
                done = dict(event)
                session.terminal_seen = True
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
                                   "queue_ns": dispatched - session.submitted_ns,
                                   "service_ns": time.monotonic_ns() - session.submitted_ns}):
            raise RequestCancelled("response delivery interrupted")
        with self._lock:
            self._completed += 1
        session.outcome = "ok"
        session.tokens = int(done.get("completion_tokens", generated) or 0)
        session.end_ns = time.monotonic_ns()
        if self.serving_log is not None:
            end = time.monotonic_ns()
            self.serving_log.append({
                "arrive": session.submitted_ns / 1e9, "start": dispatched / 1e9,
                "first": (session.first_ns or end) / 1e9, "end": end / 1e9,
                "prompt_tokens": int(done.get("prompt_tokens", prompt_tokens) or 0),
                "new_tokens": int(done.get("completion_tokens", generated) or 0),
                "width": session.width, "joined": session.joined})

    def _perform(self, session: _Session) -> None:
        stream = None
        try:
            if session.cancelled.is_set():
                self._emit(session, RequestCancelled("request cancelled"))
                with self._lock:
                    self._cancelled += 1
                return
            remaining = None if session.deadline is None else session.deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise RequestTimeout("request expired in queue")
            if self.backend is None or not self.backend.ready:
                raise BackendUnavailable("model worker is unavailable")
            session.dispatched_ns = time.monotonic_ns()
            stream = self.backend.stream(session.request, cancel=session.backend_cancel,
                                         timeout=self._backend_timeout([session.deadline]))
            self._consume_events(session, stream)
        except RequestCancelled:
            self._finish_cancelled(session, stream, terminal_seen=session.terminal_seen)
        except ProductError as exc:
            if session.cancelled.is_set():
                self._finish_cancelled(session, stream, terminal_seen=session.terminal_seen)
            else:
                self._emit(session, exc)
                with self._lock:
                    self._failed += 1
        except Exception:
            if session.cancelled.is_set():
                self._finish_cancelled(session, stream, terminal_seen=session.terminal_seen)
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

    def _perform_batch(self, sessions: list[_Session]) -> None:
        """Execute one already-admitted group.  Batch output is buffered only."""
        handled: set[str] = set()
        try:
            if self.backend is None or not self.backend.ready:
                raise BackendUnavailable("model worker is unavailable")
            now = time.monotonic()
            live: list[_Session] = []
            for item in sessions:
                if item.cancelled.is_set():
                    self._finish_cancelled(item, None, terminal_seen=True)
                    handled.add(item.request.request_id)
                elif item.deadline is not None and item.deadline <= now:
                    self._emit(item, RequestTimeout("request expired in queue"))
                    with self._lock:
                        self._failed += 1
                    handled.add(item.request.request_id)
                else:
                    live.append(item)
            if not live:
                return
            if len(live) == 1 and not getattr(self.backend, "streams_batches", False):
                # Filtering may leave one valid session.  It has not been
                # dispatched as a group, so use the normal single path.
                self._perform(live[0])
                handled.add(live[0].request.request_id)
                return
            timeout = self._backend_timeout([item.deadline for item in live])
            for item in live:
                item.dispatched_ns = time.monotonic_ns()
            if getattr(self.backend, "streams_batches", False):
                self._stream_batch(live, timeout, handled)
                return
            rows = self.backend.complete_batch(
                [item.request for item in live], cancels=[item.backend_cancel for item in live], timeout=timeout,
            )
            if not isinstance(rows, list) or len(rows) != len(live):
                raise BackendUnavailable("model worker returned an invalid batch")
            expected = [item.request.request_id for item in live]
            if [row.get("request_id") if isinstance(row, dict) else None for row in rows] != expected:
                raise BackendUnavailable("model worker returned an invalid batch")
            with self._lock:
                self._last_batch_size = len(live)
                self._batch_count += 1
            for item, row in zip(live, rows):
                try:
                    frames = row.get("events")
                    if not isinstance(frames, list):
                        raise BackendUnavailable("model worker returned an invalid batch")
                    self._consume_events(item, iter(frames))
                    handled.add(item.request.request_id)
                except RequestCancelled:
                    self._finish_cancelled(item, None, terminal_seen=True)
                    handled.add(item.request.request_id)
                except ProductError as exc:
                    if item.cancelled.is_set():
                        self._finish_cancelled(item, None, terminal_seen=True)
                    else:
                        self._emit(item, exc)
                        with self._lock:
                            self._failed += 1
                    handled.add(item.request.request_id)
        except Exception:
            # Dispatch happened as one unit: never conceal a failed batch by
            # replaying it serially.
            for item in sessions:
                if item.request.request_id in handled:
                    continue
                if item.cancelled.is_set():
                    self._finish_cancelled(item, None, terminal_seen=True)
                else:
                    self._emit(item, BackendUnavailable("model worker failed"))
                    with self._lock:
                        self._failed += 1

    def _deliver(self, item: _Session, frames: Iterator[dict[str, Any]], handled: set[str]) -> None:
        """One grouped request's frames to its caller, with the single path's error handling."""
        try:
            self._consume_events(item, frames)
        except RequestCancelled:
            self._finish_cancelled(item, None, terminal_seen=True)
        except ProductError as exc:
            if item.cancelled.is_set():
                self._finish_cancelled(item, None, terminal_seen=True)
            else:
                self._emit(item, exc)
                with self._lock:
                    self._failed += 1
        handled.add(item.request.request_id)

    def _stream_batch(self, live: list[_Session], timeout: float | None, handled: set[str]) -> None:
        """SERVE4: every grouped stream receives its frames as the worker emits them.

        Each session drains its own queue on its own thread through `_consume_events`; a failed
        group ends every queue, and each session reports the failure the way the single path does.
        """
        queues: dict[str, queue.Queue[dict[str, Any] | None]] = {item.request.request_id: queue.Queue()
                                                                  for item in live}
        consumers: list[threading.Thread] = []
        joined: list[_Session] = []

        def consume(item: _Session) -> None:
            queues[item.request.request_id] = queue.Queue()
            consumer = threading.Thread(target=self._deliver, name="ironmule-batch-stream", daemon=True,
                                        args=(item, iter(queues[item.request.request_id].get, None), handled))
            consumers.append(consumer)
            consumer.start()

        def joins() -> list[tuple[GenerationRequest, threading.Event]]:
            """SERVE8: requests waiting now join the running group while fewer than its capacity
            stream at once; the backend's metadata gate decides as for a new group."""
            taken = []
            if not self.policy.join:  # RSI4: a learned policy may queue newcomers instead
                return taken
            while len(consumers) < self._batch_capacity():  # members over the group's life, as before STAB1
                with self._lock:
                    item = self._carry.popleft() if self._carry else None
                if item is None:
                    try:
                        item = self._pending.get_nowait()
                    except queue.Empty:
                        break
                if item.cancelled.is_set() or not self.backend.can_batch_requests([live[0].request, item.request]):
                    with self._lock:
                        self._carry.appendleft(item)  # keeps its place; the scheduler serves it next
                    break
                item.dispatched_ns = time.monotonic_ns()
                item.joined, item.width = True, len(consumers) + 1
                item.group = live[0].group
                self.note("join", id=item.request.request_id, group=item.group)
                with self._lock:
                    self._active.add(item.request.request_id)
                consume(item)
                joined.append(item)
                taken.append((item.request, item.backend_cancel))
            return taken

        def deliver(request_id: str, event: dict[str, Any]) -> None:
            queues[request_id].put(event)
            if event.get("type") == "done":  # STAB1: a finished member ends now, not when its group does
                queues[request_id].put(None)

        queues.clear()
        for item in live:
            consume(item)
        try:
            rows = self.backend.complete_batch(
                [item.request for item in live], cancels=[item.backend_cancel for item in live], timeout=timeout,
                on_event=deliver, joins=joins,
            )
            if not isinstance(rows, list) or len(rows) != len(live) + len(joined):
                raise BackendUnavailable("model worker returned an invalid batch")
            with self._lock:
                self._last_batch_size = len(live) + len(joined)
                self._batch_count += 1
        finally:
            for request_id in list(queues):
                queues[request_id].put(None)
            for consumer in consumers:
                consumer.join()
            for item in joined:  # taken from the queue here, so retired here
                self._retire(item)

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            for session in self._sessions.values():
                session.cancelled.set()
                session.backend_cancel.set()
        with self._lock:  # STAB1: _recover() publishes a replacement under this lock
            starting, backend, restarting = self._starting, self.backend, self._restarting
        process = getattr(starting, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()  # a replacement still loading is ours to stop; its start then fails
        if backend is not None:
            backend.close()
        self._thread.join(timeout=30.0 if restarting else 2.0)
        if self.learning is not None:
            try:
                self.learning.close()  # persists experience and policy state
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            self._sessions.clear()
        with self._lock:
            while self._carry:
                self._carry.popleft()
                self._pending.task_done()
        while True:
            try:
                self._pending.get_nowait()
                self._pending.task_done()
            except queue.Empty:
                break
