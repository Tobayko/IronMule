"""Executors: sequential batch-1, and asynchronous grouped batch-1 at width <= 4.

Grouping never changes a tensor shape. Every execution stays batch 1; the only
difference is that up to `max_width` independent decode steps are submitted without
an intermediate barrier and completed together. E14b attributed the gain to device
execution overlapping host submission rather than to cheaper host work, and E16
replicated it at +15.10% to +17.16% across forty independent OS processes with zero
correctness failures.

Width 4 is the shipped maximum because E15 found the whole gain available there and
E14b found width 8 regressing. Realised width falls below the maximum whenever fewer
requests are ready, which is the intended behaviour and not a degradation: the
executor never waits to fill a group.

Everything the executors need from a model sits behind `DecodeBackend`, so the
scheduling, fallback and isolation logic is testable without loading one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .telemetry import RequestMetrics, Telemetry

MAX_GROUP_WIDTH = 4
now = time.perf_counter_ns


class DecodeBackend(Protocol):
    """The minimum a model must offer. Implemented for real by `ironmule.service`."""

    eos_ids: tuple[int, ...]

    def capacity_for(self, prompt_lens: Sequence[int], max_tokens: int) -> int: ...
    def prefill(self, prompt_ids: Sequence[int], plan, capacity: int) -> tuple[Any, int]: ...
    def reset_state(self, base_state: Any, offset: int) -> Any: ...
    def step(self, state: Any, token: int, capacity: int) -> Any: ...
    def complete(self, handles: Sequence[Any]) -> None: ...
    def read(self, handle: Any) -> tuple[int, Any]: ...


@dataclass
class Session:
    """One in-flight request. State objects are never shared between sessions."""

    rid: str
    prompt_ids: list[int]
    max_tokens: int
    plan: Any
    arrival_ms: float = 0.0
    base_state: Any = None
    state: Any = None
    tokens: list[int] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""
    metrics: RequestMetrics | None = None

    def restart(self, backend: DecodeBackend) -> None:
        """Return to the post-prefill state. Used only by the fallback path."""
        self.state = backend.reset_state(self.base_state, len(self.prompt_ids))
        self.tokens = self.tokens[:1]
        self.done = False
        self.stop_reason = ""


def _accept_token(session: Session, token: int, when: int,
                  eos_ids: Sequence[int]) -> None:
    """Accept one physical output token and apply the generation contract."""
    session.tokens.append(token)
    metrics = session.metrics
    metrics.token_times_ns.append(when)
    if metrics.first_token_ns == 0:
        metrics.first_token_ns = when
    metrics.generated_tokens = len(session.tokens)
    metrics.visible_generated_tokens = sum(
        value not in eos_ids for value in session.tokens)
    if token in eos_ids:
        session.done, session.stop_reason = True, "eos"
    elif len(session.tokens) >= session.max_tokens:
        session.done, session.stop_reason = True, "length"
    if session.done:
        metrics.finished_ns = when
        metrics.stop_reason = session.stop_reason


class _Runner:
    def __init__(self, backend: DecodeBackend, telemetry: Telemetry):
        self.backend = backend
        self.telemetry = telemetry

    def _record_token(self, session: Session, token: int, when: int) -> None:
        _accept_token(session, token, when, self.backend.eos_ids)

    def _finish_sequentially(self, sessions: list[Session], capacity: int) -> None:
        for session in sessions:
            while not session.done:
                handle = self.backend.step(session.state, session.tokens[-1], capacity)
                self.backend.complete([handle])
                token, state = self.backend.read(handle)
                session.state = state
                self._record_token(session, token, now())


class SequentialExecutor(_Runner):
    """Interactive mode: one request at a time, synchronised every step.

    Lowest latency for a single caller, no economy of scale whatsoever — E14b
    measured submission and wait per request flat across every batch size.
    """

    name = "sequential"

    def run(self, sessions: list[Session], capacity: int) -> None:
        started = now()
        for session in sorted(sessions, key=lambda s: (s.arrival_ms, s.rid)):
            wait_ms = session.arrival_ms - (now() - started) / 1e6
            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)
            if not session.done:
                self._finish_sequentially([session], capacity)
                self.telemetry.realised_widths.append(1)
        self.telemetry.wall_ns = now() - started


class AsyncGroupedB1Executor(_Runner):
    """Throughput mode: up to `max_width` ready requests submitted as one group.

    Never waits to fill a group. Serves whatever is ready, in arrival order with the
    request id as a stable tie-break, and rotates served requests to the back so no
    request can be starved.
    """

    name = "async_grouped_b1"

    def __init__(self, backend: DecodeBackend, telemetry: Telemetry,
                 max_width: int = MAX_GROUP_WIDTH):
        super().__init__(backend, telemetry)
        if not 1 <= max_width <= MAX_GROUP_WIDTH:
            raise ValueError(f"max_width must be between 1 and {MAX_GROUP_WIDTH}")
        self.max_width = max_width

    def run(self, sessions: list[Session], capacity: int) -> None:
        started = now()
        pending = sorted(sessions, key=lambda s: (s.arrival_ms, s.rid))
        active: list[Session] = []

        while pending or active:
            while pending and (now() - started) / 1e6 >= pending[0].arrival_ms:
                admitted = pending.pop(0)
                if not admitted.done:
                    active.append(admitted)
            if not active:
                if not pending:
                    break
                wait_ms = pending[0].arrival_ms - (now() - started) / 1e6
                if wait_ms > 0:
                    time.sleep(wait_ms / 1000.0)
                continue

            group = active[:self.max_width]
            try:
                handles = [self.backend.step(s.state, s.tokens[-1], capacity) for s in group]
                self.backend.complete(handles)
                stamp = now()
                for session, handle in zip(group, handles):
                    token, state = self.backend.read(handle)
                    session.state = state
                    self._record_token(session, token, stamp)
            except Exception as exc:                       # noqa: BLE001 - deliberate
                self._fall_back(group, capacity, exc)
                active = [s for s in active if not s.done]
                continue

            self.telemetry.realised_widths.append(len(group))
            active = active[len(group):] + [s for s in group if not s.done]

        self.telemetry.wall_ns = now() - started

    def _fall_back(self, group: list[Session], capacity: int, exc: Exception) -> None:
        """A failed group leaves its states unknown, so those requests restart.

        Tokens already produced by the failed group are discarded rather than
        trusted. That wastes work and is the only choice that keeps the output
        identical to a clean sequential run.
        """
        self.telemetry.fallbacks += 1
        self.telemetry.fallback_reasons.append(f"{type(exc).__name__}: {exc}")
        for session in group:
            session.restart(self.backend)
            if session.metrics is not None:
                session.metrics.fell_back = True
                session.metrics.token_times_ns = session.metrics.token_times_ns[:1]
                session.metrics.first_token_ns = (session.metrics.token_times_ns[0]
                                                  if session.metrics.token_times_ns else 0)
                session.metrics.generated_tokens = len(session.tokens)
                session.metrics.visible_generated_tokens = sum(
                    value not in self.backend.eos_ids for value in session.tokens)
        self._finish_sequentially(group, capacity)


def build_sessions(requests, backend: DecodeBackend, telemetry: Telemetry,
                   capacity: int) -> list[Session]:
    """Prefill every request under its own plan. Plans are applied, never chosen."""
    sessions = []
    for request in requests:
        metrics = RequestMetrics(rid=request.rid, arrival_ns=now(),
                                 prompt_tokens=len(request.prompt_ids))
        metrics.engine_start_ns = now()
        state, first = backend.prefill(request.prompt_ids, request.plan, capacity)
        session = Session(rid=request.rid, prompt_ids=list(request.prompt_ids),
                          max_tokens=request.max_tokens, plan=request.plan,
                          arrival_ms=request.arrival_ms, base_state=state,
                          state=backend.reset_state(state, len(request.prompt_ids)),
                          metrics=metrics)
        _accept_token(session, first, now(), backend.eos_ids)
        telemetry.requests.append(metrics)
        telemetry.plan_kinds.append(getattr(request.plan, "kind", "unknown"))
        sessions.append(session)
    return sessions
