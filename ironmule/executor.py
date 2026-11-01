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

import json
import os
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


class TokenStream:
    """Hands each output token of one request to `emit` once, in order (SERVE6).

    A replay of the same request (a fallback restart, a comparison arm) must reproduce what was
    already emitted: a streamed token cannot be taken back, so a different one raises.
    """

    def __init__(self, emit, cancelled=None):
        # RSI3-LIVE-P2: `cancelled()`, when given, ends the request at its next token boundary;
        # a caller that cancels never receives the rest, and the group's other requests go on.
        self.emit, self.tokens, self.cancelled = emit, [], cancelled

    def accept(self, index: int, token: int) -> None:
        if index < len(self.tokens):
            if self.tokens[index] != token:
                raise RuntimeError("streamed output diverged on replay")
            return
        if index != len(self.tokens):
            raise RuntimeError("streamed output skipped a token")
        self.tokens.append(token)
        self.emit(token)


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
    stream: TokenStream | None = None
    capacity: int = 0  # X1: this request's own KV capacity; 0 leaves it to the run's

    def restart(self, backend: DecodeBackend) -> None:
        """Return to the post-prefill state. Used only by the fallback path."""
        self.state = backend.reset_state(self.base_state, len(self.prompt_ids))
        self.tokens = self.tokens[:1]
        self.done = False
        self.stop_reason = ""


def _capacity(session: Any, capacity: int) -> int:
    """The KV capacity a session decodes at: its own when it has one (X1), else the run's."""
    return getattr(session, "capacity", 0) or capacity


def _accept_token(session: Session, token: int, when: int,
                  eos_ids: Sequence[int]) -> None:
    """Accept one physical output token and apply the generation contract."""
    session.tokens.append(token)
    if session.stream is not None:
        session.stream.accept(len(session.tokens) - 1, token)
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


def _stop_if_cancelled(session: Session) -> bool:
    """End a caller-cancelled request now instead of decoding it to its limit (RSI3-LIVE-P2).

    Only a stream given a `cancelled` predicate can stop; every other request runs exactly as
    before. Grouped batch-1 keeps each other request's tokens whatever the group width."""
    stream = session.stream
    if session.done or stream is None or stream.cancelled is None or not stream.cancelled():
        return False
    session.done, session.stop_reason = True, "cancelled"
    if session.metrics is not None:
        session.metrics.finished_ns = now()
        session.metrics.stop_reason = "cancelled"
    return True


class _Runner:
    def __init__(self, backend: DecodeBackend, telemetry: Telemetry):
        self.backend = backend
        self.telemetry = telemetry

    def _record_token(self, session: Session, token: int, when: int) -> None:
        _accept_token(session, token, when, self.backend.eos_ids)

    def _finish_sequentially(self, sessions: list[Session], capacity: int) -> None:
        every = getattr(self.backend, "readback", 1) if hasattr(self.backend, "steps") else 1
        for session in sessions:
            own = _capacity(session, capacity)
            if not session.done and len(session.tokens) == 1 and hasattr(self.backend, "speculate"):
                drafted = self.backend.speculate(session.prompt_ids, session.tokens[0], session.state,
                                                 session.max_tokens, own)
                if drafted is not None:
                    tokens, state = drafted
                    when = now()
                    for token in tokens:
                        self._record_token(session, token, when)
                        if session.done or _stop_if_cancelled(session):
                            break
                    session.state = self.backend.rewind(state, len(session.prompt_ids) + len(session.tokens) - 1)
                    continue
            while not session.done and not _stop_if_cancelled(session):
                count = min(every, session.max_tokens - len(session.tokens))
                chain = self.backend.steps(session.state, session.tokens[-1], own, count) if count > 1 else None
                if chain:
                    self._finish_chain(session, chain)
                    continue
                handle = self.backend.step(session.state, session.tokens[-1], own)
                self.backend.complete([handle])
                token, state = self.backend.read(handle)
                session.state = state
                self._record_token(session, token, now())

    def _finish_chain(self, session: Session, chain: list) -> None:
        """Accept a chained batch up to EOS or the length limit, then rewind the cache."""
        self.backend.complete_chain(chain)
        self._accept_chain(session, chain, now())

    def _accept_chain(self, session: Session, chain: list, when: int) -> None:
        # A chain read ahead; a stream cancelled at one of its tokens still ends right there (RSI3-LIVE-P2).
        for handle in chain:
            token, _ = self.backend.read(handle)
            self._record_token(session, token, when)
            if session.done or _stop_if_cancelled(session):
                break
        session.state = self.backend.rewind(chain[-1][0][1], len(session.prompt_ids) + len(session.tokens) - 1)


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

    def _step_group(self, group: list[Session], capacity: int):
        """One submission per session. A subclass may submit a group as one step."""

        return [self.backend.step(s.state, s.tokens[-1], _capacity(s, capacity)) for s in group]

    def _chain_group(self, group: list[Session], capacity: int):
        """Chained readback per session (SERVE1) when every session can chain; else None.

        Only the default one-submission-per-session grouping chains; a subclass that
        submits a group as one step keeps its own stepping.
        """
        every = getattr(self.backend, "readback", 1) if hasattr(self.backend, "steps") else 1
        if every <= 1 or type(self)._step_group is not AsyncGroupedB1Executor._step_group:
            return None
        chains = []
        for session in group:
            count = min(every, session.max_tokens - len(session.tokens))
            chain = (self.backend.steps(session.state, session.tokens[-1], _capacity(session, capacity), count)
                     if count > 1 else None)
            if not chain:
                return None
            chains.append(chain)
        return chains

    def run(self, sessions: list[Session], capacity: int, admit=None) -> None:
        """Serve `sessions`; `admit()` (SERVE8), when given, returns sessions that arrived while
        this run was going, already prefilled (each at its own capacity, which fits this one), and
        they join the rotation before the next round. The run ends when nothing is active, waiting
        or newly admitted."""
        started = now()
        pending = sorted(sessions, key=lambda s: (s.arrival_ms, s.rid))
        active: list[Session] = []
        trace_path = os.environ.get("IRONMULE_TRACE_ROUNDS")  # SERVE8-T: opt-in per-round timing

        while True:
            t0 = now()
            joined = 0
            memory0 = _memory() if trace_path else None
            if admit is not None:
                fresh = [session for session in admit() if not session.done]
                joined = len(fresh)
                active.extend(fresh)
            t1 = now()
            memory1 = _memory() if trace_path else None
            if not (pending or active):
                break
            while pending and (now() - started) / 1e6 >= pending[0].arrival_ms:
                admitted = pending.pop(0)
                if not admitted.done:
                    active.append(admitted)
            active = [s for s in active if not _stop_if_cancelled(s)]
            if not active:
                if not pending:
                    break
                wait_ms = pending[0].arrival_ms - (now() - started) / 1e6
                if wait_ms > 0:
                    time.sleep(wait_ms / 1000.0)
                continue

            group = active[:self.max_width]
            try:
                chains = self._chain_group(group, capacity)
                if chains:
                    self.backend.complete_chains(chains)
                    stamp = now()
                    for session, chain in zip(group, chains):
                        self._accept_chain(session, chain, stamp)
                else:
                    handles = self._step_group(group, capacity)
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
            if trace_path:
                with open(trace_path, "a") as stream:
                    stream.write(json.dumps({"t_ms": (t0 - started) / 1e6, "admit_ms": (t1 - t0) / 1e6,
                                             "joined": joined, "width": len(group), "active": len(active),
                                             "chained": bool(chains), "step_ms": (now() - t1) / 1e6,
                                             "capacity": capacity, "memory_before_admit": memory0,
                                             "memory_after_admit": memory1, "memory_after_step": _memory()})
                                 + "\n")
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


def _memory() -> dict[str, int] | None:
    """SERVE8-T2: MLX allocator counters (active, cache, peak bytes) for the round trace; None without MLX."""
    try:
        import mlx.core as mx
        return {"active": mx.get_active_memory(), "cache": mx.get_cache_memory(), "peak": mx.get_peak_memory()}
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def build_sessions(requests, backend: DecodeBackend, telemetry: Telemetry,
                   capacity: int, dispatch_ns: int | None = None) -> list[Session]:
    """Prefill every request under its own plan. Plans are applied, never chosen.

    Each request gets the KV capacity it would get alone, from its own prompt length and output
    limit, never the group's `capacity` (X1): on CUDA the cache length moves decode logits, so a
    shared capacity made a request's tokens depend on what it was grouped with. The group's
    capacity, sized to the longest prompt and largest limit, still bounds every one of them.

    `dispatch_ns` is when the caller handed these requests over. It defaults to now,
    which is right when a dispatch is one `serve`. A caller that splits one dispatch
    into several `serve` calls must pass the original timestamp, or every service TTFT
    and latency after the first call would be measured from a moment the caller never
    saw and the wait between the calls would vanish from the record.
    """
    sessions = []
    for request in requests:
        metrics = RequestMetrics(rid=request.rid,
                                 arrival_ns=now() if dispatch_ns is None else dispatch_ns,
                                 prompt_tokens=len(request.prompt_ids))
        metrics.engine_start_ns = now()
        stream = getattr(request, "stream", None)
        if stream is not None and stream.cancelled is not None and stream.cancelled():
            # RSI3-LIVE-P2: cancelled before its prefill; it computes nothing.
            metrics.finished_ns, metrics.stop_reason = now(), "cancelled"
            session = Session(rid=request.rid, prompt_ids=list(request.prompt_ids), max_tokens=request.max_tokens,
                              plan=request.plan, arrival_ms=request.arrival_ms, metrics=metrics, stream=stream,
                              done=True, stop_reason="cancelled")
            telemetry.requests.append(metrics)
            telemetry.plan_kinds.append(getattr(request.plan, "kind", "unknown"))
            sessions.append(session)
            continue
        own = backend.capacity_for([len(request.prompt_ids)], request.max_tokens)
        state, first = backend.prefill(request.prompt_ids, request.plan, own)
        session = Session(rid=request.rid, prompt_ids=list(request.prompt_ids),
                          max_tokens=request.max_tokens, plan=request.plan,
                          arrival_ms=request.arrival_ms, base_state=state,
                          state=backend.reset_state(state, len(request.prompt_ids)),
                          metrics=metrics, stream=getattr(request, "stream", None), capacity=own)
        _accept_token(session, first, now(), backend.eos_ids)
        telemetry.requests.append(metrics)
        telemetry.plan_kinds.append(getattr(request.plan, "kind", "unknown"))
        sessions.append(session)
    return sessions
