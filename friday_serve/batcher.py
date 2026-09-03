"""Continuous Dynamic Micro-Batcher for Friday HTTP Server.

Coordinates up to `max_width` (default: 4) concurrent inference requests on Apple Silicon.
Executes batch-1 decode steps for all active sessions within a single Metal evaluation,
saturating Unified Memory bandwidth without modifying tensor shapes or risking cross-request
attention bleeding.

Guarantees:
1. 100% Token Identity: Identical to single-flight greedy execution.
2. Independent SSE Streams: Each client receives chunks at step cadence.
3. Zero-Contention: Only the batch worker thread interacts with MLX/Metal.
4. Critic Gate 1: Rejects excess requests with 429 when concurrency limit is reached.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import mlx.core as mx


@dataclass
class BatchSession:
    """One in-flight inference session in the batcher."""

    session_id: str
    prompt_ids: list[int]
    max_tokens: int
    knobs: Mapping[str, Any]
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    created_at: float = field(default_factory=time.time)
    cancelled: threading.Event = field(default_factory=threading.Event)

    # Runtime state set by worker
    capacity: int = 0
    state: Any = None
    curr_token: Any = None
    logical_tokens: list[int] = field(default_factory=list)
    done: bool = False
    prefill_ns: int = 0
    start_decode_ns: int = 0

    def stream(self, timeout: float = 30.0) -> Iterator[dict[str, Any]]:
        """Yield events as they are produced by the batch worker."""
        while True:
            try:
                event = self.event_queue.get(timeout=timeout)
                yield event
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield {
                    "type": "error",
                    "error": "Timeout waiting for next token from inference engine",
                }
                break

    def cancel(self) -> None:
        self.cancelled.set()


class ContinuousBatcher:
    """Worker thread running continuous micro-batching over MLX Engine."""

    def __init__(
        self,
        backend: Any,
        max_concurrency: int = 4,
        max_width: int = 4,
    ) -> None:
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.max_width = min(max_width, max_concurrency)
        self.incoming_queue: queue.Queue[BatchSession] = queue.Queue()
        self.active_sessions: list[BatchSession] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_count = 0

        self.worker_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="FridayContinuousBatcher"
        )
        self.worker_thread.start()

    def submit(
        self, prompt_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> BatchSession:
        """Submit a request to the batcher. Raises RuntimeError if concurrency limit reached."""
        with self.lock:
            # Critic Gate 1: Check total admitted + pending requests
            total_in_flight = self.active_count + self.incoming_queue.qsize()
            if total_in_flight >= self.max_concurrency:
                raise RuntimeError(
                    f"Concurrency limit reached ({self.max_concurrency}). GPU queue is full."
                )

        session = BatchSession(
            session_id=f"sess-{uuid.uuid4().hex[:8]}",
            prompt_ids=[int(t) for t in prompt_ids],
            max_tokens=max_tokens,
            knobs=knobs,
        )
        self.incoming_queue.put(session)
        return session

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

    def _leaves(self, tree: Any) -> list[Any]:
        flat = []
        if isinstance(tree, mx.array):
            return [tree]
        if isinstance(tree, dict):
            for v in tree.values():
                flat.extend(self._leaves(v))
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                flat.extend(self._leaves(v))
        return flat

    def _decode_text(self, tokens: Sequence[int]) -> str:
        if hasattr(self.backend, "_decode_text"):
            return self.backend._decode_text(tokens)
        tokenizer = getattr(self.backend, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                eos = getattr(self.backend, "eos_ids", ())
                visible = [t for t in tokens if t not in eos]
                return tokenizer.decode(visible) if visible else ""
            except Exception:
                return ""
        return ""

    def _admit_session(self, session: BatchSession) -> bool:
        """Run prefill for a newly admitted session and emit the first token."""
        if session.cancelled.is_set():
            return False

        engine = self.backend._engine(session.knobs)
        prompt_ids = session.prompt_ids
        capacity = engine._capacity(len(prompt_ids), session.max_tokens)
        session.capacity = capacity

        t0 = time.perf_counter_ns()
        radix = getattr(self.backend, "radix_cache", None)
        match_len = 0
        cached_state = None
        if radix is not None:
            match_len, cached_state, _ = radix.match_prefix(prompt_ids)

        if match_len > 0 and cached_state is not None and match_len < len(prompt_ids):
            suffix = prompt_ids[match_len:]
            warm_state = {"position": {"offset": mx.array(match_len, dtype=mx.int32)}, "layers": cached_state}
            state, hidden = engine._feed(warm_state, suffix, capacity)
            from ironmule.runtime import _project
            logits = _project(engine.model, hidden[:, -1:, :] if engine.knobs.head_skip_prefill else hidden)
            token = mx.argmax(logits[:, -1, :], axis=-1).reshape((1, 1))
            mx.eval(token, *self._leaves(state))
            mx.synchronize()
            hits = radix.hits
        else:
            state, token = engine._prefill(prompt_ids, capacity)
            mx.eval(token, *self._leaves(state))
            mx.synchronize()
            hits = getattr(getattr(engine, "prefix_cache", None), "hits", 0)
            if radix is not None and len(prompt_ids) >= 32 and hasattr(state, "__getitem__") and "layers" in state:
                radix.insert(prompt_ids, state["layers"])
        session.prefill_ns = time.perf_counter_ns() - t0

        first_tok = int(token.reshape((-1,)).item())
        session.state = state
        session.curr_token = token
        session.logical_tokens = [first_tok]
        session.start_decode_ns = time.perf_counter_ns()

        eos_ids = getattr(self.backend, "eos_ids", (1,))

        # Emit first token
        session.event_queue.put(
            {
                "type": "token",
                "token": first_tok,
                "tokens": [first_tok],
                "text": self._decode_text([first_tok]),
                "is_first": True,
                "prefill_ns": session.prefill_ns,
                "prefix_cache_hits": hits,
            }
        )

        if first_tok in eos_ids or session.max_tokens <= 1:
            session.done = True
            session.event_queue.put(
                {
                    "type": "done",
                    "total_tokens": 1,
                    "decode_ns": 0,
                    "total_ns": session.prefill_ns,
                    "knobs": dict(session.knobs),
                    "prefix_cache_hits": hits,
                    "logical_tokens": [first_tok],
                }
            )
            return False

        return True

    def _run_loop(self) -> None:
        eos_ids = getattr(self.backend, "eos_ids", (1,))

        while not self.stop_event.is_set():
            # 1. Admit new sessions up to max_width
            while len(self.active_sessions) < self.max_width:
                try:
                    timeout = 0.005 if self.active_sessions else 0.05
                    new_session = self.incoming_queue.get(timeout=timeout)
                except queue.Empty:
                    break

                try:
                    admitted = self._admit_session(new_session)
                    if admitted:
                        self.active_sessions.append(new_session)
                except Exception as exc:
                    new_session.event_queue.put({"type": "error", "error": str(exc)})

            with self.lock:
                self.active_count = len(self.active_sessions)

            if not self.active_sessions:
                time.sleep(0.001)
                continue

            # Filter out cancelled sessions
            self.active_sessions = [s for s in self.active_sessions if not s.cancelled.is_set()]
            if not self.active_sessions:
                continue

            # 2. Grouped Decode Step with Cadence Bundling across all active sessions
            group = self.active_sessions[: self.max_width]
            cadence = min(4, min(s.max_tokens - len(s.logical_tokens) for s in group))
            if cadence < 1:
                cadence = 1

            session_generated: dict[int, list[Any]] = {id(s): [] for s in group}
            flat_eval = []

            for _ in range(cadence):
                for s in group:
                    engine = self.backend._engine(s.knobs)
                    body = engine._body(s.capacity, 1)
                    out = body(s.curr_token, s.state)
                    picks = engine._picks(out)
                    next_tok, next_state = picks[:, -1:], out[1]
                    s.curr_token = next_tok
                    s.state = next_state
                    session_generated[id(s)].append(next_tok)
                    flat_eval.append(next_tok)
                    flat_eval.extend(self._leaves(next_state))

            # 3. Unified Metal Evaluation: Submit all grouped steps together
            mx.async_eval(*flat_eval)
            mx.eval(*flat_eval)
            mx.synchronize()

            # 4. Process Tokens & Dispatch to individual SSE queues
            for s in group:
                engine = self.backend._engine(s.knobs)
                tokens_for_s = session_generated[id(s)]
                hit_eos = False

                for next_tok in tokens_for_s:
                    tok_val = int(next_tok.reshape((-1,)).item())
                    s.logical_tokens.append(tok_val)
                    hit_eos = tok_val in eos_ids
                    at_limit = len(s.logical_tokens) >= s.max_tokens

                    tok_text = self._decode_text([tok_val])
                    s.event_queue.put(
                        {
                            "type": "token",
                            "token": tok_val,
                            "tokens": [tok_val],
                            "text": tok_text,
                            "is_first": False,
                        }
                    )

                    if hit_eos or at_limit:
                        s.done = True
                        decode_ns = time.perf_counter_ns() - s.start_decode_ns
                        hits = getattr(getattr(engine, "prefix_cache", None), "hits", 0)
                        s.event_queue.put(
                            {
                                "type": "done",
                                "total_tokens": len(s.logical_tokens),
                                "decode_ns": decode_ns,
                                "total_ns": s.prefill_ns + decode_ns,
                                "knobs": dict(s.knobs),
                                "prefix_cache_hits": hits,
                                "logical_tokens": list(s.logical_tokens),
                            }
                        )
                        break

            # Keep only active, non-done sessions
            self.active_sessions = [s for s in self.active_sessions if not s.done]
            with self.lock:
                self.active_count = len(self.active_sessions)
