"""The runtime: plans, modes, a small public API, and a safe sequential fallback.

Two service modes, chosen by the caller and never by the runtime:

  InteractiveMode   sequential batch-1. Lowest latency for one caller.
  ThroughputMode    grouped batch-1 at width <= 4. Higher aggregate throughput and
                    much lower service TTFT under concurrency, at a measured cost in
                    median per-request latency.

E15 and E16 measured that trade: +15% to +17% throughput, median latency +26% to
+31%, tail latency -8% to -17%, and service TTFT falling roughly tenfold. Neither
mode is a default that suits everything, which is why both are explicit.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .executor import (MAX_GROUP_WIDTH, AsyncGroupedB1Executor, SequentialExecutor,
                       build_sessions)
from .fingerprint import build as build_fingerprint, usable
from .plans import ExecutionPlan, ReusableSessionPlan, StrictOneShotPlan, plan_kind
from .telemetry import Telemetry

CAPACITY_CEILING = 8192          # refuse rather than allocate an unbounded KV cache


@dataclass
class Request:
    prompt_ids: Sequence[int]
    max_tokens: int = 64
    plan: ExecutionPlan = field(default_factory=StrictOneShotPlan)
    arrival_ms: float = 0.0
    rid: str = ""

    def __post_init__(self):
        if not self.rid:
            self.rid = uuid.uuid4().hex[:8]
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass
class Result:
    rid: str
    tokens: list[int]
    text: str
    stop_reason: str
    metrics: dict


class InteractiveMode:
    name = "interactive"

    def executor(self, backend, telemetry):
        return SequentialExecutor(backend, telemetry)


class ThroughputMode:
    name = "throughput"

    def __init__(self, max_width: int = MAX_GROUP_WIDTH):
        self.max_width = max_width

    def executor(self, backend, telemetry):
        return AsyncGroupedB1Executor(backend, telemetry, max_width=self.max_width)


class MLXBackend:
    """Adapts `ironmule.runtime.Engine` to the executor's `DecodeBackend` protocol."""

    def __init__(self, engine, eos_ids: tuple[int, ...]):
        self.engine = engine
        self.eos_ids = tuple(eos_ids)

    def capacity_for(self, prompt_lens: Sequence[int], max_tokens: int) -> int:
        needed = max(prompt_lens) + max_tokens + 8
        capacity = ((needed + 63) // 64) * 64
        if capacity > CAPACITY_CEILING:
            raise ValueError(f"capacity {capacity} exceeds the ceiling {CAPACITY_CEILING}")
        return capacity

    def prefill(self, prompt_ids, plan, capacity):
        plan.apply(self.engine)
        try:
            state, token = self.engine._prefill(list(prompt_ids), capacity)
        finally:
            plan.release(self.engine)
        return state, int(token.reshape((-1,)).item())

    def reset_state(self, base_state, offset: int):
        import mlx.core as mx
        return {"position": {"offset": mx.array(offset, dtype=mx.int32)},
                "layers": [{"keys": l["keys"], "values": l["values"]}
                           for l in base_state["layers"]]}

    def step(self, state, token: int, capacity: int):
        import mlx.core as mx
        body = self.engine._body(capacity, 1)
        out = body(mx.array([[token]]), state)
        pick = mx.argmax(out[0][:, -1, :].astype(mx.float32), axis=-1)
        return out, pick

    def complete(self, handles) -> None:
        import mlx.core as mx
        from .runtime import _leaves
        flat = [pick for _, pick in handles]
        flat += [leaf for out, _ in handles for leaf in _leaves(out[1])]
        mx.async_eval(*flat)
        mx.eval(*flat)
        mx.synchronize()

    def read(self, handle):
        out, pick = handle
        return int(pick.item()), out[1]

    def kv_hash(self, state, offset: int) -> str:
        import hashlib
        import mlx.core as mx
        import numpy as np
        digest = hashlib.sha256()
        for layer in state["layers"]:
            for name in ("keys", "values"):
                arr = layer[name][..., :offset, :]
                view = {2: mx.uint16, 4: mx.uint32}[arr.dtype.size]
                digest.update(np.asarray(arr.view(view)).tobytes())
        return digest.hexdigest()


class Runtime:
    """A loaded model plus a service mode. Plans travel with requests."""

    def __init__(self, engine, tokenizer, mode=None, model_id: str = "",
                 quantisation: Any = None):
        from .tune import _eos_ids
        self.engine = engine
        self.tokenizer = tokenizer
        self.mode = mode or InteractiveMode()
        self.model_id = model_id
        self.quantisation = quantisation
        self.backend = MLXBackend(engine, _eos_ids(tokenizer))
        self.telemetry = Telemetry(mode=self.mode.name)

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, model_id: str | None = None, mode=None, use_tuned_profile: bool = True):
        from .runtime import BASELINE, Knobs
        from .tune import DEFAULT_MODEL, load_engine, load_profile
        model_id = model_id or DEFAULT_MODEL
        knobs = BASELINE
        if use_tuned_profile:
            profile = load_profile(model_id)
            if profile:
                knobs = Knobs(**profile["knobs"])
        engine, tokenizer = load_engine(model_id, knobs)
        return cls(engine, tokenizer, mode=mode, model_id=model_id)

    # -- helpers --------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return list(self.tokenizer.encode(rendered, add_special_tokens=False))

    def session_plan(self, shared_prefix: str, name: str = "session") -> ReusableSessionPlan:
        """Build a reusable-session plan from the shared part of a prompt."""
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": shared_prefix + "\n\n@@CUT@@"}],
            tokenize=False, add_generation_prompt=True).split("@@CUT@@")[0]
        return ReusableSessionPlan(self.tokenizer.encode(rendered, add_special_tokens=False),
                                   name=name)

    # -- serving --------------------------------------------------------------
    def serve(self, requests: Sequence[Request]) -> list[Result]:
        import mlx.core as mx
        if not requests:
            return []
        for request in requests:
            if plan_kind(request.plan) not in ("strict_one_shot", "reusable_session"):
                raise ValueError(f"unknown execution plan: {request.plan!r}")

        self.telemetry = Telemetry(mode=self.mode.name)
        capacity = self.backend.capacity_for([len(r.prompt_ids) for r in requests],
                                             max(r.max_tokens for r in requests))
        sessions = build_sessions(requests, self.backend, self.telemetry, capacity)

        executor = self.mode.executor(self.backend, self.telemetry)
        try:
            executor.run(sessions, capacity)
        except Exception as exc:                          # noqa: BLE001 - deliberate
            # Last-resort safety net: a whole-executor failure restarts every
            # unfinished request on the sequential path from its prefill state.
            self.telemetry.fallbacks += 1
            self.telemetry.fallback_reasons.append(f"executor: {type(exc).__name__}: {exc}")
            for session in sessions:
                if not session.done:
                    session.restart(self.backend)
                    if session.metrics is not None:
                        session.metrics.fell_back = True
            SequentialExecutor(self.backend, self.telemetry).run(sessions, capacity)

        self.telemetry.peak_memory_bytes = mx.get_peak_memory()
        results = []
        for session in sessions:
            visible = [t for t in session.tokens if t not in self.backend.eos_ids]
            results.append(Result(rid=session.rid, tokens=list(session.tokens),
                                  text=self.tokenizer.decode(visible),
                                  stop_reason=session.stop_reason,
                                  metrics=session.metrics.as_dict()))
        return results

    def generate(self, prompt: str | None = None, *, prompt_ids: Sequence[int] | None = None,
                 plan: ExecutionPlan | None = None, max_tokens: int = 64) -> Result:
        ids = list(prompt_ids) if prompt_ids is not None else self.encode(prompt or "")
        request = Request(prompt_ids=ids, max_tokens=max_tokens,
                          plan=plan or StrictOneShotPlan())
        return self.serve([request])[0]

    # -- validity -------------------------------------------------------------
    def fingerprint(self, plan: ExecutionPlan | None = None,
                    workload: dict | None = None) -> dict:
        return build_fingerprint(self.model_id, self.quantisation,
                                 plan_kind(plan or StrictOneShotPlan()),
                                 self.mode.name, workload)

    def revalidate(self, store: Path | None = None, plan: ExecutionPlan | None = None,
                   workload: dict | None = None) -> dict:
        """Compare the current identity against the last one recorded here."""
        from .hw import STORE
        path = store or (STORE / "runtime_fingerprint.json")
        current = self.fingerprint(plan, workload)
        stored = None
        if path.is_file():
            try:
                stored = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                stored = None
        if stored is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(current, indent=1, sort_keys=True, default=str))
            return {"verdict": "recorded_first_fingerprint", "current": current}
        ok, why = usable(stored, current)
        verdict = ("valid" if ok and not why["drifted"]
                   else "valid_with_workload_drift" if ok else "revalidation_required")
        if not ok:
            path.write_text(json.dumps(current, indent=1, sort_keys=True, default=str))
        return {"verdict": verdict, "current": current, "stored_digest": stored.get("digest"),
                **why}
