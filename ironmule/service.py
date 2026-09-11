"""The runtime: plans, modes, a small public API, and a safe sequential fallback.

Two service modes, chosen by the caller and never by the runtime:

  InteractiveMode   sequential batch-1. Lowest latency for one caller.
  ThroughputMode    grouped batch-1 at width <= 4. Higher aggregate throughput and
                    much lower service TTFT under concurrency, at a measured cost in
                    median per-request latency.
  PairedThroughputMode
                    opt-in research mode: two ready requests share one weight sweep.
                    Never a default, admitted only inside its qualified box.
  AutomaticMode     opt-in rule-based choice between the two above, from the tuned
                    profile's service-strategy record. Off unless asked for, and it
                    keeps the established mode wherever the record does not apply.

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
from .model_identity import ModelIdentity, ModelIdentityError
from .plans import ExecutionPlan, ReusableSessionPlan, StrictOneShotPlan, plan_kind
from .telemetry import Telemetry

CAPACITY_CEILING = 8192          # refuse rather than allocate an unbounded KV cache


#: What a caller is optimising for this request. `None` means they did not say, and an
#: unspecified request keeps whatever the runtime was already doing -- which is how every
#: call written before B56 keeps its behaviour exactly.
OBJECTIVES = ("latency", "throughput")


@dataclass
class Request:
    prompt_ids: Sequence[int]
    max_tokens: int = 64
    plan: ExecutionPlan = field(default_factory=StrictOneShotPlan)
    arrival_ms: float = 0.0
    rid: str = ""
    objective: str | None = None

    def __post_init__(self):
        if not self.rid:
            self.rid = uuid.uuid4().hex[:8]
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.objective is not None and self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES} or None, "
                             f"got {self.objective!r}")


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


class PairedThroughputMode:
    """Opt-in research mode: two ready requests share one weight sweep.

    Off unless a caller names it. Admission runs once, on the first executor built for a
    loaded engine, and refuses loudly outside the qualified model, hardware, library and
    projection set: a caller who asked for this must not silently get something else.
    Measured on a pair of requests only; a lone request runs the ordinary single path and
    never waits for a partner. See `docs/BACKLOG.md` entries `B45` and `B46`.
    """

    name = "paired_throughput"

    def __init__(self, *, share: bool = True, share_aligned: bool = False):
        # `share_aligned` extends sharing to o_proj and down_proj. Off by default: it is
        # qualified at kernel level (`B48`) but has not earned a product gate.
        self.share = share
        self.share_aligned = share_aligned
        self.admission: dict | None = None
        self._executor = None

    def executor(self, backend, telemetry):
        from .paired_research import PairedGroupedExecutor, admit
        if self.admission is None:
            self.admission = admit(backend.engine.model,
                                   getattr(backend.engine, "model_identity", None))
        self._executor = PairedGroupedExecutor(backend, telemetry, share=self.share,
                                               share_aligned=self.share_aligned)
        return self._executor

    def status(self) -> dict:
        """Disabled, admitted, and how the last run actually executed."""

        executor = self._executor
        return {
            "mode": self.name,
            "enabled": True,
            "sharing": self.share,
            "sharing_aligned_projections": self.share_aligned,
            "admitted": self.admission is not None,
            "admitted_projections": (self.admission or {}).get("admitted", 0),
            "paired_steps": getattr(executor, "paired_steps", 0),
            "solo_steps_no_partner": getattr(executor, "solo_steps", 0),
        }


def paired_status(mode) -> dict:
    """One shape of status for any mode, so `disabled` is a real answer."""

    reporter = getattr(mode, "status", None)
    if reporter is None:
        return {"mode": getattr(mode, "name", "unknown"), "enabled": False,
                "sharing": False, "sharing_aligned_projections": False,
                "admitted": False, "admitted_projections": 0,
                "paired_steps": 0, "solo_steps_no_partner": 0}
    return reporter()


class _AutomaticExecutor:
    """Chooses once per `serve`, before any decode step, then gets out of the way.

    The number of ready requests is only known when the sessions arrive, which is also
    the last moment at which nothing has been decoded yet. Choosing here means the state
    change happens at a boundary the shipped executor already treats as safe, and no
    check is added to the per-token path.
    """

    name = "automatic"

    def __init__(self, mode, backend, telemetry):
        self.mode, self.backend, self.telemetry = mode, backend, telemetry

    def run(self, sessions, capacity) -> None:
        delegate = self.mode.delegate_for(sessions, self.backend, self.telemetry)
        delegate.run(sessions, capacity)


class AutomaticMode:
    """Rule-based choice between the established mode and the qualified paired path.

    It selects nothing unless the caller opted in *and* the tuned profile carries a
    complete service-strategy record admitting this machine, model, library build and
    number of ready requests. Anything else keeps the established mode, which is what an
    absent record, an older profile or an unknown configuration all mean.

    Only facts known at decision time enter: how many requests are ready now, and the
    identity of the machine, model and libraries. The response a request will produce is
    not used, and no length is predicted.
    """

    name = "automatic"

    def __init__(self, profile: dict | None = None, *, opt_in: bool = False,
                 identity_sha256: str | None = None, fingerprint: str | None = None,
                 mlx: str = "", mlx_lm: str = ""):
        self.profile = profile
        self.opt_in = bool(opt_in)
        self.identity_sha256 = identity_sha256
        self.fingerprint = fingerprint
        self.mlx = mlx
        self.mlx_lm = mlx_lm
        self.decision: dict | None = None
        # One decision per `serve`, counted so a run can show that nothing was added to
        # the per-token path.
        self.decisions_made = 0
        self._established = ThroughputMode()
        self._paired = PairedThroughputMode()

    def executor(self, backend, telemetry):
        return _AutomaticExecutor(self, backend, telemetry)

    @staticmethod
    def ready_now(sessions) -> int:
        """How many requests are actually ready to take a step, not how many exist.

        A request with a later arrival is held back by the scheduler, and one that
        finished during prefill never takes a step at all. Counting the whole group
        would claim a partner that is not there.
        """

        return sum(1 for session in sessions
                   if getattr(session, "arrival_ms", 0.0) <= 0.0
                   and not getattr(session, "done", False))

    def delegate_for(self, sessions, backend, telemetry):
        """The chosen executor, and the decision that produced it."""

        from .service_strategy import STRATEGY_PAIRED, select

        sessions = list(sessions)
        ready_requests = self.ready_now(sessions)
        decision = select(self.profile, opt_in=self.opt_in,
                          ready_requests=ready_requests,
                          identity_sha256=self.identity_sha256,
                          fingerprint=self.fingerprint,
                          mlx=self.mlx, mlx_lm=self.mlx_lm)
        decision["ready_requests"] = ready_requests
        decision["group_requests"] = len(sessions)
        self.decisions_made += 1
        if decision["strategy"] == STRATEGY_PAIRED:
            try:
                executor = self._paired.executor(backend, telemetry)
            except Exception as exc:                      # noqa: BLE001 - deliberate
                # The profile said yes, the loaded model said no. Refusing to the
                # established mode is the documented answer for an unknown condition;
                # falling through to the sequential net would be a silent downgrade.
                decision["strategy"] = "throughput"
                decision["reason"] = (
                    f"profile admitted the paired path, the model refused it: "
                    f"{type(exc).__name__}: {exc}"
                )
                decision["evidence_run_ids"] = []
            else:
                self.decision = decision
                return executor
        self.decision = decision
        return self._established.executor(backend, telemetry)

    def status(self) -> dict:
        """What was chosen, why, and the runs that authorise it."""

        from .service_strategy import STRATEGY_PAIRED

        decision = self.decision or {}
        chose_paired = decision.get("strategy") == STRATEGY_PAIRED
        paired = self._paired.status() if chose_paired else paired_status(self._established)
        return {
            **paired,
            "mode": self.name,
            "opt_in": self.opt_in,
            "strategy": decision.get("strategy", "throughput"),
            "reason": decision.get("reason", "nothing served yet"),
            "record_present": bool(decision.get("record_present", False)),
            "ready_requests": decision.get("ready_requests", 0),
            "group_requests": decision.get("group_requests", 0),
            "decisions_made": self.decisions_made,
            "evidence_run_ids": list(decision.get("evidence_run_ids", [])),
            "correctness_contract": decision.get("correctness_contract", ""),
        }


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
        from .runtime import _copy_state_layers
        return {"position": {"offset": mx.array(offset, dtype=mx.int32)},
                "layers": _copy_state_layers(base_state["layers"])}

    def step(self, state, token: int, capacity: int):
        import mlx.core as mx
        body = self.engine._body(capacity, 1)
        out = body(mx.array([[token]]), state)
        if getattr(getattr(self.engine, "knobs", None), "fused_argmax", False):
            pick = out[0]
        else:
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
        import numpy as np
        from .runtime import _state_layer_kind
        digest = hashlib.sha256()
        kinds = [_state_layer_kind(layer) for layer in state["layers"]]
        hybrid = "arrays" in kinds
        for layer, kind in zip(state["layers"], kinds):
            if kind == "kv":
                for name in ("keys", "values"):
                    arr = layer[name][..., :offset, :]
                    if hybrid:
                        digest.update(name.encode("ascii"))
                        digest.update(str(arr.shape).encode("ascii"))
                        digest.update(str(arr.dtype).encode("ascii"))
                        digest.update(_raw_bit_bytes(arr))
                    else:
                        # Keep the established all-KV digest byte-for-byte stable.
                        digest.update(_raw_bit_bytes(arr, legacy=True))
            elif kind == "arrays":
                if not hybrid:
                    raise TypeError("unsupported cache state layer")
                digest.update(b"arrays")
                for index, arr in enumerate(layer["arrays"]):
                    digest.update(str(index).encode("ascii"))
                    if arr is None:
                        digest.update(b"none")
                    else:
                        digest.update(str(arr.shape).encode("ascii"))
                        digest.update(str(arr.dtype).encode("ascii"))
                        digest.update(_raw_bit_bytes(arr))
            else:
                raise TypeError("unsupported cache state layer")
        return digest.hexdigest()


def _raw_bit_bytes(arr, *, legacy: bool = False) -> bytes:
    """Return MLX array bits without numeric conversion (including bfloat16)."""
    import mlx.core as mx
    import numpy as np

    views = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32}
    try:
        view = views[arr.dtype.size]
    except KeyError as exc:
        raise TypeError(f"unsupported cache dtype size: {arr.dtype.size}") from exc
    # `legacy` documents the all-KV path's historical view-based digest.  It is
    # intentionally the same operation as the hybrid path for supported dtypes.
    del legacy
    return np.asarray(arr.view(view)).tobytes()


class Runtime:
    """A loaded model plus a service mode. Plans travel with requests."""

    def __init__(self, engine, tokenizer, mode=None, model_id: str = "",
                 quantisation: Any = None, model_identity: ModelIdentity | None = None):
        try:
            from .tune import _eos_ids
            engine_identity = getattr(engine, "model_identity", None)
            if (model_identity is not None and engine_identity is not None
                    and model_identity != engine_identity):
                raise ModelIdentityError(
                    "explicit Runtime identity conflicts with loaded Engine identity"
                )
            identity = model_identity or engine_identity
            if identity is not None and not isinstance(identity, ModelIdentity):
                raise ModelIdentityError("Runtime model identity has the wrong type")
            if identity is not None and model_id and model_id != identity.model_id:
                local = Path(model_id).expanduser()
                local_id = f"local:{local.resolve().name}" if local.is_dir() else None
                if local_id != identity.model_id:
                    raise ModelIdentityError("Runtime model_id conflicts with exact model identity")
            if identity is not None and quantisation is not None and quantisation != identity.quantisation:
                raise ModelIdentityError("Runtime quantisation conflicts with exact model identity")
            self.engine = engine
            self.tokenizer = tokenizer
            self.mode = mode or InteractiveMode()
            self.model_identity = identity
            self.model_id = identity.model_id if identity is not None else model_id
            self.quantisation = identity.quantisation if identity is not None else quantisation
            self.backend = MLXBackend(engine, _eos_ids(tokenizer))
            self.telemetry = Telemetry(mode=self.mode.name)
        except BaseException as exc:
            close = getattr(engine, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as cleanup_error:
                    exc.add_note(
                        "Runtime engine cleanup failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    def close(self) -> None:
        """Release the loaded engine and any process-global state it owns."""
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, model_id: str | None = None, mode=None, use_tuned_profile: bool = True,
             revision: str | None = None, automatic_service_mode: bool = False):
        """Load a model, and only on request let the profile choose the service mode.

        `automatic_service_mode=True` is the opt-in. Without it nothing about the stored
        profile can change which mode serves a request, which is why an older profile and
        a profile carrying a new record behave identically until a caller asks.
        """

        from .runtime import BASELINE, Knobs
        from .tune import DEFAULT_MODEL, load_engine, load_profile, resolve_local_model
        model_id = model_id or DEFAULT_MODEL
        if automatic_service_mode and mode is not None:
            raise ValueError("automatic_service_mode replaces an explicit mode; pass one")
        resolved = resolve_local_model(model_id, revision)
        knobs = BASELINE
        profile = None
        if use_tuned_profile or automatic_service_mode:
            profile = load_profile(
                model_id, revision=revision, model_identity=resolved.identity
            )
            if profile and use_tuned_profile:
                knobs = Knobs(**profile["knobs"])
        if automatic_service_mode:
            mode = cls._automatic_mode(profile, resolved.identity)
        engine, tokenizer = load_engine(
            model_id, knobs, revision=revision, resolved_source=resolved
        )
        return cls(
            engine, tokenizer, mode=mode, model_id=resolved.identity.model_id,
            model_identity=resolved.identity,
        )

    @staticmethod
    def _automatic_mode(profile: dict | None, identity):
        """The opt-in mode, told once what this machine and library build are."""

        import mlx.core as mx
        import mlx_lm

        from .hw import fingerprint
        return AutomaticMode(profile, opt_in=True,
                             identity_sha256=identity.identity_sha256,
                             fingerprint=fingerprint(),
                             mlx=mx.__version__, mlx_lm=mlx_lm.__version__)

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
    def serve(self, requests: Sequence[Request],
              dispatch_ns: int | None = None) -> list[Result]:
        """Serve one group of requests. `dispatch_ns` says when the caller handed them
        over, for a caller that splits one dispatch across several calls; it changes
        recorded arrival only, never scheduling or output."""
        import mlx.core as mx
        if not requests:
            return []
        for request in requests:
            if plan_kind(request.plan) not in ("strict_one_shot", "reusable_session"):
                raise ValueError(f"unknown execution plan: {request.plan!r}")

        self.telemetry = Telemetry(mode=self.mode.name)
        capacity = self.backend.capacity_for([len(r.prompt_ids) for r in requests],
                                             max(r.max_tokens for r in requests))
        sessions = build_sessions(requests, self.backend, self.telemetry, capacity,
                                  dispatch_ns=dispatch_ns)

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
        if self.model_identity is None:
            raise ModelIdentityError("Runtime fingerprint requires exact model identity")
        return build_fingerprint(self.model_id, self.quantisation,
                                 plan_kind(plan or StrictOneShotPlan()),
                                 self.mode.name, workload,
                                 model_identity=self.model_identity)

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
