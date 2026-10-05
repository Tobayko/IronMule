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
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from .executor import (MAX_GROUP_WIDTH, AsyncGroupedB1Executor, SequentialExecutor,
                       build_sessions)
from .fingerprint import build as build_fingerprint, usable
from .model_identity import ModelIdentity, ModelIdentityError, canonical_json
from .plans import ExecutionPlan, ReusableSessionPlan, StrictOneShotPlan, plan_kind
from .telemetry import Telemetry

CAPACITY_CEILING = 8192          # refuse rather than allocate an unbounded KV cache


def runtime_identity(runtime: "Runtime", *, action_contract: Mapping | None = None,
                     hardware_binding: Mapping | None = None) -> dict[str, Any]:
    """Bind online control to this loaded engine, outside the request path.

    Identity construction reads source files and backend facts once. It never
    downloads a model or grants a performance qualification. A runtime without
    exact model identity cannot use the online controller.
    """
    from friday_evidence.canonical import canonical_sha256
    import mlx.core as mx
    from .hw import device_identity
    from .runtime import _cache_kinds, _new_cache

    if runtime.model_identity is None or type(runtime.backend) is not MLXBackend:
        raise ModelIdentityError("online control requires an identified MLX backend")
    fingerprint = runtime.fingerprint(StrictOneShotPlan())
    required = ("hardware_fingerprint", "chip", "memory_bytes", "os", "mlx", "mlx_lm",
                "model_identity_sha256", "model_revision", "model_manifest_sha256",
                "tokenizer_sha256")
    if any(not fingerprint.get(name) for name in required):
        raise ModelIdentityError("online control requires complete runtime identity")
    knobs = runtime.engine.knobs.as_dict()
    kinds = tuple(_cache_kinds(_new_cache(runtime.engine.model)))
    if not kinds or any(kind not in ("kv", "arrays") for kind in kinds):
        raise ModelIdentityError("online control requires known cache layer kinds")
    root = Path(__file__).parent
    names = ("service.py", "executor.py", "plans.py", "runtime.py", "fast.py", "hw.py",
             "telemetry.py", "fingerprint.py", "model_identity.py", "online_controller.py")
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in names}
    sources["ironmule_controller.py"] = hashlib.sha256(
        (root.parent / "ironmule_controller.py").read_bytes()).hexdigest()
    identity = {
        "schema": "ironmule.online_runtime_identity.v1",
        "runtime": {key: value for key, value in fingerprint.items()
                    if key not in ("digest", "workload")},
        "backend": "mlx", "backend_device": device_identity(),
        "execution_device": str(mx.default_device()),
        "knobs": knobs, "compute_dtype": getattr(runtime.engine, "compute_dtype", None),
        "cache_kinds": list(kinds), "eos_ids": list(runtime.backend.eos_ids),
        "profiles": {"0": "sequential.v1", "1": "grouped4.v1"},
        "capacity_ceiling": CAPACITY_CEILING, "source_files": sources,
        "grouping_supported": all(kind == "kv" for kind in kinds),
    }
    if action_contract is not None:
        identity["action_contract"] = json.loads(canonical_json(action_contract))
        identity["profiles"] = {str(index): profile_id
                                for index, profile_id in enumerate(action_contract["profiles"])}
        bridge = root.parent / "ironmule_product" / "engine_bridge.py"
        identity["source_files"]["ironmule_product/engine_bridge.py"] = hashlib.sha256(
            bridge.read_bytes()).hexdigest()
    if hardware_binding is not None:
        # Probe timing samples and TTL timestamps are diagnostic data. They do
        # not change identity or discard learned state when a cache refreshes.
        identity["hardware_probe_binding"] = json.loads(canonical_json(hardware_binding))
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


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
    #: Sequential: one request's state never meets another's. A mode that does not say
    #: `groups = False` is treated as grouping by the hybrid-cache refusal below.
    groups = False

    def executor(self, backend, telemetry):
        return SequentialExecutor(backend, telemetry)


def _refuse_grouping_on_a_hybrid_cache(engine: Any, mode: Any) -> None:
    """Grouped execution is unqualified on a model with recurrent cache layers.

    Measured on a Kaggle T4 (PORT2 run 4, Qwen 3.5 9B, whose layers alternate an
    `ArraysCache` gated-delta state with a `KVCache`): every throughput arm disagreed with
    the sequential reference on 2-3 of 6 requests, and an arm with *no knobs at all* was
    already non-deterministic across processes — two runs of the same arm produced two
    different output digests. The bisection therefore names the grouped path itself, not
    any knob: a per-layer recurrent state is not separable per sequence the way keys and
    values are, so a group shares what it must not share.

    Refusing costs a hybrid model its throughput mode and nothing else. Letting it through
    costs correctness, silently and irreproducibly, which the knob contract forbids.
    """
    # NEXT1-D: decided by what the mode runs, not by its name. `AutomaticMode` always hands
    # a group to the throughput or the paired executor, and the paired mode groups too, so
    # a check of `name == "throughput"` let both through. `Runtime.serve` repeats this before
    # every prefill, because a router or a caller can swap `Runtime.mode` after construction.
    if getattr(mode, "groups", True) is False:
        return
    from .runtime import _cache_kinds, _new_cache

    model = getattr(engine, "model", None)
    if model is None:
        return
    try:
        cache = _new_cache(model)
    except Exception:  # noqa: BLE001 - no cache can be built, so nothing can be grouped either
        return
    # An unknown cache type raises here (fail closed): grouping it is not qualified either.
    kinds = _cache_kinds(cache)
    if "arrays" in kinds:
        raise ValueError(
            f"{getattr(mode, 'name', 'this')} mode groups requests and is unsupported on a "
            "model with recurrent cache layers "
            f"({kinds.count('arrays')} of {len(kinds)} layers): grouped execution has been "
            "measured to change tokens and to do so non-deterministically. Use interactive "
            "mode, or see PORT2 in the backlog."
        )


class ThroughputMode:
    name = "throughput"
    groups = True

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
    never waits for a partner. See `research/LEDGER.md` entries `B45` and `B46`.
    """

    name = "paired_throughput"
    groups = True

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
    #: Both of its delegates, the throughput and the paired executor, group requests.
    groups = True

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

    @property
    def readback(self) -> int:
        """Steps the sequential path may chain before one host read (`readback_every`)."""
        return max(1, int(getattr(getattr(self.engine, "knobs", None), "readback_every", 1)))

    def steps(self, state, token: int, capacity: int, count: int):
        """Chain `count` decode steps without reading on the host; None for hybrid state.

        Each step's input is the previous step's lazy pick, as `Engine._decode` chains them.
        Recurrent (hybrid) state cannot be rewound to an earlier position, so it stays
        step-wise.
        """
        import mlx.core as mx
        from .runtime import _state_is_hybrid
        if _state_is_hybrid(state):
            return None
        body = self.engine._body(capacity, 1)
        fused = getattr(getattr(self.engine, "knobs", None), "fused_argmax", False)
        handles, current = [], mx.array([[token]])
        for _ in range(count):
            out = body(current, state)
            pick = out[0] if fused else mx.argmax(out[0][:, -1, :].astype(mx.float32), axis=-1)
            handles.append((out, pick))
            state, current = out[1], pick.reshape((1, 1))
        return handles

    def speculate(self, prompt_ids, first: int, state, max_tokens: int, capacity: int):
        """Draft-gated speculative decode of one fresh session (SPEC1); None when off.

        Returns the physical tokens after the first and the final state. Hybrid state
        cannot be rolled back over rejected drafts, so it is never speculated.
        """
        import mlx.core as mx
        from .runtime import _state_is_hybrid
        if getattr(getattr(self.engine, "knobs", None), "speculate_k", 0) <= 0 or _state_is_hybrid(state):
            return None
        physical, _, final = self.engine._decode_speculative(
            state, mx.array([[first]]), list(prompt_ids), max_tokens, self.eos_ids, capacity)
        return physical[1:], final

    def complete_chain(self, handles) -> None:
        """Evaluate every pick of a chain and only its final state."""
        self.complete_chains([handles])

    def complete_chains(self, chains) -> None:
        """One synchronisation for several sessions' chains (grouped execution)."""
        import mlx.core as mx
        from .runtime import _leaves
        flat = [pick for chain in chains for _, pick in chain]
        flat += [leaf for chain in chains for leaf in _leaves(chain[-1][0][1])]
        mx.async_eval(*flat)
        mx.eval(*flat)
        mx.synchronize()

    @staticmethod
    def rewind(state, offset: int):
        """The chain's final cache with its position back at the last accepted token.

        Later steps wrote only slots at or beyond that position, so the cache below it,
        the region every reader and `kv_hash` sees, equals the step-wise state.
        """
        import mlx.core as mx
        return {"position": {"offset": mx.array(offset, dtype=mx.int32)}, "layers": state["layers"]}

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
                 quantisation: Any = None, model_identity: ModelIdentity | None = None,
                 online_controller: Any = None):
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
            _refuse_grouping_on_a_hybrid_cache(engine, self.mode)
            self.model_identity = identity
            self.model_id = identity.model_id if identity is not None else model_id
            self.quantisation = identity.quantisation if identity is not None else quantisation
            self.backend = MLXBackend(engine, _eos_ids(tokenizer))
            self.telemetry = Telemetry(mode=self.mode.name)
            self.online_controller = None
            if online_controller is not None:
                self.attach_online_controller(online_controller)
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
        controller = getattr(self, "online_controller", None)
        lock = getattr(self, "_online_lock", None)
        if controller is None or lock is None:
            self.engine.close()
            return
        with lock:
            try:
                controller.close()
            finally:
                getattr(self, "_online_compiled_cache", {}).clear()
                self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
        return False

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, model_id: str | None = None, mode=None, use_tuned_profile: bool = True,
             revision: str | None = None, automatic_service_mode: bool = False,
             compute_dtype: str | None = None, *,
             learning_checkpoint: str | Path | None = None,
             controller_config: Any = None,
             controller_library_path: str | Path | None = None,
             probe_cache_dir: str | Path | None = None):
        """Load a model, and only on request let the profile choose the service mode.

        `automatic_service_mode=True` is the opt-in. Without it nothing about the stored
        profile can change which mode serves a request, which is why an older profile and
        a profile carrying a new record behave identically until a caller asks.
        """

        from .runtime import BASELINE, Knobs
        from .tune import DEFAULT_MODEL, load_engine, load_profile, resolve_local_model
        model_id = model_id or DEFAULT_MODEL
        if learning_checkpoint is not None:
            if automatic_service_mode or (mode is not None and type(mode) is not InteractiveMode):
                raise ValueError("hardware learning requires the interactive reference mode")
            from .hw import probe
            # Fresh characterization can allocate substantial scratch buffers.
            # Run the existing probe before loading this model, never beside it.
            probe(cache_dir=Path(probe_cache_dir) if probe_cache_dir is not None else None)
        if automatic_service_mode and mode is not None:
            raise ValueError("automatic_service_mode replaces an explicit mode; pass one")
        resolved = resolve_local_model(model_id, revision)
        knobs = BASELINE
        profile = None
        if use_tuned_profile or automatic_service_mode:
            profile = load_profile(
                model_id, revision=revision, model_identity=resolved.identity,
                compute_dtype=compute_dtype,
            )
            if profile and use_tuned_profile:
                knobs = Knobs(**profile["knobs"])
        if automatic_service_mode:
            mode = cls._automatic_mode(profile, resolved.identity)
        engine, tokenizer = load_engine(
            model_id, knobs, revision=revision, resolved_source=resolved,
            compute_dtype=compute_dtype,
        )
        # `generate` renders the chat template, so a chat turn's end marker is an end of
        # sequence here exactly as in the product worker (Gemma 3 1B names only `<eos>`).
        from ironmule_product.worker import stop_at_end_of_turn
        stop_at_end_of_turn(tokenizer)
        runtime = cls(
            engine, tokenizer, mode=mode, model_id=resolved.identity.model_id,
            model_identity=resolved.identity,
        )
        if learning_checkpoint is not None:
            try:
                runtime.enable_hardware_learning(
                    learning_checkpoint, library_path=controller_library_path,
                    config=controller_config, probe_cache_dir=probe_cache_dir)
            except BaseException as exc:
                try:
                    runtime.close()
                except BaseException as cleanup_error:
                    exc.add_note(f"Runtime cleanup failed: {type(cleanup_error).__name__}")
                raise
        return runtime

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
    def encode(self, text: str, **template_options: Any) -> list[int]:
        """The prompt as the model's chat template renders it.

        `template_options` go to the template unchanged, e.g. Qwen 3's
        `enable_thinking=False` for its direct mode (NEXT1-Q). None are passed by default,
        so the prompt contract stays the template's own; choosing a mode is the caller's.
        """
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
            **template_options)
        return list(self.tokenizer.encode(rendered, add_special_tokens=False))

    def session_plan(self, shared_prefix: str, name: str = "session") -> ReusableSessionPlan:
        """Build a reusable-session plan from the shared part of a prompt."""
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": shared_prefix + "\n\n@@CUT@@"}],
            tokenize=False, add_generation_prompt=True).split("@@CUT@@")[0]
        return ReusableSessionPlan(self.tokenizer.encode(rendered, add_special_tokens=False),
                                   name=name)

    # -- serving --------------------------------------------------------------
    def attach_online_controller(self, controller: Any) -> None:
        """Opt in to strict-group control after binding exact startup identity.

        Attach before serving. The controller must be configured for the mapping
        returned by ``runtime_identity(self)``. Existing explicit modes remain
        caller-owned, and later mode changes bypass online control. This runtime
        owns controller shutdown; do not share it with another runtime.
        """
        if type(self.mode) is not InteractiveMode:
            raise ValueError("online control requires the interactive reference mode")
        supplied_contract = getattr(controller, "action_contract", None)
        schema = supplied_contract.get("schema") if isinstance(supplied_contract, Mapping) else None
        broad = schema in ("ironmule.execution_profiles.v2", "ironmule.execution_profiles.v3")
        identity = runtime_identity(self)
        if broad:
            from ironmule_product.engine_bridge import serving_profile_contract
            expected_contract = serving_profile_contract(
                self.engine.knobs.as_dict(), grouping_supported=identity["grouping_supported"],
                speculative=schema == "ironmule.execution_profiles.v3")
            if canonical_json(supplied_contract) != canonical_json(expected_contract):
                raise ModelIdentityError("online serving profiles differ from this runtime")
            binding = getattr(controller, "identity", {}).get("hardware_probe_binding")
            identity = runtime_identity(self, action_contract=expected_contract,
                                        hardware_binding=binding)
        if canonical_json(getattr(controller, "identity", {})) != canonical_json(identity):
            raise ModelIdentityError("online controller identity differs from this runtime")
        self._online_runtime_identity = controller.identity
        self._online_reference_mode = self.mode
        self._online_backend = self.backend
        self._online_engine = self.engine
        self._online_model = getattr(self.engine, "model", None)
        self._online_knobs = self.engine.knobs.as_dict()
        self._online_profile_contract = supplied_contract if broad else None
        self._online_compiled_cache: dict[str, tuple[Any, Any]] = {}
        self._online_lock = threading.Lock()
        self.online_controller = controller

    def enable_hardware_learning(self, checkpoint_path: str | Path, *,
                                 library_path: str | Path | None = None,
                                 config: Any = None,
                                 probe_cache_dir: str | Path | None = None):
        """Opt in to measured hardware knowledge and the existing local learner.

        Setup reuses the hardware probe and profile catalog. Model weights,
        precision and caller plans remain fixed. The returned controller owns
        automatic qualification; unknown or incompatible checkpoints fail closed.
        """
        if getattr(self, "online_controller", None) is not None:
            raise ValueError("this runtime already owns an online controller")
        if type(self.mode) is not InteractiveMode:
            raise ValueError("hardware learning requires the interactive reference mode")
        from .hw import probe, validate_probe
        from .online_controller import ControllerConfig, OnlineController
        from ironmule_product.engine_bridge import serving_profile_contract

        base_identity = runtime_identity(self)
        record = probe(cache_dir=Path(probe_cache_dir) if probe_cache_dir is not None else None,
                       allow_measure=False)
        valid, reasons = validate_probe(record, facts=record.get("static"))
        if not valid or not isinstance(record.get("binding"), Mapping):
            raise ModelIdentityError("hardware probe is invalid: " + "; ".join(reasons))
        if record["fingerprint"] != base_identity["runtime"]["hardware_fingerprint"]:
            raise ModelIdentityError("hardware probe fingerprint differs from this runtime")
        contract = serving_profile_contract(
            self.engine.knobs.as_dict(), grouping_supported=base_identity["grouping_supported"],
            speculative=bool(getattr(config, "speculative_profiles", False)))
        identity = runtime_identity(self, action_contract=contract, hardware_binding=record["binding"])
        effective_config = replace(config or ControllerConfig(), verify_exploration=True)
        controller = OnlineController(
            identity, checkpoint_path=Path(checkpoint_path),
            library_path=Path(library_path) if library_path is not None else None,
            config=effective_config, action_contract=contract, hardware_record=record)
        try:
            self.attach_online_controller(controller)
        except BaseException:
            controller.close()
            raise
        self._online_hardware_record = json.loads(canonical_json(record))
        return controller

    def serve(self, requests: Sequence[Request],
              dispatch_ns: int | None = None) -> list[Result]:
        """Serve one group of requests. `dispatch_ns` says when the caller handed them
        over, for a caller that splits one dispatch across several calls; it changes
        recorded arrival only, never scheduling or output."""
        controller = getattr(self, "online_controller", None)
        lock = getattr(self, "_online_lock", None)
        if controller is not None and lock is not None:
            # Comparisons and ordinary/bypassed groups share one engine owner.
            started = time.perf_counter_ns()
            with lock:
                return self._serve_dispatch(requests, dispatch_ns, controller, started)
        return self._serve_dispatch(requests, dispatch_ns, None)

    def _serve_dispatch(self, requests, dispatch_ns, controller, online_start_ns=None):
        if not requests:
            return []
        for request in requests:
            if plan_kind(request.plan) not in ("strict_one_shot", "reusable_session"):
                raise ValueError(f"unknown execution plan: {request.plan!r}")

        bound = getattr(self, "_online_runtime_identity", None)
        supported = (
            controller is not None and bound is not None and dispatch_ns is None
            and type(self.mode) is InteractiveMode
            and self.mode is getattr(self, "_online_reference_mode", None)
            and self.backend is getattr(self, "_online_backend", None)
            and self.engine is getattr(self, "_online_engine", None)
            and getattr(self.engine, "model", None) is getattr(self, "_online_model", None)
            and all(type(request.plan) is StrictOneShotPlan and request.arrival_ms == 0
                    and request.objective != "latency" for request in requests)
            and len({request.rid for request in requests}) == len(requests)
            and self.engine.knobs.as_dict() == self._online_knobs
            and self.model_identity is not None
            and self.model_identity.identity_sha256 == bound["runtime"]["model_identity_sha256"]
            and list(self.backend.eos_ids) == list(bound["eos_ids"])
            and getattr(self.engine, "compute_dtype", None) == bound["compute_dtype"]
        )
        if supported:
            return self._serve_online(requests, dispatch_ns, controller, bound, online_start_ns)
        return self._serve_group(requests, dispatch_ns, self.mode)

    def _serve_online(self, requests, dispatch_ns, controller, identity, started):
        from friday_evidence.canonical import canonical_sha256
        from .online_controller import ControllerContext, ExecutionOutcome

        requests = tuple(requests)
        capacity = self.backend.capacity_for([len(r.prompt_ids) for r in requests],
                                             max(r.max_tokens for r in requests))
        # Only a numeric digest enters controller storage, never prompts or tokens.
        workload = {
            "schema": "ironmule.strict_group_workload.v1", "plan": "strict_one_shot",
            "prompt_tokens": [len(r.prompt_ids) for r in requests],
            "token_caps": [r.max_tokens for r in requests], "capacity": capacity,
            "eos_ids": list(self.backend.eos_ids),
            "request_sha256": canonical_sha256([
                {"rid": r.rid, "tokens": list(r.prompt_ids), "max_tokens": r.max_tokens,
                 "objective": r.objective} for r in requests]),
        }
        grouped = len(requests) > 1 and bool(identity["grouping_supported"])
        contract = getattr(self, "_online_profile_contract", None)
        definitions = contract["definitions"] if contract is not None else None
        eligible_mask = (sum(1 << index for index, profile in enumerate(definitions)
                             if profile["mode"] == "interactive" or grouped)
                         if definitions is not None else 3 if grouped else 1)
        context = ControllerContext(identity=identity, request_count=len(requests),
                                    max_tokens=max(r.max_tokens for r in requests),
                                    workload=workload, eligible_mask=eligible_mask)

        def execute(action):
            if (type(action) is not int or action < 0
                    or action >= (len(definitions) if definitions is not None else 2)
                    or eligible_mask & (1 << action) == 0):
                raise ValueError("controller selected an inadmissible runtime profile")
            arm_started = time.perf_counter_ns()
            if definitions is None:
                mode = InteractiveMode() if action == 0 else ThroughputMode(max_width=4)
                profile_id = "sequential.v1" if action == 0 else "grouped4.v1"
            else:
                profile = definitions[action]
                mode = (InteractiveMode() if profile["mode"] == "interactive"
                        else ThroughputMode(max_width=profile["max_width"]))
                profile_id = profile["profile_id"]
            comparing = getattr(controller, "last_decision", None) or {}
            capture = ([] if definitions is not None
                       and (comparing.get("comparison") or comparing.get("verification")) else None)
            # Every arm builds new sessions from immutable caller-owned strict plans.
            value = (self._serve_group(requests, arm_started, mode) if definitions is None
                     else self._serve_profile_group(requests, arm_started, mode, profile, capture))
            telemetry = self.telemetry
            telemetry.routing["execution_profile"] = {"profile_id": profile_id}
            signature = tuple((r.rid, tuple(r.tokens), r.stop_reason,
                               r.metrics["physical_generated_tokens"],
                               r.metrics["visible_generated_tokens"], r.text) for r in value)
            if capture is not None:
                # Online control serves strict plans only; their terminal cache is discarded.
                offset_only = getattr(getattr(controller, "config", None), "state_signature", "hash") == "offset"
                signature = (signature, tuple((rid, offset) for rid, offset, _ in capture)
                             if offset_only else tuple(capture))
            return ExecutionOutcome(
                value=(value, telemetry), signature=signature,
                elapsed_s=(time.perf_counter_ns() - arm_started) / 1e9,
                ttft_s=tuple(r.metrics["service_ttft_ms"] / 1000 for r in value
                             if r.metrics["service_ttft_ms"] is not None),
                latencies_s=tuple(r.metrics["latency_ms"] / 1000 for r in value
                                  if r.metrics["latency_ms"] is not None),
                generated_tokens=sum(len(r.tokens) for r in value),
                fallback_count=telemetry.fallbacks,
                memory_bytes=telemetry.peak_memory_bytes or None,
            )

        results, telemetry = controller.run(context, execute)
        self.telemetry = telemetry
        decision = dict(getattr(controller, "last_decision", None) or {})
        if decision.get("correctness") == "matched":
            telemetry.correctness_check_performed = True
            telemetry.correctness_checked_requests = len(results)
        outer_wall_ns = time.perf_counter_ns() - started
        for result in results:
            # Buffered library results are delivered after the entire group and
            # its optional comparisons. Existing metrics describe the chosen arm.
            result.metrics["caller_return_latency_ms"] = outer_wall_ns / 1e6
        self.telemetry.routing["online_controller"] = {
            "profile": telemetry.routing["execution_profile"]["profile_id"],
            "profile_id": telemetry.routing["execution_profile"]["profile_id"],
            "outer_wall_ns": outer_wall_ns,
            "metric_scope": "completed_execution_arm",
            "caller_return_latency_ms": outer_wall_ns / 1e6,
            "eligible_mask": context.eligible_mask,
            "decision": decision,
            "fallback_count": telemetry.fallbacks,
        }
        return results

    def _serve_profile_group(self, requests, dispatch_ns, mode, profile, capture=None):
        """Borrow one effective configuration without mutating the loaded model."""
        from .runtime import Knobs

        engine = self.engine
        original_knobs = engine.knobs
        had_compiled = hasattr(engine, "_compiled")
        had_capacity = hasattr(engine, "_compiled_capacity")
        original_compiled = getattr(engine, "_compiled", None)
        original_capacity = getattr(engine, "_compiled_capacity", None)
        selected = (original_knobs if original_knobs.as_dict() == profile["knobs"]
                    else Knobs(**profile["knobs"]))
        key = selected.key()
        compiled, capacity = self._online_compiled_cache.get(
            key, (original_compiled, original_capacity) if selected is original_knobs else (None, None))
        succeeded = False
        try:
            engine.knobs = selected
            engine._compiled, engine._compiled_capacity = compiled, capacity
            result = (self._serve_group(requests, dispatch_ns, mode) if capture is None
                      else self._serve_group(requests, dispatch_ns, mode, capture=capture))
            succeeded = True
            return result
        finally:
            if succeeded:
                self._online_compiled_cache[key] = (engine._compiled, engine._compiled_capacity)
            else:
                self._online_compiled_cache.pop(key, None)
            engine.knobs = original_knobs
            if had_compiled:
                engine._compiled = original_compiled
            else:
                del engine._compiled
            if had_capacity:
                engine._compiled_capacity = original_capacity
            else:
                del engine._compiled_capacity

    def _serve_group(self, requests, dispatch_ns, mode, *, capture=None):
        import mlx.core as mx

        # NEXT1-D: before any prefill, against the mode that will actually run. A Runtime
        # built without an engine (a test double) has no model to group.
        _refuse_grouping_on_a_hybrid_cache(getattr(self, "engine", None), mode)
        self.telemetry = Telemetry(mode=mode.name)
        capacity = self.backend.capacity_for([len(r.prompt_ids) for r in requests],
                                             max(r.max_tokens for r in requests))
        sessions = build_sessions(requests, self.backend, self.telemetry, capacity,
                                  dispatch_ns=dispatch_ns)

        executor = mode.executor(self.backend, self.telemetry)
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
        if capture is not None:
            hasher = getattr(self.backend, "kv_hash", None)
            if not callable(hasher):
                raise RuntimeError("terminal state hashing is unsupported for this backend")
            for session in sessions:
                offset = len(session.prompt_ids) + len(session.tokens) - 1
                if type(self.backend) is MLXBackend:
                    actual_offset = int(session.state["position"]["offset"].item())
                    if actual_offset != offset:
                        raise RuntimeError("terminal state position violates the execution contract")
                digest = hasher(session.state, offset)
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    raise RuntimeError("terminal state hashing returned an invalid digest")
                capture.append((session.rid, offset, digest))
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
        kind = plan_kind(plan or StrictOneShotPlan())
        compute_dtype = getattr(self.engine, "compute_dtype", None)
        if compute_dtype:
            # A numeric plan changes output: its evidence must never match a native record.
            kind = f"{kind}@{compute_dtype}"
        return build_fingerprint(self.model_id, self.quantisation, kind,
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
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_text(json.dumps(current, indent=1, sort_keys=True, default=str))
            return {"verdict": "recorded_first_fingerprint", "current": current}
        ok, why = usable(stored, current)
        verdict = ("valid" if ok and not why["drifted"]
                   else "valid_with_workload_drift" if ok else "revalidation_required")
        if not ok:
            path.write_text(json.dumps(current, indent=1, sort_keys=True, default=str))
        return {"verdict": verdict, "current": current, "stored_digest": stored.get("digest"),
                **why}
