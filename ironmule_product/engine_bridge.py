"""An opt-in bridge from an already loaded product model to IronMule.

Importing this module is intentionally dependency-free.  In particular, it must
not import MLX, load another model, or change Hugging Face process settings.
The heavy imports happen only after the product loader has policy-validated and
loaded the model passed to :meth:`CurrentEngineBridge.from_loaded`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any, Literal, Sequence


class CurrentEngineBridgeBlocked(RuntimeError):
    """The registered snapshot cannot be bound to an exact IronMule identity."""


@dataclass(frozen=True)
class EngineBridgeResult:
    """Completed, non-streaming output from one strict IronMule request."""

    tokens: list[int] = field(repr=False)
    text: str = field(repr=False)
    finish_reason: str
    token_count: int
    metrics: dict[str, Any] = field(repr=False)
    metadata: dict[str, Any]


BridgeConfiguration = Literal[
    "current_profile",
    "dense_stacked",
    "baseline_interactive",
    "core_interactive",
    "baseline_throughput",
    "core_throughput",
]

_B39D_SOURCE = "B39d_gemma12b_combined_20260828"
_B39D_KNOBS = {
    "capacity_slack": 0,
    "fuse_projections": False,
    "fused_argmax": False,
    "prefill_into_fixed": False,
    "readback_every": 1,
    "speculate_k": 0,
    "speculate_ngram": 3,
    "wired_fraction": 0.0,
}
_HISTORICAL_CONFIGURATIONS = {
    "baseline_interactive": {
        "knobs": {**_B39D_KNOBS, "compiled_fixed_cache": False, "head_skip_prefill": False},
        "max_width": None,
    },
    "core_interactive": {
        "knobs": {**_B39D_KNOBS, "compiled_fixed_cache": True, "head_skip_prefill": True},
        "max_width": None,
    },
    "baseline_throughput": {
        "knobs": {**_B39D_KNOBS, "compiled_fixed_cache": False, "head_skip_prefill": False},
        "max_width": 4,
    },
    "core_throughput": {
        "knobs": {**_B39D_KNOBS, "compiled_fixed_cache": True, "head_skip_prefill": True},
        "max_width": 4,
    },
}
_CONFIGURATIONS = frozenset(("current_profile", "dense_stacked", *_HISTORICAL_CONFIGURATIONS))
DENSE_WIDTH = 8  # KB1: the dense plan stacks up to eight requests into one decode step
MAX_GROUP_REQUESTS = 32  # Admitted requests; actual grouped execution width remains <=4.


def serving_profile_contract(reference_knobs: Mapping[str, Any], *,
                             grouping_supported: bool = True,
                             speculative: bool = False) -> dict[str, Any]:
    """Project existing candidates onto the caller's effective serving knobs.

    This pure catalog grants no qualification. Installation, numeric precision,
    plan and all other knobs remain caller-owned. Historical templates are the
    single source for the two serving toggles; no template is copied elsewhere.
    """
    toggles = ("compiled_fixed_cache", "head_skip_prefill")
    if (not isinstance(reference_knobs, Mapping)
            or any(type(reference_knobs.get(key)) is not bool for key in toggles)
            or type(grouping_supported) is not bool):
        raise ValueError("serving profiles require canonical knobs and grouping support")
    definitions: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def append(name: str, knobs: dict[str, Any], grouped: bool) -> None:
        key = (tuple(sorted(knobs.items())), grouped)
        if key in seen or len(definitions) == 4:
            return
        seen.add(key)
        definitions.append({
            "profile_id": f"{name}.{'grouped4' if grouped else 'sequential'}.v1",
            "mode": "throughput" if grouped else "interactive",
            "max_width": 4 if grouped else 1, "knobs": dict(knobs),
        })

    append("current", dict(reference_knobs), False)
    if grouping_supported:
        append("current", dict(reference_knobs), True)
    if speculative and reference_knobs.get("speculate_k", 0) == 0:
        # CTRL23: the caller's knobs plus draft-gated speculation, sequential only.
        append("speculative", dict(reference_knobs, speculate_k=4), False)
    order = (("baseline_interactive", "core_interactive")
             if all(reference_knobs[key] for key in toggles)
             else ("core_interactive", "baseline_interactive"))
    for name in order:
        candidate = _HISTORICAL_CONFIGURATIONS[name]
        projected = dict(reference_knobs)
        projected.update({key: candidate["knobs"][key] for key in toggles})
        stem = name.removesuffix("_interactive")
        append(stem, projected, False)
        if grouping_supported:
            append(stem, projected, True)
    return {
        "schema": "ironmule.execution_profiles.v3" if speculative else "ironmule.execution_profiles.v2",
        "profiles": [item["profile_id"] for item in definitions],
        "definitions": definitions, "plan": "strict_one_shot",
        "transition": "idle_complete_group", "max_width": 4,
        "precision_change": False,
    }


def _validate_spec(spec: Any) -> tuple[str, str, Path]:
    """Reject a non-canonical registry entry before importing runtime code."""
    model_id = getattr(spec, "model_id", None)
    revision = getattr(spec, "revision", None)
    snapshot_path = getattr(spec, "snapshot_path", None)
    weight_bytes = getattr(spec, "weight_bytes", None)
    if (not isinstance(model_id, str) or not model_id or model_id != model_id.strip()
            or not isinstance(revision, str) or not revision or revision != revision.strip()
            or not isinstance(snapshot_path, str) or not snapshot_path
            or type(weight_bytes) is not int or weight_bytes <= 0):
        raise CurrentEngineBridgeBlocked("registered model specification is not canonical")
    path = Path(snapshot_path)
    if not path.is_absolute():
        raise CurrentEngineBridgeBlocked("registered model snapshot path is not absolute")
    return model_id, revision, path


def _validate_request(prompt_ids: Sequence[int], max_tokens: int) -> list[int]:
    """Validate canonical token IDs without importing the runtime."""
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    ids = list(prompt_ids)
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("prompt_ids must contain non-negative integer token IDs")
    return ids


def _validate_configuration(configuration: str) -> BridgeConfiguration:
    if not isinstance(configuration, str) or configuration not in _CONFIGURATIONS:
        choices = ", ".join(sorted(_CONFIGURATIONS))
        raise ValueError(f"unknown bridge configuration {configuration!r}; expected one of {choices}")
    return configuration  # type: ignore[return-value]


def _historical_configuration_metadata(configuration: BridgeConfiguration) -> dict[str, Any] | None:
    """Return a descriptive B39d record, never a current qualification claim."""
    candidate = _HISTORICAL_CONFIGURATIONS.get(configuration)
    if candidate is None:
        return None
    return {
        "name": configuration,
        "source": _B39D_SOURCE,
        "activation_allowed": False,
        "requires_fresh_native_qualification": True,
        "mode": "throughput" if candidate["max_width"] else "interactive",
        "max_width": candidate["max_width"],
        "knobs": dict(candidate["knobs"]),
    }


def _result_from_raw(raw: Any, max_tokens: int, metadata: dict[str, Any]) -> EngineBridgeResult:
    """Validate the narrow Runtime result protocol and preserve its exact payload."""
    tokens = getattr(raw, "tokens", None)
    text = getattr(raw, "text", None)
    reason = getattr(raw, "stop_reason", None)
    metrics = getattr(raw, "metrics", None)
    if (not isinstance(tokens, list) or (not tokens and reason != "cancelled")
            or any(type(token) is not int or token < 0 for token in tokens)
            or len(tokens) > max_tokens):
        raise RuntimeError("IronMule returned invalid output token IDs")
    if not isinstance(text, str):
        raise RuntimeError("IronMule returned non-string output text")
    if not isinstance(metrics, dict):
        raise RuntimeError("IronMule returned non-mapping metrics")
    if reason == "eos":
        finish_reason = "stop"
    elif reason == "length":
        finish_reason = "length"
    elif reason == "cancelled":
        finish_reason = "cancelled"  # RSI3-LIVE-P2: stopped early because its caller cancelled
    else:
        raise RuntimeError("IronMule returned an unknown stop reason")
    return EngineBridgeResult(
        tokens=list(tokens), text=text, finish_reason=finish_reason,
        token_count=len(tokens), metrics=dict(metrics), metadata=dict(metadata),
    )


class CurrentEngineBridge:
    """Run one strict one-shot request against a product-owned loaded model.

    This is deliberately not a streaming adapter: callers buffer until
    ``Runtime.serve`` has completed and must not infer TTFT or log probabilities.
    The bridge owns the Engine/Runtime it constructs, but never the model itself.
    """

    def __init__(self, runtime: Any, *, identity: Any, knobs: Any,
                 profile_available: bool, profile_source: str,
                 configuration: BridgeConfiguration,
                 historical_configuration_candidate: dict[str, Any] | None,
                 dense_plan: dict[str, Any] | None = None) -> None:
        self._runtime = runtime
        self._dense_plan = dense_plan
        self._identity = identity
        self._knobs = knobs
        self._profile_available = profile_available
        self._profile_source = profile_source
        self._configuration = configuration
        self._historical_configuration_candidate = historical_configuration_candidate
        self._closed = False

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, spec: Any, *,
                    configuration: BridgeConfiguration = "current_profile",
                    class_a_knobs: dict[str, Any] | None = None) -> "CurrentEngineBridge":
        """Construct from exactly the already loaded model and registered snapshot.

        Identity construction reads the supplied snapshot and fails closed; it does
        not use cache discovery, model loading, or inferred revisions.

        ``class_a_knobs`` (KP1), ``{"knobs": ..., "source": label}``, serve the current
        profile when no compatible stored profile exists; they must differ from the
        baseline only in knobs that leave every answer unchanged.
        """
        model_id, revision, snapshot_path = _validate_spec(spec)
        configuration = _validate_configuration(configuration)
        requested = None
        if class_a_knobs is not None:
            from .knob_profile import class_a_knobs as decode_class_a

            mapping = (decode_class_a(class_a_knobs.get("knobs")) if isinstance(class_a_knobs, dict)
                       and configuration == "current_profile" else None)
            if mapping is not None:
                from ironmule.runtime import Knobs as RuntimeKnobs  # `Knobs` is bound further down

                requested = RuntimeKnobs(**mapping)
            source = class_a_knobs.get("source") if isinstance(class_a_knobs, dict) else None
            if requested is None or not isinstance(source, str) or not 1 <= len(source) <= 64:
                raise ValueError("class-A knobs serve the current profile only, and only class-A knobs")
        # Keep all IronMule imports here.  Engine's constructor imports MLX only
        # after this method is deliberately invoked by the product worker.
        from ironmule.model_identity import ModelIdentityError, build_model_identity
        from ironmule.runtime import BASELINE, Engine, Knobs
        from ironmule.service import InteractiveMode, Runtime, ThroughputMode
        from ironmule.tune import load_profile

        try:
            identity = build_model_identity(
                model_id, snapshot_path, revision,
            )
        except (ModelIdentityError, OSError, ValueError) as exc:
            raise CurrentEngineBridgeBlocked(
                "cannot construct exact model identity from the registered snapshot"
            ) from exc

        historical_candidate = _historical_configuration_metadata(configuration)
        dense_record = None
        if configuration == "dense_stacked":
            # KB1: the opt-in dense numeric plan; `install` refuses anything it was not built for and probes the
            # model's own weights (stacked rows bit-identical) before it serves.
            import mlx.core as mx
            from ironmule.dense_plan import install
            from ironmule.service import DenseStackedMode
            plan, dense_record = install(model, mx.device_info() if mx.cuda.is_available() else None)
            knobs, profile_available, profile_source = BASELINE, False, "dense_numeric_plan"
            mode = DenseStackedMode(plan, max_width=DENSE_WIDTH)
        elif historical_candidate is not None:
            # These values are an explicit, archived B39d candidate.  They are
            # neither a compatible tune profile nor an automatic selection.
            knobs = Knobs(**historical_candidate["knobs"])
            profile_available = False
            profile_source = "historical_b39d_candidate_explicitly_requested"
            mode = (ThroughputMode(max_width=historical_candidate["max_width"])
                    if historical_candidate["max_width"] else InteractiveMode())
        else:
            # ``load_profile`` verifies the complete current identity itself.
            # Its absence is an explicit baseline choice, never a claim that no
            # historical configuration exists or that baseline is currently best.
            profile = load_profile(model_id, revision=revision, model_identity=identity)
            if profile is None and requested is not None:
                knobs = requested
                profile_available = False
                profile_source = f"class_a:{source}"
            elif profile is None:
                knobs = BASELINE
                profile_available = False
                profile_source = "baseline_no_compatible_profile"
            else:
                knobs = Knobs(**profile["knobs"])
                profile_available = True
                profile_source = "compatible_tuned_profile"
            mode = InteractiveMode()

        engine = Engine(model, tokenizer, knobs)
        try:
            runtime = Runtime(
                engine, tokenizer, mode=mode, model_id="", model_identity=identity,
            )
        except BaseException:
            # Runtime normally closes on construction failure; retain this guard for
            # compatible Runtime implementations used by isolated workers/tests.
            engine.close()
            raise
        return cls(runtime, identity=identity, knobs=knobs,
                   profile_available=profile_available, profile_source=profile_source,
                   configuration=configuration,
                   historical_configuration_candidate=historical_candidate, dense_plan=dense_record)

    def generate(self, prompt_ids: Sequence[int], max_tokens: int, on_token=None,
                 boundaries: Sequence[int] | None = None) -> EngineBridgeResult:
        """Serve exact supplied token IDs under ``StrictOneShotPlan`` only.

        ``on_token`` receives each physical output token once, in order, as the engine reads
        it back (SERVE6); the returned result is unchanged."""
        if self._closed:
            raise RuntimeError("CurrentEngineBridge is closed")
        ids = _validate_request(prompt_ids, max_tokens)
        return self.generate_many([ids], max_tokens, on_tokens=[on_token],
                                  boundaries=None if boundaries is None else [boundaries])[0]

    def generate_many(self, prompt_ids: Sequence[Sequence[int]],
                      max_tokens: int | Sequence[int], on_tokens=None,
                      boundaries=None, admit=None, cancelled=None) -> list[EngineBridgeResult]:
        """Serve admitted strict requests; mode bounds the realised execution width."""
        if self._closed:
            raise RuntimeError("CurrentEngineBridge is closed")
        requests_ids = list(prompt_ids)
        if not requests_ids:
            return []
        if len(requests_ids) > MAX_GROUP_REQUESTS:
            raise ValueError("generate_many accepts at most 32 requests")
        if type(max_tokens) is int:
            limits = [max_tokens] * len(requests_ids)
        else:
            limits = list(max_tokens)
            if len(limits) != len(requests_ids):
                raise ValueError("max_tokens must be one integer or match prompt_ids length")
        validated = [_validate_request(ids, limit) for ids, limit in zip(requests_ids, limits)]
        from ironmule.executor import TokenStream
        from ironmule.plans import ChatTurnsPlan, StrictOneShotPlan
        from ironmule.service import Request

        emitters = list(on_tokens) if on_tokens is not None else [None] * len(validated)
        if len(emitters) != len(validated):
            raise ValueError("on_tokens must match prompt_ids length")
        turns = list(boundaries) if boundaries is not None else [None] * len(validated)
        if len(turns) != len(validated):
            raise ValueError("boundaries must match prompt_ids length")
        stops = list(cancelled) if cancelled is not None else [None] * len(validated)
        if len(stops) != len(validated) or any(stop is not None and emit is None
                                               for stop, emit in zip(stops, emitters)):
            raise ValueError("cancelled must match prompt_ids and needs a streamed request")
        dense = self._configuration == "dense_stacked"
        if dense and any(cuts is not None for cuts in turns):
            raise ValueError("the dense plan serves one-shot prompts; chat-turn reuse is another plan")
        # SERVE7: a request with chat-turn boundaries runs the opt-in ChatTurnsPlan.
        requests = [Request(prompt_ids=ids, max_tokens=limit,
                            plan=StrictOneShotPlan() if cuts is None else ChatTurnsPlan(cuts),
                            stream=TokenStream(emit, stop) if emit is not None else None)
                    for ids, limit, emit, cuts, stop in zip(validated, limits, emitters, turns, stops)]
        from ironmule.service import DenseStackedMode, ThroughputMode
        mode = self._runtime.mode
        if self._configuration == "current_profile" and (len(requests) > 1 or admit is not None) and self._may_group():
            # SERVE4: simultaneous requests interleave their decode steps (grouped batch-1, exact)
            # instead of each waiting for the whole answer before it.
            self._runtime.mode = ThroughputMode(max_width=4 if admit is not None else min(4, len(requests)))
        joined_limits: list[int] = []

        def take(capacity: int) -> list[Any]:
            """SERVE8: (ids, limit, on_token, chat-turn boundaries or None) rows admitted into this run."""
            fresh = []
            for ids, limit, emit, cuts in admit(capacity):
                fresh.append(Request(prompt_ids=_validate_request(ids, limit), max_tokens=limit,
                                     plan=StrictOneShotPlan() if cuts is None else ChatTurnsPlan(cuts),
                                     stream=TokenStream(emit)))
                joined_limits.append(limit)
            return fresh
        try:
            groups_joins = type(self._runtime.mode) in (ThroughputMode, DenseStackedMode)
            raw_results = (self._runtime.serve(requests) if admit is None or not groups_joins
                           else self._runtime.serve(requests, admit=take))
        finally:
            self._runtime.mode = mode
        if not isinstance(raw_results, list) or len(raw_results) != len(validated) + len(joined_limits):
            raise RuntimeError("IronMule returned an invalid result count")
        limits = limits + joined_limits
        telemetry = self._runtime.telemetry
        fallback_count = getattr(telemetry, "fallbacks", None)
        if type(fallback_count) is not int or fallback_count < 0:
            raise RuntimeError("IronMule returned invalid fallback telemetry")
        widths = list(telemetry.realised_widths)
        widest = DENSE_WIDTH if dense else 4
        if any(type(width) is not int or not 1 <= width <= widest for width in widths):
            raise RuntimeError("IronMule returned invalid realised-width telemetry")
        results = []
        for raw, limit in zip(raw_results, limits):
            metadata = self.metadata()
            # Preserve the Runtime's per-request fallback signal without deriving
            # TTFT, log probabilities, or a per-request result from session data.
            fall_back = bool(getattr(raw, "metrics", {}).get("fell_back", False))
            metadata.update({
                "fall_back": fall_back,
                "runtime_telemetry_fallback_count": fallback_count,
                # Compatibility aliases for the original single-request consumer.
                "fallback_count": fallback_count,
                "fallback_used": fall_back,
                "admitted_requests": len(validated),
                "max_realised_width": max(widths) if widths else 0,
                "grouped_rounds": len(widths),
                **self.counters(),
            })
            results.append(_result_from_raw(raw, limit, metadata))
        return results

    def _may_group(self) -> bool:
        """Grouping is refused on a recurrent (hybrid) cache, as `Runtime` refuses it; such a
        model serves simultaneous requests sequentially, which is still exact."""
        if not hasattr(self, "_groupable"):
            from ironmule.service import ThroughputMode, _refuse_grouping_on_a_hybrid_cache
            try:
                _refuse_grouping_on_a_hybrid_cache(getattr(self._runtime, "engine", None), ThroughputMode())
                self._groupable = True
            except ValueError:
                self._groupable = False
        return self._groupable

    def metadata(self) -> dict[str, Any]:
        """Safe startup metadata; contains no request, prompt, or response data."""
        return {
            "selected_knobs": self._knobs.as_dict(),
            "profile_available": self._profile_available,
            "profile_source": self._profile_source,
            "configuration": self._configuration,
            "historical_configuration_candidate": self._historical_configuration_candidate,
            "current_identity": self._identity.to_dict(),
            "plan": "strict_one_shot",
            **({"dense_plan": dict(self._dense_plan)} if self._dense_plan is not None else {}),
        }

    def counters(self) -> dict[str, Any]:
        """KP1: decode bodies this worker's engine built and evicted so far, and its card (no request data)."""
        return {"decode_bodies": dict(getattr(getattr(self._runtime, "engine", None), "body_counts", None)
                                      or {"built": 0, "evicted": 0}),
                "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def eos_ids(self) -> tuple[int, ...]:
        """The token ids the engine stops on, so a stream ends exactly where the engine does."""
        return tuple(self._runtime.backend.eos_ids)

    def free_slots(self) -> int | None:
        """KB1: free slots in the dense plan's pool, which bound who may join a run; None for other plans."""
        if self._configuration != "dense_stacked":
            return None
        return self._runtime.backend.free_slots()

    def close(self) -> None:
        """Release only Engine/Runtime state created and owned by this bridge."""
        if self._closed:
            return
        self._runtime.close()
        self._closed = True

    def __enter__(self) -> "CurrentEngineBridge":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> bool:
        self.close()
        return False
