"""An opt-in bridge from an already loaded product model to IronMule.

Importing this module is intentionally dependency-free.  In particular, it must
not import MLX, load another model, or change Hugging Face process settings.
The heavy imports happen only after the product loader has policy-validated and
loaded the model passed to :meth:`CurrentEngineBridge.from_loaded`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
_CONFIGURATIONS = frozenset(("current_profile", *_HISTORICAL_CONFIGURATIONS))
MAX_GROUP_REQUESTS = 32  # Admitted requests; actual grouped execution width remains <=4.


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
    if (not isinstance(tokens, list) or not tokens
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
                 historical_configuration_candidate: dict[str, Any] | None) -> None:
        self._runtime = runtime
        self._identity = identity
        self._knobs = knobs
        self._profile_available = profile_available
        self._profile_source = profile_source
        self._configuration = configuration
        self._historical_configuration_candidate = historical_configuration_candidate
        self._closed = False

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, spec: Any, *,
                    configuration: BridgeConfiguration = "current_profile") -> "CurrentEngineBridge":
        """Construct from exactly the already loaded model and registered snapshot.

        Identity construction reads the supplied snapshot and fails closed; it does
        not use cache discovery, model loading, or inferred revisions.
        """
        model_id, revision, snapshot_path = _validate_spec(spec)
        configuration = _validate_configuration(configuration)
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
        if historical_candidate is not None:
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
            if profile is None:
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
                   historical_configuration_candidate=historical_candidate)

    def generate(self, prompt_ids: Sequence[int], max_tokens: int) -> EngineBridgeResult:
        """Serve exact supplied token IDs under ``StrictOneShotPlan`` only."""
        if self._closed:
            raise RuntimeError("CurrentEngineBridge is closed")
        ids = _validate_request(prompt_ids, max_tokens)
        return self.generate_many([ids], max_tokens)[0]

    def generate_many(self, prompt_ids: Sequence[Sequence[int]],
                      max_tokens: int | Sequence[int]) -> list[EngineBridgeResult]:
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
        from ironmule.plans import StrictOneShotPlan
        from ironmule.service import Request

        raw_results = self._runtime.serve([
            Request(prompt_ids=ids, max_tokens=limit, plan=StrictOneShotPlan())
            for ids, limit in zip(validated, limits)
        ])
        if not isinstance(raw_results, list) or len(raw_results) != len(validated):
            raise RuntimeError("IronMule returned an invalid result count")
        telemetry = self._runtime.telemetry
        fallback_count = getattr(telemetry, "fallbacks", None)
        if type(fallback_count) is not int or fallback_count < 0:
            raise RuntimeError("IronMule returned invalid fallback telemetry")
        widths = list(telemetry.realised_widths)
        if any(type(width) is not int or not 1 <= width <= 4 for width in widths):
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
            })
            results.append(_result_from_raw(raw, limit, metadata))
        return results

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
        }

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
