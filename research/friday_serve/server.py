"""``generate(prompt, max_tokens)`` — the entry point the project did not have.

Until now the only real generation in this repository happened inside
qualification measurements. Four ~15-line wrappers exposed CLI subcommands;
nothing exposed a request. This module is that request path, and it is
deliberately one function on top of the engine F1 already measured through,
not a new engine and not a framework.

Three properties are carried over from the sealed runtimes unchanged, because
they are what makes serving without a human in the loop defensible:

* scope is derived from the actual tokens and the loaded model (stage B), never
  from a caller's assertion;
* a knob is on only if *this device's* profile verified it as token-identical;
* a failure on an optimised path latches the breaker — and now the latch
  survives the restart, so a knob that failed once stays off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol, Sequence

from friday_calibrate.profile import DeviceProfile
from friday_runtime_core.controller import (
    DispatchController,
    DispatchDecision,
    RuntimeExecutionError,
)

from .dispatch import explain, knobs_for
from .rl_controller import AdaptiveRLController, detect_ngram_overlap
from .scope import RequestScope, in_calibrated_scope, observe

BASELINE_PLAN = "baseline_greedy"
DEVICE_PROFILE_PLAN = "device_profile_dispatch"

BASELINE_DECISION = DispatchDecision("baseline", BASELINE_PLAN, "baseline", None)


class GenerationBackend(Protocol):
    model_id: str
    model_revision: str

    def encode(self, prompt: str) -> Sequence[int]: ...

    def generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def stream_generate(
        self, token_ids: Sequence[int], max_tokens: int, knobs: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]: ...


@dataclass(frozen=True)
class Generation:
    """One answer, plus exactly what produced it."""

    tokens: tuple[int, ...]
    text: str | None
    token_sha256: str
    prefill_ns: int
    decode_ns: int
    plan: str
    reason: str
    knobs: Mapping[str, Any]
    prefix_cache_hits: int = 0

    @property
    def total_ns(self) -> int:
        return self.prefill_ns + self.decode_ns


class Server:
    """A loaded model, a device profile, and one dispatch decision per request."""

    def __init__(
        self,
        backend: GenerationBackend,
        profile: DeviceProfile | None,
        *,
        latch=None,
        rl_controller: AdaptiveRLController | None = None,
        alternate_backends: dict[str, GenerationBackend] | None = None,
    ) -> None:
        self.backend = backend
        self.profile = profile
        self.rl_controller = rl_controller
        self.alternate_backends = alternate_backends or {}
        self.controller = DispatchController(
            evidence=profile,
            decide=self._decide,
            fallback=BASELINE_DECISION,
            latch=latch,
        )

    def get_backend(self, model_hint: str | None = None) -> GenerationBackend:
        if not model_hint or model_hint == "auto":
            return self.backend
        m = model_hint.lower()
        for k, b in self.alternate_backends.items():
            if k.lower() in m or m in k.lower():
                return b
        return self.backend

    # -- decision -------------------------------------------------------------
    def _decide(self, profile: DeviceProfile | None, scope: RequestScope | None):
        if profile is None:
            return DispatchDecision("baseline", BASELINE_PLAN, "no_device_profile", None)
        allowed, reason = in_calibrated_scope(scope, profile)
        if not allowed:
            return DispatchDecision("baseline", BASELINE_PLAN, reason, profile)
        if not knobs_for(profile):
            return DispatchDecision("baseline", BASELINE_PLAN, "no_verified_knob", profile)
        return DispatchDecision("dispatched", DEVICE_PROFILE_PLAN, reason, profile)

    def scope_for(self, prompt: str, max_tokens: int, **kwargs) -> RequestScope | None:
        return observe(
            model_id=getattr(self.backend, "model_id", None),
            model_revision=getattr(self.backend, "model_revision", None),
            token_ids=self.backend.encode(prompt),
            output_tokens=max_tokens,
            **kwargs,
        )

    def plan_request(
        self,
        token_ids: Sequence[int],
        max_tokens: int,
        *,
        backend: GenerationBackend | None = None,
        **kwargs,
    ) -> tuple[DispatchDecision, dict[str, Any]]:
        """The one gate every serving path shares: scope -> decision -> knobs.

        Returns the dispatch decision and the engine knob overrides this request
        is authorised to use (empty when the request is out of the calibrated
        scope, the breaker is latched, or no knob was verified). The batcher and
        the single-flight path both go through here so neither can apply a knob
        the profile did not earn.
        """

        backend = backend or self.backend
        scope = observe(
            model_id=getattr(backend, "model_id", None),
            model_revision=getattr(backend, "model_revision", None),
            token_ids=token_ids,
            output_tokens=max_tokens,
            **kwargs,
        )
        decision = self.controller.decide_scope(scope)
        knobs = {} if self.controller.is_fallback(decision) else knobs_for(self.profile)
        return decision, knobs

    def explain(self) -> dict[str, Any]:
        described = explain(self.profile)
        described["circuit_reason"] = self.controller.circuit_reason
        return described

    def set_prefix_cache(self, prompt: str | Sequence[int] | None) -> None:
        """Encode prefix prompt (if string) and configure prefix cache on backend."""
        if prompt is None:
            prefix_ids = None
        elif isinstance(prompt, str):
            prefix_ids = list(self.backend.encode(prompt))
        else:
            prefix_ids = [int(token) for token in prompt]
        if hasattr(self.backend, "set_prefix_cache"):
            self.backend.set_prefix_cache(prefix_ids)
        else:
            raise AttributeError(
                f"{type(self.backend).__name__} does not implement set_prefix_cache"
            )

    # -- serving --------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = 32, *, model: str | None = None, **kwargs) -> Generation:
        backend = self.get_backend(model or kwargs.get("model_id"))
        try:
            token_ids = tuple(backend.encode(prompt))
        except Exception as exc:
            raise RuntimeExecutionError("prompt encoding failed") from exc
        scope = observe(
            model_id=getattr(backend, "model_id", None),
            model_revision=getattr(backend, "model_revision", None),
            token_ids=token_ids,
            output_tokens=max_tokens,
            **kwargs,
        )
        decision = self.controller.decide_scope(scope)
        knobs = {} if self.controller.is_fallback(decision) else knobs_for(self.profile)
        self._shadow_log(scope, decision, knobs, token_ids)
        with self.controller.guard(decision):
            result = backend.generate(token_ids, max_tokens, knobs)
            self._check_marker(result, knobs, max_tokens)
        return _generation(result, decision, knobs)

    def stream_generate(
        self, prompt: str | Sequence[int], max_tokens: int = 128, *, model: str | None = None, **kwargs
    ) -> Iterator[dict[str, Any]]:
        backend = self.get_backend(model or kwargs.get("model_id"))
        if isinstance(prompt, str):
            try:
                token_ids = tuple(backend.encode(prompt))
            except Exception as exc:
                raise RuntimeExecutionError("prompt encoding failed") from exc
        else:
            try:
                token_ids = tuple(int(x) for x in prompt)
            except Exception as exc:
                raise RuntimeExecutionError("invalid token sequence") from exc

        scope = observe(
            model_id=getattr(backend, "model_id", None),
            model_revision=getattr(backend, "model_revision", None),
            token_ids=token_ids,
            output_tokens=max_tokens,
            **kwargs,
        )
        decision = self.controller.decide_scope(scope)
        knobs = {} if self.controller.is_fallback(decision) else knobs_for(self.profile)
        self._shadow_log(scope, decision, knobs, token_ids)

        total_tokens_seen = 0
        has_done = False

        with self.controller.guard(decision):
            for event in backend.stream_generate(token_ids, max_tokens, knobs):
                event_type = event.get("type")
                if event_type == "token":
                    if event.get("is_first"):
                        total_tokens_seen += 1
                    else:
                        chunk = event.get("tokens")
                        if chunk:
                            total_tokens_seen += len(chunk)
                        else:
                            total_tokens_seen += 1
                    if total_tokens_seen > max_tokens:
                        raise RuntimeExecutionError("engine returned more tokens than requested")
                elif event_type == "done":
                    has_done = True
                    used = event.get("knobs")
                    if not isinstance(used, Mapping):
                        raise RuntimeExecutionError("engine did not report the knobs it used")
                    for name, value in knobs.items():
                        if used.get(name) != value:
                            raise RuntimeExecutionError(
                                f"authorised knob was not applied: {name}={value!r} vs {used.get(name)!r}"
                            )
                    if "plan" not in event:
                        event["plan"] = decision.plan
                    if "reason" not in event:
                        event["reason"] = decision.reason
                yield event

            if not has_done and total_tokens_seen == 0:
                raise RuntimeExecutionError("engine returned no tokens")

    def _shadow_log(
        self,
        scope: RequestScope | None,
        decision: DispatchDecision,
        applied_knobs: Mapping[str, Any],
        token_ids: Sequence[int],
    ) -> None:
        """RL shadow: record the action the bandit *would* pick. Never applies it.

        No reward is observed and no weight is updated from a serving request —
        there is no measured reward here, and RL stays NO-GO until R2.
        """

        if (
            self.rl_controller is None
            or scope is None
            or self.controller.is_fallback(decision)
        ):
            return
        has_overlap = detect_ngram_overlap(token_ids)
        action, shadow_knobs, score = self.rl_controller.select_action(
            scope.model_id,
            scope.prompt_tokens,
            scope.output_tokens,
            has_ngram_overlap=has_overlap,
        )
        self.rl_controller.log_decision(
            model_id=scope.model_id,
            prompt_tokens=scope.prompt_tokens,
            output_tokens=scope.output_tokens,
            shadow_action=action,
            shadow_knobs=shadow_knobs,
            shadow_score=score,
            applied_knobs=applied_knobs,
            applied_plan=decision.plan,
            has_ngram_overlap=has_overlap,
        )

    @staticmethod
    def _check_marker(
        result: Mapping[str, Any], knobs: Mapping[str, Any], max_tokens: int
    ) -> None:
        """The optimised path must prove it ran the way it was authorised to.

        The sealed head-skip runtime checks ``head_calls == 1`` after the fact
        (``executor.py:212-218``); the same idea generalises: the engine reports
        the knobs it actually used, and a mismatch between authorised and used
        is a failure, not a detail.
        """

        used = result.get("knobs")
        if not isinstance(used, Mapping):
            raise RuntimeExecutionError("engine did not report the knobs it used")
        for name, value in knobs.items():
            if used.get(name) != value:
                raise RuntimeExecutionError(
                    f"authorised knob was not applied: {name}={value!r} vs {used.get(name)!r}"
                )
        tokens = result.get("logical_tokens")
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or not tokens:
            raise RuntimeExecutionError("engine returned no tokens")
        if len(tokens) > max_tokens:
            raise RuntimeExecutionError("engine returned more tokens than requested")


def _generation(
    result: Mapping[str, Any], decision: DispatchDecision, knobs: Mapping[str, Any]
) -> Generation:
    from friday_evidence.canonical import canonical_sha256

    tokens = tuple(int(value) for value in result["logical_tokens"])
    text = result.get("text")
    return Generation(
        tokens=tokens,
        text=text if isinstance(text, str) else None,
        token_sha256=canonical_sha256(list(tokens)),
        prefill_ns=int(result.get("prefill_ns", 0)),
        decode_ns=int(result.get("decode_ns", 0)),
        plan=decision.plan,
        reason=decision.reason,
        knobs=dict(knobs),
        prefix_cache_hits=int(result.get("prefix_cache_hits", 0)),
    )


__all__ = [
    "BASELINE_PLAN",
    "DEVICE_PROFILE_PLAN",
    "Generation",
    "GenerationBackend",
    "Server",
]
