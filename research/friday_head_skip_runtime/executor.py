"""Narrow generation controller with observable fallback and a circuit breaker."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from friday_evidence.canonical import canonical_sha256

from .constants import BASELINE_PLAN, HEAD_SKIP_PLAN
from .policy import (
    PolicyDecision,
    PolicyEvidence,
    RequestScope,
    decision_for,
    load_gpu_qualification_policy,
    load_policy,
    load_runtime_policy,
)


class RuntimeExecutionError(RuntimeError):
    """The selected execution path failed without an implicit same-call retry."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt_content: str
    output_tokens: int
    temperature: float = 0.0
    prompt_logprobs: bool = False
    fixed_horizon: bool = True
    batch: int = 1


@dataclass(frozen=True)
class GenerationOutput:
    token_ids: tuple[int, ...]
    token_sha256: str
    text: str | None
    prefill_ns: int
    total_ns: int
    prefill_blocks: int
    head_calls: int
    memory: Mapping[str, int | None]


@dataclass(frozen=True)
class ExecutionResult:
    output: GenerationOutput
    decision: PolicyDecision


class GenerationBackend(Protocol):
    model_id: str
    model_revision: str
    prefill_chunk: int

    def encode_prompt(self, prompt_content: str) -> Sequence[int]: ...

    def generate_baseline(
        self, token_ids: Sequence[int], request: GenerationRequest
    ) -> GenerationOutput: ...

    def generate_head_skip(
        self, token_ids: Sequence[int], request: GenerationRequest
    ) -> GenerationOutput: ...


def observe_request(
    backend: GenerationBackend,
    request: GenerationRequest,
    token_ids: Sequence[int],
) -> RequestScope | None:
    model_id = getattr(backend, "model_id", None)
    model_revision = getattr(backend, "model_revision", None)
    prefill_chunk = getattr(backend, "prefill_chunk", None)
    if (
        not isinstance(request.prompt_content, str)
        or not request.prompt_content
        or isinstance(request.output_tokens, bool)
        or not isinstance(request.output_tokens, int)
        or request.output_tokens <= 0
        or isinstance(request.batch, bool)
        or not isinstance(request.batch, int)
        or request.batch <= 0
        or type(request.prompt_logprobs) is not bool
        or type(request.fixed_horizon) is not bool
        or not isinstance(model_id, str)
        or not isinstance(model_revision, str)
        or isinstance(prefill_chunk, bool)
        or not isinstance(prefill_chunk, int)
        or prefill_chunk <= 0
        or isinstance(token_ids, (str, bytes))
        or not isinstance(token_ids, Sequence)
        or not token_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        )
    ):
        return None
    temperature = request.temperature
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        return None
    return RequestScope(
        model_id=model_id,
        model_revision=model_revision,
        prompt_content_sha256=hashlib.sha256(
            request.prompt_content.encode("utf-8")
        ).hexdigest(),
        prompt_tokens=len(token_ids),
        prefill_chunk=prefill_chunk,
        batch=request.batch,
        temperature=float(temperature),
        prompt_logprobs=request.prompt_logprobs,
        fixed_horizon=request.fixed_horizon,
        output_tokens=request.output_tokens,
    )


def _validate_output(output: GenerationOutput, request: GenerationRequest) -> None:
    if (
        not isinstance(output, GenerationOutput)
        or type(output.token_ids) is not tuple
        or len(output.token_ids) != request.output_tokens
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in output.token_ids
        )
        or output.token_sha256 != canonical_sha256(list(output.token_ids))
        or (output.text is not None and not isinstance(output.text, str))
        or isinstance(output.prefill_ns, bool)
        or not isinstance(output.prefill_ns, int)
        or output.prefill_ns <= 0
        or isinstance(output.total_ns, bool)
        or not isinstance(output.total_ns, int)
        or output.total_ns < output.prefill_ns
        or isinstance(output.prefill_blocks, bool)
        or not isinstance(output.prefill_blocks, int)
        or output.prefill_blocks <= 0
        or isinstance(output.head_calls, bool)
        or not isinstance(output.head_calls, int)
        or output.head_calls <= 0
        or not isinstance(output.memory, Mapping)
        or any(
            not isinstance(key, str)
            or (value is not None and (type(value) is not int or value < 0))
            for key, value in output.memory.items()
        )
    ):
        raise RuntimeExecutionError("backend output violates the runtime contract")


class RuntimeController:
    """Cache verified evidence and latch optimized failures for this process."""

    def __init__(self, evidence: PolicyEvidence) -> None:
        self.evidence = evidence
        self._circuit_reason: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_evidence(cls, *args: Any, **kwargs: Any) -> "RuntimeController":
        return cls(load_runtime_policy(*args, **kwargs))

    @classmethod
    def for_cpu_qualification(cls, *args: Any, **kwargs: Any) -> "RuntimeController":
        return cls(load_policy(*args, **kwargs))

    @classmethod
    def for_gpu_qualification(cls, *args: Any, **kwargs: Any) -> "RuntimeController":
        return cls(load_gpu_qualification_policy(*args, **kwargs))

    @property
    def circuit_reason(self) -> str | None:
        with self._lock:
            return self._circuit_reason

    def decide_scope(self, scope: RequestScope | None) -> PolicyDecision:
        with self._lock:
            tripped = self._circuit_reason is not None
        if tripped:
            return PolicyDecision(
                "baseline", BASELINE_PLAN, "circuit_breaker_latched", self.evidence
            )
        return decision_for(self.evidence, scope)

    def _trip(self, failure: BaseException) -> None:
        with self._lock:
            if self._circuit_reason is None:
                self._circuit_reason = type(failure).__name__

    def execute(
        self, backend: GenerationBackend, request: GenerationRequest
    ) -> ExecutionResult:
        try:
            token_ids = tuple(backend.encode_prompt(request.prompt_content))
        except Exception as exc:
            raise RuntimeExecutionError("baseline prompt encoding failed") from exc
        decision = self.decide_scope(observe_request(backend, request, token_ids))
        try:
            if decision.plan == HEAD_SKIP_PLAN:
                output = backend.generate_head_skip(token_ids, request)
                _validate_output(output, request)
                expected_blocks = (
                    len(token_ids) + backend.prefill_chunk - 1
                ) // backend.prefill_chunk
                if output.prefill_blocks != expected_blocks or output.head_calls != 1:
                    raise RuntimeExecutionError("optimized path marker differs")
            elif decision.plan == BASELINE_PLAN:
                output = backend.generate_baseline(token_ids, request)
                _validate_output(output, request)
            else:
                raise RuntimeExecutionError("unregistered runtime plan")
        except Exception as exc:
            if decision.plan == HEAD_SKIP_PLAN:
                self._trip(exc)
                raise RuntimeExecutionError(
                    "head-skip path failed; circuit breaker latched; current call was not retried"
                ) from exc
            raise RuntimeExecutionError("baseline path failed") from exc
        return ExecutionResult(output=output, decision=decision)


__all__ = [
    "ExecutionResult",
    "GenerationBackend",
    "GenerationOutput",
    "GenerationRequest",
    "RuntimeController",
    "RuntimeExecutionError",
    "observe_request",
]
