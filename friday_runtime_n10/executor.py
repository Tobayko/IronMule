"""Narrow N10 AVO-lite boundary for evidence-authorized batched dispatch."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .constants import BATCHED_PLAN, OPERATION, SERIAL_PLAN
from .policy import PolicyDecision, PolicyEvidence, Workload, decision_for, load_policy


class Backend(Protocol):
    def matmul(self, left: Any, right: Any) -> Any: ...

    def eval_many(self, values: Sequence[Any]) -> None: ...

    def synchronize(self) -> None: ...


class RuntimeExecutionError(RuntimeError):
    """The selected plan failed; an optimized failure is never retried implicitly."""


@dataclass(frozen=True)
class ExecutionResult:
    outputs: tuple[Any, ...]
    decision: PolicyDecision


def _shape(value: Any) -> tuple[int, int] | None:
    raw = getattr(value, "shape", None)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in raw):
        return None
    return int(raw[0]), int(raw[1])


def _dtype(value: Any) -> str | None:
    raw = getattr(value, "dtype", None)
    if raw is None:
        return None
    text = str(raw).lower()
    if text.startswith("mlx.core."):
        text = text.removeprefix("mlx.core.")
    return text


def observe_workload(left: Any, operands: Sequence[Any]) -> Workload | None:
    """Derive scope from actual tensors; caller labels never authorize batching."""

    if isinstance(operands, (str, bytes)) or not isinstance(operands, Sequence) or not operands:
        return None
    left_shape = _shape(left)
    left_dtype = _dtype(left)
    operand_shapes = {_shape(value) for value in operands}
    operand_dtypes = {_dtype(value) for value in operands}
    if (
        left_shape is None
        or left_dtype is None
        or None in operand_shapes
        or None in operand_dtypes
        or len(operand_shapes) != 1
        or len(operand_dtypes) != 1
        or operand_dtypes != {left_dtype}
    ):
        return None
    rhs_shape = next(iter(operand_shapes))
    if left_shape[1] != rhs_shape[0]:
        return None
    return Workload(
        operation=OPERATION,
        dtype=left_dtype,
        lhs_shape=left_shape,
        rhs_shape=rhs_shape,
        output_shape=(left_shape[0], rhs_shape[1]),
        rhs_count=len(operands),
    )


def execute_serial(backend: Backend, left: Any, operands: Sequence[Any]) -> tuple[Any, ...]:
    outputs: list[Any] = []
    for operand in operands:
        value = backend.matmul(left, operand)
        backend.eval_many([value])
        backend.synchronize()
        outputs.append(value)
    return tuple(outputs)


def execute_batched(backend: Backend, left: Any, operands: Sequence[Any]) -> tuple[Any, ...]:
    outputs = tuple(backend.matmul(left, operand) for operand in operands)
    backend.eval_many(outputs)
    backend.synchronize()
    return outputs


class RuntimeController:
    """Cache evidence once and latch optimized failures to serial for the process lifetime."""

    def __init__(self, evidence: PolicyEvidence) -> None:
        self.evidence = evidence
        self._circuit_reason: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_evidence(cls, *args: Any, **kwargs: Any) -> "RuntimeController":
        return cls(load_policy(*args, **kwargs))

    @property
    def circuit_reason(self) -> str | None:
        with self._lock:
            return self._circuit_reason

    def decide(self, left: Any, operands: Sequence[Any]) -> PolicyDecision:
        workload = observe_workload(left, operands)
        with self._lock:
            circuit_reason = self._circuit_reason
        if circuit_reason is not None:
            return PolicyDecision(
                "serial", SERIAL_PLAN, "circuit_breaker_latched", self.evidence
            )
        return decision_for(self.evidence, workload)

    def _trip(self, failure: BaseException) -> None:
        with self._lock:
            if self._circuit_reason is None:
                self._circuit_reason = type(failure).__name__

    def execute(
        self, backend: Backend, left: Any, operands: Sequence[Any]
    ) -> ExecutionResult:
        decision = self.decide(left, operands)
        try:
            if decision.plan == BATCHED_PLAN:
                outputs = execute_batched(backend, left, operands)
            elif decision.plan == SERIAL_PLAN:
                outputs = execute_serial(backend, left, operands)
            else:  # Defensive against a future widened plan registry.
                raise RuntimeExecutionError("unregistered runtime plan")
        except Exception as exc:
            if decision.plan == BATCHED_PLAN:
                self._trip(exc)
                raise RuntimeExecutionError(
                    "batched plan failed; circuit breaker latched; current call was not retried"
                ) from exc
            raise RuntimeExecutionError("serial plan failed") from exc
        return ExecutionResult(outputs=outputs, decision=decision)


__all__ = [
    "Backend",
    "ExecutionResult",
    "RuntimeController",
    "RuntimeExecutionError",
    "execute_batched",
    "execute_serial",
    "observe_workload",
]
