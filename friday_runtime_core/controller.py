"""The dispatch boundary all three runtimes share, with the breaker made durable.

The copies differ in what they dispatch — matmul plans, prefill plans — but not
in how. In every one of them the controller caches verified evidence, refuses
the optimised plan while the breaker is latched, and treats a failure on the
optimised path differently from a failure on the fallback path: the first
latches and is never implicitly retried, the second is simply a failure.

The scope check stays with the caller on purpose. ``decide_scope`` receives a
scope the caller derived from the **actual** tensors and tokens, never from a
label the caller asserts; that property is what makes an unattended dispatch
safe and it is not weakened here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .breaker import CircuitBreaker, Latch

CIRCUIT_REASON = "circuit_breaker_latched"


class RuntimeExecutionError(RuntimeError):
    """The selected execution path failed without an implicit same-call retry."""


@dataclass(frozen=True)
class DispatchDecision:
    mode: str
    plan: str
    reason: str
    evidence: Any = None


class DispatchController:
    """Cache evidence once, refuse optimised plans while latched, latch durably."""

    def __init__(
        self,
        evidence: Any,
        *,
        decide: Callable[[Any, Any], DispatchDecision],
        fallback: DispatchDecision,
        latch: Latch | None = None,
    ) -> None:
        if not callable(decide):
            raise RuntimeExecutionError("decide must be callable")
        if not isinstance(fallback, DispatchDecision):
            raise RuntimeExecutionError("fallback must be a DispatchDecision")
        self.evidence = evidence
        self._decide = decide
        self._fallback = fallback
        self.breaker = CircuitBreaker(latch)

    @property
    def circuit_reason(self) -> str | None:
        return self.breaker.reason

    def fallback_decision(self, reason: str) -> DispatchDecision:
        return DispatchDecision(
            self._fallback.mode, self._fallback.plan, reason, self.evidence
        )

    def decide_scope(self, scope: Any) -> DispatchDecision:
        """Return the plan for this scope, or the fallback while latched."""

        latched = self.breaker.reason
        if latched is not None:
            return self.fallback_decision(CIRCUIT_REASON)
        decision = self._decide(self.evidence, scope)
        if not isinstance(decision, DispatchDecision):
            raise RuntimeExecutionError("decide returned an unregistered decision")
        return decision

    def is_fallback(self, decision: DispatchDecision) -> bool:
        return decision.plan == self._fallback.plan

    @contextmanager
    def guard(self, decision: DispatchDecision) -> Iterator[None]:
        """Run one attempt under this decision, latching an optimised failure.

        A failure on the optimised path trips the breaker before the error
        leaves, so the *next* process starts on the baseline too. The current
        call is never silently retried on the fallback: the caller sees the
        failure and decides.
        """

        try:
            yield
        except Exception as exc:
            if self.is_fallback(decision):
                raise RuntimeExecutionError("baseline path failed") from exc
            self.breaker.trip(exc)
            raise RuntimeExecutionError(
                f"{decision.plan} failed; circuit breaker latched; "
                "current call was not retried"
            ) from exc


__all__ = [
    "CIRCUIT_REASON",
    "DispatchController",
    "DispatchDecision",
    "RuntimeExecutionError",
]
