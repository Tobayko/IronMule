"""Fail-closed N8/N10 policy composition with shadow-only enforcement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from friday_runtime.executor import RuntimeController as N8Controller
from friday_runtime.policy import PolicyDecision as N8PolicyDecision
from friday_runtime_n10.executor import RuntimeController as N10Controller
from friday_runtime_n10.policy import PolicyDecision as N10PolicyDecision

from .constants import ENFORCED_PLAN, N10_RHS_COUNT, N8_RHS_COUNT, SERIAL_PLAN


Route = Literal["n8", "n10", "serial"]
UnderlyingDecision = N8PolicyDecision | N10PolicyDecision


@dataclass(frozen=True)
class ShadowDecision:
    route: Route
    recommendation_strategy: str
    recommendation_plan: str
    enforced_plan: str
    reason: str
    rhs_count: int | None
    n8_authorized: bool
    n10_authorized: bool
    decision_record_ids: dict[str, str | None]

    def as_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "recommendation_strategy": self.recommendation_strategy,
            "recommendation_plan": self.recommendation_plan,
            "enforced_plan": self.enforced_plan,
            "reason": self.reason,
            "rhs_count": self.rhs_count,
            "n8_authorized": self.n8_authorized,
            "n10_authorized": self.n10_authorized,
            "decision_record_ids": dict(self.decision_record_ids),
        }


def _count(operands: object) -> int | None:
    if isinstance(operands, (str, bytes)) or not isinstance(operands, Sequence):
        return None
    try:
        return len(operands)
    except (OverflowError, TypeError, ValueError):
        return None


class ShadowRouter:
    """Compose both sealed policies while enforcing serial shadow execution."""

    def __init__(self, n8: N8Controller, n10: N10Controller) -> None:
        self.n8 = n8
        self.n10 = n10

    @classmethod
    def from_evidence(cls, *args: Any, **kwargs: Any) -> "ShadowRouter":
        n8_args = kwargs.pop("n8_args", ())
        n8_kwargs = kwargs.pop("n8_kwargs", {})
        n10_args = kwargs.pop("n10_args", ())
        n10_kwargs = kwargs.pop("n10_kwargs", {})
        if args or kwargs:
            raise TypeError("router evidence must use explicit n8/n10 arguments")
        return cls(
            N8Controller.from_evidence(*n8_args, **n8_kwargs),
            N10Controller.from_evidence(*n10_args, **n10_kwargs),
        )

    @property
    def ready(self) -> bool:
        return bool(self.n8.evidence.authorized and self.n10.evidence.authorized)

    @property
    def decision_record_ids(self) -> dict[str, str | None]:
        return {
            "n8": self.n8.evidence.decision_record_id,
            "n10": self.n10.evidence.decision_record_id,
        }

    def direct_decision(
        self, left: Any, operands: Sequence[Any]
    ) -> tuple[Route, UnderlyingDecision | None]:
        count = _count(operands)
        if count == N8_RHS_COUNT:
            decision = self.n8.decide(left, operands)
            return ("n8" if decision.strategy == "batched" else "serial"), decision
        if count == N10_RHS_COUNT:
            decision = self.n10.decide(left, operands)
            return ("n10" if decision.strategy == "batched" else "serial"), decision
        return "serial", None

    def decide(self, left: Any, operands: Sequence[Any]) -> ShadowDecision:
        count = _count(operands)
        ids = self.decision_record_ids
        if not self.ready:
            return ShadowDecision(
                route="serial",
                recommendation_strategy="serial",
                recommendation_plan=SERIAL_PLAN,
                enforced_plan=ENFORCED_PLAN,
                reason="router_evidence_incomplete",
                rhs_count=count,
                n8_authorized=self.n8.evidence.authorized,
                n10_authorized=self.n10.evidence.authorized,
                decision_record_ids=ids,
            )

        route, decision = self.direct_decision(left, operands)
        if decision is None:
            return ShadowDecision(
                route="serial",
                recommendation_strategy="serial",
                recommendation_plan=SERIAL_PLAN,
                enforced_plan=ENFORCED_PLAN,
                reason="unregistered_rhs_count",
                rhs_count=count,
                n8_authorized=True,
                n10_authorized=True,
                decision_record_ids=ids,
            )
        return ShadowDecision(
            route=route,
            recommendation_strategy=decision.strategy,
            recommendation_plan=decision.plan,
            enforced_plan=ENFORCED_PLAN,
            reason=decision.reason,
            rhs_count=count,
            n8_authorized=True,
            n10_authorized=True,
            decision_record_ids=ids,
        )

    def evidence_summary(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "enforced_plan": ENFORCED_PLAN,
            "decision_record_ids": self.decision_record_ids,
            "n8": {
                "authorized": self.n8.evidence.authorized,
                "reason": self.n8.evidence.reason,
                "records": self.n8.evidence.evidence_records,
            },
            "n10": {
                "authorized": self.n10.evidence.authorized,
                "reason": self.n10.evidence.reason,
                "records": self.n10.evidence.evidence_records,
            },
        }
