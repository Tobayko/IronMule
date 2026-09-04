"""Preregistered exploration campaign — the only honest source of overlap.

R2 needs a decision corpus in which more than one action was actually taken,
otherwise every importance weight is zero and off-policy evaluation refuses.
A preregistered single-candidate study cannot supply that: its action is fixed
in advance, so every propensity is one.

This module registers the opposite: the *rule* is sealed instead of the
action.  A campaign fixes the policy, the seed base, the point count and the
per-point budget before any measurement, so the drawn sequence is reproducible
from the seal and cannot be reordered afterwards.  That keeps the draws
preregistered while still producing propensity overlap.

Planning only.  It starts no model, reserves no hardware and authorises
nothing; every drawn point still needs its own user-started, gated run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes, sha256_hex
from .candidates import CandidateRegistry
from .decisions import DecisionError, DecisionEvent, SelectionPolicy, decide
from .fingerprint import ExactFingerprint

CAMPAIGN_SCHEMA = "friday.optimizer.campaign.v1"

#: Measured wall time per gated measurement point, in seconds.  Sealed
#: evidence: matmul-compile-ab spent 1000.41 s wall on six runs, of which
#: 937.71 s was mandated break.  The cooldown dominates the wall clock by
#: roughly 28x the GPU work, so a block's capacity is set by breaks, not by
#: how fast the device computes.
MEASURED_POINT_SECONDS = 167.0

#: One user-started block, per the standing 30-minute limit.
BLOCK_SECONDS = 1800.0

MAX_POINTS = 512


class CampaignError(ValueError):
    """Malformed campaign plan."""


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise CampaignError(f"{field} must be a positive finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """A sealed exploration plan: the rule is registered, not the action."""

    campaign_id: str
    policy: SelectionPolicy
    seed_base: int
    points: int
    hints: tuple[str, ...] = ()
    point_seconds: float = MEASURED_POINT_SECONDS
    block_seconds: float = BLOCK_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SelectionPolicy):
            raise CampaignError("campaign requires a SelectionPolicy")
        if self.policy.rule not in ("epsilon_greedy",):
            raise CampaignError("a campaign without a stochastic rule produces no overlap")
        if isinstance(self.seed_base, bool) or not isinstance(self.seed_base, int) or not 0 <= self.seed_base < 2**32:
            raise CampaignError("seed_base must be a bounded non-negative integer")
        if isinstance(self.points, bool) or not isinstance(self.points, int) or not 1 <= self.points <= MAX_POINTS:
            raise CampaignError(f"points must be between 1 and {MAX_POINTS}")
        object.__setattr__(self, "hints", tuple(self.hints))
        object.__setattr__(self, "point_seconds", _positive_number(self.point_seconds, "point_seconds"))
        object.__setattr__(self, "block_seconds", _positive_number(self.block_seconds, "block_seconds"))

    @property
    def points_per_block(self) -> int:
        """How many points actually fit into one user-started block."""

        return max(1, int(self.block_seconds // self.point_seconds))

    @property
    def blocks(self) -> int:
        """Approved blocks this campaign needs, rounded up."""

        return math.ceil(self.points / self.points_per_block)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": self.campaign_id,
            "policy": self.policy.as_dict(),
            "policy_hash": self.policy.policy_hash,
            "seed_base": self.seed_base,
            "points": self.points,
            "hints": list(self.hints),
            "point_seconds": self.point_seconds,
            "block_seconds": self.block_seconds,
            "points_per_block": self.points_per_block,
            "blocks": self.blocks,
        }

    @property
    def campaign_hash(self) -> str:
        return sha256_hex(canonical_bytes(self.as_dict()))

    def seed_for(self, index: int) -> int:
        """Deterministic per-point seed, derived from the sealed base."""

        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.points:
            raise CampaignError("point index is outside the campaign")
        digest = sha256_hex(canonical_bytes({"campaign": self.campaign_id, "base": self.seed_base, "index": index}))
        return int(digest[:15], 16)

    def decisions(
        self,
        fingerprint: ExactFingerprint,
        *,
        registry: CandidateRegistry | None = None,
        qualified: Iterable[str] = (),
    ) -> tuple[DecisionEvent, ...]:
        """Draw the whole sealed sequence without measuring anything."""

        used = registry if registry is not None else CandidateRegistry()
        events = []
        for index in range(self.points):
            events.append(
                decide(
                    self.policy, fingerprint, registry=used, qualified=qualified,
                    hints=self.hints, seed=self.seed_for(index),
                    decision_id=f"{self.campaign_id}.{index:04d}",
                )
            )
        return tuple(events)


def action_probability(policy: SelectionPolicy, candidates: Sequence[str], action: str, hints: Sequence[str] = ()) -> float:
    """Probability this policy assigns to *action* over *candidates*."""

    distribution = policy.distribution(candidates, hints)
    if action not in distribution:
        raise CampaignError("action is outside the candidate set")
    return float(distribution[action])


def expected_effective_samples(
    *,
    logging_policy: SelectionPolicy,
    target_policy: SelectionPolicy,
    candidates: Sequence[str],
    points: int,
    hints: Sequence[str] = (),
    target_hints: Sequence[str] | None = None,
) -> float:
    """Expected Kish effective sample size of a campaign, before running it.

    For a deterministic target the weights collapse onto one action, and the
    expectation reduces to ``points * p_logging(target action)``: exploring a
    rarely logged action is exactly as expensive as its rarity.

    ``target_hints`` defaults to ``hints`` and mirrors the identically named
    parameter in :mod:`friday_optimizer.replay`: the default reads hints as
    context shared by every policy, a different value scores a policy that
    would have been hinted differently.  Either way the corpus itself is never
    rewritten; only the target being priced changes.
    """

    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise CampaignError("points must be a positive integer")
    logged = logging_policy.distribution(candidates, hints)
    target = target_policy.distribution(candidates, hints if target_hints is None else target_hints)
    total = 0.0
    squares = 0.0
    for action, probability in logged.items():
        if probability <= 0.0:
            continue
        weight = target.get(action, 0.0) / probability
        total += probability * weight
        squares += probability * weight * weight
    if squares <= 0.0:
        return 0.0
    return points * (total * total) / squares


def points_for_effective_samples(
    *,
    logging_policy: SelectionPolicy,
    target_policy: SelectionPolicy,
    candidates: Sequence[str],
    required: int,
    hints: Sequence[str] = (),
    target_hints: Sequence[str] | None = None,
) -> int | None:
    """Smallest point count reaching *required* effective samples, or ``None``.

    ``None`` means no overlap: the target puts mass where the logging policy
    never draws, and no amount of measurement fixes that.
    """

    if isinstance(required, bool) or not isinstance(required, int) or required < 1:
        raise CampaignError("required must be a positive integer")
    unit = expected_effective_samples(
        logging_policy=logging_policy, target_policy=target_policy,
        candidates=candidates, points=1, hints=hints, target_hints=target_hints,
    )
    if unit <= 0.0:
        return None
    return math.ceil(required / unit)


def plan_for_target(
    *,
    campaign_id: str,
    logging_policy: SelectionPolicy,
    target_policy: SelectionPolicy,
    candidates: Sequence[str],
    required: int,
    seed_base: int,
    hints: Sequence[str] = (),
    target_hints: Sequence[str] | None = None,
    point_seconds: float = MEASURED_POINT_SECONDS,
    block_seconds: float = BLOCK_SECONDS,
) -> CampaignPlan | None:
    """Size a campaign so its corpus can actually answer the target question."""

    points = points_for_effective_samples(
        logging_policy=logging_policy, target_policy=target_policy,
        candidates=candidates, required=required, hints=hints, target_hints=target_hints,
    )
    if points is None or points > MAX_POINTS:
        return None
    return CampaignPlan(
        campaign_id=campaign_id, policy=logging_policy, seed_base=seed_base, points=points,
        hints=tuple(hints), point_seconds=point_seconds, block_seconds=block_seconds,
    )


__all__ = [
    "BLOCK_SECONDS",
    "CAMPAIGN_SCHEMA",
    "MAX_POINTS",
    "MEASURED_POINT_SECONDS",
    "CampaignError",
    "CampaignPlan",
    "action_probability",
    "expected_effective_samples",
    "plan_for_target",
    "points_for_effective_samples",
]
