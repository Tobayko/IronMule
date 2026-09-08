"""Fail-closed selection from independently validated integration evidence.

This module is deliberately path and storage free.  It does not benchmark hardware,
load models, or turn historical measurements into product authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_IDS = frozenset({
    "reference",
    "prefix_reuse",
    "baseline_interactive",
    "core_interactive",
    "baseline_throughput",
    "core_throughput",
})
REFERENCE_BY_PROFILE = {
    "interactive": "reference",
    "throughput": "reference",
}
_PROFILES = frozenset(REFERENCE_BY_PROFILE)
_SPLITS = frozenset({"train", "validation", "heldout"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SelectionContractError(ValueError):
    """Input is malformed and cannot be interpreted conservatively."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SelectionContractError(f"{name} must be a non-empty string")
    return value


def _sha(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _SHA256.fullmatch(value):
        raise SelectionContractError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SelectionContractError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise SelectionContractError(f"{name} must be a {'positive ' if positive else ''}finite number")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SelectionContractError(f"{name} must be a boolean")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SelectionContractError(f"{name} has missing or unsupported fields")


@dataclass(frozen=True)
class RequestContext:
    model_id: str
    model_revision: str
    model_sha256: str
    hardware_sha256: str
    environment_sha256: str
    code_sha256: str
    source_manifest_sha256: str
    profile: str
    session_requests: int
    context_tokens: int
    max_new_tokens: int
    stream: bool
    group_width: int
    prefix_hit: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RequestContext":
        fields = set(cls.__dataclass_fields__)
        _exact_keys(value, fields, "request context")
        profile = _text(value["profile"], "profile")
        if profile not in _PROFILES:
            raise SelectionContractError("profile is unsupported")
        return cls(
            model_id=_text(value["model_id"], "model_id"),
            model_revision=_text(value["model_revision"], "model_revision"),
            model_sha256=_sha(value["model_sha256"], "model_sha256"),
            hardware_sha256=_sha(value["hardware_sha256"], "hardware_sha256"),
            environment_sha256=_sha(value["environment_sha256"], "environment_sha256"),
            code_sha256=_sha(value["code_sha256"], "code_sha256"),
            source_manifest_sha256=_sha(value["source_manifest_sha256"], "source_manifest_sha256"),
            profile=profile,
            session_requests=_integer(value["session_requests"], "session_requests", minimum=1),
            context_tokens=_integer(value["context_tokens"], "context_tokens"),
            max_new_tokens=_integer(value["max_new_tokens"], "max_new_tokens", minimum=1),
            stream=_boolean(value["stream"], "stream"),
            group_width=_integer(value["group_width"], "group_width", minimum=1),
            prefix_hit=_boolean(value["prefix_hit"], "prefix_hit"),
        )


@dataclass(frozen=True)
class IntegrationEvidence:
    audit_id: str
    candidate_id: str
    baseline_candidate_id: str
    model_id: str
    model_revision: str
    model_sha256: str
    hardware_sha256: str
    environment_sha256: str
    code_sha256: str
    source_manifest_sha256: str
    heldout: bool
    quality_exact: bool
    profile: str
    min_session_requests: int
    max_session_requests: int
    min_context_tokens: int
    max_context_tokens: int
    min_new_tokens: int
    max_new_tokens: int
    streamable: bool
    group_widths: tuple[int, ...]
    prefix_hit_capable: bool
    prefix_hit_required: bool
    cold_setup_included: bool
    estimated_e2e_ms: float
    pair_ratio: float
    confidence_upper: float
    frozen_min_effect: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntegrationEvidence":
        fields = set(cls.__dataclass_fields__)
        _exact_keys(value, fields, "integration evidence")
        candidate = _text(value["candidate_id"], "candidate_id")
        baseline = _text(value["baseline_candidate_id"], "baseline_candidate_id")
        if candidate not in CANDIDATE_IDS or baseline not in CANDIDATE_IDS or candidate == baseline:
            raise SelectionContractError("candidate/baseline identity is invalid")
        profile = _text(value["profile"], "profile")
        if profile not in _PROFILES:
            raise SelectionContractError("profile is unsupported")
        widths_value = value["group_widths"]
        if not isinstance(widths_value, (list, tuple)) or not widths_value:
            raise SelectionContractError("group_widths must be a non-empty array")
        widths = tuple(_integer(item, "group_width", minimum=1) for item in widths_value)
        if tuple(sorted(set(widths))) != widths:
            raise SelectionContractError("group_widths must be unique and sorted")
        result = cls(
            audit_id=_text(value["audit_id"], "audit_id"),
            candidate_id=candidate,
            baseline_candidate_id=baseline,
            model_id=_text(value["model_id"], "model_id"),
            model_revision=_text(value["model_revision"], "model_revision"),
            model_sha256=_sha(value["model_sha256"], "model_sha256"),
            hardware_sha256=_sha(value["hardware_sha256"], "hardware_sha256"),
            environment_sha256=_sha(value["environment_sha256"], "environment_sha256"),
            code_sha256=_sha(value["code_sha256"], "code_sha256"),
            source_manifest_sha256=_sha(value["source_manifest_sha256"], "source_manifest_sha256"),
            heldout=_boolean(value["heldout"], "heldout"),
            quality_exact=_boolean(value["quality_exact"], "quality_exact"),
            profile=profile,
            min_session_requests=_integer(value["min_session_requests"], "min_session_requests", minimum=1),
            max_session_requests=_integer(value["max_session_requests"], "max_session_requests", minimum=1),
            min_context_tokens=_integer(value["min_context_tokens"], "min_context_tokens"),
            max_context_tokens=_integer(value["max_context_tokens"], "max_context_tokens"),
            min_new_tokens=_integer(value["min_new_tokens"], "min_new_tokens", minimum=1),
            max_new_tokens=_integer(value["max_new_tokens"], "max_new_tokens", minimum=1),
            streamable=_boolean(value["streamable"], "streamable"),
            group_widths=widths,
            prefix_hit_capable=_boolean(value["prefix_hit_capable"], "prefix_hit_capable"),
            prefix_hit_required=_boolean(value["prefix_hit_required"], "prefix_hit_required"),
            cold_setup_included=_boolean(value["cold_setup_included"], "cold_setup_included"),
            estimated_e2e_ms=_number(value["estimated_e2e_ms"], "estimated_e2e_ms", positive=True),
            pair_ratio=_number(value["pair_ratio"], "pair_ratio", positive=True),
            confidence_upper=_number(value["confidence_upper"], "confidence_upper", positive=True),
            frozen_min_effect=_number(value["frozen_min_effect"], "frozen_min_effect"),
        )
        if (result.max_session_requests < result.min_session_requests
                or result.max_context_tokens < result.min_context_tokens
                or result.max_new_tokens < result.min_new_tokens):
            raise SelectionContractError("evidence token bounds are inverted")
        if not 0 <= result.frozen_min_effect < 1:
            raise SelectionContractError("frozen_min_effect must be in [0, 1)")
        if result.confidence_upper < result.pair_ratio:
            raise SelectionContractError("confidence_upper cannot be below pair_ratio")
        if result.prefix_hit_required and not result.prefix_hit_capable:
            raise SelectionContractError("required prefix hits must be supported")
        if result.candidate_id == "prefix_reuse":
            if not result.prefix_hit_capable:
                raise SelectionContractError("prefix_reuse must support prefix hits")
            if not result.prefix_hit_required and not result.cold_setup_included:
                raise SelectionContractError("cold prefix evidence must include setup")
        return result

    def matches(self, request: RequestContext) -> bool:
        return (
            self.model_id == request.model_id
            and self.model_revision == request.model_revision
            and self.model_sha256 == request.model_sha256
            and self.hardware_sha256 == request.hardware_sha256
            and self.environment_sha256 == request.environment_sha256
            and self.code_sha256 == request.code_sha256
            and self.source_manifest_sha256 == request.source_manifest_sha256
            and self.profile == request.profile
            and self.min_session_requests <= request.session_requests <= self.max_session_requests
            and self.min_context_tokens <= request.context_tokens <= self.max_context_tokens
            and self.min_new_tokens <= request.max_new_tokens <= self.max_new_tokens
            and (not request.stream or self.streamable)
            and request.group_width in self.group_widths
            and (not self.prefix_hit_required or request.prefix_hit)
        )

    @property
    def proves_improvement(self) -> bool:
        return (self.heldout and self.quality_exact and self.confidence_upper < 1.0
                and self.confidence_upper <= 1.0 - self.frozen_min_effect)


@dataclass(frozen=True)
class SelectionDecision:
    candidate_id: str
    reason: str
    evidence_audit_id: str | None
    explored: bool = False


def select_candidate(
    request: RequestContext,
    evidence: Iterable[IntegrationEvidence],
    *,
    incumbent_candidate_id: str | None = None,
) -> SelectionDecision:
    """Choose a proven arm, otherwise return the profile's reference arm.

    Switching from a non-reference incumbent needs a direct held-out comparison
    against that incumbent.  No exploration occurs in this request path.
    """
    reference = REFERENCE_BY_PROFILE[request.profile]
    incumbent = incumbent_candidate_id or reference
    if incumbent not in CANDIDATE_IDS:
        raise SelectionContractError("incumbent_candidate_id is unsupported")
    eligible = [row for row in evidence if row.matches(request) and row.proves_improvement
                and row.baseline_candidate_id == incumbent]
    if not eligible:
        return SelectionDecision(reference, "no_matching_independent_heldout_promotion", None)
    winner = min(eligible, key=lambda row: (row.estimated_e2e_ms, row.candidate_id, row.audit_id))
    return SelectionDecision(winner.candidate_id, "fastest_proven_matching_e2e_candidate", winner.audit_id)


@dataclass(frozen=True)
class RewardObservation:
    audit_id: str
    candidate_id: str
    split: str
    cluster_id: str
    model_id: str
    model_sha256: str
    profile: str
    session_requests: int
    prefix_hit: bool
    actual_reward: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RewardObservation":
        _exact_keys(value, set(cls.__dataclass_fields__), "reward observation")
        candidate = _text(value["candidate_id"], "candidate_id")
        split = _text(value["split"], "split")
        if candidate not in CANDIDATE_IDS or split not in _SPLITS:
            raise SelectionContractError("reward candidate or split is unsupported")
        profile = _text(value["profile"], "profile")
        if profile not in _PROFILES:
            raise SelectionContractError("reward profile is unsupported")
        return cls(_text(value["audit_id"], "audit_id"), candidate, split,
                   _text(value["cluster_id"], "cluster_id"),
                   _text(value["model_id"], "model_id"),
                   _sha(value["model_sha256"], "model_sha256"), profile,
                   _integer(value["session_requests"], "session_requests", minimum=1),
                   _boolean(value["prefix_hit"], "prefix_hit"),
                   _number(value["actual_reward"], "actual_reward"))


@dataclass(frozen=True)
class RankProposal:
    candidate_id: str
    model_id: str
    model_sha256: str
    profile: str
    session_requests: int
    prefix_hit: bool
    observed_count: int
    mean_reward: float
    uncertainty: float
    score: float


def propose_ranks(observations: Sequence[RewardObservation]) -> tuple[RankProposal, ...]:
    """Rank bounded arms from actual audited train/validation rewards only.

    This transparent mean-minus-standard-error score is a proposal, not a
    promotion and not a claim that an RL policy was trained.
    """
    audit_ids: set[str] = set()
    split_by_cluster: dict[str, str] = {}
    grouped: dict[tuple[str, str, str, str, int, bool], list[float]] = {}
    for row in observations:
        if row.audit_id in audit_ids:
            raise SelectionContractError("reward audit_id must be unique")
        audit_ids.add(row.audit_id)
        previous = split_by_cluster.setdefault(row.cluster_id, row.split)
        if previous != row.split:
            raise SelectionContractError("run/process cluster crosses data splits")
        if row.split != "heldout":
            key = (row.candidate_id, row.model_id, row.model_sha256, row.profile,
                   row.session_requests, row.prefix_hit)
            grouped.setdefault(key, []).append(row.actual_reward)
    proposals = []
    for key, rewards in grouped.items():
        candidate = key[0]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / max(1, len(rewards) - 1)
        uncertainty = math.sqrt(variance / len(rewards)) if len(rewards) > 1 else math.inf
        score = mean - uncertainty
        proposals.append(RankProposal(candidate, key[1], key[2], key[3], key[4],
                                      key[5], len(rewards), mean, uncertainty, score))
    return tuple(sorted(proposals, key=lambda row: (-row.score, row.candidate_id,
                                                    row.model_id, row.session_requests,
                                                    row.prefix_hit)))


__all__ = [
    "CANDIDATE_IDS", "IntegrationEvidence", "RankProposal", "RequestContext",
    "RewardObservation", "SelectionContractError", "SelectionDecision",
    "propose_ranks", "select_candidate",
]
