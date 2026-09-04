"""R0 — RL-ready decision logging for Optimization Memory v2.

Every tuner decision is written as an immutable feature record (context,
full candidate set, chosen action, selection rule, propensity) and every
outcome as a matching label record (reward plus censoring status).  Without
propensities no later off-policy evaluation is possible; with them the
offline corpus is a by-product of ordinary measurement.

The module is a data contract and a selection rule.  It executes no model,
probes no hardware, writes no threshold, and grounds no learning claim.  It
deliberately reuses ``RecordKind.SYSTEM`` instead of adding a SQL kind: the
memory schema is verified byte-exactly, so a new ``kind`` value would
invalidate every existing database.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes, sha256_hex
from .candidates import CANDIDATE_IDS, CandidateRegistry
from .fingerprint import ExactFingerprint
from .records import DataPhase, OptimizationRecord, QualityClass, RecordKind


DECISION_SCHEMA = "friday.optimizer.decision.v1"
OUTCOME_SCHEMA = "friday.optimizer.decision-outcome.v1"

#: Rules that may produce a decision.  Stochastic rules must log a seed.
SELECTION_RULES = ("deterministic_order", "epsilon_greedy", "user_forced")
STOCHASTIC_RULES = ("epsilon_greedy",)

#: Censoring status of a reward.  Only ``observed`` carries a number.
CENSORING = (
    "observed",
    "censored_timeout",
    "censored_error",
    "censored_gate_failed",
    "not_run",
)

REWARD_METRICS = ("ratio_median", "ttft_ratio", "decode_ratio")

MAX_CONTEXT_KEYS = 64
MAX_CONTEXT_STRING = 256
MAX_NOTE_LENGTH = 512

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class DecisionError(ValueError):
    """Malformed decision, outcome, or policy input."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise DecisionError(f"{field} must be a bounded safe identifier")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise DecisionError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _member(value: Any, allowed: Sequence[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise DecisionError(f"{field} must be one of {', '.join(allowed)}")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DecisionError(f"{field} must be a finite number")
    return float(value)


def _context_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecisionError("context values must be finite")
        return value
    if isinstance(value, str):
        if len(value) > MAX_CONTEXT_STRING:
            raise DecisionError("context string exceeds the bound")
        return value
    raise DecisionError("context values must be JSON scalars")


def decision_context(fingerprint: ExactFingerprint) -> dict[str, Any]:
    """Flatten an exact fingerprint into a bounded, scalar context vector.

    Features are shape- and parameter-generic on purpose, so one expensive
    measurement can inform many later decisions on the same device.  No
    cross-device transfer is implied or permitted by this projection.
    """

    if not isinstance(fingerprint, ExactFingerprint):
        raise DecisionError("context requires an ExactFingerprint")
    flat: dict[str, Any] = {}
    for section in ("environment", "model", "workload"):
        values = getattr(fingerprint, section).as_dict()
        for key in sorted(values):
            flat[f"{section}.{key}"] = _context_value(values[key])
    if len(flat) > MAX_CONTEXT_KEYS:
        raise DecisionError("context exceeds the bounded feature count")
    return flat


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """A named, hashable selection rule over the closed candidate registry.

    The policy only orders and samples an already eligible set; it can never
    widen the action space.  ``epsilon`` is the exploration share of an
    epsilon-greedy rule and is logged with every decision it produces.
    """

    policy_id: str
    rule: str = "deterministic_order"
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(self, "rule", _member(self.rule, SELECTION_RULES, "rule"))
        epsilon = _finite(self.epsilon, "epsilon")
        if not 0.0 <= epsilon <= 1.0:
            raise DecisionError("epsilon must be within [0, 1]")
        if self.rule not in STOCHASTIC_RULES and epsilon != 0.0:
            raise DecisionError("only stochastic rules may carry a non-zero epsilon")
        if self.rule == "epsilon_greedy" and epsilon <= 0.0:
            raise DecisionError("epsilon_greedy requires a positive epsilon")
        object.__setattr__(self, "epsilon", epsilon)

    def as_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "rule": self.rule, "epsilon": self.epsilon}

    @property
    def policy_hash(self) -> str:
        return sha256_hex(canonical_bytes({"schema": DECISION_SCHEMA, **self.as_dict()}))

    def greedy_action(self, candidates: Sequence[str], hints: Sequence[str] = ()) -> str:
        """Return the deterministic choice: the best hinted eligible action.

        Without a historical hint the greedy action stays ``baseline``.  An
        empty corpus therefore recommends the unchanged reference path, which
        is the only honest default while nothing has been measured.
        """

        available = tuple(candidates)
        if not available:
            raise DecisionError("candidate set must not be empty")
        for hint in hints:
            if hint in available and hint != "baseline":
                return hint
        return "baseline" if "baseline" in available else available[0]

    def distribution(self, candidates: Sequence[str], hints: Sequence[str] = ()) -> dict[str, float]:
        """Return the exact action probabilities this rule would use."""

        available = tuple(candidates)
        if len(set(available)) != len(available):
            raise DecisionError("candidate set contains duplicates")
        greedy = self.greedy_action(available, hints)
        if self.rule not in STOCHASTIC_RULES or len(available) == 1:
            return {action: (1.0 if action == greedy else 0.0) for action in available}
        share = self.epsilon / len(available)
        return {
            action: share + (1.0 - self.epsilon if action == greedy else 0.0)
            for action in available
        }

    def select(
        self,
        candidates: Sequence[str],
        *,
        hints: Sequence[str] = (),
        seed: int | None = None,
    ) -> tuple[str, float]:
        """Return ``(action, propensity)`` for one decision.

        The propensity is the exact probability of the action that was
        actually taken, not an approximation, so later importance-sampling
        estimators are unbiased.
        """

        probabilities = self.distribution(candidates, hints)
        if self.rule not in STOCHASTIC_RULES:
            if seed is not None:
                raise DecisionError("deterministic rules must not carry a seed")
            action = self.greedy_action(candidates, hints)
            return action, probabilities[action]
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise DecisionError("stochastic rules require a bounded non-negative integer seed")
        draw = random.Random(seed).random()
        cumulative = 0.0
        for action in candidates:
            cumulative += probabilities[action]
            if draw < cumulative:
                return action, probabilities[action]
        action = tuple(candidates)[-1]
        return action, probabilities[action]


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """One logged tuner decision, sufficient for off-policy evaluation."""

    decision_id: str
    fingerprint_hash: str
    context: Mapping[str, Any]
    candidate_set: tuple[str, ...]
    chosen: str
    selection_rule: str
    propensity: float
    policy_id: str
    policy_hash: str
    registry_hash: str
    hints: tuple[str, ...] = ()
    seed: int | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "fingerprint_hash", _digest(self.fingerprint_hash, "fingerprint_hash"))
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id"))
        object.__setattr__(self, "policy_hash", _digest(self.policy_hash, "policy_hash"))
        object.__setattr__(self, "registry_hash", _digest(self.registry_hash, "registry_hash"))
        object.__setattr__(self, "selection_rule", _member(self.selection_rule, SELECTION_RULES, "selection_rule"))
        if not isinstance(self.context, Mapping) or isinstance(self.context, (str, bytes)):
            raise DecisionError("context must be a mapping")
        context = {str(key): _context_value(value) for key, value in self.context.items()}
        if len(context) > MAX_CONTEXT_KEYS:
            raise DecisionError("context exceeds the bounded feature count")
        object.__setattr__(self, "context", _freeze(context))
        candidates = tuple(self.candidate_set)
        if not candidates or len(set(candidates)) != len(candidates):
            raise DecisionError("candidate_set must be a non-empty set of unique ids")
        if any(action not in CANDIDATE_IDS for action in candidates):
            raise DecisionError("candidate_set escapes the sealed allowlist")
        object.__setattr__(self, "candidate_set", candidates)
        if self.chosen not in candidates:
            raise DecisionError("chosen action is outside the logged candidate set")
        hints = tuple(self.hints)
        if len(set(hints)) != len(hints) or any(hint not in CANDIDATE_IDS for hint in hints):
            raise DecisionError("hints must be unique ids from the sealed allowlist")
        object.__setattr__(self, "hints", hints)
        propensity = _finite(self.propensity, "propensity")
        if not 0.0 < propensity <= 1.0:
            raise DecisionError("propensity must be within (0, 1]")
        object.__setattr__(self, "propensity", propensity)
        if self.selection_rule in STOCHASTIC_RULES:
            if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
                raise DecisionError("stochastic decisions must log a non-negative integer seed")
        else:
            if self.seed is not None:
                raise DecisionError("deterministic decisions must not log a seed")
            if propensity != 1.0:
                raise DecisionError("deterministic decisions have propensity 1.0")
        if not isinstance(self.created_at, str) or len(self.created_at) > 40:
            raise DecisionError("created_at must be a bounded string")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "decision_id": self.decision_id,
            "fingerprint_hash": self.fingerprint_hash,
            "context": _thaw(self.context),
            "candidate_set": list(self.candidate_set),
            "chosen": self.chosen,
            "selection_rule": self.selection_rule,
            "propensity": self.propensity,
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "registry_hash": self.registry_hash,
            "hints": list(self.hints),
            "seed": self.seed,
        }

    def as_record(self) -> OptimizationRecord:
        """Return the append-only feature record for Optimization Memory v2."""

        return OptimizationRecord(
            record_id=f"decision:{self.decision_id}",
            kind=RecordKind.SYSTEM,
            quality=QualityClass.EXPLORATORY,
            phase=DataPhase.FEATURE,
            payload=self.payload(),
            created_at=self.created_at or None,
        )

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "DecisionEvent":
        if not isinstance(value, Mapping) or value.get("schema") != DECISION_SCHEMA:
            raise DecisionError("payload is not a decision record")
        return cls(
            decision_id=value["decision_id"],
            fingerprint_hash=value["fingerprint_hash"],
            context=value.get("context", {}),
            candidate_set=tuple(value["candidate_set"]),
            chosen=value["chosen"],
            selection_rule=value["selection_rule"],
            propensity=value["propensity"],
            policy_id=value["policy_id"],
            policy_hash=value["policy_hash"],
            registry_hash=value["registry_hash"],
            hints=tuple(value.get("hints", ())),
            seed=value.get("seed"),
        )


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    """The reward of one decision, with its censoring status.

    A censored run is never discarded: a timeout, a compile error, or a
    failed gate is information about the action, and dropping it biases every
    later estimate towards the actions that happened to survive.
    """

    decision_id: str
    censoring: str
    reward: float | None = None
    reward_metric: str = "ratio_median"
    evidence_hash: str | None = None
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "censoring", _member(self.censoring, CENSORING, "censoring"))
        object.__setattr__(self, "reward_metric", _member(self.reward_metric, REWARD_METRICS, "reward_metric"))
        if self.censoring == "observed":
            object.__setattr__(self, "reward", _finite(self.reward, "reward"))
        elif self.reward is not None:
            raise DecisionError("a censored outcome must not carry a reward")
        if self.evidence_hash is not None:
            object.__setattr__(self, "evidence_hash", _digest(self.evidence_hash, "evidence_hash"))
        if not isinstance(self.notes, str) or len(self.notes) > MAX_NOTE_LENGTH:
            raise DecisionError("notes are unbounded")
        if not isinstance(self.created_at, str) or len(self.created_at) > 40:
            raise DecisionError("created_at must be a bounded string")

    @property
    def observed(self) -> bool:
        return self.censoring == "observed"

    def payload(self) -> dict[str, Any]:
        return {
            "schema": OUTCOME_SCHEMA,
            "decision_id": self.decision_id,
            "censoring": self.censoring,
            "reward": self.reward,
            "reward_metric": self.reward_metric,
            "evidence_hash": self.evidence_hash,
            "notes": self.notes,
        }

    def as_record(self) -> OptimizationRecord:
        """Return the append-only label record for Optimization Memory v2."""

        return OptimizationRecord(
            record_id=f"outcome:{self.decision_id}",
            kind=RecordKind.SYSTEM,
            quality=QualityClass.EXPLORATORY,
            phase=DataPhase.LABEL,
            payload=self.payload(),
            created_at=self.created_at or None,
        )

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "OutcomeEvent":
        if not isinstance(value, Mapping) or value.get("schema") != OUTCOME_SCHEMA:
            raise DecisionError("payload is not an outcome record")
        return cls(
            decision_id=value["decision_id"],
            censoring=value["censoring"],
            reward=value.get("reward"),
            reward_metric=value.get("reward_metric", "ratio_median"),
            evidence_hash=value.get("evidence_hash"),
            notes=value.get("notes", ""),
        )


def decide(
    policy: SelectionPolicy,
    fingerprint: ExactFingerprint,
    *,
    registry: CandidateRegistry | None = None,
    qualified: Iterable[str] = (),
    hints: Sequence[str] = (),
    seed: int | None = None,
    decision_id: str | None = None,
    created_at: str = "",
) -> DecisionEvent:
    """Make and log one decision over the eligible, masked candidate set.

    The mask comes from the sealed registry, never from the policy: a policy
    output is untrusted input and cannot widen the action space.
    """

    used = registry if registry is not None else CandidateRegistry()
    context = decision_context(fingerprint)
    candidates = used.ordered_ids(fingerprint, qualified=tuple(qualified), historical_hints=tuple(hints))
    chosen, propensity = policy.select(candidates, hints=hints, seed=seed)
    identifier = decision_id or sha256_hex(
        canonical_bytes(
            {
                "fingerprint": fingerprint.fingerprint_hash,
                "policy": policy.policy_hash,
                "candidates": list(candidates),
                "chosen": chosen,
                "seed": seed,
            }
        )
    )[:32]
    return DecisionEvent(
        decision_id=identifier,
        fingerprint_hash=fingerprint.fingerprint_hash,
        context=context,
        candidate_set=candidates,
        chosen=chosen,
        selection_rule=policy.rule,
        propensity=propensity,
        policy_id=policy.policy_id,
        policy_hash=policy.policy_hash,
        registry_hash=used.registry_hash,
        hints=tuple(hints),
        seed=seed,
        created_at=created_at,
    )


__all__ = [
    "CENSORING",
    "DECISION_SCHEMA",
    "OUTCOME_SCHEMA",
    "REWARD_METRICS",
    "SELECTION_RULES",
    "STOCHASTIC_RULES",
    "DecisionError",
    "DecisionEvent",
    "OutcomeEvent",
    "SelectionPolicy",
    "decide",
    "decision_context",
]
