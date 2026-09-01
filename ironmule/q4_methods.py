"""Deterministic, evidence-bound method replay for the Q4 pilot.

This module is deliberately a small, dependency-free research boundary.  It
does not know how to run a model and it cannot write a profile or choose a
runtime path.  The evaluator owns correctness, resource and rollback gates;
the methods below only rank already-recorded observations.

The public classes are intentionally usable with either the strict Q4 contract
objects or small mapping/dataclass fixtures.  That keeps synthetic replay tests
honest without importing the runtime package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # The package may be loaded under the test harness without __init__.py.
    from .q4_contracts import (
        KNOB_ACTIONS,
        LEGAL_KNOB_ACTIONS,
        ALL_DECLARED_KNOB_ACTIONS,
        KNOB_CANDIDATES,
        KNOB_CANDIDATE_BY_ID,
        SCHEDULE_ACTIONS,
        KNOB_ACTION_BY_ID,
        SCHEDULE_ACTION_BY_ID,
        KNOB_DELTA_IDS,
        Q4State,
        RewardRecord,
        ScheduleAction,
        Stage,
        PlanKind,
        canonical_sha256,
    )
except ImportError:  # pragma: no cover - direct source loading convenience
    from q4_contracts import (  # type: ignore[no-redef]
        KNOB_ACTIONS,
        LEGAL_KNOB_ACTIONS,
        ALL_DECLARED_KNOB_ACTIONS,
        KNOB_CANDIDATES,
        KNOB_CANDIDATE_BY_ID,
        SCHEDULE_ACTIONS,
        KNOB_ACTION_BY_ID,
        SCHEDULE_ACTION_BY_ID,
        KNOB_DELTA_IDS,
        Q4State,
        RewardRecord,
        ScheduleAction,
        Stage,
        PlanKind,
        canonical_sha256,
    )


SEED = "Q4-RL-20260901"
GAMMA = 0.9
RIDGE_ALPHA = 1.0
FQI_ITERATIONS = 20
FQI_TOLERANCE = 1e-9
BEHAVIOUR_LAMBDA = 0.1
MIN_SUPPORT = 3
FOLD_COUNT = 5
WIS_RATIO_CLIP = 10.0
OPE_UNSUPPORTED = "OPE_UNSUPPORTED"
KNOB_CANDIDATE_IDS = tuple(item.candidate_id for item in KNOB_CANDIDATES)


class Method(str, Enum):
    """The seven equal-budget policies named by the preregistration."""

    BASELINE = "BASELINE"
    CURRENT_COORDINATE = "CURRENT_COORDINATE"
    SEEDED_RANDOM = "SEEDED_RANDOM"
    SURROGATE = "SURROGATE"
    BO = "BO"
    CONTEXTUAL_BANDIT = "CONTEXTUAL_BANDIT"
    OFFLINE_RL = "OFFLINE_RL"
    # Internal descriptive aliases; serialized values remain the frozen
    # external names BO and OFFLINE_RL.
    DETERMINISTIC_BO = "BO"
    EB_HCORL = "OFFLINE_RL"


class MethodStatus(str, Enum):
    READY = "READY"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DataInsufficientError(ValueError):
    """Raised when a contract row cannot be joined to a trainable state/reward."""


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _action_id(action: Any) -> str:
    return str(_value(action, "action_id", action if isinstance(action, str) else ""))


def _policy_action_id(row: Any) -> str:
    """Stage-1 policy ID; retain dynamic delta IDs for transition identity."""
    candidate = _value(row, "candidate_id", None)
    if candidate:
        return str(candidate)
    field = _value(row, "changed_field", None)
    target = _value(row, "target_value", None)
    if field is not None and target is not None:
        return canonical_sha256({"schema": "ironmule.q4_knob_candidate.v1", "changed_field": field, "target_value": target})
    return _action_id(_value(row, "action_id", ""))


def _ope_action_id(row: Any) -> str:
    """Use candidate slots for knob OPE; retain strategy IDs unchanged."""
    return _policy_action_id(row) if _stage(row) == "KNOB_DELTA" else _action_id(_value(row, "action_id", ""))


def _legal_remaining_candidates(state: Any, action_ids: Iterable[str], evaluated: Iterable[str] = ()) -> tuple[str, ...]:
    """Return one-field candidate slots still legal from this complete state."""
    current_id = _value(state, "knob_action_id", None)
    current = ALL_DECLARED_KNOB_ACTIONS.get(str(current_id)) if current_id else None
    if current is None:
        return ()
    already = {str(item) for item in evaluated}
    result = []
    for candidate_id in sorted(set(str(item) for item in action_ids)):
        if candidate_id in already:
            continue
        candidate = KNOB_CANDIDATE_BY_ID.get(candidate_id)
        if candidate is not None and getattr(current, candidate.changed_field) != candidate.target_value:
            result.append(candidate_id)
    return tuple(result)


def _group_id(row: Any) -> str:
    return str(
        _value(row, "group_id", None)
        or _value(row, "context_id", None)
        or _value(_value(row, "context", None), "context_id", None)
        or _value(row, "trajectory_id", "")
    )


def _split(row: Any) -> str:
    value = _value(row, "split", "")
    return str(_value(value, "value", value))


def _stage(row: Any) -> str:
    value = _value(row, "stage", "")
    return str(_value(value, "value", value))


def _state(row: Any) -> Any:
    return _value(row, "state", None)


def _features(state: Any) -> tuple[float, ...]:
    if state is None:
        return (1.0,)
    vector = _value(state, "feature_vector", None)
    if callable(vector):
        vector = vector()
    if vector is None:
        vector = _value(state, "features", None)
    if vector is None and isinstance(state, Mapping):
        vector = state.get("feature_vector", state.get("features"))
    if vector is None:
        return (1.0,)
    try:
        values = tuple(float(x) for x in vector)
    except (TypeError, ValueError):
        return (1.0,)
    if not values or not all(math.isfinite(x) for x in values):
        return (1.0,)
    return values


def _in_domain(state: Any) -> bool:
    if state is None:
        return False
    result = _value(state, "in_domain", None)
    if callable(result):
        result = result()
    if result is not None:
        return bool(result)
    if isinstance(state, Mapping):
        model = state.get("model_size")
        buckets = tuple(state.get(name, "") for name in (
            "memory_bucket", "gpu_core_bucket", "prompt_bucket",
            "output_bucket", "concurrency_bucket",
        ))
        return model in {"1B", "4B", "12B"} and all(x in {"small", "medium", "large"} for x in buckets)
    return False


def _is_failed(row: Any) -> bool:
    explicit = _value(row, "failed", None)
    if explicit is not None:
        return bool(explicit)
    if bool(_value(row, "censored", False)):
        return True
    outcome = _value(row, "outcome", None)
    if outcome is not None:
        complete = _value(outcome, "complete_safe", None)
        if complete is not None:
            return not bool(complete)
        status = str(_value(_value(outcome, "status", ""), "value", _value(outcome, "status", "")))
        return status not in {"MEASURED", "COMPLETE", "SAFE"}
    return False


def _reward(row: Any) -> float:
    value = _value(row, "reward", None)
    if value is None:
        value = _value(row, "r", None)
    if value is None:
        # A contract transition may point to an outcome but does not contain a
        # reward.  Missing values remain missing for the caller; zero is only a
        # neutral value for a ranking fixture and is never evidence.
        return 0.0
    return _finite(value)


def _propensity(row: Any) -> float:
    value = _value(row, "propensity", None)
    if value is None:
        value = _value(row, "behaviour_propensity", None)
    result = _finite(value, -1.0)
    return result


def _fold(group: str) -> int:
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % FOLD_COUNT


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def q_lcb(values: Sequence[float], grouped_support: int) -> float:
    """The frozen performance lower confidence bound."""
    if not values:
        return float("-inf")
    numbers = tuple(_finite(value) for value in values)
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    support = max(0, int(grouped_support))
    return mean - math.sqrt(variance) - 1.0 / math.sqrt(support + 1)


def failure_ucb(failures: int, trials: int) -> float:
    """The frozen one-sided failure-risk upper confidence bound."""
    failures = max(0, int(failures))
    trials = max(0, int(trials))
    return min(1.0, (failures + 1.0) / (trials + 2.0) + math.sqrt(math.log(20.0) / (2.0 * max(trials, 1))))


def behaviour_score(performance_lcb: float, propensity: float) -> float:
    """LCB plus the exact frozen behaviour-cloning log prior."""
    if not math.isfinite(performance_lcb):
        return float("-inf")
    return performance_lcb + BEHAVIOUR_LAMBDA * math.log(max(float(propensity), 1e-6))


def grouped_support(rows: Iterable[Any], action_id: str) -> int:
    return len({_group_id(row) for row in rows if _action_id(_value(row, "action_id", "")) == action_id})


def _policy_grouped_support(rows: Iterable[Any], action_id: str) -> int:
    return len({_group_id(row) for row in rows if _policy_action_id(row) == action_id})


def _policy_failure_stats(rows: Iterable[Any], action_id: str) -> tuple[int, int]:
    selected = [row for row in rows if _policy_action_id(row) == action_id]
    return sum(_is_failed(row) for row in selected), len(selected)


@dataclass(frozen=True, slots=True)
class DerivedRewardRecord:
    """Canonical reward join emitted by the offline evidence builder.

    A plain ``{transition_id: 0.4}`` map is intentionally not accepted for
    strict data.  The record binds the reward to the transition, reference
    outcome and both original positive costs before a value head can see it.
    """

    transition_id: str
    reference_outcome_id: str
    objective_class: str
    current_cost: float
    candidate_cost: float
    reward: float
    candidate_outcome_id: str | None = None
    metric: str = "wall_time"
    candidate_p95_full_response_ms: float | None = None
    candidate_physical_tokens_per_second: float | None = None
    p95_inflation: float | None = None
    safety_passed: bool = True
    reward_id: str = ""

    SCHEMA = "ironmule.q4_derived_reward.v1"

    def __post_init__(self) -> None:
        for name in ("transition_id", "reference_outcome_id"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.candidate_outcome_id is not None and not _is_digest(self.candidate_outcome_id):
            raise ValueError("candidate_outcome_id must be a lowercase SHA-256 digest")
        if self.objective_class not in {"LATENCY", "THROUGHPUT"}:
            raise ValueError("objective_class must be LATENCY or THROUGHPUT")
        for name in ("current_cost", "candidate_cost", "reward"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name != "reward" and value <= 0):
                raise ValueError(f"{name} is not a finite positive value")
        expected = math.log(self.current_cost / self.candidate_cost)
        if abs(float(self.reward) - expected) > 1e-9:
            raise ValueError("reward is not the canonical log cost ratio")
        if type(self.safety_passed) is not bool:
            raise ValueError("safety_passed must be boolean")
        if self.reward_id:
            if self.reward_id != canonical_sha256(self._semantic_dict()):
                raise ValueError("reward_id does not match canonical content")
        object.__setattr__(self, "reward_id", canonical_sha256(self._semantic_dict()))

    def _semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA, "transition_id": self.transition_id,
            "reference_outcome_id": self.reference_outcome_id,
            "objective_class": self.objective_class, "current_cost": self.current_cost,
            "candidate_cost": self.candidate_cost, "reward": self.reward, "candidate_outcome_id": self.candidate_outcome_id, "metric": self.metric,
            "candidate_p95_full_response_ms": self.candidate_p95_full_response_ms,
            "candidate_physical_tokens_per_second": self.candidate_physical_tokens_per_second,
            "p95_inflation": self.p95_inflation, "safety_passed": self.safety_passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "reward_id": self.reward_id}

    def valid_for(self, transition: Any) -> bool:
        if not self.safety_passed or self.transition_id != str(_value(transition, "transition_id", "")) or self.reference_outcome_id != str(_value(transition, "reference_outcome_id", "")):
            return False
        if self.candidate_outcome_id is not None and self.candidate_outcome_id != str(_value(transition, "outcome_id", "")):
            return False
        if _stage(transition) != "STRATEGY_SELECT":
            return True
        if self.metric == "contract":
            # Dataset.derive_rewards has already called Outcome.metric_value,
            # including the p95>=20 and throughput p95<=10% gates.
            return True
        if self.objective_class == "LATENCY":
            return self.candidate_p95_full_response_ms is not None and self.candidate_p95_full_response_ms > 0
        return (self.candidate_physical_tokens_per_second is not None
                and self.candidate_physical_tokens_per_second > 0
                and self.p95_inflation is not None and self.p95_inflation <= 0.10)


def action_failure_stats(rows: Iterable[Any], action_id: str) -> tuple[int, int]:
    selected = [row for row in rows if _action_id(_value(row, "action_id", "")) == action_id]
    return sum(_is_failed(row) for row in selected), len(selected)


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    """Minimal reward-bearing view used by offline method replay.

    It is intentionally separate from ``q4_contracts.Transition`` because the
    latter correctly stores evaluator-owned outcome IDs rather than a derived
    policy reward.  ``from_contract`` joins the two only in the caller's
    offline report.
    """

    state: Any
    action_id: str
    reward: float
    next_state: Any = None
    terminal: bool = False
    stage: str = "KNOB_DELTA"
    split: str = "Q4_TRAIN"
    context_id: str = ""
    group_id: str = ""
    propensity: float = 1.0
    behaviour_policy_digest: str = ""
    failed: bool = False
    censored: bool = False
    trajectory_id: str = ""
    step_index: int | None = None
    candidate_id: str = ""
    evaluated_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if self.state is None:
            raise DataInsufficientError("state object or state_lookup entry is required")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if not 0 < float(self.propensity) <= 1:
            raise ValueError("propensity must satisfy 0 < p <= 1")
        if self.step_index is not None and (type(self.step_index) is not int or not 0 <= self.step_index <= 16):
            raise ValueError("step_index must be an integer from 0 through 16")
        if self.group_id == "" and self.context_id:
            object.__setattr__(self, "group_id", self.context_id)

    @property
    def group(self) -> str:
        return self.group_id or self.context_id or self.trajectory_id

    @classmethod
    def from_contract(
        cls,
        transition: Any,
        outcome: Any = None,
        reward: float | None = None,
        *,
        state_lookup: Mapping[str, Any] | Callable[[str], Any] | None = None,
        reward_record: DerivedRewardRecord | RewardRecord | Mapping[str, Any] | None = None,
        synthetic: bool = False,
    ) -> "ReplayTransition":
        if reward_record is not None:
            reward_record = _coerce_reward_record(reward_record)
            if not reward_record.valid_for(transition):
                raise DataInsufficientError("derived reward record is not bound to a safe transition")
            reward = reward_record.reward
        if reward is None:
            reward = _value(transition, "reward", None)
        if reward is None and outcome is not None:
            reward = _value(outcome, "reward", None)
        if reward is None:
            raise DataInsufficientError("contract transition has no joined reward")
        if reward_record is None and not synthetic:
            raise DataInsufficientError("strict contract rows require a canonical derived reward record")
        state = _value(transition, "state", None) or _value(transition, "state_digest", None)
        next_state = _value(transition, "next_state", None) or _value(transition, "next_state_digest", None)
        if isinstance(state, str):
            state = state_lookup(state) if callable(state_lookup) else (state_lookup or {}).get(state) if state_lookup is not None else None
        if isinstance(next_state, str):
            next_digest = next_state
            next_state = state_lookup(next_digest) if callable(state_lookup) else (state_lookup or {}).get(next_digest) if state_lookup is not None else None
        return cls(
            state=state,
            action_id=_action_id(_value(transition, "action_id", "")),
            reward=float(reward),
            next_state=next_state,
            terminal=bool(_value(transition, "terminal", False)),
            stage=str(_value(_value(transition, "stage", ""), "value", _value(transition, "stage", ""))),
            split=str(_value(_value(transition, "split", ""), "value", _value(transition, "split", ""))),
            context_id=str(_value(_value(transition, "context", None), "context_id", _value(transition, "context_id", ""))),
            group_id=str(_value(transition, "group_id", "")),
            propensity=float(_value(transition, "behaviour_propensity", _value(transition, "propensity", 1.0))),
            behaviour_policy_digest=str(_value(transition, "behaviour_policy_digest", "")),
            failed=_is_failed(outcome if outcome is not None else transition),
            trajectory_id=str(_value(transition, "trajectory_id", "")),
            step_index=_value(transition, "step_index", None),
            candidate_id=(lambda value: "" if value is None else str(value))(_value(transition, "candidate_id", "")),
            evaluated_candidate_ids=tuple(_value(transition, "evaluated_candidate_ids", ()) or ()),
        )


def join_contract_rows(
    transitions: Iterable[Any],
    *,
    reward_lookup: Mapping[str, float] | Callable[[str], float] | None = None,
    derived_rewards: Mapping[str, DerivedRewardRecord | RewardRecord | Mapping[str, Any]] | Iterable[DerivedRewardRecord | RewardRecord | Mapping[str, Any]] | None = None,
    reward_records: Mapping[str, DerivedRewardRecord | RewardRecord | Mapping[str, Any]] | Iterable[DerivedRewardRecord | RewardRecord | Mapping[str, Any]] | None = None,
    state_lookup: Mapping[str, Any] | Callable[[str], Any] | None = None,
    synthetic: bool = False,
) -> tuple[ReplayTransition, ...]:
    """Join strict contract IDs to state objects and computed rewards.

    Q4 deliberately keeps rewards out of the evaluator-owned transition.  A
    caller must therefore provide an explicit reward lookup (or a transition
    field in a synthetic fixture); missing joins raise ``DataInsufficientError``
    instead of becoming zero-valued training examples.
    """
    typed_source = derived_rewards if derived_rewards is not None else reward_records
    typed: dict[str, DerivedRewardRecord] = {}
    if typed_source is not None:
        values = typed_source.items() if isinstance(typed_source, Mapping) else ((str(_value(item, "transition_id", "")), item) for item in typed_source)
        for key, value in values:
            record = _coerce_reward_record(value)
            if str(key) != record.transition_id:
                raise DataInsufficientError("derived reward lookup key must equal transition_id")
            typed[record.transition_id] = record
    if reward_lookup is not None and not synthetic:
        raise DataInsufficientError("plain scalar reward_lookup is permitted only for synthetic fixtures")
    result: list[ReplayTransition] = []
    for transition in transitions:
        transition_id = str(_value(transition, "transition_id", ""))
        record = typed.get(transition_id)
        if record is None and typed_source is not None:
            raise DataInsufficientError("missing canonical reward record for transition")
        if record is not None:
            result.append(ReplayTransition.from_contract(transition, state_lookup=state_lookup, reward_record=record))
            continue
        reward = None
        if reward_lookup is not None:
            reward = reward_lookup(transition_id) if callable(reward_lookup) else reward_lookup.get(transition_id)
        result.append(ReplayTransition.from_contract(transition, reward=reward, state_lookup=state_lookup, synthetic=synthetic))
    return tuple(result)


def _coerce_reward_record(value: DerivedRewardRecord | RewardRecord | Mapping[str, Any]) -> DerivedRewardRecord:
    if isinstance(value, DerivedRewardRecord):
        return value
    if isinstance(value, RewardRecord) or (isinstance(value, Mapping) and value.get("schema") == RewardRecord.SCHEMA):
        # The strict Dataset emits this typed, canonical record after joining
        # both evaluator-owned Outcomes.  Preserve its transition/reference
        # binding; metric gates were checked by Dataset.derive_rewards.
        if isinstance(value, Mapping):
            value = RewardRecord.from_dict(value)
        return DerivedRewardRecord(
            transition_id=value.transition_id,
            reference_outcome_id=value.reference_outcome_id,
            objective_class=value.objective_class,
            current_cost=value.reference_cost,
            candidate_cost=value.candidate_cost,
            reward=value.reward,
            candidate_outcome_id=value.candidate_outcome_id,
            metric="contract",
            safety_passed=True,
        )
    if not isinstance(value, Mapping) or value.get("schema") != DerivedRewardRecord.SCHEMA:
        raise DataInsufficientError("reward record is not a typed canonical derived-reward object")
    expected = set(DerivedRewardRecord.__dataclass_fields__) | {"schema"}
    legacy_expected = expected - {"candidate_outcome_id"}
    if set(value) not in {expected, legacy_expected}:
        raise DataInsufficientError("derived reward fields differ")
    return DerivedRewardRecord(
        transition_id=value["transition_id"], reference_outcome_id=value["reference_outcome_id"],
        objective_class=value["objective_class"], current_cost=value["current_cost"],
        candidate_cost=value["candidate_cost"], reward=value["reward"], metric=value["metric"],
        candidate_outcome_id=value.get("candidate_outcome_id"),
        candidate_p95_full_response_ms=value["candidate_p95_full_response_ms"],
        candidate_physical_tokens_per_second=value["candidate_physical_tokens_per_second"],
        p95_inflation=value["p95_inflation"], safety_passed=value["safety_passed"], reward_id=value["reward_id"],
    )


@dataclass(frozen=True, slots=True)
class ActionScore:
    action_id: str
    score: float
    q_lcb: float
    support: int
    failure_ucb: float
    allowed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "score": self.score,
            "q_lcb": self.q_lcb,
            "support": self.support,
            "failure_ucb": self.failure_ucb,
            "allowed": self.allowed,
            "reason": self.reason,
        }


def _action_set(stage: str, plan_kind: str | None = None) -> tuple[str, ...]:
    if stage == "KNOB_DELTA":
        # Policy decisions are the eleven declared candidate specs.  Concrete
        # transitions still carry dynamic KnobDelta IDs separately.
        return tuple(KNOB_CANDIDATE_IDS)
    if stage == "STRATEGY_SELECT" or stage == "REVALIDATE":
        actions = SCHEDULE_ACTIONS
        if plan_kind is not None:
            try:
                actions = ScheduleAction.safe_pool(plan_kind)
            except Exception:
                actions = ()
        return tuple(action.action_id for action in actions if not action.is_risk_probe)
    return ()


def mask_actions(
    state: Any,
    action_ids: Iterable[str],
    rows: Iterable[Any] = (),
    *,
    incumbent_failure: float | None = None,
    minimum_support: int = MIN_SUPPORT,
    risk_probe_ids: Iterable[str] = (),
) -> tuple[ActionScore, ...]:
    """Apply domain, support, risk and risk-probe masks without inventing reward."""
    rows = tuple(rows)
    risk_ids = set(risk_probe_ids)
    result: list[ActionScore] = []
    for action_id in sorted(set(str(item) for item in action_ids)):
        support = grouped_support(rows, action_id)
        failures, trials = action_failure_stats(rows, action_id)
        risk = failure_ucb(failures, trials)
        rewards = [_reward(row) for row in rows if _action_id(_value(row, "action_id", "")) == action_id and not _is_failed(row)]
        lcb = q_lcb(rewards, support)
        propensities = [_propensity(row) for row in rows if _action_id(_value(row, "action_id", "")) == action_id and 0 < _propensity(row) <= 1]
        prior = sum(propensities) / len(propensities) if propensities else 1e-6
        allowed = True
        reason = ""
        if not _in_domain(state):
            allowed, reason = False, "OUT_OF_DOMAIN"
        elif action_id in risk_ids or action_id in {getattr(action, "action_id", "") for action in SCHEDULE_ACTIONS if getattr(action, "is_risk_probe", False)}:
            allowed, reason = False, "RISK_PROBE"
        elif support < minimum_support:
            allowed, reason = False, "INSUFFICIENT_SUPPORT"
        elif incumbent_failure is not None and risk > incumbent_failure:
            allowed, reason = False, "FAILURE_RISK"
        result.append(ActionScore(action_id, behaviour_score(lcb, prior), lcb, support, risk, allowed, reason))
    return tuple(result)


def _solve_ridge(rows: Sequence[tuple[tuple[float, ...], float]], alpha: float = RIDGE_ALPHA) -> tuple[float, ...]:
    """Solve ridge regression using sparse sufficient statistics.

    Q4State has roughly 1,057 columns (and the action block adds a few more),
    but each tabular row has only a handful of non-zero one-hot entries.  The
    old dense ``p×p`` accumulation was therefore quadratic in the state width.
    We retain only active feature indices, form their sufficient statistics,
    solve that compact system deterministically, and expand the zero columns
    back into the exact feature width.
    """
    if not rows:
        return ()
    width = len(rows[0][0])
    sparse_rows: list[tuple[tuple[tuple[int, float], ...], float]] = []
    active: set[int] = set()
    for vector, target in rows:
        if len(vector) != width:
            continue
        entries = tuple((index, float(value)) for index, value in enumerate(vector) if value != 0.0 and math.isfinite(float(value)))
        if not entries:
            continue
        sparse_rows.append((entries, float(target)))
        active.update(index for index, _ in entries)
    if not active:
        return (0.0,) * width
    columns = tuple(sorted(active))
    positions = {column: offset for offset, column in enumerate(columns)}
    compact = len(columns)
    matrix = [[0.0] * (compact + 1) for _ in range(compact)]
    for entries, target in sparse_rows:
        for index, value in entries:
            i = positions[index]
            matrix[i][compact] += value * target
            for other_index, other_value in entries:
                matrix[i][positions[other_index]] += value * other_value
    for index in range(compact):
        matrix[index][index] += alpha
    for col in range(compact):
        pivot = max(range(col, compact), key=lambda row: (abs(matrix[row][col]), -row))
        if abs(matrix[pivot][col]) <= 1e-15:
            continue
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        scale = matrix[col][col]
        for j in range(col, compact + 1):
            matrix[col][j] /= scale
        for row in range(compact):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0.0:
                continue
            for j in range(col, compact + 1):
                matrix[row][j] -= factor * matrix[col][j]
    result = [0.0] * width
    for column, index in positions.items():
        result[column] = matrix[index][compact]
    return tuple(result)


def _dot(weights: Sequence[float], vector: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(weights, vector))


def _action_features(state: Any, action_id: str, action_ids: Sequence[str]) -> tuple[float, ...]:
    base = _features(state)
    return base + tuple(float(action_id == candidate) for candidate in action_ids)


class _FoldEnsemble:
    def __init__(self, action_ids: Sequence[str], action_key: Callable[[Any], str] | None = None) -> None:
        self.action_ids = tuple(sorted(set(action_ids)))
        self.action_key = action_key or (lambda row: row.action_id)
        self.weights: tuple[tuple[float, ...], ...] = ()

    def fit(self, rows: Sequence[ReplayTransition], targets: Mapping[int, Sequence[tuple[ReplayTransition, float]]] | None = None) -> None:
        heads: list[tuple[float, ...]] = []
        for fold in range(FOLD_COUNT):
            if targets is None:
                subset = [row for row in rows if _fold(row.group) != fold]
                pairs = [(_action_features(row.state, self.action_key(row), self.action_ids), row.reward) for row in subset]
            else:
                pairs = [(_action_features(row.state, self.action_key(row), self.action_ids), target) for row, target in targets.get(fold, ())]
            heads.append(_solve_ridge(pairs, RIDGE_ALPHA))
        self.weights = tuple(heads)

    def predict(self, state: Any, action_id: str) -> tuple[float, ...]:
        vector = _action_features(state, action_id, self.action_ids)
        if not self.weights:
            return (0.0,) * FOLD_COUNT
        return tuple(_dot(weights, vector) if weights else 0.0 for weights in self.weights)


class KnobFQI:
    """Five-fold conservative fitted-Q iteration for knob deltas only."""

    def __init__(self, *, gamma: float = GAMMA, iterations: int = FQI_ITERATIONS, tolerance: float = FQI_TOLERANCE) -> None:
        if float(gamma) != GAMMA or type(iterations) is not int or iterations != FQI_ITERATIONS or float(tolerance) != FQI_TOLERANCE:
            raise ValueError("Q4 knob FQI hyperparameters are frozen")
        self.gamma = float(gamma)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.rows: tuple[ReplayTransition, ...] = ()
        self.risk_rows: tuple[ReplayTransition, ...] = ()
        self.action_ids: tuple[str, ...] = ()
        self.ensemble = _FoldEnsemble((), _policy_action_id)
        self.converged_delta: float | None = None

    def fit(self, rows: Iterable[Any]) -> "KnobFQI":
        normalized = tuple(row if isinstance(row, ReplayTransition) else ReplayTransition.from_contract(row) for row in rows)
        self.risk_rows = tuple(row for row in normalized if row.failed or row.censored)
        # A failed/censored row or a nonterminal row without a joined next
        # state cannot be a reward label.  It remains in the evidence corpus,
        # but is excluded from the fitted value head rather than imputed.
        self.rows = tuple(
            row for row in normalized
            if _stage(row) in {"KNOB_DELTA", ""}
            and not row.failed and not row.censored
            and row.state is not None
            and _in_domain(row.state)
            and (row.terminal or row.next_state is not None)
        )
        self.action_ids = tuple(sorted({_policy_action_id(row) for row in self.rows}))
        if not self.action_ids:
            self.action_ids = tuple(sorted(KNOB_CANDIDATE_IDS))
        self.ensemble = _FoldEnsemble(self.action_ids, _policy_action_id)
        previous: _FoldEnsemble | None = None
        last_delta = float("inf")
        for _iteration in range(self.iterations):
            targets: dict[int, list[tuple[ReplayTransition, float]]] = {fold: [] for fold in range(FOLD_COUNT)}
            for fold in range(FOLD_COUNT):
                # Every head is trained on groups outside its held-out fold.
                # The target uses the same head's previous iteration, so no
                # information crosses the deterministic group boundary.
                for row in self.rows:
                    if _fold(row.group) == fold:
                        continue
                    next_values: list[float] = []
                    knob_terminal = row.terminal or (row.step_index is not None and row.step_index >= 10)
                    if not knob_terminal and row.next_state is not None:
                        for action_id in _legal_remaining_candidates(row.next_state, self.action_ids, _value(row, "evaluated_candidate_ids", ())):
                            support = _policy_grouped_support(self.rows, action_id)
                            if support >= MIN_SUPPORT:
                                if previous is not None:
                                    next_values.append(previous.predict(row.next_state, action_id)[fold])
                    target = row.reward + (self.gamma * max(next_values) if next_values else 0.0)
                    targets[fold].append((row, target))
            self.ensemble = _FoldEnsemble(self.action_ids, _policy_action_id)
            self.ensemble.fit(self.rows, targets)
            if previous is not None:
                deltas = []
                for row in self.rows:
                    for action_id in self.action_ids:
                        before = previous.predict(row.state, action_id)
                        after = self.ensemble.predict(row.state, action_id)
                        deltas.extend(abs(after[fold] - before[fold]) for fold in range(FOLD_COUNT))
                last_delta = max(deltas, default=0.0)
                if last_delta <= self.tolerance:
                    break
            previous = self.ensemble
        self.converged_delta = last_delta
        return self

    def values(self, state: Any, action_id: str, support: int | None = None) -> tuple[float, ...]:
        return self.ensemble.predict(state, action_id)

    def score(self, state: Any, action_id: str, rows: Iterable[Any] | None = None, *, incumbent_failure: float | None = None) -> ActionScore:
        rows = tuple(self.rows if rows is None else rows)
        risk_rows = tuple(self.risk_rows) + tuple(row for row in rows if _is_failed(row) and row not in self.risk_rows)
        values = self.values(state, action_id)
        support = _policy_grouped_support(rows, action_id)
        failures, trials = _policy_failure_stats(tuple(rows) + risk_rows, action_id)
        risk = failure_ucb(failures, trials)
        lcb = q_lcb(values, support)
        propensities = [_propensity(row) for row in rows if _policy_action_id(row) == action_id and 0 < _propensity(row) <= 1]
        prior = sum(propensities) / len(propensities) if propensities else 1e-6
        allowed = support >= MIN_SUPPORT and _in_domain(state) and (incumbent_failure is None or risk <= incumbent_failure)
        current_knob = _value(state, "knob_action_id", None)
        if current_knob is not None and action_id not in _legal_remaining_candidates(state, (action_id,), _value(state, "evaluated_candidate_ids", ())):
            allowed, reason = False, "ILLEGAL_OR_REPEATED_CANDIDATE"
        else:
            reason = "" if allowed else ("OUT_OF_DOMAIN" if not _in_domain(state) else "INSUFFICIENT_SUPPORT" if support < MIN_SUPPORT else "FAILURE_RISK")
        return ActionScore(action_id, behaviour_score(lcb, prior), lcb, support, risk, allowed, reason)

    def select(self, state: Any, action_ids: Iterable[str], rows: Iterable[Any] | None = None, *, incumbent_failure: float | None = None) -> ActionScore:
        candidates = tuple(action_ids)
        if _value(state, "knob_action_id", None) is not None:
            candidates = _legal_remaining_candidates(state, candidates, _value(state, "evaluated_candidate_ids", ()))
        scores = [self.score(state, action_id, rows, incumbent_failure=incumbent_failure) for action_id in candidates]
        allowed = [item for item in scores if item.allowed]
        return max(allowed or scores, key=lambda item: (item.score, item.action_id)) if scores else ActionScore("", float("-inf"), float("-inf"), 0, 1.0, False, "NO_ACTIONS")


class StrategyImmediateRidge:
    """Five-fold contextual ridge head for independent strategy candidates."""

    def __init__(self, *, alpha: float = RIDGE_ALPHA) -> None:
        if float(alpha) != RIDGE_ALPHA:
            raise ValueError("Q4 strategy ridge alpha is frozen at 1")
        self.alpha = float(alpha)
        self.rows: tuple[ReplayTransition, ...] = ()
        self.risk_rows: tuple[ReplayTransition, ...] = ()
        self.action_ids: tuple[str, ...] = ()
        self.ensemble = _FoldEnsemble(())

    def fit(self, rows: Iterable[Any]) -> "StrategyImmediateRidge":
        normalized = tuple(row if isinstance(row, ReplayTransition) else ReplayTransition.from_contract(row) for row in rows)
        self.risk_rows = tuple(row for row in normalized if row.failed or row.censored)
        self.rows = tuple(row for row in normalized if _stage(row) == "STRATEGY_SELECT" and not row.failed and not row.censored and row.state is not None and _in_domain(row.state))
        self.action_ids = tuple(sorted({_action_id(row.action_id) for row in self.rows}))
        if not self.action_ids:
            self.action_ids = tuple(sorted(action.action_id for action in SCHEDULE_ACTIONS if not action.is_risk_probe))
        self.ensemble = _FoldEnsemble(self.action_ids)
        self.ensemble.fit(self.rows)
        # The generic ensemble uses alpha=1 as required by Q4.  Keep the
        # constructor alpha visible in the contract and validate it strictly.
        return self

    def values(self, state: Any, action_id: str) -> tuple[float, ...]:
        return self.ensemble.predict(state, action_id)

    def score(self, state: Any, action_id: str, rows: Iterable[Any] | None = None, *, incumbent_failure: float | None = None) -> ActionScore:
        rows = tuple(self.rows if rows is None else rows)
        risk_rows = tuple(self.risk_rows) + tuple(row for row in rows if _is_failed(row) and row not in self.risk_rows)
        values = self.values(state, action_id)
        support = grouped_support(rows, action_id)
        failures, trials = action_failure_stats(tuple(rows) + risk_rows, action_id)
        risk = failure_ucb(failures, trials)
        lcb = q_lcb(values, support)
        propensities = [_propensity(row) for row in rows if _action_id(_value(row, "action_id", "")) == action_id and 0 < _propensity(row) <= 1]
        prior = sum(propensities) / len(propensities) if propensities else 1e-6
        allowed = support >= MIN_SUPPORT and _in_domain(state) and (incumbent_failure is None or risk <= incumbent_failure)
        reason = "" if allowed else ("OUT_OF_DOMAIN" if not _in_domain(state) else "INSUFFICIENT_SUPPORT" if support < MIN_SUPPORT else "FAILURE_RISK")
        return ActionScore(action_id, behaviour_score(lcb, prior), lcb, support, risk, allowed, reason)

    def select(self, state: Any, action_ids: Iterable[str], rows: Iterable[Any] | None = None, *, incumbent_failure: float | None = None) -> ActionScore:
        scores = [self.score(state, action_id, rows, incumbent_failure=incumbent_failure) for action_id in action_ids]
        allowed = [item for item in scores if item.allowed]
        return max(allowed or scores, key=lambda item: (item.score, item.action_id)) if scores else ActionScore("", float("-inf"), float("-inf"), 0, 1.0, False, "NO_ACTIONS")


class EBHCORL:
    """Evidence-Bound Hierarchical Conservative Offline RL policy."""

    def __init__(self) -> None:
        self.knob_head = KnobFQI()
        self.strategy_head = StrategyImmediateRidge()
        self.rows: tuple[ReplayTransition, ...] = ()

    def fit(self, rows: Iterable[Any]) -> "EBHCORL":
        self.rows = tuple(row if isinstance(row, ReplayTransition) else ReplayTransition.from_contract(row) for row in rows)
        self.knob_head.fit(row for row in self.rows if row.stage in {"KNOB_DELTA", ""})
        self.strategy_head.fit(row for row in self.rows if row.stage == "STRATEGY_SELECT")
        return self

    def select_knob(self, state: Any, action_ids: Iterable[str] = KNOB_CANDIDATE_IDS, *, incumbent_failure: float | None = None) -> ActionScore:
        return self.knob_head.select(state, action_ids, self.knob_head.rows, incumbent_failure=incumbent_failure)

    def select_strategy(self, state: Any, action_ids: Iterable[str], *, incumbent_failure: float | None = None) -> ActionScore:
        return self.strategy_head.select(state, action_ids, self.strategy_head.rows, incumbent_failure=incumbent_failure)


def _seeded_order(items: Iterable[str], seed: str) -> tuple[str, ...]:
    return tuple(sorted(set(items), key=lambda item: hashlib.sha256(f"{seed}/{item}".encode("utf-8")).hexdigest()))


def _normal_method(method: Method | str) -> Method:
    raw = str(getattr(method, "value", method))
    if raw in {"BO", "DETERMINISTIC_BO"}:
        return Method.BO
    if raw in {"OFFLINE_RL", "EB_HCORL"}:
        return Method.OFFLINE_RL
    return Method(raw)


def normalize_method(method: Method | str) -> Method:
    """Normalize prose aliases while retaining exactly seven enum values."""
    return _normal_method(method)


def deterministic_policy_order(method: Method | str, *, seed: str = SEED, plan_kind: str | None = None) -> tuple[str, ...]:
    """Return a byte-stable full-catalogue order for simple replay policies."""
    method = _normal_method(method)
    knob = tuple(sorted(KNOB_CANDIDATE_IDS))
    strategy = _action_set("STRATEGY_SELECT", plan_kind)
    if method is Method.BASELINE:
        return knob[:11] + strategy[:5]
    if method is Method.CURRENT_COORDINATE:
        return tuple(sorted(knob))[:11] + tuple(sorted(strategy))[:5]
    if method is Method.SEEDED_RANDOM:
        return _seeded_order(knob, f"{seed}/knob")[:11] + _seeded_order(strategy, f"{seed}/strategy")[:5]
    return tuple(sorted(knob))[:11] + tuple(sorted(strategy))[:5]


def _mean_reward(rows: Sequence[Any], action_id: str) -> float:
    selected = [_reward(row) for row in rows if _policy_action_id(row) == action_id and not _is_failed(row)]
    return sum(selected) / len(selected) if selected else 0.0


def _std_reward(rows: Sequence[Any], action_id: str) -> float:
    selected = [_reward(row) for row in rows if _policy_action_id(row) == action_id and not _is_failed(row)]
    if not selected:
        return 1.0
    mean = sum(selected) / len(selected)
    return math.sqrt(sum((item - mean) ** 2 for item in selected) / len(selected))


def _rank_catalogue(method: Method, action_ids: Sequence[str], rows: Sequence[Any], seed: str) -> tuple[str, ...]:
    """Rank finite catalogues with deterministic, non-learning comparators.

    These are replay policies, not claims that a summary row is Q4 evidence.
    They only use rows explicitly supplied by the caller and preserve a
    lexicographic tie-break after the method score.
    """
    unique = tuple(sorted(set(action_ids)))
    if method is Method.SEEDED_RANDOM:
        return _seeded_order(unique, seed)
    if method in {Method.BASELINE, Method.CURRENT_COORDINATE}:
        return unique
    if method is Method.SURROGATE:
        return tuple(sorted(unique, key=lambda action: (-_mean_reward(rows, action) + _std_reward(rows, action), action)))
    if method is Method.DETERMINISTIC_BO:
        # Finite-catalogue optimistic surrogate: uncertainty is useful only to
        # break ties in favour of unexplored actions, never to bypass masks.
        return tuple(sorted(unique, key=lambda action: (-(_mean_reward(rows, action) + _std_reward(rows, action)), action)))
    if method is Method.CONTEXTUAL_BANDIT:
        # The contextual ridge head is the RL module's immediate component;
        # this comparator uses grouped action means and requires no hidden
        # extrapolation when a context is absent.
        return tuple(sorted(unique, key=lambda action: (-_mean_reward(rows, action), action)))
    return unique


def rank_catalogue(method: Method | str, action_ids: Sequence[str], rows: Iterable[Any] = (), *, seed: str = SEED) -> tuple[str, ...]:
    """Public deterministic ranking helper for equal-budget replay."""
    return _rank_catalogue(_normal_method(method), tuple(action_ids), tuple(rows), seed)


@dataclass(frozen=True, slots=True)
class OPEEstimate:
    status: str
    estimate: float | None
    trajectory_count: int
    support: Mapping[str, int] = field(default_factory=dict)
    effective_sample_size: float | None = None
    reason: str = ""
    estimates_by_stage: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "estimate": self.estimate,
            "trajectory_count": self.trajectory_count,
            "support": dict(sorted(self.support.items())),
            "effective_sample_size": self.effective_sample_size,
            "reason": self.reason,
            "estimates_by_stage": dict(sorted(self.estimates_by_stage.items())),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _trajectory_rows(trajectory: Any) -> tuple[Any, ...]:
    rows = _value(trajectory, "transitions", None)
    if rows is None:
        rows = _value(trajectory, "rows", None)
    if rows is None and isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
        rows = trajectory
    return tuple(rows or ())


def _model_state_key(row: Any) -> tuple[Any, ...]:
    state = _state(row)
    features = _features(state)
    return (_stage(row), _value(row, "step_index", None), features)


def _target_action(target_policy: Mapping[Any, str] | Callable[[Any], str], row: Any) -> str | None:
    """Resolve a target action from state/step identity, never observed action."""
    if callable(target_policy):
        value = target_policy(row)
        return None if value is None else str(value)
    state = _state(row)
    state_digest = str(_value(state, "state_digest", _value(row, "state_digest", "")))
    step = _value(row, "step_index", None)
    context = _group_id(row)
    keys = ((state_digest, step), (context, step), state_digest)
    for key in keys:
        if key in target_policy:
            return str(target_policy[key])
    return None


def _ope_inputs(trajectories: Iterable[Any], target_policy: Mapping[Any, str] | Callable[[Any], str] | None) -> tuple[tuple[tuple[Any, ...], ...], dict[str, int], str | None]:
    groups: list[tuple[Any, ...]] = []
    support: dict[str, set[str]] = {}
    if target_policy is None:
        return (), {}, "target policy is missing"
    for trajectory in trajectories:
        rows = _trajectory_rows(trajectory)
        if not rows:
            return (), {}, "empty trajectory"
        if len(rows) != 17:
            return (), {}, "OPE requires complete horizon-17 trajectories"
        splits = {_split(row) for row in rows}
        if not splits.issubset({"Q4_TRAIN", "Q4_VALIDATION"}):
            return (), {}, "OPE is restricted to train/validation rows"
        if len(splits) != 1:
            return (), {}, "a trajectory cannot cross Q4 splits"
        steps = [_value(row, "step_index", None) for row in rows]
        if any(step is None for step in steps) or sorted(steps) != list(range(17)):
            return (), {}, "trajectory step markers do not prove H17"
        expected_stages = (("KNOB_DELTA",) * 11) + (("STRATEGY_SELECT",) * 5) + (("REVALIDATE",) * 1)
        if tuple(_stage(row) for row in sorted(rows, key=lambda item: _value(item, "step_index", -1))) != expected_stages:
            return (), {}, "trajectory stage sequence does not prove H17"
        trajectory_ids = {str(_value(row, "trajectory_id", "")) for row in rows}
        if len(trajectory_ids) != 1 or not _is_digest(next(iter(trajectory_ids))):
            return (), {}, "trajectory identity is missing or not a lowercase digest"
        contexts = {_value(row, "context_id", None) or _value(row, "group_id", None) for row in rows}
        if len(contexts) != 1 or not _is_digest(next(iter(contexts))):
            return (), {}, "context identity is missing or not a lowercase digest"
        groups_in_trajectory = {_group_id(row) for row in rows}
        if len(groups_in_trajectory) != 1:
            return (), {}, "group identity is not stable across the trajectory"
        if any(not _is_digest(_group_id(row)) for row in rows):
            return (), {}, "context/group identity is not a lowercase digest"
        if any(not _is_digest(_value(row, "behaviour_policy_digest", "")) for row in rows):
            return (), {}, "behaviour policy digest is missing or non-canonical"
        if any(_value(row, "terminal", False) for row in rows[:-1]) or not bool(_value(rows[-1], "terminal", False)):
            return (), {}, "trajectory terminal marker is incomplete"
        groups.append(rows)
        for row in rows:
            observed = _ope_action_id(row)
            p = _propensity(row)
            desired = _target_action(target_policy, row)
            if not observed or desired is None or not desired or not 0 < p <= 1:
                return (), {}, "missing state/step target or invalid propensity"
            support.setdefault(desired, set()).add(_group_id(row))
    counts = {action: len(group_set) for action, group_set in support.items()}
    if any(count < MIN_SUPPORT for count in counts.values()):
        return (), counts, "minimum grouped support is 3 for every target action"
    if len({_fold(_group_id(row)) for rows in groups for row in rows}) < FOLD_COUNT:
        return (), counts, "one or more grouped OPE folds are empty"
    # A p=1 trajectory is the deterministic coordinate/holdout behaviour
    # policy.  It has no counterfactual overlap, even if a malformed fixture
    # happens to mention multiple actions.
    if all(_propensity(row) == 1.0 for rows in (row for group in groups for row in group)):
        return (), counts, "deterministic propensity-1 behaviour is OPE unsupported"
    return tuple(groups), counts, None


def weighted_importance_sampling(trajectories: Iterable[Any], target_policy: Mapping[Any, str] | Callable[[Any], str]) -> OPEEstimate:
    """Return grouped WIS with per-decision ratios clipped at exactly 10."""
    groups, counts, reason = _ope_inputs(trajectories, target_policy)
    if reason:
        return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, None, reason)
    weighted_rewards: list[tuple[str, float, dict[str, float]]] = []
    for rows in groups:
        weight = 1.0
        stage_rewards: dict[str, float] = {}
        for row in rows:
            desired = _target_action(target_policy, row)
            observed = _ope_action_id(row)
            if desired != observed:
                weight = 0.0
                continue
            weight *= min(WIS_RATIO_CLIP, 1.0 / _propensity(row))
            stage = _stage(row)
            stage_rewards[stage] = stage_rewards.get(stage, 0.0) + _reward(row)
        weighted_rewards.append((_group_id(rows[0]), weight, stage_rewards))
    stage_context_values: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for context, weight, rewards in weighted_rewards:
        for stage, reward in rewards.items():
            stage_context_values.setdefault(stage, {}).setdefault(context, []).append((weight, reward))
    per_stage: dict[str, float] = {}
    all_weights: list[float] = []
    for stage, contexts in stage_context_values.items():
        per_context: list[float] = []
        for values in contexts.values():
            denominator = sum(weight for weight, _ in values)
            if denominator <= 0:
                continue
            per_context.append(sum(weight * reward for weight, reward in values) / denominator)
            all_weights.extend(weight for weight, _ in values)
        if per_context:
            per_stage[stage] = sum(per_context) / len(per_context)
    if not per_stage:
        return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, 0.0, "target policy has no overlap")
    # The two hierarchy heads remain a vector.  A single-stage fixture may
    # expose a scalar for convenience, but Q4 never sums knob and strategy
    # rewards into one policy score.
    estimate = next(iter(per_stage.values())) if len(per_stage) == 1 else None
    denominator = sum(all_weights)
    ess = denominator * denominator / max(1e-12, sum(weight * weight for weight in all_weights))
    return OPEEstimate("SUPPORTED", estimate, len(groups), counts, ess, "", per_stage)


def grouped_doubly_robust(trajectories: Iterable[Any], target_policy: Mapping[Any, str] | Callable[[Any], str]) -> OPEEstimate:
    """Five-fold grouped DR diagnostic with fold-local action means.

    This is deliberately conservative: missing groups, invalid propensities or
    no overlap return ``OPE_UNSUPPORTED`` rather than a guessed estimate.
    """
    groups, counts, reason = _ope_inputs(trajectories, target_policy)
    if reason:
        return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, None, reason)
    rows = [row for group in groups for row in group]
    fold_values: dict[str, list[float]] = {}
    for fold in range(FOLD_COUNT):
        train = [row for row in rows if _fold(_group_id(row)) != fold]
        test_groups = [group for group in groups if _fold(_group_id(group[0])) == fold]
        if not train or not test_groups:
            return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, None, "empty grouped DR fold")
        means: dict[tuple[Any, ...], float] = {}
        for action in {_ope_action_id(row) for row in train}:
            for state_key in {_model_state_key(row) for row in train if _ope_action_id(row) == action}:
                selected = [_reward(row) for row in train if _ope_action_id(row) == action and _model_state_key(row) == state_key]
                if selected:
                    means[(*state_key, action)] = sum(selected) / len(selected)
        trajectory_values: dict[str, list[float]] = {}
        for group in test_groups:
            totals: dict[str, float] = {}
            for row in group:
                observed = _ope_action_id(row)
                desired = _target_action(target_policy, row)
                desired_key = (*_model_state_key(row), desired)
                if desired_key not in means:
                    return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, None, "desired action absent from fold-local DR model")
                q = means[desired_key]
                correction = 0.0
                if desired == observed:
                    observed_key = (*_model_state_key(row), observed)
                    if observed_key not in means:
                        return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, None, "observed action absent from fold-local DR model")
                    correction = min(WIS_RATIO_CLIP, 1.0 / _propensity(row)) * (_reward(row) - means[observed_key])
                stage = _stage(row)
                totals[stage] = totals.get(stage, 0.0) + q + correction
            for stage, total in totals.items():
                trajectory_values.setdefault(stage, []).append(total)
        for stage, values in trajectory_values.items():
            fold_values.setdefault(stage, []).append(sum(values) / len(values))
    per_stage = {stage: sum(values) / len(values) for stage, values in fold_values.items() if values}
    if not per_stage:
        return OPEEstimate(OPE_UNSUPPORTED, None, len(groups), counts, None, "empty grouped DR estimate")
    estimate = next(iter(per_stage.values())) if len(per_stage) == 1 else None
    return OPEEstimate("SUPPORTED", estimate, len(groups), counts, None, "", per_stage)


def ope_diagnostics(trajectories: Iterable[Any], target_policy: Mapping[str, str] | Callable[[Any], str]) -> dict[str, Any]:
    wis = weighted_importance_sampling(trajectories, target_policy)
    dr = grouped_doubly_robust(trajectories, target_policy)
    return {"wis": wis.to_dict(), "dr": dr.to_dict(), "status": "SUPPORTED" if wis.status == dr.status == "SUPPORTED" else OPE_UNSUPPORTED}


__all__ = [
    "SEED", "GAMMA", "RIDGE_ALPHA", "FQI_ITERATIONS", "FQI_TOLERANCE", "BEHAVIOUR_LAMBDA",
    "MIN_SUPPORT", "FOLD_COUNT", "WIS_RATIO_CLIP", "OPE_UNSUPPORTED", "KNOB_CANDIDATE_IDS", "Method", "MethodStatus",
    "DataInsufficientError",
    "ReplayTransition", "ActionScore", "KnobFQI", "StrategyImmediateRidge", "EBHCORL",
    "q_lcb", "failure_ucb", "behaviour_score", "grouped_support", "action_failure_stats",
    "mask_actions", "deterministic_policy_order", "OPEEstimate", "weighted_importance_sampling",
    "grouped_doubly_robust", "ope_diagnostics", "rank_catalogue", "normalize_method", "join_contract_rows",
]
