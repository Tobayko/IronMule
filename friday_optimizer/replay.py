"""R1 — offline replay environment and off-policy evaluation.

The environment is a deterministic view over logged decisions in Optimization
Memory v2.  It runs no model, touches no hardware, and adds no reward that was
not measured under a gate: a counterfactual action simply has no reward, and a
censored run keeps its own conservative terminal value instead of being
dropped.

Reward convention: the stored metric is a ratio ``candidate / baseline`` where
values below ``1`` are faster.  ``default_reward`` therefore returns
``1 - ratio``, the relative gain, so higher is better for every estimator in
this module.
"""

from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canonical import loads_strict
from .decisions import (
    DECISION_SCHEMA,
    OUTCOME_SCHEMA,
    DecisionEvent,
    OutcomeEvent,
    SelectionPolicy,
)
from .records import DataPhase, RecordKind


REPLAY_SCHEMA = "friday.optimizer.replay.v1"

#: Minimum effective sample size before an estimate may be read as a result.
#: Below it every estimator reports ``insufficient_data`` and nothing else.
DEFAULT_MIN_SAMPLES = 30

MAX_STEPS = 100_000
ESTIMATORS = ("ips", "snips", "doubly_robust", "replayer")
STATUSES = ("ok", "insufficient_data", "no_overlap", "no_labels")


class ReplayError(ValueError):
    """Malformed replay input or an action outside the logged mask."""


def default_reward(outcome: OutcomeEvent) -> float:
    """Return the relative gain of an observed ratio (higher is better)."""

    if not outcome.observed or outcome.reward is None:
        raise ReplayError("only an observed outcome carries a reward")
    return 1.0 - outcome.reward


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One logged decision with its label, if the label already exists."""

    decision: DecisionEvent
    outcome: OutcomeEvent | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, DecisionEvent):
            raise ReplayError("step requires a DecisionEvent")
        if self.outcome is not None:
            if not isinstance(self.outcome, OutcomeEvent):
                raise ReplayError("step outcome must be an OutcomeEvent")
            if self.outcome.decision_id != self.decision.decision_id:
                raise ReplayError("outcome does not belong to this decision")

    @property
    def labelled(self) -> bool:
        return self.outcome is not None

    @property
    def observed(self) -> bool:
        return self.outcome is not None and self.outcome.observed

    @property
    def action_mask(self) -> tuple[str, ...]:
        return self.decision.candidate_set

    @property
    def context(self) -> Mapping[str, Any]:
        return self.decision.context


def _payloads(source: Any) -> list[Mapping[str, Any]]:
    if hasattr(source, "list"):
        rows: list[sqlite3.Row] = []
        for phase in (DataPhase.FEATURE, DataPhase.LABEL):
            offset = 0
            while True:
                page = source.list(kind=RecordKind.SYSTEM, phase=phase, limit=1_000, offset=offset)
                rows.extend(page)
                if len(page) < 1_000 or len(rows) > MAX_STEPS * 2:
                    break
                offset += len(page)
        payloads = []
        for row in rows:
            try:
                payloads.append(loads_strict(bytes(row["payload"])))
            except (ValueError, TypeError):
                continue
        return payloads
    return [value for value in source if isinstance(value, Mapping)]


def load_steps(source: Any) -> tuple[ReplayStep, ...]:
    """Pair logged decisions with their outcomes, in stable memory order.

    ``source`` is an Optimization Memory (or read-only view), or any iterable
    of record payloads.  Unknown payloads are ignored; an outcome without a
    decision is an integrity error, not a silent drop.
    """

    decisions: dict[str, DecisionEvent] = {}
    order: list[str] = []
    outcomes: dict[str, OutcomeEvent] = {}
    for payload in _payloads(source):
        schema = payload.get("schema")
        if schema == DECISION_SCHEMA:
            event = DecisionEvent.from_payload(payload)
            if event.decision_id in decisions:
                raise ReplayError(f"duplicate decision {event.decision_id!r}")
            decisions[event.decision_id] = event
            order.append(event.decision_id)
        elif schema == OUTCOME_SCHEMA:
            outcome = OutcomeEvent.from_payload(payload)
            if outcome.decision_id in outcomes:
                raise ReplayError(f"duplicate outcome for {outcome.decision_id!r}")
            outcomes[outcome.decision_id] = outcome
    if len(order) > MAX_STEPS:
        raise ReplayError("replay exceeds the bounded step count")
    unknown = set(outcomes) - set(decisions)
    if unknown:
        raise ReplayError(f"{len(unknown)} outcome(s) without a logged decision")
    return tuple(ReplayStep(decisions[key], outcomes.get(key)) for key in order)


@dataclass(frozen=True, slots=True)
class Observation:
    """What a policy may see before acting: context plus the legal actions."""

    context: Mapping[str, Any]
    action_mask: tuple[str, ...]
    index: int


@dataclass(frozen=True, slots=True)
class Transition:
    """The result of one replayed action.

    ``matched`` is false when the evaluated policy chose a different action
    than the logged one.  The reward is then ``None``: the counterfactual was
    never measured and must not be imputed by the environment.
    """

    reward: float | None
    matched: bool
    censoring: str
    done: bool
    logged_action: str


class ReplayEnv:
    """Deterministic, read-only bandit replay over logged decisions.

    One episode is one decision: there is no state transition to learn, which
    is exactly why this stays a contextual bandit and not an MDP.
    """

    __slots__ = ("_steps", "_index", "_reward_fn", "_censored_reward", "_skip_unlabelled")

    def __init__(
        self,
        steps: Sequence[ReplayStep] | Iterable[ReplayStep],
        *,
        reward_fn: Callable[[OutcomeEvent], float] = default_reward,
        censored_reward: float = 0.0,
        skip_unlabelled: bool = True,
    ) -> None:
        values = tuple(steps)
        if any(not isinstance(step, ReplayStep) for step in values):
            raise ReplayError("replay steps must be ReplayStep instances")
        if len(values) > MAX_STEPS:
            raise ReplayError("replay exceeds the bounded step count")
        if not callable(reward_fn):
            raise ReplayError("reward_fn must be callable")
        if isinstance(censored_reward, bool) or not isinstance(censored_reward, (int, float)) or not math.isfinite(censored_reward):
            raise ReplayError("censored_reward must be a finite number")
        if censored_reward > 0.0:
            raise ReplayError("a censored run must never earn a positive reward")
        if skip_unlabelled:
            values = tuple(step for step in values if step.labelled)
        self._steps = values
        self._reward_fn = reward_fn
        self._censored_reward = float(censored_reward)
        self._skip_unlabelled = bool(skip_unlabelled)
        self._index = 0

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> tuple[ReplayStep, ...]:
        return self._steps

    @property
    def censored_reward(self) -> float:
        return self._censored_reward

    def reward_of(self, step: ReplayStep) -> float | None:
        """Return the logged reward of *step*, or ``None`` when unlabelled."""

        if step.outcome is None:
            return None
        if step.outcome.observed:
            return float(self._reward_fn(step.outcome))
        return self._censored_reward

    def reset(self) -> Observation | None:
        self._index = 0
        return self.observe()

    def observe(self) -> Observation | None:
        if self._index >= len(self._steps):
            return None
        step = self._steps[self._index]
        return Observation(step.context, step.action_mask, self._index)

    def action_mask(self) -> tuple[str, ...]:
        observation = self.observe()
        return () if observation is None else observation.action_mask

    def step(self, action: str) -> Transition:
        """Replay one decision under *action*, enforcing the logged mask."""

        if self._index >= len(self._steps):
            raise ReplayError("replay is exhausted; call reset() first")
        current = self._steps[self._index]
        if not isinstance(action, str) or action not in current.action_mask:
            raise ReplayError(f"action {action!r} is masked for this decision")
        self._index += 1
        matched = action == current.decision.chosen
        reward = self.reward_of(current) if matched else None
        censoring = current.outcome.censoring if current.outcome is not None else "not_run"
        return Transition(
            reward=reward,
            matched=matched,
            censoring=censoring,
            done=self._index >= len(self._steps),
            logged_action=current.decision.chosen,
        )


@dataclass(frozen=True, slots=True)
class OffPolicyEstimate:
    """One estimator result with its honest status and confidence interval."""

    estimator: str
    value: float | None
    samples: int
    effective_samples: float
    status: str
    ci_low: float | None = None
    ci_high: float | None = None
    min_samples: int = DEFAULT_MIN_SAMPLES
    censored_samples: int = 0

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ReplayError("unknown estimator")
        if self.status not in STATUSES:
            raise ReplayError("unknown estimate status")

    @property
    def conclusive(self) -> bool:
        """Whether this estimate may be read as a result at all."""

        return self.status == "ok" and self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAY_SCHEMA,
            "estimator": self.estimator,
            "value": self.value,
            "samples": self.samples,
            "effective_samples": self.effective_samples,
            "status": self.status,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "min_samples": self.min_samples,
            "censored_samples": self.censored_samples,
            "conclusive": self.conclusive,
        }


def _target_probability(policy: SelectionPolicy, step: ReplayStep) -> float:
    distribution = policy.distribution(step.decision.candidate_set, step.decision.hints)
    return float(distribution[step.decision.chosen])


def _weights(env: ReplayEnv, policy: SelectionPolicy) -> list[tuple[float, float, ReplayStep]]:
    rows: list[tuple[float, float, ReplayStep]] = []
    for step in env.steps:
        reward = env.reward_of(step)
        if reward is None:
            continue
        weight = _target_probability(policy, step) / step.decision.propensity
        rows.append((weight, reward, step))
    return rows


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish effective sample size; ``0`` when no weight has any mass."""

    total = math.fsum(weights)
    squares = math.fsum(weight * weight for weight in weights)
    if squares <= 0.0:
        return 0.0
    return (total * total) / squares


def _interval(values: Sequence[float], *, seed: int, resamples: int) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(resamples):
        means.append(math.fsum(values[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    low = means[max(0, int(0.025 * resamples) - 1)]
    high = means[min(resamples - 1, int(0.975 * resamples))]
    return (low, high)


def _estimate(
    name: str,
    contributions: Sequence[float],
    weights: Sequence[float],
    *,
    censored: int,
    min_samples: int,
    seed: int,
    resamples: int,
    normaliser: float | None = None,
) -> OffPolicyEstimate:
    samples = len(contributions)
    ess = effective_sample_size(weights)
    if samples == 0:
        return OffPolicyEstimate(name, None, 0, 0.0, "no_labels", min_samples=min_samples, censored_samples=censored)
    if ess <= 0.0:
        return OffPolicyEstimate(name, None, samples, 0.0, "no_overlap", min_samples=min_samples, censored_samples=censored)
    denominator = samples if normaliser is None else normaliser
    if denominator <= 0.0:
        return OffPolicyEstimate(name, None, samples, ess, "no_overlap", min_samples=min_samples, censored_samples=censored)
    value = math.fsum(contributions) / denominator
    status = "ok" if ess >= min_samples else "insufficient_data"
    low, high = _interval(contributions, seed=seed, resamples=resamples) if normaliser is None else (None, None)
    return OffPolicyEstimate(name, value, samples, ess, status, low, high, min_samples, censored)


def ips(
    env: ReplayEnv,
    policy: SelectionPolicy,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = 0,
    resamples: int = 2_000,
) -> OffPolicyEstimate:
    """Inverse propensity scoring: unbiased, high variance, needs overlap."""

    rows = _weights(env, policy)
    censored = sum(1 for _, _, step in rows if not step.observed)
    return _estimate(
        "ips",
        [weight * reward for weight, reward, _ in rows],
        [weight for weight, _, _ in rows],
        censored=censored,
        min_samples=min_samples,
        seed=seed,
        resamples=resamples,
    )


def snips(
    env: ReplayEnv,
    policy: SelectionPolicy,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = 0,
    resamples: int = 2_000,
) -> OffPolicyEstimate:
    """Self-normalised IPS: slightly biased, far more stable at small n."""

    rows = _weights(env, policy)
    censored = sum(1 for _, _, step in rows if not step.observed)
    weights = [weight for weight, _, _ in rows]
    return _estimate(
        "snips",
        [weight * reward for weight, reward, _ in rows],
        weights,
        censored=censored,
        min_samples=min_samples,
        seed=seed,
        resamples=resamples,
        normaliser=math.fsum(weights),
    )


def doubly_robust(
    env: ReplayEnv,
    policy: SelectionPolicy,
    reward_model: Callable[[Mapping[str, Any], str], float],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = 0,
    resamples: int = 2_000,
) -> OffPolicyEstimate:
    """Doubly robust: consistent if either propensities or the model are right.

    ``reward_model`` must return a finite prediction for any masked action; it
    is a variance-reduction device only and never becomes evidence on its own.
    """

    if not callable(reward_model):
        raise ReplayError("reward_model must be callable")
    rows = _weights(env, policy)
    censored = sum(1 for _, _, step in rows if not step.observed)
    contributions: list[float] = []
    for weight, reward, step in rows:
        distribution = policy.distribution(step.decision.candidate_set, step.decision.hints)
        baseline = 0.0
        for action, probability in distribution.items():
            if probability <= 0.0:
                continue
            prediction = reward_model(step.context, action)
            if isinstance(prediction, bool) or not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
                raise ReplayError("reward_model must return a finite number")
            baseline += probability * float(prediction)
        logged_prediction = float(reward_model(step.context, step.decision.chosen))
        if not math.isfinite(logged_prediction):
            raise ReplayError("reward_model must return a finite number")
        contributions.append(baseline + weight * (reward - logged_prediction))
    return _estimate(
        "doubly_robust",
        contributions,
        [weight for weight, _, _ in rows],
        censored=censored,
        min_samples=min_samples,
        seed=seed,
        resamples=resamples,
    )


def replayer(
    env: ReplayEnv,
    policy: SelectionPolicy,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = 0,
    resamples: int = 2_000,
) -> OffPolicyEstimate:
    """Rejection-sampling replay: keep only steps where the actions agree.

    Unbiased for a uniform logging policy and easy to read, but it discards
    every mismatch, so its effective sample size collapses quickly.
    """

    matched: list[float] = []
    censored = 0
    for step in env.steps:
        reward = env.reward_of(step)
        if reward is None:
            continue
        distribution = policy.distribution(step.decision.candidate_set, step.decision.hints)
        best = max(distribution.items(), key=lambda item: (item[1], item[0]))[0]
        if best != step.decision.chosen:
            continue
        matched.append(reward)
        if not step.observed:
            censored += 1
    return _estimate(
        "replayer",
        matched,
        [1.0] * len(matched),
        censored=censored,
        min_samples=min_samples,
        seed=seed,
        resamples=resamples,
    )


def evaluate(
    env: ReplayEnv,
    policy: SelectionPolicy,
    *,
    reward_model: Callable[[Mapping[str, Any], str], float] | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = 0,
    resamples: int = 2_000,
) -> Mapping[str, OffPolicyEstimate]:
    """Run every applicable estimator and return them by name."""

    results = {
        "ips": ips(env, policy, min_samples=min_samples, seed=seed, resamples=resamples),
        "snips": snips(env, policy, min_samples=min_samples, seed=seed, resamples=resamples),
        "replayer": replayer(env, policy, min_samples=min_samples, seed=seed, resamples=resamples),
    }
    if reward_model is not None:
        results["doubly_robust"] = doubly_robust(
            env, policy, reward_model, min_samples=min_samples, seed=seed, resamples=resamples
        )
    return MappingProxyType(results)


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "ESTIMATORS",
    "REPLAY_SCHEMA",
    "STATUSES",
    "Observation",
    "OffPolicyEstimate",
    "ReplayEnv",
    "ReplayError",
    "ReplayStep",
    "Transition",
    "default_reward",
    "doubly_robust",
    "effective_sample_size",
    "evaluate",
    "ips",
    "load_steps",
    "replayer",
    "snips",
]
