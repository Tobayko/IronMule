"""Equal-budget replay and shadow-only hierarchical optimisation for Q4.

The optimiser is a ranking/reporting layer.  It has no worker, executor,
model, MLX, profile or persistence dependency.  A caller supplies immutable
panel evidence; this module returns deterministic candidates and a report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
import hashlib
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

try:
    from .q4_contracts import (
        KNOB_ACTIONS,
        KNOB_ACTION_BY_ID,
        ALL_DECLARED_KNOB_ACTIONS,
        LEGAL_KNOB_ACTIONS,
        KNOB_CANDIDATES,
        KnobAction,
        KnobCandidateSpec,
        INTERACTION_KNOB_ANCHORS,
        SCHEDULE_ACTIONS,
        ScheduleAction,
        PlanKind,
        HybridAction,
        Outcome,
        Q4State,
        Dataset,
        canonical_json,
        canonical_sha256,
    )
    from .q4_methods import (
        ActionScore,
        EBHCORL,
        Method,
        ReplayTransition,
        deterministic_policy_order,
        grouped_support,
        q_lcb,
        failure_ucb,
        behaviour_score,
        rank_catalogue,
        normalize_method,
        DataInsufficientError,
        join_contract_rows,
        MIN_SUPPORT,
    )
except ImportError:  # pragma: no cover - direct source loading convenience
    from q4_contracts import (  # type: ignore[no-redef]
        KNOB_ACTIONS, KNOB_ACTION_BY_ID, ALL_DECLARED_KNOB_ACTIONS, LEGAL_KNOB_ACTIONS,
        KNOB_CANDIDATES, KnobAction, KnobCandidateSpec, INTERACTION_KNOB_ANCHORS,
        SCHEDULE_ACTIONS, ScheduleAction, PlanKind, HybridAction, Outcome, Q4State, Dataset, canonical_json, canonical_sha256,
    )
    from q4_methods import (  # type: ignore[no-redef]
        ActionScore, EBHCORL, Method, ReplayTransition,
        deterministic_policy_order, grouped_support, q_lcb, failure_ucb,
        behaviour_score, MIN_SUPPORT,
        rank_catalogue,
        normalize_method, DataInsufficientError, join_contract_rows,
    )


KNOB_DECISION_BUDGET = 11
STRATEGY_DECISION_BUDGET = 5
TOTAL_DECISION_BUDGET = 16
SHADOW_RECOMMENDATION = "SHADOW_RECOMMENDATION"
NOT_COMPUTABLE = "NOT_COMPUTABLE"
KEEP_IF_RATIO_BELOW = 0.995
_VERIFICATION_TOKEN = object()


class Decision(str, Enum):
    RL_WINS = "RL_WINS"
    SIMPLER_WINS = "SIMPLER_WINS"
    TIE_NO_RL = "TIE_NO_RL"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    SAFETY_FAILURE = "SAFETY_FAILURE"
    OPE_UNSUPPORTED = "OPE_UNSUPPORTED"


_DEFAULT_INTERACTION_ANCHORS = tuple(KNOB_ACTIONS) if len(KNOB_ACTIONS) == 12 else tuple(INTERACTION_KNOB_ANCHORS)


def _id(item: Any) -> str:
    if isinstance(item, str):
        return item
    return str(getattr(item, "action_id", ""))


def _plan(item: Any) -> str:
    value = getattr(item, "plan_kind", item.get("plan_kind") if isinstance(item, Mapping) else "")
    return str(getattr(value, "value", value))


def _safe_strategies(plan_kind: str | None) -> tuple[Any, ...]:
    if plan_kind is not None:
        try:
            return tuple(ScheduleAction.safe_pool(plan_kind))
        except Exception:
            return ()
    return tuple(item for item in SCHEDULE_ACTIONS if not item.is_risk_probe)


def _action_values(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    return {name: getattr(item, name) for name in (
        "fuse_projections", "compiled_fixed_cache", "fused_argmax",
        "head_skip_prefill", "prefill_into_fixed", "readback_every",
        "speculate_k", "speculate_ngram", "capacity_slack", "wired_fraction",
    ) if hasattr(item, name)}


def _one_field_diff(left: Any, right: Any) -> bool:
    a, b = _action_values(left), _action_values(right)
    keys = tuple(sorted(set(a) | set(b)))
    return sum(a.get(key) != b.get(key) for key in keys) == 1


def _apply_candidate(current: Any, candidate: Any) -> Any | None:
    values = dict(_action_values(current))
    field = getattr(candidate, "changed_field", candidate.get("changed_field") if isinstance(candidate, Mapping) else None)
    target = getattr(candidate, "target_value", candidate.get("target_value") if isinstance(candidate, Mapping) else None)
    if field not in values or target is None or values.get(field) == target:
        return None
    values[field] = target
    try:
        return KnobAction(**values)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """A panel slot, kept separate from a transition delta.

    Panel slots are complete configurations.  A trajectory chooses a legal
    one-field delta from the *current* slot, so a rejected candidate never
    causes the next source ID to drift or silently assume a fixed catalogue
    order.
    """

    action_id: str
    action: Any = None
    label: str = ""
    safe: bool = True

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("candidate action_id is required")
        if self.safe is not True:
            # Risk probes can be represented in an inventory but not a policy.
            object.__setattr__(self, "safe", False)

    @classmethod
    def from_contract(cls, candidate: Any) -> "CandidateSpec":
        return cls(str(getattr(candidate, "candidate_id")), candidate, str(getattr(candidate, "changed_field", "")))


@dataclass(frozen=True, slots=True)
class ActionPanel:
    """Immutable complete panel descriptor used by both stages."""

    # Stage 1 is the eleven declared candidate specs, not the 12 absolute
    # interaction anchors.  Stage 2 uses the latter independently.
    knob_candidates: tuple[CandidateSpec, ...] = tuple(CandidateSpec.from_contract(item) for item in KNOB_CANDIDATES)
    strategy_candidates: tuple[CandidateSpec, ...] = ()
    interaction_anchors: tuple[Any, ...] = _DEFAULT_INTERACTION_ANCHORS

    def __post_init__(self) -> None:
        if len({item.action_id for item in self.knob_candidates}) != len(self.knob_candidates):
            raise ValueError("duplicate knob panel action")
        if any(not item.safe for item in self.knob_candidates):
            raise ValueError("knob panel may contain only safe policy candidates")
        if self.strategy_candidates and any(not item.safe for item in self.strategy_candidates):
            raise ValueError("strategy panel may contain only safe policy candidates")
        if len(self.interaction_anchors) != 12 or any(_id(item) not in ALL_DECLARED_KNOB_ACTIONS for item in self.interaction_anchors):
            raise ValueError("interaction anchor panel must contain the declared twelve safe anchors")

    @property
    def knob_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.knob_candidates)

    @property
    def interaction_knob_ids(self) -> tuple[str, ...]:
        return tuple(_id(item) for item in self.interaction_anchors)

    def strategies(self, plan_kind: str | None = None) -> tuple[CandidateSpec, ...]:
        if self.strategy_candidates:
            candidates = tuple(item for item in self.strategy_candidates if item.safe and (plan_kind is None or _plan(item.action) == plan_kind))
            return candidates
        return tuple(CandidateSpec(item.action_id, item, getattr(item, "label", "")) for item in _safe_strategies(plan_kind))

    def complete_knob_panel(self) -> bool:
        return len(self.knob_candidates) == len(KNOB_CANDIDATES) and set(self.knob_ids) == set(item.candidate_id for item in KNOB_CANDIDATES)

    def complete_strategy_panel(self, plan_kind: str) -> bool:
        return len(self.strategies(plan_kind)) == 5 and len({item.action_id for item in self.strategies(plan_kind)}) == 5

    def complete_cross_product(self, measured_pairs: Iterable[tuple[str, str]], plan_kind: str) -> bool:
        expected = {(knob_id, strategy.action_id) for knob_id in self.interaction_knob_ids for strategy in self.strategies(plan_kind)}
        return len(expected) == 60 and expected == set((str(k), str(s)) for k, s in measured_pairs)


@dataclass(frozen=True, slots=True)
class BudgetLedger:
    knob_actions: tuple[str, ...]
    strategy_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.knob_actions) > KNOB_DECISION_BUDGET or len(self.strategy_actions) > STRATEGY_DECISION_BUDGET:
            raise ValueError("equal-budget action cap exceeded")
        if len(set(self.knob_actions)) != len(self.knob_actions) or len(set(self.strategy_actions)) != len(self.strategy_actions):
            raise ValueError("a candidate cannot be evaluated twice in a stage")

    @property
    def total(self) -> int:
        return len(self.knob_actions) + len(self.strategy_actions)

    def to_dict(self) -> dict[str, Any]:
        return {"knob_actions": list(self.knob_actions), "strategy_actions": list(self.strategy_actions), "total": self.total, "budget": TOTAL_DECISION_BUDGET}


@dataclass(frozen=True, slots=True)
class EvaluatorCell:
    """Strict metric cell joining one evaluator-owned Outcome to its context."""

    context_id: str
    action_id: str
    outcome: Outcome
    cost: float
    stage: str = "KNOB_DELTA"
    candidate_id: str | None = None
    split: str = "Q4_SEALED_HOLDOUT"

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome) or not self.outcome.complete_safe:
            raise ValueError("metric cell requires a complete safe Outcome")
        if not _digest_id(self.context_id) or not _digest_id(self.action_id):
            raise ValueError("metric cell IDs must be lowercase digests")
        if not math.isfinite(float(self.cost)) or self.cost <= 0:
            raise ValueError("metric cell cost must be finite and positive")
        if self.stage not in {"KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"}:
            raise ValueError("metric cell stage is outside the frozen vocabulary")
        if self.split != "Q4_SEALED_HOLDOUT":
            raise ValueError("direct EvaluatorCell must belong to Q4_SEALED_HOLDOUT")
        expected = self.outcome.metric_value(self.stage)
        if abs(float(self.cost) - expected) > max(1e-12, abs(expected) * 1e-12):
            raise ValueError("metric cell cost does not match Outcome.metric_value(stage)")
        if self.stage in {"STRATEGY_SELECT", "REVALIDATE"} and self.outcome.strategy_action_id != self.action_id:
            raise ValueError("strategy metric cell action is not bound to Outcome.strategy_action_id")
        if self.stage == "KNOB_DELTA" and self.candidate_id is None and self.outcome.knob_action_id != self.action_id:
            raise ValueError("knob metric cell action is not bound to Outcome.knob_action_id")


def legal_knob_neighbors(current: Any, panel: ActionPanel) -> tuple[CandidateSpec, ...]:
    """Return panel slots that are legal one-field deltas from ``current``."""
    result = []
    for item in panel.knob_candidates:
        target = _apply_candidate(current, item.action)
        if target is not None and _one_field_diff(current, target):
            result.append(CandidateSpec(item.action_id, target, item.label))
    return tuple(result)


def _rows_for_action(rows: Iterable[Any], action_id: str) -> tuple[Any, ...]:
    return tuple(row for row in rows if _id(getattr(row, "action_id", row.get("action_id", "") if isinstance(row, Mapping) else "")) == action_id)


def _safe_row(row: Any) -> bool:
    failed = getattr(row, "failed", row.get("failed", False) if isinstance(row, Mapping) else False)
    censored = getattr(row, "censored", row.get("censored", False) if isinstance(row, Mapping) else False)
    complete = getattr(row, "complete_safe", row.get("complete_safe") if isinstance(row, Mapping) else None)
    return not bool(failed or censored) and (complete is None or complete is True)


def _safe_qualified_evidence(row: Any) -> bool:
    """Check an evaluator-owned safe/raw outcome without trusting a label."""
    outcome = getattr(row, "outcome", row.get("outcome") if isinstance(row, Mapping) else None)
    # A caller-provided mapping/boolean is not evaluator ownership.  Only the
    # strict Q4 Outcome contract can authorize a hybrid composition.
    if not isinstance(outcome, Outcome):
        return False
    return bool(outcome.complete_safe and outcome.raw_artifact_refs and outcome.raw_sample_count > 0)


def _digest_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validated_dataset_pairs(dataset: Any, plan_kind: str) -> tuple[tuple[str, str], ...]:
    """Derive trusted Stage-2 pairs from strict Dataset-owned records only."""
    if not isinstance(dataset, Dataset):
        return ()
    artifact_by_id = {item.artifact_id: item for item in dataset.source_artifacts}
    outcomes = {item.outcome_id: item for item in dataset.outcomes}
    contexts = {item.context_id: item for item in dataset.contexts}
    safe_ids = {item.action_id for item in _safe_strategies(plan_kind)}
    expected = {(knob.action_id, strategy.action_id) for knob in KNOB_ACTIONS for strategy in _safe_strategies(plan_kind)}
    panel_by_context: dict[str, set[tuple[str, str]]] = {}
    for cell in getattr(dataset, "panel_cells", ()):
        context = contexts.get(cell.context_id)
        outcome = outcomes.get(cell.outcome_id)
        reference = outcomes.get(cell.reference_outcome_id)
        if context is None or outcome is None or reference is None:
            continue
        if not outcome.complete_safe or not reference.complete_safe:
            continue
        if outcome.context_id != cell.context_id or reference.context_id != cell.context_id:
            continue
        if cell.strategy_action_id not in safe_ids or outcome.strategy_action_id != cell.strategy_action_id:
            continue
        if outcome.knob_action_id != cell.knob_action_id or reference.knob_action_id != cell.knob_action_id:
            continue
        if reference.strategy_action_id != _safe_strategies(plan_kind)[0].action_id:
            continue
        if outcome.plan_kind != plan_kind or reference.plan_kind != plan_kind:
            continue
        if any(ref.artifact_id not in artifact_by_id or artifact_by_id[ref.artifact_id].sha256 != ref.sha256 for ref in outcome.raw_artifact_refs + reference.raw_artifact_refs):
            continue
        panel_by_context.setdefault(cell.context_id, set()).add((cell.knob_action_id, cell.strategy_action_id))
    if not panel_by_context or any(pairs != expected for pairs in panel_by_context.values()):
        return ()
    return tuple(sorted(expected))


def _coordinate_order(panel: ActionPanel, rows: Sequence[Any]) -> tuple[str, ...]:
    """Build an order from the current incumbent using dynamic neighbors."""
    current = next((item.action for item in panel.knob_candidates if item.action_id == KNOB_ACTIONS[0].action_id), KNOB_ACTIONS[0])
    remaining = list(panel.knob_candidates)
    chosen: list[str] = []
    for _ in range(KNOB_DECISION_BUDGET):
        neighbors = [candidate for candidate in legal_knob_neighbors(current, ActionPanel(tuple(remaining), panel.strategy_candidates)) if candidate.action_id not in chosen]
        if not neighbors:
            break
        # Existing coordinate semantics use frozen panel order; the incumbent
        # changes only after a safe measured improvement.
        candidate = neighbors[0]
        chosen.append(candidate.action_id)
        candidate_rows = [row for row in rows if _id(getattr(row, "candidate_id", row.get("candidate_id", "") if isinstance(row, Mapping) else "")) == candidate.action_id or _id(getattr(row, "action_id", row.get("action_id", "") if isinstance(row, Mapping) else "")) == candidate.action_id]
        ratios = []
        for row in candidate_rows:
            if not _safe_row(row):
                continue
            ratio = getattr(row, "ratio", row.get("ratio") if isinstance(row, Mapping) else None)
            if ratio is None:
                reward = getattr(row, "reward", row.get("reward") if isinstance(row, Mapping) else None)
                ratio = math.exp(-float(reward)) if reward is not None and math.isfinite(float(reward)) else None
            if ratio is not None and math.isfinite(float(ratio)) and float(ratio) > 0:
                ratios.append(float(ratio))
        # Existing Q2/E11 keep semantics: accept strictly below 0.995.  A
        # missing ratio is not a win and never advances the incumbent.
        if ratios and sum(ratios) / len(ratios) < KEEP_IF_RATIO_BELOW and candidate.action is not None:
            current = candidate.action
        remaining = [item for item in remaining if item.action_id != candidate.action_id]
    return tuple(chosen)


def equal_budget_plan(method: Method | str, panel: ActionPanel | None = None, rows: Iterable[Any] = (), *, seed: str = "Q4-RL-20260901", plan_kind: str | None = None) -> BudgetLedger:
    """Create a deterministic 11+5 plan for any declared method."""
    method = normalize_method(method)
    panel = panel or ActionPanel()
    rows = tuple(rows)
    if method is Method.BASELINE:
        # BASE is a shared external reference, not a fabricated sixteen-action
        # trajectory and not a learner that can consume tuned evidence.
        return BudgetLedger((), ())
    knob_ids = panel.knob_ids
    strategy_ids = tuple(item.action_id for item in panel.strategies(plan_kind))
    if method is Method.CURRENT_COORDINATE:
        ordered_knobs = _coordinate_order(panel, rows)
    else:
        ordered_knobs = rank_catalogue(method, knob_ids, rows, seed=f"{seed}/knob")
    # A custom panel can contain identifiers absent from the frozen catalogue;
    # preserve its explicit order and still keep deterministic no-repeat rules.
    ordered_knobs = ordered_knobs + tuple(item for item in knob_ids if item not in ordered_knobs)
    ordered_strategies = rank_catalogue(method, strategy_ids, rows, seed=f"{seed}/strategy")
    ordered_strategies = ordered_strategies + tuple(item for item in strategy_ids if item not in ordered_strategies)
    return BudgetLedger(tuple(ordered_knobs[:KNOB_DECISION_BUDGET]), tuple(ordered_strategies[:STRATEGY_DECISION_BUDGET]))


@dataclass(frozen=True, slots=True)
class MethodReport:
    method: str
    ledger: BudgetLedger
    status: str = "READY"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "status": self.status, "reason": self.reason, **self.ledger.to_dict()}


def _row_cost(row: Any) -> float | None:
    value = getattr(row, "cost", row.get("cost") if isinstance(row, Mapping) else None)
    if value is None:
        value = getattr(row, "original_cost", row.get("original_cost") if isinstance(row, Mapping) else None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _strict_metric_row(row: Any) -> bool:
    if isinstance(row, EvaluatorCell):
        return True
    outcome = row if isinstance(row, Outcome) else getattr(row, "outcome", None)
    return isinstance(outcome, Outcome) and outcome.complete_safe


def _metric_action_key(row: Any) -> str:
    candidate = getattr(row, "candidate_id", None)
    if candidate:
        return str(candidate)
    return _id(getattr(row, "action_id", row.get("action_id", "") if isinstance(row, Mapping) else ""))


def _row_context(row: Any) -> str:
    value = getattr(row, "context_id", row.get("context_id") if isinstance(row, Mapping) else None)
    return str(value or getattr(row, "group_id", row.get("group_id", "") if isinstance(row, Mapping) else ""))


def safe_context_oracle(rows: Iterable[Any], required_action_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Select the oracle only when every context has a complete safe panel."""
    required = set(str(item) for item in (required_action_ids or ()))
    candidates: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        cost = _row_cost(row)
        if cost is None or not _safe_row(row) or not _strict_metric_row(row):
            continue
        action = _id(getattr(row, "action_id", row.get("action_id", "") if isinstance(row, Mapping) else ""))
        if action:
            candidates.setdefault(_row_context(row), []).append((action, cost))
    per_context: dict[str, dict[str, Any]] = {}
    for context, values in sorted(candidates.items()):
        if required and required - {action for action, _ in values}:
            continue
        action, cost = min(values, key=lambda item: (item[1], item[0]))
        per_context[context] = {"action_id": action, "cost": cost}
    return {
        "status": "COMPUTABLE" if per_context else NOT_COMPUTABLE,
        "contexts": per_context,
        "context_count": len(per_context),
    }


def time_to_best(costs: Sequence[float], oracle_cost: float, *, tolerance: float = 0.01) -> int | None:
    """Return the first original-cost decision within 1% of the oracle."""
    if not costs or not math.isfinite(oracle_cost) or oracle_cost <= 0:
        return None
    for index, cost in enumerate(costs):
        if math.isfinite(cost) and cost > 0 and cost <= oracle_cost * (1.0 + tolerance):
            return index + 1
    return None


def normalized_regret(method_cost: float | None, oracle_cost: float | None) -> float | None:
    if method_cost is None or oracle_cost is None or not math.isfinite(method_cost) or not math.isfinite(oracle_cost) or oracle_cost <= 0:
        return None
    return (method_cost - oracle_cost) / oracle_cost


def regression_rates(rows: Iterable[Any], baseline_cost: float | None) -> dict[str, float | None]:
    rows = tuple(rows)
    unsafe = sum(not _safe_row(row) for row in rows)
    unsafe_rate = unsafe / len(rows) if rows else None
    safe = [row for row in rows if _safe_row(row) and _row_cost(row) is not None]
    regressions = sum(float(_row_cost(row)) > baseline_cost * 1.02 for row in safe) if baseline_cost is not None else None
    safe_rate = regressions / len(safe) if regressions is not None and safe else (0.0 if regressions == 0 else None)
    return {"unsafe_censored_rate": unsafe_rate, "safe_regression_rate": safe_rate, "attempted": len(rows), "safe": len(safe)}


def regression_rates_by_context(rows: Iterable[Any], baseline_cost_by_context: Mapping[str, float]) -> dict[str, float | None]:
    """Compute exact rates per context, then aggregate them equally."""
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_row_context(row), []).append(row)
    if not grouped:
        return {"unsafe_censored_rate": None, "safe_regression_rate": None, "context_count": 0}
    context_rates = [regression_rates(values, baseline_cost_by_context.get(context)) for context, values in grouped.items()]
    unsafe = [item["unsafe_censored_rate"] for item in context_rates if item["unsafe_censored_rate"] is not None]
    safe = [item["safe_regression_rate"] for item in context_rates if item["safe_regression_rate"] is not None]
    return {
        "unsafe_censored_rate": sum(unsafe) / len(unsafe) if unsafe else None,
        "safe_regression_rate": sum(safe) / len(safe) if safe else None,
        "context_count": len(grouped),
    }


def grouped_bootstrap_ci(values_by_context: Mapping[str, float], *, seed: str = "Q4-RL-20260901", samples: int = 1000) -> dict[str, Any]:
    """Deterministic equal-context percentile interval (95%)."""
    values = tuple(float(value) for _, value in sorted(values_by_context.items()) if math.isfinite(float(value)))
    if len(values) < 2:
        return {"status": NOT_COMPUTABLE, "lower": None, "upper": None, "context_count": len(values)}
    draws: list[float] = []
    count = max(100, int(samples))
    for draw in range(count):
        selected = []
        for index in range(len(values)):
            digest = hashlib.sha256(f"{seed}/{draw}/{index}".encode("utf-8")).digest()
            selected.append(values[int.from_bytes(digest[:8], "big") % len(values)])
        draws.append(sum(selected) / len(selected))
    draws.sort()
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return {"status": "COMPUTABLE", "lower": lower, "upper": upper, "context_count": len(values), "samples": count}


def uncertainty_calibration(intervals: Iterable[tuple[float, float]], observed: Iterable[float], *, level: float = 0.90) -> dict[str, Any]:
    pairs = tuple((float(low), float(high), float(value)) for (low, high), value in zip(intervals, observed) if math.isfinite(float(low)) and math.isfinite(float(high)) and math.isfinite(float(value)) and low <= high)
    if not pairs:
        return {"status": NOT_COMPUTABLE, "coverage": None, "interval_width": None, "log_score": None, "level": level}
    coverage = sum(low <= value <= high for low, high, value in pairs) / len(pairs)
    width = sum(high - low for low, high, _ in pairs) / len(pairs)
    log_scores = []
    for low, high, value in pairs:
        width_i = max(high - low, 1e-12)
        distance = 0.0 if low <= value <= high else min(abs(value - low), abs(value - high))
        log_scores.append(-math.log(width_i) - distance / width_i)
    return {"status": "COMPUTABLE", "coverage": coverage, "interval_width": width, "log_score": sum(log_scores) / len(log_scores), "level": level, "count": len(pairs)}


def grouped_advantage_ci(rl_cost_by_context: Mapping[str, float], comparator_cost_by_context: Mapping[str, float], *, seed: str = "Q4-RL-20260901") -> dict[str, Any]:
    """95% grouped CI for positive lower-is-better RL advantage."""
    common = sorted(set(rl_cost_by_context) & set(comparator_cost_by_context))
    values = {context: (float(comparator_cost_by_context[context]) - float(rl_cost_by_context[context])) / float(comparator_cost_by_context[context]) for context in common if float(comparator_cost_by_context[context]) > 0 and math.isfinite(float(rl_cost_by_context[context]))}
    return grouped_bootstrap_ci(values, seed=seed)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    method: str
    status: str
    decision: str
    oracle: Mapping[str, Any]
    best_cost: float | None
    regret: float | None
    time_to_best: int | None
    experiments_to_best: int | None
    regression: Mapping[str, Any]
    confidence_interval: Mapping[str, Any]
    calibration: Mapping[str, Any]
    budget: Mapping[str, Any]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "status": self.status, "decision": self.decision,
            "oracle": dict(self.oracle), "best_cost": self.best_cost, "regret": self.regret,
            "time_to_best": self.time_to_best, "experiments_to_best": self.experiments_to_best,
            "regression": dict(self.regression), "confidence_interval": dict(self.confidence_interval),
            "calibration": dict(self.calibration), "budget": dict(self.budget), "reason": self.reason,
        }


def evaluate_method(
    method: Method | str,
    rows: Iterable[Any],
    *,
    oracle_rows: Iterable[Any] | None = None,
    baseline_cost_by_context: Mapping[str, float] | None = None,
    plan: BudgetLedger | None = None,
    split: str = "Q4_SEALED_HOLDOUT",
) -> EvaluationReport:
    """Compute direct, context-normalized metrics with explicit missingness."""
    method = normalize_method(method)
    rows = tuple(rows)
    if split != "Q4_SEALED_HOLDOUT":
        empty = safe_context_oracle(())
        return EvaluationReport(method.value, NOT_COMPUTABLE, Decision.DATA_INSUFFICIENT.value, empty, None, None, None, None, regression_rates(rows, None), {"status": NOT_COMPUTABLE}, {"status": NOT_COMPUTABLE}, (plan or BudgetLedger((), ())).to_dict(), "direct sealed evaluator accepts only Q4_SEALED_HOLDOUT")
    if oracle_rows is None:
        empty = safe_context_oracle(())
        return EvaluationReport(method.value, NOT_COMPUTABLE, Decision.DATA_INSUFFICIENT.value, empty, None, None, None, None, regression_rates(rows, None), {"status": NOT_COMPUTABLE}, {"status": NOT_COMPUTABLE}, (plan or BudgetLedger((), ())).to_dict(), "complete safe oracle panel is required")
    oracle_source = tuple(oracle_rows)
    oracle_action_ids = {_id(getattr(row, "action_id", row.get("action_id", "") if isinstance(row, Mapping) else "")) for row in oracle_source}
    oracle = safe_context_oracle(oracle_source, oracle_action_ids)
    if plan is None:
        return EvaluationReport(method.value, NOT_COMPUTABLE, Decision.DATA_INSUFFICIENT.value, oracle, None, None, None, None, regression_rates(rows, None), {"status": NOT_COMPUTABLE}, {"status": NOT_COMPUTABLE}, BudgetLedger((), ()).to_dict(), "ledger-selected actions are required")
    selected_ids = set(plan.knob_actions) | set(plan.strategy_actions)
    if method is Method.BASELINE:
        selected_ids = {KNOB_ACTIONS[0].action_id}
    rows = tuple(row for row in rows if _metric_action_key(row) in selected_ids)
    if not rows or oracle["status"] != "COMPUTABLE":
        return EvaluationReport(method.value, NOT_COMPUTABLE, Decision.DATA_INSUFFICIENT.value, oracle, None, None, None, None, regression_rates(rows, None), {"status": NOT_COMPUTABLE}, {"status": NOT_COMPUTABLE}, (plan or BudgetLedger((), ())).to_dict(), "missing safe original-cost rows or sealed oracle")
    per_context_cost: dict[str, float] = {}
    per_context_sequences: dict[str, list[float]] = {}
    for row in rows:
        cost = _row_cost(row)
        context = _row_context(row)
        if cost is None or not _safe_row(row) or not _strict_metric_row(row):
            continue
        per_context_cost[context] = min(per_context_cost.get(context, float("inf")), cost)
        per_context_sequences.setdefault(context, []).append(cost)
    if not per_context_cost:
        return EvaluationReport(method.value, NOT_COMPUTABLE, Decision.DATA_INSUFFICIENT.value, oracle, None, None, None, None, regression_rates(rows, None), {"status": NOT_COMPUTABLE}, {"status": NOT_COMPUTABLE}, (plan or BudgetLedger((), ())).to_dict(), "no computable safe method cost")
    common = sorted(set(per_context_cost) & set(oracle["contexts"]))
    if not common:
        return EvaluationReport(method.value, NOT_COMPUTABLE, Decision.DATA_INSUFFICIENT.value, oracle, None, None, None, None, regression_rates(rows, None), {"status": NOT_COMPUTABLE}, {"status": NOT_COMPUTABLE}, (plan or BudgetLedger((), ())).to_dict(), "method and oracle have no common context")
    best = sum(per_context_cost[item] for item in common) / len(common)
    oracle_cost = sum(float(oracle["contexts"][item]["cost"]) for item in common) / len(common)
    baseline = sum(baseline_cost_by_context[item] for item in common) / len(common) if baseline_cost_by_context and all(item in baseline_cost_by_context for item in common) else None
    regret = normalized_regret(best, oracle_cost)
    ci = grouped_bootstrap_ci({item: normalized_regret(per_context_cost[item], oracle["contexts"][item]["cost"]) or 0.0 for item in common})
    times = [time_to_best(per_context_sequences[item], float(oracle["contexts"][item]["cost"])) for item in common]
    if any(value is None for value in times):
        ttb = None
        exp = None
    else:
        ordered = sorted(int(value) for value in times if value is not None)
        mid = len(ordered) // 2
        ttb = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        exp = ttb
    regression = regression_rates_by_context(rows, baseline_cost_by_context or {}) if baseline_cost_by_context else regression_rates(rows, baseline)
    return EvaluationReport(method.value, "COMPUTABLE", Decision.TIE_NO_RL.value, oracle, best, regret, ttb, exp, regression, ci, {"status": NOT_COMPUTABLE}, (plan or BudgetLedger((), ())).to_dict(), "direct metrics only; final decision requires comparator gates")


def decide_rl(
    *,
    advantage: float | None,
    lower_bound: float | None,
    rl_time_to_best: float | None,
    simpler_time_to_best: float | None,
    rl_experiments_to_best: float | None,
    simpler_experiments_to_best: float | None,
    rl_regression_rate: float | None,
    simpler_regression_rate: float | None,
    coverage: float | None,
    support_pass: bool,
    model_strata_pass: bool,
    data_complete: bool,
    safety_ok: bool,
    ope_status: str = "SUPPORTED",
) -> Decision:
    """Apply the preregistered frozen RL decision gates."""
    if not safety_ok:
        return Decision.SAFETY_FAILURE
    if ope_status != "SUPPORTED":
        return Decision.OPE_UNSUPPORTED
    if not data_complete or not support_pass or any(value is None for value in (advantage, lower_bound, rl_time_to_best, simpler_time_to_best, rl_experiments_to_best, simpler_experiments_to_best, rl_regression_rate, simpler_regression_rate, coverage)):
        return Decision.DATA_INSUFFICIENT
    if lower_bound > 0.02 and advantage > 0.01 and rl_time_to_best < simpler_time_to_best and rl_experiments_to_best < simpler_experiments_to_best and rl_regression_rate <= simpler_regression_rate + 0.01 and 0.80 <= coverage <= 0.98 and model_strata_pass:
        return Decision.RL_WINS
    if advantage < -0.01:
        return Decision.SIMPLER_WINS
    return Decision.TIE_NO_RL


class ReplayEngine:
    """Replay method orders against supplied rows without evaluating candidates."""

    def __init__(self, panel: ActionPanel | None = None, *, seed: str = "Q4-RL-20260901") -> None:
        self.panel = panel or ActionPanel()
        self.seed = seed

    def replay(self, method: Method | str, rows: Iterable[Any] = (), *, plan_kind: str | None = None, dataset: Any = None) -> MethodReport:
        method = normalize_method(method)
        rows = tuple(rows)
        if method is Method.EB_HCORL:
            # An isolated unit fit is useful for tests, but a method replay is
            # structurally applicable only to the frozen 24/72/1224 dataset.
            if not isinstance(dataset, Dataset):
                return MethodReport(method.value, BudgetLedger((), ()), "NOT_APPLICABLE", "OFFLINE_RL requires a strict Q4 Dataset")
            try:
                ready, reasons = dataset._rl_readiness()
            except (AttributeError, TypeError, ValueError) as exc:
                return MethodReport(method.value, BudgetLedger((), ()), "DATA_INSUFFICIENT", f"dataset readiness unavailable: {exc}")
            if not ready:
                return MethodReport(method.value, BudgetLedger((), ()), "NOT_APPLICABLE", "; ".join(reasons) or "Q4 dataset readiness gate failed")
            try:
                derived = dataset.derive_rewards()
                reward_rows = getattr(derived, "rewards", derived)
                state_lookup = {item.state_digest: item for item in dataset.states}
                rows = join_contract_rows(dataset.transitions, state_lookup=state_lookup, derived_rewards=reward_rows)
            except (AttributeError, DataInsufficientError, TypeError, ValueError) as exc:
                return MethodReport(method.value, BudgetLedger((), ()), "DATA_INSUFFICIENT", f"strict dataset reward/state join failed: {exc}")
            try:
                policy = EBHCORL().fit(rows)
            except DataInsufficientError as exc:
                return MethodReport(method.value, BudgetLedger((), ()), "DATA_INSUFFICIENT", str(exc))
            state = next((getattr(row, "state", row.get("state") if isinstance(row, Mapping) else None) for row in rows), None)
            if state is None:
                return MethodReport(method.value, equal_budget_plan(method, self.panel, rows, seed=self.seed, plan_kind=plan_kind), "DATA_INSUFFICIENT", "no state evidence")
            first = policy.select_knob(state, self.panel.knob_ids)
            fallback = equal_budget_plan(method, self.panel, rows, seed=self.seed, plan_kind=plan_kind)
            knobs = (first.action_id,) + tuple(item for item in fallback.knob_actions if item != first.action_id)
            return MethodReport(method.value, BudgetLedger(knobs[:KNOB_DECISION_BUDGET], fallback.strategy_actions), "READY" if first.allowed else "DATA_INSUFFICIENT", first.reason)
        return MethodReport(method.value, equal_budget_plan(method, self.panel, rows, seed=self.seed, plan_kind=plan_kind))


@dataclass(frozen=True, slots=True)
class HybridRecommendation:
    schema: str
    status: str
    knob_action_id: str
    strategy_action_id: str
    stage_order: tuple[str, ...]
    objective_class: str
    plan_kind: str
    knob_score: float | None
    strategy_score: float | None
    knob_uncertainty: tuple[float, float] | None
    strategy_uncertainty: tuple[float, float] | None
    safety_decision: str
    eligible_for_future_revalidation: bool = field(default=False, init=False)
    evidence_ids: tuple[str, ...] = ()
    reason: str = ""
    recommendation_id: str = ""
    _eligibility_token: object = field(default=None, init=False, repr=False, compare=False)

    SCHEMA = "ironmule.q4_hybrid_recommendation.v1"

    def __post_init__(self) -> None:
        if self.status != SHADOW_RECOMMENDATION:
            raise ValueError("Q4 hybrid recommendations are shadow-only")
        if self.stage_order != ("KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"):
            raise ValueError("stage order is frozen")
        if self.objective_class not in {"LATENCY", "THROUGHPUT"}:
            raise ValueError("objective_class is outside the frozen vocabulary")
        if self.plan_kind not in {"StrictOneShotPlan", "ReusableSessionPlan"}:
            raise ValueError("plan_kind is outside the frozen vocabulary")
        if self.knob_action_id not in ALL_DECLARED_KNOB_ACTIONS:
            raise ValueError("recommendation knob action is outside the frozen panel")
        strategy = next((item for item in SCHEDULE_ACTIONS if item.action_id == self.strategy_action_id), None)
        if strategy is None or strategy.is_risk_probe or _plan(strategy) != self.plan_kind:
            raise ValueError("recommendation strategy must be a safe action")
        for interval in (self.knob_uncertainty, self.strategy_uncertainty):
            if interval is not None:
                if len(interval) != 2 or not all(math.isfinite(float(value)) for value in interval) or interval[0] > interval[1]:
                    raise ValueError("uncertainty must be a finite ordered interval")
        for score in (self.knob_score, self.strategy_score):
            if score is not None and not math.isfinite(float(score)):
                raise ValueError("recommendation scores must be finite")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if any(not _digest_id(item) for item in evidence_ids):
            raise ValueError("recommendation evidence IDs must be lowercase digests")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        computed = canonical_sha256(self._semantic_dict())
        if self.recommendation_id and self.recommendation_id != computed:
            raise ValueError("recommendation_id does not match canonical content")
        object.__setattr__(self, "recommendation_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "status": self.status,
            "knob_action_id": self.knob_action_id, "strategy_action_id": self.strategy_action_id,
            "stage_order": self.stage_order, "objective_class": self.objective_class, "plan_kind": self.plan_kind,
            "knob_score": self.knob_score, "strategy_score": self.strategy_score,
            "knob_uncertainty": self.knob_uncertainty, "strategy_uncertainty": self.strategy_uncertainty,
            "safety_decision": self.safety_decision,
            "eligible_for_future_revalidation": self.eligible_for_future_revalidation,
            "evidence_ids": self.evidence_ids, "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "recommendation_id": self.recommendation_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HybridRecommendation":
        expected = {
            "schema", "status", "knob_action_id", "strategy_action_id", "stage_order",
            "objective_class", "plan_kind", "knob_score", "strategy_score", "knob_uncertainty",
            "strategy_uncertainty", "safety_decision", "eligible_for_future_revalidation",
            "evidence_ids", "reason", "recommendation_id",
        }
        if set(data) != expected or data.get("schema") != cls.SCHEMA:
            raise ValueError("hybrid recommendation fields differ")
        return cls(
            schema=data["schema"], status=data["status"], knob_action_id=data["knob_action_id"],
            strategy_action_id=data["strategy_action_id"], stage_order=tuple(data["stage_order"]),
            objective_class=data["objective_class"], plan_kind=data["plan_kind"], knob_score=data["knob_score"],
            strategy_score=data["strategy_score"], knob_uncertainty=None if data["knob_uncertainty"] is None else tuple(data["knob_uncertainty"]),
            strategy_uncertainty=None if data["strategy_uncertainty"] is None else tuple(data["strategy_uncertainty"]),
            safety_decision=data["safety_decision"],
            evidence_ids=tuple(data["evidence_ids"]), reason=data["reason"], recommendation_id=data["recommendation_id"] if not data["eligible_for_future_revalidation"] else "",
        )

    def validate(self) -> bool:
        """Re-run strict identity checks on an already materialized report."""
        return self.recommendation_id == canonical_sha256(self._semantic_dict())

    def _with_eligibility(self) -> "HybridRecommendation":
        trusted = object.__new__(type(self))
        for item in fields(type(self)):
            object.__setattr__(trusted, item.name, getattr(self, item.name))
        object.__setattr__(trusted, "eligible_for_future_revalidation", True)
        object.__setattr__(trusted, "_eligibility_token", _VERIFICATION_TOKEN)
        object.__setattr__(trusted, "recommendation_id", canonical_sha256(trusted._semantic_dict()))
        return trusted


class ShadowSigner(Protocol):
    """Signer interface; cryptographic implementation remains caller-owned."""

    key_id: str
    algorithm: str

    def sign(self, payload: bytes) -> str | bytes: ...


class ShadowVerifier(Protocol):
    def verify(self, payload: bytes, signature: str, key_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ShadowRecommendationEnvelope:
    """Optional signed envelope for an inspectable shadow report.

    No cryptography is implemented here.  Without both a signer and a
    verifier, the envelope is explicitly unsigned/ineligible.
    """

    recommendation: HybridRecommendation
    signer_key_id: str | None = None
    signature_algorithm: str | None = None
    signature: str | None = None
    verified: bool = field(default=False, init=False)
    _verification_token: object = field(default=None, init=False, repr=False, compare=False)
    envelope_id: str = ""

    SCHEMA = "ironmule.q4_shadow_recommendation_envelope.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation, HybridRecommendation):
            raise ValueError("envelope recommendation must be strict HybridRecommendation")
        if self.signature is not None and (not self.signer_key_id or self.signature_algorithm != "Ed25519"):
            raise ValueError("signed envelope requires Ed25519 and a signer key id")
        computed = canonical_sha256(self._semantic_dict())
        if self.envelope_id and self.envelope_id != computed:
            raise ValueError("envelope_id does not match canonical content")
        object.__setattr__(self, "envelope_id", computed)

    @property
    def eligible(self) -> bool:
        return bool(self.verified and self._verification_token is _VERIFICATION_TOKEN and self.signature and self.recommendation.eligible_for_future_revalidation)

    def _semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA, "recommendation": self.recommendation.to_dict(),
            "signer_key_id": self.signer_key_id, "signature_algorithm": self.signature_algorithm,
            "signature": self.signature, "verified": self.verified,
        }

    def signing_bytes(self) -> bytes:
        # The envelope ID/signature are excluded from the signed payload to
        # avoid a circular digest; the recommendation is fully canonical.
        return canonical_json({
            "schema": self.SCHEMA, "recommendation": self.recommendation.to_dict(),
            "signer_key_id": self.signer_key_id, "signature_algorithm": self.signature_algorithm,
        }).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "envelope_id": self.envelope_id, "eligible": self.eligible}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowRecommendationEnvelope":
        expected = {"schema", "recommendation", "signer_key_id", "signature_algorithm", "signature", "verified", "envelope_id", "eligible"}
        if set(data) != expected or data.get("schema") != cls.SCHEMA:
            raise ValueError("shadow envelope fields differ")
        # Serialized trust flags are untrusted claims.  They are deliberately
        # discarded; callers must invoke verify() with their local verifier.
        envelope_id = data["envelope_id"] if not data["verified"] and not data["eligible"] else ""
        envelope = cls(
            recommendation=HybridRecommendation.from_dict(data["recommendation"]),
            signer_key_id=data["signer_key_id"], signature_algorithm=data["signature_algorithm"],
            signature=data["signature"], envelope_id=envelope_id,
        )
        return envelope

    def verify(self, verifier: ShadowVerifier | None) -> "ShadowRecommendationEnvelope":
        """Return a trusted copy only after an external verifier succeeds."""
        if verifier is None or not self.signature or not callable(getattr(verifier, "verify", None)):
            return self
        try:
            valid = bool(verifier.verify(self.signing_bytes(), self.signature, str(self.signer_key_id)))
        except Exception:
            valid = False
        if not valid:
            return self
        trusted = object.__new__(type(self))
        for item in fields(type(self)):
            object.__setattr__(trusted, item.name, getattr(self, item.name))
        object.__setattr__(trusted, "verified", True)
        object.__setattr__(trusted, "_verification_token", _VERIFICATION_TOKEN)
        object.__setattr__(trusted, "envelope_id", canonical_sha256(trusted._semantic_dict()))
        return trusted


class HybridOptimizer:
    """Two-stage optimizer that can only return an inspectable shadow report."""

    def __init__(self, *, method: Method | str = Method.EB_HCORL, seed: str = "Q4-RL-20260901") -> None:
        self.method = normalize_method(method)
        self.seed = seed

    def recommend(
        self,
        *,
        objective_class: str,
        plan_kind: str,
        knob_action_id: str | None = None,
        strategy_action_id: str | None = None,
        knob_score: float | None = None,
        strategy_score: float | None = None,
        knob_uncertainty: tuple[float, float] | None = None,
        strategy_uncertainty: tuple[float, float] | None = None,
        measured_pairs: Iterable[tuple[str, str]] = (),
        panel: ActionPanel | None = None,
        evidence_ids: Iterable[str] = (),
        dataset: Any = None,
    ) -> HybridRecommendation:
        panel = panel or ActionPanel()
        measured_pairs = _validated_dataset_pairs(dataset, plan_kind) if dataset is not None else ()
        evidence_ids = tuple(str(item) for item in evidence_ids)
        strategies = panel.strategies(plan_kind)
        base_knob = KNOB_ACTIONS[0].action_id
        base_strategy = strategies[0].action_id if strategies else _safe_strategies(plan_kind)[0].action_id
        requested_knob = knob_action_id if knob_action_id in ALL_DECLARED_KNOB_ACTIONS else base_knob
        requested_strategy = strategy_action_id if strategy_action_id in {item.action_id for item in strategies} else base_strategy
        complete = panel.complete_knob_panel() and panel.complete_strategy_panel(plan_kind) and panel.complete_cross_product(measured_pairs, plan_kind)
        if not complete:
            return HybridRecommendation(
                self._recommendation_schema(), SHADOW_RECOMMENDATION, base_knob, base_strategy,
                ("KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"), objective_class, plan_kind,
                None, None, None, None, "BASE_FALLBACK", tuple(evidence_ids),
                "missing or incomplete exact knob×strategy cross-product; BASE required",
            )
        # Stage 2 is allowed to consume only the selected Stage-1 action and a
        # measured exact pair.  It cannot change knobs or plan/mode.
        pairs = set(measured_pairs)
        if (requested_knob, requested_strategy) not in pairs:
            requested_knob, requested_strategy = base_knob, base_strategy
            safety = "BASE_FALLBACK"
            eligible = False
            reason = "selected pair is not measured under the exact final knob"
        else:
            safety = "CROSS_PRODUCT_BACKED"
            artifact_digests = {item.sha256 for item in dataset.source_artifacts} if isinstance(dataset, Dataset) else set()
            eligible = isinstance(dataset, Dataset) and bool(evidence_ids) and set(evidence_ids).issubset(artifact_digests)
            reason = "exact cross-product evidence present"
        recommendation = HybridRecommendation(
            self._recommendation_schema(), SHADOW_RECOMMENDATION, requested_knob, requested_strategy,
            ("KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"), objective_class, plan_kind,
            knob_score, strategy_score, knob_uncertainty, strategy_uncertainty,
            safety, tuple(evidence_ids), reason,
        )
        if safety == "CROSS_PRODUCT_BACKED" and eligible:
            return recommendation._with_eligibility()
        return recommendation

    def recommend_from_evidence(
        self,
        *,
        state: Any,
        rows: Iterable[Any],
        objective_class: str,
        plan_kind: str,
        panel: ActionPanel | None = None,
        dataset: Any = None,
        artifact_registry: Mapping[str, Any] | None = None,
    ) -> HybridRecommendation:
        """Fit Stage 1, condition Stage 2 on its selected knob, then report.

        This method is still offline: ``rows`` are evaluator-owned evidence,
        and the return value is always a shadow recommendation.  Incomplete
        pair coverage, missing safe/raw outcomes, absent state/reward joins,
        or invalid evidence hashes immediately produce the BASE report.
        """
        panel = panel or ActionPanel()
        source_rows = tuple(rows)
        if dataset is not None and not source_rows:
            outcomes_by_id = {item.outcome_id: item for item in getattr(dataset, "outcomes", ())}
            materialized: list[dict[str, Any]] = []
            for transition in getattr(dataset, "transitions", ()):
                outcome = outcomes_by_id.get(transition.outcome_id)
                if outcome is None:
                    continue
                try:
                    cost = outcome.metric_value(transition.stage)
                except (AttributeError, ValueError):
                    continue
                materialized.append({
                    "outcome": outcome, "transition_id": transition.transition_id,
                    "outcome_id": outcome.outcome_id, "knob_action_id": outcome.knob_action_id,
                    "strategy_action_id": outcome.strategy_action_id, "action_id": transition.action_id,
                    "context_id": transition.context.context_id, "cost": cost,
                    "reference_outcome_id": transition.reference_outcome_id,
                })
            source_rows = tuple(materialized)
        if dataset is None and artifact_registry is None:
            return self.recommend(objective_class=objective_class, plan_kind=plan_kind, panel=panel, measured_pairs=(), evidence_ids=())
        dataset_artifacts = {}
        dataset_outcomes = {}
        dataset_transitions = {}
        if dataset is not None:
            dataset_artifacts = {str(getattr(item, "artifact_id", "")): item for item in getattr(dataset, "source_artifacts", ())}
            dataset_outcomes = {str(getattr(item, "outcome_id", "")): item for item in getattr(dataset, "outcomes", ())}
            dataset_transitions = {str(getattr(item, "transition_id", "")): item for item in getattr(dataset, "transitions", ())}
        registry = dict(artifact_registry or {})
        safe_rows = tuple(row for row in source_rows if _safe_qualified_evidence(row))
        pair_rows: list[tuple[str, str]] = []
        evidence_ids: list[str] = []
        for row in safe_rows:
            outcome = getattr(row, "outcome", row.get("outcome") if isinstance(row, Mapping) else None)
            if not isinstance(outcome, Outcome):
                continue
            knob = getattr(row, "knob_action_id", row.get("knob_action_id") if isinstance(row, Mapping) else None)
            strategy = getattr(row, "strategy_action_id", row.get("strategy_action_id") if isinstance(row, Mapping) else None)
            knob = knob or outcome.knob_action_id
            strategy = strategy or outcome.strategy_action_id
            if strategy is None:
                candidate = getattr(row, "action_id", row.get("action_id") if isinstance(row, Mapping) else None)
                if candidate in {item.action_id for item in panel.strategies(plan_kind)}:
                    strategy = candidate
            raw = getattr(row, "evidence_ids", row.get("evidence_ids") if isinstance(row, Mapping) else ()) or ()
            if dataset is not None:
                outcome_id = str(getattr(row, "outcome_id", row.get("outcome_id", "") if isinstance(row, Mapping) else ""))
                if outcome_id and dataset_outcomes.get(outcome_id) != outcome:
                    continue
                reference_id = str(getattr(row, "reference_outcome_id", row.get("reference_outcome_id", "") if isinstance(row, Mapping) else ""))
                reference = dataset_outcomes.get(reference_id)
                if reference is None or not reference.complete_safe:
                    continue
                transition_id = str(getattr(row, "transition_id", row.get("transition_id", "") if isinstance(row, Mapping) else ""))
                if transition_id and transition_id not in dataset_transitions:
                    continue
            if dataset is None and not registry:
                continue
            references = tuple(outcome.raw_artifact_refs) + tuple(reference.raw_artifact_refs) if dataset is not None else tuple(outcome.raw_artifact_refs)
            if any((getattr(item, "artifact_id", "") not in dataset_artifacts and getattr(item, "sha256", "") not in registry) for item in references):
                continue
            if knob in panel.interaction_knob_ids and strategy in {item.action_id for item in panel.strategies(plan_kind)}:
                pair_rows.append((str(knob), str(strategy)))
            refs = getattr(outcome, "raw_artifact_refs", getattr(outcome, "raw_sample_refs", ())) if outcome is not None else ()
            for item in tuple(raw) + tuple(refs or ()):
                digest = getattr(item, "sha256", item.get("sha256") if isinstance(item, Mapping) else item)
                if _digest_id(digest):
                    evidence_ids.append(digest)
        # Require every pair to have a qualified safe/raw observation.  A row
        # merely mentioning an action ID is not enough to qualify composition.
        pair_set = set(pair_rows)
        exact = panel.complete_knob_panel() and panel.complete_strategy_panel(plan_kind) and panel.complete_cross_product(pair_set, plan_kind)
        if not exact or not evidence_ids or any(not _digest_id(item) for item in evidence_ids):
            return self.recommend(
                objective_class=objective_class, plan_kind=plan_kind, panel=panel,
            measured_pairs=(), evidence_ids=(), dataset=dataset,
            )
        train_rows: list[ReplayTransition] = []
        for row in source_rows:
            if isinstance(row, ReplayTransition):
                train_rows.append(row)
                continue
            try:
                reward = getattr(row, "reward", row.get("reward") if isinstance(row, Mapping) else None)
                state_value = getattr(row, "state", row.get("state") if isinstance(row, Mapping) else None)
                action = getattr(row, "action_id", row.get("action_id") if isinstance(row, Mapping) else None)
                if reward is not None and state_value is not None and action:
                    train_rows.append(ReplayTransition(
                        state=state_value, action_id=str(action), reward=float(reward),
                        next_state=getattr(row, "next_state", row.get("next_state") if isinstance(row, Mapping) else None),
                        terminal=bool(getattr(row, "terminal", row.get("terminal", True) if isinstance(row, Mapping) else True)),
                        stage=str(getattr(row, "stage", row.get("stage", "STRATEGY_SELECT") if isinstance(row, Mapping) else "STRATEGY_SELECT")),
                        context_id=str(getattr(row, "context_id", row.get("context_id", "evidence") if isinstance(row, Mapping) else "evidence")),
                        group_id=str(getattr(row, "group_id", row.get("group_id", "evidence") if isinstance(row, Mapping) else "evidence")),
                        propensity=float(getattr(row, "behaviour_propensity", row.get("behaviour_propensity", 1.0) if isinstance(row, Mapping) else 1.0)),
                    ))
            except (TypeError, ValueError):
                continue
        if not train_rows and dataset is not None:
            try:
                derived = dataset.derive_rewards()
                reward_rows = getattr(derived, "rewards", None)
                if reward_rows is None:
                    raise DataInsufficientError("Dataset.derive_rewards() did not return a RewardDerivationResult.rewards mapping")
                state_lookup = {item.state_digest: item for item in getattr(dataset, "states", ())}
                train_rows.extend(join_contract_rows(
                    getattr(dataset, "transitions", ()),
                    state_lookup=state_lookup,
                    derived_rewards=reward_rows,
                ))
            except (AttributeError, DataInsufficientError, TypeError, ValueError) as exc:
                base = self.recommend(objective_class=objective_class, plan_kind=plan_kind, panel=panel, measured_pairs=(), evidence_ids=())
                return replace(base, reason=f"strict dataset reward/state join failed: {exc}")
        if not train_rows:
            return self.recommend(objective_class=objective_class, plan_kind=plan_kind, panel=panel, measured_pairs=(), evidence_ids=())
        policy = EBHCORL().fit(train_rows)
        knob_score = policy.select_knob(state, panel.knob_ids)
        current_id = str(getattr(state, "knob_action_id", state.get("knob_action_id", KNOB_ACTIONS[0].action_id) if isinstance(state, Mapping) else KNOB_ACTIONS[0].action_id))
        if current_id not in ALL_DECLARED_KNOB_ACTIONS:
            current_id = KNOB_ACTIONS[0].action_id
        selected_knob = current_id
        if knob_score.allowed:
            selected_candidate = next((item for item in panel.knob_candidates if item.action_id == knob_score.action_id), None)
            target = _apply_candidate(ALL_DECLARED_KNOB_ACTIONS[current_id], selected_candidate.action if selected_candidate else None)
            if target is not None:
                selected_knob = target.action_id
        # Stage 2 receives the selected complete knob identity explicitly.  It
        # cannot alter that identity or substitute another plan.
        if isinstance(state, Q4State):
            stage2_state = replace(state, stage="STRATEGY_SELECT", step_index=11, strategy_candidate_index=0, knob_action_id=selected_knob)
        elif isinstance(state, Mapping):
            stage2_state = dict(state)
            stage2_state["knob_action_id"] = selected_knob
            stage2_state["stage"] = "STRATEGY_SELECT"
            stage2_state["step_index"] = 11
            stage2_state["strategy_candidate_index"] = 0
        else:
            stage2_state = {"features": getattr(state, "feature_vector", lambda: (1.0,))(), "in_domain": bool(getattr(state, "in_domain", False)), "knob_action_id": selected_knob}
        strategies = panel.strategies(plan_kind)
        strategy_score = policy.select_strategy(stage2_state, tuple(item.action_id for item in strategies))
        selected_strategy = strategy_score.action_id if strategy_score.allowed and (selected_knob, strategy_score.action_id) in pair_set else strategies[0].action_id
        recommendation = self.recommend(
            objective_class=objective_class, plan_kind=plan_kind,
            knob_action_id=selected_knob, strategy_action_id=selected_strategy,
            knob_score=knob_score.score if knob_score.allowed else None,
            strategy_score=strategy_score.score if strategy_score.allowed else None,
            knob_uncertainty=(knob_score.q_lcb, knob_score.q_lcb) if knob_score.allowed else None,
            strategy_uncertainty=(strategy_score.q_lcb, strategy_score.q_lcb) if strategy_score.allowed else None,
            measured_pairs=pair_set, panel=panel, evidence_ids=tuple(sorted(set(evidence_ids))), dataset=dataset,
        )
        # This is the only path that can mark future revalidation eligible;
        # all strict Outcome/raw/evidence checks happened above.
        if recommendation.safety_decision == "CROSS_PRODUCT_BACKED":
            return recommendation._with_eligibility()
        return recommendation

    def optimize(self, **kwargs: Any) -> HybridRecommendation:
        return self.recommend(**kwargs)

    def sign_recommendation(
        self,
        recommendation: HybridRecommendation,
        *,
        signer: ShadowSigner | None = None,
        verifier: ShadowVerifier | None = None,
    ) -> ShadowRecommendationEnvelope:
        if signer is None:
            return ShadowRecommendationEnvelope(recommendation)
        key_id = str(getattr(signer, "key_id", "")) or None
        algorithm = str(getattr(signer, "algorithm", "")) or None
        if algorithm != "Ed25519" or not key_id or not callable(getattr(signer, "sign", None)):
            return ShadowRecommendationEnvelope(recommendation)
        provisional = ShadowRecommendationEnvelope(recommendation, key_id, algorithm, None)
        try:
            signature_value = signer.sign(provisional.signing_bytes())
            signature = signature_value.decode("utf-8") if isinstance(signature_value, bytes) else str(signature_value)
        except Exception:
            return ShadowRecommendationEnvelope(recommendation)
        if not signature or verifier is None or not callable(getattr(verifier, "verify", None)):
            return ShadowRecommendationEnvelope(recommendation, key_id, algorithm, signature, False)
        signed = ShadowRecommendationEnvelope(recommendation, key_id, algorithm, signature)
        return signed.verify(verifier)

    def recommend_signed(self, **kwargs: Any) -> ShadowRecommendationEnvelope:
        signer = kwargs.pop("signer", None)
        verifier = kwargs.pop("verifier", None)
        recommendation = self.recommend(**kwargs)
        return self.sign_recommendation(recommendation, signer=signer, verifier=verifier)

    @staticmethod
    def _recommendation_schema() -> str:
        return HybridRecommendation.SCHEMA


__all__ = [
    "KNOB_DECISION_BUDGET", "STRATEGY_DECISION_BUDGET", "TOTAL_DECISION_BUDGET",
    "SHADOW_RECOMMENDATION", "NOT_COMPUTABLE", "Decision", "CandidateSpec", "ActionPanel", "BudgetLedger",
    "KEEP_IF_RATIO_BELOW",
    "legal_knob_neighbors", "equal_budget_plan", "MethodReport", "ReplayEngine",
    "safe_context_oracle", "time_to_best", "normalized_regret", "regression_rates",
    "grouped_bootstrap_ci", "uncertainty_calibration", "EvaluationReport", "evaluate_method", "decide_rl",
    "HybridRecommendation", "ShadowSigner", "ShadowVerifier", "ShadowRecommendationEnvelope", "HybridOptimizer",
]
