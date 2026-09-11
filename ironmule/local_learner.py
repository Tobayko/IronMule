"""What this machine has earned about one action, remembered across restarts, acting on nothing.

A fresh install knows nothing and serves the reference. `B75` showed it can measure its way
to a local decision in under five minutes; `B76` showed the *sign* of that decision is stable
over fourteen sessions while its magnitude moves; `B77` showed that machine-state features
made the forecast worse and that a plain mean and a small Bayesian posterior are the same
answer on this evidence. This module is the persistent form of exactly that much and no more.

**Shadow only.** `ExecutionRouter.decide` builds its route from the same dispatch facts it
always did and never reads this. A recommendation travels beside a decision as annotation, in
`RouteDecision.local_learning`, and is computed in `annotate()` -- once per dispatch, after
the route exists. `B70` measured why: an eager diagnostic inside `decide()` cost `78%` of a
`4.3 us` decision. Here the annotation is a dictionary built once per learner update and
looked up by workload class, so the per-dispatch cost is one mapping lookup.

**Fail closed, everywhere.** Missing state, corrupt state, a foreign fingerprint, a different
model revision, a different `mlx` build, evidence that failed correctness or the `B65`
resource gate, evidence marked `BLOCKED` or `INVALID`, an A/A control too noisy to read, an
interval that touches `1.0` -- every one of them yields `reference`. There is no partial
belief and no nearest guess.

**No false precision.** A preference and a gain are different things and are stored
separately. `B76` measured a stable sign with a variable magnitude, so this records
`action_preference = candidate` with an `expected_gain_distribution` that keeps its spread,
and never a single number like "15 per cent" as though it were a property of the silicon.

**No context features.** Load, free memory and swap stay out. `B77` re-scored `B76`'s sealed
predictions and found a ridge over those three features worse than a constant model on
prediction error, coverage and regret, with an error larger than the whole between-session
spread. Reopening that needs new evidence, not another fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "ironmule.local_learning.v1"

#: The four states a single action can be in on this machine. There is no fifth, and no
#: state in which something other than the reference runs without a completed qualification.
UNKNOWN = "UNKNOWN"
COLLECTING = "COLLECTING"
CANDIDATE_QUALIFIED = "CANDIDATE_QUALIFIED"
REFERENCE_ONLY = "REFERENCE_ONLY"
STATES = (UNKNOWN, COLLECTING, CANDIDATE_QUALIFIED, REFERENCE_ONLY)

REFERENCE = "reference"
CANDIDATE = "candidate"
DECISION_BOUNDARY = 1.0

#: Independent sessions required before any qualification is even considered. Three is the
#: smallest number from which an empirical interval over *sessions* can be formed at all;
#: two gives a spread of one gap. `B76` ran fourteen and its first prediction that named the
#: right action with the truth inside its interval arrived after two.
MIN_SESSIONS = 3
#: The A/A gate `B75` introduced and `B76` kept, applied here to the accepted evidence as a
#: whole: the control must sit on 1.0 and must not be wide.
AA_MAX_OFFSET = 0.05
AA_MAX_SPREAD = 0.05
#: The prior for estimator B. Deliberately wide enough that three sessions move it, and
#: centred on "no effect" so an unqualified action is never flattered.
PRIOR_MEAN = 1.0
PRIOR_SD = 0.15

#: Why `B69` is kept and not averaged in. Stated here because a rule that lives only in a
#: ledger is a rule nobody applies.
INCLUSION_RULE = (
    "evidence enters `action_preference` when it passes identity, correctness, the B65 "
    "resource gate and is not BLOCKED or INVALID. It enters `expected_gain_distribution` "
    "only if it *also* carries its own A/A control within "
    f"{AA_MAX_OFFSET} of 1.0 with a half width at most {AA_MAX_SPREAD}. B69 is valid "
    "historical evidence and its sign stands, but B77 measured its A/A control at a half "
    "width of 0.0848 with its candidate interval overlapping its own control, so its "
    "magnitude is recorded and never averaged into the local gain estimate. A study whose "
    "own control cannot separate the arms cannot set the scale of the effect")

#: Every field of an evidence row. Exact: an unknown field is a refusal, not a warning.
EVIDENCE_FIELDS = frozenset({
    "evidence_id", "action_id", "workload_class", "reference_stack",
    "hardware_fingerprint", "gpu_architecture", "mlx", "mlx_lm",
    "model_id", "model_identity_sha256", "model_revision",
    "ratio", "ci_low", "ci_high", "within_session_se", "aa_median", "aa_half_width",
    "correctness_passed", "resource_gate_passed", "status", "measured_at",
})
VALID = "VALID"


class LocalLearningError(ValueError):
    """State that cannot be fully understood. It is never partially believed."""


# --------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class Evidence:
    """One independent session's qualified result for one action on one workload class."""

    evidence_id: str
    action_id: str
    workload_class: str
    reference_stack: str
    hardware_fingerprint: str
    gpu_architecture: str
    mlx: str
    mlx_lm: str
    model_id: str
    model_identity_sha256: str
    model_revision: str
    ratio: float
    ci_low: float
    ci_high: float
    within_session_se: float | None
    aa_median: float | None
    aa_half_width: float | None
    correctness_passed: bool
    resource_gate_passed: bool
    status: str
    measured_at: str

    @property
    def aa_is_readable(self) -> bool:
        """Its own control sat on 1.0 and was narrow enough to separate the arms."""
        return bool(self.aa_median is not None and self.aa_half_width is not None
                    and abs(self.aa_median - DECISION_BOUNDARY) <= AA_MAX_OFFSET
                    and self.aa_half_width <= AA_MAX_SPREAD)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(EVIDENCE_FIELDS)}


def evidence_from(raw: Mapping[str, Any]) -> Evidence:
    """Parse one row, or refuse it. Nothing is defaulted and nothing is coerced."""
    if not isinstance(raw, Mapping):
        raise LocalLearningError("an evidence row is not a mapping")
    found = set(raw)
    missing = EVIDENCE_FIELDS - found
    unknown = found - EVIDENCE_FIELDS
    if missing:
        raise LocalLearningError(f"evidence is missing {sorted(missing)}")
    if unknown:
        raise LocalLearningError(f"evidence carries unknown {sorted(unknown)}")
    for name in ("evidence_id", "action_id", "workload_class", "reference_stack",
                 "hardware_fingerprint", "gpu_architecture", "mlx", "mlx_lm", "model_id",
                 "model_identity_sha256", "model_revision", "status", "measured_at"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise LocalLearningError(f"{name} must be a non-empty string")
    for name in ("ratio", "ci_low", "ci_high"):
        if isinstance(raw[name], bool) or not isinstance(raw[name], (int, float)):
            raise LocalLearningError(f"{name} must be a number")
        if not math.isfinite(float(raw[name])) or float(raw[name]) <= 0.0:
            raise LocalLearningError(f"{name} must be finite and positive")
    for name in ("within_session_se", "aa_median", "aa_half_width"):
        value = raw[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LocalLearningError(f"{name} must be a number or null")
        if not math.isfinite(float(value)):
            raise LocalLearningError(f"{name} must be finite")
    for name in ("correctness_passed", "resource_gate_passed"):
        if not isinstance(raw[name], bool):
            raise LocalLearningError(f"{name} must be a boolean")
    if not raw["ci_low"] <= raw["ratio"] <= raw["ci_high"]:
        raise LocalLearningError("ratio must lie inside its own interval")
    return Evidence(**raw)


@dataclass(frozen=True)
class IntakeContext:
    """What this installation is. Evidence measured under anything else is not ours."""

    hardware_fingerprint: str = ""
    gpu_architecture: str = ""
    mlx: str = ""
    mlx_lm: str = ""
    model_id: str = ""
    model_identity_sha256: str = ""
    model_revision: str = ""

    @property
    def identified(self) -> bool:
        return bool(self.hardware_fingerprint and self.model_identity_sha256
                    and self.mlx and self.mlx_lm and self.model_revision)


#: The order matters: the cheapest and most selective check runs first, and the first
#: failure is the reason reported. A rejection that cannot say why is not fail-closed, it
#: is silence.
def intake_reason(evidence: Evidence, context: IntakeContext) -> str:
    """"" when the row may be learned from, otherwise why it may not."""
    if not context.identified:
        return "this installation does not identify its machine, model and libraries"
    if evidence.hardware_fingerprint != context.hardware_fingerprint:
        return "hardware fingerprint"
    if evidence.model_identity_sha256 != context.model_identity_sha256:
        return "model identity"
    if evidence.model_revision != context.model_revision:
        return "model revision"
    if evidence.mlx != context.mlx:
        return "mlx build"
    if evidence.mlx_lm != context.mlx_lm:
        return "mlx_lm build"
    if context.gpu_architecture and evidence.gpu_architecture != context.gpu_architecture:
        return "gpu architecture"
    if evidence.status != VALID:
        return f"evidence status is {evidence.status}, not {VALID}"
    if not evidence.correctness_passed:
        return "correctness did not pass"
    if not evidence.resource_gate_passed:
        return "the B65 resource gate did not pass"
    return ""


# --------------------------------------------------------------------------- estimators


def running_mean(ratios: Sequence[float]) -> dict[str, Any] | None:
    """Estimator A. The mean of what was seen, with the spread of what was seen."""
    if len(ratios) < 2:
        return None
    centre = statistics.fmean(ratios)
    spread = statistics.stdev(ratios)
    # A predictive interval for the next session, not a confidence interval for the mean.
    predictive = spread * math.sqrt(1.0 + 1.0 / len(ratios))
    return {"estimator": "running_mean", "n": len(ratios), "center": centre,
            "uncertainty": predictive,
            "predictive_interval": [centre - 1.96 * predictive, centre + 1.96 * predictive],
            "observed_min": min(ratios), "observed_max": max(ratios), "observed_sd": spread}


def bayesian(ratios: Sequence[float],
             within_session_ses: Sequence[float | None]) -> dict[str, Any] | None:
    """Estimator B. Non-contextual, with its own uncertainty kept apart from the data's."""
    if len(ratios) < 2:
        return None
    errors = [se for se in within_session_ses if se is not None]
    within = statistics.fmean([se ** 2 for se in errors]) if errors else 0.0
    between = max(0.0, statistics.variance(ratios) - within)

    precision = 1.0 / PRIOR_SD ** 2
    weighted = PRIOR_MEAN / PRIOR_SD ** 2
    for index, ratio in enumerate(ratios):
        se = within_session_ses[index]
        variance = max((se ** 2 if se is not None else within) + between, 1e-9)
        precision += 1.0 / variance
        weighted += ratio / variance
    centre = weighted / precision
    posterior_variance = 1.0 / precision
    predictive = math.sqrt(posterior_variance + between + within)
    return {"estimator": "bayesian", "n": len(ratios), "center": centre,
            "uncertainty": predictive,
            "predictive_interval": [centre - 1.96 * predictive, centre + 1.96 * predictive],
            "posterior_sd": math.sqrt(posterior_variance),
            "between_session_variance": between, "mean_within_session_variance": within}


#: Both estimators must clear the boundary. `B77` measured them as the same answer on this
#: machine's evidence -- identical actions, identical regret, a prediction-error gap of one
#: per cent of the between-session spread -- so the data cannot pick one. Requiring both is
#: the conservative reading of a choice the evidence does not support.
ESTIMATOR_RULE = ("both the running mean and the Bayesian posterior must place their whole "
                  "95 per cent predictive interval below 1.0. B77 found the two "
                  "indistinguishable, so neither is trusted alone")


# --------------------------------------------------------------------------- policy


@dataclass(frozen=True)
class Recommendation:
    """What the learner would say, if anyone asked it to decide. Nobody does yet."""

    local_learning_state: str
    recommended_action: str
    estimated_ratio: float | None
    prediction_interval: tuple[float, float] | None
    evidence_count: int
    confidence: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        # Built by hand. `asdict` deep-copies, and `B70` measured what that costs.
        return {"local_learning_state": self.local_learning_state,
                "recommended_action": self.recommended_action,
                "estimated_ratio": self.estimated_ratio,
                "prediction_interval": (list(self.prediction_interval)
                                        if self.prediction_interval else None),
                "evidence_count": self.evidence_count,
                "confidence": self.confidence, "reason": self.reason,
                "shadow_only": True, "affects_dispatch": False}


NO_STATE = Recommendation(
    UNKNOWN, REFERENCE, None, None, 0, "none",
    "no local learning state: a fresh install serves the reference until it has earned "
    "something else")


def decide_state(accepted: Sequence[Evidence]) -> Recommendation:
    """The whole policy, in one pure function over accepted evidence.

    Nothing below reads a clock, a file, an environment variable or the machine. The same
    evidence always produces the same recommendation, which is what makes the cold-start
    replay a test rather than a demonstration.
    """
    if not accepted:
        return NO_STATE
    count = len(accepted)
    for_gain = [row for row in accepted if row.aa_is_readable]
    if count < MIN_SESSIONS:
        return Recommendation(
            COLLECTING, REFERENCE, None, None, count, "insufficient",
            f"{count} of {MIN_SESSIONS} independent sessions collected. The reference "
            f"serves until the evidence exists, which is not a failure but the design")
    if len(for_gain) < MIN_SESSIONS:
        return Recommendation(
            COLLECTING, REFERENCE, None, None, count, "insufficient",
            f"{count} sessions accepted but only {len(for_gain)} carry an A/A control "
            f"narrow enough to set a scale. The others stay as evidence of the sign")

    controls = [row.aa_median for row in for_gain if row.aa_median is not None]
    aa_centre = statistics.fmean(controls) if controls else None
    aa_spread = statistics.stdev(controls) if len(controls) > 1 else 0.0
    if aa_centre is None or abs(aa_centre - DECISION_BOUNDARY) > AA_MAX_OFFSET \
            or aa_spread > AA_MAX_SPREAD:
        return Recommendation(
            REFERENCE_ONLY, REFERENCE, None, None, count, "unusable",
            f"the A/A control over {len(for_gain)} sessions sits at {aa_centre} with a "
            f"spread of {aa_spread}. A machine whose own control cannot separate two "
            f"identical arms cannot qualify a third")

    ratios = [row.ratio for row in for_gain]
    mean_estimate = running_mean(ratios)
    bayes_estimate = bayesian(ratios, [row.within_session_se for row in for_gain])
    if mean_estimate is None or bayes_estimate is None:
        return Recommendation(COLLECTING, REFERENCE, None, None, count, "insufficient",
                              "not enough sessions to form a spread")

    intervals = (mean_estimate["predictive_interval"], bayes_estimate["predictive_interval"])
    if all(high < DECISION_BOUNDARY for _low, high in intervals):
        return Recommendation(
            CANDIDATE_QUALIFIED, CANDIDATE, bayes_estimate["center"],
            tuple(bayes_estimate["predictive_interval"]), count, "qualified",
            f"{len(for_gain)} of {count} accepted sessions carry an A/A control narrow "
            f"enough to set a scale, every one correctness-checked and past the B65 gate. "
            f"Their pooled control sits at {aa_centre:.4f} and both estimators place their "
            f"whole 95 per cent interval below {DECISION_BOUNDARY}")
    if all(low > DECISION_BOUNDARY for low, _high in intervals):
        return Recommendation(
            REFERENCE_ONLY, REFERENCE, bayes_estimate["center"],
            tuple(bayes_estimate["predictive_interval"]), count, "qualified",
            "the candidate is measurably slower than the reference on this machine")
    return Recommendation(
        COLLECTING, REFERENCE, bayes_estimate["center"],
        tuple(bayes_estimate["predictive_interval"]), count, "uncertain",
        f"an interval that touches {DECISION_BOUNDARY} has not decided anything. The "
        f"reference serves and the evidence keeps accumulating")


# --------------------------------------------------------------------------- the learner


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


class LocalLearner:
    """Persistent, per action and workload class, and inert.

    It owns accepted evidence and the recommendation that follows from it. It owns no
    model, starts no process, and is never consulted while a route is chosen.
    """

    def __init__(self, context: IntakeContext | None = None):
        self.context = context or IntakeContext()
        self._accepted: dict[tuple[str, str], list[Evidence]] = {}
        self._rejected: list[dict[str, Any]] = []
        self._recommendations: dict[tuple[str, str], Recommendation] = {}
        #: Precomputed once per update and looked up per dispatch. This is the whole of
        #: `B70`'s lesson: the hot path does no work a cold path could have done.
        self._shadow: dict[str, dict[str, Any]] = {}
        #: The digest, computed when the evidence changes and not when someone asks. An
        #: activation layer reads it once per dispatch, and `as_dict()` rebuilds every
        #: estimator over every row, which is not work a dispatch may pay for.
        self._digest: str = ""
        self.last_valid_update: str = ""

    # -- intake ---------------------------------------------------------------
    def observe(self, evidence: Evidence) -> dict[str, Any]:
        """Offer one row. Accepted rows change the state; rejected rows say why."""
        reason = intake_reason(evidence, self.context)
        if reason:
            record = {"evidence_id": evidence.evidence_id, "accepted": False,
                      "reason": reason, "action_id": evidence.action_id,
                      "workload_class": evidence.workload_class}
            self._rejected.append(record)
            return record
        key = (evidence.action_id, evidence.workload_class)
        known = self._accepted.setdefault(key, [])
        if any(row.evidence_id == evidence.evidence_id for row in known):
            record = {"evidence_id": evidence.evidence_id, "accepted": False,
                      "reason": "already observed: raw evidence is immutable and is never "
                                "counted twice",
                      "action_id": evidence.action_id,
                      "workload_class": evidence.workload_class}
            self._rejected.append(record)
            return record
        known.append(evidence)
        self.last_valid_update = evidence.measured_at
        self._recompute(key)
        return {"evidence_id": evidence.evidence_id, "accepted": True, "reason": "",
                "action_id": evidence.action_id, "workload_class": evidence.workload_class,
                "used_for_gain_distribution": evidence.aa_is_readable,
                "evidence_count": len(known)}

    def observe_all(self, rows: Iterable[Evidence]) -> list[dict[str, Any]]:
        return [self.observe(row) for row in rows]

    def _recompute(self, key: tuple[str, str]) -> None:
        self._digest = ""
        recommendation = decide_state(self._accepted[key])
        self._recommendations[key] = recommendation
        self._shadow = {}
        for (action_id, workload_class), row in self._recommendations.items():
            entry = {**row.as_dict(), "action_id": action_id}
            current = self._shadow.get(workload_class)
            # One action per class is the normal case. With several, a qualified one is
            # reported and anything else leaves the class unnamed rather than guessed.
            if current is None:
                self._shadow[workload_class] = entry
            elif row.local_learning_state == CANDIDATE_QUALIFIED:
                self._shadow[workload_class] = entry
            elif current["local_learning_state"] != CANDIDATE_QUALIFIED:
                self._shadow[workload_class] = {
                    **NO_STATE.as_dict(), "action_id": "",
                    "reason": "several actions are known for this workload class and none "
                              "is qualified, so none is named"}

    # -- reading --------------------------------------------------------------
    @property
    def digest(self) -> str:
        """The state's digest, cached until the evidence changes."""
        if not self._digest:
            self._digest = self.as_dict()["digest"]
        return self._digest

    def recommendation(self, action_id: str, workload_class: str) -> Recommendation:
        return self._recommendations.get((action_id, workload_class), NO_STATE)

    def shadow_for(self, workload_class: str) -> dict[str, Any] | None:
        """The router's way in. One mapping lookup, no allocation, no computation."""
        return self._shadow.get(workload_class)

    @property
    def rejected(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._rejected)

    def as_dict(self) -> dict[str, Any]:
        """The persistent form. Preference and gain are separate, deliberately."""
        actions = []
        for (action_id, workload_class), rows in sorted(self._accepted.items()):
            recommendation = self._recommendations[(action_id, workload_class)]
            for_gain = [row for row in rows if row.aa_is_readable]
            ratios = [row.ratio for row in for_gain]
            actions.append({
                "action_id": action_id,
                "workload_class": workload_class,
                "reference_stack": rows[0].reference_stack,
                "state": recommendation.local_learning_state,
                # Never a single number posing as a hardware parameter. B76 measured a
                # stable sign with a magnitude that moves, so both are written and the
                # preference does not carry the gain's precision.
                "action_preference": {
                    "preferred": recommendation.recommended_action,
                    "confidence": recommendation.confidence,
                    "reason": recommendation.reason,
                    "sessions_supporting_the_sign": sum(
                        1 for row in rows if row.ci_high < DECISION_BOUNDARY),
                    "sessions_accepted": len(rows),
                },
                "expected_gain_distribution": (
                    {"n": len(ratios), "observed_ratios": ratios,
                     "running_mean": running_mean(ratios),
                     "bayesian": bayesian(ratios,
                                          [row.within_session_se for row in for_gain]),
                     "inclusion_rule": INCLUSION_RULE,
                     "excluded_for_a_noisy_control": [
                         row.evidence_id for row in rows if not row.aa_is_readable]}
                    if ratios else None),
                # The rows themselves, not only their ids. `restore` rebuilds the state by
                # offering each one back through the same intake gates, so a file cannot
                # assert a qualification it cannot re-earn.
                "evidence": [row.as_dict() for row in rows],
                "aa_noise_estimate": {
                    "n": len(for_gain),
                    "center": (statistics.fmean([row.aa_median for row in for_gain])
                               if for_gain else None),
                    "spread": (statistics.stdev([row.aa_median for row in for_gain])
                               if len(for_gain) > 1 else 0.0 if for_gain else None)},
                "evidence_ids": [row.evidence_id for row in rows],
            })
        body = {"schema": SCHEMA, "actions": actions,
                "last_valid_update": self.last_valid_update,
                "context": {"hardware_fingerprint": self.context.hardware_fingerprint,
                            "gpu_architecture": self.context.gpu_architecture,
                            "mlx": self.context.mlx, "mlx_lm": self.context.mlx_lm,
                            "model_id": self.context.model_id,
                            "model_identity_sha256": self.context.model_identity_sha256,
                            "model_revision": self.context.model_revision},
                "estimator_rule": ESTIMATOR_RULE,
                "shadow_only": True, "activation": "none"}
        return {**body, "digest": _canonical_digest(body)}

    # -- persistence ----------------------------------------------------------
    def save(self, path: Path) -> Path:
        """Atomic, so a killed process leaves the previous state rather than half of one."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def restore(cls, path: Path, context: IntakeContext,
                evidence: Iterable[Evidence] = ()) -> "LocalLearner":
        """Rebuild from disk, or return a learner that knows nothing.

        The stored digest is checked, the stored context is compared against this
        installation's, and the state is rebuilt from the *evidence*, never from the
        stored summary. A summary is a report; evidence is the thing that can be re-checked.
        """
        learner = cls(context)
        learner.observe_all(evidence)
        try:
            stored = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return learner
        if not isinstance(stored, Mapping) or stored.get("schema") != SCHEMA:
            return learner
        digest = stored.get("digest")
        body = {name: value for name, value in stored.items() if name != "digest"}
        if not isinstance(digest, str) or _canonical_digest(body) != digest:
            return learner
        context_row = stored.get("context")
        if not isinstance(context_row, Mapping):
            return learner
        for name in ("hardware_fingerprint", "model_identity_sha256", "model_revision",
                     "mlx", "mlx_lm"):
            if context_row.get(name) != getattr(context, name):
                return learner
        # Rebuilt from the evidence, never from the summary beside it. Every row goes back
        # through `observe`, so a stored row that would be refused today is refused now.
        for action in stored.get("actions") or ():
            if not isinstance(action, Mapping):
                return cls(context)
            for raw in action.get("evidence") or ():
                try:
                    learner.observe(evidence_from(raw))
                except LocalLearningError:
                    # One unreadable row makes the whole file untrustworthy, so the learner
                    # is thrown away rather than partly believed.
                    return cls(context)
        learner.restored_from = str(path)
        return learner


def default_state_path() -> Path:
    """Beside the tuned profiles, under `IRONMULE_HOME` when that is set."""
    from .hw import _store

    return _store() / "local_learning.json"


def _self_check() -> None:
    """The policy's own smallest proof: it never leaves the reference without earning it."""
    context = IntakeContext("fp", "arch", "0.32.0", "0.31.3", "m", "sha", "rev")

    def row(index: int, ratio: float, aa: float = 1.0, aa_half: float = 0.005,
            **overrides) -> Evidence:
        base = {"evidence_id": f"e{index}", "action_id": "a", "workload_class": "single_short",
                "reference_stack": "A", "hardware_fingerprint": "fp",
                "gpu_architecture": "arch", "mlx": "0.32.0", "mlx_lm": "0.31.3",
                "model_id": "m", "model_identity_sha256": "sha", "model_revision": "rev",
                "ratio": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01,
                "within_session_se": 0.005, "aa_median": aa, "aa_half_width": aa_half,
                "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
                "measured_at": f"2026-09-11T0{index}:00:00+00:00"}
        return evidence_from({**base, **overrides})

    empty = LocalLearner(context)
    assert empty.recommendation("a", "single_short").recommended_action == REFERENCE
    assert empty.recommendation("a", "single_short").local_learning_state == UNKNOWN

    learner = LocalLearner(context)
    learner.observe(row(1, 0.96))
    assert learner.recommendation("a", "single_short").local_learning_state == COLLECTING
    learner.observe(row(2, 0.955))
    assert learner.recommendation("a", "single_short").recommended_action == REFERENCE, \
        "two sessions is below the minimum and must not qualify anything"
    learner.observe(row(3, 0.965))
    qualified = learner.recommendation("a", "single_short")
    assert qualified.local_learning_state == CANDIDATE_QUALIFIED, qualified
    assert qualified.recommended_action == CANDIDATE
    assert qualified.prediction_interval[1] < DECISION_BOUNDARY

    foreign = LocalLearner(IntakeContext("other", "arch", "0.32.0", "0.31.3", "m", "sha", "rev"))
    assert foreign.observe(row(1, 0.96))["reason"] == "hardware fingerprint"
    assert foreign.recommendation("a", "single_short").recommended_action == REFERENCE

    noisy = LocalLearner(context)
    noisy.observe_all([row(index, 0.96, aa=1.0, aa_half=0.4) for index in (1, 2, 3)])
    assert noisy.recommendation("a", "single_short").recommended_action == REFERENCE, \
        "a control too wide to separate identical arms cannot qualify a third"
    print("local_learner self-check passed")


if __name__ == "__main__":
    _self_check()
