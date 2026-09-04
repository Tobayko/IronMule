"""Turn one measured session into one label.

R0 built the decision log and R1 the replay environment, but nothing ever
connected a measurement to its reward: ``record_outcome`` is reachable only from
the CLI, with the number typed in by hand (``cli.py:917``). That is workable for
a handful of points and impossible for the four hundred R2 needs, so this module
closes the gap.

**Where the number comes from.** The session result persists, per pair and per
arm, the raw ``total_ns`` series (``real_session._measurement_arm``). That series
*is* the request's wall time, so the ratio is read straight off it:

    per pair   ratio = median(candidate.total_ns) / median(baseline.total_ns)
    reward     ratio_median = median(ratio over pairs)

Paired first, then aggregated -- the pairing carries the run-to-run variance that
dwarfs every effect in this project (unpaired `20,5 %` against paired `1,32 %`,
``docs/ERGEBNISSE.md``).

**Why not compose it from the phase ratios** the evaluator already reports
(``metrics.ttft`` and ``metrics.decode_tps``): because that is E04, the measured
dead end. Phase ratios do not multiply, they average time-weighted, and a
composed number here would be a projection wearing a measurement's clothes.

**Why censored runs still produce a record.** Dropping a timeout or a failed gate
biases every later estimate toward the actions that happened to survive
(``decisions.OutcomeEvent`` docstring). A censored outcome carries no reward but
is written all the same.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from .decisions import DecisionError, OutcomeEvent

#: The schema ``real_session`` stamps on its result file.
RESULT_SCHEMA = "friday.optimizer.session-result.v1"
EVIDENCE_SCHEMA = "friday.optimizer.measurement-evidence.v1"

#: The only metric this bridge emits. ``replay.default_reward`` applies
#: ``1 - reward`` without consulting ``reward_metric``, which is correct for a
#: time ratio and silently sign-flipped for ``decode_ratio`` (throughput, higher
#: is better). Emitting one time-like metric keeps that trap shut.
REWARD_METRIC = "ratio_median"


class RewardError(ValueError):
    """The session result cannot be turned into a label."""


def _series(arm: Any, name: str) -> list[float] | None:
    if not isinstance(arm, Mapping):
        return None
    values = arm.get(name)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return None
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return None
        out.append(float(value))
    return out


def ratio_median(test_stage: Mapping[str, Any]) -> float | None:
    """Paired request-time ratio, or ``None`` when the evidence is incomplete."""

    pairs = test_stage.get("pairs") if isinstance(test_stage, Mapping) else None
    if not isinstance(pairs, Sequence) or not pairs:
        return None
    ratios: list[float] = []
    for pair in pairs:
        arms = pair.get("arms") if isinstance(pair, Mapping) else None
        if not isinstance(arms, Mapping):
            return None
        baseline = _series(arms.get("baseline_samples"), "total_ns")
        candidate = _series(arms.get("candidate_samples"), "total_ns")
        if baseline is None or candidate is None:
            return None
        ratios.append(statistics.median(candidate) / statistics.median(baseline))
    return statistics.median(ratios) if ratios else None


def censoring_for(reason: str) -> str:
    """Which kind of censoring a failed run represents.

    ``censored_gate_failed`` has had no caller since it was defined; a readiness
    refusal is exactly the case it was named for.
    """

    lowered = str(reason or "").lower()
    if "readiness" in lowered or "gate" in lowered or "blocked" in lowered:
        return "censored_gate_failed"
    if "timeout" in lowered or "deadline" in lowered:
        return "censored_timeout"
    return "censored_error"


def outcome_for(result: Mapping[str, Any], *, decision_id: str, created_at: str = "") -> OutcomeEvent:
    """The single label for one session result. Never raises on a failed run."""

    if not isinstance(result, Mapping) or result.get("schema") != RESULT_SCHEMA:
        raise RewardError("not a session result")
    reason = str(result.get("reason", ""))
    evidence = result.get("measurement_evidence")
    complete = (
        isinstance(evidence, Mapping)
        and evidence.get("schema") == EVIDENCE_SCHEMA
        and evidence.get("status") == "complete"
    )
    digest = evidence.get("evidence_sha256") if isinstance(evidence, Mapping) else None
    if not isinstance(digest, str) or len(digest) != 64:
        digest = None

    if not complete or not bool(result.get("run_ok")):
        return OutcomeEvent(
            decision_id=decision_id,
            censoring=censoring_for(reason),
            evidence_hash=digest,
            notes=reason[:120],
            created_at=created_at,
        )

    value = ratio_median(evidence.get("test", {}))
    if value is None:
        # Evidence declared itself complete but does not carry a usable series.
        # That is a defect in the run, not an observation about the action.
        return OutcomeEvent(
            decision_id=decision_id,
            censoring="censored_error",
            evidence_hash=digest,
            notes="measurement_evidence_unusable",
            created_at=created_at,
        )
    try:
        return OutcomeEvent(
            decision_id=decision_id,
            censoring="observed",
            reward=value,
            reward_metric=REWARD_METRIC,
            evidence_hash=digest,
            notes=reason[:120],
            created_at=created_at,
        )
    except DecisionError as exc:  # non-finite ratio, guarded upstream but not assumed
        raise RewardError(f"ratio is not a usable reward: {exc}") from exc


__all__ = [
    "EVIDENCE_SCHEMA",
    "RESULT_SCHEMA",
    "REWARD_METRIC",
    "RewardError",
    "censoring_for",
    "outcome_for",
    "ratio_median",
]
