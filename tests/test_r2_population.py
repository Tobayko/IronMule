"""Pure arithmetic and reconstruction guards for the corrected R2 evaluator."""

from __future__ import annotations

import pytest

from experiments.r2_campaign.corrected_evaluation import (
    ReconstructionError,
    population_estimate,
)
from friday_optimizer.decisions import DecisionEvent, OutcomeEvent
from friday_optimizer.replay import ReplayStep


def _decision(index: int, action: str, propensity: float) -> DecisionEvent:
    return DecisionEvent(
        decision_id=f"r2-corpus-20260904-01.{index:04d}",
        fingerprint_hash="a" * 64,
        context={},
        candidate_set=("baseline", "head_skip_prefill", "persistent_process", "fixed_compiled_cache", "readback_every_2"),
        chosen=action,
        selection_rule="epsilon_greedy",
        propensity=propensity,
        policy_id="r2-logging-v1",
        policy_hash="b" * 64,
        registry_hash="c" * 64,
        hints=("head_skip_prefill",),
        seed=index,
    )


def _step(index: int, action: str, reward: float, propensity: float) -> ReplayStep:
    decision = _decision(index, action, propensity)
    return ReplayStep(decision, OutcomeEvent(decision.decision_id, "observed", reward=reward))


def test_population_denominator_keeps_omitted_zero_contributions() -> None:
    schedule = {
        0: ("head_skip_prefill", 0.5, 0),
        1: ("head_skip_prefill", 0.5, 1),
        2: ("persistent_process", 0.12, 2),
        3: ("persistent_process", 0.12, 3),
    }
    steps = {0: _step(0, "head_skip_prefill", 0.8, 0.5), 1: _step(1, "head_skip_prefill", 0.9, 0.5)}
    row = population_estimate(steps, schedule, range(4), "head_skip_prefill", resamples=20)

    # Two observed gains are weighted by 1/.5, while two omitted population
    # units contribute exact zero; denominator is four, not observed count two.
    assert row["numerator"] == pytest.approx(0.6)
    assert row["denominator"] == 4
    assert row["ips"] == pytest.approx(0.15)
    assert row["samples"] == 2
    assert row["support_samples"] == 2
    assert row["normalised_diagnostics"]["observed_fraction"] == pytest.approx(0.5)


def test_target_zero_overlap_is_not_invented_for_omitted_action() -> None:
    schedule = {i: ("persistent_process", 0.12, i) for i in range(3)}
    steps = {}
    row = population_estimate(steps, schedule, range(3), "baseline")
    assert row["numerator"] == 0.0
    assert row["denominator"] == 3
    assert row["samples"] == 0
    assert row["support_samples"] == 0
    assert row["snips"] is None


def test_missing_target_supported_record_is_rejected() -> None:
    schedule = {i: ("head_skip_prefill", 0.5, i) for i in range(3)}
    steps = {0: _step(0, "head_skip_prefill", 0.8, 0.5)}
    with pytest.raises(ReconstructionError, match="missing target-supported"):
        population_estimate(steps, schedule, range(3), "head_skip_prefill")


def test_observed_action_and_propensity_must_match_schedule() -> None:
    schedule = {0: ("head_skip_prefill", 0.5, 0)}
    steps = {0: _step(0, "head_skip_prefill", 0.8, 0.4)}
    with pytest.raises(ReconstructionError, match="action/propensity mismatch"):
        population_estimate(steps, schedule, (0,), "head_skip_prefill")


def test_population_indices_and_schedule_propensities_are_strict() -> None:
    with pytest.raises(ReconstructionError, match="indices must be unique"):
        population_estimate(
            {},
            {0: ("persistent_process", 0.12, 0)},
            (0, 0),
            "baseline",
        )
    with pytest.raises(ReconstructionError, match="invalid schedule propensity"):
        population_estimate(
            {},
            {0: ("persistent_process", 0.0, 0)},
            (0,),
            "baseline",
        )


def test_population_split_reports_zero_for_non_target_observations() -> None:
    schedule = {
        0: ("baseline", 0.12, 0),
        1: ("head_skip_prefill", 0.52, 1),
        2: ("fixed_compiled_cache", 0.12, 2),
    }
    steps = {
        0: _step(0, "baseline", 1.0, 0.12),
        1: _step(1, "head_skip_prefill", 0.9, 0.52),
        2: _step(2, "fixed_compiled_cache", 0.98, 0.12),
    }
    row = population_estimate(steps, schedule, range(3), "baseline", resamples=20)
    assert row["ips"] == pytest.approx(0.0)
    assert row["effective_samples"] == pytest.approx(1.0)
    assert row["support_samples"] == 1
    assert row["normalised_diagnostics"]["zero_contribution_fraction"] == pytest.approx(2 / 3)
