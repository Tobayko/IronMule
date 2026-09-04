"""Campaign contract tests: sealed draws, overlap, and honest sizing."""

from __future__ import annotations

import pytest

from friday_optimizer.campaign import (
    BLOCK_SECONDS,
    MEASURED_POINT_SECONDS,
    CampaignError,
    CampaignPlan,
    expected_effective_samples,
    plan_for_target,
    points_for_effective_samples,
)
from friday_optimizer.candidates import CandidateRegistry
from friday_optimizer.decisions import OutcomeEvent, SelectionPolicy
from friday_optimizer.replay import DEFAULT_MIN_SAMPLES, ReplayEnv, ReplayStep, ips
from tests.test_optimizer_decisions import make_fingerprint

HINT = "head_skip_prefill"
LOGGING = SelectionPolicy("log-v1", rule="epsilon_greedy", epsilon=0.5)
GREEDY = SelectionPolicy("target-greedy-v1")


def plan(points=20, seed_base=7):
    return CampaignPlan(campaign_id="test-campaign", policy=LOGGING, seed_base=seed_base,
                        points=points, hints=(HINT,))


def test_a_deterministic_rule_cannot_form_a_campaign():
    # A preregistered single-candidate study has propensity 1.0 everywhere and
    # therefore produces no overlap; refusing it early is the whole point.
    with pytest.raises(CampaignError):
        CampaignPlan(campaign_id="bad", policy=GREEDY, seed_base=1, points=10)


def test_draws_are_reproducible_from_the_seal():
    fingerprint = make_fingerprint()
    first = plan().decisions(fingerprint)
    second = plan().decisions(fingerprint)
    assert [event.chosen for event in first] == [event.chosen for event in second]
    assert [event.seed for event in first] == [event.seed for event in second]
    other = plan(seed_base=8).decisions(fingerprint)
    assert [event.chosen for event in first] != [event.chosen for event in other]
    assert plan().campaign_hash != plan(seed_base=8).campaign_hash


def test_a_campaign_actually_explores_more_than_one_action():
    events = plan(points=40).decisions(make_fingerprint())
    chosen = {event.chosen for event in events}
    assert len(chosen) > 1, "without overlap the corpus can never be evaluated"
    assert all(event.propensity < 1.0 for event in events)
    assert all(event.seed is not None for event in events)


def test_block_capacity_comes_from_the_measured_break_dominated_wall_clock():
    value = plan(points=25)
    assert value.points_per_block == int(BLOCK_SECONDS // MEASURED_POINT_SECONDS) == 10
    assert value.blocks == 3
    assert plan(points=10).blocks == 1


def test_expected_effective_samples_matches_the_replayed_corpus():
    fingerprint = make_fingerprint()
    registry = CandidateRegistry()
    candidates = registry.ordered_ids(fingerprint, historical_hints=(HINT,))
    points = 50
    predicted = expected_effective_samples(
        logging_policy=LOGGING, target_policy=GREEDY, candidates=candidates,
        points=points, hints=(HINT,),
    )
    events = CampaignPlan(campaign_id="sizing", policy=LOGGING, seed_base=7, points=points,
                          hints=(HINT,)).decisions(fingerprint, registry=registry)
    steps = [ReplayStep(event, OutcomeEvent(event.decision_id, "observed", reward=0.9)) for event in events]
    measured = ips(ReplayEnv(steps), GREEDY, resamples=200)
    assert predicted == pytest.approx(30.0)
    assert measured.effective_samples == pytest.approx(predicted, rel=0.15)


def test_sizing_reports_a_rare_target_as_expensive_not_as_impossible():
    fingerprint = make_fingerprint()
    candidates = CandidateRegistry().ordered_ids(fingerprint, historical_hints=(HINT,))
    greedy_points = points_for_effective_samples(
        logging_policy=LOGGING, target_policy=GREEDY, candidates=candidates,
        required=DEFAULT_MIN_SAMPLES, hints=(HINT,),
    )
    rare = SelectionPolicy("target-rare-v1")
    rare_points = points_for_effective_samples(
        logging_policy=LOGGING, target_policy=rare, candidates=candidates,
        required=DEFAULT_MIN_SAMPLES, hints=("fixed_compiled_cache",),
    )
    assert greedy_points == 50
    assert rare_points is not None and rare_points >= greedy_points


def test_no_overlap_is_reported_as_none_rather_than_a_huge_number():
    # A deterministic logging policy puts all mass on one action, so a target
    # preferring any other action can never be scored: no amount of extra
    # measurement creates overlap that the logging rule never produced.
    candidates = ("baseline", "head_skip_prefill")
    deterministic = SelectionPolicy("log-deterministic-v1")
    target = SelectionPolicy("target-other-v1")
    # Logged deterministically on the hinted action; the target would prefer
    # baseline. That corpus can never score the target, however long it runs.
    assert expected_effective_samples(
        logging_policy=deterministic, target_policy=target,
        candidates=candidates, points=1_000, hints=(HINT,), target_hints=(),
    ) == 0.0
    assert points_for_effective_samples(
        logging_policy=deterministic, target_policy=target,
        candidates=candidates, required=DEFAULT_MIN_SAMPLES, hints=(HINT,), target_hints=(),
    ) is None
    assert plan_for_target(
        campaign_id="impossible", logging_policy=deterministic, target_policy=target,
        candidates=candidates, required=DEFAULT_MIN_SAMPLES, seed_base=1,
        hints=(HINT,), target_hints=(),
    ) is None
    # The same campaign with an exploring logger does have overlap.
    assert points_for_effective_samples(
        logging_policy=LOGGING, target_policy=target,
        candidates=candidates, required=DEFAULT_MIN_SAMPLES, hints=(HINT,), target_hints=(),
    ) is not None


def test_plan_for_target_sizes_and_seals_in_one_step():
    fingerprint = make_fingerprint()
    candidates = CandidateRegistry().ordered_ids(fingerprint, historical_hints=(HINT,))
    sized = plan_for_target(
        campaign_id="r2-v1", logging_policy=LOGGING, target_policy=GREEDY,
        candidates=candidates, required=DEFAULT_MIN_SAMPLES, seed_base=3, hints=(HINT,),
    )
    assert sized is not None
    assert sized.points == 50 and sized.blocks == 5
    assert len(sized.campaign_hash) == 64


def test_plan_validates_its_bounds():
    for kwargs in (
        {"points": 0}, {"points": 10_000}, {"seed_base": -1},
    ):
        with pytest.raises(CampaignError):
            CampaignPlan(campaign_id="bad", policy=LOGGING, seed_base=kwargs.get("seed_base", 1),
                         points=kwargs.get("points", 10))
    with pytest.raises(CampaignError):
        CampaignPlan(campaign_id="bad", policy=LOGGING, seed_base=1, points=10, point_seconds=0)
