"""R1 contract tests: replay masking, censoring, off-policy estimators."""

from __future__ import annotations

import math

import pytest

from friday_optimizer.decisions import DecisionEvent, OutcomeEvent, SelectionPolicy, decide
from friday_optimizer.replay import (
    DEFAULT_MIN_SAMPLES,
    ReplayEnv,
    ReplayError,
    ReplayStep,
    default_reward,
    doubly_robust,
    effective_sample_size,
    evaluate,
    ips,
    load_steps,
    replayer,
    snips,
)
from tests.test_optimizer_decisions import make_fingerprint


LOGGING = SelectionPolicy("log-v1", rule="epsilon_greedy", epsilon=0.9)
TARGET = SelectionPolicy("det-v1")


def make_steps(count=60, *, censored_every=0, reward=0.8):
    steps = []
    for index in range(count):
        decision = decide(
            LOGGING, make_fingerprint(), hints=("head_skip_prefill",), seed=index,
            decision_id=f"d{index:04d}",
        )
        if censored_every and index % censored_every == 0:
            outcome = OutcomeEvent(decision.decision_id, "censored_timeout")
        else:
            outcome = OutcomeEvent(decision.decision_id, "observed", reward=reward)
        steps.append(ReplayStep(decision, outcome))
    return steps


def test_env_enforces_the_logged_action_mask():
    env = ReplayEnv(make_steps(3))
    observation = env.reset()
    assert observation is not None and "baseline" in observation.action_mask
    with pytest.raises(ReplayError):
        env.step("throughput_width_4")
    transition = env.step(observation.action_mask[0])
    assert transition.logged_action in observation.action_mask


def test_counterfactual_actions_have_no_imputed_reward():
    steps = make_steps(2)
    env = ReplayEnv(steps)
    env.reset()
    logged = steps[0].decision.chosen
    other = next(action for action in steps[0].action_mask if action != logged)
    transition = env.step(other)
    assert transition.matched is False and transition.reward is None
    matched = env.step(steps[1].decision.chosen)
    assert matched.matched is True and matched.reward == pytest.approx(1.0 - 0.8)


def test_censored_runs_are_kept_with_a_conservative_reward():
    steps = make_steps(4, censored_every=2)
    env = ReplayEnv(steps, censored_reward=0.0)
    censored = [step for step in steps if not step.observed]
    assert censored, "fixture must contain censored steps"
    assert env.reward_of(censored[0]) == 0.0
    assert len(env) == len(steps), "a censored run is information, never a drop"
    with pytest.raises(ReplayError):
        ReplayEnv(steps, censored_reward=0.5)


def test_replay_is_exhaustible_and_resettable():
    env = ReplayEnv(make_steps(2))
    env.reset()
    env.step(env.action_mask()[0])
    last = env.step(env.action_mask()[0])
    assert last.done is True
    with pytest.raises(ReplayError):
        env.step("baseline")
    assert env.reset() is not None


def test_ips_recovers_a_constant_reward_under_full_overlap():
    env = ReplayEnv(make_steps(200))
    estimate = ips(env, TARGET, min_samples=DEFAULT_MIN_SAMPLES, seed=7, resamples=200)
    assert estimate.status == "ok" and estimate.conclusive
    # Every action pays the same reward, so any reweighting must return it.
    assert estimate.value == pytest.approx(1.0 - 0.8, rel=0.25)
    assert estimate.ci_low is not None and estimate.ci_low <= estimate.value <= estimate.ci_high


def test_snips_is_stabler_than_ips_on_the_same_sample():
    env = ReplayEnv(make_steps(200))
    plain = ips(env, TARGET, seed=7, resamples=200)
    normalised = snips(env, TARGET, seed=7, resamples=200)
    assert normalised.value == pytest.approx(1.0 - 0.8, rel=0.02)
    assert abs(normalised.value - 0.2) <= abs(plain.value - 0.2) + 1e-9


def test_small_corpus_is_reported_as_insufficient_never_as_a_result():
    env = ReplayEnv(make_steps(2))
    for estimate in evaluate(env, TARGET, min_samples=DEFAULT_MIN_SAMPLES, resamples=50).values():
        assert estimate.status in ("insufficient_data", "no_overlap", "no_labels")
        assert estimate.conclusive is False


def test_no_labels_and_no_overlap_are_distinct_statuses():
    unlabelled = [ReplayStep(step.decision) for step in make_steps(4)]
    env = ReplayEnv(unlabelled)
    assert len(env) == 0
    assert ips(env, TARGET, resamples=50).status == "no_labels"

    steps = make_steps(4)
    forced = [
        ReplayStep(
            DecisionEvent(
                decision_id=step.decision.decision_id, fingerprint_hash=step.decision.fingerprint_hash,
                context=step.decision.context, candidate_set=step.decision.candidate_set,
                chosen=next(a for a in step.decision.candidate_set if a != "baseline"),
                selection_rule="epsilon_greedy", propensity=0.1, policy_id="log-v1",
                policy_hash=step.decision.policy_hash, registry_hash=step.decision.registry_hash,
                hints=(), seed=1,
            ),
            step.outcome,
        )
        for step in steps
    ]
    # The deterministic target puts all mass on baseline, so no logged action
    # overlaps and the estimator must refuse rather than extrapolate.
    assert ips(ReplayEnv(forced), TARGET, resamples=50).status == "no_overlap"


def test_doubly_robust_matches_ips_when_the_model_predicts_zero():
    env = ReplayEnv(make_steps(200))
    plain = ips(env, TARGET, seed=3, resamples=200)
    robust = doubly_robust(env, TARGET, lambda context, action: 0.0, seed=3, resamples=200)
    assert robust.value == pytest.approx(plain.value)
    perfect = doubly_robust(env, TARGET, lambda context, action: 0.2, seed=3, resamples=200)
    assert perfect.value == pytest.approx(0.2, rel=1e-6)
    with pytest.raises(ReplayError):
        doubly_robust(env, TARGET, lambda context, action: float("nan"), resamples=50)


def test_replayer_discards_mismatches_and_reports_its_own_sample_size():
    env = ReplayEnv(make_steps(200))
    estimate = replayer(env, TARGET, resamples=200)
    assert 0 < estimate.samples < 200
    assert estimate.value == pytest.approx(0.2)


def test_effective_sample_size_penalises_concentrated_weights():
    assert effective_sample_size([1.0] * 10) == pytest.approx(10.0)
    assert effective_sample_size([10.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_sample_size([0.0, 0.0]) == 0.0


def test_load_steps_pairs_records_and_rejects_orphans(tmp_path):
    from friday_optimizer.memory import OptimizationMemoryV2

    steps = make_steps(3, censored_every=3)
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        for step in steps:
            memory.append(step.decision.as_record())
            memory.append(step.outcome.as_record())
        loaded = load_steps(memory)
    assert [step.decision.decision_id for step in loaded] == [s.decision.decision_id for s in steps]
    assert all(step.labelled for step in loaded)

    orphan = [steps[0].outcome.payload()]
    with pytest.raises(ReplayError):
        load_steps(orphan)


def test_default_reward_is_the_relative_gain_and_rejects_censored_input():
    assert default_reward(OutcomeEvent("d1", "observed", reward=0.846)) == pytest.approx(0.154)
    with pytest.raises(ReplayError):
        default_reward(OutcomeEvent("d1", "censored_error"))


def test_target_hints_price_a_policy_that_would_have_hinted_differently():
    steps = make_steps(200)
    env = ReplayEnv(steps)
    # Default: the target is scored under the logged hints, so it concentrates
    # on the same action the logger favoured.
    logged = ips(env, TARGET, resamples=200)
    assert logged.status == "ok"
    # Overridden: the same policy object now concentrates on another action,
    # which the logger drew rarely, so overlap and effective size fall.
    other = ips(env, TARGET, resamples=200, target_hints=("fixed_compiled_cache",))
    assert other.effective_samples < logged.effective_samples
    assert other.value is not None


def test_target_hints_do_not_change_the_corpus():
    steps = make_steps(60)
    env = ReplayEnv(steps)
    before = [(step.decision.chosen, step.decision.hints, step.decision.propensity) for step in env.steps]
    ips(env, TARGET, resamples=100, target_hints=("persistent_process",))
    after = [(step.decision.chosen, step.decision.hints, step.decision.propensity) for step in env.steps]
    assert before == after


def test_every_estimator_accepts_the_same_target_hints():
    env = ReplayEnv(make_steps(120))
    hints = ("persistent_process",)
    results = evaluate(env, TARGET, min_samples=DEFAULT_MIN_SAMPLES, resamples=100, target_hints=hints)
    direct = snips(env, TARGET, resamples=100, target_hints=hints)
    assert results["snips"].value == pytest.approx(direct.value)
    assert set(results) == {"ips", "snips", "replayer"}
