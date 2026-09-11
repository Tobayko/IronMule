"""Every way the local controller can be wrong, and the reference it falls back to each time.

The controller is allowed to recommend a candidate on exactly one condition: this machine
measured it, repeatedly, under gates that passed. These check the other cases -- corrupt
evidence, a foreign machine, a different model, a different `mlx`, a control too noisy to
read, an interval that touches `1.0`, one wild session, a lost or damaged state file -- and
that every one of them ends at the reference.

They also check the part that is structural rather than behavioural: `ExecutionRouter.decide`
cannot read the controller, so a recommendation cannot change a route however wrong it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ironmule.local_learner import (AA_MAX_SPREAD, CANDIDATE, CANDIDATE_QUALIFIED, COLLECTING,
                                    Evidence, INCLUSION_RULE, IntakeContext, LocalLearner,
                                    LocalLearningError, MIN_SESSIONS, REFERENCE,
                                    REFERENCE_ONLY, SCHEMA, UNKNOWN, VALID, evidence_from)
from ironmule.router import ExecutionRouter, RouteDecision

CONTEXT = IntakeContext("fp", "applegpu_g13s", "0.32.0", "0.31.3", "model", "sha", "rev")
ACTION, CLASS = "k3840_geometry_sg4_r8", "single_short"


def row(index: int, ratio: float = 0.96, **overrides) -> Evidence:
    base = {"evidence_id": f"e{index}", "action_id": ACTION, "workload_class": CLASS,
            "reference_stack": "A", "hardware_fingerprint": "fp",
            "gpu_architecture": "applegpu_g13s", "mlx": "0.32.0", "mlx_lm": "0.31.3",
            "model_id": "model", "model_identity_sha256": "sha", "model_revision": "rev",
            "ratio": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01,
            "within_session_se": 0.005, "aa_median": 1.0, "aa_half_width": 0.005,
            "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
            "measured_at": f"2026-09-11T{index:02d}:00:00+00:00"}
    return evidence_from({**base, **overrides})


def qualified_learner(count: int = 6) -> LocalLearner:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([row(i, 0.955 + 0.002 * i) for i in range(count)])
    return learner


# --------------------------------------------------------------------------- the happy path


def test_a_fresh_install_is_unknown_and_serves_the_reference() -> None:
    learner = LocalLearner(CONTEXT)
    recommendation = learner.recommendation(ACTION, CLASS)
    assert recommendation.local_learning_state == UNKNOWN
    assert recommendation.recommended_action == REFERENCE
    assert recommendation.evidence_count == 0
    assert learner.shadow_for(CLASS) is None


def test_the_candidate_needs_the_preregistered_minimum_of_sessions() -> None:
    learner = LocalLearner(CONTEXT)
    for index in range(MIN_SESSIONS - 1):
        learner.observe(row(index, 0.96))
        state = learner.recommendation(ACTION, CLASS)
        assert state.local_learning_state == COLLECTING
        assert state.recommended_action == REFERENCE, (
            f"{index + 1} sessions is below {MIN_SESSIONS} and must not qualify anything")
    learner.observe(row(MIN_SESSIONS, 0.961))
    assert learner.recommendation(ACTION, CLASS).local_learning_state == CANDIDATE_QUALIFIED


def test_a_qualified_action_keeps_its_spread_and_never_one_number() -> None:
    stored = qualified_learner().as_dict()
    action = stored["actions"][0]
    assert action["action_preference"]["preferred"] == CANDIDATE
    distribution = action["expected_gain_distribution"]
    assert distribution["running_mean"]["observed_min"] < distribution["running_mean"]["observed_max"]
    assert distribution["bayesian"]["uncertainty"] > 0.0
    assert "gain" not in action["action_preference"], (
        "a preference must not carry a magnitude: B76 measured a stable sign with a "
        "magnitude that moves")


def test_evidence_that_disagrees_with_itself_qualifies_nothing() -> None:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([row(i, r) for i, r in enumerate([0.90, 1.10, 0.95, 1.05, 0.98])])
    state = learner.recommendation(ACTION, CLASS)
    assert state.recommended_action == REFERENCE
    assert state.confidence == "uncertain"


def test_a_measurably_slower_candidate_is_reference_only() -> None:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([row(i, 1.05 + 0.002 * i) for i in range(5)])
    state = learner.recommendation(ACTION, CLASS)
    assert state.local_learning_state == REFERENCE_ONLY
    assert state.recommended_action == REFERENCE


# --------------------------------------------------------------------------- fail closed


@pytest.mark.parametrize("field,value,reason", [
    ("hardware_fingerprint", "other", "hardware fingerprint"),
    ("model_identity_sha256", "other", "model identity"),
    ("model_revision", "other", "model revision"),
    ("mlx", "0.33.0", "mlx build"),
    ("mlx_lm", "0.32.0", "mlx_lm build"),
    ("gpu_architecture", "applegpu_g16", "gpu architecture"),
])
def test_foreign_evidence_is_refused_with_the_axis_that_failed(field, value, reason) -> None:
    learner = LocalLearner(CONTEXT)
    for index in range(MIN_SESSIONS + 2):
        outcome = learner.observe(row(index, 0.96, **{field: value}))
        assert outcome["accepted"] is False
        assert outcome["reason"] == reason
    assert learner.recommendation(ACTION, CLASS).recommended_action == REFERENCE


@pytest.mark.parametrize("field,value,reason", [
    ("status", "BLOCKED", "evidence status is BLOCKED, not VALID"),
    ("status", "INVALID", "evidence status is INVALID, not VALID"),
    ("correctness_passed", False, "correctness did not pass"),
    ("resource_gate_passed", False, "the B65 resource gate did not pass"),
])
def test_evidence_that_failed_its_own_gates_is_never_learned(field, value, reason) -> None:
    learner = LocalLearner(CONTEXT)
    for index in range(MIN_SESSIONS + 2):
        assert learner.observe(row(index, 0.80, **{field: value}))["reason"] == reason
    assert learner.recommendation(ACTION, CLASS).local_learning_state == UNKNOWN


@pytest.mark.parametrize("payload", [
    "not a mapping",
    {"evidence_id": "e"},
    None,
])
def test_corrupt_evidence_is_refused_before_it_becomes_evidence(payload) -> None:
    with pytest.raises(LocalLearningError):
        evidence_from(payload)


def test_an_unknown_field_is_a_refusal_and_not_a_warning() -> None:
    good = row(1).as_dict()
    with pytest.raises(LocalLearningError, match="unknown"):
        evidence_from({**good, "surprise": 1})


@pytest.mark.parametrize("field,value", [
    ("ratio", float("nan")), ("ratio", float("inf")), ("ratio", -1.0), ("ratio", 0.0),
    ("ci_low", "0.9"), ("correctness_passed", "yes"), ("evidence_id", ""),
    ("within_session_se", float("nan")),
])
def test_a_number_that_is_not_a_number_is_refused(field, value) -> None:
    good = row(1).as_dict()
    with pytest.raises(LocalLearningError):
        evidence_from({**good, field: value})


def test_a_ratio_outside_its_own_interval_is_refused() -> None:
    good = row(1).as_dict()
    with pytest.raises(LocalLearningError, match="inside its own interval"):
        evidence_from({**good, "ratio": 0.5})


def test_the_same_evidence_is_never_counted_twice() -> None:
    learner = LocalLearner(CONTEXT)
    learner.observe(row(1, 0.96))
    repeated = learner.observe(row(1, 0.96))
    assert repeated["accepted"] is False
    assert "immutable" in repeated["reason"]
    assert learner.recommendation(ACTION, CLASS).evidence_count == 1


def test_an_unidentified_installation_learns_nothing() -> None:
    learner = LocalLearner(IntakeContext())
    outcome = learner.observe(row(1, 0.96))
    assert outcome["accepted"] is False
    assert "does not identify" in outcome["reason"]


# --------------------------------------------------------------------------- noise


def test_a_control_too_wide_to_read_never_sets_a_scale() -> None:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([row(i, 0.80, aa_half_width=AA_MAX_SPREAD * 4)
                         for i in range(MIN_SESSIONS + 3)])
    state = learner.recommendation(ACTION, CLASS)
    assert state.recommended_action == REFERENCE
    assert "A/A control" in state.reason


def test_a_control_sitting_off_one_is_refused_as_a_scale() -> None:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([row(i, 0.90, aa_median=1.25) for i in range(MIN_SESSIONS + 2)])
    assert learner.recommendation(ACTION, CLASS).recommended_action == REFERENCE


def test_a_noisy_session_still_counts_towards_the_sign() -> None:
    """B76 session 0 is the real case: valid evidence, a control too wide to set a scale."""
    learner = LocalLearner(CONTEXT)
    learner.observe(row(0, 0.955, aa_half_width=0.0754))
    learner.observe_all([row(i, 0.955 + 0.002 * i) for i in (1, 2, 3)])
    action = learner.as_dict()["actions"][0]
    assert "e0" in action["evidence_ids"]
    assert "e0" in action["expected_gain_distribution"]["excluded_for_a_noisy_control"]
    assert action["expected_gain_distribution"]["n"] == 3
    assert action["action_preference"]["sessions_accepted"] == 4


def test_one_wild_session_pulls_the_interval_back_over_the_boundary() -> None:
    learner = qualified_learner()
    assert learner.recommendation(ACTION, CLASS).recommended_action == CANDIDATE
    learner.observe(row(99, 1.30))
    state = learner.recommendation(ACTION, CLASS)
    assert state.recommended_action == REFERENCE, (
        "a single extreme session widens the spread, and a widened interval that touches "
        "1.0 must stop recommending anything")
    assert state.prediction_interval[1] >= 1.0


def test_an_interval_that_touches_the_boundary_decides_nothing() -> None:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([row(i, r) for i, r in enumerate([0.97, 0.99, 1.01, 0.98])])
    state = learner.recommendation(ACTION, CLASS)
    assert state.local_learning_state == COLLECTING
    assert state.recommended_action == REFERENCE
    assert state.prediction_interval[0] < 1.0 < state.prediction_interval[1]


def test_both_estimators_must_clear_the_boundary() -> None:
    from ironmule.local_learner import bayesian, running_mean

    ratios = [0.955 + 0.002 * i for i in range(6)]
    mean = running_mean(ratios)
    bayes = bayesian(ratios, [0.005] * len(ratios))
    assert mean["predictive_interval"][1] < 1.0 and bayes["predictive_interval"][1] < 1.0
    assert running_mean(ratios[:1]) is None and bayesian(ratios[:1], [0.005]) is None


# --------------------------------------------------------------------------- persistence


def test_state_survives_a_restart(tmp_path: Path) -> None:
    learner = qualified_learner()
    path = learner.save(tmp_path / "local_learning.json")
    evidence = [row(i, 0.955 + 0.002 * i) for i in range(6)]
    restored = LocalLearner.restore(path, CONTEXT, evidence)
    assert (restored.recommendation(ACTION, CLASS).as_dict()
            == learner.recommendation(ACTION, CLASS).as_dict())
    assert restored.as_dict()["digest"] == learner.as_dict()["digest"]


def test_a_missing_state_file_leaves_a_learner_that_knows_nothing(tmp_path: Path) -> None:
    restored = LocalLearner.restore(tmp_path / "absent.json", CONTEXT)
    assert restored.recommendation(ACTION, CLASS).local_learning_state == UNKNOWN
    assert restored.recommendation(ACTION, CLASS).recommended_action == REFERENCE


@pytest.mark.parametrize("content", [
    "{not json",
    "[]",
    json.dumps({"schema": "something.else", "actions": [], "digest": "x"}),
])
def test_a_state_file_that_cannot_be_understood_is_not_partially_believed(
        tmp_path: Path, content: str) -> None:
    path = tmp_path / "local_learning.json"
    path.write_text(content, encoding="utf-8")
    restored = LocalLearner.restore(path, CONTEXT)
    assert restored.recommendation(ACTION, CLASS).recommended_action == REFERENCE


def test_a_tampered_digest_is_refused(tmp_path: Path) -> None:
    path = qualified_learner().save(tmp_path / "local_learning.json")
    stored = json.loads(path.read_text())
    stored["last_valid_update"] = "2099-01-01T00:00:00+00:00"
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert LocalLearner.restore(path, CONTEXT).recommendation(
        ACTION, CLASS).recommended_action == REFERENCE


def test_state_written_for_another_machine_is_refused(tmp_path: Path) -> None:
    path = qualified_learner().save(tmp_path / "local_learning.json")
    foreign = IntakeContext("other", "applegpu_g13s", "0.32.0", "0.31.3", "model", "sha", "rev")
    restored = LocalLearner.restore(path, foreign)
    assert restored.recommendation(ACTION, CLASS).recommended_action == REFERENCE


def test_a_saved_state_names_itself_and_claims_no_activation(tmp_path: Path) -> None:
    stored = json.loads(qualified_learner().save(tmp_path / "s.json").read_text())
    assert stored["schema"] == SCHEMA
    assert stored["shadow_only"] is True and stored["activation"] == "none"
    assert stored["last_valid_update"]
    assert INCLUSION_RULE in json.dumps(stored)


# --------------------------------------------------------------------------- shadow only


def test_decide_cannot_read_the_controller() -> None:
    """Structural, not a promise: the method does not mention it."""
    import inspect

    source = inspect.getsource(ExecutionRouter.decide)
    assert "local_learner" not in source and "local_learning" not in source


def test_a_qualified_controller_changes_no_route() -> None:
    from ironmule.plans import StrictOneShotPlan
    from ironmule.service import Request

    requests = [Request(prompt_ids=[1, 2, 3], max_tokens=16, plan=StrictOneShotPlan())]
    plain = ExecutionRouter(None).decide(requests)
    routed = ExecutionRouter(None, local_learner=qualified_learner()).decide(requests)
    assert (plain.route, plain.reason) == (routed.route, routed.reason)
    assert routed.local_learning is None, "decide() attaches nothing"


def test_the_recommendation_arrives_only_as_annotation() -> None:
    from ironmule.plans import StrictOneShotPlan
    from ironmule.service import Request

    router = ExecutionRouter(None, local_learner=qualified_learner())
    requests = [Request(prompt_ids=[1, 2, 3], max_tokens=16, plan=StrictOneShotPlan())]
    decision = router.annotate(router.decide(requests))
    assert decision.workload_class == CLASS
    assert decision.local_learning["recommended_action"] == CANDIDATE
    assert decision.local_learning["shadow_only"] is True
    assert decision.local_learning["affects_dispatch"] is False
    assert decision.route == "reference", "the route is what it always was"


def test_an_unnamed_workload_class_is_annotated_with_nothing() -> None:
    from ironmule.plans import StrictOneShotPlan
    from ironmule.service import Request

    router = ExecutionRouter(None, local_learner=qualified_learner())
    requests = [Request(prompt_ids=[1], max_tokens=16, plan=StrictOneShotPlan()),
                Request(prompt_ids=[2], max_tokens=16, plan=StrictOneShotPlan())]
    decision = router.annotate(router.decide(requests))
    assert decision.workload_class == ""
    assert decision.local_learning is None


def test_a_router_without_a_controller_is_exactly_as_it_was() -> None:
    from ironmule.plans import StrictOneShotPlan
    from ironmule.service import Request

    router = ExecutionRouter(None)
    requests = [Request(prompt_ids=[1, 2], max_tokens=16, plan=StrictOneShotPlan())]
    decision = router.annotate(router.decide(requests))
    assert decision.local_learning is None
    assert decision.as_dict()["local_learning"] is None


def test_the_shadow_entry_is_prebuilt_and_not_computed_per_dispatch() -> None:
    learner = qualified_learner()
    first = learner.shadow_for(CLASS)
    assert first is learner.shadow_for(CLASS), (
        "the annotation is a stored dictionary, not one built on each lookup")
