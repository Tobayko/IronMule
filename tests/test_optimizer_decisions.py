"""R0 contract tests: propensity logging, masking, censored rewards."""

from __future__ import annotations

import pytest

from friday_optimizer.candidates import CANDIDATE_IDS, CandidateRegistry
from friday_optimizer.decisions import (
    CENSORING,
    DECISION_SCHEMA,
    OUTCOME_SCHEMA,
    DecisionError,
    DecisionEvent,
    OutcomeEvent,
    SelectionPolicy,
    decide,
    decision_context,
)
from friday_optimizer.fingerprint import (
    EnvironmentFingerprint,
    ExactFingerprint,
    ModelFingerprint,
    WorkloadFingerprint,
)
from friday_optimizer.records import DataPhase, RecordKind


def make_fingerprint(*, mode="interactive"):
    environment = EnvironmentFingerprint(
        chip="M1 Max", gpu="Apple GPU", ram_bytes=32 * 1024**3, cpu_cores=10,
        macos="14.5", mlx="0.32.0", mlx_lm="0.0.13", python="3.12.13", runtime_commit="a" * 64,
    )
    model = ModelFingerprint(
        model_id="mlx-community/gemma-3-4b-it-4bit", revision="r1", manifest="b" * 64,
        architecture="gemma3", quant_bits=4, quant_group_size=64, tokenizer="tok-r1",
    )
    workload = WorkloadFingerprint(
        prompt_family="chat", tokenizer="tok-r1", generator="gen-r1", context_bucket="short",
        batch=1, concurrency=1, max_tokens=64, greedy=True, prompt_logprobs=False,
        power_mode="performance", mode=mode,
    )
    return ExactFingerprint(environment, model, workload)


def test_context_is_bounded_scalar_projection():
    context = decision_context(make_fingerprint())
    assert all(key.startswith(("environment.", "model.", "workload.")) for key in context)
    assert all(value is None or isinstance(value, (str, int, float, bool)) for value in context.values())
    assert context["workload.mode"] == "interactive"


def test_deterministic_policy_defaults_to_baseline_without_hints():
    policy = SelectionPolicy("det-v1")
    event = decide(policy, make_fingerprint())
    assert event.chosen == "baseline"
    assert event.propensity == 1.0
    assert event.seed is None
    assert set(event.candidate_set) <= set(CANDIDATE_IDS)


def test_hint_moves_the_deterministic_choice_but_cannot_widen_the_mask():
    policy = SelectionPolicy("det-v1")
    event = decide(policy, make_fingerprint(), hints=("head_skip_prefill",))
    assert event.chosen == "head_skip_prefill"
    assert event.hints == ("head_skip_prefill",)
    # A throughput-only action stays masked for an interactive workload.
    masked = decide(policy, make_fingerprint(), hints=("throughput_width_4",))
    assert "throughput_width_4" not in masked.candidate_set
    assert masked.chosen == "baseline"


def test_epsilon_greedy_logs_the_exact_probability_of_the_taken_action():
    policy = SelectionPolicy("eps-v1", rule="epsilon_greedy", epsilon=0.5)
    fingerprint = make_fingerprint()
    seen = {}
    for seed in range(40):
        event = decide(policy, fingerprint, hints=("head_skip_prefill",), seed=seed)
        distribution = policy.distribution(event.candidate_set, event.hints)
        assert event.propensity == pytest.approx(distribution[event.chosen])
        seen[event.chosen] = seen.get(event.chosen, 0) + 1
    assert len(seen) > 1, "an exploring policy must actually explore"
    assert sum(policy.distribution(
        decide(policy, fingerprint, seed=0).candidate_set, ()).values()) == pytest.approx(1.0)


def test_stochastic_decisions_are_reproducible_from_the_logged_seed():
    policy = SelectionPolicy("eps-v1", rule="epsilon_greedy", epsilon=0.3)
    first = decide(policy, make_fingerprint(), seed=1234)
    second = decide(policy, make_fingerprint(), seed=1234)
    assert (first.chosen, first.propensity, first.decision_id) == (second.chosen, second.propensity, second.decision_id)


def test_policy_rejects_inconsistent_configuration():
    with pytest.raises(DecisionError):
        SelectionPolicy("bad", rule="epsilon_greedy", epsilon=0.0)
    with pytest.raises(DecisionError):
        SelectionPolicy("bad", rule="deterministic_order", epsilon=0.2)
    with pytest.raises(DecisionError):
        SelectionPolicy("bad", rule="policy_gradient")


def test_decision_rejects_actions_outside_the_sealed_allowlist():
    registry = CandidateRegistry()
    base = decide(SelectionPolicy("det-v1"), make_fingerprint())
    with pytest.raises(DecisionError):
        DecisionEvent(
            decision_id=base.decision_id, fingerprint_hash=base.fingerprint_hash,
            context=base.context, candidate_set=("baseline", "rewrite_kernel"), chosen="rewrite_kernel",
            selection_rule="deterministic_order", propensity=1.0, policy_id="det-v1",
            policy_hash=base.policy_hash, registry_hash=registry.registry_hash,
        )
    with pytest.raises(DecisionError):
        DecisionEvent(
            decision_id=base.decision_id, fingerprint_hash=base.fingerprint_hash,
            context=base.context, candidate_set=("baseline",), chosen="baseline",
            selection_rule="deterministic_order", propensity=0.5, policy_id="det-v1",
            policy_hash=base.policy_hash, registry_hash=registry.registry_hash,
        )


def test_records_are_feature_and_label_phases_of_the_same_decision():
    event = decide(SelectionPolicy("det-v1"), make_fingerprint())
    outcome = OutcomeEvent(event.decision_id, "observed", reward=0.85)
    feature = event.as_record()
    label = outcome.as_record()
    assert feature.kind is RecordKind.SYSTEM and feature.phase is DataPhase.FEATURE
    assert label.kind is RecordKind.SYSTEM and label.phase is DataPhase.LABEL
    assert feature.payload["schema"] == DECISION_SCHEMA
    assert label.payload["schema"] == OUTCOME_SCHEMA
    assert DecisionEvent.from_payload(feature.payload) == event
    assert OutcomeEvent.from_payload(label.payload) == outcome


def test_censored_outcomes_carry_no_reward_but_stay_representable():
    for status in CENSORING:
        if status == "observed":
            continue
        outcome = OutcomeEvent("d1", status)
        assert outcome.reward is None and not outcome.observed
        with pytest.raises(DecisionError):
            OutcomeEvent("d1", status, reward=0.9)
    with pytest.raises(DecisionError):
        OutcomeEvent("d1", "observed")


def test_decision_survives_the_memory_round_trip(tmp_path):
    from friday_optimizer.memory import OptimizationMemoryV2

    event = decide(SelectionPolicy("det-v1"), make_fingerprint(), hints=("fixed_compiled_cache",))
    outcome = OutcomeEvent(event.decision_id, "observed", reward=0.93, notes="offline replay seed")
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        memory.append(event.as_record())
        memory.append(outcome.as_record())
        assert memory.verify_chain()
        rows = memory.list(kind=RecordKind.SYSTEM)
    assert len(rows) == 2
