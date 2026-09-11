"""The line between watching and learning, and every way the watcher must stay quiet or speak.

`B80`'s whole risk is that a monitor built from ordinary dispatches starts believing them. A
candidate request that came back quickly says nothing about the reference, so no number of
them may qualify anything. These check that the separation is structural rather than a habit,
that stable use costs a user nothing, and that a real shift is caught and ends at the
reference.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ironmule import monitoring as monitoring_module
from ironmule.activation import ActivationContext, LearnedDispatchActivation, REFERENCE
from ironmule.local_learner import CANDIDATE_QUALIFIED, IntakeContext, LocalLearner, VALID
from ironmule.local_learner import evidence_from
from ironmule.monitoring import (BASELINE_MIN, DriftMonitor, MONITORING, MonitoringError,
                                 Observation, REQUALIFICATION_REQUIRED, SEGMENT_FIELDS,
                                 WARMING_UP, WINDOW, observation_from, requalification_path)
from ironmule.qmv_variant import QUALIFIED_ACTION_ID

SOURCE = Path(monitoring_module.__file__)
CLASS = "single_short"
MATERIAL = 0.0226


def raw(index: int, milliseconds: float, **overrides) -> dict:
    base = {"observed_at": f"2026-09-11T00:{index % 60:02d}:00+00:00",
            "hardware_fingerprint": "fp", "gpu_architecture": "applegpu_g13s",
            "model_identity_sha256": "sha", "model_revision": "rev",
            "quantization_bits": 4, "quantization_group_size": 64,
            "mlx": "0.32.0", "mlx_lm": "0.31.3", "workload_class": CLASS,
            "action_id": QUALIFIED_ACTION_ID, "action_code_digest": "digest",
            "effective_action": "candidate", "end_to_end_ms": milliseconds,
            "service_ttft_ms": 20.0, "tokens_per_second": 33.0, "new_tokens": 32,
            "prompt_tokens": 27, "fallbacks": 0, "correctness_errors": 0,
            "memory_pressure_level": 1, "swap_used_bytes": 0, "controller_digest": "c"}
    return {**base, **overrides}


def primed(monitor: DriftMonitor | None = None, centre: float = 921.0,
           count: int = BASELINE_MIN) -> DriftMonitor:
    monitor = monitor or DriftMonitor(min_material_shift=MATERIAL)
    for index in range(count):
        monitor.observe(observation_from(raw(index, centre + (index % 4))))
    return monitor


# --------------------------------------------------------------------------- separation


def test_the_monitor_cannot_reach_the_controller_at_all() -> None:
    """Structural. The docstring may name it; the code may not."""
    text = SOURCE.read_text()
    tree = ast.parse(text)
    body = ast.get_source_segment(text, tree.body[-1]) or ""
    code = text[text.index(body):] if body else text
    assert "LocalLearner" not in code
    assert "Evidence" not in code
    assert not any(module and "local_learner" in module
                   for module in (node.module for node in tree.body
                                  if isinstance(node, ast.ImportFrom)))


def test_an_observation_carries_no_ratio_and_no_counterfactual() -> None:
    fields = set(monitoring_module.OBSERVATION_FIELDS)
    for forbidden in ("ratio", "ci_low", "ci_high", "reference_ms", "candidate_ms"):
        assert forbidden not in fields, f"{forbidden} would make an observation comparative"


def test_a_thousand_fast_candidate_dispatches_qualify_nothing() -> None:
    learner = LocalLearner(IntakeContext("fp", "applegpu_g13s", "0.32.0", "0.31.3",
                                         "model", "sha", "rev"))
    monitor = DriftMonitor(min_material_shift=MATERIAL)
    for index in range(1000):
        monitor.observe(observation_from(raw(index, 500.0)))
    assert learner.recommendation(QUALIFIED_ACTION_ID, CLASS).recommended_action == REFERENCE
    assert learner.recommendation(QUALIFIED_ACTION_ID, CLASS).evidence_count == 0


# --------------------------------------------------------------------------- intake


@pytest.mark.parametrize("payload", ["not a mapping", None, {"observed_at": "x"}])
def test_a_record_that_is_not_an_observation_is_refused(payload) -> None:
    with pytest.raises(MonitoringError):
        observation_from(payload)


def test_an_unknown_field_is_a_refusal() -> None:
    with pytest.raises(MonitoringError, match="unknown"):
        observation_from({**raw(1, 900.0), "surprise": 1})


@pytest.mark.parametrize("field,value", [
    ("end_to_end_ms", 0.0), ("end_to_end_ms", -1.0), ("end_to_end_ms", float("nan")),
    ("new_tokens", -1), ("fallbacks", True), ("workload_class", ""),
])
def test_a_number_that_is_not_a_number_is_refused(field, value) -> None:
    with pytest.raises(MonitoringError):
        observation_from({**raw(1, 900.0), field: value})


def test_a_refused_record_is_recorded_and_changes_nothing() -> None:
    monitor = primed()
    outcome = monitor.observe_raw({"nonsense": True})
    assert outcome["accepted"] is False
    assert monitor.state_for(CLASS, QUALIFIED_ACTION_ID) == MONITORING


# --------------------------------------------------------------------------- the rule


def test_a_baseline_needs_the_preregistered_minimum() -> None:
    monitor = DriftMonitor(min_material_shift=MATERIAL)
    for index in range(BASELINE_MIN - 1):
        monitor.observe(observation_from(raw(index, 921.0)))
        assert monitor.state_for(CLASS, QUALIFIED_ACTION_ID) == WARMING_UP
    monitor.observe(observation_from(raw(99, 921.0)))
    assert monitor.state_for(CLASS, QUALIFIED_ACTION_ID) == MONITORING


def test_stable_use_never_costs_a_user_the_gain() -> None:
    monitor = primed()
    for index in range(200):
        monitor.observe(observation_from(raw(index, 921.0 + (index % 5))))
    assert not monitor.requalification_required


def test_one_slow_request_decides_nothing() -> None:
    monitor = primed()
    monitor.observe(observation_from(raw(50, 5000.0)))
    assert not monitor.requalification_required


def test_a_sustained_material_shift_is_caught() -> None:
    monitor = primed()
    for index in range(WINDOW):
        monitor.observe(observation_from(raw(60 + index, 921.0 * 1.25)))
    assert monitor.requalification_required
    assert monitor.state_for(CLASS, QUALIFIED_ACTION_ID) == REQUALIFICATION_REQUIRED


def test_a_shift_smaller_than_the_gain_is_not_worth_catching() -> None:
    """One per cent is statistically obvious here and still leaves the action winning."""
    monitor = primed()
    for index in range(3 * WINDOW):
        monitor.observe(observation_from(raw(60 + index, 921.0 * 1.01)))
    assert not monitor.requalification_required


def test_a_machine_that_became_erratic_has_drifted_at_the_same_median() -> None:
    monitor = primed()
    for index in range(3 * WINDOW):
        offset = 120.0 if index % 2 else -120.0
        monitor.observe(observation_from(raw(60 + index, 921.0 + offset)))
    assert monitor.requalification_required


def test_requalification_is_one_way_and_observation_cannot_undo_it() -> None:
    monitor = primed()
    for index in range(WINDOW):
        monitor.observe(observation_from(raw(60 + index, 921.0 * 1.25)))
    assert monitor.requalification_required
    for index in range(200):
        monitor.observe(observation_from(raw(index, 921.0)))
    assert monitor.requalification_required, (
        "only an explicit requalification producing comparative evidence may clear this")


# --------------------------------------------------------------------------- segments


@pytest.mark.parametrize("field,value", [
    ("mlx", "0.33.0"), ("mlx_lm", "0.32.0"), ("model_revision", "other"),
    ("model_identity_sha256", "other"), ("hardware_fingerprint", "other"),
    ("action_code_digest", "changed"), ("quantization_bits", 8),
    ("gpu_architecture", "applegpu_g16"),
])
def test_a_change_starts_a_new_segment_rather_than_drifting_the_old_one(field, value) -> None:
    monitor = primed()
    monitor.observe(observation_from(raw(99, 921.0, **{field: value})))
    assert len(monitor.segments) == 2
    assert any(segment.superseded for segment in monitor.segments.values())
    assert not monitor.requalification_required
    assert list(monitor.segments.values())[-1].state == WARMING_UP


def test_the_action_that_ran_is_part_of_the_key() -> None:
    assert "effective_action" in SEGMENT_FIELDS
    monitor = primed()
    for index in range(BASELINE_MIN + WINDOW):
        monitor.observe(observation_from(raw(index, 965.0, effective_action="reference")))
    assert not monitor.requalification_required, (
        "a reference dispatch is slower by construction and is not the candidate drifting")
    assert len(monitor.segments) == 2


def test_a_sibling_action_does_not_supersede_the_one_being_watched() -> None:
    monitor = primed()
    monitor.observe(observation_from(raw(99, 965.0, effective_action="reference")))
    watched = [s for s in monitor.segments.values()
               if s.fields["effective_action"] == "candidate"]
    assert watched and not watched[0].superseded


def test_age_alone_deletes_nothing() -> None:
    assert "deletes nothing" in monitoring_module.AGING_RULE
    monitor = primed()
    first = next(iter(monitor.segments.values()))
    for index in range(50):
        monitor.observe(observation_from(
            raw(index, 921.0, observed_at=f"2027-01-01T00:{index % 60:02d}:00+00:00")))
    assert first.baseline is not None and not first.superseded


# --------------------------------------------------------------------------- persistence


def test_state_survives_a_restart(tmp_path: Path) -> None:
    monitor = primed()
    path = monitor.save(tmp_path / "monitoring.json")
    restored = DriftMonitor.restore(path, min_material_shift=MATERIAL)
    assert restored.state_for(CLASS, QUALIFIED_ACTION_ID) == MONITORING
    assert restored.as_dict()["digest"] == monitor.as_dict()["digest"]


def test_a_requalification_survives_a_restart(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    monitor = primed(DriftMonitor(state, min_material_shift=MATERIAL))
    for index in range(WINDOW):
        monitor.observe(observation_from(raw(60 + index, 921.0 * 1.25)))
    assert requalification_path(state).exists()
    restored = DriftMonitor(state, min_material_shift=MATERIAL)
    assert restored.requalification_required


@pytest.mark.parametrize("content", ["{not json", "[]",
                                     json.dumps({"schema": "other", "segments": []})])
def test_a_state_that_cannot_be_understood_is_not_partially_believed(
        tmp_path: Path, content: str) -> None:
    path = tmp_path / "monitoring.json"
    path.write_text(content, encoding="utf-8")
    restored = DriftMonitor.restore(path, min_material_shift=MATERIAL)
    assert restored.segments == {}
    assert restored.state_for(CLASS, QUALIFIED_ACTION_ID) == WARMING_UP


def test_a_tampered_digest_is_refused(tmp_path: Path) -> None:
    path = primed().save(tmp_path / "monitoring.json")
    stored = json.loads(path.read_text())
    stored["segments"][0]["observations"] = 9999
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert DriftMonitor.restore(path, min_material_shift=MATERIAL).segments == {}


def test_an_unreadable_requalification_record_counts_as_required(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    requalification_path(state).write_text("{not json", encoding="utf-8")
    assert DriftMonitor(state).requalification_required


def test_the_state_does_not_grow_with_use(tmp_path: Path) -> None:
    short = primed().save(tmp_path / "short.json").stat().st_size
    monitor = primed()
    for index in range(2000):
        monitor.observe(observation_from(raw(index, 921.0 + (index % 5))))
    long = monitor.save(tmp_path / "long.json").stat().st_size
    assert long < short * 1.5, "a baseline and a window, not the observations themselves"


# --------------------------------------------------------------------------- the runtime


def _qualified_learner() -> LocalLearner:
    intake = IntakeContext("fp", "applegpu_g13s", "0.32.0", "0.31.3", "model", "sha", "rev")
    learner = LocalLearner(intake)
    for index in range(6):
        ratio = 0.955 + 0.002 * index
        learner.observe(evidence_from({
            "evidence_id": f"e{index}", "action_id": QUALIFIED_ACTION_ID,
            "workload_class": CLASS, "reference_stack": "A", "hardware_fingerprint": "fp",
            "gpu_architecture": "applegpu_g13s", "mlx": "0.32.0", "mlx_lm": "0.31.3",
            "model_id": "model", "model_identity_sha256": "sha", "model_revision": "rev",
            "ratio": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01,
            "within_session_se": 0.005, "aa_median": 1.0, "aa_half_width": 0.005,
            "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
            "measured_at": f"2026-09-11T{index:02d}:00:00+00:00"}))
    assert learner.recommendation(QUALIFIED_ACTION_ID, CLASS).local_learning_state \
        == CANDIDATE_QUALIFIED
    return learner


def test_a_requalification_takes_the_candidate_away() -> None:
    context = ActivationContext("fp", "applegpu_g13s", "model", "sha", "rev", 4, 64, 3840,
                                (15360,), "0.32.0", "0.31.3")
    monitor = primed()
    layer = LearnedDispatchActivation(_qualified_learner(), context, enabled=True,
                                      monitor=monitor)
    layer.prepare()
    assert layer.for_dispatch(CLASS, "interactive").effective_action == "candidate"
    for index in range(WINDOW):
        monitor.observe(observation_from(raw(60 + index, 921.0 * 1.25)))
    record = layer.for_dispatch(CLASS, "interactive")
    assert record.effective_action == REFERENCE
    assert record.local_learning_state == REQUALIFICATION_REQUIRED
    assert layer.gate.active is False


def test_the_runtime_still_defaults_to_off() -> None:
    import inspect

    from ironmule.router import AppleRuntime

    assert inspect.signature(AppleRuntime.load).parameters[
        "enable_local_learned_dispatch"].default is False
