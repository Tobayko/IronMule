"""The switch that can finally change a dispatch, and every way it must refuse to.

`B78` could be wrong for free. `B79` cannot: a mistake here runs a kernel on a user's request.
These check the three things that make that safe -- the default is off, admission agrees on
every axis before anything is installed, and there is exactly one direction out of trouble --
and the structural claims that hold them up.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ironmule import activation as activation_module
from ironmule.activation import (ActivationContext, CANDIDATE, LearnedDispatchActivation,
                                 REFERENCE, kill_path, read_kill, write_kill)
from ironmule.local_learner import (CANDIDATE_QUALIFIED, Evidence, IntakeContext, LocalLearner,
                                    VALID, evidence_from)
from ironmule.qmv_variant import (GatedVariantLinear, QUALIFIED_ACTION_ID, VariantGate,
                                  uninstall)
from ironmule.router import AppleRuntime, ExecutionRouter

CLASS = "single_short"
ROUTE = "interactive"
INTAKE = IntakeContext("fp", "applegpu_g13s", "0.32.0", "0.31.3", "model", "sha", "rev")
CONTEXT = ActivationContext("fp", "applegpu_g13s", "model", "sha", "rev", 4, 64, 3840,
                            (15360,), "0.32.0", "0.31.3")


def row(index: int, ratio: float, **overrides) -> Evidence:
    base = {"evidence_id": f"e{index}", "action_id": QUALIFIED_ACTION_ID,
            "workload_class": CLASS, "reference_stack": "A", "hardware_fingerprint": "fp",
            "gpu_architecture": "applegpu_g13s", "mlx": "0.32.0", "mlx_lm": "0.31.3",
            "model_id": "model", "model_identity_sha256": "sha", "model_revision": "rev",
            "ratio": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01,
            "within_session_se": 0.005, "aa_median": 1.0, "aa_half_width": 0.005,
            "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
            "measured_at": f"2026-09-11T{index:02d}:00:00+00:00"}
    return evidence_from({**base, **overrides})


def qualified(count: int = 6) -> LocalLearner:
    learner = LocalLearner(INTAKE)
    learner.observe_all([row(i, 0.955 + 0.002 * i) for i in range(count)])
    assert learner.recommendation(QUALIFIED_ACTION_ID, CLASS).local_learning_state \
        == CANDIDATE_QUALIFIED
    return learner


def prepared(**kwargs) -> LearnedDispatchActivation:
    layer = LearnedDispatchActivation(kwargs.pop("learner", qualified()),
                                      kwargs.pop("context", CONTEXT),
                                      enabled=kwargs.pop("enabled", True), **kwargs)
    layer.prepare()
    return layer


# --------------------------------------------------------------------------- the default


def test_the_default_is_off_in_the_signature_itself() -> None:
    parameter = inspect.signature(AppleRuntime.load).parameters["enable_local_learned_dispatch"]
    assert parameter.default is False


def test_not_opting_in_never_activates_however_good_the_evidence() -> None:
    layer = LearnedDispatchActivation(qualified(), CONTEXT)
    assert layer.enabled is False
    assert layer.admit().admitted is False
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE


def test_opting_in_without_a_controller_activates_nothing() -> None:
    layer = LearnedDispatchActivation(None, CONTEXT, enabled=True)
    assert layer.admit().admitted is False
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE


def test_a_controller_that_has_not_qualified_activates_nothing() -> None:
    learner = LocalLearner(INTAKE)
    learner.observe(row(0, 0.96))
    layer = prepared(learner=learner)
    assert layer.admit().admitted is False
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE


# --------------------------------------------------------------------------- admission


@pytest.mark.parametrize("field,value", [
    ("hardware_fingerprint", "other"), ("model_identity_sha256", "other"),
    ("model_revision", "other"), ("mlx", "0.33.0"), ("mlx_lm", "0.32.0"),
])
def test_every_identity_axis_must_agree_before_anything_is_installed(field, value) -> None:
    from dataclasses import replace

    layer = LearnedDispatchActivation(qualified(), replace(CONTEXT, **{field: value}),
                                      enabled=True)
    admission = layer.admit()
    assert admission.admitted is False
    assert field in admission.reason or "does not match" in admission.reason
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE


@pytest.mark.parametrize("field,value,fragment", [
    ("quantization_bits", 8, "quantisation"),
    ("quantization_group_size", 32, "quantisation"),
    ("hidden_size", 4096, "K is 4096"),
    ("projection_widths", (17,), "whole number of"),
])
def test_shape_and_quantisation_must_agree(field, value, fragment) -> None:
    from dataclasses import replace

    admission = LearnedDispatchActivation(qualified(), replace(CONTEXT, **{field: value}),
                                          enabled=True).admit()
    assert admission.admitted is False and fragment in admission.reason


def test_admission_names_the_axes_it_checked() -> None:
    admission = LearnedDispatchActivation(qualified(), CONTEXT, enabled=True).admit()
    assert admission.admitted is True
    for axis in ("hardware_fingerprint", "model_identity_sha256", "model_revision", "mlx",
                 "mlx_lm", "quantization", "k", "admitted_n", "action_id",
                 "correctness_contract", "local_preference", "controller_digest"):
        assert axis in admission.checked, axis
    assert admission.checked["local_preference"] == CANDIDATE_QUALIFIED


def test_only_the_workload_classes_the_controller_qualified_are_offered() -> None:
    admission = LearnedDispatchActivation(qualified(), CONTEXT, enabled=True).admit()
    assert admission.workload_classes == (CLASS,)


@pytest.mark.parametrize("workload_class", ["single_long", "session_warm", "", "throughput"])
def test_no_other_workload_class_ever_activates(workload_class) -> None:
    layer = prepared()
    assert layer.for_dispatch(workload_class, ROUTE).effective_action == REFERENCE


@pytest.mark.parametrize("route", ["throughput", "reference", ""])
def test_no_other_route_ever_activates(route) -> None:
    layer = prepared()
    assert layer.for_dispatch(CLASS, route).effective_action == REFERENCE


def test_the_qualified_pair_is_the_only_one_that_activates() -> None:
    layer = prepared()
    record = layer.for_dispatch(CLASS, ROUTE)
    assert record.effective_action == CANDIDATE
    assert record.base_action == REFERENCE
    assert layer.gate.active is True


# --------------------------------------------------------------------------- the record


def test_the_decision_record_carries_what_it_must() -> None:
    record = prepared().for_dispatch(CLASS, ROUTE).as_dict()
    for field in ("base_action", "effective_action", "local_learning_state",
                  "evidence_count", "controller_digest", "activation_reason"):
        assert field in record and record[field] not in (None, ""), field
    assert record["default_enabled"] is False


def test_no_magnitude_reaches_the_activation_decision() -> None:
    """Structural: B76 measured a magnitude that moves, so it decides nothing."""
    source = inspect.getsource(activation_module)
    for forbidden in ("estimated_ratio", "prediction_interval", "predicted_ratio"):
        assert forbidden not in source, f"{forbidden} must not decide an activation"


def test_decide_names_neither_the_controller_nor_the_activation() -> None:
    source = inspect.getsource(ExecutionRouter.decide)
    assert "local_learn" not in source and "activation" not in source


def test_a_shut_gate_calls_the_original_module_and_not_a_copy_of_it() -> None:
    source = inspect.getsource(GatedVariantLinear.__call__)
    assert "self._source(x)" in source, (
        "the fallback must be the module that was replaced, not a reconstruction of it")


# --------------------------------------------------------------------------- one way out


def _telemetry(fallbacks: int = 0, correctness_errors: int = 0) -> SimpleNamespace:
    return SimpleNamespace(fallbacks=fallbacks, correctness_errors=correctness_errors)


def test_a_fallback_in_the_candidate_path_kills_activation(tmp_path: Path) -> None:
    layer = prepared(state_path=tmp_path / "local_learning.json")
    record = layer.for_dispatch(CLASS, ROUTE)
    assert record.effective_action == CANDIDATE
    event = layer.after_dispatch(record, _telemetry(fallbacks=1))
    assert event and event["killed"] is True
    assert layer.gate.killed is True
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE
    assert kill_path(tmp_path / "local_learning.json").exists()


def test_a_correctness_error_kills_activation(tmp_path: Path) -> None:
    layer = prepared(state_path=tmp_path / "local_learning.json")
    record = layer.for_dispatch(CLASS, ROUTE)
    assert layer.after_dispatch(record, _telemetry(correctness_errors=1))["killed"] is True
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE


def test_a_clean_dispatch_kills_nothing() -> None:
    layer = prepared()
    record = layer.for_dispatch(CLASS, ROUTE)
    assert layer.after_dispatch(record, _telemetry()) is None
    assert layer.gate.killed is False


def test_a_reference_dispatch_is_never_judged_by_the_candidate_rules() -> None:
    layer = prepared()
    record = layer.for_dispatch("single_long", ROUTE)
    assert layer.after_dispatch(record, _telemetry(fallbacks=3)) is None, (
        "a fallback on the reference path is not the candidate's failure"
    )


def test_a_killed_gate_never_reopens_in_this_process() -> None:
    layer = prepared()
    layer.for_dispatch(CLASS, ROUTE)
    layer.kill("deliberate")
    for _ in range(3):
        assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE
    assert layer.gate.active is False and layer.gate.killed is True


def test_a_persisted_kill_outranks_the_opt_in(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    write_kill(state, "a fallback was recorded earlier")
    layer = LearnedDispatchActivation(qualified(), CONTEXT, enabled=True, state_path=state)
    assert layer.enabled is False
    assert layer.admit().admitted is False
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE


def test_an_unreadable_kill_record_counts_as_killed(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    kill_path(state).write_text("{not json", encoding="utf-8")
    assert read_kill(state) is not None
    layer = LearnedDispatchActivation(qualified(), CONTEXT, enabled=True, state_path=state)
    assert layer.enabled is False


def test_a_controller_that_stops_qualifying_kills_activation(tmp_path: Path) -> None:
    learner = qualified()
    layer = prepared(learner=learner, state_path=tmp_path / "local_learning.json")
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == CANDIDATE
    # New valid local evidence that widens the interval back over the boundary.
    learner.observe(row(99, 1.30))
    assert learner.recommendation(QUALIFIED_ACTION_ID, CLASS).local_learning_state \
        != CANDIDATE_QUALIFIED
    assert layer.for_dispatch(CLASS, ROUTE).effective_action == REFERENCE
    assert layer.gate.killed is True


def test_the_kill_record_says_what_happened_and_survives(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    layer = prepared(state_path=state)
    layer.for_dispatch(CLASS, ROUTE)
    layer.kill("a fallback in the candidate path")
    stored = json.loads(kill_path(state).read_text())
    assert stored["schema"] == activation_module.SCHEMA
    assert "fallback" in stored["reason"]
    assert "until a person removes this file" in stored["effect"]


def test_the_only_direction_out_is_the_reference() -> None:
    """There is no second experimental action anywhere in the layer."""
    source = inspect.getsource(activation_module)
    assert source.count("effective_action=") == 0
    records = [line for line in source.splitlines() if "ActivationRecord(" in line]
    assert records, "the layer builds records"
    assert all("REFERENCE, CANDIDATE" in line or "REFERENCE, REFERENCE" in line
               or "ActivationRecord(" == line.strip().split("=")[-1].strip()
               or "class ActivationRecord" in line or "-> ActivationRecord" in line
               for line in records), records


# --------------------------------------------------------------------------- the gate


def test_the_gate_starts_shut_and_a_kill_is_permanent() -> None:
    gate = VariantGate()
    assert gate.active is False
    gate.open("eligible")
    assert gate.active is True
    gate.kill("anything")
    gate.open("try again")
    assert gate.active is False and gate.killed is True


def test_uninstalling_a_model_that_was_never_installed_is_a_no_op() -> None:
    import mlx.nn as nn

    assert uninstall(nn.Module()) == 0
