"""A cross-hardware test must refuse the machine it was trained on, and seal before it looks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "research" / "raw"
FROZEN = RAW / "B73_frozen_model_20260910.json"
ELIGIBILITY = RAW / "B73_eligibility_20260910.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "b73", ROOT / "tools" / "b73_cross_hardware.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_this_machine_is_refused():
    """It is the training machine. A prediction here would be a memory, not a forecast."""
    row = _module().eligibility()
    assert row["is_unknown_machine"] is False
    assert row["hardware_fingerprint"] in row["training_fingerprints"]
    assert "memory, not a forecast" in row["refusal"]


def test_require_eligible_stops_rather_than_continuing():
    module = _module()
    with pytest.raises(SystemExit):
        module.require_eligible()


def test_the_recorded_eligibility_says_not_started():
    record = json.loads(ELIGIBILITY.read_text())
    assert record["verdict"] == "B73_NOT_STARTED"
    assert record["machine"]["is_unknown_machine"] is False


def test_the_model_is_frozen_with_everything_a_prediction_needs():
    frozen = json.loads(FROZEN.read_text())
    assert frozen["schema"] == "ironmule.frozen_cost_model.v1"
    for key in ("coefficients", "feature_mean", "feature_scale", "feature_names",
                "residual_spread", "known_actions", "known_metrics",
                "training_fingerprints", "policy_thresholds"):
        assert frozen[key], key
    assert len(frozen["coefficients"]) == len(frozen["feature_names"]) + 1


def test_the_freeze_records_what_it_cannot_do():
    frozen = json.loads(FROZEN.read_text())
    assert frozen["hardware_block_constant_in_training"] is True
    assert "never seen vary" in frozen["what_this_cannot_do"]


def test_the_decision_is_the_one_b69_confirmed():
    module = _module()
    assert module.DECISION["actions"] == ["reference_stack", "k3840_geometry_sg4_r8"]
    assert module.DECISION["shape"]["k"] == 3840
    assert module.DECISION["shape"]["decode_width"] == 1
    assert set(module.DECISION["workload_classes"]) == {
        "single_short", "single_long", "session_warm"}


def test_a_prediction_on_unknown_hardware_always_abstains_for_the_stated_reason():
    """Not a policy choice: no coefficient on a hardware feature was fitted on variation."""
    module = _module()
    vector = json.loads((RAW / "B71_vector_m1max_20260910_v3.json").read_text())["vector"]
    machine = dict(module.eligibility(), is_unknown_machine=True)
    answer = module.predict(vector, machine)
    for name, row in answer["answers"].items():
        assert row["abstained"], name
        assert "never varied in training" in row["reason"], name


def test_a_deep_probe_may_only_target_a_decision_relevant_feature():
    module = _module()
    assert "geometry_4_8_ratio" in module.DECISION_RELEVANT_FEATURES
    assert "k_unaligned_to_aligned_ratio" not in module.DECISION_RELEVANT_FEATURES


def test_the_verdict_step_refuses_a_prediction_written_after_the_truth(tmp_path):
    module = _module()
    truth = tmp_path / "truth.json"
    prediction = tmp_path / "prediction.json"
    truth.write_text(json.dumps({"comparisons": {}, "resource_gate": {"passed": True},
                                 "correctness": {"identical": True}}))
    prediction.write_text(json.dumps({"prediction": {"answers": {}}}))
    with pytest.raises(SystemExit, match="not sealed first"):
        module.main(["verdict", "--prediction", str(prediction),
                     "--ground-truth-record", str(truth), "--out", str(tmp_path / "v.json")])
