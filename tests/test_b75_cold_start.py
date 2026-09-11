"""A cold start must not be able to peek, and must not act without its own evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "research" / "raw"
LEARNER = ROOT / "tools" / "b75_cold_start.py"
SEALED = RAW / "B75_cold_start_20260910_v2.json"
TRUTH = RAW / "B75_ground_truth_20260910.json"


def _module():
    spec = importlib.util.spec_from_file_location("b75", LEARNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_learner_opens_no_historical_record():
    """Isolation by construction: it has no path to a known winner."""
    source = LEARNER.read_text()
    for forbidden in ("B66_", "B69_stack_proof_20260910", "B72_", "silicon_profile",
                      "B74_", "B71_vector"):
        assert forbidden not in source, forbidden


def test_it_starts_with_the_reference_and_nothing_else():
    record = json.loads(SEALED.read_text())
    assert record["initial_state"]["qualified_actions"] == ["reference"]
    assert record["initial_state"]["knowledge"] == "empty"
    assert record["initial_state"]["confidence"] == "unknown"


def test_a_point_estimate_is_not_decision_grade():
    """The first run answered REFERENCE from a number with no spread. That was the defect."""
    module = _module()
    state = {"features": {"geometry_4_8_ratio": {"value": 0.9376}}}
    answer = module.decide(state)
    assert answer["decision"] == "MEASURE_MORE"
    assert answer["target"] == "geometry_4_8_ratio"
    assert "no interval" in answer["reason"]


def test_a_missing_feature_asks_for_the_deep_probe_by_name():
    module = _module()
    answer = module.decide({"features": {}})
    assert answer["decision"] == "MEASURE_MORE"
    assert answer["target"] == "geometry_4_8_ratio"


def test_an_interval_above_one_keeps_the_reference():
    module = _module()
    state = {"features": {"geometry_4_8_ratio": {"value": 1.02, "ci": [0.99, 1.05]}}}
    assert module.decide(state)["decision"] == "REFERENCE"


def test_an_interval_below_one_reaches_the_candidate():
    module = _module()
    state = {"features": {"geometry_4_8_ratio": {"value": 0.90, "ci": [0.88, 0.93]}}}
    assert module.decide(state)["decision"] == "CANDIDATE"


def test_the_sealed_run_took_the_deep_probe_before_deciding():
    steps = [h for h in json.loads(SEALED.read_text())["history"]
             if "probe" in h or h.get("step") == "decision"]
    kinds = [h.get("probe") or h.get("decision") for h in steps]
    assert kinds[:4] == ["fast", "MEASURE_MORE", "deep_geometry", "CANDIDATE"]


def test_the_qualification_passed_every_gate():
    qualification = next(h for h in json.loads(SEALED.read_text())["history"]
                         if h.get("probe") == "qualification")
    assert qualification["correctness_identical"]
    assert qualification["fallbacks"] == 0
    assert qualification["disturbed_blocks"] == []
    assert qualification["resource_gate_passed"]
    assert not qualification["child_failures"]
    for row in qualification["adoption"].values():
        assert row["aa_ok"], "the A/A arm has to clear before anything is adopted"


def test_nothing_is_qualified_without_local_evidence():
    record = json.loads(SEALED.read_text())
    if record["final_decision"]["decision"] == "CANDIDATE":
        qualification = next(h for h in record["history"]
                             if h.get("probe") == "qualification")
        assert qualification["blocks"] >= 1
        assert any(row["adopted"] for row in qualification["adoption"].values())
    else:
        assert record["qualified_actions_after"] == ["reference"]


def test_the_cost_is_reported_and_dominated_by_the_qualification():
    cost = json.loads(SEALED.read_text())["cost"]
    assert cost["time_to_useful_hardware_knowledge_seconds"] > 0
    assert cost["probe_count"] == 3
    assert cost["probe_breakdown"]["qualification"] > cost["probe_breakdown"]["fast"]


def test_the_ground_truth_was_opened_after_the_seal():
    truth = json.loads(TRUTH.read_text())
    assert "before this file opened any" in truth["order_held"]
    assert truth["sealed_decision"]["digest"]


def test_the_disagreement_in_magnitude_is_stated_not_smoothed():
    truth = json.loads(TRUTH.read_text())
    assert truth["magnitude"]["intervals_overlap"] is False
    assert "MAGNITUDE does not" in truth["magnitude"]["reading"]


def test_no_claim_is_made_about_other_machines():
    truth = json.loads(TRUTH.read_text())
    assert "anything about another Mac" in truth["what_this_does_not_show"]
