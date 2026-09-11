"""A shadow cost model must abstain rather than guess, and must never touch a route."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "research" / "raw" / "B72_cost_model_20260910_v3.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "b72", ROOT / "tools" / "b72_cost_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record():
    return json.loads(RECORD.read_text())


def test_the_split_has_no_study_on_both_sides():
    split = _record()["split"]
    assert split["leakage"] == []
    assert set(split["train_studies"]) & set(split["holdout_studies"]) == set()
    assert set(split["train_studies"]) & set(split["validation_studies"]) == set()


def test_no_blocked_or_invalid_evidence_was_used():
    dataset = json.loads((ROOT / "research" / "raw" /
                          "B72_cost_dataset_20260910.json").read_text())
    for row in dataset["rows"]:
        assert row["validity"] not in ("BLOCKED", "EXPLORATORY_INVALID", "NOT_STARTED")


def test_the_hardware_block_is_reported_as_constant():
    """One machine. A model that claimed to learn from it would be claiming a constant."""
    record = _record()
    assert record["hardware_block_is_constant"] is True
    assert not any(name.startswith("hardware.")
                   for name in record["features_actually_varying_in_train"])


def test_tiny_variance_is_not_called_variation():
    """1e-16 across identical floats is float representation, not information."""
    module = _module()
    column = np.full(10, 0.9465755303192629)
    assert float(np.std(column)) <= 1e-12


def test_every_holdout_context_abstained():
    for name, result in _record()["results"].items():
        for decision in result["decisions_holdout"]:
            assert decision["abstained"], name
            assert decision["reason"]
            assert decision["predicted_action"] is None


def test_an_unseen_action_is_named_as_the_reason():
    reasons = " ".join(d["reason"] for r in _record()["results"].values()
                       for d in r["decisions_holdout"])
    assert "no evidence in training" in reasons


def test_actions_that_do_different_work_are_never_ranked():
    module = _module()
    assert "kernel_wall_time_ratio_vs_width_4" in module.INCOMPARABLE_METRICS
    reasons = " ".join(d["reason"] for r in _record()["results"].values()
                       for d in r["decisions_validation"])
    assert "do not do equal work" in reasons


def test_the_ablation_says_it_is_not_testable_rather_than_passed():
    record = _record()
    assert record["ablation_testable"] is False
    for name, row in record["ablation_hardware_removed"].items():
        assert row["testable"] is False, name
        assert "not answerable" in row["why"], name


def test_the_models_are_small_and_fast():
    for name, result in _record()["results"].items():
        assert result["model_size_bytes"] < 8192, name
        assert result["inference_ns_per_context"] < 100_000, name


def test_the_models_are_deterministic():
    module = _module()
    dataset = json.loads((ROOT / "research" / "raw" /
                          "B72_cost_dataset_20260910.json").read_text())
    rows = dataset["rows"]
    matrix, _names = module.featurise(rows)
    y = np.array([r["measured_cost"]["value"] for r in rows])
    first = module.Ridge().fit(rows, matrix, y).predict(rows, matrix)
    second = module.Ridge().fit(rows, matrix, y).predict(rows, matrix)
    assert np.array_equal(first, second)


def test_missing_features_get_an_explicit_indicator():
    module = _module()
    dataset = json.loads((ROOT / "research" / "raw" /
                          "B72_cost_dataset_20260910.json").read_text())
    _matrix, names = module.featurise(dataset["rows"])
    assert "hardware.geometry_4_8_ratio.missing" in names
    assert "workload.k.missing" in names


def test_nothing_here_reaches_the_router():
    router = (ROOT / "ironmule" / "router.py").read_text()
    assert "b72" not in router.lower()
    assert "cost_model" not in router


def test_no_hardware_generalisation_is_claimed():
    record = _record()
    assert "No hardware generalisation is claimed" in record["one_machine"]
    assert "H1 is not reported as tested" in record["one_machine"]
