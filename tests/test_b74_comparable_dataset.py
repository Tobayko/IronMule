"""Only actions that did the same work, on the same order, may be ranked against each other."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "raw" / "B74_comparable_dataset_20260910_v2.json"
COMPANION = ROOT / "research" / "raw" / "B74_comparable_dataset_20260910_v2_v1shape.json"
OUTCOME = ROOT / "research" / "raw" / "B74_outcome_20260910.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "b74", ROOT / "tools" / "b74_comparable_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset():
    return json.loads(DATASET.read_text())


def test_every_quality_check_passed():
    for name, ok in _dataset()["quality_checks"].items():
        assert ok, name


def test_unequal_work_is_never_ranked():
    for row in _dataset()["rows"]:
        if row["measured_cost"]["metric"] == "kernel_wall_time_ratio_vs_width_4":
            assert row["comparability"] != "COMPARABLE"
            assert "not a free action" in row["comparability_reason"]


def test_a_set_shares_one_metric_and_one_work_unit():
    rows = [r for r in _dataset()["rows"] if r["comparability"] == "COMPARABLE"]
    by_context = {}
    for row in rows:
        by_context.setdefault(row["comparison_context_id"], []).append(row)
    for context, members in by_context.items():
        assert len({m["measured_cost"]["metric"] for m in members}) == 1, context
        assert len({m["work_units"] for m in members}) == 1, context
        assert len({m["workload_features"].get("workload_class") for m in members}) == 1, context


def test_the_same_intervention_has_one_name():
    module = _module()
    assert module.canonical("kernel_geometry_sg4_r8") == "k3840_geometry_sg4_r8"
    assert module.canonical("k3840_geometry_4_8") == "k3840_geometry_sg4_r8"
    assert module.canonical("mlx_command_buffer_ops_mb_400") == "mlx_command_buffer_ops_mb_400"


def test_the_reference_is_an_action_at_one():
    rows = [r for r in _dataset()["rows"] if r.get("is_reference_action")]
    assert rows
    for row in rows:
        assert row["normalized_cost"] == 1.0
        assert row["uncertainty"]["kind"] == "definitional"


def test_no_normalisation_is_invented():
    """A row without a work unit in its own source carries no normalised cost."""
    for row in _dataset()["rows"]:
        if row["normalized_cost"] is None:
            assert row["comparability"] != "COMPARABLE"
        if row["comparability"] == "COMPARABLE":
            assert row["normalization_method"]


def test_no_synthetic_training_rows():
    """Every non-reference row traces to a study that measured it."""
    source = json.loads((ROOT / "research" / "raw" /
                         "B72_cost_dataset_20260910.json").read_text())
    measured = {(r["evidence_id"], r["action"]) for r in source["rows"]}
    for row in _dataset()["rows"]:
        if row.get("is_reference_action"):
            continue
        assert (row["evidence_id"], row["original_action"]) in measured


def test_the_split_keeps_studies_whole():
    split = _dataset()["split"]
    assert not set(split["train_studies"]) & set(split["holdout_studies"])
    assert not set(split["train_studies"]) & set(split["validation_studies"])


def test_coverage_labels_unseen_actions():
    for context, row in _dataset()["coverage"].items():
        assert set(row["known_action"]) | set(row["unseen_action"]) <= set(row["actions"])
        if row["unseen_action"]:
            assert not row["usable_as_a_top1_test"], context


def test_the_companion_only_carries_rankable_rows():
    companion = json.loads(COMPANION.read_text())
    assert companion["schema"] == "ironmule.cost_dataset.v1"
    metrics = {r["measured_cost"]["metric"] for r in companion["rows"]}
    assert "kernel_wall_time_ratio_vs_width_4" not in metrics


def test_the_outcome_states_the_trivial_policy_matches_top_1():
    """Top-1 on three easy contexts is not evidence a cost model was learned."""
    outcome = json.loads(OUTCOME.read_text())
    trivial = outcome["the_caveat_that_matters_most"]["trivial_policy"]
    assert trivial["always_the_non_reference_action"] == trivial["n"]
    assert "NOT evidence" in outcome["the_caveat_that_matters_most"]["reading"]


def test_the_outcome_records_the_tree_failure():
    outcome = json.loads(OUTCOME.read_text())
    assert "catastrophic" in outcome["the_tree_failed"]
    assert outcome["replay"]["per_model"]["regression_tree_depth_3"]["catastrophic"] == 1


def test_no_hardware_generalisation_is_claimed():
    outcome = json.loads(OUTCOME.read_text())
    assert "shows nothing about" in outcome["b73_is_still_the_first_real_test"]
