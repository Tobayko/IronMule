"""Goal/ranking regressions only; these cases are not hardware evidence."""
import math

import pytest

from friday_evidence.portable.learning import LearningError, _validation_runtime_score, train_and_evaluate
from friday_evidence.portable.objectives import runtime_objective


def test_goal_prioritizes_runtime_and_does_not_claim_cross_device_success():
    goal = runtime_objective()
    assert goal["primary"] == "maximize_end_to_end_inference_speedup"
    assert goal["secondary"] == "minimize_discovery_cost_within_free_budget"
    assert goal["proxy_oracle_runtime_ratio_limit"] == 1.0
    assert goal["kaggle_money_budget_eur"] == 0
    assert goal["kaggle_quota_policy_changed"] is False
    assert goal["proxy_is_runtime_goal_evidence"] is False
    refs = {item["reference"]: item for item in goal["targets"]}
    assert 1 / refs["E13"]["maximum_ratio"] == pytest.approx(4.90269232, abs=1e-6)
    assert refs["E13"]["historical_token_identity"] is False
    assert refs["E13"]["current_quality_contract_requires_new_validation"] is True
    assert refs["D5"]["requires_fresh_qualification"] is True
    assert refs["B39d"]["separate_token_rate_minimum_ratio"] == 1.2202787058
    goal["targets"].clear()
    assert len(runtime_objective()["targets"]) == 3


def test_selection_error_cannot_hide_behind_a_better_prediction_rmse():
    # Counterexample for the selection bug: closer numerical predictions can
    # still rank the slower arm first. No estimator is trained in this test.
    rows = [{"run_id": "run", "case_id": "case", "stratum_sha256": "stratum",
             "group_sha256": "group", "candidate": name, "label": {"median_ratio": ratio}}
            for name, ratio in (("fast", 0.5), ("slow", 0.6))]
    wrong_choice = _validation_runtime_score(rows, [math.log(.56), math.log(.55)])
    right_choice = _validation_runtime_score(rows, [math.log(.3), math.log(.8)])
    assert wrong_choice == pytest.approx(1.2)
    assert right_choice == 1.0
    with pytest.raises(LearningError):
        _validation_runtime_score(rows, [float("nan"), 0.0])


def test_empty_corpus_reports_new_goal_as_unmet_without_training(tmp_path):
    result = train_and_evaluate(tmp_path)
    assert result["schema"] == "ironmule.learning.v2"
    assert result["runtime_goal_met"] is False
    assert result["performance_claim"] is False
    assert result["model"] is None
    assert result["runtime_objective"]["primary"] == "maximize_end_to_end_inference_speedup"
