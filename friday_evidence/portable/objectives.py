"""Prospective user objective; historical references are targets, not promotions."""
from __future__ import annotations

from copy import deepcopy


_OBJECTIVE = {
    "schema": "ironmule.objective.v2",
    "primary": "maximize_end_to_end_inference_speedup",
    "minimum": "match_or_exceed_historical_best_in_comparable_workload",
    "secondary": "minimize_discovery_cost_within_free_budget",
    "proxy_oracle_runtime_ratio_limit": 1.0,
    "proxy_is_runtime_goal_evidence": False,
    "required_comparators": ["native_stock_backend", "best_compatible_ironmule_configuration"],
    "quality_policy": "retain_current_exact_or_explicit_efficiency_contract",
    "kaggle_money_budget_eur": 0,
    "kaggle_quota_policy_changed": False,
    "targets": [
        {
            "reference": "E13",
            "workload": "gemma_4b_eight_questions_shared_document",
            "metric": "complete_session_wall_candidate_over_strict",
            "maximum_ratio": 0.2039695603648146,
            "evidence": "historical_secondary_performance_with_quality_difference",
            "historical_token_identity": False,
            "historical_accuracy_loss_bound_pp": 1.1363636363636365,
            "current_quality_contract_requires_new_validation": True,
            "source": "research/raw/E13_summary.json",
        },
        {
            "reference": "D5",
            "workload": "gemma_1b_greedy_897_prompt_32_output_batch_one",
            "metric": "complete_request_wall_candidate_over_knobs_off",
            "maximum_ratio": 0.6959773070789428,
            "evidence": "historical_exploratory_formal_claim_false",
            "historical_token_identity": True,
            "requires_fresh_qualification": True,
            "source": "experiments/serve_gain/gain_1b_32_b.json",
        },
        {
            "reference": "B39d",
            "workload": "gemma_12b_six_requests_48_tokens",
            "metric": "complete_service_wall_core_throughput_over_baseline_interactive",
            "maximum_ratio": 0.8194867050,
            "separate_token_rate_minimum_ratio": 1.2202787058,
            "evidence": "historical_qualified_activation_not_allowed",
            "historical_token_identity": True,
            "source": "research/raw/B39d_public_summary_20260828.json",
        },
    ],
}


def runtime_objective() -> dict:
    """Return an independent JSON projection; never infer cross-device success."""
    return deepcopy(_OBJECTIVE)
