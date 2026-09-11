"""The four quantities B77 separates, and the rule that used to collapse them.

The point of `B77` is that a miscalibrated interval, a wrong action and a real cost are three
different things. These check that the classifier keeps them apart, that only a genuine
confident action error is flagged as dangerous, and that the lexicographic choice cannot be
won by a state-feature model whose features make the forecast worse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "tools" / "b77_rule_review.py"


def _module():
    spec = importlib.util.spec_from_file_location("b77_rule_review", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b77 = _module()


def _prediction(ratio, interval, action=None):
    return {"predicted_ratio": ratio, "prediction_interval": interval,
            "predicted_action": action or ("ABSTAIN" if interval is None else
                                           ("CANDIDATE" if interval[1] < 1.0 else
                                            ("REFERENCE" if interval[0] > 1.0 else "ABSTAIN")))}


def _truth(ratio, half=0.01):
    return {"ratio": ratio, "ci_low": ratio - half, "ci_high": ratio + half}


# --------------------------------------------------------------------------- definitions


def test_a_near_miss_is_a_calibration_miss_and_nothing_worse() -> None:
    """B76 session 6: the interval missed by 0.0018 and the action was still right."""
    row = b77._classify(_prediction(0.9782, [0.9597, 0.9967]), _truth(0.9579))
    assert row["CALIBRATION_MISS"] is True
    assert row["ACTION_ERROR"] is False
    assert row["CONFIDENT_ACTION_ERROR"] is False
    assert row["REGRET"] == 0.0
    assert row["old_rule_called_it_confidently_wrong"] is True, (
        "which is exactly the conflation B77 exists to separate")


def test_a_confident_action_error_needs_both_sides_of_the_boundary() -> None:
    row = b77._classify(_prediction(0.95, [0.94, 0.96]), _truth(1.08))
    assert row["CONFIDENT_ACTION_ERROR"] is True
    assert row["ACTION_ERROR"] is True
    assert row["REGRET"] == pytest.approx(0.08)


def test_an_abstention_never_counts_as_a_confident_error() -> None:
    row = b77._classify(_prediction(1.0, [0.90, 1.10]), _truth(0.94))
    assert row["CONFIDENT_ACTION_ERROR"] is False
    assert row["ACTION_ERROR"] is True
    assert row["action_error_kind"] == "abstention"
    assert row["REGRET"] == pytest.approx(0.06), "abstaining forgoes the gain, it does not lose"


def test_a_covered_prediction_on_the_wrong_side_is_still_not_confident() -> None:
    """A wide interval that spans the boundary has not committed, however wrong it lands."""
    row = b77._classify(_prediction(0.99, [0.97, 1.03]), _truth(1.05))
    assert row["CALIBRATION_MISS"] is True
    assert row["CONFIDENT_ACTION_ERROR"] is False


def test_no_absolute_threshold_enters_any_of_the_four_quantities() -> None:
    wide = b77._classify(_prediction(0.95, [0.80, 0.99]), _truth(0.70))
    narrow = b77._classify(_prediction(0.95, [0.949, 0.951]), _truth(0.70))
    for row in (wide, narrow):
        assert row["CALIBRATION_MISS"] is True
        assert row["CONFIDENT_ACTION_ERROR"] is False
        assert row["REGRET"] == 0.0, "both chose the candidate and the candidate won"


def test_the_interval_score_punishes_width_and_distance() -> None:
    tight_hit = b77._interval_score([0.94, 0.96], 0.95)
    wide_hit = b77._interval_score([0.90, 1.00], 0.95)
    tight_miss = b77._interval_score([0.94, 0.96], 0.99)
    assert tight_hit < wide_hit < tight_miss


# --------------------------------------------------------------------------- selection


def _scores(**overrides):
    base = {name: {"confident_action_errors": 0, "cumulative_regret": 0.1,
                   "coverage_95": 1.0, "mean_absolute_error": 0.005,
                   "rows": [], "abstain_rate": 0.0}
            for name in b77.MODELS}
    for name, row in overrides.items():
        base[name].update(row)
    return base


def test_a_confident_action_error_loses_however_good_the_error_is() -> None:
    scores = _scores(C_bayes={"confident_action_errors": 1, "mean_absolute_error": 0.0001})
    assert b77.choose(scores)["winner"] != "C_bayes"


def test_regret_outranks_prediction_error() -> None:
    scores = _scores(B_ridge={"cumulative_regret": 0.5, "mean_absolute_error": 0.0001},
                     C_bayes={"cumulative_regret": 0.5, "mean_absolute_error": 0.0002},
                     A_constant={"cumulative_regret": 0.1, "mean_absolute_error": 0.01})
    assert b77.choose(scores)["winner"] == "A_constant", (
        "the cheapest action wins even with fifty times the prediction error")


def test_the_simplest_model_wins_a_genuine_draw() -> None:
    assert b77.choose(_scores())["winner"] == "A_constant"


def test_a_state_feature_model_cannot_win_while_its_features_hurt() -> None:
    scores = _scores(B_ridge={"cumulative_regret": 0.05, "mean_absolute_error": 0.02},
                     A_constant={"cumulative_regret": 0.10, "mean_absolute_error": 0.005})
    choice = b77.choose(scores)
    assert choice["state_feature_model_hurts"] is True
    assert choice["winner"] != "B_ridge"


def test_practically_identical_needs_the_same_actions_and_the_same_cost() -> None:
    rows = [{"predicted_action": "CANDIDATE"}, {"predicted_action": "CANDIDATE"}]
    left = {"rows": rows, "cumulative_regret": 0.075, "coverage_95": 1.0,
            "mean_absolute_error": 0.005772}
    right = {"rows": rows, "cumulative_regret": 0.075, "coverage_95": 1.0,
             "mean_absolute_error": 0.005864}
    assert b77.practically_identical(left, right, 0.006888)["practically_identical"] is True
    diverging = {**right, "rows": [{"predicted_action": "ABSTAIN"},
                                   {"predicted_action": "CANDIDATE"}]}
    assert b77.practically_identical(left, diverging, 0.006888)["practically_identical"] is False


# --------------------------------------------------------------------------- guards


def test_the_review_refits_nothing() -> None:
    text = SOURCE.read_text()
    for forbidden in ("np.linalg", "numpy", "ridge(", "fit(", "model_b_ridge", "predict_all"):
        assert forbidden not in text, f"{forbidden} would mean a prediction was recomputed"


def test_b76_is_read_but_never_rewritten() -> None:
    text = SOURCE.read_text()
    assert "write_once" in text
    assert "B76_temporal_learning_20260910.json" in text
    for forbidden in ("write_text", "unlink", "replace(", "open(", "'w'"):
        assert forbidden not in text, f"{forbidden} could alter a sealed file"


# --------------------------------------------------------------------------- B69 axes


def _b69(candidate_ratios, aa_median, aa_low, aa_high):
    return {"comparisons": {"single_short": {
        "candidate": {"median": 0.8469, "ci_low": 0.7580, "ci_high": 0.9488,
                      "ratios": candidate_ratios},
        "reference_aa": {"median": aa_median, "ci_low": aa_low, "ci_high": aa_high}}}}


def _b76(block_ratios):
    return {"sessions": [{"block_ratios": {"single_short": {"candidate": block_ratios}}}],
            "aa_distribution": {"median": 1.0, "sd": 0.0069, "max_absolute_offset": 0.019,
                                "n": 14}}


def test_a_wide_aa_control_is_reported_as_failing_the_later_gate() -> None:
    noise = b77.b69_noise_floor(_b69([0.96, 0.68, 0.83, 0.85, 0.85, 0.94], 1.0054, 0.8880, 1.0576),
                                _b76([0.95, 0.96, 0.97]))
    assert noise["b69_aa_would_pass_the_b75_gate"] is False
    assert noise["b69_candidate_and_aa_intervals_overlap"] is True
    assert noise["gate"]["applied_to_b69_verdict"] is False, (
        "B77 reports what the control showed; it does not reopen B69's verdict")


def test_a_quiet_aa_control_passes_and_does_not_overlap() -> None:
    noise = b77.b69_noise_floor(_b69([0.95, 0.96, 0.95], 1.0, 0.99, 1.01),
                                _b76([0.95, 0.96, 0.97]))
    assert noise["b69_aa_would_pass_the_b75_gate"] is True
    assert noise["b69_candidate_and_aa_intervals_overlap"] is False


def test_the_arm_decomposition_holds_each_arm_at_the_other_study_level() -> None:
    arms = {"candidate": {"b69": {"median_ms": 938.0}, "b76": {"median_ms": 1036.3}},
            "reference": {"b69": {"median_ms": 1132.2}, "b76": {"median_ms": 1075.4}}}
    row = b77.b69_arm_decomposition(arms)
    assert row["b69_as_measured"] < row["b69_candidate_over_b76_reference"]
    assert row["b69_candidate_over_b76_reference"] < row["b76_candidate_over_b69_reference"]
    assert row["b76_candidate_over_b69_reference"] < row["b76_as_measured"], (
        "neither arm alone closes the gap")
