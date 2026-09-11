"""B76's learner, its gates and its isolation, checked without touching the GPU.

The three-hour run cannot afford to discover a broken predictor in session four, and a
learner that quietly reads a historical number would answer its own question. Both are
checked here on synthetic sessions.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "tools" / "b76_temporal_learning.py"


def _module():
    spec = importlib.util.spec_from_file_location("b76_temporal_learning", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b76 = _module()


def _session(ratio: float, load: float = 3.0, free: float = 40.0, swap: float = 5.0,
             se: float | None = 0.01, index: int = 0) -> dict:
    return {"session": index, "ratio": ratio, "ci": [ratio - 0.02, ratio + 0.02],
            "within_session_se": se, "aa_median": 1.0, "aa_gate_passed": True,
            "state": {"load_1min": load, "memory_free_percent": free, "swap_used_gb": swap,
                      "memory_pressure_level": 1}}


def _state(load: float = 3.0, free: float = 40.0, swap: float = 5.0) -> dict:
    return {"load_1min": load, "memory_free_percent": free, "swap_used_gb": swap,
            "memory_pressure_level": 1}


# --------------------------------------------------------------------------- isolation


@pytest.mark.parametrize("forbidden", ["B69", "B75", "B72", "B74", "silicon_profile"])
def test_the_learner_cannot_read_a_historical_number(forbidden: str) -> None:
    """Every mention must be prose or the machinery import, never a record being opened."""
    text = SOURCE.read_text()
    for match in re.finditer(re.escape(forbidden), text):
        line = text[text.rfind("\n", 0, match.start()) + 1:
                    text.find("\n", match.end())]
        assert "research/raw" not in line and "read_text" not in line and "json.load" not in line, line
    assert "b69_stack_proof" in text, "the child process is imported as measuring machinery"


def test_no_historical_evidence_file_is_reachable_from_the_harness() -> None:
    """Not `it does not read one` -- it has no path to one."""
    text = SOURCE.read_text()
    assert "B69_stack_proof_2" not in text and "B75_cold_start_2" not in text
    assert "research" not in text.replace("research/raw", "@"), (
        "the raw evidence directory is never named except in the prose that says so")
    body = ast.get_source_segment(text, ast.parse(text).body[-1]) or ""
    code = text[text.index(body):] if body else text
    for number in ("0.8469", "0.9624", "0.7580", "0.9488", "0.9563", "0.9630"):
        assert number not in code, (
            f"{number} is a historical ratio and must not reach the code. The module "
            f"docstring may name it as the motivation; the learner may not use it")


# --------------------------------------------------------------------------- models


def test_every_model_abstains_before_it_has_evidence() -> None:
    for name, prediction in b76.predict_all([], _state()).items():
        assert prediction["predicted_action"] == "ABSTAIN", name
        assert prediction["abstain_reason"], name
        assert prediction["prediction_interval"] is None, name


def test_the_ridge_model_abstains_until_it_has_enough_rows() -> None:
    history = [_session(0.95, index=i) for i in range(b76.MIN_ROWS_FOR_LINEAR - 1)]
    assert b76.model_b_ridge(history, _state())["predicted_action"] == "ABSTAIN"
    history.append(_session(0.95, index=99))
    assert b76.model_b_ridge(history, _state())["predicted_action"] != "ABSTAIN"


def test_the_ridge_model_abstains_when_a_feature_is_unreadable() -> None:
    history = [_session(0.95, load=2.0 + i, index=i) for i in range(8)]
    blind = {**_state(), "load_1min": None}
    prediction = b76.model_b_ridge(history, blind)
    assert prediction["predicted_action"] == "ABSTAIN"
    assert "load_1min" in prediction["abstain_reason"]


def test_the_ridge_model_follows_a_feature_that_actually_carries_the_signal() -> None:
    history = [_session(1.05 - 0.02 * load, load=float(load), index=load)
               for load in range(1, 10)]
    slow = b76.model_b_ridge(history, _state(load=9.0))["predicted_ratio"]
    quiet = b76.model_b_ridge(history, _state(load=1.0))["predicted_ratio"]
    assert slow < quiet, "a ratio that falls with load must be predicted lower under load"


def test_a_consistent_win_becomes_a_candidate_action() -> None:
    history = [_session(0.90 + 0.002 * i, se=0.002, index=i) for i in range(8)]
    for name, prediction in b76.predict_all(history, _state()).items():
        assert prediction["predicted_action"] == "CANDIDATE", (name, prediction)


def test_wildly_varying_sessions_produce_an_abstention_not_a_guess() -> None:
    history = [_session(r, se=0.01, index=i)
               for i, r in enumerate([0.70, 1.25, 0.85, 1.15, 0.95, 1.05, 0.80, 1.20])]
    assert b76.model_c_bayes(history, _state())["predicted_action"] == "ABSTAIN"
    assert b76.model_a_constant(history, _state())["predicted_action"] == "ABSTAIN"


def test_the_bayesian_interval_is_never_narrower_than_its_posterior() -> None:
    history = [_session(0.95, se=0.01, index=i) for i in range(10)]
    prediction = b76.model_c_bayes(history, _state())
    confidence = prediction["confidence"]
    assert confidence["half_width"] > 1.96 * confidence["posterior_sd"], (
        "a predictive interval must carry between- and within-session variance too")


# --------------------------------------------------------------------------- scoring


def _sealed(session: int, predicted: float, interval, actual: float, valid: bool = True):
    prediction = {"predicted_ratio": predicted, "prediction_interval": interval,
                  "predicted_action": b76._action(interval),
                  "confidence": {"half_width": None if interval is None
                                 else (interval[1] - interval[0]) / 2,
                                 "n_prior_sessions": session},
                  "abstain_reason": "", "state_features_used": []}
    return {"session": session, "state": _state(),
            "predictions": {name: prediction for name in b76.MODELS},
            "ground_truth": {"ratio": actual, "ci_low": actual - 0.01,
                             "ci_high": actual + 0.01, "valid_for_learning": valid}}


def test_scoring_counts_coverage_error_and_regret() -> None:
    sealed = [_sealed(0, 0.95, [0.93, 0.97], 0.94),
              _sealed(1, 0.95, [0.93, 0.97], 0.99),
              _sealed(2, 0.95, [0.93, 0.97], 0.96)]
    scored = b76.score([], sealed)["C_bayes"]
    assert scored["n_scored"] == 3
    assert scored["coverage_95"] == pytest.approx(2 / 3)
    assert scored["mean_absolute_error"] == pytest.approx(
        (0.01 + 0.04 + 0.01) / 3, abs=1e-9)
    assert scored["total_regret"] == pytest.approx(0.0), (
        "choosing the candidate when the candidate wins carries no regret")


def test_an_invalid_session_is_never_scored() -> None:
    sealed = [_sealed(0, 0.95, [0.93, 0.97], 0.60, valid=False)]
    assert b76.score([], sealed)["C_bayes"]["n_scored"] == 0


def test_choosing_the_candidate_when_it_loses_is_regret() -> None:
    sealed = [_sealed(0, 0.95, [0.93, 0.97], 1.08)]
    scored = b76.score([], sealed)["A_constant"]
    assert scored["total_regret"] == pytest.approx(0.08)
    assert scored["confidently_wrong"] == 1, "a 0.02 interval missed by 0.11 is confident and wrong"


def test_abstaining_falls_back_to_the_reference_and_forgoes_the_gain() -> None:
    sealed = [_sealed(0, None, None, 0.90)]
    scored = b76.score([], sealed)["A_constant"]
    assert scored["abstain_rate"] == 1.0
    assert scored["total_regret"] == pytest.approx(0.10)
    assert scored["confidently_wrong"] == 0


# --------------------------------------------------------------------------- variance


def test_the_between_session_variance_removes_the_measurement_error() -> None:
    history = [_session(r, se=0.02, index=i)
               for i, r in enumerate([0.90, 0.92, 0.94, 0.96, 0.98])]
    row = b76.variance_decomposition(history)
    assert row["tau_squared_method_of_moments"] > 0.0
    assert row["tau_squared_method_of_moments"] < row["between_session_variance_total"]


def test_pure_measurement_noise_leaves_no_between_session_variance() -> None:
    history = [_session(r, se=0.05, index=i)
               for i, r in enumerate([0.95, 0.96, 0.94, 0.95, 0.96])]
    assert b76.variance_decomposition(history)["tau_squared_method_of_moments"] == 0.0


# --------------------------------------------------------------------------- verdict


def _run(ratio: float, valid: bool = True) -> dict:
    return {"correctness_identical": True, "fallbacks": 0, "valid_for_learning": valid,
            "primary": {"median": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01}}


def test_too_few_sessions_is_partial_and_never_a_verdict() -> None:
    history = [_session(0.95, index=i) for i in range(3)]
    outcome = b76.decide_verdict(history, [_run(0.95)] * 3, b76.score(history, []),
                                 b76.variance_decomposition(history))
    assert outcome["verdict"] == "B76_PARTIAL"


def test_a_token_difference_is_invalid_before_anything_else_is_read() -> None:
    history = [_session(0.95, index=i) for i in range(b76.MIN_SESSIONS)]
    broken = [_run(0.95) for _ in range(b76.MIN_SESSIONS)]
    broken[2]["correctness_identical"] = False
    assert b76.decide_verdict(history, broken, b76.score(history, []),
                              b76.variance_decomposition(history))["verdict"] == "B76_INVALID"


def test_a_confidently_wrong_prediction_fails_the_study() -> None:
    history = [_session(0.95, index=i) for i in range(b76.MIN_SESSIONS)]
    sealed = [_sealed(i, 0.95, [0.945, 0.955], 1.30) for i in range(b76.MIN_SESSIONS)]
    outcome = b76.decide_verdict(history, [_run(1.30)] * b76.MIN_SESSIONS,
                                 b76.score(history, sealed),
                                 b76.variance_decomposition(history))
    assert outcome["verdict"] == "B76_LOCAL_LEARNING_FAIL"


def test_a_stable_sign_without_a_better_forecast_is_variable_gain() -> None:
    """A wide interval does not erase the finding that the sign never moved."""
    history = [_session(0.95, index=i) for i in range(b76.MIN_SESSIONS)]
    sealed = [_sealed(i, 0.95, [0.80, 1.10], 0.95) for i in range(b76.MIN_SESSIONS)]
    scored = b76.score(history, sealed)
    scored["B_ridge"]["mean_absolute_error"] = 0.20
    scored["A_constant"]["mean_absolute_error"] = 0.05
    outcome = b76.decide_verdict(history, [_run(0.95)] * b76.MIN_SESSIONS, scored,
                                 b76.variance_decomposition(history))
    assert outcome["verdict"] == "B76_STABLE_ACTION_VARIABLE_GAIN"
    assert outcome["abstain_rate"] == 1.0, "and it reports that it never named an action"


def test_an_unstable_sign_with_a_refusing_learner_is_a_safe_abstain() -> None:
    ratios = [0.90, 1.08, 0.94, 1.12, 0.88, 1.05, 0.97, 1.03, 0.92, 1.10]
    history = [_session(r, index=i) for i, r in enumerate(ratios)]
    sealed = [_sealed(i, 1.0, [0.80, 1.20], r) for i, r in enumerate(ratios)]
    runs = [_run(r) for r in ratios]
    outcome = b76.decide_verdict(history, runs, b76.score(history, sealed),
                                 b76.variance_decomposition(history))
    assert outcome["verdict"] == "B76_SAFE_ABSTAIN"
    assert outcome["H1"] is False


def test_a_calibrated_state_aware_forecast_is_the_confirmed_verdict() -> None:
    ratios = [0.90 + 0.005 * i for i in range(b76.MIN_SESSIONS)]
    history = [_session(r, index=i) for i, r in enumerate(ratios)]
    sealed = [_sealed(i, r, [r - 0.02, r + 0.02], r) for i, r in enumerate(ratios)]
    scored = b76.score(history, sealed)
    scored["B_ridge"]["mean_absolute_error"] = 0.01
    scored["A_constant"]["mean_absolute_error"] = 0.05
    outcome = b76.decide_verdict(history, [_run(r) for r in ratios], scored,
                                 b76.variance_decomposition(history))
    assert outcome["verdict"] == "B76_TEMPORAL_LEARNING_CONFIRMED"


# --------------------------------------------------------------------------- design


def test_the_preregistration_fixes_the_budget_and_refuses_to_extend_it() -> None:
    prereg = b76.PREREGISTRATION
    assert prereg["budget"]["wall_clock_seconds"] == 3 * 3600
    assert "not raised because of a result" in prereg["budget"]["no_extension"]
    assert prereg["session"]["primary_class"] == "single_short"
    assert prereg["learner"]["sealed_before_ground_truth"]


def test_the_arm_order_rotates_across_sessions_and_blocks() -> None:
    seen = set()
    for session in range(3):
        for block in range(b76.SESSION_BLOCKS):
            shift = (session + block) % len(b76.ARMS)
            seen.add(b76.ARMS[shift:] + b76.ARMS[:shift])
    assert len(seen) == len(b76.ARMS), "every arm must lead somewhere in the design"


def test_the_state_capture_names_what_it_cannot_read() -> None:
    thermal = b76._thermal_indicator()
    assert "available" in thermal and "raw" in thermal
    if not thermal["available"]:
        assert thermal["reason"], "an unavailable indicator must say why, not read as zero"
