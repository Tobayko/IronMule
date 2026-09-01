"""F1 contract tests: request-level composition and preregistered thresholds."""

from __future__ import annotations

import pytest

from friday_optimizer.evaluator import MetricSample
from friday_optimizer.integration import (
    ARMS,
    CONFIRMED_RATIOS,
    IntegrationError,
    evaluate_integration,
    paired_request_ratios,
    prefill_share,
    project_request_ratio,
    request_seconds,
)


def arm_samples(ttft, decode_tps, *, arm, count=6, tokens=32, status="ok", orders=True):
    return tuple(
        MetricSample(
            session_id=f"s{index}", pair_id=f"p{index}", arm=arm,
            order=("AB" if index % 2 == 0 else "BA") if orders else "",
            fingerprint="f" * 64, workload="workload",
            ttft_seconds=ttft, decode_tps=decode_tps, tokens=tokens, status=status,
        )
        for index in range(count)
    )


def test_request_seconds_composes_prefill_and_decode():
    sample = MetricSample(ttft_seconds=1.8, decode_tps=62.0, tokens=32)
    assert request_seconds(sample) == pytest.approx(1.8 + 32 / 62.0)


def test_request_seconds_refuses_to_complete_missing_evidence():
    assert request_seconds(MetricSample(ttft_seconds=1.8, decode_tps=62.0)) is None
    assert request_seconds(MetricSample(ttft_seconds=1.8, tokens=32)) is None
    assert request_seconds(MetricSample(ttft_seconds=0.0, decode_tps=62.0, tokens=32)) is None
    assert request_seconds(MetricSample(ttft_seconds=1.8, decode_tps=62.0, tokens=32, status="error")) is None


def test_prefill_share_matches_the_measured_warm_profile():
    # Sealed persistent-process evidence: 1.7851 s prefill, 32 tokens at 70.99 tok/s.
    share = prefill_share(ttft_seconds=1.7851, tokens=32, decode_tps=70.99)
    assert share == pytest.approx(0.7984, abs=5e-4)


def test_projection_is_identity_without_a_change():
    assert project_request_ratio(ttft_seconds=1.8, tokens=32, decode_tps=70.0) == pytest.approx(1.0)


def test_phase_gains_do_not_multiply():
    ttft_ratio = CONFIRMED_RATIOS["head_skip_prefill"]
    decode_ratio = CONFIRMED_RATIOS["fixed_compiled_cache"]
    composed = project_request_ratio(
        ttft_seconds=1.7851, tokens=32, decode_tps=70.99,
        ttft_ratio=ttft_ratio, decode_tps_ratio=1.0 / decode_ratio,
    )
    naive = ttft_ratio * decode_ratio
    # The naive product promises about 21 %; the weighted truth is about 14 %.
    assert composed == pytest.approx(0.8632, abs=1e-3)
    assert composed > naive + 0.05


def test_a_decode_gain_shrinks_as_the_prompt_dominates():
    long_prompt = project_request_ratio(
        ttft_seconds=1.7851, tokens=32, decode_tps=70.99, decode_tps_ratio=1.0 / 0.9296
    )
    long_answer = project_request_ratio(
        ttft_seconds=1.7851, tokens=512, decode_tps=70.99, decode_tps_ratio=1.0 / 0.9296
    )
    assert long_prompt > long_answer, "a decode lever must matter more when decode dominates"
    assert 1 - long_prompt < 0.02


def test_projection_rejects_invalid_inputs():
    for kwargs in (
        {"ttft_seconds": 0.0, "tokens": 32, "decode_tps": 70.0},
        {"ttft_seconds": 1.8, "tokens": 0, "decode_tps": 70.0},
        {"ttft_seconds": 1.8, "tokens": 32, "decode_tps": float("inf")},
        {"ttft_seconds": 1.8, "tokens": True, "decode_tps": 70.0},
    ):
        with pytest.raises(IntegrationError):
            project_request_ratio(**kwargs)


def test_pairing_inherits_the_evaluator_gate():
    baseline = arm_samples(1.8, 70.0, arm="baseline")
    candidate = arm_samples(1.5, 75.0, arm="candidate")
    ratios, reasons = paired_request_ratios(baseline, candidate)
    assert reasons == () and len(ratios) == 6
    _, unbalanced = paired_request_ratios(baseline, arm_samples(1.5, 75.0, arm="candidate", orders=False))
    assert unbalanced and all(reason.startswith("ab_") for reason in unbalanced)


def test_qualified_only_when_the_interval_clears_the_threshold():
    baseline = arm_samples(1.8, 70.0, arm="baseline")
    candidate = arm_samples(1.5, 76.0, arm="candidate")
    result = evaluate_integration(baseline, candidate, arm="warm", min_gain=0.10, mde=0.05)
    assert result.status == "qualified" and result.qualified
    assert result.gain_percent > 10.0
    assert result.as_dict()["formal_claim"] is False

    modest = evaluate_integration(baseline, candidate, arm="warm", min_gain=0.30, mde=0.05)
    assert modest.status == "below_threshold" and not modest.qualified


def test_a_confirmed_regression_is_rejected_not_merely_missed():
    baseline = arm_samples(1.8, 70.0, arm="baseline")
    slower = arm_samples(2.6, 55.0, arm="candidate")
    result = evaluate_integration(baseline, slower, arm="cold", min_gain=0.10, mde=0.05)
    assert result.status == "rejected"
    assert "statistically_confirmed_request_regression" in result.reasons


def test_thin_evidence_is_inconclusive_and_never_a_number():
    baseline = arm_samples(1.8, 70.0, arm="baseline", count=2)
    candidate = arm_samples(1.5, 76.0, arm="candidate", count=2)
    result = evaluate_integration(baseline, candidate, arm="warm", min_gain=0.10, mde=0.05)
    assert result.status == "inconclusive" and result.ratio_median is None
    assert result.reasons == ("integration_requires_more_paired_sessions",)


def test_arms_and_configuration_are_validated():
    baseline = arm_samples(1.8, 70.0, arm="baseline")
    candidate = arm_samples(1.5, 76.0, arm="candidate")
    assert ARMS == ("cold", "warm")
    with pytest.raises(IntegrationError):
        evaluate_integration(baseline, candidate, arm="lukewarm", min_gain=0.1, mde=0.05)
    with pytest.raises(IntegrationError):
        evaluate_integration(baseline, candidate, arm="warm", min_gain=1.0, mde=0.05)


def test_evidence_hash_is_stable_and_binds_the_samples():
    baseline = arm_samples(1.8, 70.0, arm="baseline")
    candidate = arm_samples(1.5, 76.0, arm="candidate")
    first = evaluate_integration(baseline, candidate, arm="warm", min_gain=0.10, mde=0.05)
    second = evaluate_integration(baseline, candidate, arm="warm", min_gain=0.10, mde=0.05)
    changed = evaluate_integration(baseline, arm_samples(1.51, 76.0, arm="candidate"), arm="warm", min_gain=0.10, mde=0.05)
    assert first.evidence_hash == second.evidence_hash != changed.evidence_hash
