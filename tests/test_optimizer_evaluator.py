from dataclasses import replace

import pytest

from friday_optimizer.evaluator import (
    CalibrationReport,
    CorrectnessResult,
    EvaluationError,
    Evaluator,
    MetricSample,
    ResourceResult,
    calibrate_aa,
)
from friday_optimizer.fingerprint import EnvironmentFingerprint, ExactFingerprint, ModelFingerprint, WorkloadFingerprint


def fingerprint():
    return ExactFingerprint(
        EnvironmentFingerprint("M1 Max", "Apple GPU", 32 * 1024**3, 10, "14.5", "0.32.0", "0.31.3", "3.12", "a" * 64),
        ModelFingerprint("google/gemma-3-4b-it", "r1", "b" * 64, "gemma", 4, 64, "tok"),
        WorkloadFingerprint("chat", "tok", "gen", "short", 1, 1, 32, True, False, "performance", "interactive"),
    )


def samples(ttft, decode, count=6, *, status="ok", arm="aa_left", orders=True):
    return tuple(
        MetricSample(
            session_id=str(i), pair_id=str(i), arm=arm,
            order=("AB" if i % 2 == 0 else "BA") if orders else "",
            fingerprint=fingerprint().fingerprint_hash, workload="workload",
            ttft_seconds=ttft, decode_tps=decode, status=status,
        ) for i in range(count)
    )


def resource():
    return (ResourceResult(peak_memory_bytes=100, peak_rss_bytes=200, swap_delta_bytes=0, status="ok"),)


def aa_kwargs():
    return {"aa_baseline_samples": samples(1.0, 10.0, arm="aa_left"), "aa_control_samples": samples(1.0, 10.0, arm="aa_right")}


def correctness():
    return (CorrectnessResult(response_hash="same"), CorrectnessResult(response_hash="same"))


def test_nan_and_bool_are_rejected():
    with pytest.raises(EvaluationError):
        MetricSample(ttft_seconds=float("nan"), decode_tps=1)
    with pytest.raises(EvaluationError):
        MetricSample(ttft_seconds=True, decode_tps=1)


def test_strings_are_not_fingerprint_authority():
    with pytest.raises(EvaluationError):
        Evaluator().evaluate("fp", "fixed_compiled_cache", (), ())


def test_candidate_scope_and_parameter_gate_are_fail_closed():
    evaluator = Evaluator()
    unknown = evaluator.evaluate(fingerprint(), "unknown", (), ())
    assert unknown.status == "rejected"
    # width candidates are not valid for interactive workloads.
    decision = evaluator.evaluate(fingerprint(), "throughput_width_4", (), ())
    assert decision.status in {"ood", "rejected"}
    assert "candidate_scope" in decision.reasons[0]


def test_correctness_overrides_a_speed_gain_and_keeps_evidence():
    bad = (CorrectnessResult(token_ids=(1, 2), text="a", stop_reason="eos", physical_tokens=2, visible_tokens=2, response_hash="a"),
           CorrectnessResult(token_ids=(1, 3), text="a", stop_reason="eos", physical_tokens=2, visible_tokens=2, response_hash="b"))
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"),
        **aa_kwargs(), correctness=bad, resources=resource(),
    )
    assert result.status == "rejected"
    assert any("correctness_mismatch" in reason for reason in result.reasons)
    assert len(result.evidence_hash) == 64


def test_ttft_improve_decode_regress_is_rejected():
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(1, 8, arm="candidate"),
        **aa_kwargs(), correctness=correctness(), resources=resource(),
    )
    assert result.status == "rejected"


def test_both_metrics_improve_and_shadow_never_activates():
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"),
        **aa_kwargs(), correctness=correctness(), resources=resource(),
    )
    assert result.status == "qualified"
    assert result.qualified and result.no_activation


def test_single_aa_or_wide_ci_is_inconclusive():
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=samples(1, 10, 1), aa_right=samples(1, 10, 1, arm="aa_right"), bootstrap_resamples=100)
    baseline = samples(1, 10, arm="baseline")
    candidate = tuple(MetricSample(session_id=str(i), pair_id=str(i), arm="candidate", order=("AB" if i % 2 == 0 else "BA"), fingerprint=fingerprint().fingerprint_hash, workload="workload", ttft_seconds=(.4 if i % 2 else 1.4), decode_tps=(6 if i % 2 else 14)) for i in range(6))
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", baseline, candidate,
        **aa_kwargs(), correctness=correctness(), resources=resource(),
    )
    assert result.status == "inconclusive"


def test_aa_old_positional_api_and_structural_pair_errors_are_rejected():
    left = samples(1, 10, arm="aa_left")
    right = samples(1, 10, arm="aa_right")
    with pytest.raises(TypeError):
        calibrate_aa(left, right)  # type: ignore[call-arg]
    with pytest.raises(EvaluationError):
        swapped_orders = (replace(right[0], pair_id="1"), replace(right[1], pair_id="0"), *right[2:])
        calibrate_aa(aa_left=left, aa_right=swapped_orders)
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=left, aa_right=right[:-1] + (replace(right[-1], pair_id=right[-2].pair_id),))
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=left, aa_right=tuple(replace(item, pair_id="") for item in right))
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=left, aa_right=tuple(replace(item, order="") for item in right))
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=left, aa_right=tuple(replace(item, order="AB") for item in right))


def test_aa_pairing_is_order_independent_but_evidence_bound():
    left = samples(1, 10, arm="aa_left")
    right = samples(1, 10, arm="aa_right")
    first = calibrate_aa(aa_left=left, aa_right=right, bootstrap_resamples=100)
    second = calibrate_aa(aa_left=tuple(reversed(left)), aa_right=tuple(reversed(right)), bootstrap_resamples=100)
    assert first == second
    mismatched = tuple(replace(item, fingerprint="other") for item in right)
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=left, aa_right=mismatched)
    split_workload = tuple(replace(item, workload="other") if item.pair_id == "0" else item for item in right)
    with pytest.raises(EvaluationError):
        calibrate_aa(aa_left=left, aa_right=split_workload)


def test_resource_errors_are_retained_and_unknown_is_inconclusive():
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"),
        **aa_kwargs(), correctness=correctness(), resources=(ResourceResult(peak_memory_bytes=100, peak_rss_bytes=200, swap_delta_bytes=1024, status="resource", error="swap"),),
    )
    assert result.status == "rejected"
    unknown = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"),
        **aa_kwargs(), correctness=correctness(), resources=(ResourceResult(status="unknown"),),
    )
    assert unknown.status == "inconclusive"


def test_forged_calibration_report_cannot_override_raw_aa():
    raw = calibrate_aa(aa_left=samples(1, 10), aa_right=samples(1, 10, arm="aa_right"), bootstrap_resamples=100)
    forged = CalibrationReport(raw.pair_count, 0.0, 0.0, 0.001, True, raw.confidence_intervals, (), raw.evidence_hash)
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"),
        calibration=forged, **aa_kwargs(), correctness=correctness(), resources=resource(),
    )
    assert result.status == "inconclusive"
    assert "aa_report_does_not_match_raw_evidence" in result.reasons


def test_negative_resource_value_is_a_hard_safety_failure():
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"),
        **aa_kwargs(), correctness=correctness(), resources=(ResourceResult(peak_memory_bytes=-1, peak_rss_bytes=200, swap_delta_bytes=0, status="ok"),),
    )
    assert result.status == "rejected"
    assert "resource_or_safety_gate_failed" in result.reasons


def test_pair_identity_and_balanced_order_are_mandatory():
    baseline = samples(1, 10, arm="baseline", orders=False)
    candidate = samples(.8, 12, arm="candidate", orders=False)
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", baseline, candidate,
        **aa_kwargs(), correctness=correctness(), resources=resource(),
    )
    assert result.status == "rejected"
    assert "ab_order_unbalanced" in result.reasons


def test_pair_order_mismatch_is_rejected_even_when_counts_match():
    baseline = samples(1, 10, arm="baseline")
    candidate = tuple(MetricSample(session_id=str(i), pair_id=str(i), arm="candidate", order=("BA" if i % 2 == 0 else "AB"), fingerprint=fingerprint().fingerprint_hash, workload="workload", ttft_seconds=.8, decode_tps=12) for i in range(6))
    result = Evaluator(bootstrap_resamples=100).evaluate(
        fingerprint(), "fixed_compiled_cache", baseline, candidate,
        **aa_kwargs(), correctness=correctness(), resources=resource(),
    )
    assert result.status == "rejected"
    assert "ab_pair_order_mismatch" in result.reasons


def test_ranking_and_evidence_hash_are_reproducible():
    evaluator = Evaluator(bootstrap_resamples=100)
    args = (fingerprint(), "fixed_compiled_cache", samples(1, 10, arm="baseline"), samples(.8, 12, arm="candidate"))
    kwargs = {**aa_kwargs(), "resources": resource(), "correctness": correctness()}
    first = evaluator.evaluate(*args, **kwargs)
    second = evaluator.evaluate(*args, **kwargs)
    assert first == second
    assert evaluator.rank((first, second)) == evaluator.rank((second, first))
