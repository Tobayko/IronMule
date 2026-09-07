"""Pure evaluator contracts using synthetic report metadata only."""

from __future__ import annotations

import copy
import math

from ironmule_product.calibration_plan import plan_id, schedule
from ironmule_product.evaluation import evaluate_report


HASH = "a" * 64


def _report(*, ratio: float = 0.9, completion_limit: bool = True) -> dict:
    identity = {
        "identity_sha256": HASH,
        "model_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
        "code_sha256": "d" * 64,
        "hardware_sha256": "e" * 64,
    }
    workers = [{"worker_index": i, "started": True, "closed": True, "pid": 1000 + i} for i in range(3)]
    worker_timings = [
        {"worker_index": 0, "load_started_monotonic": 0.0, "ready_monotonic": 1.0, "closed_monotonic": 200.0},
        {"worker_index": 1, "load_started_monotonic": 260.0, "ready_monotonic": 261.0, "closed_monotonic": 460.0},
        {"worker_index": 2, "load_started_monotonic": 520.0, "ready_monotonic": 521.0, "closed_monotonic": 720.0},
    ]
    samples = []
    for descriptor in schedule():
        limit = descriptor["limit"]
        completion = limit if completion_limit else max(1, limit - 1)
        traced = descriptor["trace_forwards"]
        reference = descriptor["variant"] == "reference"
        sample = {
            **descriptor,
            "status": "passed",
            "correctness": True,
            "tokens_match": True,
            "text_match": True,
            "identity_sha256": HASH,
            "token_sha256": "f" * 64,
            "output_sha256": "0" * 64,
            "http_wall_seconds": 1.0 if reference else ratio,
            "peak_memory_bytes": 100_000_000,
            "swap_used_bytes": 0,
            "prompt_tokens": 4,
            "completion_tokens": completion,
            "finish_reason": "length",
            "forward_count": ((10 + limit) if reference else (9 + limit)) if traced and completion_limit else (10 if reference else 10) if traced else None,
        }
        samples.append(sample)
    resource_events = []
    for index, sample in enumerate(samples):
        worker = sample["worker_index"]
        start = 1.0 + worker * 260.0 + (index - worker * 30) * 5.0
        resource_events.append({"sample_index": index, "start_monotonic": start,
                                "end_monotonic": start + 1.0, "upper_bound_seconds": 1.0})
    return {
        "schema": "ironmule.calibration.v1",
        "plan_id": plan_id(),
        "status": "measured",
        "identity_before": identity,
        "identity_after": copy.deepcopy(identity),
        "memory_total_bytes": 1_000_000_000,
        "swap_baseline_bytes": 0,
        "resource_valid": True,
        "workers": workers,
        "worker_timings": worker_timings,
        "samples": samples,
        "resource_events": resource_events,
        "budget": {
            "gpu_work_seconds": 90.0,
            "max_continuous_gpu_seconds": 1.0,
            "required_break_seconds": 360.0,
            "cooldown_seconds": 120.0,
            "wall_seconds": 721.0,
            "gpu_work_limit_seconds": 120.0,
            "continuous_gpu_limit_seconds": 6.0,
            "duty_cycle_limit": 0.25,
            "wall_limit_seconds": 1200.0,
            "candidate_cooldown_seconds": 60.0,
            "required_break_limit_seconds": 4.0,
        },
    }


def test_valid_fixture_returns_math_only_calibration_signal() -> None:
    result = evaluate_report(_report())
    assert result["verdict"] == "calibration_signal"
    assert result["activation_allowed"] is False
    assert result["performance_claim_allowed"] is False
    assert result["per_limit"]["1"]["cluster"]["median_ratio"] == 0.9


def test_boundary_resource_values_are_allowed() -> None:
    report = _report()
    for sample in report["samples"]:
        sample["peak_memory_bytes"] = 600_000_000
        sample["swap_used_bytes"] = 256 * 1024**2
    assert evaluate_report(report)["verdict"] == "calibration_signal"


def test_early_eos_requires_zero_forward_delta_and_cannot_signal_mechanism() -> None:
    result = evaluate_report(_report(completion_limit=False))
    assert result["verdict"] != "calibration_signal"


def test_no_gain_is_classified_without_activation() -> None:
    result = evaluate_report(_report(ratio=1.0))
    assert result["verdict"] == "no_gain"
    assert result["activation_allowed"] is False


def test_malformed_report_never_throws_and_fails_closed() -> None:
    report = _report()
    report["samples"][0]["http_wall_seconds"] = float("nan")
    report["workers"][1]["worker_index"] = report["workers"][0]["worker_index"]
    result = evaluate_report(report)
    assert result["verdict"] == "invalid"
    assert result["activation_allowed"] is False


def test_bool_and_infinite_boundaries_are_invalid() -> None:
    for key, value in (("resource_valid", 1), ("memory_total_bytes", True), ("swap_baseline_bytes", math.nan)):
        report = _report()
        report[key] = value
        assert evaluate_report(report)["verdict"] == "invalid"


def test_wrong_plan_and_schedule_are_invalid() -> None:
    report = _report()
    report["plan_id"] = "0" * 64
    assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["samples"][0]["variant"] = "unexpected"
    assert evaluate_report(report)["verdict"] == "invalid"


def test_boolean_descriptor_numbers_are_invalid() -> None:
    for key in ("sample_index", "worker_index", "limit"):
        report = _report()
        report["samples"][0][key] = True
        assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["samples"][0]["trace_forwards"] = 1
    assert evaluate_report(report)["verdict"] == "invalid"


def test_finish_reason_and_length_count_are_closed() -> None:
    report = _report()
    report["samples"][0]["finish_reason"] = "unexpected"
    assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["samples"][0]["finish_reason"] = "length"
    report["samples"][0]["completion_tokens"] = 0
    assert evaluate_report(report)["verdict"] == "invalid"


def test_conflicting_comparison_aliases_are_invalid() -> None:
    report = _report()
    report["samples"][0]["actual_tokens_match"] = False
    assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["samples"][0]["actual_text_match"] = False
    assert evaluate_report(report)["verdict"] == "invalid"


def test_resource_timeline_and_budget_accounting_are_mandatory() -> None:
    report = _report()
    report.pop("resource_events")
    assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["resource_events"][1]["start_monotonic"] = report["resource_events"][0]["end_monotonic"]
    assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["budget"]["gpu_work_seconds"] = 89.0
    assert evaluate_report(report)["verdict"] == "invalid"


def test_resource_upper_bound_and_worker_cooldown_are_closed() -> None:
    report = _report()
    report["resource_events"][0]["upper_bound_seconds"] = 0.5
    assert evaluate_report(report)["verdict"] == "invalid"
    report = _report()
    report["worker_timings"][1]["load_started_monotonic"] = 459.5
    assert evaluate_report(report)["verdict"] == "invalid"
