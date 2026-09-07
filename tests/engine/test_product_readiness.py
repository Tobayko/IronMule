"""Pure readiness policy tests plus one bounded native probe smoke."""

from __future__ import annotations

import math

import pytest

import ironmule_product.readiness as readiness_module
from ironmule_product.readiness import ReadinessPolicy, ReadinessWindow, evaluate, hardware_identity, probe


def snapshot(**overrides):
    value = {
        "observed_unix_ns": 1_000_000_000,
        "observed_monotonic": 100.0,
        "duration_seconds": 0.01,
        "platform_supported": True,
        "power_source": "ac",
        "low_power_mode": False,
        "thermal_state": 0,
        "cpu_count": 10,
        "load_1m": 2.0,
        "load_ratio": 0.2,
        "memory_total_bytes": 16 * 1024**3,
        "swap_used_bytes": 0,
        "memory_free_percent": 50.0,
        "errors": [],
    }
    value.update(overrides)
    return value


def test_native_probe_is_bounded_numeric_metadata_without_mlx_import():
    value = probe()
    assert set(value) == {
        "observed_unix_ns", "observed_monotonic", "duration_seconds", "platform_supported",
        "power_source", "low_power_mode", "thermal_state", "cpu_count", "load_1m", "load_ratio",
        "memory_total_bytes", "swap_used_bytes", "memory_free_percent", "errors",
    }
    assert type(value["observed_unix_ns"]) is int and value["observed_unix_ns"] > 0
    for field in ("observed_monotonic", "duration_seconds"):
        assert isinstance(value[field], float) and math.isfinite(value[field])
    assert value["power_source"] in {"ac", "battery", "unknown"}
    assert isinstance(value["errors"], list)


def test_native_hardware_identity_is_stable_and_redacted_to_gpu_fields():
    value = hardware_identity()
    assert set(value) == {"chip_name", "gpu_devices", "hardware_errors"}
    assert value["chip_name"] is None or isinstance(value["chip_name"], str)
    assert isinstance(value["gpu_devices"], list)
    for device in value["gpu_devices"]:
        assert set(device) == {"model", "cores", "metal_support"}
        assert device["model"] is None or isinstance(device["model"], str)
        assert device["cores"] is None or type(device["cores"]) is int
        assert device["metal_support"] is None or isinstance(device["metal_support"], str)
    assert isinstance(value["hardware_errors"], list)
    serialized = str(value).lower()
    assert "serial" not in serialized and "resolution" not in serialized and "ndrvs" not in serialized

    original_command = readiness_module._hardware_command
    readiness_module._hardware_command = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("identity must be cached"))
    try:
        assert hardware_identity() == value
    finally:
        readiness_module._hardware_command = original_command


def test_evaluate_accepts_only_fresh_complete_eligible_snapshot():
    policy = ReadinessPolicy()
    decision = evaluate(snapshot(), policy, now_monotonic=100.0)
    assert decision == {"eligible": True, "reasons": []}
    stale = evaluate(snapshot(), policy, now_monotonic=110.1)
    assert stale["eligible"] is False
    assert "sample_too_old" in stale["reasons"]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"power_source": "battery"}, "power_not_ac"),
        ({"low_power_mode": True}, "low_power_mode_unavailable_or_enabled"),
        ({"thermal_state": 2}, "thermal_state_not_allowed"),
        ({"load_ratio": 0.81}, "load_ratio_too_high"),
        ({"memory_free_percent": 9.9}, "memory_free_percent_too_low"),
        ({"load_ratio": float("nan")}, "invalid_load_ratio"),
        ({"errors": ["memory_free_probe_failed"]}, "probe_errors_present"),
    ],
)
def test_evaluate_rejects_ineligible_or_nonfinite_metadata(changes, reason):
    decision = evaluate(snapshot(**changes), ReadinessPolicy(), now_monotonic=100.0)
    assert decision["eligible"] is False
    assert reason in decision["reasons"]


def test_evaluate_rejects_missing_and_invalid_types():
    value = snapshot()
    del value["cpu_count"]
    value["memory_total_bytes"] = True
    decision = evaluate(value, ReadinessPolicy(), now_monotonic=100.0)
    assert decision["eligible"] is False
    assert "missing_cpu_count" in decision["reasons"]
    assert "invalid_memory_total_bytes" in decision["reasons"]


def test_evaluate_fails_closed_for_huge_numbers_unhashable_fields_and_bad_errors():
    value = snapshot(
        load_ratio=10**10_000,
        power_source=[],
        errors="",
    )
    decision = evaluate(value, ReadinessPolicy(), now_monotonic=100.0)
    assert decision["eligible"] is False
    assert "invalid_load_ratio" in decision["reasons"]
    assert "invalid_power_source" in decision["reasons"]
    assert "invalid_errors" in decision["reasons"]


def test_evaluate_rejects_inconsistent_load_ratio():
    decision = evaluate(snapshot(load_1m=10.0, cpu_count=100, load_ratio=0.2), ReadinessPolicy(), now_monotonic=100.0)
    assert decision["eligible"] is False
    assert "load_ratio_inconsistent" in decision["reasons"]


def test_policy_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        ReadinessPolicy(max_gap_s=1.0, sample_interval_s=2.0)
    with pytest.raises(ValueError):
        ReadinessPolicy(max_load_ratio=float("inf"))
    with pytest.raises(ValueError):
        ReadinessPolicy(sample_interval_s=0)
    with pytest.raises(ValueError):
        ReadinessPolicy(max_sample_age_s=0)


def test_window_requires_three_spaced_samples_and_ignores_fast_duplicates():
    window = ReadinessWindow()
    assert window.observe(snapshot(observed_monotonic=100.0), now_monotonic=100.0)["consecutive"] == 1
    duplicate = window.observe(snapshot(observed_monotonic=100.0), now_monotonic=100.1)
    assert duplicate["consecutive"] == 1 and duplicate["stable"] is False
    assert window.observe(snapshot(observed_monotonic=105.0), now_monotonic=105.0)["consecutive"] == 2
    final = window.observe(snapshot(observed_monotonic=110.0), now_monotonic=110.0)
    assert final["consecutive"] == 3 and final["stable"] is True
    replay = window.observe(snapshot(observed_monotonic=110.0), now_monotonic=110.1)
    assert replay["consecutive"] == 3 and replay["stable"] is False
    fast = window.observe(snapshot(observed_monotonic=111.0), now_monotonic=111.0)
    assert fast["consecutive"] == 3 and fast["stable"] is False


def test_window_resets_on_ineligible_gap_and_out_of_order_samples():
    window = ReadinessWindow()
    window.observe(snapshot(observed_monotonic=100.0), now_monotonic=100.0)
    assert window.observe(snapshot(observed_monotonic=105.0), now_monotonic=105.0)["consecutive"] == 2
    reset = window.observe(snapshot(observed_monotonic=110.0, power_source="battery"), now_monotonic=110.0)
    assert reset["consecutive"] == 0 and reset["stable"] is False
    assert window.observe(snapshot(observed_monotonic=111.0), now_monotonic=111.0)["consecutive"] == 1

    window = ReadinessWindow()
    window.observe(snapshot(observed_monotonic=100.0), now_monotonic=100.0)
    gap = window.observe(snapshot(observed_monotonic=120.0), now_monotonic=120.0)
    assert gap["consecutive"] == 1 and "sampling_gap_too_large" in gap["reasons"]
    out_of_order = window.observe(snapshot(observed_monotonic=119.0), now_monotonic=120.0)
    assert out_of_order["consecutive"] == 0 and "out_of_order_sample" in out_of_order["reasons"]
