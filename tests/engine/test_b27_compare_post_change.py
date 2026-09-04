import copy

from research.b27_compare_post_change import bootstrap_ratio, compare


def record(model="org/model", *, wall=10.0, rate=20.0):
    raw = []
    for value in (0.99, 1.0, 1.01, 1.0):
        raw.append({
            "phase": "measure",
            "snapshot": {
                "outer_wall_ms": wall * value,
                "physical_tokens_per_second": rate / value,
            },
        })
    arms = {
        arm: {"raw": copy.deepcopy(raw)} for arm in ("interactive", "throughput")
    }
    result = {
        "status": "BASELINE_CAPTURED",
        "model_binding": {
            "model_id": model, "revision": "rev", "model_manifest_sha256": "m" * 64,
            "architecture": "arch", "quantisation": {"bits": 4, "group_size": 64},
        },
        "environment": {
            "python": "3.12", "mlx": "1", "mlx_lm": "2", "os": "3", "power_source": "AC",
            "low_power_mode": False, "thermal": {},
        },
        "protocol": {
            "requests": 6, "max_tokens": 48, "warmup": 2, "repeats": 4,
            "plan": "strict", "knobs": "BASELINE", "offline_cached_snapshot_only": True,
        },
        "runtime_binding": {"git_head": "commit", "runtime_tree_sha256": "t" * 64},
        "benchmark": {"comparison": {"token_identity": True}, "arms": arms},
        "system_before": {"swap_used_bytes": 0, "memory_free_percent": 90},
        "resource_summary": {
            "fallbacks": 0, "correctness_errors": 0, "swap_delta_bytes": 0,
            "mlx_peak_memory_bytes": 100,
        },
    }
    for arm in arms.values():
        arm["runtime_fingerprint"] = {
            "chip": "Apple Test", "machine": "arm64", "memory_bytes": 32 * 1024**3,
            "gpu_cores": 32, "hardware_fingerprint": "hardware", "runtime_version": "0.1.0",
        }
    return result


def test_bootstrap_ratio_is_deterministic_and_centered():
    first = bootstrap_ratio([1, 2, 3], [1, 2, 3], resamples=100, seed=7)
    second = bootstrap_ratio([1, 2, 3], [1, 2, 3], resamples=100, seed=7)
    assert first == second
    assert first["median_ratio"] == 1.0


def test_identical_domain_and_samples_have_no_observed_regression():
    before = record()
    after = copy.deepcopy(before)
    after["runtime_binding"] = {"git_head": "new", "runtime_tree_sha256": "n" * 64}
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "NO_REGRESSION_OBSERVED"
    assert result["regression_kind"] == "NONE"
    assert not result["activation_allowed"] and not result["valid_for_qualification"]


def test_domain_change_is_evidence_drift_before_performance_interpretation():
    before = record()
    after = copy.deepcopy(before)
    after["environment"]["mlx"] = "changed"
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "REVALIDATION_REQUIRED"
    assert result["regression_kind"] == "EVIDENCE_DRIFT"
    assert result["cells"][0]["domain_drift"] == ["mlx"]


def test_system_condition_class_change_requires_revalidation():
    before = record()
    after = copy.deepcopy(before)
    after["environment"]["low_power_mode"] = True
    after["system_before"] = {"swap_used_bytes": 1024, "memory_free_percent": 50}
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "REVALIDATION_REQUIRED"
    assert set(result["cells"][0]["domain_drift"]) == {
        "low_power_mode", "memory_free_class", "swap_preflight_class",
    }


def test_slow_same_domain_result_is_potential_regression_not_cherry_picked():
    before = record()
    after = record(wall=12.0, rate=16.0)
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "INCONCLUSIVE_POTENTIAL_REGRESSION"
    assert result["regression_kind"] == "POTENTIAL_CODE_REGRESSION"
    assert result["cells"][0]["performance_misses"]


def test_correctness_failure_is_code_regression_in_same_domain():
    before = record()
    after = copy.deepcopy(before)
    after["benchmark"]["comparison"]["token_identity"] = False
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "CODE_REGRESSION"
    assert result["cells"][0]["hard_failures"] == ["token_identity"]
