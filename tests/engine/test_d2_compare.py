import copy

from research.d2_compare import canonical_sha256, compare, derive_identity


def record(model_id="org/model", *, wall=10.0, rate=20.0, exact_identity=True):
    manifest = [
        {"path": "config.json", "bytes": 10, "sha256": "a" * 64},
        {"path": "model.safetensors", "bytes": 20, "sha256": "b" * 64},
        {"path": "tokenizer.json", "bytes": 30, "sha256": "c" * 64},
    ]
    binding = {
        "model_id": model_id, "revision": "rev",
        "model_manifest_sha256": canonical_sha256(manifest), "manifest": manifest,
        "architecture": "arch", "quantisation": {"bits": 4, "group_size": 64},
    }
    raw = [
        {"phase": "measure", "snapshot": {
            "outer_wall_ms": wall * value,
            "physical_tokens_per_second": rate / value,
        }}
        for value in (0.99, 1.0, 1.01, 1.0)
    ]
    arms = {arm: {"raw": copy.deepcopy(raw)} for arm in ("interactive", "throughput")}
    result = {
        "status": "BASELINE_CAPTURED", "model_binding": binding,
        "environment": {
            "python": "3.12", "mlx": "1", "mlx_lm": "2", "os": "3",
            "power_source": "AC", "low_power_mode": False, "thermal": {},
        },
        "protocol": {
            "requests": 6, "max_tokens": 48, "warmup": 2, "repeats": 4,
            "plan": "strict", "knobs": "BASELINE", "stored_profile_reuse": False,
            "offline_cached_snapshot_only": True,
        },
        "runtime_binding": {"git_head": "commit", "runtime_tree_sha256": "t" * 64},
        "benchmark": {"comparison": {"token_identity": True}, "arms": arms},
        "system_before": {"swap_used_bytes": 0, "memory_free_percent": 90},
        "resource_summary": {
            "fallbacks": 0, "correctness_errors": 0, "swap_delta_bytes": 0,
            "mlx_peak_memory_bytes": 100,
        },
    }
    for arm_name, arm in arms.items():
        arm["runtime_fingerprint"] = {
            "chip": "Apple Test", "machine": "arm64", "memory_bytes": 32 * 1024**3,
            "gpu_cores": 32, "hardware_fingerprint": "hardware", "runtime_version": "0.1.0",
            "service_mode": arm_name,
        }
    if exact_identity:
        attach_exact_identity(result)
    return result


def attach_exact_identity(result):
    identity = derive_identity(result["model_binding"])
    result["runtime_model_identity"] = identity
    fields = {
        "fingerprint_schema": "ironmule.runtime_fingerprint.v2",
        "model_id": identity["model_id"], "model_revision": identity["revision"],
        "model_manifest_sha256": identity["model_manifest_sha256"],
        "model_architecture": identity["architecture"],
        "quantisation": identity["quantisation"],
        "quantisation_sha256": identity["quantisation_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "model_identity_sha256": identity["identity_sha256"],
    }
    for arm in result["benchmark"]["arms"].values():
        arm["runtime_fingerprint"].update(fields)


def test_independent_identity_derivation_is_deterministic_and_complete():
    first = derive_identity(record()["model_binding"])
    second = derive_identity(record()["model_binding"])
    assert first == second
    assert first["schema"] == "ironmule.model_identity.v1"
    assert first["manifest_file_count"] == 3
    assert first["manifest_bytes"] == 60
    assert first["tokenizer_file_count"] == 1


def test_exact_post_identity_and_same_domain_pass():
    before = record(exact_identity=False)
    after = record()
    after["runtime_binding"] = {"git_head": "new", "runtime_tree_sha256": "n" * 64}
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "NO_REGRESSION_OBSERVED"
    assert result["cells"][0]["hard_failures"] == []
    assert result["cells"][0]["expected_model_identity"] == after["runtime_model_identity"]
    assert not result["activation_allowed"] and not result["valid_for_qualification"]


def test_missing_or_tampered_runtime_identity_is_hard_regression():
    before = record(exact_identity=False)
    missing = record(exact_identity=False)
    result = compare([("a" * 64, before)], [("b" * 64, missing)], resamples=200)
    assert result["classification"] == "CODE_REGRESSION"
    assert set(result["cells"][0]["hard_failures"]) == {
        "runtime_model_identity",
        "interactive.runtime_fingerprint_identity",
        "throughput.runtime_fingerprint_identity",
    }

    tampered = record()
    tampered["runtime_model_identity"]["tokenizer_sha256"] = "f" * 64
    result = compare([("a" * 64, before)], [("b" * 64, tampered)], resamples=200)
    assert result["classification"] == "CODE_REGRESSION"
    assert "runtime_model_identity" in result["cells"][0]["hard_failures"]


def test_domain_drift_precedes_identity_or_performance_interpretation():
    before = record(exact_identity=False)
    after = record(wall=12.0, rate=16.0)
    after["environment"]["mlx"] = "changed"
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "REVALIDATION_REQUIRED"
    assert result["cells"][0]["domain_drift"] == ["mlx"]
    assert result["cells"][0]["performance_misses"]


def test_manifest_claim_is_independently_recomputed():
    before = record(exact_identity=False)
    after = record()
    after["model_binding"]["manifest"][0]["bytes"] += 1
    result = compare([("a" * 64, before)], [("b" * 64, after)], resamples=200)
    assert result["classification"] == "CODE_REGRESSION"
    assert result["cells"][0]["hard_failures"] == ["model_binding_identity:ValueError"]
