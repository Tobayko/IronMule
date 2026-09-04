import copy
import hashlib
import json

from research.b27_verify_public_summary import verify


def _raw(model="org/model"):
    return {
        "schema": "ironmule.main_baseline.v1",
        "status": "BASELINE_CAPTURED",
        "activation_allowed": False,
        "valid_for_performance": True,
        "runtime_binding": {"git_head": "abc", "runtime_tree_sha256": "tree"},
        "model_binding": {
            "model_id": model, "revision": "rev", "model_manifest_sha256": "manifest",
            "architecture": "arch", "quantisation": {"bits": 4, "group_size": 64},
        },
        "environment": {"python": "3.12", "mlx": "1", "mlx_lm": "2", "os": "3", "power_source": "AC"},
        "protocol": {"requests": 2, "max_tokens": 8, "warmup": 1, "repeats": 2,
                     "knobs": "BASELINE", "offline_cached_snapshot_only": True,
                     "fresh_process_per_model": True},
        "benchmark": {
            "comparison": {
                "token_identity": True,
                "primary_wall_ratio_throughput_over_interactive": {"median_ratio": .9, "ci_low": .8, "ci_high": .95},
                "primary_rate_ratio_throughput_over_interactive": {"median_ratio": 1.1, "ci_low": 1.05, "ci_high": 1.2},
            },
            "arms": {
                "interactive": {
                    "summary": {"outer_wall_ms": {"median": 10}, "physical_tokens_per_second": {"median": 5}},
                    "runtime_fingerprint": {"chip": "Apple Test"},
                },
                "throughput": {
                    "summary": {"outer_wall_ms": {"median": 9}, "physical_tokens_per_second": {"median": 5.5}},
                },
            },
        },
        "resource_summary": {"mlx_peak_memory_bytes": 100, "swap_delta_bytes": 0,
                             "fallbacks": 0, "correctness_errors": 0},
    }


def _digest(value):
    # Tests provide the same digest as the synthetic public record; production main
    # computes it from the exact raw bytes before calling verify().
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _summary(raw, digest):
    return {
        "schema": "ironmule.main_baseline.public.v1",
        "status": "BASELINE_CAPTURED", "activation_allowed": False,
        "valid_for_qualification": False, "base_commit": "abc",
        "runtime_tree_sha256": "tree",
        "protocol": {"requests": 2, "max_tokens": 8, "warmup": 1, "repeats": 2,
                     "knobs": "BASELINE", "offline_cached_snapshots_only": True,
                     "fresh_process_per_model": True},
        "premeasurement_failures": [],
        "cells": [{
            "status": "BASELINE_CAPTURED", "raw_sha256": digest,
            "model": {"id": "org/model", "revision": "rev", "manifest_sha256": "manifest",
                      "architecture": "arch", "bits": 4, "group_size": 64},
            "environment": {"apple_chip": "Apple Test", "python": "3.12", "mlx": "1",
                            "mlx_lm": "2", "os": "3", "power": "AC"},
            "interactive": {"outer_wall_ms_median": 10, "physical_tokens_per_second_median": 5},
            "throughput": {"outer_wall_ms_median": 9, "physical_tokens_per_second_median": 5.5},
            "comparison": {
                "token_identity": True,
                "wall_ratio_throughput_over_interactive": {"median": .9, "ci_low": .8, "ci_high": .95},
                "physical_rate_ratio_throughput_over_interactive": {"median": 1.1, "ci_low": 1.05, "ci_high": 1.2},
            },
            "resources": {"mlx_peak_memory_bytes": 100, "swap_delta_bytes": 0,
                          "fallbacks": 0, "correctness_errors": 0},
        }],
    }


def test_verifier_matches_every_published_cell_field_and_stays_non_activating():
    raw = _raw()
    digest = _digest(raw)
    result = verify(_summary(raw, digest), [(digest, raw)], [])
    assert result == {
        "schema": "ironmule.public_summary_verification.v1",
        "ok": True, "errors": [], "checked_cells": 1, "checked_failures": 0,
        "activation_allowed": False, "qualification_changed": False,
        "raw_hashes": [digest], "failure_hashes": [],
    }


def test_verifier_rejects_changed_endpoint_and_local_path_leak():
    raw = _raw()
    digest = _digest(raw)
    summary = _summary(raw, digest)
    summary["cells"][0]["throughput"]["outer_wall_ms_median"] = 8
    summary["local_note"] = "/Users/example/private"
    result = verify(summary, [(digest, raw)], [])
    assert not result["ok"]
    assert any("throughput wall median" in error for error in result["errors"])
    assert any("local-path material" in error for error in result["errors"])


def test_verifier_requires_preserved_failure_hash_and_no_benchmark():
    raw = _raw()
    digest = _digest(raw)
    failure = {"status": "INCONCLUSIVE", "stage": "binding",
               "error": {"type": "ResolverError"}}
    failure_digest = _digest(failure)
    summary = _summary(raw, digest)
    summary["premeasurement_failures"] = [{
        "raw_sha256": failure_digest, "stage": "binding", "type": "ResolverError",
    }]
    assert verify(summary, [(digest, raw)], [(failure_digest, failure)])["ok"]

    bad = copy.deepcopy(failure)
    bad["benchmark"] = {"comparison": {}}
    result = verify(summary, [(digest, raw)], [(failure_digest, bad)])
    assert not result["ok"]
    assert any("benchmark evidence must be absent" in error for error in result["errors"])
