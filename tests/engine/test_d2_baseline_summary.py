import copy

import pytest

from research.d2_baseline_summary import summarize


def record(model_id="org/model"):
    summary = {
        "outer_wall_ms": {"median": 10.0, "n": 6},
        "physical_tokens_per_second": {"median": 20.0, "n": 6},
    }
    return {
        "status": "BASELINE_CAPTURED",
        "model_binding": {
            "model_id": model_id, "revision": "rev", "model_manifest_sha256": "m" * 64,
            "architecture": "arch", "quantisation": {"bits": 4, "group_size": 64},
        },
        "runtime_binding": {"git_head": "commit", "runtime_tree_sha256": "t" * 64},
        "environment": {
            "python": "3.12", "mlx": "1", "mlx_lm": "2", "os": "3",
            "power_source": "AC", "low_power_mode": False,
        },
        "protocol": {
            "requests": 6, "max_tokens": 48, "warmup": 2, "repeats": 6,
            "plan": "strict", "knobs": "BASELINE", "stored_profile_reuse": False,
            "offline_cached_snapshot_only": True,
        },
        "benchmark": {
            "comparison": {"token_identity": True},
            "arms": {"interactive": {"summary": summary}, "throughput": {"summary": summary}},
        },
        "resource_summary": {
            "fallbacks": 0, "correctness_errors": 0, "swap_delta_bytes": 0,
        },
        "system_before": {"memory_free_percent": 90, "swap_used_bytes": 0},
    }


def test_summary_is_path_free_deterministic_and_requires_two_clean_cells():
    records = [("a" * 64, record("org/4b")), ("b" * 64, record("org/12b"))]
    first = summarize(records, phase="pre", experiment_id="D2a")
    second = summarize(reversed(records), phase="pre", experiment_id="D2a")
    assert first == second
    assert first["classification"] == "BASELINE_CAPTURED"
    assert not first["activation_allowed"] and not first["valid_for_qualification"]


def test_summary_retains_failures_and_rejects_paths():
    bad = record("org/12b")
    bad["benchmark"]["comparison"]["token_identity"] = False
    result = summarize(
        [("a" * 64, record("org/4b")), ("b" * 64, bad)],
        phase="post", experiment_id="D2b",
    )
    assert result["classification"] == "INCONCLUSIVE"
    failed = next(cell for cell in result["cells"] if cell["model"]["model_id"] == "org/12b")
    assert failed["hard_failures"] == ["token_identity"]

    leaked = copy.deepcopy(record("org/12b"))
    leaked["runtime_model_identity"] = {"source_root": "/Users/private"}
    with pytest.raises(ValueError, match="local path"):
        summarize(
            [("a" * 64, record("org/4b")), ("b" * 64, leaked)],
            phase="post", experiment_id="D2b",
        )
