from types import SimpleNamespace

import pytest
import research.b27_main_baseline as baseline_module

from research.b27_main_baseline import (
    classify,
    parse_memory_free_percent,
    parse_swap_used_bytes,
    select_cached_snapshot,
)


def _benchmark(*, identical=True, fallbacks=0, errors=0):
    raw = [{
        "phase": "measure",
        "snapshot": {"fallbacks": fallbacks, "correctness_errors": errors},
    }]
    return {
        "comparison": {"token_identity": identical},
        "arms": {"interactive": {"raw": raw}, "throughput": {"raw": raw}},
    }


def test_preflight_parsers_are_explicit_and_binary():
    assert parse_swap_used_bytes(
        "vm.swapusage: total = 8192.00M used = 12.50M free = 8179.50M"
    ) == 12.5 * 1024 * 1024
    assert parse_swap_used_bytes("unknown") is None
    assert parse_memory_free_percent("System-wide memory free percentage: 93%") == 93
    assert parse_memory_free_percent("unknown") is None


def test_baseline_classification_fails_closed():
    before = {"swap_used_bytes": 0}
    after = {"swap_used_bytes": 10}
    assert classify(_benchmark(), before, after, swap_delta_ceiling_bytes=20) == (
        "BASELINE_CAPTURED", []
    )

    status, failures = classify(
        _benchmark(identical=False, fallbacks=1, errors=1),
        before,
        {"swap_used_bytes": 30},
        swap_delta_ceiling_bytes=20,
    )
    assert status == "INCONCLUSIVE"
    assert failures == ["token_identity", "fallbacks", "correctness_errors", "swap_delta"]


def test_missing_swap_evidence_is_inconclusive():
    status, failures = classify(
        _benchmark(), {"swap_used_bytes": None}, {"swap_used_bytes": None},
        swap_delta_ceiling_bytes=1,
    )
    assert status == "INCONCLUSIVE"
    assert failures == ["swap_unavailable"]


def test_cached_snapshot_selection_is_exact_and_does_not_download(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    cache = SimpleNamespace(repos=[
        SimpleNamespace(
            repo_id="org/model",
            revisions=[
                SimpleNamespace(commit_hash="abc", snapshot_path=snapshot),
            ],
        ),
    ])
    assert select_cached_snapshot(cache, "org/model", "abc") == snapshot.resolve()
    with pytest.raises(RuntimeError, match="found 0"):
        select_cached_snapshot(cache, "org/model", "other")


def test_cli_propagates_explicit_experiment_id(monkeypatch, tmp_path):
    observed = {}

    def fake_run(args):
        observed["experiment_id"] = args.experiment_id
        observed["runtime_root"] = args.runtime_root
        return {"status": "BASELINE_CAPTURED", "failures": [], "elapsed_seconds": 0.0}

    monkeypatch.setattr(baseline_module, "run", fake_run)
    status = baseline_module.main([
        "--model", "org/model", "--revision", "abc", "--experiment-id", "B27d",
        "--runtime-root", str(tmp_path), "--output", str(tmp_path / "result.json"),
    ])
    assert status == 0
    assert observed == {"experiment_id": "B27d", "runtime_root": tmp_path}
