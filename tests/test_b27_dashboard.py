from research.b27_dashboard import render


def test_dashboard_is_local_escaped_and_includes_history():
    summary = {
        "schema": "ironmule.main_baseline.public.v1",
        "status": "BASELINE_CAPTURED",
        "base_commit": "abc",
        "runtime_tree_sha256": "def",
        "corpus_inventory": {"unique_artifacts": 2, "artifacts": 3, "local_only_or_ignored": 1},
        "tests": {"non_integration": "2 passed"},
        "premeasurement_failures": [{"type": "Bad<Resolver>", "stage": "binding"}],
        "cells": [{
            "status": "BASELINE_CAPTURED",
            "model": {"id": "org/<model>", "revision": "1234567890abcdef"},
            "comparison": {
                "wall_ratio_throughput_over_interactive": {"median": 0.9, "ci_low": 0.8, "ci_high": 0.95},
                "physical_rate_ratio_throughput_over_interactive": {"median": 1.1, "ci_low": 1.05, "ci_high": 1.2},
            },
            "interactive": {"outer_wall_ms_median": 10},
            "throughput": {"outer_wall_ms_median": 9},
            "resources": {"mlx_peak_memory_bytes": 1_000_000_000, "swap_delta_bytes": 0},
        }],
    }
    page = render(summary, {
        "ok": True, "errors": [], "checked_cells": 1, "checked_failures": 1,
    })
    assert "Bad&lt;Resolver&gt;" in page
    assert "org/&lt;model&gt;" in page
    assert "Pre-measurement" in page and "Baseline cell" in page
    assert "Evidence integrity" in page and "VERIFIED" in page
    assert "http://" not in page and "https://" not in page
    assert "<script" not in page
