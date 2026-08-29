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
    page = render(
        summary,
        {"ok": True, "errors": [], "checked_cells": 1, "checked_failures": 1},
        {
            "classification": "INCONCLUSIVE_POTENTIAL_REGRESSION",
            "regression_kind": "POTENTIAL_CODE_REGRESSION",
            "cells": [{
                "model_id": "org/model",
                "performance_misses": ["interactive.outer_wall"],
                "comparisons": {
                    arm: {
                        "outer_wall_post_over_pre": {"median_ratio": 1.06, "ci_low": 1.05, "ci_high": 1.07},
                        "physical_rate_post_over_pre": {"median_ratio": .94, "ci_low": .93, "ci_high": .95},
                    } for arm in ("interactive", "throughput")
                },
            }],
        },
        {
            "ok": True,
            "byte_identical_recomputation": True,
            "tests": {"non_integration": "146 passed"},
        },
        {
            "classification": "ORDER_OR_TEMPORAL_DRIFT",
            "b27d_consequence": "B27D_REMAINS_INCONCLUSIVE",
            "blocks": [{
                "block": 0, "order": ["old", "d1"],
                "ratios": {
                    arm: {"d1_over_old_wall": .99, "d1_over_old_rate": 1.01}
                    for arm in ("interactive", "throughput")
                },
            }],
        },
        {"ok": True, "byte_identical_recomputation": True},
        {
            "classification": "BASELINE_CAPTURED",
            "cells": [{
                "model": {"model_id": "org/model", "revision": "revision"},
                "interactive": {
                    "outer_wall_ms": {"median": 10},
                    "physical_tokens_per_second": {"median": 20},
                },
                "throughput": {
                    "outer_wall_ms": {"median": 8},
                    "physical_tokens_per_second": {"median": 25},
                },
            }],
        },
        {"ok": True, "byte_identical_recomputation": True},
        {
            "classification": "BASELINE_CAPTURED",
            "cells": [{
                "model": {
                    "model_id": "org/model", "revision": "revision",
                    "runtime_identity": {"identity_sha256": "a" * 64},
                },
                "interactive": {
                    "outer_wall_ms": {"median": 10},
                    "physical_tokens_per_second": {"median": 20},
                },
                "throughput": {
                    "outer_wall_ms": {"median": 8},
                    "physical_tokens_per_second": {"median": 25},
                },
                "resources": {"mlx_peak_memory_bytes": 100, "swap_delta_bytes": 0},
            }],
        },
        {
            "classification": "NO_REGRESSION_OBSERVED",
            "regression_kind": "NONE",
            "cells": [{
                "model_id": "org/model", "performance_misses": [],
                "comparisons": {
                    arm: {
                        "outer_wall_post_over_pre": {
                            "median_ratio": 1.0, "ci_low": .99, "ci_high": 1.01,
                        },
                        "physical_rate_post_over_pre": {
                            "median_ratio": 1.0, "ci_low": .99, "ci_high": 1.01,
                        },
                    } for arm in ("interactive", "throughput")
                },
            }],
        },
        {
            "classification": "VERIFIED", "errors": [],
            "tests": {"non_integration": "178 passed"},
        },
    )
    assert "Bad&lt;Resolver&gt;" in page
    assert "org/&lt;model&gt;" in page
    assert "Pre-measurement" in page and "Baseline cell" in page
    assert "Evidence integrity" in page and "VERIFIED" in page
    assert "D1 post/pre regression screen" in page
    assert "INCONCLUSIVE_POTENTIAL_REGRESSION" in page
    assert "D1 comparison integrity" in page
    assert "Regression suite</small><strong>178 passed" in page
    assert "B27e mirrored OLD/D1 control" in page
    assert "ORDER_OR_TEMPORAL_DRIFT" in page
    assert "D2a same-day pre-change baseline" in page
    assert "D2a evidence integrity" in page
    assert "D2b measured post cells" in page
    assert "D2b exact-identity post/pre screen" in page
    assert "NO_REGRESSION_OBSERVED" in page
    assert "D2b evidence integrity" in page
    protected = page.split("D1 post/pre regression screen", 1)[0]
    assert "org/&lt;model&gt;" in protected
    assert "http://" not in page and "https://" not in page
    assert "<script" not in page
