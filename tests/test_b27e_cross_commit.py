import copy
import json
from pathlib import Path

import pytest

from research.b27e_cross_commit import (
    _sha256,
    analyze,
    entries_from_parent,
    execution_surface_digest,
    parser,
)


def record(*, wall=100.0, rate=10.0):
    rows = [
        {"phase": "measure", "snapshot": {
            "outer_wall_ms": wall * scale,
            "physical_tokens_per_second": rate / scale,
        }} for scale in (.99, 1.0, 1.01, 1.0)
    ]
    arms = {
        arm: {
            "raw": copy.deepcopy(rows),
            "runtime_fingerprint": {
                "chip": "Apple Test", "machine": "arm64", "memory_bytes": 32,
                "gpu_cores": 32, "hardware_fingerprint": "hardware",
                "runtime_version": "0.1.0",
            },
        } for arm in ("interactive", "throughput")
    }
    return {
        "status": "BASELINE_CAPTURED",
        "b27e_target": {"commit": "", "label": ""},
        "model_binding": {
            "model_id": "org/model", "revision": "rev", "model_manifest_sha256": "m" * 64,
            "architecture": "arch", "quantisation": {"bits": 4, "group_size": 64},
        },
        "environment": {
            "python": "3.12", "mlx": "1", "mlx_lm": "2", "os": "3",
            "power_source": "AC", "low_power_mode": False, "thermal": {},
        },
        "protocol": {
            "requests": 6, "max_tokens": 48, "warmup": 2, "repeats": 4,
            "plan": "strict", "knobs": "BASELINE", "offline_cached_snapshot_only": True,
        },
        "benchmark": {"comparison": {"token_identity": True}, "arms": arms},
        "resource_summary": {
            "fallbacks": 0, "correctness_errors": 0, "swap_delta_bytes": 0,
        },
    }


def entries(*, d1_wall=100.0, d1_rate=10.0, flip_second=False):
    result = []
    orders = (("old", "d1"), ("d1", "old"))
    for block, order in enumerate(orders):
        for position, label in enumerate(order):
            if label == "old":
                value = record()
                commit = "old"
            else:
                wall = (90.0 if flip_second and block == 1 else d1_wall)
                rate = (11.0 if flip_second and block == 1 else d1_rate)
                value = record(wall=wall, rate=rate)
                commit = "d1"
            value["b27e_target"] = {"commit": commit, "label": label}
            result.append({
                "block": block, "position": position, "label": label,
                "sha256": ("a" if label == "old" else "b") * 64,
                "record": value,
            })
    return result


def test_execution_surface_digest_is_path_independent_and_content_bound(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "a.py").write_text("x = 1\n")
    files = ("pkg/a.py",)
    assert execution_surface_digest(first, files) == execution_surface_digest(second, files)
    (second / "pkg" / "a.py").write_text("x = 2\n")
    assert execution_surface_digest(first, files) != execution_surface_digest(second, files)


def test_mirrored_equal_commits_support_common_mode_drift_reading():
    result = analyze(entries(), old_commit="old", d1_commit="d1",
                     execution_surface_sha256="s" * 64)
    assert result["classification"] == "COMMITS_INDISTINGUISHABLE"
    assert result["b27d_consequence"] == "COMMON_MODE_TEMPORAL_DRIFT_SUPPORTED"
    assert not result["activation_allowed"] and not result["valid_for_qualification"]


def test_d1_slowdown_must_reproduce_in_both_mirrored_orders():
    result = analyze(entries(d1_wall=110.0, d1_rate=9.0), old_commit="old",
                     d1_commit="d1", execution_surface_sha256="s" * 64)
    assert result["classification"] == "D1_SLOWER_REPRODUCED"


def test_order_flip_remains_inconclusive():
    result = analyze(entries(d1_wall=110.0, d1_rate=9.0, flip_second=True),
                     old_commit="old", d1_commit="d1",
                     execution_surface_sha256="s" * 64)
    assert result["classification"] == "ORDER_OR_TEMPORAL_DRIFT"
    assert result["b27d_consequence"] == "B27D_REMAINS_INCONCLUSIVE"


def test_domain_or_correctness_failure_precedes_timing():
    changed = entries()
    changed[1]["record"]["environment"]["mlx"] = "changed"
    drift = analyze(changed, old_commit="old", d1_commit="d1",
                    execution_surface_sha256="s" * 64)
    assert drift["classification"] == "REVALIDATION_REQUIRED"

    failed = entries()
    failed[1]["record"]["benchmark"]["comparison"]["token_identity"] = False
    regression = analyze(failed, old_commit="old", d1_commit="d1",
                         execution_surface_sha256="s" * 64)
    assert regression["classification"] == "CODE_REGRESSION"


def test_parent_requires_explicit_artifact_date():
    argv = [
        "run", "--old-root", "/old", "--d1-root", "/d1",
        "--old-commit", "old", "--d1-commit", "d1",
        "--d1-evidence-sha256", "a" * 64, "--model", "org/model",
        "--revision", "rev", "--output-dir", "/out",
        "--raw-output", "/out/raw.json", "--public-output", "/out/public.json",
    ]
    with pytest.raises(SystemExit):
        parser().parse_args(argv)
    parsed = parser().parse_args([*argv, "--artifact-date", "20260829"])
    assert parsed.artifact_date == "20260829"
    with pytest.raises(SystemExit):
        parser().parse_args([*argv, "--artifact-date", "2026-08-29"])


def test_reanalysis_rejects_changed_child_artifact(tmp_path):
    child = tmp_path / "child.json"
    child.write_text("{}")
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({
        "children": [{
            "block": 0, "position": 0, "label": "old",
            "artifact_id": child.name, "sha256": _sha256(child),
        }],
    }))
    assert entries_from_parent(parent)[0]["record"] == {}
    child.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="hash mismatch"):
        entries_from_parent(parent)
