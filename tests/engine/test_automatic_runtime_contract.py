"""Metadata-only contract tests; no fake runtime or performance claim."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from friday_evidence.canonical import canonical_sha256
from ironmule_product.automatic_runtime import AutomaticRuntime
from ironmule_product.types import ModelSpec


SHA = "a" * 64


def identity(**changes: object) -> dict[str, object]:
    code_files = {"ironmule_product/example.py": "1" * 64}
    environment = {"python": "test", "platform": "test", "packages": {}}
    hardware = {"platform": "test", "gpu_devices": []}
    value: dict[str, object] = {
        "model_id": "local/model", "revision": "immutable/rev",
        "model_sha256": SHA, "hardware_sha256": canonical_sha256(hardware),
        "environment_sha256": canonical_sha256(environment),
        "code_sha256": canonical_sha256(code_files),
        "model_files": [], "code_files": code_files,
        "environment": environment, "hardware": hardware, "weight_bytes": 1,
    }
    value.update(changes)
    value["identity_sha256"] = canonical_sha256(value)
    return value


def evidence(**changes: object) -> dict[str, object]:
    bound = identity()
    value: dict[str, object] = {
        "audit_id": "audit-1", "candidate_id": "core_interactive",
        "baseline_candidate_id": "reference", "model_id": "local/model",
        "model_revision": "immutable/rev", "model_sha256": bound["model_sha256"],
        "hardware_sha256": bound["hardware_sha256"],
        "environment_sha256": bound["environment_sha256"],
        "code_sha256": bound["code_sha256"],
        "source_manifest_sha256": bound["code_sha256"], "heldout": True,
        "quality_exact": True,
        "profile": "interactive", "min_session_requests": 1,
        "max_session_requests": 1, "min_context_tokens": 2,
        "max_context_tokens": 2, "min_new_tokens": 1, "max_new_tokens": 8,
        "streamable": False, "group_widths": [1], "prefix_hit_capable": False,
        "prefix_hit_required": False, "cold_setup_included": False,
        "estimated_e2e_ms": 8.0, "pair_ratio": .8,
        "confidence_upper": .88, "frozen_min_effect": .05,
    }
    value.update(changes)
    return value


def runtime(rows=(), **kwargs: object) -> AutomaticRuntime:
    spec = ModelSpec("local/model", "immutable/rev", "/private/tmp/model", 1)
    return AutomaticRuntime(object(), object(), spec, identity(), rows,
                            prefix_cache_max_entries=2,
                            prefix_cache_max_bytes=1024, **kwargs)


def test_actual_context_selects_only_exact_matching_evidence() -> None:
    item = runtime([evidence()])
    try:
        decision = item.choose([4, 5], 8)
        assert decision.candidate_id == "core_interactive"
        assert item.last_decision == {
            "candidate_id": "core_interactive",
            "reason": "fastest_proven_matching_e2e_candidate",
            "evidence_audit_id": "audit-1", "explored": False,
            "rejected_evidence_records": 0,
        }
    finally:
        item.close()


def test_unknown_or_malformed_records_fail_closed_without_leaking_context() -> None:
    item = runtime([evidence(extra="unknown"), {"unknown": "record"}])
    try:
        assert item.choose([11, 12], 8).candidate_id == "reference"
        assert item.last_decision["rejected_evidence_records"] == 2
        assert "prompt" not in repr(item.last_decision)
        assert "scope" not in repr(item.last_decision)
    finally:
        item.close()


def test_empty_and_mixed_batches_use_stock_reference() -> None:
    item = runtime([evidence(profile="throughput", group_widths=[2],
                             min_session_requests=2, max_session_requests=2),
                    evidence(audit_id="wide", profile="throughput", group_widths=[2],
                             min_session_requests=2, max_session_requests=2,
                             min_context_tokens=3, max_context_tokens=3)])
    try:
        assert item.choose_many([], 8).candidate_id == "reference"
        mixed = item.choose_many([[1, 2], [1, 2, 3]], 8)
        assert mixed.candidate_id == "reference"
        assert mixed.reason == "mixed_batch_has_no_common_eligible_candidate"
    finally:
        item.close()


def test_execution_manifest_is_distinct_and_defaults_to_runtime_code_digest() -> None:
    item = runtime([evidence(source_manifest_sha256="b" * 64)])
    try:
        assert item.choose([1, 2], 8).candidate_id == "reference"
    finally:
        item.close()
    item = runtime([evidence(source_manifest_sha256="b" * 64)],
                   source_manifest_sha256="b" * 64)
    try:
        assert item.choose([1, 2], 8).candidate_id == "core_interactive"
    finally:
        item.close()


def test_identity_and_engine_candidate_validation_fail_closed() -> None:
    spec = ModelSpec("local/model", "immutable/rev", "/private/tmp/model", 1)
    with pytest.raises(ValueError, match="complete runtime_identity"):
        AutomaticRuntime(object(), object(), spec, {"model_id": "local/model"}, [],
                         prefix_cache_max_entries=1, prefix_cache_max_bytes=1)
    tampered = identity()
    tampered["code_sha256"] = "b" * 64
    tampered["identity_sha256"] = canonical_sha256({
        key: value for key, value in tampered.items() if key != "identity_sha256"
    })
    with pytest.raises(ValueError, match="component digest"):
        AutomaticRuntime(object(), object(), spec, tampered, [],
                         prefix_cache_max_entries=1, prefix_cache_max_bytes=1)
    item = runtime()
    try:
        with pytest.raises(ValueError, match="explicit B39"):
            item.engine_for("current_profile")
        with pytest.raises(ValueError, match="explicit B39"):
            item.engine_for("reference")
    finally:
        item.close()


def test_batch_defaults_bind_actual_request_count_and_reject_undercount() -> None:
    item = runtime([evidence(profile="throughput", group_widths=[2],
                             min_session_requests=2, max_session_requests=2)])
    try:
        assert item.choose_many([[1, 2], [1, 2]], 8).candidate_id == "core_interactive"
        with pytest.raises(ValueError, match="undercount"):
            item.choose_many([[1, 2], [1, 2]], 8, session_requests=1)
    finally:
        item.close()


def test_import_is_metadata_only() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    command = ("import sys; " f"sys.path.insert(0, {str(root)!r}); "
               "import ironmule_product.automatic_runtime; "
               "assert 'mlx' not in sys.modules; "
               "assert 'ironmule.runtime' not in sys.modules; "
               "assert 'ironmule.service' not in sys.modules")
    completed = subprocess.run([sys.executable, "-I", "-c", command],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
