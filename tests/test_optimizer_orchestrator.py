from __future__ import annotations

from pathlib import Path

from friday_optimizer.orchestrator import OptimizerConfig, OptimizerOrchestrator


def test_doctor_and_status_are_read_only_on_fresh_root(tmp_path: Path) -> None:
    config = OptimizerConfig(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    orchestrator = OptimizerOrchestrator(config)
    assert orchestrator.doctor().ok
    assert not orchestrator.status().memory_exists
    after = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    assert before == after


def test_shadow_requires_exact_fingerprint_and_never_activates() -> None:
    from friday_optimizer.evaluator import MetricSample

    orchestrator = OptimizerOrchestrator(root=".")
    try:
        orchestrator.shadow(None, "baseline", (), ())
    except TypeError:
        pass
    else:
        raise AssertionError("shadow must require ExactFingerprint")


def test_shadow_request_binds_dataset_and_code_hashes(tmp_path: Path) -> None:
    from friday_optimizer.orchestrator import ShadowRequest
    from friday_optimizer.fingerprint import EnvironmentFingerprint, ExactFingerprint, ModelFingerprint, WorkloadFingerprint
    fp = ExactFingerprint(
        EnvironmentFingerprint("M1", "GPU", 1, 1, "14", "0.32", "0.31", "3", "a" * 64),
        ModelFingerprint("gemma-1b", "r", "b" * 64, "gemma", 4, 64, "tok"),
        WorkloadFingerprint("w", "tok", "gen", "short", 1, 1, 8, True, False, "performance", "interactive"),
    )
    import pytest
    with pytest.raises(ValueError):
        ShadowRequest(fp, "baseline")
    request = ShadowRequest(fp, "baseline", dataset_hash="c" * 64, code_hash="d" * 64)
    assert request.dataset_hash == "c" * 64


def test_shadow_request_roundtrip_is_canonical(tmp_path: Path) -> None:
    from friday_optimizer.evaluator import MetricSample
    from friday_optimizer.fingerprint import EnvironmentFingerprint, ExactFingerprint, ModelFingerprint, WorkloadFingerprint
    from friday_optimizer.orchestrator import ShadowRequest
    fp = ExactFingerprint(EnvironmentFingerprint("M1", "GPU", 1, 1, "14", "0.32", "0.31", "3", "a" * 64), ModelFingerprint("gemma-1b", "r", "b" * 64, "gemma", 4, 64, "tok"), WorkloadFingerprint("w", "tok", "g", "short", 1, 1, 8, True, False, "performance", "interactive"))
    sample = MetricSample("s", "p", "baseline", "AB", fp.fingerprint_hash, "w", 1.0, 2.0)
    request = ShadowRequest(fp, "persistent_process", (sample,), (sample,), (sample,), (sample,), dataset_hash="c" * 64, code_hash="d" * 64, parameters={"x": 1}, qualified=("baseline",))
    replay = ShadowRequest.from_dict(request.to_dict())
    assert replay == request
    assert replay.canonical_bytes == request.canonical_bytes
    assert replay.request_hash == request.request_hash


def test_config_paths_are_frozen(tmp_path: Path) -> None:
    config = OptimizerConfig(tmp_path)
    try:
        config.memory_path = tmp_path / "other"  # type: ignore[misc]
    except Exception as exc:
        assert type(exc).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("config paths must be immutable")


def test_inventory_identity_is_compact_and_changes_for_record_or_issue_mutation(tmp_path: Path) -> None:
    from dataclasses import replace
    from friday_optimizer.corpus import CorpusIssue, CorpusInventory, normalize_record
    from friday_optimizer.orchestrator import _inventory_identity_hash
    record = normalize_record({"latency": 1}, source_path="relative.json", source_sha256="a" * 64)
    inventory = CorpusInventory(str(tmp_path), (), (record,))
    first = _inventory_identity_hash(inventory)
    changed_record = replace(record, content_fingerprint="b" * 64)
    changed = _inventory_identity_hash(replace(inventory, records=(changed_record,)))
    changed_payload = _inventory_identity_hash(replace(inventory, records=(replace(record, data={"latency": 2}),)))
    issue_changed = _inventory_identity_hash(replace(inventory, issues=(CorpusIssue("x", "code", "detail"),)))
    assert len(first) == 64
    assert first != changed
    assert first != changed_payload
    assert first != issue_changed
