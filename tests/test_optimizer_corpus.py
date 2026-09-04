from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from friday_optimizer.adapters import (
    DiscoveryLimits,
    SQLiteReadOnlyAdapter,
    discover_files,
    read_bounded_json,
    read_stable_bytes,
)
from friday_optimizer.corpus import CorpusAuditor, QualityClass, normalize_record, verify_archive_manifest


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_discovery_is_bounded_sorted_and_does_not_follow_symlinks(tmp_path: Path) -> None:
    root = tmp_path / ".friday-data"
    _write_json(root / "z.json", {"timing": {"p50": 1}, "model": "gemma-1b"})
    _write_json(root / "a.json", {"timing": {"p50": 2}, "model": "gemma-1b"})
    outside = tmp_path / "outside.json"
    _write_json(outside, {"timing": {"p50": 3}})
    (root / "link.json").symlink_to(outside)
    rows = discover_files(tmp_path)
    assert [row.relative_path for row in rows] == [".friday-data/a.json", ".friday-data/z.json"]


def test_discovery_limit_is_terminal_not_silent(tmp_path: Path) -> None:
    _write_json(tmp_path / ".friday-data" / "a.json", {"x": 1})
    _write_json(tmp_path / ".friday-data" / "b.json", {"x": 2})
    with pytest.raises(ValueError):
        discover_files(tmp_path, limits=DiscoveryLimits(max_files=1))
    audit = CorpusAuditor(tmp_path, limits=DiscoveryLimits(max_files=1)).audit()
    assert audit.issues and audit.issues[0].terminal and not audit.usable_records


def test_known_model_cache_prefix_is_explicitly_excluded(tmp_path: Path) -> None:
    cache = tmp_path / ".friday-data" / "models"
    cache.mkdir(parents=True)
    (cache / "weights.safetensors").write_bytes(b"x" * (1024 * 1024))
    outside = tmp_path / "outside.json"
    _write_json(outside, {"timing": 1})
    (cache / "escape.json").symlink_to(outside)
    rows = discover_files(tmp_path)
    assert any(row.excluded_reason == "known_local_model_cache" for row in rows)
    assert all("models/weights" not in row.relative_path for row in rows)
    inventory = CorpusAuditor(tmp_path).audit()
    assert inventory.exclusions == {"known_local_model_cache": 1}


def test_python_bytecode_cache_is_named_and_genuine_sources_remain(tmp_path: Path) -> None:
    root = tmp_path / "experiments" / "run"
    (root / "__pycache__").mkdir(parents=True)
    (root / "__pycache__" / "worker.cpython-312.pyc").write_bytes(b"cache")
    (root / "worker.py").write_text("print('evidence')", encoding="utf-8")
    _write_json(root / "result.json", {"status": "complete"})
    rows = discover_files(tmp_path)
    assert any(row.excluded_reason == "python_bytecode_cache" for row in rows)
    assert any(row.relative_path.endswith("worker.py") and row.excluded_reason is None for row in rows)
    assert any(row.relative_path.endswith("result.json") and row.excluded_reason is None for row in rows)
    assert CorpusAuditor(tmp_path).audit().exclusions == {"python_bytecode_cache": 1}


def test_optimizer_control_state_prefix_is_exact_and_nonrecursive_elsewhere(tmp_path: Path) -> None:
    data = tmp_path / ".friday-data"
    data.mkdir()
    _write_json(data / "optimizer-dataset-v1.json", {"control": True})
    (data / "optimizer-v2.sqlite3").write_bytes(b"not evidence")
    evidence = tmp_path / "experiments" / "optimizer-result.json"
    _write_json(evidence, {"timing": {"p50": 1}})
    rows = discover_files(tmp_path)
    assert {row.excluded_reason for row in rows if row.relative_path.startswith(".friday-data/optimizer-")} == {"optimizer_control_plane_state"}
    assert any(row.relative_path.endswith("optimizer-result.json") and row.excluded_reason is None for row in rows)
    inventory = CorpusAuditor(tmp_path).audit()
    assert inventory.exclusions["optimizer_control_plane_state"] == 2


def test_unknown_oversized_evidence_remains_terminal(tmp_path: Path) -> None:
    evidence = tmp_path / "experiments"
    evidence.mkdir()
    (evidence / "unexpected.json").write_bytes(b"x" * 32)
    with pytest.raises(ValueError):
        discover_files(tmp_path, limits=DiscoveryLimits(max_file_bytes=16))


def test_strict_json_rejects_duplicate_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_bounded_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":Infinity}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_bounded_json(nonfinite)


def test_stable_reader_rejects_swap_and_growth_after_bound_identity(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    _write_json(path, {"timing": 1})
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size)
    path.unlink()
    _write_json(path, {"timing": 123456})
    with pytest.raises(ValueError):
        read_bounded_json(path, expected_identity=identity)
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size)
    with path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError):
        read_stable_bytes(path, expected_identity=identity)


def test_sqlite_adapter_is_read_only_and_inventories_schema(tmp_path: Path) -> None:
    path = tmp_path / "evidence.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("create table records (record_id text, payload_json text)")
    conn.execute("insert into records values ('r1', '{\"timing\": 1}')")
    conn.commit()
    conn.close()
    before = path.read_bytes()
    with SQLiteReadOnlyAdapter(path) as adapter:
        inventory = adapter.inventory()
        assert inventory.tables[0].row_count == 1
        assert adapter.records()[0]["record_id"] == "r1"
        with pytest.raises(ValueError):
            adapter.known_rows("not_allowlisted")
    assert path.read_bytes() == before


def test_archive_manifest_verifies_dedupe_and_tamper(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    payload = b'{"timing":{"p50":3}}'
    digest = hashlib.sha256(payload).hexdigest()
    stored = archive / digest[:2] / f"{digest}-a.json"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(payload)
    manifest = archive / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {"file": "a.json", "sha256": digest, "bytes": len(payload), "stored_as": stored.relative_to(archive).as_posix()},
                    {"file": "a-copy.json", "sha256": digest, "bytes": len(payload), "duplicate_of": stored.relative_to(archive).as_posix()},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = verify_archive_manifest(manifest)
    assert result.ok and result.unique_content == 1 and result.duplicate_entries == 1
    stored.write_bytes(b"tampered")
    result = verify_archive_manifest(manifest)
    assert result.terminal and result.hash_mismatches


def test_partial_is_quarantined_and_never_eligible(tmp_path: Path) -> None:
    path = tmp_path / ".friday-data" / ".run.json.partial"
    _write_json(path, {"timing": {"p50": 1}, "model": "gemma-1b"})
    inventory = CorpusAuditor(tmp_path).audit()
    assert inventory.records[0].quality is QualityClass.QUARANTINED
    assert not inventory.records[0].training_eligible


def test_self_declared_feature_label_contract_is_not_registered() -> None:
    record = normalize_record(
        {"features": {"x": 1}, "labels": {"y": 2}, "study_id": "s", "run_id": "r", "observed_at": "t", "hardware_fingerprint": "h", "model_fingerprint": "m", "workload_fingerprint": "w", "prompt_family": "p"},
        source_sha256="0" * 64,
        source_verified=True,
        manifest_verified=True,
    )
    assert not record.contract_verified and not record.training_eligible


def test_contract_filename_match_is_exact_not_suffix() -> None:
    from friday_optimizer.corpus import Q2_PROFILES_CONTRACT
    assert Q2_PROFILES_CONTRACT.matches("evil-Q2_profiles.json", {}) is False
    assert Q2_PROFILES_CONTRACT.matches("Q2_profiles.json", {}) is True


def test_manifest_tamper_quarantines_every_archive_record(tmp_path: Path) -> None:
    archive = tmp_path / ".friday-data" / "ironmule-evidence-archive"
    payload = b'{"features":{"model":"gemma-1b"},"labels":{"timing":1},"study_id":"s","run_id":"r","observed_at":"t","hardware_fingerprint":"h","model_fingerprint":"m","workload_fingerprint":"w","prompt_family":"p"}'
    digest = hashlib.sha256(payload).hexdigest()
    stored = archive / digest[:2] / f"{digest}-record.json"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(payload)
    manifest = archive / "MANIFEST.json"
    manifest.write_text(json.dumps({"entries": [{"file": "record.json", "sha256": digest, "bytes": len(payload), "stored_as": stored.relative_to(archive).as_posix()}]}), encoding="utf-8")
    stored.write_bytes(payload + b"tamper")
    inventory = CorpusAuditor(tmp_path).audit()
    archive_records = [record for record in inventory.records if "ironmule-evidence-archive/" in record.source_path]
    assert archive_records and all(not record.training_eligible for record in archive_records)
    assert inventory.manifest is not None and inventory.manifest.terminal


def test_real_project_smoke_is_read_only() -> None:
    project = Path(__file__).resolve().parents[1]
    before = (project / ".friday-data/ironmule-evidence-archive/MANIFEST.json").read_bytes()
    inventory = CorpusAuditor(project).audit()
    after = (project / ".friday-data/ironmule-evidence-archive/MANIFEST.json").read_bytes()
    assert inventory.files and inventory.manifest is not None
    assert inventory.manifest.entries_checked >= 1
    assert before == after
