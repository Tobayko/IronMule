from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from friday_optimizer.corpus import CorpusAuditor, Q2_PROFILES_CONTRACT, normalize_record
from friday_optimizer.memory import OptimizationMemoryV2
from friday_optimizer.record_bridge import MemoryRecordSink, normalized_to_memory_records


def _eligible():
    return normalize_record(
        {
            "run": {
                "conditions": {
                    "chip": "Apple M1 Max", "fingerprint": "a" * 16, "model_id": "gemma-1b",
                    "model_revision": "rev", "model_manifest_sha256": "2" * 64,
                    "model_identity_sha256": "0" * 64, "quantisation_sha256": "1" * 64,
                    "tokenizer_sha256": "3" * 64,
                    "execution_plan": "single_shot", "max_tokens": 1, "prompt_tokens": 1,
                },
                "confirmation": {"ratio": {"decode_ns": {"median_ratio": 1.25}, "prefill_ns": {"median_ratio": 1.0}, "total_ns": {"median_ratio": 1.1}}},
                "tuned_at": 1.0,
            },
        },
        source_path="Q2_profiles.json",
        source_sha256="0" * 64,
        source_verified=True,
        manifest_verified=True,
        contract=Q2_PROFILES_CONTRACT,
    )


def test_bridge_converts_compact_feature_and_label_records() -> None:
    records = normalized_to_memory_records(_eligible())
    assert [record.kind.value for record in records] == ["import", "environment", "workload", "candidate", "benchmark"]
    assert records[-1].phase.value == "label"
    assert all(len(record.payload_bytes) < 100_000 for record in records)


def test_sink_is_deterministic_idempotent_and_transactional(tmp_path: Path) -> None:
    memory = OptimizationMemoryV2(tmp_path / "memory.sqlite3")
    sink = MemoryRecordSink(memory)
    record = _eligible()
    sink.accept_many((record,))
    first = sink.last_snapshot_hash
    count = len(memory.list())
    sink.accept_many((record,))
    assert sink.last_snapshot_hash == first
    assert len(memory.list()) == count
    assert memory.verify_chain()
    memory.close()


def test_wrong_contract_hash_rejects_before_memory_write(tmp_path: Path) -> None:
    memory = OptimizationMemoryV2(tmp_path / "memory.sqlite3")
    sink = MemoryRecordSink(memory)
    forged = replace(_eligible(), contract_hash="0" * 64)
    with pytest.raises(ValueError, match="version/hash"):
        sink.accept_many((forged,))
    assert memory.list() == []
    memory.close()


def test_bridge_rejects_foreign_source_and_label_field_atomically(tmp_path: Path) -> None:
    memory = OptimizationMemoryV2(tmp_path / "memory.sqlite3")
    sink = MemoryRecordSink(memory)
    forged = replace(_eligible(), source_path="evil-Q2_profiles.json")
    with pytest.raises(ValueError, match="source path"):
        sink.accept_many((forged,))
    assert memory.list() == []
    forged = replace(_eligible(), label_fields=_eligible().label_fields + ("stale_gain",))
    with pytest.raises(ValueError, match="feature/label"):
        sink.accept_many((forged,))
    assert memory.list() == []
    memory.close()


def test_bridge_rejects_missing_or_uppercase_source_hash(tmp_path: Path) -> None:
    memory = OptimizationMemoryV2(tmp_path / "memory.sqlite3")
    sink = MemoryRecordSink(memory)
    for value in ("", "A" * 64):
        forged = replace(_eligible(), source_sha256=value)
        with pytest.raises(ValueError, match="source_sha256"):
            sink.accept_many((forged,))
        assert memory.list() == []
    memory.close()


def test_bridge_imports_real_archive_without_raw_huge_payload(tmp_path: Path) -> None:
    inventory = CorpusAuditor(Path(__file__).resolve().parents[1], roots=(".friday-data/ironmule-evidence-archive",)).audit()
    memory = OptimizationMemoryV2(tmp_path / "archive-memory.sqlite3")
    sink = MemoryRecordSink(memory)
    sink.accept_many(inventory.records)
    assert sink.last_records
    assert all(len(record.payload_bytes) <= 1_048_576 for record in sink.last_records)
    assert memory.verify_chain()
    memory.close()
