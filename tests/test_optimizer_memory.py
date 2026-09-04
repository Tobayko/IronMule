"""Focused stdlib-only tests for the offline Optimization Memory v2 core."""

from __future__ import annotations

import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from friday_optimizer import (
    DataPhase,
    MemoryConflictError,
    MemoryLimitError,
    OptimizationMemoryV2,
    OptimizationRecord,
    QualityClass,
    RecordKind,
    UnsafeDatabaseError,
    canonical_bytes,
    loads_strict,
)
from friday_optimizer.memory import APPLICATION_ID, USER_VERSION


def record(record_id: str, payload: dict | None = None) -> OptimizationRecord:
    return OptimizationRecord(
        record_id=record_id,
        kind=RecordKind.BENCHMARK,
        quality=QualityClass.FORMAL,
        phase=DataPhase.LABEL,
        payload=payload or {"tokens": 4, "latency_ms": 3.25},
        created_at="2026-08-30T00:00:00+00:00",
    )


def test_canonical_json_is_stable_and_strict() -> None:
    assert canonical_bytes({"z": 1, "a": [True, None]}) == b'{"a":[true,null],"z":1}'
    with pytest.raises(ValueError):
        loads_strict('{"a": 1, "a": 2}')
    with pytest.raises(ValueError):
        loads_strict("[NaN]")
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("inf")})
    with pytest.raises(ValueError):
        canonical_bytes([[[1]]], max_depth=1)
    with pytest.raises(ValueError):
        canonical_bytes({"x": "12345"}, max_bytes=4)
    with pytest.raises(ValueError):
        canonical_bytes({"x": "\ud800"})
    with pytest.raises(ValueError):
        loads_strict('{"x":"\ud800"}')
    with pytest.raises(ValueError):
        loads_strict("1" * 5_000)
    with pytest.raises(ValueError):
        loads_strict("[" * 100 + "0" + "]" * 100, max_depth=32)
    with pytest.raises((TypeError, ValueError)):
        OptimizationRecord("id", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.LABEL, {}, created_at=1)  # type: ignore[arg-type]


def test_migration_identity_and_secure_permissions(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with OptimizationMemoryV2(path) as memory:
        assert memory.application_id == APPLICATION_ID
        assert memory.user_version == USER_VERSION
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == USER_VERSION


def test_open_read_only_never_creates_or_mutates_database(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with OptimizationMemoryV2(path) as memory:
        memory.append(record("ro"))
    before = (path.stat().st_mode, path.stat().st_ino, path.stat().st_mtime_ns)
    view = OptimizationMemoryV2.open_read_only(path)
    assert view.schema_ok
    assert view.integrity().ok
    with view.read_connection() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("UPDATE optimization_records SET quality='formal'")
    with pytest.raises(UnsafeDatabaseError):
        view.append(record("blocked"))
    assert before == (path.stat().st_mode, path.stat().st_ino, path.stat().st_mtime_ns)


def test_append_idempotence_conflict_and_chain(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        first = memory.append(record("r1"))
        again = memory.append(record("r1"))
        assert first["record_hash"] == again["record_hash"]
        with pytest.raises(MemoryConflictError):
            memory.append(record("r1", {"different": True}))
        for changed in (
            OptimizationRecord("r1", RecordKind.CANDIDATE, QualityClass.FORMAL, DataPhase.LABEL, {"tokens": 4, "latency_ms": 3.25}, created_at="2026-08-30T00:00:00Z"),
            OptimizationRecord("r1", RecordKind.BENCHMARK, QualityClass.EXPLORATORY, DataPhase.LABEL, {"tokens": 4, "latency_ms": 3.25}, created_at="2026-08-30T00:00:00Z"),
            OptimizationRecord("r1", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.FEATURE, {"tokens": 4, "latency_ms": 3.25}, created_at="2026-08-30T00:00:00Z"),
            OptimizationRecord("r1", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.LABEL, {"tokens": 4, "latency_ms": 3.25}, source_hash="a" * 64, created_at="2026-08-30T00:00:00Z"),
            OptimizationRecord("r1", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.LABEL, {"tokens": 4, "latency_ms": 3.25}, created_at="2026-08-30T00:00:01Z"),
        ):
            with pytest.raises(MemoryConflictError):
                memory.append(changed)
        memory.append(record("r2", {"ok": True}))
        assert memory.verify_chain()
        assert memory.integrity().ok


def test_batch_rollback_leaves_no_partial_records(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        with pytest.raises(MemoryConflictError):
            memory.append_many([record("r1"), record("r1", {"bad": True})])
        assert memory.list() == []


def test_trigger_rejects_update_and_delete(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        memory.append(record("r1"))
        with sqlite3.connect(memory.path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("UPDATE optimization_records SET quality='invalid'")
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM optimization_records")
        assert memory.verify_chain()


def test_tamper_is_detected_when_trigger_is_removed(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        memory.append(record("r1"))
        with sqlite3.connect(memory.path) as connection:
            connection.execute("DROP TRIGGER optimization_records_no_update")
            connection.execute("UPDATE optimization_records SET payload=?", (b'{"x":9}',))
        assert not memory.verify_chain()


def test_symlink_and_non_regular_files_are_refused(tmp_path: Path) -> None:
    target = tmp_path / "real.sqlite3"
    target.touch()
    link = tmp_path / "link.sqlite3"
    link.symlink_to(target)
    with pytest.raises(UnsafeDatabaseError):
        OptimizationMemoryV2(link)
    directory = tmp_path / "directory.sqlite3"
    directory.mkdir()
    with pytest.raises(UnsafeDatabaseError):
        OptimizationMemoryV2(directory)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(UnsafeDatabaseError):
        OptimizationMemoryV2(parent_link / "nested.sqlite3")
    with OptimizationMemoryV2(tmp_path / "name with ? and &.sqlite3") as memory:
        assert memory.verify_chain()


def test_nested_payload_is_deep_frozen(tmp_path: Path) -> None:
    payload = {"nested": {"values": [1, {"ok": True}]}}
    item = OptimizationRecord("frozen", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.FEATURE, payload)
    payload["nested"]["values"][1]["ok"] = False
    assert item.payload["nested"]["values"][1]["ok"] is True
    with pytest.raises(TypeError):
        item.payload["nested"]["values"][1]["ok"] = False  # type: ignore[index]
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        memory.append(item)
        assert memory.verify_chain()


def test_existing_schema_must_be_exact(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    memory = OptimizationMemoryV2(path)
    memory.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER optimization_records_no_update")
    with pytest.raises(UnsafeDatabaseError):
        OptimizationMemoryV2(path)
    path2 = tmp_path / "memory2.sqlite3"
    memory = OptimizationMemoryV2(path2)
    memory.close()
    with sqlite3.connect(path2) as connection:
        connection.execute("DROP INDEX optimization_records_kind_seq")
    with pytest.raises(UnsafeDatabaseError):
        OptimizationMemoryV2(path2)
    path3 = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path3) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(UnsafeDatabaseError):
        OptimizationMemoryV2(path3)


def test_bounded_generator_and_incremental_snapshot(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3", max_records=2) as memory:
        consumed = 0

        def endless():
            nonlocal consumed
            while True:
                consumed += 1
                yield record(f"r{consumed}")

        with pytest.raises(MemoryLimitError):
            memory.append_many(endless())
        assert consumed == 4
        assert memory.list() == []
    with OptimizationMemoryV2(tmp_path / "small-snapshot.sqlite3", max_snapshot_bytes=200) as memory:
        memory.append(record("snapshot", {"value": "x" * 100}))
        with pytest.raises(MemoryLimitError):
            memory.snapshot()


def test_same_inode_tamper_is_rejected_before_append(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with OptimizationMemoryV2(path) as memory:
        memory.append(record("original"))
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TRIGGER optimization_records_no_update")
            connection.execute("UPDATE optimization_records SET payload=?", (b'{"tampered":true}',))
        with pytest.raises(UnsafeDatabaseError):
            memory.append(record("new"))
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0] == 1


def test_shared_store_is_serialized_across_threads(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        records = [record(f"thread-{index}") for index in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(memory.append, records))
        assert len(rows) == 4
        assert memory.verify_chain()
        assert len(memory.list()) == 4


def test_preappend_integrity_and_sidecar_size_gate(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    with OptimizationMemoryV2(path) as memory:
        memory.append(record("first"))
        sidecar = Path(f"{path}-shm")
        sidecar.write_bytes(b"x" * 128)
        memory.max_database_bytes = path.stat().st_size + 64
        with pytest.raises(MemoryLimitError):
            memory.append(record("blocked"))
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM optimization_records").fetchone()[0] == 1
        sidecar.unlink()


def test_limits_snapshot_and_deterministic_queries(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3", max_records=2, max_payload_bytes=40) as memory:
        memory.append(record("b", {"v": 1}))
        memory.append(record("a", {"v": 2}))
        assert [row["record_id"] for row in memory.list()] == ["b", "a"]
        assert memory.snapshot() == memory.snapshot()
        with pytest.raises(MemoryLimitError):
            memory.append(record("c", {"v": 3}))
    with OptimizationMemoryV2(tmp_path / "small.sqlite3", max_payload_bytes=8) as memory:
        with pytest.raises(MemoryLimitError):
            memory.append(record("large", {"too": "large"}))


def test_read_connection_is_query_only_and_foreign_keys_enabled(tmp_path: Path) -> None:
    with OptimizationMemoryV2(tmp_path / "memory.sqlite3") as memory:
        with memory.read_connection() as connection:
            assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE forbidden(x INTEGER)")
