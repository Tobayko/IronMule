"""Filesystem-backed tests for the metadata-only calibration event journal."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import uuid

import pytest

from friday_evidence.events import EventJournal, EventJournalError
import friday_evidence.events as events_module


KINDS = ("run_started", "sample", "validation", "run_finished")


def _journal_path(tmp_path: Path) -> Path:
    parent = tmp_path / "product-state"
    parent.mkdir(mode=0o700)
    os.chmod(parent, 0o700)
    return parent / "events.sqlite3"


def _run_id(index: int = 0) -> str:
    return f"{index:032x}"


def test_append_history_filter_latest_and_chain(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path) as journal:
        first = journal.append(_run_id(1), "run_started", {"model": "gemma", "prompt_tokens": 12})
        second = journal.append(_run_id(1), "sample", {"step": 1, "elapsed_ms": 2.5})
        journal.append(_run_id(2), "pause", {"reason": "thermal_guard"})
        assert first["seq"] == 1
        assert second["prev_sha256"] == first["sha256"]
        assert [event["seq"] for event in journal.events()] == [1, 2, 3]
        assert [event["seq"] for event in journal.events(_run_id(1))] == [1, 2]
        assert [event["seq"] for event in journal.events(after_seq=1, limit=1)] == [2]
        assert journal.latest()["seq"] == 3
        assert journal.latest(_run_id(1))["seq"] == 2
        assert journal.verify() == 3
    assert not path.with_name(path.name + "-wal").exists()


def test_read_only_missing_does_not_create_database_or_parent(tmp_path: Path):
    parent = tmp_path / "missing-parent"
    path = parent / "events.sqlite3"
    with pytest.raises(EventJournalError):
        EventJournal(path, read_only=True)
    assert not parent.exists()


def test_read_only_queries_and_mutation_are_bounded(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path) as journal:
        journal.append(_run_id(), "sample", {"step": 1})
    with EventJournal(path, read_only=True) as journal:
        assert journal.events(limit=1)[0]["kind"] == "sample"
        with pytest.raises(EventJournalError):
            journal.append(_run_id(), "sample", {"step": 2})
        for limit in (0, 1001, True):
            with pytest.raises(EventJournalError):
                journal.events(limit=limit)
        for after_seq in (-1, True):
            with pytest.raises(EventJournalError):
                journal.events(after_seq=after_seq)


def test_sqlite_extension_loading_is_denied_on_real_connection(tmp_path: Path):
    with EventJournal(_journal_path(tmp_path)) as journal:
        with pytest.raises(sqlite3.Error):
            journal._connection.execute("SELECT load_extension('missing')")  # noqa: SLF001


def test_extension_guard_fallback_handles_connection_without_python_api():
    real = sqlite3.connect(":memory:")

    class WithoutLoadExtension:
        def __getattr__(self, name):
            if name == "enable_load_extension":
                raise AttributeError(name)
            return getattr(real, name)

    try:
        events_module._secure_extension_controls(WithoutLoadExtension())
        with pytest.raises(sqlite3.Error):
            real.execute("SELECT load_extension('missing')")
    finally:
        real.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "secret"},
        {"nested": {"messages": []}},
        {"text": "secret"},
        {"token_ids": [1, 2]},
        {"prompt_tokens": 12, "token_sha256": "a" * 64},
        float("nan"),
        {"unsupported": {1, 2}},
    ],
)
def test_payload_contract_rejects_user_content_and_unsupported_values(tmp_path: Path, payload):
    with EventJournal(_journal_path(tmp_path)) as journal:
        if payload == {"prompt_tokens": 12, "token_sha256": "a" * 64}:
            assert journal.append(_run_id(), "sample", payload)["payload"] == payload
        else:
            with pytest.raises(EventJournalError):
                journal.append(_run_id(), "sample", payload)


def test_payload_depth_and_size_are_bounded(tmp_path: Path):
    with EventJournal(_journal_path(tmp_path)) as journal:
        deep: object = "leaf"
        for _ in range(33):
            deep = [deep]
        with pytest.raises(EventJournalError):
            journal.append(_run_id(), "sample", deep)
        with pytest.raises(EventJournalError):
            journal.append(_run_id(), "sample", {"metadata": "x" * (256 * 1024)})


def test_registry_and_run_id_are_closed(tmp_path: Path):
    with EventJournal(_journal_path(tmp_path)) as journal:
        with pytest.raises(EventJournalError):
            journal.append("not-a-uuid", "sample", {})
        with pytest.raises(EventJournalError):
            journal.append(_run_id(), "unknown", {})
        with pytest.raises(EventJournalError):
            journal.events("not-a-uuid")


def test_schema_corruption_and_append_only_triggers_fail_closed(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path) as journal:
        journal.append(_run_id(), "sample", {"step": 1})
        with pytest.raises(sqlite3.DatabaseError):
            journal._connection.execute("UPDATE event_journal SET kind='pause' WHERE seq=1")  # noqa: SLF001
        journal._connection.rollback()  # noqa: SLF001
        with pytest.raises(sqlite3.DatabaseError):
            journal._connection.execute("DELETE FROM event_journal WHERE seq=1")  # noqa: SLF001
        journal._connection.rollback()  # noqa: SLF001
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA application_id=123")
    connection.commit()
    connection.close()
    with pytest.raises(EventJournalError):
        EventJournal(path, read_only=True)


def test_reopened_writable_journal_reapplies_page_limit(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path):
        pass
    with EventJournal(path) as journal:
        page_size = journal._connection.execute("PRAGMA page_size").fetchone()[0]  # noqa: SLF001
        page_limit = journal._connection.execute("PRAGMA max_page_count").fetchone()[0]  # noqa: SLF001
    assert page_limit <= (64 * 1024 * 1024) // page_size


def test_trigger_body_tampering_is_rejected_even_when_names_remain(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path):
        pass
    connection = sqlite3.connect(path)
    connection.executescript(
        "DROP TRIGGER event_journal_no_update;"
        "CREATE TRIGGER event_journal_no_update BEFORE UPDATE ON event_journal "
        "BEGIN SELECT RAISE(ABORT, 'changed trigger'); END;"
    )
    connection.commit()
    connection.close()
    with pytest.raises(EventJournalError):
        EventJournal(path, read_only=True)


def test_history_and_latest_recheck_chain_after_external_corruption(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path) as journal:
        journal.append(_run_id(), "sample", {"step": 1})
        connection = sqlite3.connect(path)
        connection.executescript(
            "DROP TRIGGER event_journal_no_update; DROP TRIGGER event_journal_no_delete;"
        )
        connection.execute("UPDATE event_journal SET payload_json=? WHERE seq=?", (json.dumps({"step": 2}, separators=(",", ":")), 1))
        connection.executescript(
            "CREATE TRIGGER event_journal_no_update BEFORE UPDATE ON event_journal "
            "BEGIN SELECT RAISE(ABORT, 'event journal is append-only'); END;"
            "CREATE TRIGGER event_journal_no_delete BEFORE DELETE ON event_journal "
            "BEGIN SELECT RAISE(ABORT, 'event journal is append-only'); END;"
        )
        connection.commit()
        connection.close()
        with pytest.raises(EventJournalError):
            journal.events()
        with pytest.raises(EventJournalError):
            journal.latest()


def test_symlink_hardlink_and_public_permissions_are_rejected(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path) as journal:
        journal.append(_run_id(), "sample", {})
    link = path.with_name("link.sqlite3")
    link.symlink_to(path)
    with pytest.raises(EventJournalError):
        EventJournal(link, read_only=True)
    hardlink = path.with_name("hardlink.sqlite3")
    os.link(path, hardlink)
    with pytest.raises(EventJournalError):
        EventJournal(hardlink, read_only=True)
    os.chmod(path.parent, 0o755)
    with pytest.raises(EventJournalError):
        EventJournal(path, read_only=True)


def test_concurrent_connections_append_without_lost_chain_records(tmp_path: Path):
    path = _journal_path(tmp_path)
    with EventJournal(path):
        pass

    def append(index: int):
        with EventJournal(path) as journal:
            return journal.append(_run_id(index), "sample", {"step": index})["seq"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        sequences = list(pool.map(append, range(20)))
    with EventJournal(path, read_only=True) as journal:
        assert len(set(sequences)) == 20
        assert journal.verify() == 20
