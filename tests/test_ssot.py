from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from friday_evidence import ssot


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "Owner",
            "GIT_AUTHOR_EMAIL": "owner@example.invalid",
            "GIT_COMMITTER_NAME": "Owner",
            "GIT_COMMITTER_EMAIL": "owner@example.invalid",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(root),
        },
    )


def _study_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE records (record_id TEXT PRIMARY KEY, entity_key TEXT, record_kind TEXT,"
        " status TEXT, created_at_unix_ns INTEGER, report_json TEXT, provenance_json TEXT)"
    )
    connection.execute(
        "INSERT INTO records VALUES(?,?,?,?,?,?,?)",
        (
            "a" * 64,
            "pair-1",
            "benchmark",
            "passed",
            1_700_000_000_000_000_000,
            json.dumps({"median_ms": 4.5, "samples": [1.0, 2.0, 3.0]}),
            json.dumps({"git_revision": "REVISION"}),
        ),
    )
    connection.commit()
    connection.close()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "experiments" / "demo").mkdir(parents=True)
    (root / ".friday-data").mkdir(parents=True)
    (root / "experiments" / "demo" / "run.json").write_text(
        json.dumps({"status": "ok", "tokens_per_second": 12.5, "nested": {"peak_gb": 3.25}}),
        encoding="utf-8",
    )
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(
        root,
        "commit",
        "-q",
        "-m",
        "feat: add demo run\n\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>",
    )
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    database = root / ".friday-data" / "study.sqlite3"
    _study_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE records SET provenance_json = ?",
            (json.dumps({"git_revision": revision}),),
        )
    return root


def test_build_indexes_every_source_and_never_writes_to_one(repository: Path) -> None:
    sources = sorted(ssot.discover(repository))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    assert len(sources) == 2

    index = repository / ".friday-data" / "ssot.sqlite3"
    totals = ssot.build(index, repository)

    assert totals["sources"] == 2
    assert totals["runs"] == 2
    assert totals["failed_sources"] == 0
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources} == before

    rows = ssot.query("SELECT study, native_id, status FROM runs ORDER BY study", index)
    assert [row["study"] for row in rows] == ["experiments/demo", "study"]


def test_metrics_are_flattened_and_long_arrays_summarised(repository: Path) -> None:
    index = repository / ".friday-data" / "ssot.sqlite3"
    ssot.build(index, repository)

    values = {
        row["metric"]: row["value"]
        for row in ssot.query("SELECT metric, value FROM measurement", index)
    }
    assert values["tokens_per_second"] == pytest.approx(12.5)
    assert values["nested.peak_gb"] == pytest.approx(3.25)
    assert values["median_ms"] == pytest.approx(4.5)
    assert values["samples[1]"] == pytest.approx(2.0)

    summary = ssot.flatten_metrics({"raw": list(range(ssot.ARRAY_SUMMARY_THRESHOLD + 1))})
    assert summary["raw.count"] == pytest.approx(ssot.ARRAY_SUMMARY_THRESHOLD + 1)
    assert summary["raw.max"] == pytest.approx(ssot.ARRAY_SUMMARY_THRESHOLD)
    assert "raw[0]" not in summary


def test_runs_carry_the_agent_that_produced_them(repository: Path) -> None:
    index = repository / ".friday-data" / "ssot.sqlite3"
    ssot.build(index, repository)

    rows = {
        row["study"]: (row["agent"], row["agent_source"])
        for row in ssot.query("SELECT study, agent, agent_source FROM runs", index)
    }
    assert rows["experiments/demo"] == ("claude", "git_history")
    assert rows["study"] == ("claude", "provenance_commit")


def test_commit_attribution_reads_trailer_scope_and_default() -> None:
    trailer = ssot._attribute_commit(
        "0" * 40, "fix: thing", "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
    )
    assert (trailer.agent, trailer.source) == ("claude", "commit_trailer")

    scope = ssot._attribute_commit("1" * 40, "feat(gemini): cockpit", "")
    assert (scope.agent, scope.source) == ("gemini", "commit_scope")

    default = ssot._attribute_commit("2" * 40, "feat(product): add worker", "")
    assert (default.agent, default.source) == ("codex", "repo_default_codex")


def test_journal_export_json_is_indexed_event_by_event(repository: Path) -> None:
    export = {
        "schema": "ironmule.optimization_export.v2",
        "events": [
            {
                "seq": 1,
                "recorded_unix_ns": 1_788_000_000_000_000_000,
                "run_id": "abc",
                "kind": "run_started",
                "payload": {"model_id": "gemma"},
                "prev_sha256": "0" * 64,
                "sha256": "a" * 64,
            },
            {
                "seq": 2,
                "recorded_unix_ns": 1_788_000_000_000_000_001,
                "run_id": "abc",
                "kind": "run_finished",
                "payload": {"status": "failed", "wall_seconds": 1.5},
                "prev_sha256": "a" * 64,
                "sha256": "b" * 64,
            },
        ],
    }
    (repository / "research").mkdir(parents=True, exist_ok=True)
    (repository / "research" / "export.json").write_text(json.dumps(export), encoding="utf-8")

    index = repository / ".friday-data" / "ssot.sqlite3"
    ssot.build(index, repository)

    rows = ssot.query(
        "SELECT native_id, kind, status FROM runs WHERE study = 'research' ORDER BY native_id",
        index,
    )
    assert [row["native_id"] for row in rows] == ["abc:1", "abc:2", "export"]
    assert [row["kind"] for row in rows] == [
        "run_started",
        "run_finished",
        "ironmule.optimization_export.v2",
    ]
    assert [row["status"] for row in rows] == [None, "failed", None]

    seconds = ssot.query(
        "SELECT value FROM measurement WHERE metric = 'wall_seconds' AND native_id = 'abc:2'",
        index,
    )
    assert [row["value"] for row in seconds] == [1.5]
