from __future__ import annotations

import hashlib
import json
import os
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
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
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

    assert totals["sources"] == 3  # two files and the committed source occurrence
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
    assert (default.agent, default.source) == ("unattributed", "no_evidence")


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
    from friday_evidence.canonical import canonical_sha256

    previous = "0" * 64
    for event in export["events"]:
        event["prev_sha256"] = previous
        event["sha256"] = canonical_sha256({key: value for key, value in event.items()
                                             if key != "sha256"})
        previous = event["sha256"]
    (repository / "research").mkdir(parents=True, exist_ok=True)
    (repository / "research" / "export.json").write_text(json.dumps(export), encoding="utf-8")

    index = repository / ".friday-data" / "ssot.sqlite3"
    ssot.build(index, repository)

    rows = ssot.query(
        "SELECT native_id, kind, status FROM runs WHERE study = 'research' ORDER BY native_id",
        index,
    )
    assert [row["native_id"] for row in rows] == ["abc:1", "abc:2"]
    assert [row["kind"] for row in rows] == [
        "run_started",
        "run_finished",
    ]
    assert [row["status"] for row in rows] == [None, "failed"]

    seconds = ssot.query(
        "SELECT value FROM measurement WHERE metric = 'wall_seconds' AND native_id = 'abc:2'",
        index,
    )
    assert [row["value"] for row in seconds] == [1.5]


def test_source_objects_are_lossless_and_survive_original_removal(repository: Path) -> None:
    artifact = repository / '.friday-data' / 'trace-bytes'
    content = bytes(range(256)) * (ssot.OBJECT_CHUNK_BYTES // 128 + 1)
    artifact.write_bytes(content)
    duplicate = artifact.with_name('trace-copy')
    duplicate.write_bytes(content)
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['sources'] == 5
    assert totals['objects'] == 3
    digest = hashlib.sha256(content).hexdigest()
    artifact.unlink()
    duplicate.unlink()
    assert b''.join(ssot.source_content(digest, index)) == content
    assert ssot.verify(index)['state'] == 'verified'
    with pytest.raises(OSError):
        ssot.verify(index, check_sources=True)


def test_same_identified_record_has_one_metric_and_all_origins(repository: Path) -> None:
    path = repository / 'experiments' / 'demo' / 'identified.json'
    document = {'run_id': 'experiment-1', 'median_ms': 4.5, 'created_at': '2026-09-12T10:00:00Z'}
    path.write_text(json.dumps(document))
    path.with_name('copy.json').write_text(json.dumps(document, indent=2))
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['occurrences'] == 5
    assert totals['runs'] == 3
    assert totals['duplicate_occurrences'] == 2  # one copy and one Git alias
    rows = ssot.query("SELECT run_id FROM runs WHERE canonical_key LIKE 'record:%' AND payload_json LIKE '%experiment-1%'", index)
    assert len(rows) == 1
    identifier = rows[0]['run_id']
    assert len(ssot.query(f'SELECT * FROM origins WHERE run_id={identifier}', index)) == 2
    assert len(ssot.query(f'SELECT * FROM metrics WHERE run_id={identifier} AND path=\'median_ms\'', index)) == 1


def test_equal_values_without_shared_identity_are_not_collapsed(repository: Path) -> None:
    path = repository / 'experiments' / 'demo' / 'run.json'
    path.with_name('independent-repeat.json').write_bytes(path.read_bytes())
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['objects'] == 2
    assert totals['runs'] == 3
    assert len(ssot.query("SELECT * FROM measurement WHERE metric='tokens_per_second'", index)) == 2
    assert len(ssot.query('SELECT * FROM payloads', index)) == 2


def test_native_journal_and_exports_share_verified_event_identity(repository: Path) -> None:
    from friday_evidence.events import EventJournal

    directory = repository / '.friday-data' / 'events'
    directory.mkdir(mode=0o700)
    path = directory / 'events.sqlite3'
    with EventJournal(path) as journal:
        journal.append(run_id='a' * 32, kind='run_started', payload={'number': 1})
        journal.append(run_id='a' * 32, kind='run_finished', payload={'wall_seconds': 1.5})
        events = journal.events()
    export = {'schema': 'ironmule.optimization_export.v2', 'events': events}
    (directory / 'export.json').write_text(json.dumps(export))
    (directory / 'second-export.json').write_text(json.dumps(export))
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['occurrences'] == 9
    assert totals['runs'] == 4
    rows = ssot.query("SELECT * FROM measurement WHERE metric='wall_seconds'", index)
    assert len(rows) == 1
    assert len(ssot.query(f"SELECT * FROM origins WHERE run_id={rows[0]['run_id']}", index)) == 3
    assert ssot.verify(index, check_sources=True)['state'] == 'verified'


def test_unknown_schema_preserves_previous_index(repository: Path) -> None:
    index = repository / '.friday-data' / 'ssot.sqlite3'
    ssot.build(index, repository)
    before = index.read_bytes()
    with sqlite3.connect(repository / '.friday-data' / 'unknown.db') as connection:
        connection.execute('CREATE TABLE unfamiliar (result TEXT)')
        connection.execute("INSERT INTO unfamiliar VALUES ('historical data')")
    with pytest.raises(ssot.SsotError, match='unsupported'):
        ssot.build(index, repository)
    assert index.read_bytes() == before
    assert not list(index.parent.glob('*.building'))


def test_corrupt_json_is_not_silently_converted_to_a_successful_record(repository: Path) -> None:
    (repository / 'experiments' / 'broken.json').write_text('{broken')
    with pytest.raises(ValueError):
        ssot.build(repository / '.friday-data' / 'ssot.sqlite3', repository)


def test_discovery_excludes_tooling_and_includes_profiles_and_archive_inputs(repository: Path) -> None:
    tool = repository / '.friday-data' / 'tooling' / 'package.json'
    tool.parent.mkdir()
    tool.write_text('{}')
    profile = repository / 'profiles' / 'machine.json'
    profile.parent.mkdir()
    profile.write_text('{}')
    archive = repository / ssot.ARCHIVE_MANIFEST.parent
    archive.mkdir()
    specification = archive / 'preregistration.md'
    specification.write_text('Frozen original specification.\n')
    digest = hashlib.sha256(specification.read_bytes()).hexdigest()
    manifest = {'total_entries': 2, 'unique_files': 1, 'entries': [
        {'stored_as': specification.name, 'sha256': digest, 'bytes': specification.stat().st_size},
        {'duplicate_of': specification.name, 'sha256': digest, 'bytes': specification.stat().st_size},
    ]}
    (repository / ssot.ARCHIVE_MANIFEST).write_text(json.dumps(manifest))
    discovered = ssot.discover(repository)
    assert specification in discovered and profile in discovered and tool not in discovered
    index = repository / '.friday-data' / 'ssot.sqlite3'
    assert ssot.build(index, repository)['archive_entries'] == 2
    assert b''.join(ssot.source_content(digest, index)) == specification.read_bytes()
    specification.write_text('Changed source.\n')
    with pytest.raises(ssot.SsotError, match='missing or changed'):
        ssot.build(index, repository)


def test_source_change_during_build_is_detected_and_previous_index_survives(repository: Path, monkeypatch) -> None:
    index = repository / '.friday-data' / 'ssot.sqlite3'
    ssot.build(index, repository)
    before = index.read_bytes()
    original = ssot._insert_runs

    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        (repository / 'experiments' / 'demo' / 'run.json').write_text('{"changed": true}')
        return result

    monkeypatch.setattr(ssot, '_insert_runs', change_after_read)
    with pytest.raises(ssot.SsotError, match='source changed'):
        ssot.build(index, repository)
    assert index.read_bytes() == before


def test_output_cannot_replace_a_source_database(repository: Path) -> None:
    source = repository / '.friday-data' / 'study.sqlite3'
    before = source.read_bytes()
    with pytest.raises(ssot.SsotError, match='output must be absent'):
        ssot.build(source, repository)
    assert source.read_bytes() == before


def test_corrupt_stored_object_fails_verification_and_export_is_not_left_partial(repository: Path) -> None:
    index = repository / '.friday-data' / 'ssot.sqlite3'
    ssot.build(index, repository)
    digest = ssot.query('SELECT sha256 FROM source_objects ORDER BY sha256 LIMIT 1', index)[0]['sha256']
    with sqlite3.connect(index) as connection:
        connection.execute('UPDATE source_chunks SET compressed=? WHERE sha256=?', (b'corrupt', digest))
    destination = repository / 'restored.bin'
    assert ssot.main(['--database', str(index), 'export-source', digest, str(destination)]) == 65
    assert not destination.exists()
    destination.write_bytes(b'preserve existing file')
    assert ssot.main(['--database', str(index), 'export-source', digest, str(destination)]) == 66
    assert destination.read_bytes() == b'preserve existing file'


def test_trace_buffer_aliases_share_content_and_keep_link_metadata(repository: Path) -> None:
    target = repository / '.friday-data' / 'buffer-0'
    target.write_bytes(b'original trace buffer')
    alias = target.with_name('buffer-1')
    alias.symlink_to(target.name)
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['sources'] == 5 and totals['objects'] == 3
    assert ssot.query("SELECT link_target FROM sources WHERE path='.friday-data/buffer-1'", index) == [
        {'link_target': 'buffer-0'}
    ]
    assert ssot.verify(index, check_sources=True)['state'] == 'verified'


def test_evidence_alias_cannot_capture_files_outside_repository(repository: Path, tmp_path: Path) -> None:
    external = tmp_path / 'private.bin'
    external.write_bytes(b'not an evidence source')
    (repository / '.friday-data' / 'escape').symlink_to(external)
    with pytest.raises(ssot.SsotError, match='alias leaves'):
        ssot.discover(repository)


def test_active_sqlite_wal_is_not_misrepresented_as_stored_source_bytes(repository: Path) -> None:
    database = repository / '.friday-data' / 'study.sqlite3'
    connection = sqlite3.connect(database)
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute("UPDATE records SET status='changed'")
        connection.commit()
        with pytest.raises(ssot.SsotError, match='active SQLite journal'):
            ssot.build(repository / '.friday-data' / 'ssot.sqlite3', repository)
    finally:
        connection.close()


def test_identical_values_with_different_run_ids_remain_distinct(repository: Path) -> None:
    directory = repository / 'experiments' / 'demo'
    for identifier in ('independent-1', 'independent-2'):
        (directory / f'{identifier}.json').write_text(json.dumps({'run_id': identifier, 'elapsed_ms': 4.5}))
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['runs'] == 4
    assert totals['occurrences'] == 5  # committed source remains one canonical record
    assert len(ssot.query("SELECT * FROM measurement WHERE metric='elapsed_ms'", index)) == 2


def test_tampered_journal_digest_cannot_deduplicate_unrelated_content() -> None:
    from friday_evidence.canonical import canonical_sha256

    event = {'seq': 1, 'recorded_unix_ns': 1, 'run_id': 'a' * 32,
             'kind': 'sample', 'payload': {'elapsed_ms': 1.0}, 'prev_sha256': '0' * 64}
    event['sha256'] = canonical_sha256(event)
    assert ssot._journal_identity(event) == 'journal:' + event['sha256']
    event['payload']['elapsed_ms'] = 2.0
    with pytest.raises(ssot.SsotError, match='digest'):
        ssot._journal_identity(event)


def test_deleted_historical_data_versions_are_retained_and_readable(repository: Path) -> None:
    path = repository / 'experiments' / 'demo' / 'run.json'
    first = path.read_bytes()
    second = json.dumps({'tokens_per_second': 99.0, 'run_id': 'second-version'}).encode()
    path.write_bytes(second)
    _git(repository, 'add', str(path.relative_to(repository)))
    _git(repository, 'commit', '-qm', 'test: record another version')
    path.unlink()
    _git(repository, 'add', '-u')
    _git(repository, 'commit', '-qm', 'test: retire the source file')
    index = repository / '.friday-data' / 'ssot.sqlite3'
    totals = ssot.build(index, repository)
    assert totals['git_history_objects'] == 2
    assert totals['runs'] == 3
    for data in (first, second):
        assert b''.join(ssot.source_content(hashlib.sha256(data).hexdigest(), index)) == data
    assert ssot.verify(index, check_sources=True)['state'] == 'verified'


def test_verified_object_reuse_matches_fresh_build_and_rejects_corruption(repository: Path) -> None:
    first = repository / '.friday-data' / 'ssot.sqlite3'
    ssot.build(first, repository)
    second = repository / '.friday-data' / 'another-corpus.sqlite3'
    reused = ssot.build(second, repository, reuse_from=first)
    assert reused['runs'] == 2
    assert ssot.verify(second, check_sources=True)['state'] == 'verified'
    assert ssot.query('SELECT sha256,bytes FROM source_objects ORDER BY sha256', first) == ssot.query(
        'SELECT sha256,bytes FROM source_objects ORDER BY sha256', second)
    original_second = second.read_bytes()
    with sqlite3.connect(first) as connection:
        connection.execute('UPDATE source_chunks SET compressed=?', (b'corrupted prior object',))
    with pytest.raises(Exception, match='decompress|header|incorrect'):
        ssot.build(second, repository, reuse_from=first)
    assert second.read_bytes() == original_second


def test_archive_manifest_establishes_a_copy_without_merging_unrelated_repetitions(repository: Path) -> None:
    source = repository / 'experiments' / 'demo' / 'run.json'
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    archive = repository / ssot.ARCHIVE_MANIFEST.parent
    archive.mkdir()
    stored = archive / f'{digest}-run.json'
    stored.write_bytes(content)
    (repository / ssot.ARCHIVE_MANIFEST).write_text(json.dumps({
        'total_entries': 1, 'unique_files': 1, 'entries': [
            {'file': 'run.json', 'stored_as': stored.name, 'sha256': digest, 'bytes': len(content)}
        ]
    }))
    independent = source.with_name('independent-trial.json')
    independent.write_bytes(content)
    index = repository / '.friday-data' / 'ssot.sqlite3'
    ssot.build(index, repository)
    records = ssot.query(f"SELECT run_id,canonical_key FROM runs WHERE payload_sha256='{hashlib.sha256(ssot._canonical(json.loads(content)).encode()).hexdigest()}'", index)
    assert len(records) == 2  # archived/current copy plus the unrelated repetition
    archived = next(row for row in records if row['canonical_key'].startswith('archive:'))
    assert len(ssot.query(f"SELECT * FROM origins WHERE run_id={archived['run_id']}", index)) == 3


def test_legacy_nonfinite_values_are_preserved_and_explicitly_labeled(repository: Path) -> None:
    import math

    document = {'run_id': 'legacy-vector', 'headroom': float('inf'), 'elapsed_ms': 2.0}
    raw = json.dumps(document).encode()
    path = repository / 'experiments' / 'legacy.json'
    path.write_bytes(raw)
    index = repository / '.friday-data' / 'ssot.sqlite3'
    ssot.build(index, repository)
    record = ssot.query("SELECT payload_json,payload_format FROM runs WHERE payload_json LIKE '%legacy-vector%'", index)[0]
    assert record['payload_format'] == 'legacy_nonfinite_json'
    assert math.isinf(json.loads(record['payload_json'])['headroom'])
    assert b''.join(ssot.source_content(hashlib.sha256(raw).hexdigest(), index)) == raw
    assert ssot.status(index)['nonstandard_payloads'] == 1
    assert ssot.verify(index)['state'] == 'verified'
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE payloads SET format='json'")
    with pytest.raises(ssot.SsotError, match='encoding label'):
        ssot.verify(index)


def test_git_history_failure_is_not_misreported_as_no_history(repository: Path, monkeypatch) -> None:
    original = subprocess.check_output

    def fail_history(arguments, *args, **kwargs):
        if 'rev-list' in arguments:
            raise subprocess.CalledProcessError(128, arguments)
        return original(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'check_output', fail_history)
    with pytest.raises(ssot.SsotError, match='historical Git data'):
        ssot.build(repository / '.friday-data' / 'ssot.sqlite3', repository)
    assert not list((repository / '.friday-data').glob('*.building'))
