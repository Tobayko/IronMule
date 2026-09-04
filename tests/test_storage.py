import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from friday_h0.canonical import canonical_json, canonical_sha256
from friday_h0.protocol import ProtocolError, close_manifest
from friday_h0.storage import (
    PERSISTENCE_JSON_DEPTH,
    PERSISTENCE_MAX_EVENT_BYTES,
    PersistenceOutcome,
    Storage,
    StorageError,
)
from tests.test_manifest import valid_manifest


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "h0.sqlite3"
        self.storage = Storage.open(self.path)
        self.storage.create_run(valid_manifest(), created_at_unix_ns=1)

    def tearDown(self):
        self.storage.close()
        self.tempdir.cleanup()

    def _closed_result(self, run_id="persist-001"):
        manifest = valid_manifest("analysis_known_win")
        manifest["run_id"] = run_id
        closed = close_manifest(manifest)
        result = {
            "schema_version": 1,
            "run_id": closed.run_id,
            "mode": closed.mode,
            "manifest_sha256": closed.sha256,
            "status": "completed",
            "classification": "promoted",
            "action": "promoted",
            "error": None,
            "evidence": {"fixture": "analysis_known_win"},
        }
        return closed, result

    @staticmethod
    def _children():
        return {
            "raw_samples": [
                {"session_id": "s0", "sample_kind": "timing", "sample_index": 0,
                 "block_index": 0, "arm": "baseline", "value": 1.5, "unit": "ns",
                 "observed_at_ns": 10}
            ],
            "scalar_metrics": [
                {"metric_name": "median", "value": 1.5, "unit": "ns", "scope": "run",
                 "recorded_at_ns": 11},
                {"metric_name": "rss", "value": None, "unit": "bytes", "scope": "run",
                 "missing_reason": "unavailable", "recorded_at_ns": 12},
            ],
            "correctness_metrics": [
                {"case_name": "oracle", "metric_name": "max_abs", "value": 0.0,
                 "unit": "abs", "passed": True, "detail": {"atol": 0.0}}
            ],
            "artifacts": [
                {"artifact_name": "manifest", "artifact_kind": "json", "sha256": "a" * 64,
                 "metadata": {"role": "manifest"}}
            ],
        }

    def _persist(self, run_id="persist-001", **overrides):
        closed, result = self._closed_result(run_id)
        children = self._children()
        children.update(overrides)
        outcome = self.storage.persist_common_result(
            closed, result, created_at_unix_ns=100,
            raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
            correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"],
            recorded_at_ns=101,
        )
        return closed, result, children, outcome

    def _fresh_database(self, name):
        path = Path(self.tempdir.name) / f"{name}.sqlite3"
        with Storage.open(path) as storage:
            storage.create_run(valid_manifest(), created_at_unix_ns=1)
        return path

    @staticmethod
    def _count_run_rows(path, run_id):
        connection = sqlite3.connect(path)
        try:
            return {
                table: connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                for table in (
                    "runs", "status_events", "raw_samples", "scalar_metrics",
                    "correctness_metrics", "artifacts",
                )
            }
        finally:
            connection.close()

    def test_persist_common_result_stores_complete_bundle_atomically(self):
        closed, result, children, outcome = self._persist()
        self.assertIsInstance(outcome, PersistenceOutcome)
        self.assertEqual(outcome.state, "inserted")
        self.assertEqual(outcome.run_id, closed.run_id)
        for table, expected in (("raw_samples", 1), ("scalar_metrics", 2),
                                ("correctness_metrics", 1), ("artifacts", 1)):
            self.assertEqual(len(self.storage.rows(table, closed.run_id)), expected)
        event = self.storage.rows("status_events", closed.run_id)
        self.assertEqual(len(event), 1)
        self.assertEqual(event[0]["event_kind"], "common_result")
        self.assertEqual(event[0]["status"], result["status"])
        payload = json.loads(event[0]["payload_json"])
        self.assertEqual(payload["result"], result)
        self.assertEqual(payload["bundle"]["raw_samples"], children["raw_samples"])
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["result_sha256"], canonical_sha256(result))

    def test_invalid_last_child_leaves_zero_rows(self):
        closed, result = self._closed_result("rollback-last")
        children = self._children()
        children["artifacts"].append({
            "artifact_name": "bad", "artifact_kind": "json", "sha256": "not-a-sha",
            "metadata": {},
        })
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(
                closed, result, created_at_unix_ns=100,
                raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
                correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"],
            )
        for table in ("runs", "status_events", "raw_samples", "scalar_metrics", "correctness_metrics", "artifacts"):
            self.assertEqual(self.storage.connection.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (closed.run_id,)).fetchone()[0], 0)

    def test_identical_replay_is_idempotent_without_new_rows(self):
        closed, result, children, first = self._persist()
        counts_before = {
            table: self.storage.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("runs", "status_events", "raw_samples", "scalar_metrics", "correctness_metrics", "artifacts")
        }
        second = self.storage.persist_common_result(
            closed, result, created_at_unix_ns=999,
            raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
            correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"],
        )
        self.assertEqual(second.state, "idempotent")
        counts_after = {
            table: self.storage.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in counts_before
        }
        self.assertEqual(counts_before, counts_after)

    def test_read_only_bundle_verifier_is_hash_and_child_complete(self):
        closed, result, children, _ = self._persist("verify-clean")
        before = self.storage.connection.total_changes
        self.assertEqual(
            self.storage.verify_common_result_bundle(
                closed, result,
                raw_samples=children["raw_samples"],
                scalar_metrics=children["scalar_metrics"],
                correctness_metrics=children["correctness_metrics"],
                artifacts=children["artifacts"],
            ),
            "verified",
        )
        self.assertEqual(self.storage.connection.total_changes, before)
        self.storage.close()
        with Storage.open(self.path, read_only=True) as readonly:
            before = readonly.connection.total_changes
            self.assertEqual(
                readonly.verify_common_result_bundle(
                    closed, result,
                    raw_samples=children["raw_samples"],
                    scalar_metrics=children["scalar_metrics"],
                    correctness_metrics=children["correctness_metrics"],
                    artifacts=children["artifacts"],
                ),
                "verified",
            )
            self.assertEqual(readonly.connection.total_changes, before)

    def test_read_only_bundle_verifier_rejects_wrapper_and_each_child_tamper(self):
        closed, result, children, _ = self._persist("verify-tamper")
        args = {
            "raw_samples": children["raw_samples"],
            "scalar_metrics": children["scalar_metrics"],
            "correctness_metrics": children["correctness_metrics"],
            "artifacts": children["artifacts"],
        }
        trigger_sql = {}
        for table in ("status_events", "raw_samples", "scalar_metrics", "correctness_metrics", "artifacts"):
            for operation in ("update", "delete"):
                name = f"{table}_append_only_{operation}"
                trigger_sql[name] = self.storage.connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
                ).fetchone()[0]
                self.storage.connection.execute(f"DROP TRIGGER {name}")
        self.storage.connection.commit()
        try:
            event = self.storage.rows("status_events", closed.run_id)[0]
            self.storage.connection.execute(
                "UPDATE status_events SET status=? WHERE event_id=?", ("invalid", event["event_id"])
            )
            self.storage.connection.commit()
            with self.assertRaises(StorageError):
                self.storage.verify_common_result_bundle(closed, result, **args)
            self.storage.connection.execute(
                "UPDATE status_events SET status=? WHERE event_id=?", (result["status"], event["event_id"])
            )
            self.storage.connection.commit()
            self.storage.connection.execute(
                "UPDATE status_events SET payload_hash=? WHERE event_id=?", ("0" * 64, event["event_id"])
            )
            self.storage.connection.commit()
            with self.assertRaises(StorageError):
                self.storage.verify_common_result_bundle(closed, result, **args)
            self.storage.connection.execute(
                "UPDATE status_events SET payload_hash=? WHERE event_id=?", (event["payload_hash"], event["event_id"])
            )
            self.storage.connection.commit()
            for table, column, value in (
                ("raw_samples", "value", 2.5),
                ("scalar_metrics", "value", 2.5),
                ("correctness_metrics", "value", 2.5),
                ("artifacts", "sha256", "b" * 64),
            ):
                row = self.storage.connection.execute(
                    f"SELECT rowid FROM {table} WHERE run_id=? LIMIT 1", (closed.run_id,)
                ).fetchone()
                self.storage.connection.execute(
                    f"UPDATE {table} SET {column}=? WHERE rowid=?", (value, row[0])
                )
                self.storage.connection.commit()
                with self.assertRaises(StorageError):
                    self.storage.verify_common_result_bundle(closed, result, **args)
                original = {
                    "raw_samples": children["raw_samples"][0].get(column),
                    "scalar_metrics": children["scalar_metrics"][0].get(column),
                    "correctness_metrics": children["correctness_metrics"][0].get(column),
                    "artifacts": children["artifacts"][0].get(column),
                }[table]
                self.storage.connection.execute(
                    f"UPDATE {table} SET {column}=? WHERE rowid=?", (original, row[0])
                )
                self.storage.connection.commit()

            self.storage.connection.execute(
                "DELETE FROM raw_samples WHERE run_id=?", (closed.run_id,)
            )
            self.storage.connection.commit()
            with self.assertRaises(StorageError):
                self.storage.verify_common_result_bundle(closed, result, **args)
        finally:
            for sql in trigger_sql.values():
                self.storage.connection.execute(sql)
            self.storage.connection.commit()

    def test_different_result_manifest_or_child_collides(self):
        closed, result, children, _ = self._persist()
        changed_result = dict(result)
        changed_result["evidence"] = {"fixture": "different"}
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, changed_result, created_at_unix_ns=100,
                                               raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
                                               correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"])
        changed_children = {key: list(value) for key, value in children.items()}
        changed_children["raw_samples"] = [dict(children["raw_samples"][0], value=2.0)]
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=100,
                                               raw_samples=changed_children["raw_samples"], scalar_metrics=children["scalar_metrics"],
                                               correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"])
        changed_manifest = valid_manifest("analysis_known_win")
        changed_manifest["run_id"] = closed.run_id
        changed_manifest["provenance"]["spec_sha256"] = "d" * 64
        changed_closed = close_manifest(changed_manifest)
        changed_result["manifest_sha256"] = changed_closed.sha256
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(changed_closed, changed_result, created_at_unix_ns=100,
                                               raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
                                               correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"])

    def test_preexisting_partial_run_is_hard_conflict(self):
        manifest = valid_manifest("analysis_known_win")
        manifest["run_id"] = "partial-run"
        self.storage.create_run(manifest, created_at_unix_ns=1)
        closed, result = self._closed_result("partial-run")
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=2)
        self.assertEqual(self.storage.connection.execute("SELECT COUNT(*) FROM status_events WHERE run_id='partial-run'").fetchone()[0], 0)

    def test_duplicate_child_keys_result_binding_and_missing_xor_rejected(self):
        closed, result = self._closed_result("rejects")
        duplicate = self._children()
        duplicate["raw_samples"].append(dict(duplicate["raw_samples"][0]))
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=1, **duplicate)
        duplicate = self._children()
        duplicate["scalar_metrics"].append(dict(duplicate["scalar_metrics"][0]))
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=1, **duplicate)
        bad_result = dict(result, manifest_sha256="0" * 64)
        with self.assertRaises(ProtocolError):
            self.storage.persist_common_result(closed, bad_result, created_at_unix_ns=1)
        bad_missing = self._children()
        bad_missing["scalar_metrics"] = [{"metric_name": "x", "value": None, "unit": "ns"}]
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=1, **bad_missing)

    def test_bounded_nonfinite_and_read_only_rejection(self):
        closed, result = self._closed_result("bounded")
        bad = self._children()
        bad["raw_samples"] = [dict(bad["raw_samples"][0], value=float("nan"))]
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=1, **bad)
        bad = self._children()
        bad["raw_samples"] = [dict(bad["raw_samples"][0], value=10 ** 1000)]
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=1, **bad)
        bad = self._children()
        bad["artifacts"] = [{"artifact_name": "x", "artifact_kind": "json", "sha256": "a" * 64,
                              "metadata": {"x": "z" * (64 * 1024)}}]
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(closed, result, created_at_unix_ns=1, **bad)
        self.storage.close()
        with Storage.open(self.path, read_only=True) as readonly:
            with self.assertRaises(StorageError):
                readonly.persist_common_result(closed, result, created_at_unix_ns=1)

    def test_schema_integrity_rejects_missing_or_altered_required_objects(self):
        mutations = (
            ("missing-trigger", "DROP TRIGGER runs_append_only_update"),
            ("missing-table", "DROP TABLE artifacts"),
            (
                "missing-migration-row",
                "DROP TRIGGER schema_migrations_append_only_delete; "
                "DELETE FROM schema_migrations WHERE version=1",
            ),
            ("wrong-column", "ALTER TABLE db_identity RENAME COLUMN value TO wrong_value"),
        )
        for name, sql in mutations:
            path = self._fresh_database(name)
            connection = sqlite3.connect(path)
            try:
                connection.executescript(sql)
            finally:
                connection.close()
            with self.assertRaises(StorageError):
                Storage.open(path)

    def test_schema_integrity_is_exact_and_never_repairs_temp_databases(self):
        mutations = (
            ("extra-table", "CREATE TABLE extra_table (id INTEGER PRIMARY KEY)"),
            ("extra-view", "CREATE VIEW extra_view AS SELECT 1"),
            ("extra-index", "CREATE INDEX extra_index ON runs(mode)"),
            (
                "extra-trigger",
                "CREATE TRIGGER extra_trigger AFTER INSERT ON runs BEGIN SELECT 1; END",
            ),
            ("extra-column", "ALTER TABLE db_identity ADD COLUMN extra TEXT"),
            (
                "changed-trigger-body",
                "DROP TRIGGER runs_append_only_update; "
                "CREATE TRIGGER runs_append_only_update BEFORE UPDATE ON runs "
                "BEGIN SELECT RAISE(ABORT, 'different body'); END",
            ),
            (
                "wrong-index-columns",
                "CREATE INDEX wrong_index_columns ON raw_samples(sample_index)",
            ),
            (
                "wrong-index-uniqueness",
                "CREATE UNIQUE INDEX wrong_index_uniqueness ON raw_samples(run_id)",
            ),
        )
        for name, sql in mutations:
            path = self._fresh_database(f"exact-{name}")
            with sqlite3.connect(path) as connection:
                connection.executescript(sql)
            mutated_digest = hashlib.sha256(path.read_bytes()).digest()
            for read_only in (True, False):
                with self.assertRaises(StorageError, msg=f"{name} read_only={read_only}"):
                    Storage.open(path, read_only=read_only)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).digest(),
                    mutated_digest,
                    f"schema verification repaired {name}",
                )

    def test_current_v1_schema_passes_read_only_and_writable_exact_checks(self):
        path = self._fresh_database("exact-current")
        with Storage.open(path, read_only=True) as readonly:
            self.assertEqual(readonly.connection.execute("PRAGMA query_only").fetchone()[0], 1)
        with Storage.open(path) as writable:
            self.assertEqual(writable.connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_replay_checks_all_manifest_mirror_fields(self):
        closed, result, children, _ = self._persist("mirror-fields")
        self.storage.connection.execute("DROP TRIGGER runs_append_only_update")
        self.storage.connection.execute(
            "UPDATE runs SET environment_sha256=? WHERE run_id=?", ("e" * 64, closed.run_id)
        )
        self.storage.connection.commit()
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(
                closed, result, created_at_unix_ns=999,
                raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
                correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"],
            )

    def test_status_event_payload_is_bounded_and_depth_limited(self):
        with self.assertRaises(StorageError):
            self.storage.append_status_event(
                "run-001", "large", "open", {"payload": "x" * PERSISTENCE_MAX_EVENT_BYTES}
            )
        nested = {}
        root = nested
        for _ in range(PERSISTENCE_JSON_DEPTH + 1):
            child = {}
            nested["child"] = child
            nested = child
        with self.assertRaises(StorageError):
            self.storage.append_status_event("run-001", "deep", "open", root)
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(StorageError):
            self.storage.append_status_event("run-001", "cycle", "open", cyclic)
        with self.assertRaises(StorageError):
            self.storage.append_status_event("run-001", "timestamp", "open", {}, recorded_at_ns=1 << 63)
        self.assertEqual(
            self.storage.connection.execute(
                "SELECT COUNT(*) FROM status_events WHERE run_id='run-001'"
            ).fetchone()[0],
            0,
        )

    def test_all_sqlite_integer_inputs_are_bounded(self):
        with self.assertRaises(StorageError):
            self.storage.create_run(valid_manifest("eager_baseline"), created_at_unix_ns=1 << 63)
        with self.assertRaises(StorageError):
            self.storage.append_raw_sample(
                "run-001", "s0", 1 << 63, "baseline", 1.0, "ns"
            )
        with self.assertRaises(StorageError):
            self.storage.append_scalar_metric(
                "run-001", "bounded-time", 1.0, "ns", recorded_at_ns=1 << 63
            )

    def test_in_transaction_sql_failure_rolls_back_every_bundle_row(self):
        closed, result = self._closed_result("late-sql-failure")
        children = self._children()
        self.storage.connection.execute(
            """CREATE TRIGGER forced_storage_failure
               BEFORE INSERT ON artifacts
               BEGIN SELECT RAISE(ABORT, 'forced storage test failure'); END;"""
        )
        self.storage.connection.commit()
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(
                closed, result, created_at_unix_ns=100,
                raw_samples=children["raw_samples"], scalar_metrics=children["scalar_metrics"],
                correctness_metrics=children["correctness_metrics"], artifacts=children["artifacts"],
            )
        self.assertEqual(self._count_run_rows(self.path, closed.run_id), {
            "runs": 0, "status_events": 0, "raw_samples": 0,
            "scalar_metrics": 0, "correctness_metrics": 0, "artifacts": 0,
        })

    def test_duplicate_or_tampered_common_result_is_not_replayable(self):
        closed, result, children, _ = self._persist("duplicate-event")
        self.storage.connection.execute(
            """INSERT INTO status_events
               (run_id,event_kind,status,payload_json,payload_hash,recorded_at_ns)
               SELECT run_id,event_kind,status,payload_json,payload_hash,recorded_at_ns
               FROM status_events WHERE run_id=? AND event_kind='common_result'""",
            (closed.run_id,),
        )
        self.storage.connection.commit()
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(
                closed, result, created_at_unix_ns=100, **children
            )

        closed, result, children, _ = self._persist("tampered-wrapper")
        self.storage.connection.execute("DROP TRIGGER status_events_append_only_update")
        row = self.storage.connection.execute(
            "SELECT event_id FROM status_events WHERE run_id=? AND event_kind='common_result'",
            (closed.run_id,),
        ).fetchone()
        self.storage.connection.execute(
            "UPDATE status_events SET payload_hash=? WHERE event_id=?", ("0" * 64, row["event_id"])
        )
        self.storage.connection.commit()
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(
                closed, result, created_at_unix_ns=100, **children
            )

        closed, result, children, _ = self._persist("tampered-bundle")
        row = self.storage.connection.execute(
            "SELECT event_id,payload_json FROM status_events WHERE run_id=? AND event_kind='common_result'",
            (closed.run_id,),
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["bundle_sha256"] = "0" * 64
        self.storage.connection.execute(
            "UPDATE status_events SET payload_json=? WHERE event_id=?",
            (canonical_json(payload), row["event_id"]),
        )
        self.storage.connection.commit()
        with self.assertRaises(StorageError):
            self.storage.persist_common_result(
                closed, result, created_at_unix_ns=100, **children
            )

    def test_same_run_two_connection_race_has_no_duplicate_or_partial_bundle(self):
        path = self._fresh_database("race")
        closed, result = self._closed_result("race-run")
        children = self._children()
        barrier = threading.Barrier(2)
        outcomes = []
        lock = threading.Lock()

        def persist_from_second_connection():
            try:
                with Storage.open(path) as storage:
                    barrier.wait(timeout=5)
                    outcome = storage.persist_common_result(
                        closed, result, created_at_unix_ns=100, **children
                    )
                    with lock:
                        outcomes.append(outcome.state)
            except Exception as exc:  # test the bounded SQLite contention contract
                with lock:
                    outcomes.append(exc)

        threads = [threading.Thread(target=persist_from_second_connection) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcomes), 2)
        errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
        self.assertLessEqual(len(errors), 1)
        if errors:
            self.assertIsInstance(errors[0], StorageError)
            self.assertIn("database is locked", str(errors[0]))
        else:
            self.assertEqual(sorted(outcomes), ["idempotent", "inserted"])
        self.assertEqual(self._count_run_rows(path, closed.run_id), {
            "runs": 1, "status_events": 1, "raw_samples": 1,
            "scalar_metrics": 2, "correctness_metrics": 1, "artifacts": 1,
        })

    def test_migration_identity_and_append_tables(self):
        self.storage.append_status_event("run-001", "created", "open", {"x": 1})
        self.storage.append_raw_sample("run-001", "s0", 0, "baseline", 1.5, "ns", sample_kind="timing", sample_index=0)
        self.storage.append_raw_sample("run-001", "s0", 0, "baseline", 1.6, "ns", sample_kind="timing", sample_index=1)
        self.storage.append_scalar_metric("run-001", "median", 1.5, "ns")
        self.storage.append_scalar_metric("run-001", "rss", None, "bytes", missing_reason="unavailable")
        self.storage.append_correctness_metric("run-001", "case", "max", 0.0, "abs", True)
        self.storage.append_artifact("run-001", "manifest", "json", "a" * 64, {"kind": "manifest"})
        self.assertEqual(self.storage.get_run("run-001")["manifest"]["phase"], "H0")
        self.assertEqual(len(self.storage.rows("raw_samples", "run-001")), 2)
        self.assertEqual(self.storage.get_run("run-001")["created_at_unix_ns"], 1)
        self.storage.close()
        with Storage.open(self.path) as reopened:
            self.assertEqual(reopened.get_run("run-001")["manifest"]["mode"], "eager_baseline")

    def test_foreign_keys_and_append_only_triggers(self):
        with self.assertRaises(StorageError):
            self.storage.append_status_event("missing", "x", "y", {})
        with self.assertRaises(sqlite3.IntegrityError):
            self.storage.connection.execute(
                """INSERT INTO runs
                   (run_id, phase, mode, manifest_json, manifest_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                ("run-invalid-mode", "H0", "free_code", "{}", "a" * 64),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.storage.connection.execute("UPDATE runs SET mode='aa_gpu' WHERE run_id='run-001'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.storage.connection.execute("DELETE FROM runs WHERE run_id='run-001'")
        for table in ("schema_migrations", "db_identity"):
            with self.assertRaises(sqlite3.IntegrityError):
                column = "name" if table == "schema_migrations" else "value"
                self.storage.connection.execute(f"UPDATE {table} SET {column}='changed'")
            with self.assertRaises(sqlite3.IntegrityError):
                self.storage.connection.execute(f"DELETE FROM {table}")

    def test_mode_ro_dashboard_cannot_write(self):
        self.storage.close()
        with Storage.open(self.path, read_only=True) as readonly:
            self.assertEqual(readonly.get_run("run-001")["run_id"], "run-001")
            with self.assertRaises(StorageError):
                readonly.append_scalar_metric("run-001", "x", 1.0, "ns")
            self.assertEqual(readonly.connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(readonly.connection.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                readonly.connection.execute("INSERT INTO status_events(run_id, event_kind, status, payload_json, payload_hash) VALUES ('run-001','x','x','{}',?)", ("a" * 64,))

    def test_migration_is_idempotent_and_values_are_validated(self):
        self.storage.close()
        with Storage.open(self.path) as reopened:
            self.assertEqual(reopened.connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0], 1)
            with self.assertRaises(StorageError):
                reopened.append_scalar_metric("run-001", "x", float("nan"), "ns")
            with self.assertRaises(StorageError):
                reopened.append_scalar_metric("run-001", "x", None, "bytes")
            with self.assertRaises(StorageError):
                reopened.append_scalar_metric("run-001", "x", 0.0, "bytes", missing_reason="also present")
            with self.assertRaises(StorageError):
                reopened.append_correctness_metric("run-001", "case", "x", 0.0, "x", 1)

    def test_failed_migration_rolls_back_partial_schema(self):
        path = Path(self.tempdir.name) / "failed.sqlite3"
        connection = sqlite3.connect(path)
        with self.assertRaises(sqlite3.Error):
            connection.executescript(
                "BEGIN IMMEDIATE; CREATE TABLE partial_marker(x INTEGER); SELECT * FROM absent_table; COMMIT;"
            )
        connection.rollback()
        self.assertEqual(
            connection.execute("SELECT 1 FROM sqlite_master WHERE name='partial_marker'").fetchone(),
            None,
        )
        connection.close()

    def test_unversioned_nonempty_database_is_not_silently_upgraded(self):
        path = Path(self.tempdir.name) / "unversioned.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.commit()
        connection.close()
        with self.assertRaises(StorageError):
            Storage.open(path)
