from __future__ import annotations

import copy
import hashlib
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from friday_h01.analysis import analyze_trace
from friday_h01 import storage as storage_module
from friday_h01.canonical import canonical_sha256
from friday_h01.import_h0 import (
    COMPLETED_ADAPTER,
    INVALID_ADAPTER,
    STATIC_ADAPTER_REGISTRY,
    W1V3_COMPLETED_ADAPTER,
)
from friday_h01.storage import (
    BundleError,
    ReadOnlyStorageError,
    SchemaError,
    Storage,
    StorageConflict,
    StorageError,
    build_bundle,
    legacy_warmup_statistics,
)
from tests.test_h01_analysis import BASE_NS, hash_noise_main
from tests.test_h01_protocol import make_manifest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def legacy_payload(
    entity_id: str = "legacy-h0-observation-1",
    *,
    adapter_id: str = COMPLETED_ADAPTER,
    observation: object | None = None,
) -> dict[str, object]:
    source_run_id = "h0-parent-offline-fixture"
    warmup_count = {
        COMPLETED_ADAPTER: 11,
        W1V3_COMPLETED_ADAPTER: 8,
        INVALID_ADAPTER: 16,
    }[adapter_id]
    warmup_ns = [1_000_000 + index for index in range(warmup_count)]
    descriptor = next(
        item for item in STATIC_ADAPTER_REGISTRY.descriptors if item.adapter_id == adapter_id
    )
    raw_warmup_sha256 = canonical_sha256(warmup_ns)
    binding = {
        "registry_schema_version": 1,
        "registry_sha256": STATIC_ADAPTER_REGISTRY.registry_sha256,
        "descriptor_sha256": descriptor.descriptor_sha256,
        "selector_sha256": canonical_sha256(descriptor.selector),
        "parser_id": descriptor.parser_id,
        "raw_warmup_sha256": raw_warmup_sha256,
    }
    completed = adapter_id != INVALID_ADAPTER
    source_status = "completed" if completed else "invalid"
    source_classification = "measurement_complete" if completed else "invalid"
    source_error_code = None if completed else "warmup_unstable"
    source_diagnostic = (
        None
        if completed
        else {
            "schema_version": 1,
            "code": "warmup_unstable",
            "details": {"warmups_ns": warmup_ns},
        }
    )
    lineage = {
        "parent_phase": "H0",
        "parent_run_id": source_run_id,
        "parent_manifest_sha256": _digest("legacy-parent-manifest"),
        "parent_result_sha256": _digest("legacy-parent-result"),
        "parent_evidence_sha256": _digest("legacy-parent-evidence"),
        "parent_bundle_sha256": _digest("legacy-parent-bundle"),
        "parent_code_sha256": _digest("legacy-parent-code"),
        "parent_spec_sha256": _digest("legacy-parent-spec"),
        "parent_environment_sha256": _digest("legacy-parent-environment"),
        "source_database_sha256": _digest("legacy-source-database"),
        "registry_sha256": binding["registry_sha256"],
        "descriptor_sha256": binding["descriptor_sha256"],
        "selector_sha256": binding["selector_sha256"],
        "raw_warmup_sha256": binding["raw_warmup_sha256"],
    }
    return {
        "entity_id": entity_id,
        "entity_kind": "legacy_h0_warmup_observation",
        "status": "legacy_observation",
        "created_at_unix_ns": 10,
        "manifest": {
            "schema_version": 1,
            "entity_id": entity_id,
            "source_phase": "H0",
            "source_run_id": source_run_id,
            "source_mode": "eager_baseline",
            "source_status": source_status,
            "source_classification": source_classification,
            "source_created_at_unix_ns": 10,
            "observation_kind": "warmup_observation",
            "adapter": adapter_id,
            **binding,
        },
        "trace": {
            "schema_version": 1,
            "observation": {
                "adapter": adapter_id,
                "source_status": source_status,
                "source_classification": source_classification,
                "source_error_code": source_error_code,
                "warmup_ns": warmup_ns,
                "statistics": legacy_warmup_statistics(warmup_ns),
                "source_diagnostic": source_diagnostic,
                **binding,
            }
            if observation is None
            else observation,
        },
        "result": {
            "schema_version": 1,
            "status": "legacy_observation",
            "conclusion": "historical_warmup_observation_only",
            "interpretation": "descriptive_only",
            "action": "no_h0_conclusion",
            "stationarity_supported": False,
            "paced_gate_applicable": False,
            "h0_reclassification": False,
            "promotion_applicable": False,
        },
        "lineage": lineage,
    }


def session_payload(*, invalid: bool = False) -> dict[str, object]:
    manifest = make_manifest("C0")
    from friday_h01.protocol import build_trace
    from friday_h01.constants import BURN_IN_SAMPLES

    durations = [BASE_NS] * BURN_IN_SAMPLES + hash_noise_main("storage-session")
    trace = build_trace(manifest, durations)
    if invalid:
        del trace["samples"][4]
    result = analyze_trace(manifest, trace)
    return {
        "entity_id": manifest["run_id"],
        "entity_kind": "paced_session",
        "status": result["status"],
        "created_at_unix_ns": 20,
        "manifest": manifest,
        "trace": trace,
        "result": result,
        "lineage": manifest["source"],
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_row_without_update_trigger(path: Path, sql: str, parameters: tuple[object, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            "DROP TRIGGER bundles_no_update;"
            "CREATE TRIGGER bundles_no_update BEFORE UPDATE ON bundles "
            "BEGIN SELECT RAISE(ABORT, 'bundles are append-only'); END;"
        )
        # Recreate-then-update would be blocked, so temporarily remove and restore atomically.
        connection.execute("DROP TRIGGER bundles_no_update")
        connection.execute(sql, parameters)
        connection.execute(
            "CREATE TRIGGER bundles_no_update BEFORE UPDATE ON bundles "
            "BEGIN SELECT RAISE(ABORT, 'bundles are append-only'); END"
        )
        connection.commit()
    finally:
        connection.close()


class H01StorageTests(unittest.TestCase):
    def test_database_file_permissions_are_private_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h01.sqlite3"
            with Storage.open(path) as storage:
                storage.persist_bundle(**legacy_payload("legacy-private-mode"))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            os.chmod(path, 0o644)
            with self.assertRaises(StorageError):
                Storage.open(path, read_only=True)
            with self.assertRaises(StorageError):
                Storage.open(path)

            os.chmod(path, 0o400)
            with Storage.open(path, read_only=True) as storage:
                self.assertEqual(storage.count(), 1)
            with self.assertRaises(StorageError):
                Storage.open(path)

            os.chmod(path, 0o600)

    def test_insert_replay_idempotence_and_conflict_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h01.sqlite3"
            payload = session_payload()
            expected = build_bundle(**payload)
            with Storage.open(path) as first_storage, Storage.open(path) as second_storage:
                first = first_storage.persist_bundle(**payload)
                second = second_storage.persist_bundle(**payload)
                self.assertEqual(first.state, "inserted")
                self.assertEqual(second.state, "idempotent")
                self.assertEqual(first.bundle_sha256, expected["bundle_sha256"])
                self.assertEqual(
                    second_storage.get_verified_bundle(payload["entity_id"]), expected
                )
                conflict = dict(payload)
                conflict["created_at_unix_ns"] = 21
                with self.assertRaises(StorageConflict):
                    second_storage.persist_bundle(**conflict)
                self.assertEqual(second_storage.count(), 1)
                self.assertEqual(
                    second_storage.get_verified_bundle(payload["entity_id"]), expected
                )

    def test_batch_persistence_is_atomic_ordered_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h01.sqlite3"
            batch = tuple(
                legacy_payload(f"legacy-batch-{index}") for index in range(3)
            )
            with Storage.open(path) as storage:
                first = storage.persist_bundles(batch)
                second = storage.persist_bundles(batch)
                self.assertEqual([item.state for item in first], ["inserted"] * 3)
                self.assertEqual([item.state for item in second], ["idempotent"] * 3)
                self.assertEqual(
                    [item.entity_id for item in first],
                    [item["entity_id"] for item in batch],
                )
                self.assertEqual(storage.count(), 3)

            rollback_path = Path(directory) / "rollback.sqlite3"
            existing = legacy_payload("legacy-batch-conflict")
            conflict = copy.deepcopy(existing)
            conflict["created_at_unix_ns"] = 11
            new_item = legacy_payload("legacy-batch-must-rollback")
            with Storage.open(rollback_path) as storage:
                storage.persist_bundle(**existing)
                with self.assertRaises(StorageConflict):
                    storage.persist_bundles((new_item, conflict))
                self.assertEqual(storage.count(), 1)
                self.assertIsNone(
                    storage.get_verified_bundle(new_item["entity_id"])
                )
                with self.assertRaises(BundleError):
                    storage.persist_bundles((existing, existing))
                self.assertEqual(storage.persist_bundles(()), ())

    def test_append_only_triggers_and_authorizer_failure_leave_no_partial_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h01.sqlite3"
            payload = legacy_payload()
            with Storage.open(path) as storage:
                connection = storage._connection
                for statement in (
                    "UPDATE bundles SET status = status",
                    "DELETE FROM bundles",
                ):
                    storage.persist_bundle(**payload)
                    with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(statement)
                    self.assertEqual(storage.count(), 1)

                stored = list(connection.execute("SELECT * FROM bundles").fetchone())
                replacements = {
                    "same_bytes": stored,
                    "different_bytes": [
                        *stored[:4],
                        stored[4] + 1,
                        *stored[5:],
                    ],
                    "different_id_same_hash": ["replacement-id", *stored[1:]],
                }
                placeholders = ",".join("?" for _ in stored)
                for recursive in (0, 1):
                    connection.execute(f"PRAGMA recursive_triggers = {recursive}")
                    self.assertEqual(
                        connection.execute("PRAGMA recursive_triggers").fetchone(),
                        (recursive,),
                    )
                    for name, replacement in replacements.items():
                        with (
                            self.subTest(recursive=recursive, replacement=name),
                            self.assertRaises(sqlite3.IntegrityError),
                        ):
                            connection.execute(
                                f"INSERT OR REPLACE INTO bundles VALUES ({placeholders})",
                                replacement,
                            )
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM bundles").fetchone(), (1,))
                connection.execute("PRAGMA recursive_triggers = ON")
                self.assertEqual(connection.execute("PRAGMA recursive_triggers").fetchone(), (1,))

            empty_path = Path(directory) / "atomic.sqlite3"
            with Storage.open(empty_path) as storage:
                def authorizer(action: int, arg1: str | None, _arg2: str | None, _db: str | None, _source: str | None) -> int:
                    if action == sqlite3.SQLITE_INSERT and arg1 == "bundles":
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK

                storage._connection.set_authorizer(authorizer)
                with self.assertRaises(sqlite3.DatabaseError):
                    storage.persist_bundle(**payload)
                storage._connection.set_authorizer(None)
                self.assertFalse(storage._connection.in_transaction)
                self.assertEqual(storage.count(), 0)

    def test_exact_schema_rejects_extra_missing_changed_column_index_and_trigger(self) -> None:
        migration_path = Path(__file__).parents[1] / "friday_h01" / "migrations" / "0001_initial.sql"
        registered = migration_path.read_text(encoding="utf-8")
        mutations = {
            "extra_table": (
                "PRAGMA user_version = 1;",
                "CREATE TABLE unexpected(value INTEGER);\nPRAGMA user_version = 1;",
            ),
            "missing_column": ("    lineage_json TEXT NOT NULL,\n", ""),
            "changed_column": ("    trace_json TEXT NOT NULL,", "    trace_json BLOB NOT NULL,"),
            "missing_index": (
                "CREATE INDEX idx_bundles_kind_created\n    ON bundles(entity_kind, created_at_unix_ns DESC, entity_id);\n",
                "",
            ),
            "changed_index": (
                "ON bundles(status, created_at_unix_ns DESC, entity_id);",
                "ON bundles(status, entity_id);",
            ),
            "missing_trigger": (
                "CREATE TRIGGER bundles_no_delete\nBEFORE DELETE ON bundles\nBEGIN\n    SELECT RAISE(ABORT, 'bundles are append-only');\nEND;\n",
                "",
            ),
            "extra_view": (
                "PRAGMA user_version = 1;",
                "CREATE VIEW unexpected_view AS SELECT entity_id FROM bundles;\n"
                "PRAGMA user_version = 1;",
            ),
            "extra_index": (
                "PRAGMA user_version = 1;",
                "CREATE INDEX unexpected_index ON bundles(created_at_unix_ns);\n"
                "PRAGMA user_version = 1;",
            ),
            "extra_trigger": (
                "PRAGMA user_version = 1;",
                "CREATE TRIGGER unexpected_trigger AFTER INSERT ON bundles "
                "BEGIN SELECT 1; END;\nPRAGMA user_version = 1;",
            ),
            "changed_trigger_body": (
                "SELECT RAISE(ABORT, 'bundles are append-only');",
                "SELECT RAISE(ABORT, 'changed append-only body');",
            ),
            "changed_autoindex": (
                "    bundle_sha256 TEXT NOT NULL UNIQUE",
                "    bundle_sha256 TEXT NOT NULL COLLATE NOCASE UNIQUE",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for index, (name, (old, new)) in enumerate(mutations.items()):
                with self.subTest(drift=name):
                    self.assertIn(old, registered)
                    path = Path(directory) / f"drift-{index}.sqlite3"
                    connection = sqlite3.connect(path, isolation_level=None)
                    try:
                        connection.executescript(registered.replace(old, new, 1))
                    finally:
                        connection.close()
                    os.chmod(path, 0o600)
                    with self.assertRaises(SchemaError):
                        Storage.open(path, read_only=True)

    def test_tampered_json_is_rejected_after_exact_trigger_restoration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h01.sqlite3"
            payload = legacy_payload()
            with Storage.open(path) as storage:
                storage.persist_bundle(**payload)
            _rewrite_row_without_update_trigger(
                path,
                "UPDATE bundles SET trace_json = ? WHERE entity_id = ?",
                ("{malformed", payload["entity_id"]),
            )
            with Storage.open(path, read_only=True) as storage:
                with self.assertRaises(BundleError):
                    storage.get_verified_bundle(payload["entity_id"])
            with self.assertRaises(BundleError):
                Storage.open(path)

    def test_read_only_verification_and_queries_do_not_change_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h01.sqlite3"
            payload = legacy_payload()
            with Storage.open(path) as storage:
                storage.persist_bundle(**payload)
            before = file_sha256(path)
            with Storage.open(path, read_only=True) as storage:
                self.assertEqual(storage._connection.execute("PRAGMA query_only").fetchone(), (1,))
                storage.verify_schema()
                self.assertIsNotNone(storage.get_verified_bundle(payload["entity_id"]))
                with self.assertRaises(ReadOnlyStorageError):
                    storage.persist_bundle(**payload)
            self.assertEqual(file_sha256(path), before)

    def test_malformed_huge_bool_int_and_legacy_stationarity_are_fail_closed(self) -> None:
        base = legacy_payload()
        cases: dict[str, dict[str, object]] = {}
        boolean_time = copy.deepcopy(base)
        boolean_time["created_at_unix_ns"] = True
        cases["bool_time"] = boolean_time
        huge_time = copy.deepcopy(base)
        huge_time["created_at_unix_ns"] = 1 << 63
        cases["huge_time"] = huge_time
        float_version = copy.deepcopy(base)
        float_version["manifest"]["schema_version"] = 1.0
        cases["float_schema_version"] = float_version
        huge_json = copy.deepcopy(base)
        huge_json["trace"]["observation"] = "x" * 70_000
        cases["huge_json_string"] = huge_json
        nonfinite = copy.deepcopy(base)
        nonfinite["trace"]["observation"] = float("nan")
        cases["nonfinite"] = nonfinite
        legacy_claim = copy.deepcopy(base)
        legacy_claim["status"] = "h01_stationarity_supported"
        legacy_claim["result"]["status"] = "h01_stationarity_supported"
        cases["legacy_stationarity"] = legacy_claim
        promotion = copy.deepcopy(base)
        promotion["result"]["promotion_applicable"] = True
        cases["legacy_promotion"] = promotion
        for name, payload in cases.items():
            with self.subTest(case=name), self.assertRaises(BundleError):
                build_bundle(**payload)

    def test_source_path_uri_missing_file_and_limit_bounds_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.sqlite3"
            with self.assertRaises(StorageError):
                Storage.open("file:/tmp/h01.sqlite3?mode=rw", read_only=True)
            with self.assertRaises(StorageError):
                Storage.open(path, read_only=True)
            real_path = Path(directory) / "h01.sqlite3"
            with Storage.open(real_path) as storage:
                with self.assertRaises(StorageError):
                    storage.recent(True)
                with self.assertRaises(StorageError):
                    storage.recent(201)

    def test_symlink_and_injected_file_swaps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.sqlite3"
            replacement = root / "replacement.sqlite3"
            with Storage.open(original) as storage:
                storage.persist_bundle(**legacy_payload("legacy-original"))
            with Storage.open(replacement) as storage:
                storage.persist_bundle(**legacy_payload("legacy-replacement"))

            link = root / "linked.sqlite3"
            os.symlink(original, link)
            with self.assertRaises(StorageError):
                Storage.open(link, read_only=True)

            backup = root / "original-backup.sqlite3"
            real_connect = storage_module._connect_database

            def swap_after_connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
                connection = real_connect(path, read_only=read_only)
                os.replace(original, backup)
                os.replace(replacement, original)
                return connection

            storage_module._connect_database = swap_after_connect
            try:
                with self.assertRaises(StorageError):
                    Storage.open(original, read_only=True)
            finally:
                storage_module._connect_database = real_connect

    def test_injected_swap_after_begin_fails_transaction_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "h01.sqlite3"
            replacement = root / "replacement.sqlite3"
            with Storage.open(path) as storage:
                storage.persist_bundle(**legacy_payload("legacy-before-begin"))
            with Storage.open(replacement) as storage:
                storage.persist_bundle(**legacy_payload("legacy-after-begin"))

            backup = root / "original.sqlite3"
            with Storage.open(path, read_only=True) as storage:
                real_after_begin = storage_module._verify_after_begin

                def swap_and_verify(binding: object, connection: sqlite3.Connection) -> None:
                    os.replace(path, backup)
                    os.replace(replacement, path)
                    real_after_begin(binding, connection)

                storage_module._verify_after_begin = swap_and_verify
                try:
                    with self.assertRaises(StorageError):
                        with storage.read_transaction():
                            self.fail("swapped transaction must not yield")
                finally:
                    storage_module._verify_after_begin = real_after_begin


if __name__ == "__main__":
    unittest.main()
