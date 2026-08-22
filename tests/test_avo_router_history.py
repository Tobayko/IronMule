from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from friday_avo_router.cli import main
from friday_avo_router.constants import ROUTER_ID, SCHEMA_VERSION
from friday_avo_router.dashboard import DashboardService, _html, _snapshot_limit
from friday_avo_router.history import History, HistoryConflict, HistoryError
from friday_n10_v2.canonical import canonical_sha256


def provenance() -> dict[str, object]:
    code_files = {"friday_avo_router/router.py": "a" * 64}
    spec_files = {"docs/AVO_SHADOW_ROUTER_SPEC.md": "b" * 64}
    environment = {"python": "test"}
    hardware = {"machine": "test"}
    value: dict[str, object] = {
        "router_id": ROUTER_ID,
        "schema_version": SCHEMA_VERSION,
        "git_revision": "1" * 40,
        "git_dirty": False,
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
    }
    value["provenance_sha256"] = canonical_sha256(value)
    return value


def report(run_id: str, kind: str = "policy_overhead", status: str = "policy_overhead_passed") -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "router_id": ROUTER_ID,
        "run_id": run_id,
        "kind": kind,
        "status": status,
        "formal_claim": False,
        "decision_record_ids": {"n8": "7" * 64, "n10": "8" * 64},
        "router": {"ready": True, "enforced_plan": "serial_shadow_only"},
        "metrics": {"gate_passed": True},
    }


class RouterHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "router.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_private_append_only_chain_and_dashboard(self) -> None:
        with History.open(self.path, initialize=True) as history:
            first = history.persist(report("policy"), provenance(), created_at_unix_ns=1)
            second = history.persist(
                report("shadow", "shadow_validation", "shadow_router_validated"),
                provenance(),
                created_at_unix_ns=2,
            )
            records = history.verified_records()
            self.assertEqual(records[1]["previous_record_id"], first.record_id)
            self.assertEqual(records[1]["record_id"], second.record_id)
            with self.assertRaises(sqlite3.DatabaseError):
                history.connection.execute("UPDATE records SET status='x'")
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        snapshot = DashboardService(self.path).snapshot(2)
        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(snapshot["total"], 2)
        self.assertTrue(snapshot["hash_chain_verified"])
        self.assertEqual(before, after)

    def test_duplicate_run_with_different_bytes_is_rejected(self) -> None:
        with History.open(self.path, initialize=True) as history:
            history.persist(report("same"), provenance(), created_at_unix_ns=1)
            changed = report("same")
            changed["metrics"] = {"gate_passed": False}
            with self.assertRaises(HistoryConflict):
                history.persist(changed, provenance(), created_at_unix_ns=2)

    def test_world_readable_database_is_rejected(self) -> None:
        with History.open(self.path, initialize=True):
            pass
        os.chmod(self.path, 0o644)
        with self.assertRaises(HistoryError):
            History.open(self.path, read_only=True)

    def test_same_named_but_modified_trigger_is_rejected(self) -> None:
        with History.open(self.path, initialize=True):
            pass
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("DROP TRIGGER records_no_update")
            connection.execute(
                "CREATE TRIGGER records_no_update BEFORE UPDATE ON records "
                "BEGIN SELECT RAISE(ABORT, 'foreign trigger'); END"
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(HistoryError):
            History.open(self.path, read_only=True)

    def test_timestamp_tampering_breaks_the_hash_chain(self) -> None:
        with History.open(self.path, initialize=True) as history:
            history.persist(report("policy"), provenance(), created_at_unix_ns=1)
        connection = sqlite3.connect(self.path)
        try:
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='records_no_update'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER records_no_update")
            connection.execute("UPDATE records SET created_at_unix_ns=2")
            connection.execute(trigger_sql)
            connection.commit()
        finally:
            connection.close()
        with History.open(self.path, read_only=True) as history:
            with self.assertRaises(HistoryError):
                history.verified_records()

    def test_internally_inconsistent_provenance_is_rejected(self) -> None:
        changed = provenance()
        changed["code_sha256"] = "c" * 64
        without_outer = dict(changed)
        without_outer.pop("provenance_sha256")
        changed["provenance_sha256"] = canonical_sha256(without_outer)
        with History.open(self.path, initialize=True) as history:
            with self.assertRaises(HistoryError):
                history.persist(report("policy"), changed, created_at_unix_ns=1)

    def test_symlink_database_is_rejected(self) -> None:
        target = Path(self.temp.name) / "target.sqlite3"
        with History.open(target, initialize=True):
            pass
        self.path.symlink_to(target)
        with self.assertRaises(HistoryError):
            History.open(self.path, read_only=True)

    def test_dashboard_contract_and_live_commands_are_closed_without_execute(self) -> None:
        self.assertIn(b"serial_shadow_only", _html())
        self.assertEqual(_snapshot_limit("/api/snapshot?limit=2"), 2)
        self.assertEqual(
            main(["--database", str(self.path), "benchmark-policy"]), 78
        )
        self.assertEqual(main(["--database", str(self.path), "validate-shadow"]), 78)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
