"""Append-only hash-chain and read-only dashboard tests."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

from friday_h1.canonical import canonical_sha256
from friday_runtime.constants import H1_DECISION_RECORD_ID, RUNTIME_ID, SCHEMA_VERSION
from friday_runtime.dashboard import DashboardError, DashboardService, _html, _target
from friday_runtime.history import History, HistoryConflict, HistoryError


def provenance() -> dict[str, object]:
    code_files = {"friday_runtime/example.py": "1" * 64}
    spec_files = {"docs/example.md": "2" * 64}
    environment = {"python": "test"}
    hardware = {"machine": "test"}
    body = {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "git_revision": "a" * 40,
        "git_dirty": False,
        "git_diff_sha256": "0" * 64,
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
    }
    body["provenance_sha256"] = canonical_sha256(body)
    return body


def report(run_id: str, *, kind: str = "policy_overhead") -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "formal_claim": False,
        "h1_decision_record_id": H1_DECISION_RECORD_ID,
        "kind": kind,
        "run_id": run_id,
        "status": f"{kind}_passed",
        "policy": {"authorized": True},
        "metrics": {"gate_passed": True, "policy_median_ns": 1000},
        "blocks": [],
    }


class RuntimeHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "runtime.sqlite3"
        self.provenance = provenance()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_history_is_private_hash_chained_and_dashboard_is_read_only(self) -> None:
        with History.open(self.path, initialize=True) as history:
            first = history.persist(report("one"), self.provenance, created_at_unix_ns=1)
            second = history.persist(
                report("two", kind="runtime_validation"),
                self.provenance,
                created_at_unix_ns=2,
            )
            rows = history.verified_records()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertIsNone(rows[0]["previous_record_id"])
        self.assertEqual(rows[1]["previous_record_id"], first.record_id)
        self.assertEqual(rows[1]["record_id"], second.record_id)

        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        snapshot = DashboardService(self.path).snapshot()
        document = _html(snapshot).decode("utf-8")
        self.assertEqual(snapshot["total"], 2)
        self.assertTrue(snapshot["hash_chain_verified"])
        self.assertIn("Friday bounded runtime", document)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_update_delete_read_only_and_duplicate_are_blocked(self) -> None:
        with History.open(self.path, initialize=True) as history:
            history.persist(report("one"), self.provenance, created_at_unix_ns=1)
            with self.assertRaises(HistoryConflict):
                history.persist(report("one"), self.provenance, created_at_unix_ns=2)
            with self.assertRaises(sqlite3.DatabaseError):
                history.connection.execute("UPDATE records SET status='changed'")
            with self.assertRaises(sqlite3.DatabaseError):
                history.connection.execute("DELETE FROM records")
        with History.open(self.path, read_only=True) as history:
            with self.assertRaises((sqlite3.DatabaseError, HistoryError)):
                history.connection.execute("INSERT INTO records(record_id) VALUES('x')")

    def test_symlink_and_invalid_request_targets_are_rejected(self) -> None:
        with History.open(self.path, initialize=True):
            pass
        link = self.path.with_name("link.sqlite3")
        os.symlink(self.path, link)
        with self.assertRaises(HistoryError):
            History.open(link, read_only=True)
        self.assertEqual(_target("/api/snapshot?limit=1"), ("/api/snapshot", {"limit": ["1"]}))
        for target in ("https://example.test/", "/x#fragment", "/ümlaut"):
            with self.subTest(target=target), self.assertRaises(DashboardError):
                _target(target)


if __name__ == "__main__":
    unittest.main()
