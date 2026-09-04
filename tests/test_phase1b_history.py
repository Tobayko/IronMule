from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from friday_phase1b.canonical import canonical_sha256
from friday_phase1b.constants import (
    BENCHMARK_RUN_ID,
    CONTRACT_ID,
    EXPERIMENT_ID,
    QUALIFICATION_RUN_ID,
    SCHEMA_VERSION,
)
from friday_phase1b.dashboard import DashboardService, _html, _snapshot_limit, _trusted_host
from friday_phase1b.experiment import scope
from friday_phase1b.history import History, HistoryConflict, HistoryError
from friday_phase1b.kernel_source import KERNEL_NAME, KERNEL_SOURCE_SHA256


def provenance() -> dict[str, object]:
    code_files = {"friday_phase1b/kernel.py": "a" * 64}
    spec_files = {"docs/PHASE1B_RESIDUAL_RMSNORM_SPEC.md": "b" * 64}
    source = {"kernel_name": KERNEL_NAME, "source_sha256": KERNEL_SOURCE_SHA256}
    environment = {"python": "test"}
    hardware = {"machine": "test"}
    value: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "contract_id": CONTRACT_ID,
        "schema_version": SCHEMA_VERSION,
        "git_revision": "1" * 40,
        "git_dirty": False,
        "git_status_sha256": hashlib.sha256(b"").hexdigest(),
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "source": source,
        "source_binding_sha256": canonical_sha256(source),
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
    }
    value["provenance_sha256"] = canonical_sha256(value)
    return value


def report(
    run_id: str = QUALIFICATION_RUN_ID,
    *,
    kind: str = "qualification",
    status: str = "qualification_passed",
    action: str = "qualification_only",
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "kind": kind,
        "status": status,
        "formal_claim": False,
        "action": action,
        "scope": scope(),
        "metrics": {"gate_passed": True},
    }


class Phase1BHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "phase1b.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_private_append_only_chain_and_read_only_dashboard(self) -> None:
        with History.open(self.path, initialize=True) as history:
            first = history.persist(report(), provenance(), created_at_unix_ns=1)
            second = history.persist(
                report(
                    BENCHMARK_RUN_ID,
                    kind="benchmark",
                    status="candidate_inconclusive",
                    action="baseline_fallback",
                ),
                provenance(),
                created_at_unix_ns=2,
            )
            records = history.verified_records()
            self.assertEqual(records[1]["previous_record_id"], first.record_id)
            self.assertEqual(records[1]["record_id"], second.record_id)
            with self.assertRaises(sqlite3.DatabaseError):
                history.connection.execute("DELETE FROM records")
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        snapshot = DashboardService(self.path).snapshot(2)
        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(snapshot["total"], 2)
        self.assertTrue(snapshot["hash_chain_verified"])
        self.assertEqual(before, after)

    def test_dashboard_host_allowlist_rejects_dns_rebinding_names(self) -> None:
        self.assertTrue(_trusted_host("127.0.0.1:8774", 8774))
        self.assertTrue(_trusted_host("localhost:8774", 8774))
        self.assertFalse(_trusted_host("attacker.example:8774", 8774))
        self.assertFalse(_trusted_host(None, 8774))

    def test_duplicate_and_tampering_are_rejected(self) -> None:
        with History.open(self.path, initialize=True) as history:
            history.persist(report(), provenance(), created_at_unix_ns=1)
            changed = report()
            changed["metrics"] = {"gate_passed": False}
            with self.assertRaises(HistoryConflict):
                history.persist(changed, provenance(), created_at_unix_ns=2)
        connection = sqlite3.connect(self.path)
        try:
            trigger = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='records_no_update'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER records_no_update")
            connection.execute("UPDATE records SET created_at_unix_ns=9")
            connection.execute(trigger)
            connection.commit()
        finally:
            connection.close()
        with History.open(self.path, read_only=True) as history:
            with self.assertRaises(HistoryError):
                history.verified_records()

    def test_symlink_world_mode_and_source_binding_are_rejected(self) -> None:
        target = Path(self.temp.name) / "target.sqlite3"
        with History.open(target, initialize=True):
            pass
        self.path.symlink_to(target)
        with self.assertRaises(HistoryError):
            History.open(self.path, read_only=True)
        os.chmod(target, 0o644)
        with self.assertRaises(HistoryError):
            History.open(target, read_only=True)
        changed = provenance()
        changed["source"] = {"kernel_name": "wrong", "source_sha256": "c" * 64}
        changed["source_binding_sha256"] = canonical_sha256(changed["source"])
        outer = dict(changed)
        outer.pop("provenance_sha256")
        changed["provenance_sha256"] = canonical_sha256(outer)
        other = Path(self.temp.name) / "other.sqlite3"
        with History.open(other, initialize=True) as history:
            with self.assertRaises(HistoryError):
                history.persist(report(), changed)

    def test_dashboard_contract_is_closed(self) -> None:
        self.assertIn(b"no runtime activation", _html())
        self.assertEqual(_snapshot_limit("/api/snapshot?limit=2"), 2)
        with self.assertRaises(Exception):
            _snapshot_limit("/api/snapshot?path=/tmp/x")


if __name__ == "__main__":
    unittest.main()
