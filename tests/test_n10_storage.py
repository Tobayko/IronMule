"""SQLite append-only and replay tests for N10-v1."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from friday_n10.canonical import canonical_sha256
from friday_n10.constants import CALIBRATION, CONFIRMATION, SCHEMA_VERSION, SESSION_ORDER, STUDY_ID
from friday_n10.protocol import (
    build_calibration_summary,
    build_confirmation_seal,
    build_preregistration,
    build_study_decision,
)
from friday_n10.storage import Storage, StorageConflict, StorageError
from tests.test_n10_protocol import PROVENANCE, complete_study, session


def provenance() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "marker": "offline-test",
    }
    value["provenance_sha256"] = canonical_sha256(value)
    return value


class FormalStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "n10.sqlite3"
        self.provenance = provenance()
        self.preregistration = build_preregistration(self.provenance["provenance_sha256"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_initialization_persistence_and_read_only_verification(self) -> None:
        with Storage.open(self.path, initialize=True) as storage:
            outcome = storage.persist(self.preregistration, self.provenance, created_at_unix_ns=1)
            self.assertEqual(outcome.state, "persisted")
            self.assertEqual(len(storage.verified_records()), 1)
        self.assertEqual(os.stat(self.path).st_mode & 0o077, 0)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with Storage.open(self.path, read_only=True) as storage:
            with storage.read_transaction():
                self.assertEqual(storage.verified_records()[0]["payload"], self.preregistration)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_duplicate_session_and_out_of_order_session_are_rejected(self) -> None:
        with Storage.open(self.path, initialize=True) as storage:
            storage.persist(self.preregistration, self.provenance, created_at_unix_ns=1)
            c0 = session(
                CALIBRATION, "C0", ratio=1.0, preregistration=self.preregistration
            )
            storage.persist(c0, self.provenance, created_at_unix_ns=2)
            with self.assertRaises(StorageConflict):
                storage.persist(c0, self.provenance, created_at_unix_ns=3)
        second = Path(self.temporary.name) / "order.sqlite3"
        with Storage.open(second, initialize=True) as storage:
            storage.persist(self.preregistration, self.provenance, created_at_unix_ns=1)
            v0 = session(
                CALIBRATION, "V0", ratio=1.0, preregistration=self.preregistration
            )
            with self.assertRaises(ValueError):
                storage.persist(v0, self.provenance, created_at_unix_ns=2)

    def test_complete_history_replays_and_sql_mutation_is_blocked(self) -> None:
        preregistration = self.preregistration
        calibration_sessions = [
            session(CALIBRATION, item, ratio=1.0, preregistration=preregistration)
            for item in SESSION_ORDER
        ]
        calibration = build_calibration_summary(calibration_sessions)
        seal = build_confirmation_seal(calibration, calibration_sessions)
        confirmation_sessions = [
            session(
                CONFIRMATION,
                item,
                ratio=0.8,
                preregistration=preregistration,
                seal=seal,
            )
            for item in SESSION_ORDER
        ]
        decision = build_study_decision(confirmation_sessions, seal)
        payloads = [
            preregistration,
            *calibration_sessions,
            calibration,
            seal,
            *confirmation_sessions,
            decision,
        ]
        with Storage.open(self.path, initialize=True) as storage:
            for index, payload in enumerate(payloads, start=1):
                storage.persist(payload, self.provenance, created_at_unix_ns=index)
            rows = storage.verified_records()
            self.assertEqual(len(rows), len(payloads))
            self.assertEqual(rows[-1]["status"], "n10_gain_confirmed")
            with self.assertRaises(sqlite3.DatabaseError):
                storage.connection.execute("UPDATE records SET status='x'")
            with self.assertRaises(sqlite3.DatabaseError):
                storage.connection.execute("DELETE FROM records")

    def test_tampered_schema_is_rejected(self) -> None:
        with Storage.open(self.path, initialize=True) as storage:
            storage.persist(self.preregistration, self.provenance, created_at_unix_ns=1)
        connection = sqlite3.connect(self.path)
        connection.execute("DROP TRIGGER records_no_update")
        connection.commit()
        connection.close()
        with self.assertRaises(StorageError):
            Storage.open(self.path, read_only=True)


if __name__ == "__main__":
    unittest.main()
