"""Append-only history, read-only UI, and release-gate tests."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from friday_evidence.canonical import canonical_sha256
from friday_head_skip_runtime.cli import main
from friday_head_skip_runtime.constants import (
    FORMAL_DECISION_SHA256,
    GPU_RUN_ID,
    POLICY_RUN_ID,
    QUALIFICATION_ID,
    RUNTIME_ID,
    SCHEMA_VERSION,
)
from friday_head_skip_runtime.dashboard import (
    DashboardError,
    DashboardService,
    _html,
    _target,
    serve,
)
from friday_head_skip_runtime.history import History, HistoryConflict, HistoryError


def provenance() -> dict[str, object]:
    code_files = {"friday_head_skip_runtime/example.py": "1" * 64}
    spec_files = {"docs/example.md": "2" * 64}
    environment = {"python": "test"}
    hardware = {"machine": "test"}
    body: dict[str, object] = {
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


def report(*, kind: str = "policy_overhead") -> dict[str, object]:
    run_id, status = {
        "policy_overhead": (POLICY_RUN_ID, "policy_overhead_passed"),
        "runtime_validation_attempt": (GPU_RUN_ID, "runtime_validation_started"),
        "runtime_validation": (GPU_RUN_ID, "runtime_validation_passed"),
    }[kind]
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": RUNTIME_ID,
        "qualification_id": QUALIFICATION_ID,
        "formal_claim": False,
        "formal_decision_sha256": FORMAL_DECISION_SHA256,
        "kind": kind,
        "run_id": run_id,
        "status": status,
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
            first = history.persist(report(), self.provenance, created_at_unix_ns=1)
            second = history.persist(
                report(kind="runtime_validation_attempt"),
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
        self.assertIn("Friday bounded head-skip runtime", document)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_update_delete_read_only_duplicate_and_symlink_are_blocked(self) -> None:
        with History.open(self.path, initialize=True) as history:
            history.persist(report(), self.provenance, created_at_unix_ns=1)
            with self.assertRaises(HistoryConflict):
                history.persist(report(), self.provenance, created_at_unix_ns=2)
            with self.assertRaises(sqlite3.DatabaseError):
                history.connection.execute("UPDATE records SET status='changed'")
            with self.assertRaises(sqlite3.DatabaseError):
                history.connection.execute("DELETE FROM records")
        with History.open(self.path, read_only=True) as history:
            with self.assertRaises((sqlite3.DatabaseError, HistoryError)):
                history.connection.execute("INSERT INTO records(record_id) VALUES('x')")
        link = self.path.with_name("link.sqlite3")
        os.symlink(self.path, link)
        with self.assertRaises(HistoryError):
            History.open(link, read_only=True)

    def test_frozen_run_and_qualification_ids_cannot_be_changed(self) -> None:
        for field, value in (
            ("run_id", "second-attempt"),
            ("qualification_id", "other-qualification"),
            ("status", "runtime_validation_passed"),
        ):
            with self.subTest(field=field):
                changed = report()
                changed[field] = value
                with History.open(self.path, initialize=True) as history:
                    with self.assertRaises(HistoryError):
                        history.persist(changed, self.provenance)

    def test_request_targets_and_keyboard_interrupt_are_bounded(self) -> None:
        with History.open(self.path, initialize=True):
            pass
        self.assertEqual(
            _target("/api/snapshot?limit=1"),
            ("/api/snapshot", {"limit": ["1"]}),
        )
        for target in ("https://example.test/", "/x#fragment", "/ümlaut"):
            with self.subTest(target=target), self.assertRaises(DashboardError):
                _target(target)

        class Server:
            daemon_threads = True

            def __init__(self, *_args, **_kwargs) -> None:
                self.closed = False

            def serve_forever(self, *, poll_interval: float) -> None:
                self.poll_interval = poll_interval
                raise KeyboardInterrupt

            def server_close(self) -> None:
                self.closed = True

        server = Server()
        with patch(
            "friday_head_skip_runtime.dashboard.ThreadingHTTPServer", return_value=server
        ):
            serve(self.path, port=8775)
        self.assertTrue(server.closed)

    def test_cli_measurements_require_explicit_release(self) -> None:
        self.assertEqual(main(["benchmark-policy"]), 78)
        self.assertEqual(main(["validate-gpu"]), 78)
        self.assertEqual(main(["unknown"]), 64)


if __name__ == "__main__":
    unittest.main()
