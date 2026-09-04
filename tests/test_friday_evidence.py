"""Offline contract tests for persistent H1/H2 evidence and hardware budgets."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from friday_evidence.budget import BudgetError, BudgetGuard
from friday_evidence.canonical import CanonicalError, canonical_json, canonical_sha256
from friday_evidence.cli import main as evidence_main
from friday_evidence.dashboard import (
    DashboardError,
    DashboardHandler,
    DashboardService,
    _html,
    _target,
)
from friday_evidence.legacy import import_legacy_summaries
from friday_evidence.provenance import ProvenanceError, collect_provenance
from friday_evidence.registry import REGISTERED_TOOLS
from friday_evidence.run import run_persisted
from friday_evidence.storage import EvidenceStorage, StorageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def provenance(tool: str = "dispatch") -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "tool": tool,
        "workload_key": REGISTERED_TOOLS[tool],
        "provenance_kind": "native",
        "git_revision": "1" * 40,
        "git_dirty": False,
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "code_files": {"tools/x.py": "3" * 64},
        "spec_files": {"docs/x.md": "4" * 64},
        "environment": {"python": "3.12"},
        "hardware": {"model": "test"},
    }
    value["code_sha256"] = canonical_sha256(value["code_files"])
    value["spec_sha256"] = canonical_sha256(value["spec_files"])
    value["environment_sha256"] = canonical_sha256(value["environment"])
    value["hardware_key"] = canonical_sha256(value["hardware"])
    value["provenance_sha256"] = canonical_sha256(value)
    return value


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class CanonicalContractTest(unittest.TestCase):
    def test_keys_are_sorted_and_nonfinite_values_are_rejected(self) -> None:
        self.assertEqual(canonical_json({"z": 1, "a": 2}), '{"a":2,"z":1}')
        with self.assertRaises(CanonicalError):
            canonical_json({"value": float("nan")})


class RootProvenanceContractTest(unittest.TestCase):
    def test_current_root_has_a_complete_self_consistent_identity(self) -> None:
        value = collect_provenance("dispatch", require_clean=False)
        digest = value.pop("provenance_sha256")
        self.assertEqual(digest, canonical_sha256(value))
        self.assertGreaterEqual(len(value["git_revision"]), 40)
        self.assertIn("tools/measure_dispatch_plan.py", value["code_files"])
        self.assertIn("docs/PHASE1_MATMUL_SPEC.md", value["spec_files"])

    def test_dirty_root_is_rejected_before_file_or_environment_collection(self) -> None:
        def git_result(*args: str) -> bytes:
            if args[:2] == ("rev-parse", "HEAD"):
                return b"1" * 40 + b"\n"
            if args and args[0] == "status":
                return b" M tools/measure_dispatch_plan.py\n"
            raise AssertionError(args)

        with mock.patch("friday_evidence.provenance._run_git", side_effect=git_result):
            with self.assertRaises(ProvenanceError):
                collect_provenance("dispatch", require_clean=True)


class BudgetGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.time = FakeTime()
        self.guard = BudgetGuard(clock=self.time.clock, sleeper=self.time.sleep)

    def test_continuous_limit_is_cumulative_until_a_real_break(self) -> None:
        self.time.now = 3.0
        self.guard.record_gpu(3.0)
        self.time.now = 6.0
        self.guard.record_gpu(3.0)
        self.time.now = 6.01
        with self.assertRaises(BudgetError):
            self.guard.record_gpu(0.01)

    def test_required_break_resets_continuous_load(self) -> None:
        self.time.now = 6.0
        self.guard.record_gpu(6.0)
        self.guard.required_break()
        self.time.now += 6.0
        self.guard.record_gpu(6.0)
        self.assertEqual(self.guard.summary()["max_continuous_gpu_seconds"], 6.0)

    def test_break_fails_closed_if_sleep_does_not_really_elapse(self) -> None:
        guard = BudgetGuard(clock=self.time.clock, sleeper=lambda _seconds: None)
        with self.assertRaises(BudgetError):
            guard.required_break()

    def test_candidate_cooldown_waits_the_missing_time(self) -> None:
        self.guard.before_candidate()
        self.guard.finish_candidate()
        self.time.now += 10.0
        self.guard.before_candidate()
        self.assertEqual(self.guard.summary()["cooldown_seconds"], 50.0)

    def test_rolling_duty_cycle_fails_closed(self) -> None:
        self.time.now = 6.0
        self.guard.record_gpu(6.0)
        self.guard.required_break()
        self.time.now += 6.0
        self.guard.record_gpu(6.0)
        self.guard.required_break()
        self.time.now += 3.1
        with self.assertRaises(BudgetError):
            self.guard.record_gpu(3.1)

    def test_wall_limit_is_enforced(self) -> None:
        self.time.now = 1200.001
        with self.assertRaises(BudgetError):
            self.guard.check_wall()


class StorageContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite3"

    def persist(self, storage: EvidenceStorage, report: dict | None = None):
        return storage.persist(
            evidence_kind="native",
            source_key="native:test",
            tool="dispatch",
            report=report or {
                "verdict": "not_beyond_threshold",
                "raw": [1, 2],
                "formal_claim": False,
            },
            provenance=provenance(),
            result_status="not_beyond_threshold",
            raw_measurements_available=True,
            observed_at_unix_ns=1,
            recorded_at_unix_ns=2,
        )

    def test_initialization_persistence_and_read_only_verification(self) -> None:
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            outcome = self.persist(storage)
            self.assertEqual(outcome.state, "inserted")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        with EvidenceStorage.open(self.path, read_only=True) as storage:
            rows = storage.verified_rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["record_id"], outcome.record_id)
            self.assertEqual(
                storage._connection.getconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE), True
            )
            self.assertEqual(storage._connection.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                storage._connection.execute("CREATE TABLE forbidden(value INTEGER)")
            with self.assertRaises(StorageError):
                self.persist(storage)

    def test_existing_database_permissions_must_remain_private(self) -> None:
        with EvidenceStorage.open(self.path, initialize=True):
            pass
        os.chmod(self.path, 0o644)
        with self.assertRaises(StorageError):
            EvidenceStorage.open(self.path, read_only=True)

    def test_native_provenance_projections_are_recomputed(self) -> None:
        changed = provenance()
        changed["code_sha256"] = "0" * 64
        changed.pop("provenance_sha256")
        changed["provenance_sha256"] = canonical_sha256(changed)
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            with self.assertRaises(StorageError):
                storage.persist(
                    evidence_kind="native",
                    source_key="native:bad-projection",
                    tool="dispatch",
                    report={"verdict": "invalid", "formal_claim": False},
                    provenance=changed,
                    result_status="invalid",
                    raw_measurements_available=True,
                    observed_at_unix_ns=1,
                    recorded_at_unix_ns=2,
                )

    def test_identical_replay_is_idempotent(self) -> None:
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            first = self.persist(storage)
            second = self.persist(storage)
        self.assertEqual(second.state, "already_present")
        self.assertEqual(first.record_id, second.record_id)

    def test_source_key_conflict_is_rejected(self) -> None:
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            self.persist(storage)
            with self.assertRaises(StorageError):
                self.persist(storage, {"verdict": "different", "formal_claim": False})
            with self.assertRaises(StorageError):
                storage.persist(
                    evidence_kind="native",
                    source_key="native:test",
                    tool="dispatch",
                    report={
                        "verdict": "not_beyond_threshold",
                        "raw": [1, 2],
                        "formal_claim": False,
                    },
                    provenance=provenance(),
                    result_status="different_status",
                    raw_measurements_available=True,
                    observed_at_unix_ns=1,
                    recorded_at_unix_ns=3,
                )

    def test_completed_native_evidence_cannot_omit_raw_measurements(self) -> None:
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            with self.assertRaises(StorageError):
                storage.persist(
                    evidence_kind="native",
                    source_key="native:no-raw",
                    tool="dispatch",
                    report={"verdict": "not_beyond_threshold", "formal_claim": False},
                    provenance=provenance(),
                    result_status="not_beyond_threshold",
                    raw_measurements_available=False,
                    observed_at_unix_ns=1,
                    recorded_at_unix_ns=2,
                )

    def test_sql_updates_and_deletes_are_blocked(self) -> None:
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            self.persist(storage)
            with self.assertRaises(sqlite3.IntegrityError):
                storage._connection.execute(
                    "UPDATE evidence_records SET result_status='changed'"
                )
            storage._connection.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                storage._connection.execute("DELETE FROM evidence_records")

    def test_unknown_schema_is_rejected(self) -> None:
        self.path.write_bytes(b"not sqlite")
        with self.assertRaises(StorageError):
            EvidenceStorage.open(self.path, read_only=True)


class LegacyAndDashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite3"
        with EvidenceStorage.open(self.path, initialize=True) as storage:
            self.outcomes = import_legacy_summaries(
                storage, PROJECT_ROOT / "experiments" / "legacy_h1h2_summaries_v1.json"
            )

    def test_all_legacy_rows_are_explicitly_downgraded(self) -> None:
        self.assertEqual(len(self.outcomes), 10)
        with EvidenceStorage.open(self.path, read_only=True) as storage:
            for row in storage.verified_rows():
                self.assertEqual(row["evidence_kind"], "legacy_summary")
                self.assertFalse(row["raw_measurements_available"])
                self.assertEqual(row["report"]["evidence_grade"], "legacy_summary")

    def test_import_is_idempotent(self) -> None:
        with EvidenceStorage.open(self.path) as storage:
            outcomes = import_legacy_summaries(
                storage, PROJECT_ROOT / "experiments" / "legacy_h1h2_summaries_v1.json"
            )
        self.assertTrue(all(item.state == "already_present" for item in outcomes))

    def test_snapshot_and_html_show_history_without_writes(self) -> None:
        before = self.path.read_bytes()
        snapshot = DashboardService(self.path).snapshot(limit=5)
        rendered = _html(snapshot).decode("utf-8")
        self.assertEqual(snapshot["total"], 10)
        self.assertIn("legacy summaries", rendered)
        self.assertEqual(before, self.path.read_bytes())

    def test_dashboard_request_target_is_closed(self) -> None:
        self.assertEqual(_target("/api/snapshot?limit=5")[0], "/api/snapshot")
        self.assertEqual(_target("/api/snapshot?unknown=")[1], {"unknown": [""]})
        for invalid in ("https://example.com/", "/x#fragment"):
            with self.subTest(target=invalid):
                with self.assertRaises(DashboardError):
                    _target(invalid)

    def test_dashboard_explicitly_rejects_mutating_and_preflight_methods(self) -> None:
        for method in ("do_POST", "do_PUT", "do_PATCH", "do_DELETE", "do_OPTIONS"):
            with self.subTest(method=method):
                self.assertIn(method, DashboardHandler.__dict__)

    def test_loopback_http_boundary_is_read_only_and_closed(self) -> None:
        before = self.path.read_bytes()
        server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        server.dashboard_service = DashboardService(self.path)  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=2.0
        )
        try:
            connection.request("GET", "/api/snapshot?limit=1")
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertTrue(payload["read_only"])
            self.assertEqual(response.getheader("Cache-Control"), "no-store")

            connection.request("POST", "/api/snapshot")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 405)

            connection.request("GET", "/api/snapshot?unknown=")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 400)
        finally:
            connection.close()
            server.shutdown()
            thread.join(timeout=2.0)
        self.assertEqual(before, self.path.read_bytes())


class EvidenceCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite3"

    def invoke(self, *arguments: str) -> tuple[int, dict]:
        with redirect_stdout(io.StringIO()) as output:
            code = evidence_main(["--database", str(self.path), *arguments])
        return code, json.loads(output.getvalue())

    def test_write_commands_require_apply_and_reads_are_explicit(self) -> None:
        code, report = self.invoke("init")
        self.assertEqual((code, report["state"]), (78, "not_applied"))
        self.assertFalse(self.path.exists())

        code, report = self.invoke("import-legacy", "--apply")
        self.assertEqual((code, report["inserted"], report["total"]), (0, 10, 10))
        self.assertEqual(self.invoke("verify")[1]["records"], 10)
        snapshot = self.invoke("snapshot", "--limit", "3")[1]
        self.assertEqual((snapshot["total"], len(snapshot["recent"])), (10, 3))


class PersistedLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "evidence.sqlite3"
        self.provenance = provenance()

    def test_report_is_persisted_before_returning(self) -> None:
        with mock.patch("friday_evidence.run.collect_provenance", return_value=self.provenance):
            report = run_persisted(
                "dispatch",
                lambda: {
                    "verdict": "effect_confirmed",
                    "replicates": [{"serial_ns": [1], "batched_ns": [1]}],
                },
                database_path=self.path,
            )
        self.assertIn("record_id", report["evidence"])
        with EvidenceStorage.open(self.path, read_only=True) as storage:
            rows = storage.verified_rows()
        self.assertEqual(rows[0]["result_status"], "effect_confirmed")
        self.assertNotIn("evidence", rows[0]["report"])

    def test_failed_live_attempt_is_also_recorded(self) -> None:
        def fail() -> dict:
            raise RuntimeError("internal detail must not be persisted")

        with mock.patch("friday_evidence.run.collect_provenance", return_value=self.provenance):
            with self.assertRaises(RuntimeError):
                run_persisted("dispatch", fail, database_path=self.path)
        with EvidenceStorage.open(self.path, read_only=True) as storage:
            row = storage.verified_rows()[0]
        self.assertEqual(row["result_status"], "measurement_failed")
        self.assertEqual(row["report"]["failure_type"], "RuntimeError")
        self.assertNotIn("internal detail", json.dumps(row["report"]))

    def test_invalid_operation_result_is_recorded_as_failure(self) -> None:
        with mock.patch("friday_evidence.run.collect_provenance", return_value=self.provenance):
            with self.assertRaises(TypeError):
                run_persisted("dispatch", lambda: [], database_path=self.path)  # type: ignore[arg-type]
        with EvidenceStorage.open(self.path, read_only=True) as storage:
            row = storage.verified_rows()[0]
        self.assertEqual(row["result_status"], "measurement_failed")
        self.assertEqual(row["report"]["failure_type"], "TypeError")

    def test_missing_registered_raw_payload_is_recorded_as_failure(self) -> None:
        with mock.patch("friday_evidence.run.collect_provenance", return_value=self.provenance):
            with self.assertRaises(ValueError):
                run_persisted(
                    "dispatch",
                    lambda: {"verdict": "effect_confirmed"},
                    database_path=self.path,
                )
        with EvidenceStorage.open(self.path, read_only=True) as storage:
            row = storage.verified_rows()[0]
        self.assertEqual(row["result_status"], "measurement_failed")
        self.assertEqual(row["report"]["failure_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
