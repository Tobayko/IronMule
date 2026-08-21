"""Read-only local H0 dashboard audit (loopback only, no production data)."""

from __future__ import annotations

import http.client
import json
import math
import re
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

from friday_h0.dashboard import CSP, DashboardService, DashboardServer, _bounded_payload, _number, _run_record, _status, serve
from friday_h0.storage import Storage
from tests.test_manifest import valid_manifest


class DashboardAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "h0.sqlite3"
        with Storage.open(self.path):
            pass
        self.server = serve(self.path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def request(self, method: str, target: str, body: bytes | None = None, headers: dict[str, str] | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=2)
        connection.request(method, target, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        response_headers = dict(response.getheaders())
        connection.close()
        return response.status, response_headers, data

    def add_run(self, run_id: str, created: int, *, ratio: float | None = None) -> None:
        manifest = valid_manifest()
        manifest["run_id"] = run_id
        with Storage.open(self.path) as storage:
            storage.create_run(manifest, created_at_unix_ns=created)
            storage.append_status_event(run_id, "decision", "promoted", {"guardrails": {"correctness": "PASS", "memory": "PASS"}})
            if ratio is not None:
                storage.append_scalar_metric(run_id, "primary_ratio", ratio, "ratio")
                storage.append_scalar_metric(run_id, "bootstrap_ci_low", ratio - .01, "ratio")
                storage.append_scalar_metric(run_id, "bootstrap_ci_high", ratio + .01, "ratio")

    def test_bind_routes_methods_and_traversal(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for target in ("/does-not-exist", "/../friday_h0/storage.py", "/api/snapshot?x=1"):
            status, _, _ = self.request("GET", target)
            self.assertEqual(status, 404)
        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
            status, headers, _ = self.request(method, "/")
            self.assertEqual(status, 405)
            self.assertEqual(headers.get("Allow"), "GET, HEAD")

    def test_security_headers_and_assets_are_local_dom_only(self):
        status, headers, html = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Security-Policy"], CSP)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertNotIn(b"http://", html); self.assertNotIn(b"https://", html)
        _, _, js = self.request("GET", "/assets/app.js")
        self.assertNotIn(b"innerHTML", js); self.assertNotIn(b"document.write", js); self.assertNotIn(b"eval(", js)
        self.assertNotIn(b"http://", js); self.assertNotIn(b"https://", js)

    def test_empty_source_has_no_mock_kpis(self):
        status, _, body = self.request("GET", "/api/snapshot")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["data_state"], "empty")
        self.assertEqual(payload["runs"], [])
        self.assertNotIn('"primary_ratio":0', body.decode())
        self.assertIn("Noch keine Messdaten", self.request("GET", "/")[2].decode())

    def test_one_and_multiple_runs_snapshot_reconciliation(self):
        self.add_run("run-one", 10, ratio=.94)
        status, _, body = self.request("GET", "/api/snapshot")
        payload = json.loads(body)
        self.assertEqual(status, 200); self.assertEqual(payload["run_count"], 1)
        self.assertEqual(payload["runs"][0]["run_id"]["value"], "run-one")
        self.add_run("run-two", 20, ratio=.91)
        payload = json.loads(self.request("GET", "/api/snapshot")[2])
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual([r["run_id"]["value"] for r in payload["runs"]], ["run-two", "run-one"])
        self.assertEqual(payload["freshness_state"], "snapshot")

    def test_snapshot_revision_tracks_append_only_evidence_without_changing_run_time(self):
        self.add_run("revision", 10)
        before = json.loads(self.request("GET", "/api/snapshot")[2])
        with Storage.open(self.path) as storage:
            storage.append_status_event("revision", "evidence", "completed", {"note": "new"})
            storage.append_raw_sample("revision", "s", 0, "baseline", 1.0, "ns", sample_index=0)
            storage.append_scalar_metric("revision", "extra", 1.0, "ns")
            storage.append_artifact("revision", "extra", "json", "d" * 64, {"role": "evidence"})
        after = json.loads(self.request("GET", "/api/snapshot")[2])
        self.assertNotEqual(before["source_revision"], after["source_revision"])
        self.assertNotEqual(before["snapshot_id"], after["snapshot_id"])
        self.assertEqual(before["source_last_updated_at"], after["source_last_updated_at"])
        self.assertEqual(
            after["freshness_basis"],
            "latest_run_created_at; evidence revision tracked without timestamps",
        )

    def test_deterministic_order_and_latest_100_cap(self):
        for index in range(105):
            self.add_run(f"r-{index:03d}", 1 if index < 2 else index)
        payload = json.loads(self.request("GET", "/api/snapshot")[2])
        self.assertEqual(payload["run_count"], 105); self.assertEqual(len(payload["runs"]), 100)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["runs"][0]["run_id"]["value"], "r-104")
        self.assertEqual(payload["runs"][99]["run_id"]["value"], "r-005")

    def test_missing_values_and_detail_raw_sample_cap(self):
        self.add_run("detail", 1)
        with Storage.open(self.path) as storage:
            for i in range(202):
                storage.append_raw_sample("detail", "s", i, "baseline", float(i + 1), "ns", sample_index=i)
        status, _, body = self.request("GET", "/api/run?id=detail")
        payload = json.loads(body)
        self.assertEqual(status, 200); self.assertEqual(len(payload["raw_samples"]), 200)
        self.assertEqual(payload["raw_sample_count"], 202); self.assertTrue(payload["raw_samples_truncated"])
        self.assertEqual(payload["model"]["missing_reason"], "not_applicable_h0")
        status, _, _ = self.request("GET", "/api/run?id=missing")
        self.assertEqual(status, 404)

    def test_status_supports_legacy_and_common_result_payloads_fail_closed(self):
        self.assertEqual(_status([{"status": "promoted", "payload": {"status": "completed"}}]), "completed")
        self.assertEqual(
            _status([{"status": "promoted", "payload": {"result": {"status": "timeout"}}}]),
            "timeout",
        )
        self.assertEqual(
            _status([{"status": "promoted", "payload": {"result": {}}}]),
            "not_recorded",
        )

    def test_nonfinite_and_non_numeric_values_are_bounded_missing_values(self):
        self.assertEqual(_number(float("nan")), {"value": None, "missing_reason": "non_finite_source_value"})
        self.assertEqual(_number(float("inf")), {"value": None, "missing_reason": "non_finite_source_value"})
        self.assertEqual(_number(True), {"value": None, "missing_reason": "invalid_source_value"})
        self.assertEqual(_number("1.5"), {"value": None, "missing_reason": "invalid_source_value"})
        payload = _bounded_payload({"value": float("nan")}, 1024)
        self.assertEqual(payload["data_state"], "error")
        self.assertEqual(payload["error"], "response_serialization_error")
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_nonfinite_metric_projection_is_missing(self):
        manifest = valid_manifest()
        manifest["run_id"] = "nonfinite"
        row = {
            "run_id": "nonfinite",
            "created_at_unix_ns": 1,
            "phase": manifest["phase"],
            "mode": manifest["mode"],
            "manifest_json": json.dumps(manifest),
            "manifest_hash": "a" * 64,
            "code_sha256": "b" * 64,
            "spec_sha256": "c" * 64,
            "environment_sha256": "d" * 64,
            "revision": None,
            "revision_missing_reason": "test_fixture",
        }
        metrics = {"primary_ratio": {"value": float("inf"), "missing_reason": None}}
        payload = _run_record(row, [], [], metrics, [], detail=False)
        ratio = payload["primary_ratio"]
        self.assertEqual(ratio, {"value": None, "missing_reason": "non_finite_source_value"})

    def test_open_closes_storage_once_when_setup_fails(self):
        for failure_point in ("pragma", "progress", "begin"):
            with self.subTest(failure_point=failure_point):
                storage = mock.Mock()
                connection = mock.Mock()
                storage.connection = connection
                if failure_point == "pragma":
                    connection.execute.side_effect = RuntimeError("pragma failure")
                elif failure_point == "progress":
                    connection.set_progress_handler.side_effect = RuntimeError("progress failure")
                else:
                    connection.execute.side_effect = [None, RuntimeError("begin failure")]
                with mock.patch("friday_h0.dashboard.Storage.open", return_value=storage):
                    with self.assertRaises(RuntimeError):
                        DashboardService(self.path)._open(0.1)
                storage.close.assert_called_once_with()

    def test_run_id_sql_xss_and_query_bounds(self):
        self.add_run("safe", 1)
        for target in ("/api/run?id=safe%27%20OR%201%3D1", "/api/run?id=../../x", "/api/run?id=safe&id=other", "/api/run?sort=run_id"):
            self.assertEqual(self.request("GET", target)[0], 400)
        self.assertEqual(self.request("GET", "/api/run?id=%3Cscript%3E")[0], 400)
        with Storage.open(self.path) as storage:
            storage.append_status_event("safe", "x", "<script>alert(1)</script>", {"note": "<img src=x onerror=1>"})
        body = self.request("GET", "/api/snapshot")[2]
        self.assertIn(b"<script>", body)  # JSON is data; the browser renders it only through textContent.

    def test_limits_and_head_match_get(self):
        status, get_headers, get_body = self.request("GET", "/assets/app.css")
        head_status, head_headers, head_body = self.request("HEAD", "/assets/app.css")
        self.assertEqual((status, get_headers.get("Content-Length")), (head_status, head_headers.get("Content-Length")))
        self.assertEqual(len(head_body), 0); self.assertEqual(int(head_headers["Content-Length"]), len(get_body))
        self.assertEqual(self.request("GET", "/" + "a" * 2100)[0], 414)
        status, _, _ = self.request("GET", "/api/snapshot", headers={"X-Large": "x" * 17000})
        self.assertIn(status, (400, 431))

    def test_read_only_mode_and_no_file_parameter(self):
        before = self.path.stat().st_size
        status, _, _ = self.request("GET", "/api/snapshot?db=/tmp/other.sqlite")
        self.assertEqual(status, 404)
        self.assertEqual(self.path.stat().st_size, before)
        with Storage.open(self.path, read_only=True) as storage:
            self.assertEqual(storage.connection.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                storage.connection.execute("CREATE TABLE forbidden(x)")

    def test_corrupt_wrong_schema_and_response_text_are_safe(self):
        wrong_path = Path(self.tmp.name) / "wrong.sqlite3"
        connection = sqlite3.connect(wrong_path); connection.execute("PRAGMA user_version=99"); connection.commit(); connection.close()
        wrong_server = serve(wrong_path); wrong_thread = threading.Thread(target=wrong_server.serve_forever, daemon=True); wrong_thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", wrong_server.server_address[1], timeout=2); connection.request("GET", "/api/snapshot"); response = connection.getresponse(); body = response.read(); connection.close()
            self.assertEqual(response.status, 503); self.assertIn(b"invalid_source", body)
        finally:
            wrong_server.shutdown(); wrong_server.server_close(); wrong_thread.join(timeout=2)
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.path.write_bytes(b"not sqlite")
        server = serve(self.path); thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2); connection.request("GET", "/api/snapshot"); response = connection.getresponse(); body = response.read(); connection.close()
            self.assertEqual(response.status, 503); self.assertIn(b"invalid_source", body); self.assertNotIn(str(self.path).encode(), body); self.assertNotIn(b"SELECT", body)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_lock_state_is_bounded_and_does_not_leak_sql(self):
        blocker = sqlite3.connect(self.path, timeout=0)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            status, _, body = self.request("GET", "/api/snapshot")
            self.assertEqual(status, 503)
            self.assertIn(b"source_busy", body)
            self.assertNotIn(b"database is locked", body)
            self.assertNotIn(b"SELECT", body)
        finally:
            blocker.rollback(); blocker.close()


if __name__ == "__main__":
    unittest.main()
