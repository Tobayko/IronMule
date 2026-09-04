from __future__ import annotations

import hashlib
import io
import json
import socket
import tempfile
import types
import unittest
from pathlib import Path

from friday_h01.analysis import analyze_trace
from friday_h01 import storage as storage_module
from friday_h01.dashboard import (
    MAX_PATH_BYTES,
    MAX_RESPONSE_BYTES,
    DashboardError,
    DashboardRequestHandler,
    DashboardService,
    _parse_detail_query,
    _parse_limit_query,
    _parse_request_target,
)
from friday_h01.protocol import build_trace
from friday_h01.storage import BundleError, Storage
from friday_h01.study import analyze_study
from tests.test_h01_analysis import BASE_NS
from tests.test_h01_protocol import make_manifest
from tests.test_h01_storage import (
    _rewrite_row_without_update_trigger,
    file_sha256,
    legacy_payload,
)
from tests.test_h01_study import make_records


def invalid_session_payload() -> dict[str, object]:
    manifest = make_manifest("C0")
    trace = build_trace(manifest, [BASE_NS] * 112)
    del trace["samples"][3]
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


def unresolved_study_payload() -> dict[str, object]:
    records = make_records(drift_session="V2")
    result = analyze_study(records)
    return {
        "entity_id": result["study_id"],
        "entity_kind": "paced_study",
        "status": result["status"],
        "created_at_unix_ns": 30,
        "manifest": {"session_records": records},
        "trace": {"session_bindings": result["session_bindings"]},
        "result": result,
        "lineage": result["shared_provenance"]["source"],
    }


def contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            contains_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(contains_key(child, forbidden) for child in value)
    return False


class H01DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.database_path = Path(cls._temporary.name) / "h01.sqlite3"
        legacy = legacy_payload()
        cls.payloads = [legacy, invalid_session_payload(), unresolved_study_payload()]
        with Storage.open(cls.database_path) as storage:
            for payload in cls.payloads:
                storage.persist_bundle(**payload)
        cls.database_hash = file_sha256(cls.database_path)
        cls.service = DashboardService(cls.database_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_snapshot_exposes_invalid_unresolved_and_legacy_history_read_only(self) -> None:
        events: list[str] = []
        real_verify_schema = storage_module._verify_schema
        real_verified_rows = storage_module._verified_rows

        def ordered_schema(connection: object) -> dict[str, object]:
            self.assertTrue(connection.in_transaction)
            events.append("schema")
            return real_verify_schema(connection)

        def ordered_rows(connection: object) -> list[dict[str, object]]:
            self.assertTrue(connection.in_transaction)
            events.append("rows")
            return real_verified_rows(connection)

        storage_module._verify_schema = ordered_schema
        storage_module._verified_rows = ordered_rows
        try:
            first = self.service.snapshot(limit=3)
        finally:
            storage_module._verify_schema = real_verify_schema
            storage_module._verified_rows = real_verified_rows
        second = self.service.snapshot(limit=3)
        self.assertEqual(events[-2:], ["schema", "rows"])
        self.assertEqual(first, second)
        self.assertEqual(first["total"], 3)
        self.assertEqual(first["by_kind"]["paced_session"], 1)
        self.assertEqual(first["by_kind"]["paced_study"], 1)
        self.assertEqual(first["by_kind"]["legacy_h0_warmup_observation"], 1)
        self.assertEqual(first["by_status"]["h01_invalid"], 1)
        self.assertEqual(first["by_status"]["h01_complete_unresolved"], 1)
        self.assertEqual(first["by_status"]["legacy_observation"], 1)
        self.assertFalse(first["cross_database_atomicity"])
        self.assertEqual(file_sha256(self.database_path), self.database_hash)

    def test_details_are_bounded_and_parent_h0_lineage_is_separate(self) -> None:
        legacy = self.service.detail(self.payloads[0]["entity_id"])
        self.assertIsNotNone(legacy)
        self.assertEqual(len(legacy["records"]), 11)
        self.assertFalse(legacy["records_truncated"])
        self.assertEqual(legacy["trace"]["warmup_count"], 11)
        self.assertEqual(
            legacy["trace"]["statistics"]["median_ns"],
            {"numerator": 1_000_005, "denominator": 1},
        )
        self.assertEqual(legacy["parent_h0_lineage"]["parent_phase"], "H0")
        session = self.service.detail(self.payloads[1]["entity_id"])
        self.assertEqual(len(session["trace_points"]), 111)
        self.assertLessEqual(len(session["trace_points"]), 200)
        study = self.service.detail(self.payloads[2]["entity_id"])
        self.assertEqual(len(study["records"]), 6)
        self.assertLessEqual(len(study["records"]), 200)
        for name, detail in (("legacy", legacy), ("session", session), ("study", study)):
            exposed = {
                key: value
                for key, value in detail.items()
                if key != "parent_h0_lineage"
            }
            with self.subTest(detail=name):
                self.assertFalse(contains_key(exposed, "source"))
                self.assertFalse(contains_key(exposed, "parent_run_id"))
        self.assertIsNone(self.service.detail("missing-valid-id"))
        self.assertEqual(file_sha256(self.database_path), self.database_hash)

    def test_snapshot_rejects_any_tampered_row_and_revision_binds_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.sqlite3"
            second_path = root / "second.sqlite3"
            first_payload = legacy_payload("legacy-revision-a")
            second_payload = legacy_payload("legacy-revision-b")
            with Storage.open(first_path) as storage:
                storage.persist_bundle(**first_payload)
            with Storage.open(second_path) as storage:
                storage.persist_bundle(**second_payload)
            first = DashboardService(first_path).snapshot(limit=1)
            second = DashboardService(second_path).snapshot(limit=1)
            self.assertEqual(first["total"], second["total"])
            self.assertEqual(
                first["recent"][0]["created_at_unix_ns"],
                second["recent"][0]["created_at_unix_ns"],
            )
            self.assertNotEqual(first["revision"], second["revision"])

            _rewrite_row_without_update_trigger(
                first_path,
                "UPDATE bundles SET trace_json = ? WHERE entity_id = ?",
                ("{}", first_payload["entity_id"]),
            )
            with self.assertRaises(BundleError):
                DashboardService(first_path).snapshot(limit=1)

    def test_request_path_and_query_contracts_are_bounded(self) -> None:
        self.assertEqual(_parse_request_target("/api/snapshot?limit=3"), ("/api/snapshot", {"limit": ["3"]}))
        self.assertEqual(_parse_limit_query({}), 50)
        self.assertEqual(_parse_limit_query({"limit": ["200"]}), 200)
        self.assertEqual(_parse_detail_query({"id": ["valid-id:1"]}), "valid-id:1")
        bad_targets = [
            "http://example.invalid/api/snapshot",
            "/api/snapshot#fragment",
            "/" + "x" * MAX_PATH_BYTES,
            "/api/snapshot?bad",
        ]
        for target in bad_targets:
            with self.subTest(target=target[:40]), self.assertRaises(DashboardError):
                _parse_request_target(target)
        bad_queries = [
            lambda: _parse_limit_query({"limit": ["0"]}),
            lambda: _parse_limit_query({"limit": ["201"]}),
            lambda: _parse_limit_query(_parse_request_target("/api/snapshot?limit=1&limit=2")[1]),
            lambda: _parse_limit_query({"other": ["1"]}),
            lambda: _parse_detail_query({"id": ["../escape"]}),
            lambda: _parse_detail_query({}),
        ]
        for index, call in enumerate(bad_queries):
            with self.subTest(query=index), self.assertRaises(DashboardError):
                call()

    def test_socketfree_get_head_and_method_rejection_emit_security_headers(self) -> None:
        socket_constructions = 0

        def blocked_socket(*_args: object, **_kwargs: object) -> socket.socket:
            nonlocal socket_constructions
            socket_constructions += 1
            raise AssertionError("real socket construction is forbidden in dashboard unit tests")

        def invoke(path: str, command: str) -> tuple[int, list[tuple[str, str]], bytes]:
            handler = object.__new__(DashboardRequestHandler)
            handler.server = types.SimpleNamespace(dashboard_service=self.service)
            handler.path = path
            handler.command = command
            handler.wfile = io.BytesIO()
            status: list[int] = []
            headers: list[tuple[str, str]] = []
            handler.send_response = lambda code, _message=None: status.append(code)
            handler.send_header = lambda name, value: headers.append((name, value))
            handler.end_headers = lambda: None
            if command == "GET":
                handler.do_GET()
            elif command == "HEAD":
                handler.do_HEAD()
            else:
                getattr(handler, f"do_{command}")()
            return status[0], headers, handler.wfile.getvalue()

        previous_socket = socket.socket
        socket.socket = blocked_socket
        try:
            get_status, get_headers, get_body = invoke("/api/snapshot?limit=2", "GET")
            head_status, head_headers, head_body = invoke("/api/snapshot?limit=2", "HEAD")
            post_status, post_headers, post_body = invoke("/api/snapshot", "POST")
            unknown_status, unknown_headers, unknown_body = invoke(
                "/api/snapshot", "PROPFIND"
            )
        finally:
            socket.socket = previous_socket
        self.assertEqual(socket_constructions, 0)
        self.assertEqual(get_status, 200)
        self.assertEqual(json.loads(get_body)["total"], 3)
        self.assertEqual(head_status, 200)
        self.assertEqual(head_body, b"")
        self.assertEqual(post_status, 405)
        self.assertIn(b"method_not_allowed", post_body)
        self.assertEqual(unknown_status, 405)
        self.assertIn(b"method_not_allowed", unknown_body)
        for headers in (get_headers, head_headers, post_headers, unknown_headers):
            names = {name for name, _value in headers}
            self.assertIn("Content-Security-Policy", names)
            self.assertIn("X-Content-Type-Options", names)
            self.assertIn("Cache-Control", names)
        self.assertEqual(file_sha256(self.database_path), self.database_hash)

    def test_serialized_response_byte_cap_fails_before_headers(self) -> None:
        handler = object.__new__(DashboardRequestHandler)
        handler.command = "GET"
        handler.wfile = io.BytesIO()
        statuses: list[int] = []
        handler.send_response = lambda code, _message=None: statuses.append(code)
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        with self.assertRaises(DashboardError):
            handler._send(
                200,
                "application/octet-stream",
                b"x" * (MAX_RESPONSE_BYTES + 1),
                head=False,
            )
        self.assertEqual(statuses, [])
        self.assertEqual(handler.wfile.getvalue(), b"")


if __name__ == "__main__":
    unittest.main()
