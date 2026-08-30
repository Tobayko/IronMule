"""Real loopback contract tests for the optimizer history dashboard."""

from __future__ import annotations

import http.client
import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from friday_optimizer import DataPhase, OptimizationMemoryV2, OptimizationRecord, QualityClass, RecordKind
from friday_optimizer.canonical import canonical_bytes
from friday_optimizer.corpus import NormalizedRecord
from friday_optimizer.dataset import DatasetBuilder
from friday_optimizer.dashboard import DashboardService, serve
from friday_optimizer.profiles import AtomicProfileStore, OptimizerProfile


class OptimizerDashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.root = Path(self.temp.name)
        self.database = self.root / "memory.sqlite3"
        with OptimizationMemoryV2(self.database) as memory:
            memory.append(
                OptimizationRecord(
                    "run-1", RecordKind.BENCHMARK, QualityClass.FORMAL, DataPhase.LABEL,
                    {
                        "status": "qualified", "prompt": "DO_NOT_RENDER", "raw_log": "DO_NOT_RENDER",
                        "fingerprint": "m1-max", "dataset_hash": "a" * 64,
                        "candidate_hash": "b" * 64, "code_sha256": "c" * 64,
                        "ttft_ms": 12.5, "decode_tps": 44.0, "ci_low": 0.8,
                        "ci_high": 0.95, "mde": 0.1, "correctness": "passed",
                        "peak_memory_mb": 512, "peak_rss_mb": 720, "swap_before_mb": 0,
                        "swap_after_mb": 0, "lease": "owned", "pid": 99, "fork": "serial",
                    },
                    created_at="2026-08-30T00:00:00Z",
                )
            )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_service_is_read_only_and_redacts_payload(self) -> None:
        before = self.database.read_bytes()
        result = DashboardService(self.database).history(1)
        after = self.database.read_bytes()
        self.assertEqual(before, after)
        encoded = json.dumps(result)
        self.assertNotIn("DO_NOT_RENDER", encoded)
        self.assertIn("dataset_hash", encoded)
        self.assertIn("decode_tps", encoded)

    def test_missing_database_fails_closed(self) -> None:
        result = DashboardService(self.root / "missing.sqlite3").status()
        self.assertEqual(result["data_state"], "unavailable")
        self.assertNotIn("Traceback", json.dumps(result))

    def test_corrupt_database_direct_methods_return_bounded_unavailable(self) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute("DROP TRIGGER optimization_records_no_update")
            connection.execute("UPDATE optimization_records SET payload=?", (b"{broken",))
        service = DashboardService(self.database)
        for value in (service.history(1), service.status()):
            self.assertEqual(value["data_state"], "unavailable")
            self.assertLess(len(json.dumps(value)), 4096)
            self.assertNotIn("broken", json.dumps(value))

    def test_memory_path_swap_is_not_served(self) -> None:
        signature = (("memory.sqlite3", 1, 2, 3, 4),)
        swapped = (("memory.sqlite3", 1, 9, 3, 4),)
        with patch("friday_optimizer.dashboard._path_signature", side_effect=[signature, swapped]):
            result = DashboardService(self.database).history(1)
        self.assertEqual(result["data_state"], "unavailable")
        self.assertEqual(result["reason"], "identity")

    def test_profile_and_dataset_tamper_fail_closed(self) -> None:
        profile_path = self.root / "profiles.json"
        AtomicProfileStore(profile_path).save(OptimizerProfile("base", "fp", "serial", {}, True), baseline=True)
        profile_value = json.loads(profile_path.read_text())
        profile_value["profiles"]["base"]["candidate"] = "tampered"
        profile_path.write_text(json.dumps(profile_value))
        profile_path.chmod(0o600)
        self.assertEqual(DashboardService(self.database, profile_path).profiles()["data_state"], "unavailable")

        dataset_path = self.root / "dataset.json"
        dataset_path.write_text("{broken")
        self.assertEqual(DashboardService(self.database, dataset_path=dataset_path).dataset()["data_state"], "unavailable")

    def test_dataset_path_swap_and_nested_redaction(self) -> None:
        dataset_path = self.root / "dataset.json"
        dataset_path.write_text(json.dumps({"schema_version": 1, "card": {"claim": "smoke_only"}}))
        signature = (("dataset.json", 1, 2, 3, 4),)
        swapped = (("dataset.json", 1, 9, 3, 4),)
        with patch("friday_optimizer.dashboard._path_signature", side_effect=[signature, swapped]):
            result = DashboardService(self.database, dataset_path=dataset_path).dataset()
        self.assertEqual(result["data_state"], "unavailable")
        self.assertEqual(result["reason"], "identity")

        with OptimizationMemoryV2(self.database) as memory:
            memory.append(
                OptimizationRecord(
                    "run-2", RecordKind.SYSTEM, QualityClass.ENGINEERING, DataPhase.FEATURE,
                    {"nested": {"input": "PRIVATE_INPUT", "source_path": "/private/source", "reason": "secret prompt"}},
                    created_at="2026-08-30T00:00:01Z",
                )
            )
        encoded = json.dumps(DashboardService(self.database).history(10))
        for forbidden in ("PRIVATE_INPUT", "/private/source", "secret prompt"):
            self.assertNotIn(forbidden, encoded)

    def test_dataset_requires_schema_and_reconstructed_hash(self) -> None:
        dataset_path = self.root / "dataset.json"
        dataset_path.write_text(json.dumps({"schema_version": 1, "card": {"claim": "smoke_only"}}))
        missing = DashboardService(self.database, dataset_path=dataset_path).dataset()
        self.assertEqual(missing["data_state"], "unavailable")
        self.assertTrue(
            set(("schema_version", "dataset_hash", "card", "record_count", "splits", "leakage", "claim", "data_state"))
            <= set(missing)
        )
        self.assertIsNone(missing["dataset_hash"])
        self.assertIsNone(missing["record_count"])
        self.assertIsNone(missing["splits"])
        self.assertIsNone(missing["leakage"])
        self.assertIsNone(missing["claim"])
        dataset_path.write_text(json.dumps({"schema_version": 2, "sha256": "a" * 64, "records": [], "splits": {}, "card": {}}))
        invalid_schema = DashboardService(self.database, dataset_path=dataset_path).dataset()
        self.assertEqual(invalid_schema["data_state"], "unavailable")
        embedded = {"schema_version": 1, "records": [], "splits": {"train": [], "validation": [], "holdout": []}, "card": {}}
        embedded["sha256"] = hashlib.sha256(canonical_bytes(embedded)).hexdigest()
        dataset_path.write_bytes(canonical_bytes(embedded))
        self.assertEqual(DashboardService(self.database, dataset_path=dataset_path).dataset()["data_state"], "unavailable")

    def test_materialized_dataset_snapshot_has_stable_success_schema(self) -> None:
        record = NormalizedRecord(
            record_id="dataset-record",
            source_path="evidence.json",
            source_kind="json",
            quality=QualityClass.ENGINEERING,
            data={"model": "gemma-1b", "workload": "matmul", "timing": {"p50": 1.0}},
            feature_fields=("model", "workload"),
            label_fields=("timing.p50",),
            source_sha256="0" * 64,
            study_id="study-1",
            run_id="run-1",
            observed_time="2026-08-30T00:00:00Z",
            hardware_fingerprint="m1-max",
            model_fingerprint="gemma-1b-q4",
            workload_fingerprint="matmul",
            prompt_family="default",
            dirty=False,
            manifest_verified=True,
            contract_verified=True,
            identity_contract_valid=True,
            contract_id="fixture.v1",
            contract_version=1,
            contract_hash="f" * 64,
            logical_source_file="evidence.json",
        )
        snapshot = DatasetBuilder([record]).build()
        dataset_path = self.root / "materialized-dataset.json"
        materialized = snapshot.as_dict()
        materialized.pop("sha256", None)
        dataset_path.write_bytes(canonical_bytes(materialized))
        value = DashboardService(self.database, dataset_path=dataset_path).dataset()
        self.assertEqual(value["data_state"], "available")
        self.assertEqual(value["dataset_hash"], hashlib.sha256(dataset_path.read_bytes()).hexdigest())
        self.assertEqual(value["record_count"], 1)
        self.assertEqual(set(value["splits"]), {"train", "validation", "holdout"})
        self.assertIsInstance(value["card"], dict)
        self.assertIsInstance(value["leakage"], dict)
        self.assertEqual(value["claim"], "no_learning_claim")

    def test_real_materialized_dataset_artifact_is_available(self) -> None:
        artifact = Path(".friday-data/optimizer-dataset-v1.json")
        if not artifact.is_file():
            self.skipTest("materialized optimizer dataset is not present")
        service = DashboardService(self.database, dataset_path=artifact, expected_dataset_hash="79ce63bfc3b786b2e26975e367ccd905e25b692568122c2d847e8925511f5c8d")
        value = service.dataset()
        self.assertEqual(value["data_state"], "available")
        self.assertEqual(value["dataset_hash"], "79ce63bfc3b786b2e26975e367ccd905e25b692568122c2d847e8925511f5c8d")
        self.assertEqual(value["record_count"], 392)
        self.assertEqual(value["splits"], {"holdout": 0, "train": 2, "validation": 0})
        self.assertEqual(value["claim"], "no_learning_claim")
        raw = artifact.read_bytes()
        for suffix, changed in (("-whitespace", raw + b"\n"), ("-tampered", raw.replace(b"no_learning_claim", b"no_learning_claIm", 1))):
            candidate = self.root / (artifact.name + suffix)
            candidate.write_bytes(changed)
            rejected = DashboardService(self.database, dataset_path=candidate, expected_dataset_hash=value["dataset_hash"]).dataset()
            self.assertEqual(rejected["data_state"], "unavailable")
            self.assertIsNone(rejected["dataset_hash"])

    def test_loopback_routes_headers_and_methods(self) -> None:
        artifact = Path(".friday-data/optimizer-dataset-v1.json")
        try:
            server = serve(self.database, 0, dataset_path=artifact if artifact.is_file() else None)
        except PermissionError as exc:  # restricted developer sandboxes may forbid bind(2)
            self.skipTest(str(exc))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method: str, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request(method, path, headers=headers or {})
                response = connection.getresponse()
                body = response.read()
                values = {key.lower(): value for key, value in response.getheaders()}
                connection.close()
                return response.status, values, body

            for path in ("/", "/assets/app.css", "/assets/app.js", "/api/status", "/api/history?limit=1", "/api/dataset", "/api/profiles", "/api/shadow"):
                status, headers, body = request("GET", path)
                self.assertIn(status, (200, 503))
                self.assertEqual(headers["cache-control"], "no-store")
                self.assertEqual(headers["x-content-type-options"], "nosniff")
                self.assertEqual(headers["x-frame-options"], "DENY")
                self.assertEqual(headers["referrer-policy"], "no-referrer")
                self.assertLessEqual(len(body), 512 * 1024)
                if path == "/api/dataset" and artifact.is_file():
                    payload = json.loads(body)
                    self.assertEqual(payload["data_state"], "available")
                    self.assertEqual(payload["dataset_hash"], "79ce63bfc3b786b2e26975e367ccd905e25b692568122c2d847e8925511f5c8d")
            status, headers, body = request("HEAD", "/api/history?limit=1")
            self.assertEqual(status, 200)
            self.assertEqual(body, b"")
            self.assertGreater(int(headers["content-length"]), 0)
            status, headers, _ = request("POST", "/api/status")
            self.assertEqual(status, 405)
            self.assertEqual(headers["allow"], "GET, HEAD")
            status, _, _ = request("GET", "/no-such-route")
            self.assertEqual(status, 404)
            status, _, _ = request("GET", "/api/status", {"Host": "evil.example"})
            self.assertEqual(status, 421)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
