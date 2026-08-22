"""Read-only snapshot and HTML tests for the formal N10-v1 UI."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from friday_n10.dashboard import DashboardError, DashboardService, _html, _target
from friday_n10.protocol import build_preregistration
from friday_n10.storage import Storage
from tests.test_n10_storage import provenance


class DashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "n10.sqlite3"
        self.provenance = provenance()
        self.preregistration = build_preregistration(self.provenance["provenance_sha256"])
        with Storage.open(self.path, initialize=True) as storage:
            storage.persist(self.preregistration, self.provenance, created_at_unix_ns=1)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_snapshot_and_html_show_verified_history_without_writes(self) -> None:
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        snapshot = DashboardService(self.path).snapshot()
        document = _html(snapshot).decode("utf-8")
        self.assertEqual(snapshot["total"], 1)
        self.assertTrue(snapshot["read_only"])
        self.assertIn("N10-v1 formal", document)
        snapshot["recent"][0]["metrics"] = {"ratio": 0.8, "mde": 0.05}
        metrics_document = _html(snapshot).decode("utf-8")
        self.assertIn("ratio=0.8", metrics_document)
        self.assertIn("mde=0.05", metrics_document)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_request_target_is_closed(self) -> None:
        self.assertEqual(_target("/api/snapshot?limit=1"), ("/api/snapshot", {"limit": ["1"]}))
        for target in ("https://example.test/", "/path#fragment", "/api?a=1&a=2&a=3"):
            with self.subTest(target=target):
                with self.assertRaises(DashboardError):
                    _target(target)


if __name__ == "__main__":
    unittest.main()
