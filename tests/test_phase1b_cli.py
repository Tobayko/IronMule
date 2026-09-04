from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from friday_phase1b.cli import main
from friday_phase1b.constants import (
    BENCHMARK_RUN_ID,
    EXPERIMENT_ID,
    QUALIFICATION_RUN_ID,
    SCHEMA_VERSION,
)
from friday_phase1b.dashboard import DashboardService
from friday_phase1b.experiment import scope
from tests.test_phase1b_history import provenance


def qualification(passed: bool = True) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": QUALIFICATION_RUN_ID,
        "kind": "qualification",
        "status": "qualification_passed" if passed else "qualification_failed",
        "formal_claim": False,
        "action": "qualification_only" if passed else "baseline_fallback",
        "scope": scope(),
        "metrics": {"gate_passed": passed},
    }


def benchmark() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": BENCHMARK_RUN_ID,
        "kind": "benchmark",
        "status": "candidate_inconclusive",
        "formal_claim": False,
        "action": "baseline_fallback",
        "scope": scope(),
        "metrics": {"gate_passed": False},
    }


class Phase1BCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "phase1b.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_main(self, arguments: list[str], *, qualify_passed: bool = True) -> int:
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "friday_phase1b.cli.DEFAULT_DATABASE_PATH", self.database
        ), patch(
            "friday_phase1b.cli.collect_provenance", return_value=provenance()
        ), patch(
            "friday_phase1b.cli.qualification_report",
            return_value=qualification(qualify_passed),
        ), patch("friday_phase1b.cli.benchmark_report", return_value=benchmark()):
            return main(arguments)

    def test_execute_lock_and_foreign_run_id_precede_database(self) -> None:
        self.assertEqual(self.run_main(["qualify"]), 78)
        self.assertFalse(self.database.exists())
        self.assertEqual(
            self.run_main(["qualify", "--run-id", "foreign", "--execute"]), 1
        )
        self.assertFalse(self.database.exists())

    def test_once_only_qualification_then_benchmark(self) -> None:
        self.assertEqual(self.run_main(["qualify", "--execute"]), 0)
        self.assertEqual(self.run_main(["benchmark", "--execute"]), 2)
        snapshot = DashboardService(self.database).snapshot(2)
        self.assertEqual(snapshot["total"], 2)
        self.assertEqual(snapshot["by_status"]["qualification_passed"], 1)
        self.assertEqual(snapshot["by_status"]["candidate_inconclusive"], 1)
        self.assertEqual(self.run_main(["qualify", "--execute"]), 1)
        self.assertEqual(self.run_main(["benchmark", "--execute"]), 1)

    def test_failed_qualification_is_terminal_and_blocks_benchmark(self) -> None:
        self.assertEqual(self.run_main(["qualify", "--execute"], qualify_passed=False), 2)
        self.assertEqual(self.run_main(["benchmark", "--execute"]), 1)


if __name__ == "__main__":
    unittest.main()
