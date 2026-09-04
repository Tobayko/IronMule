from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from friday_avo_router.cli import main
from friday_avo_router.dashboard import DashboardService
from tests.test_avo_router import router
from tests.test_avo_router_history import provenance


POLICY_METRICS = {
    "cold_load_ns": 1,
    "baseline_median_ns": 10.0,
    "router_median_ns": 20.0,
    "router_p95_ns": 21.0,
    "incremental_median_ns": 10.0,
    "gates": {"all": True},
    "gate_passed": True,
}
SHADOW_METRICS = {
    "cases": {},
    "gates": {"all": True},
    "gate_passed": True,
}


class RouterCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "router.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_main(self, command: list[str]) -> int:
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "friday_avo_router.cli.collect_provenance", return_value=provenance()
        ), patch("friday_avo_router.cli._load_router", return_value=router()), patch(
            "friday_avo_router.cli._resource_metrics", return_value={"rss_peak_bytes": 1}
        ), patch(
            "friday_avo_router.cli.benchmark_policy_overhead",
            return_value=dict(POLICY_METRICS),
        ), patch(
            "friday_avo_router.cli.validate_real_tensor_shadow",
            return_value=dict(SHADOW_METRICS),
        ):
            return main(["--database", str(self.database), *command])

    def test_policy_then_shadow_persist_exactly_once(self) -> None:
        self.assertEqual(self.run_main(["benchmark-policy", "--execute"]), 0)
        self.assertEqual(self.run_main(["validate-shadow", "--execute"]), 0)
        snapshot = DashboardService(self.database).snapshot(10)
        self.assertEqual(snapshot["total"], 2)
        self.assertEqual(snapshot["by_status"]["policy_overhead_passed"], 1)
        self.assertEqual(snapshot["by_status"]["shadow_router_validated"], 1)
        self.assertEqual(self.run_main(["benchmark-policy", "--execute"]), 1)
        self.assertEqual(self.run_main(["validate-shadow", "--execute"]), 1)

    def test_failed_policy_blocks_shadow(self) -> None:
        failed = dict(POLICY_METRICS)
        failed["gate_passed"] = False
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "friday_avo_router.cli.collect_provenance", return_value=provenance()
        ), patch("friday_avo_router.cli._load_router", return_value=router()), patch(
            "friday_avo_router.cli._resource_metrics", return_value={}
        ), patch(
            "friday_avo_router.cli.benchmark_policy_overhead", return_value=failed
        ):
            self.assertEqual(
                main(["--database", str(self.database), "benchmark-policy", "--execute"]),
                2,
            )
        self.assertEqual(self.run_main(["validate-shadow", "--execute"]), 1)

    def test_unexpected_measurement_error_is_persisted_terminally(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output), patch(
            "friday_avo_router.cli.collect_provenance", return_value=provenance()
        ), patch("friday_avo_router.cli._load_router", return_value=router()), patch(
            "friday_avo_router.cli._resource_metrics", return_value={}
        ), patch(
            "friday_avo_router.cli.benchmark_policy_overhead",
            side_effect=RuntimeError("synthetic measurement failure"),
        ):
            self.assertEqual(
                main(["--database", str(self.database), "benchmark-policy", "--execute"]),
                1,
            )
        snapshot = DashboardService(self.database).snapshot(10)
        self.assertEqual(snapshot["by_status"]["router_failed_terminal"], 1)
        self.assertEqual(self.run_main(["benchmark-policy", "--execute"]), 1)

    def test_unregistered_run_ids_fail_before_database_creation(self) -> None:
        self.assertEqual(
            self.run_main(
                ["benchmark-policy", "--run-id", "not-preregistered", "--execute"]
            ),
            1,
        )
        self.assertFalse(self.database.exists())


if __name__ == "__main__":
    unittest.main()
