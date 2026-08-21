import contextlib
import builtins
import io
import unittest
from unittest import mock

from friday_h0 import cli


class CliTests(unittest.TestCase):
    def test_locked_mlx_validates_choices_then_returns_without_runner(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(cli, "_run_offline") as offline:
            code = cli.main([
                "mlx-run", "--mode", "eager_baseline", "--process-set", "characterization", "--process-index", "0",
            ])
        self.assertEqual(code, 78)
        self.assertIn("state=not_released", output.getvalue())
        offline.assert_not_called()

    def test_locked_mlx_stays_before_runner_import_or_activity(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(cli, "_run_mlx") as live, mock.patch(
            "builtins.__import__", wraps=builtins.__import__
        ) as importer:
            code = cli.main([
                "mlx-run", "--mode", "aa_gpu", "--process-set", "characterization", "--process-index", "0",
            ])
        self.assertEqual(code, cli.EXIT_MLX_LOCKED)
        self.assertEqual(output.getvalue(), "state=not_released command=mlx-run\n")
        live.assert_not_called()
        self.assertNotIn("friday_h0.runner", [call.args[0] for call in importer.call_args_list])

    def test_execute_mlx_uses_only_closed_registered_tuple(self):
        class Persistence:
            state = "inserted"
            bundle_sha256 = "b" * 64

        class Manifest:
            run_id = "h0-aa_gpu-confirmation-2-test"

        outcome = mock.Mock(manifest=Manifest(), persistence=Persistence(), result={
            "status": "invalid", "classification": "runtime_unavailable", "action": "baseline_fallback",
        })
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch(
            "friday_h0.runner.run_mlx", return_value=outcome
        ) as run:
            code = cli.main([
                "mlx-run", "--mode", "aa_gpu", "--process-set", "confirmation", "--process-index", "2",
                "--execute",
            ])
        self.assertEqual(code, 10)
        run.assert_called_once_with("aa_gpu", "confirmation", 2)
        self.assertEqual(output.getvalue(), (
            "state=inserted run_id=h0-aa_gpu-confirmation-2-test status=invalid "
            "classification=runtime_unavailable action=baseline_fallback bundle_sha256=" + "b" * 64 + "\n"
        ))

    def test_db_init_output_is_bounded(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch(
            "friday_h0.runner.initialize_database", return_value=None
        ):
            code = cli.main(["db-init"])
        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "state=initialized identity=friday_h0.sqlite.v1\n")

    def test_offline_output_has_only_closed_fields(self):
        class Persistence:
            state = "inserted"
            bundle_sha256 = "a" * 64

        class Manifest:
            run_id = "h0-analysis_known_win-analysis-0-test"

        outcome = mock.Mock(manifest=Manifest(), persistence=Persistence(), result={
            "status": "completed", "classification": "promoted", "action": "promoted",
        })
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch(
            "friday_h0.runner.run_offline", return_value=outcome
        ):
            code = cli.main(["offline", "--mode", "analysis_known_win"])
        self.assertEqual(code, 0)
        self.assertNotIn("/", output.getvalue())
        self.assertIn("bundle_sha256=" + "a" * 64, output.getvalue())

    def test_dashboard_port_argument_is_closed(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["dashboard", "--port", "65536"])
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertEqual(output.getvalue(), "state=usage_error code=64\n")

    def test_unknown_flags_and_modes_use_static_usage_code(self):
        for argv in (
            ["offline", "--mode", "not-a-mode"],
            ["offline", "--mode", "analysis_slow", "--secret", "/tmp/private"],
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(argv)
            self.assertEqual(code, cli.EXIT_USAGE)
            self.assertEqual(output.getvalue(), "state=usage_error code=64\n")

    def test_mlx_lock_remains_before_runner_activity(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), mock.patch.object(cli, "_run_offline") as offline:
            code = cli.main([
                "mlx-run", "--mode", "aa_gpu", "--process-set", "confirmation", "--process-index", "2",
            ])
        self.assertEqual(code, cli.EXIT_MLX_LOCKED)
        self.assertEqual(output.getvalue(), "state=not_released command=mlx-run\n")
        offline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
