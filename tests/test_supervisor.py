import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from friday_h0 import supervisor
from friday_h0.protocol import ProtocolError, close_manifest
from friday_h0.supervisor import SupervisorLimits, run_supervised
from tests.test_protocol import valid_manifest


class SupervisorTests(unittest.TestCase):
    @staticmethod
    def _executable(path: Path) -> Path:
        path.write_bytes(b"test executable fixture")
        path.chmod(0o700)
        return path

    def test_trusted_executable_accepts_regular_file_without_rewriting_path(self):
        with tempfile.TemporaryDirectory() as raw_root:
            executable = self._executable(Path(raw_root) / "python")
            with mock.patch.object(supervisor.sys, "executable", str(executable)):
                identity = supervisor._trusted_executable()
        self.assertEqual(identity.lexical, str(executable))
        self.assertEqual(identity.resolved, str(executable.resolve()))

    def test_trusted_symlink_is_used_lexically_in_argv(self):
        manifest = close_manifest(valid_manifest("analysis_slow"))
        captured = {}

        class FakeProcess:
            pid = 12345
            returncode = 0
            stdout = None
            stderr = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return FakeProcess()

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            target = self._executable(root / "python-target")
            lexical = root / "python"
            lexical.symlink_to(target.name)
            with mock.patch.object(supervisor.sys, "executable", str(lexical)), mock.patch.object(
                supervisor.subprocess, "Popen", side_effect=fake_popen
            ), mock.patch.object(supervisor, "_rss_bytes", return_value=None):
                result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=0.2, cleanup_s=0.1))

            self.assertEqual(
                tuple(captured["argv"]),
                (str(lexical), "-P", "-s", "-B", "-m", "friday_h0.worker"),
            )
            self.assertNotEqual(str(lexical), str(target.resolve()))
            self.assertEqual(captured["kwargs"]["env"]["PATH"], "/usr/bin:/bin")
            self.assertNotIn("HOME", captured["kwargs"]["env"])
            self.assertEqual(result["status"], "invalid")

    def test_trusted_symlink_chain_may_revisit_a_symlink(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            real = root / "real"
            inner = real / "inner"
            real.mkdir()
            inner.mkdir()
            executable = self._executable(real / "python")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            alias_target = os.readlink(alias)
            (inner / "up").symlink_to("../..", target_is_directory=True)
            lexical = alias / "inner" / "up" / "alias" / "python"

            identity = supervisor._capture_executable_identity(str(lexical))

        self.assertEqual(identity.resolved, str(executable.resolve()))
        self.assertEqual(sum(link.link_target == alias_target for link in identity.links), 2)

    def test_trusted_executable_rejects_invalid_paths_and_targets(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            missing = root / "missing"
            dangling = root / "dangling"
            dangling.symlink_to(missing.name)
            cycle_a = root / "cycle-a"
            cycle_b = root / "cycle-b"
            cycle_a.symlink_to(cycle_b.name)
            cycle_b.symlink_to(cycle_a.name)
            directory = root / "directory"
            directory.mkdir()
            writable = self._executable(root / "writable")
            writable.chmod(0o722)
            not_executable = self._executable(root / "not-executable")
            not_executable.chmod(0o600)

            cases = {
                "empty": "",
                "relative": "relative/python",
                "missing": str(missing),
                "dangling": str(dangling),
                "cycle": str(cycle_a),
                "directory": str(directory),
                "writable-target": str(writable),
                "non-executable-target": str(not_executable),
            }
            for name, value in cases.items():
                with self.subTest(name=name), self.assertRaises(ProtocolError):
                    supervisor._capture_executable_identity(value)

    def test_trusted_executable_rejects_untrusted_owner(self):
        with tempfile.TemporaryDirectory() as raw_root:
            executable = self._executable(Path(raw_root) / "python")
            actual_uid = os.lstat(executable).st_uid
            self.assertNotEqual(actual_uid, 0, "test fixture requires an unprivileged owner")
            with mock.patch.object(supervisor, "_current_uid", return_value=actual_uid + 10_000):
                with self.assertRaisesRegex(ProtocolError, "owner is not trusted"):
                    supervisor._capture_executable_identity(str(executable))

    def test_identity_change_before_popen_fails_closed(self):
        manifest = close_manifest(valid_manifest("analysis_slow"))
        real_environment = supervisor._controlled_environment

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            first = self._executable(root / "python-first")
            second = self._executable(root / "python-second")
            lexical = root / "python"
            lexical.symlink_to(first.name)

            def swap_after_first_capture(*, manifest, test_limits):
                environment = real_environment(manifest=manifest, test_limits=test_limits)
                lexical.unlink()
                lexical.symlink_to(second.name)
                return environment

            with mock.patch.object(supervisor.sys, "executable", str(lexical)), mock.patch.object(
                supervisor, "_controlled_environment", side_effect=swap_after_first_capture
            ), mock.patch.object(supervisor.subprocess, "Popen") as popen:
                result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=0.2, cleanup_s=0.1))

        self.assertEqual(result["status"], "worker_exit")
        self.assertEqual(result["error"]["code"], "invalid_executable")
        popen.assert_not_called()

    def test_mlx_modes_fail_closed_without_importing_mlx(self):
        manifest = close_manifest(valid_manifest("eager_baseline"))
        fixture = Path(__file__).parent / "fixtures" / "runtime_unavailable_child.py"
        real_popen = subprocess.Popen

        def deterministic_popen(_argv, **kwargs):
            return real_popen([sys.executable, str(fixture)], **kwargs)

        with mock.patch("friday_h0.supervisor.subprocess.Popen", side_effect=deterministic_popen):
            result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=1.0, cleanup_s=0.25))
        self.assertEqual(result["classification"], "runtime_unavailable")
        self.assertEqual(result["action"], "baseline_fallback")
        self.assertIn("rss_peak_bytes", result["evidence"])

    def test_analysis_mode_is_pure_and_replayable(self):
        manifest = close_manifest(valid_manifest("analysis_known_win"))
        limits = SupervisorLimits.for_tests(total_s=1.0, cleanup_s=0.25)
        first = run_supervised(manifest, limits)
        second = run_supervised(manifest, limits)
        self.assertEqual(first["classification"], "promoted")
        self.assertEqual(first["evidence"]["decision"]["decision_hash"], second["evidence"]["decision"]["decision_hash"])

    def test_control_timeout_uses_short_test_seam(self):
        manifest = close_manifest(valid_manifest("control_timeout"))
        result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=0.10, cleanup_s=0.20, control_sleep_s=0.30))
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["action"], "baseline_fallback")

    def test_control_exit_70_is_bounded_fallback(self):
        manifest = close_manifest(valid_manifest("control_exit_70"))
        result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=1.0, cleanup_s=0.25))
        self.assertEqual(result["status"], "worker_exit")
        self.assertEqual(result["error"]["code"], "exit_70")

    def test_fixed_argv_environment_and_fd_contract(self):
        manifest = close_manifest(valid_manifest("analysis_slow"))
        captured = {}

        class FakeProcess:
            pid = 12345
            returncode = 0
            stdout = None
            stderr = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return FakeProcess()

        with mock.patch("friday_h0.supervisor.subprocess.Popen", side_effect=fake_popen), mock.patch(
            "friday_h0.supervisor._rss_bytes", return_value=None
        ):
            result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=0.2, cleanup_s=0.1))
        self.assertEqual(tuple(captured["argv"]), (sys.executable, "-P", "-s", "-B", "-m", "friday_h0.worker"))
        self.assertEqual(captured["kwargs"]["stdin"], subprocess.DEVNULL)
        self.assertEqual(captured["kwargs"]["start_new_session"], True)
        self.assertEqual(captured["kwargs"]["close_fds"], True)
        self.assertEqual(captured["kwargs"]["pass_fds"], ())
        self.assertEqual(captured["kwargs"]["shell"], False)
        environment = captured["kwargs"]["env"]
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertEqual(environment["PYTHONPATH"], str(Path(supervisor.__file__).resolve().parent.parent))
        self.assertNotIn("HOME", environment)
        self.assertNotIn("VIRTUAL_ENV", environment)
        self.assertEqual(result["status"], "invalid")

    def test_concurrent_noisy_streams_trigger_bounded_fallback(self):
        manifest = close_manifest(valid_manifest("analysis_slow"))
        fixture = Path(__file__).parent / "fixtures" / "noisy_child.py"
        real_popen = subprocess.Popen

        def noisy_popen(_argv, **kwargs):
            return real_popen([sys.executable, str(fixture)], **kwargs)

        limits = SupervisorLimits.for_tests(total_s=1.0, cleanup_s=0.25, stdout_bytes=128, stderr_bytes=128)
        with mock.patch("friday_h0.supervisor.subprocess.Popen", side_effect=noisy_popen), mock.patch(
            "friday_h0.supervisor._rss_bytes", return_value=None
        ):
            result = run_supervised(manifest, limits)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["error"]["code"], "stream_overflow")
        self.assertGreater(result["evidence"]["stdout_bytes"], 0)
        self.assertEqual(len(result["evidence"]["stdout_sha256"]), 64)
        self.assertLessEqual(len(result["evidence"]["stdout_preview"].encode()), 4096)
        self.assertTrue(result["evidence"]["stdout_truncated"])
        self.assertTrue(result["evidence"]["stdout_overflow"])

    def test_unconfirmed_termination_does_not_cleanup_cwd_or_claim_success(self):
        manifest = close_manifest(valid_manifest("control_timeout"))

        class NeverDies:
            pid = 54321
            returncode = None
            stdout = None
            stderr = None

            def poll(self):
                return None

            def wait(self, timeout=None):
                raise subprocess.TimeoutExpired("friday_h0.worker", timeout)

        with mock.patch("friday_h0.supervisor.subprocess.Popen", return_value=NeverDies()), mock.patch(
            "friday_h0.supervisor._kill_group"
        ) as kill_group, mock.patch("friday_h0.supervisor._rss_bytes", return_value=None):
            result = run_supervised(manifest, SupervisorLimits.for_tests(total_s=0.05, cleanup_s=0.05))
        self.assertEqual(result["error"]["code"], "termination_unconfirmed")
        self.assertEqual(result["status"], "invalid")
        kill_group.assert_called()


if __name__ == "__main__":
    unittest.main()
