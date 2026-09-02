from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from friday_optimizer.readiness import (
    HardwareLease,
    LeaseBusy,
    LeaseError,
    ProbeSnapshot,
    ReadinessPolicy,
    check_readiness,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def snapshot(**changes):
    values = dict(
        timestamp=0.0,
        ac_connected=True,
        low_power=False,
        swap_used_bytes=10,
        swap_total_bytes=1000,
        memory_available_bytes=8000,
        memory_total_bytes=16000,
        load_1m=0.1,
        cpu_percent=2.0,
        workload_active=False,
        process_tree_readable=True,
    )
    values.update(changes)
    return ProbeSnapshot(**values)


class FakeProbe:
    def __init__(self, samples):
        self.samples = iter(samples)

    def sample(self):
        return next(self.samples)


class ReadinessTests(unittest.TestCase):
    def test_requires_ac_and_low_power_off(self):
        clock = FakeClock()
        policy = ReadinessPolicy(sample_interval_seconds=1)
        result = check_readiness(
            FakeProbe([snapshot(ac_connected=False), snapshot(), snapshot()]),
            policy, sleeper=clock.sleep, clock=clock,
        )
        self.assertFalse(result.ready)
        self.assertIn("ac_not_confirmed", result.reasons)

    def test_unknown_and_unstable_samples_fail_closed(self):
        clock = FakeClock()
        result = check_readiness(
            FakeProbe([snapshot(load_1m=None), snapshot(load_1m=.1), snapshot(load_1m=.1)]),
            sleeper=clock.sleep, clock=clock,
        )
        self.assertFalse(result.ready)
        self.assertIn("load_unknown", result.reasons)
        self.assertIn("samples_unstable", result.reasons)

    def test_zero_memory_is_below_absolute_reserve(self):
        clock = FakeClock()
        result = check_readiness(
            FakeProbe([snapshot(memory_available_bytes=0)] * 3),
            sleeper=clock.sleep, clock=clock,
        )
        self.assertFalse(result.ready)
        self.assertIn("memory_reserve_too_low", result.reasons)

    def test_active_workload_and_high_load_are_blocked(self):
        clock = FakeClock()
        result = check_readiness(
            FakeProbe([snapshot(workload_active=True, load_1m=2.0)] * 3),
            sleeper=clock.sleep, clock=clock,
        )
        self.assertFalse(result.ready)
        self.assertIn("foreign_workload_or_unknown", result.reasons)
        self.assertIn("load_too_high", result.reasons)

    def test_snapshot_rejects_nan_inf_and_bad_ranges(self):
        for changes in (
            {"load_1m": float("nan")},
            {"cpu_percent": float("inf")},
            {"memory_available_bytes": -1},
            {"memory_available_bytes": 20000, "memory_total_bytes": 10000},
        ):
            with self.assertRaises((ValueError, TypeError)):
                snapshot(**changes)


class MacParserTests(unittest.TestCase):
    def _probe(self, outputs, *, max_output_bytes=64 * 1024):
        def runner(argv, **_):
            return SimpleNamespace(stdout=outputs.get(tuple(argv), ""), stderr="", returncode=0)
        return __import__("friday_optimizer.readiness", fromlist=["MacSystemProbe"]).MacSystemProbe(
            runner=runner, max_output_bytes=max_output_bytes
        )

    def test_default_command_output_bound_is_128_kib(self):
        from friday_optimizer.readiness import MacSystemProbe
        self.assertEqual(MacSystemProbe()._max_output, 128 * 1024)

    def test_pmset_sections_match_the_actual_power_source(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        p = self._probe({
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "Battery Power:\n lowpowermode 1\nAC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        })
        value = p.sample()
        self.assertEqual((value.ac_connected, value.low_power), (True, False))
        self.assertEqual(value.memory_available_bytes, 110 * 4096)

    def test_missing_low_power_key_is_explicitly_unsupported_off(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n sleep 0\nBattery Power:\n sleep 1\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        value = self._probe(outputs).sample()
        self.assertFalse(value.low_power)
        self.assertNotIn("power_unreadable", value.errors)

    def test_low_power_key_in_inactive_profile_requires_active_value(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n sleep 0\nBattery Power:\n lowpowermode 1\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        value = self._probe(outputs).sample()
        self.assertIsNone(value.low_power)
        self.assertIn("low_power_missing", value.errors)

    def test_active_low_power_one_is_not_treated_as_off(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 1\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        value = self._probe(outputs).sample()
        self.assertTrue(value.low_power)
        self.assertNotIn("low_power_unknown", value.errors)

    def test_invalid_active_low_power_value_is_unknown(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 2\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        value = self._probe(outputs).sample()
        self.assertIsNone(value.low_power)
        self.assertIn("low_power_unknown", value.errors)

    def test_stderr_is_not_measurement_and_truncation_is_unknown(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        def runner(argv, **_):
            key = tuple(argv)
            if key == (c["pmset"], "-g", "batt"):
                return SimpleNamespace(stdout="", stderr="Now drawing from 'AC Power'", returncode=0)
            return SimpleNamespace(stdout="x" * 1025, stderr="", returncode=0)
        value = MacSystemProbe(runner=runner, max_output_bytes=1024).sample()
        self.assertIsNone(value.ac_connected)
        self.assertIn("power_source_ambiguous", value.errors)
        self.assertIn("output_truncated", value.errors)

    def test_missing_page_size_and_unknown_swap_unit_block(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "Pages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00Z used = 0.00Z free = 1.00Z\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        value = self._probe(outputs).sample()
        self.assertIn("page_size_unknown", value.errors)
        self.assertIn("swap_unit_unknown", value.errors)

    def test_duplicate_low_power_values_are_ambiguous(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n lowpowermode 0\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        self.assertIn("low_power_ambiguous", self._probe(outputs).sample().errors)

    def test_missing_self_pid_is_unknown(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "0 1 0 S 0.0 launchd\n",
        }
        with patch("friday_optimizer.readiness.os.getpid", return_value=99999):
            value = self._probe(outputs).sample()
        self.assertIn("self_pid_unknown", value.errors)

    def test_own_and_validated_ancestors_are_not_foreign_workload(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "501 100 1 R 99.0 python\n0 1 0 R 99.0 node\n",
        }
        with patch("friday_optimizer.readiness.os.getpid", return_value=100):
            value = self._probe(outputs).sample()
        self.assertFalse(value.workload_active)
        self.assertEqual(value.cpu_percent, 198.0)

    def test_active_unknown_current_user_process_is_foreign_without_argv(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        current_uid = os.getuid()
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): (
                f"{current_uid} 100 1 S 0.0 python\n"
                f"{current_uid} 200 1 R 0.0 custom_gpu_worker\n"
            ),
        }
        with patch("friday_optimizer.readiness.os.getpid", return_value=100):
            value = self._probe(outputs).sample()
        self.assertTrue(value.workload_active)
        self.assertEqual(value.process_evidence, ("custom_gpu_worker",))
        self.assertNotIn("args=", value.process_evidence)

    def test_active_unknown_system_process_does_not_block(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        system_uid = os.getuid() + 1
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): (
                f"{system_uid} 200 1 R 99.0 custom_daemon\n"
            ),
        }
        with patch("friday_optimizer.readiness.os.getpid", return_value=100):
            value = self._probe(outputs).sample()
        self.assertFalse(value.workload_active)

    def test_known_runtime_name_blocks_regardless_of_uid(self):
        from friday_optimizer.readiness import MacSystemProbe
        c = MacSystemProbe.COMMANDS
        outputs = {
            (c["pmset"], "-g", "batt"): "Now drawing from 'AC Power'\n",
            (c["pmset"], "-g", "custom"): "AC Power:\n lowpowermode 0\n",
            (c["vm_stat"],): "(page size of 4096 bytes)\nPages free: 100.\nPages speculative: 10.\n",
            (c["sysctl"], "-n", "hw.memsize"): "104857600\n",
            (c["sysctl"], "vm.swapusage"): "total = 1.00G used = 0.00G free = 1.00G\n",
            (c["ps"], "-axo", "uid=,pid=,ppid=,state=,%cpu=,comm="): "999 200 1 R 0.0 mlx\n",
        }
        with patch("friday_optimizer.readiness.os.getpid", return_value=100):
            value = self._probe(outputs).sample()
        self.assertTrue(value.workload_active)
        self.assertEqual(value.process_evidence, ("mlx",))


class LeaseTests(unittest.TestCase):
    def test_lease_is_exclusive_and_0600(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lease.json"
            first = HardwareLease(path, fingerprint="fp").acquire()
            self.assertEqual(stat_mode(path), 0o600)
            with self.assertRaises(LeaseBusy):
                HardwareLease(path, fingerprint="fp").acquire()
            self.assertTrue(first.validate())
            self.assertTrue(first.heartbeat())
            first.release()
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text())["released"], True)
            second = HardwareLease(path, fingerprint="fp").acquire()
            self.assertTrue(second.validate())
            second.release()

    def test_symlink_and_nonregular_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("x")
            link = root / "lease"
            link.symlink_to(target)
            with self.assertRaises(LeaseError):
                HardwareLease(link, fingerprint="fp")
            with self.assertRaises(LeaseError):
                HardwareLease(root, fingerprint="fp")


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()


def test_a_comma_decimal_machine_can_still_be_certified():
    """macOS prints numbers in the user's locale; the gate must still read them.

    On a German system `sysctl vm.swapusage` emits "used = 1675,38M" and
    `ps -o %cpu` emits "0,5". The parser accepted a dot only, so swap and the
    process tree came back unreadable and the gate could never certify the
    machine - failing closed, but failing always, on every run this project
    ever wanted to make.
    """

    from friday_optimizer.readiness import ReadinessError, _locale_decimal, _parse_swap_value

    german = "total = 3072,00M  used = 1675,38M  free = 1396,62M  (encrypted)"
    english = "total = 3072.00M  used = 1675.38M  free = 1396.62M  (encrypted)"
    for text in (german, english):
        assert _parse_swap_value(text, "used") == int(1675.38 * 1024 * 1024)
        assert _parse_swap_value(text, "total") == int(3072.00 * 1024 * 1024)

    assert _locale_decimal("0,5") == 0.5
    assert _locale_decimal("10.6") == 10.6
    assert _locale_decimal(" 361,1 ") == 361.1


def test_an_ambiguous_number_is_refused_rather_than_guessed():
    """A thousands separator must not be read as a decimal mark."""

    from friday_optimizer.readiness import ReadinessError, _locale_decimal

    for text in ("1.234,56", "1,234.56", "1,2,3", "1.2.3", "", "   ", "abc", "nan"):
        try:
            value = _locale_decimal(text)
        except ReadinessError:
            continue
        raise AssertionError(f"{text!r} was accepted as {value}")
