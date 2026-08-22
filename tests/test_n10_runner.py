"""Offline backend and release-gate tests for the N10-v1 runner."""

from __future__ import annotations

import sys
import unittest

import numpy as np

from friday_evidence.budget import BudgetGuard
from friday_n10.cli import main
from friday_n10.constants import CALIBRATION, CONFIRMATION, N_MATMULS
from friday_n10.protocol import validate_session
from friday_n10.runner import measure_prepared_session


class FakeBackend:
    def from_host(self, value):
        return np.asarray(value)

    def matmul(self, left, right):
        return np.matmul(left, right).astype(np.float16)

    def eval_many(self, _values):
        return None

    def synchronize(self):
        return None

    def to_host(self, value):
        return np.asarray(value)

    def memory_snapshot(self):
        return {
            "mlx_active_memory_bytes": 10,
            "mlx_peak_memory_bytes": 20,
            "mlx_cache_memory_bytes": 5,
        }


class StepClock:
    def __init__(self, step: int = 1_000_000) -> None:
        self.value = 1_000_000_000
        self.step = step

    def __call__(self) -> int:
        self.value += self.step
        return self.value


def binding(stage: str) -> dict[str, object]:
    return {
        "preregistration_sha256": "a" * 64,
        "confirmation_seal_sha256": None if stage == CALIBRATION else "b" * 64,
        "provenance_sha256": "c" * 64,
    }


class PreparedRunnerTest(unittest.TestCase):
    def test_fake_backend_executes_complete_balanced_aa_session(self) -> None:
        self.assertEqual(N_MATMULS, 10)
        backend = FakeBackend()
        left = np.eye(2, dtype=np.float16)
        operands = [np.eye(2, dtype=np.float16) * (index + 1) for index in range(N_MATMULS)]
        clock = StepClock()
        result = measure_prepared_session(
            CALIBRATION,
            "C0",
            binding=binding(CALIBRATION),
            backend=backend,
            left=left,
            operands=operands,
            np_module=np,
            power_source="ac_power",
            guard=BudgetGuard(),
            clock_ns=clock,
            process_clock_ns=StepClock(step=100),
        )
        validate_session(result)
        self.assertEqual(result["metrics"]["ratio"], 1.0)
        self.assertEqual(result["correctness"]["status"], "byte_identical")

    def test_fake_backend_executes_confirmation_without_changing_results(self) -> None:
        backend = FakeBackend()
        left = np.eye(2, dtype=np.float16)
        operands = [np.eye(2, dtype=np.float16) for _ in range(N_MATMULS)]
        result = measure_prepared_session(
            CONFIRMATION,
            "C0",
            binding=binding(CONFIRMATION),
            backend=backend,
            left=left,
            operands=operands,
            np_module=np,
            power_source="ac_power",
            guard=BudgetGuard(),
            clock_ns=StepClock(),
            process_clock_ns=StepClock(step=100),
        )
        self.assertEqual(result["correctness"]["max_abs_error"], 0.0)
        self.assertEqual(result["kind"], "confirmation_session")


class ReleaseGateTest(unittest.TestCase):
    def test_session_is_locked_without_execute_and_never_imports_mlx(self) -> None:
        before = set(sys.modules)
        code = main(["session", "--stage", CALIBRATION, "--id", "C0"])
        self.assertEqual(code, 78)
        self.assertNotIn("mlx.core", set(sys.modules) - before)

    def test_offline_self_check_is_available_without_execute(self) -> None:
        self.assertEqual(main(["self-check"]), 0)


if __name__ == "__main__":
    unittest.main()
