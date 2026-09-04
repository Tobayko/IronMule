"""Offline backend and release-gate tests for the N10-v2 runner."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from friday_evidence.budget import BudgetGuard
from friday_n10_v2.cli import main
from friday_n10_v2.constants import CALIBRATION, CONFIRMATION, FIXTURE_A_SHA256, N_MATMULS
from friday_n10_v2.protocol import validate_session
from friday_n10_v2.runner import (
    RunnerError,
    _failure_type,
    _private_regular_sha256,
    decide_study,
    measure_prepared_session,
    seal_confirmation,
    summarize_calibration,
    validate_fixture_contract,
)


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
    def test_predecessor_hash_rejects_symlinks_and_broad_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite3"
            source.write_bytes(b"terminal-v1")
            os.chmod(source, 0o600)
            self.assertEqual(
                _private_regular_sha256(source),
                hashlib.sha256(b"terminal-v1").hexdigest(),
            )
            link = root / "link.sqlite3"
            link.symlink_to(source)
            with self.assertRaises(RunnerError):
                _private_regular_sha256(link)
            os.chmod(source, 0o644)
            with self.assertRaises(RunnerError):
                _private_regular_sha256(source)

    def test_frozen_production_fixture_has_the_registered_identity(self) -> None:
        identity = validate_fixture_contract()
        self.assertEqual(identity["a_sha256"], FIXTURE_A_SHA256)

    def test_stable_benchmark_failure_code_is_preserved_without_message(self) -> None:
        class TaggedError(RuntimeError):
            code = "correctness_contract"

        self.assertEqual(
            _failure_type(TaggedError("untrusted detail")),
            "TaggedError:correctness_contract",
        )

    def test_every_derived_record_rechecks_the_terminal_predecessor(self) -> None:
        for operation in (summarize_calibration, seal_confirmation, decide_study):
            with self.subTest(operation=operation.__name__):
                with patch(
                    "friday_n10_v2.runner.validate_predecessor_contract",
                    side_effect=RunnerError("predecessor changed"),
                ):
                    with self.assertRaisesRegex(RunnerError, "predecessor changed"):
                        operation(Path("must-not-open.sqlite3"))

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
