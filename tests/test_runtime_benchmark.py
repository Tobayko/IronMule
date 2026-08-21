"""Offline measurements for policy overhead and prepared runtime validation."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass

import numpy as np

from friday_runtime.benchmark import benchmark_policy_overhead, validate_prepared_runtime
from friday_runtime.cli import main
from friday_runtime.constants import H1_DECISION_RECORD_ID
from friday_runtime.executor import RuntimeController
from friday_runtime.policy import PolicyEvidence


def authorized() -> PolicyEvidence:
    return PolicyEvidence(
        authorized=True,
        reason="formal_h1_gain_exact_scope",
        decision_record_id=H1_DECISION_RECORD_ID,
        decision_sha256="d" * 64,
        preregistration_sha256="p" * 64,
        sealed_provenance_sha256="s" * 64,
        evidence_records=16,
    )


class StepClock:
    def __init__(self, step: int = 1_000_000) -> None:
        self.value = 1_000_000_000
        self.step = step

    def __call__(self) -> int:
        self.value += self.step
        return self.value


@dataclass(frozen=True)
class Tensor:
    data: np.ndarray
    shape: tuple[int, int] = (2048, 2048)
    dtype: str = "float16"


class Backend:
    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return Tensor(np.matmul(left.data, right.data).astype(np.float16))

    def eval_many(self, _values) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def to_host(self, value: Tensor) -> np.ndarray:
        return value.data

    def memory_snapshot(self):
        return {
            "mlx_active_memory_bytes": 10,
            "mlx_peak_memory_bytes": 20,
            "mlx_cache_memory_bytes": 5,
        }


class RuntimeBenchmarkTest(unittest.TestCase):
    def test_policy_benchmark_is_balanced_and_bounded(self) -> None:
        result = benchmark_policy_overhead(
            RuntimeController(authorized()),
            warmup_blocks=2,
            measurement_blocks=3,
            iterations=10,
            clock_ns=StepClock(step=10_000),
        )
        self.assertEqual([block["order"] for block in result["blocks"]], ["ab", "ba", "ab"])
        self.assertTrue(result["metrics"]["gate_passed"])
        self.assertEqual(result["policy"]["strategy"], "batched")

    def test_prepared_runtime_is_byte_identical_and_records_raw_pairs(self) -> None:
        left = Tensor(np.eye(2, dtype=np.float16))
        operands = tuple(Tensor(np.eye(2, dtype=np.float16) * (index + 1)) for index in range(8))
        result = validate_prepared_runtime(
            RuntimeController(authorized()),
            backend=Backend(),
            left=left,
            operands=operands,
            np_module=np,
            power_source="ac_power",
            warmup_pairs=2,
            measurement_blocks=4,
            clock_ns=StepClock(),
            process_clock_ns=StepClock(step=100),
        )
        self.assertTrue(result["correctness"]["byte_identical"])
        self.assertEqual(result["correctness"]["max_abs_error"], 0.0)
        self.assertEqual([block["order"] for block in result["blocks"]], ["ab", "ba", "ab", "ba"])
        self.assertEqual(result["metrics"]["ratio"], 1.0)
        self.assertFalse(result["metrics"]["gate_passed"])

    def test_cli_requires_release_before_importing_mlx_runner(self) -> None:
        before = set(sys.modules)
        code = main(["benchmark-policy"])
        self.assertEqual(code, 78)
        self.assertNotIn("friday_h1.runner", set(sys.modules) - before)
        code = main(["validate-gpu"])
        self.assertEqual(code, 78)
        self.assertNotIn("friday_h1.runner", set(sys.modules) - before)


if __name__ == "__main__":
    unittest.main()
