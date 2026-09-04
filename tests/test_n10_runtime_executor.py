"""Execution-plan, metadata-scope, and circuit-breaker tests."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from friday_runtime_n10.constants import N10_DECISION_RECORD_ID
from friday_runtime_n10.executor import RuntimeController, RuntimeExecutionError
from friday_runtime_n10.policy import PolicyEvidence


@dataclass(frozen=True)
class Tensor:
    value: int
    shape: tuple[int, int] = (2048, 2048)
    dtype: str = "float16"


class Backend:
    def __init__(self, *, fail_batch: bool = False) -> None:
        self.fail_batch = fail_batch
        self.eval_sizes: list[int] = []
        self.syncs = 0

    def matmul(self, left: Tensor, right: Tensor) -> Tensor:
        return Tensor(left.value * right.value, shape=(left.shape[0], right.shape[1]))

    def eval_many(self, values) -> None:
        self.eval_sizes.append(len(values))
        if self.fail_batch and len(values) == 10:
            raise MemoryError("simulated")

    def synchronize(self) -> None:
        self.syncs += 1


def authorized() -> PolicyEvidence:
    return PolicyEvidence(
        authorized=True,
        reason="formal_n10_gain_exact_scope",
        decision_record_id=N10_DECISION_RECORD_ID,
        decision_sha256="d" * 64,
        preregistration_sha256="p" * 64,
        sealed_provenance_sha256="s" * 64,
        formal_database_sha256="f" * 64,
        formal_snapshot_revision="r" * 64,
        evidence_records=16,
    )


class RuntimeExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.left = Tensor(2)
        self.operands = tuple(Tensor(index + 1) for index in range(10))

    def test_exact_tensor_metadata_uses_one_terminal_eval_and_sync(self) -> None:
        backend = Backend()
        result = RuntimeController(authorized()).execute(backend, self.left, self.operands)
        self.assertEqual(result.decision.strategy, "batched")
        self.assertEqual(backend.eval_sizes, [10])
        self.assertEqual(backend.syncs, 1)
        self.assertEqual(
            [value.value for value in result.outputs],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        )

    def test_out_of_scope_shape_uses_serial_baseline(self) -> None:
        backend = Backend()
        operands = (*self.operands[:-1], Tensor(10, shape=(1024, 2048)))
        result = RuntimeController(authorized()).execute(backend, self.left, operands)
        self.assertEqual(result.decision.strategy, "serial")
        self.assertEqual(result.decision.reason, "workload_out_of_scope")
        self.assertEqual(backend.eval_sizes, [1] * 10)
        self.assertEqual(backend.syncs, 10)

    def test_batch_failure_is_not_retried_and_latches_future_calls(self) -> None:
        controller = RuntimeController(authorized())
        failing = Backend(fail_batch=True)
        with self.assertRaisesRegex(RuntimeExecutionError, "was not retried"):
            controller.execute(failing, self.left, self.operands)
        self.assertEqual(failing.eval_sizes, [10])
        self.assertEqual(failing.syncs, 0)
        self.assertEqual(controller.circuit_reason, "MemoryError")

        healthy = Backend()
        result = controller.execute(healthy, self.left, self.operands)
        self.assertEqual(result.decision.reason, "circuit_breaker_latched")
        self.assertEqual(healthy.eval_sizes, [1] * 10)
        self.assertEqual(healthy.syncs, 10)


if __name__ == "__main__":
    unittest.main()
