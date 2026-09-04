from __future__ import annotations

import unittest
from dataclasses import dataclass

from friday_avo_router.benchmark import benchmark_policy_overhead
from friday_avo_router.constants import ENFORCED_PLAN, SERIAL_PLAN
from friday_avo_router.router import ShadowRouter
from friday_runtime.executor import RuntimeController as N8Controller
from friday_runtime.policy import PolicyEvidence as N8Evidence
from friday_runtime_n10.executor import RuntimeController as N10Controller
from friday_runtime_n10.policy import PolicyEvidence as N10Evidence


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, int]
    dtype: str = "float16"


def router(*, n8_authorized: bool = True, n10_authorized: bool = True) -> ShadowRouter:
    n8 = N8Controller(
        N8Evidence(
            authorized=n8_authorized,
            reason="test_n8",
            decision_record_id="1" * 64,
            decision_sha256="2" * 64,
            preregistration_sha256="3" * 64,
            sealed_provenance_sha256="4" * 64,
            evidence_records=16,
        )
    )
    n10 = N10Controller(
        N10Evidence(
            authorized=n10_authorized,
            reason="test_n10",
            decision_record_id="5" * 64,
            decision_sha256="6" * 64,
            preregistration_sha256="7" * 64,
            sealed_provenance_sha256="8" * 64,
            formal_database_sha256="9" * 64,
            formal_snapshot_revision="a" * 64,
            evidence_records=16,
        )
    )
    return ShadowRouter(n8, n10)


class ShadowRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = Tensor((2048, 2048))
        self.rhs = Tensor((2048, 2048))

    def test_exact_n8_and_n10_are_recommended_but_never_executed(self) -> None:
        controller = router()
        for count, route in ((8, "n8"), (10, "n10")):
            decision = controller.decide(self.left, tuple(self.rhs for _ in range(count)))
            self.assertEqual(decision.route, route)
            self.assertEqual(decision.recommendation_strategy, "batched")
            self.assertEqual(decision.enforced_plan, ENFORCED_PLAN)
        self.assertFalse(hasattr(controller, "execute"))

    def test_negative_scopes_fall_back(self) -> None:
        controller = router()
        count = controller.decide(self.left, tuple(self.rhs for _ in range(9)))
        shape = controller.decide(Tensor((1024, 2048)), tuple(self.rhs for _ in range(8)))
        dtype = controller.decide(
            Tensor((2048, 2048), "float32"),
            tuple(Tensor((2048, 2048), "float32") for _ in range(10)),
        )
        self.assertEqual((count.route, count.recommendation_plan), ("serial", SERIAL_PLAN))
        self.assertEqual(shape.route, "serial")
        self.assertEqual(dtype.route, "serial")
        self.assertEqual(shape.recommendation_strategy, "serial")
        self.assertEqual(dtype.recommendation_strategy, "serial")
        self.assertTrue(all(value.enforced_plan == ENFORCED_PLAN for value in (count, shape, dtype)))

    def test_one_missing_evidence_closes_both_routes(self) -> None:
        for controller in (router(n8_authorized=False), router(n10_authorized=False)):
            for count in (8, 10):
                decision = controller.decide(self.left, tuple(self.rhs for _ in range(count)))
                self.assertEqual(decision.route, "serial")
                self.assertEqual(decision.reason, "router_evidence_incomplete")

    def test_policy_benchmark_is_balanced_and_records_raw_blocks(self) -> None:
        result = benchmark_policy_overhead(
            router(), cold_load_ns=1, warmup_blocks=1, blocks=3, iterations=50
        )
        self.assertEqual(result["blocks"], 3)
        self.assertEqual(len(result["baseline_block_ns"]), 3)
        self.assertEqual(len(result["router_block_ns"]), 3)
        self.assertTrue(result["gates"]["decision_agreement"])

    def test_cold_load_gate_fails_without_changing_other_measurements(self) -> None:
        result = benchmark_policy_overhead(
            router(),
            cold_load_ns=20_000_000_000,
            warmup_blocks=1,
            blocks=1,
            iterations=10,
        )
        self.assertFalse(result["gates"]["cold_load"])
        self.assertFalse(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
