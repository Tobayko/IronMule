"""Offline checks for the dispatch-plan measurement statistics.

No GPU, no MLX, no timing.  These guard the decision logic itself: a wrong
estimator would turn noise into a finding, which is the exact failure mode the
A/A calibration exposed.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "measure_dispatch_plan", PROJECT_ROOT / "tools" / "measure_dispatch_plan.py"
)
plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plan)


class PairedRatioTest(unittest.TestCase):
    def test_clean_gain_is_recovered_exactly(self) -> None:
        summary = plan.paired_ratio([math.log(0.80)] * 20)
        self.assertAlmostEqual(summary["ratio"], 0.80, places=9)
        self.assertTrue(plan.clears_threshold(summary))

    def test_no_effect_does_not_clear_the_threshold(self) -> None:
        self.assertFalse(plan.clears_threshold(plan.paired_ratio([0.0] * 20)))

    def test_single_wild_block_cannot_manufacture_a_finding(self) -> None:
        # One extreme block among twenty must not flip a null into an effect.
        noisy = [0.0] * 19 + [math.log(0.2)]
        self.assertFalse(plan.clears_threshold(plan.paired_ratio(noisy)))

    def test_real_effect_survives_a_wild_block_the_other_way(self) -> None:
        real = [math.log(0.80)] * 19 + [math.log(3.0)]
        self.assertLess(plan.paired_ratio(real)["ratio"], 0.9)

    def test_a_slowdown_also_clears_the_threshold(self) -> None:
        # The gate is two-sided: a regression must be detectable too.
        self.assertTrue(plan.clears_threshold(plan.paired_ratio([math.log(1.25)] * 20)))

    def test_too_few_blocks_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            plan.paired_ratio([0.0])


class HierarchicalBootstrapTest(unittest.TestCase):
    def test_aggregate_is_deterministic_for_a_fixed_seed(self) -> None:
        data = [[math.log(0.85)] * 10, [math.log(0.86)] * 10, [math.log(0.84)] * 10]
        first = plan.hierarchical_bootstrap(data, seed=123, draws=500)
        second = plan.hierarchical_bootstrap(data, seed=123, draws=500)
        self.assertEqual(first, second)

    def test_consistent_replicates_give_a_tight_interval(self) -> None:
        data = [[math.log(0.85)] * 20 for _ in range(5)]
        result = plan.hierarchical_bootstrap(data, seed=7, draws=1000)
        self.assertAlmostEqual(result["ratio"], 0.85, places=6)
        self.assertTrue(plan.clears_threshold(result))

    def test_one_disagreeing_replicate_widens_rather_than_decides(self) -> None:
        # The point of aggregating: a single noisy replicate must not veto a
        # result, and must not be silently ignored either.
        agree = [[math.log(0.85)] * 20 for _ in range(4)]
        noisy = agree + [[math.log(1.10)] * 20]
        tight = plan.hierarchical_bootstrap(agree, seed=11, draws=1000)
        wide = plan.hierarchical_bootstrap(noisy, seed=11, draws=1000)
        self.assertGreater(
            wide["ci_high"] - wide["ci_low"], tight["ci_high"] - tight["ci_low"]
        )

    def test_aggregation_needs_at_least_two_replicates(self) -> None:
        with self.assertRaises(ValueError):
            plan.hierarchical_bootstrap([[0.0] * 10], seed=1, draws=10)


class ReleaseGateTest(unittest.TestCase):
    def test_measurement_is_locked_without_the_execute_flag(self) -> None:
        self.assertEqual(plan.main([]), 78)

    def test_self_check_passes(self) -> None:
        self.assertEqual(plan.main(["--self-check"]), 0)


if __name__ == "__main__":
    unittest.main()
