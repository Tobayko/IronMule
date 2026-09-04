"""Offline checks for the self-optimization loop's decision logic.

No GPU, no MLX, no timing.  The loop chooses on its own, so its ranking and
refinement rules are the part that must not be wrong: a loop that promotes an
unconfirmed candidate would manufacture findings automatically.
"""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "optimization_loop", PROJECT_ROOT / "tools" / "optimization_loop.py"
)
loop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loop)


class NeighbourProposalTest(unittest.TestCase):
    def test_proposes_the_four_adjacent_sizes(self) -> None:
        self.assertEqual(loop.neighbours(4, set()), [2, 3, 5, 6])

    def test_never_repeats_an_already_measured_size(self) -> None:
        self.assertEqual(loop.neighbours(4, {2, 3}), [5, 6])

    def test_stays_inside_the_operand_pool(self) -> None:
        self.assertEqual(loop.neighbours(2, set()), [3, 4])
        self.assertEqual(loop.neighbours(16, set()), [14, 15])

    def test_exhausted_neighbourhood_yields_nothing(self) -> None:
        self.assertEqual(loop.neighbours(4, {2, 3, 5, 6}), [])


class RankingTest(unittest.TestCase):
    def test_best_reliable_candidate_leads(self) -> None:
        ordered = loop.rank(
            [
                {"name": "weak", "ratio": 0.93, "ci_high": 0.95, "clears_mde": True},
                {"name": "strong", "ratio": 0.80, "ci_high": 0.85, "clears_mde": True},
            ]
        )
        self.assertEqual(ordered[0]["name"], "strong")

    def test_a_candidate_below_threshold_never_leads(self) -> None:
        # The whole point: a big-looking ratio that did not clear the threshold
        # must not be promoted over a smaller confirmed one.
        ordered = loop.rank(
            [
                {"name": "unconfirmed", "ratio": 0.70, "ci_high": 1.10, "clears_mde": False},
                {"name": "confirmed", "ratio": 0.90, "ci_high": 0.93, "clears_mde": True},
            ]
        )
        self.assertEqual(ordered[0]["name"], "confirmed")

    def test_lucky_outlier_loses_to_a_reliable_candidate(self) -> None:
        # The regression this ranking exists to prevent.  Ranking by the point
        # estimate picked 0.750 and 0.741 in two live runs; both regressed to
        # 0.87-0.96 on independent re-measurement and failed confirmation.
        ordered = loop.rank(
            [
                {"name": "lucky", "ratio": 0.75, "ci_high": 0.94, "clears_mde": True},
                {"name": "reliable", "ratio": 0.84, "ci_high": 0.87, "clears_mde": True},
            ]
        )
        self.assertEqual(ordered[0]["name"], "reliable")

    def test_all_failing_candidates_still_rank_without_crashing(self) -> None:
        ordered = loop.rank(
            [
                {"name": "a", "ratio": 1.02, "ci_high": 1.06, "clears_mde": False},
                {"name": "b", "ratio": 0.98, "ci_high": 1.01, "clears_mde": False},
            ]
        )
        self.assertEqual([entry["name"] for entry in ordered], ["b", "a"])


class ThresholdTest(unittest.TestCase):
    def test_clean_gain_clears_the_frozen_threshold(self) -> None:
        self.assertTrue(loop.summarize([math.log(0.80)] * 20)["clears_mde"])

    def test_no_effect_does_not_clear(self) -> None:
        self.assertFalse(loop.summarize([0.0] * 20)["clears_mde"])

    def test_effect_just_under_the_threshold_does_not_clear(self) -> None:
        # 3% is real but below the 5% the loop was told to require.
        self.assertFalse(loop.summarize([math.log(0.97)] * 20)["clears_mde"])

    def test_regression_is_also_detected(self) -> None:
        self.assertTrue(loop.summarize([math.log(1.25)] * 20)["clears_mde"])


class ReleaseGateTest(unittest.TestCase):
    def test_loop_is_locked_without_the_execute_flag(self) -> None:
        self.assertEqual(loop.main([]), 78)

    def test_self_check_passes(self) -> None:
        self.assertEqual(loop.main(["--self-check"]), 0)

    def test_statistics_are_shared_with_the_measurement_tool(self) -> None:
        # A second copy of the estimator would be a second chance to get it wrong.
        import importlib.util as util

        spec = util.spec_from_file_location(
            "mdp", PROJECT_ROOT / "tools" / "measure_dispatch_plan.py"
        )
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sample = [math.log(0.85)] * 15
        self.assertEqual(
            loop.paired_ratio(sample)["ratio"], module.paired_ratio(sample)["ratio"]
        )


if __name__ == "__main__":
    unittest.main()
