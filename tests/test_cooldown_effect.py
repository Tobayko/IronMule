"""Offline checks for the cooldown-effect measurement logic.

No GPU, no MLX, no timing.  These pin down the metric that two earlier attempts
got wrong: a cutoff rule needs a steady state that holds a fixed band, and this
device does not have one.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "measure_cooldown_effect", PROJECT_ROOT / "tools" / "measure_cooldown_effect.py"
)
cooldown = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cooldown)


class NormalizeTest(unittest.TestCase):
    def test_flat_burst_normalizes_to_one(self) -> None:
        self.assertEqual(cooldown.normalize([100] * 12), [1.0] * 12)

    def test_ratio_is_against_the_own_steady_state(self) -> None:
        burst = [300] + [100] * 11
        self.assertAlmostEqual(cooldown.normalize(burst)[0], 3.0)

    def test_short_burst_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            cooldown.normalize([1, 2, 3])

    def test_nonpositive_steady_state_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            cooldown.normalize([1] * 6 + [0] * 6)


class ExcessSamplesTest(unittest.TestCase):
    def test_settled_burst_costs_nothing(self) -> None:
        self.assertEqual(cooldown.excess_samples([1.0] * 12), 0.0)
        self.assertEqual(cooldown.excess_samples([1.04] * 12), 0.0)

    def test_ramp_is_charged_its_full_overshoot(self) -> None:
        self.assertEqual(cooldown.excess_samples([3.0, 1.5] + [1.0] * 10), 2.5)

    def test_samples_below_steady_state_cost_nothing(self) -> None:
        # Jitter downward is not a cooldown; the measure is one-sided.
        self.assertEqual(cooldown.excess_samples([0.90, 0.93] + [0.96] * 10), 0.0)

    def test_a_later_spike_is_not_charged_to_the_cooldown(self) -> None:
        # Only the leading run counts, otherwise ordinary outliers would inflate
        # every pause length equally and destroy the dose-response reading.
        self.assertEqual(cooldown.excess_samples([1.0] * 10 + [1.5, 1.0]), 0.0)

    def test_longer_ramp_costs_strictly_more(self) -> None:
        long_ramp = cooldown.excess_samples([4.0, 2.0, 1.4] + [1.0] * 9)
        short_ramp = cooldown.excess_samples([2.0, 1.2] + [1.0] * 10)
        self.assertGreater(long_ramp, short_ramp)

    def test_measure_does_not_need_a_clean_steady_state_to_exist(self) -> None:
        # The regression this replaces: a jittery tail must not be reported as
        # contamination when the leading samples are already settled.
        jittery = [1.0, 1.05] + [1.3, 0.8, 1.2, 0.9] * 2 + [1.0, 1.0]
        self.assertEqual(cooldown.excess_samples(jittery), 0.0)


class ShuffledPlanTest(unittest.TestCase):
    def test_plan_is_deterministic(self) -> None:
        self.assertEqual(
            cooldown.shuffled_plan((0.0, 1.0, 5.0), 4),
            cooldown.shuffled_plan((0.0, 1.0, 5.0), 4),
        )

    def test_plan_covers_every_pause_exactly_reps_times(self) -> None:
        plan = cooldown.shuffled_plan((0.0, 1.0, 5.0), 4)
        self.assertEqual(
            sorted(plan), sorted([(p, i) for p in (0.0, 1.0, 5.0) for i in range(4)])
        )

    def test_plan_actually_interleaves(self) -> None:
        # Grouped pause lengths would let drift over the run masquerade as a
        # pause-length effect.
        plan = cooldown.shuffled_plan((0.0, 1.0, 5.0), 4)
        self.assertNotEqual(plan, sorted(plan))


class ReleaseGateTest(unittest.TestCase):
    def test_measurement_is_locked_without_the_execute_flag(self) -> None:
        self.assertEqual(cooldown.main([]), 78)

    def test_self_check_passes(self) -> None:
        self.assertEqual(cooldown.main(["--self-check"]), 0)


if __name__ == "__main__":
    unittest.main()
