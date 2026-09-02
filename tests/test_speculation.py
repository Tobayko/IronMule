"""The speculation bandit: it must be able to switch off, and it must stay closed.

The corpus question — whether the bandit beats a fixed width on real workloads —
is decided by experiments/speculation_bandit/replay.py, not here. These tests
cover the properties that must hold regardless of how that comes out.
"""

from __future__ import annotations

import unittest

from friday_serve.speculation import (
    ACTIONS,
    CLASS_NAMES,
    BanditError,
    Posterior,
    SpeculationBandit,
    repetition_rate,
    workload_class,
)


class ActionSpaceTest(unittest.TestCase):
    def test_the_action_space_is_the_one_the_evidence_covers(self) -> None:
        # tools/measure_prompt_lookup.py:56 sweeps DRAFT_LENGTHS = (0,1,2,3,4).
        self.assertEqual(ACTIONS, (0, 1, 2, 3, 4))

    def test_off_is_a_first_class_action(self) -> None:
        self.assertIn(0, ACTIONS)

    def test_an_action_outside_the_space_is_refused(self) -> None:
        bandit = SpeculationBandit()
        for action in (8, -1, 2.5, "two"):
            with self.assertRaises(BanditError):
                bandit.propensity("sparse", action, seed=1)


class ClassTest(unittest.TestCase):
    def test_repetition_orders_prompts_the_way_the_speedups_do(self) -> None:
        sparse = list(range(200))
        repetitive = [1, 2, 3] * 60
        self.assertLess(repetition_rate(sparse), repetition_rate(repetitive))
        self.assertEqual(workload_class(sparse), "sparse")
        self.assertEqual(workload_class(repetitive), "repetitive")

    def test_every_class_name_is_reachable(self) -> None:
        self.assertEqual(len(CLASS_NAMES), 3)
        self.assertEqual(workload_class([]), "sparse")


class PosteriorTest(unittest.TestCase):
    def test_a_neutral_reward_moves_nothing(self) -> None:
        posterior = Posterior()
        posterior.update(0.5)
        self.assertAlmostEqual(posterior.mean, 0.5)

    def test_rewards_are_clamped(self) -> None:
        posterior = Posterior()
        posterior.update(9.0)
        posterior.update(-4.0)
        self.assertAlmostEqual(posterior.alpha, 2.0)
        self.assertAlmostEqual(posterior.beta, 2.0)


class BanditTest(unittest.TestCase):
    def test_greedy_decoding_scores_exactly_neutral(self) -> None:
        bandit = SpeculationBandit()
        self.assertAlmostEqual(bandit.observe("sparse", 0, 60.0), 0.5)
        self.assertAlmostEqual(bandit.observe("sparse", 0, 60.0), 0.5)

    def test_it_learns_to_switch_speculation_off_when_it_loses(self) -> None:
        """The whole point: an unfavourable workload must end up on width 0."""

        bandit = SpeculationBandit()
        rates = {0: 60.0, 1: 55.0, 2: 52.0, 3: 50.0, 4: 48.0}
        for step in range(400):
            action, _propensity = bandit.select("sparse", seed=step)
            bandit.observe("sparse", action, rates[action])
        distribution = bandit.distribution("sparse")
        self.assertGreater(distribution[0], 0.7)
        self.assertLess(distribution[4], 0.05)

    def test_it_finds_the_winning_width_when_one_exists(self) -> None:
        bandit = SpeculationBandit()
        rates = {0: 60.0, 1: 64.0, 2: 72.0, 3: 66.0, 4: 61.0}
        for step in range(400):
            action, _propensity = bandit.select("repetitive", seed=step)
            bandit.observe("repetitive", action, rates[action])
        self.assertEqual(max(bandit.distribution("repetitive"), key=lambda k: bandit.distribution("repetitive")[k]), 2)

    def test_classes_do_not_leak_into_each_other(self) -> None:
        bandit = SpeculationBandit()
        good = {0: 60.0, 1: 62.0, 2: 75.0, 3: 70.0, 4: 65.0}
        bad = {0: 60.0, 1: 54.0, 2: 50.0, 3: 48.0, 4: 45.0}
        for step in range(400):
            for name, rates in (("repetitive", good), ("sparse", bad)):
                action, _propensity = bandit.select(name, seed=step * 7 + len(name))
                bandit.observe(name, action, rates[action])
        self.assertGreater(bandit.distribution("sparse")[0], 0.6)
        self.assertLess(bandit.distribution("repetitive")[0], 0.3)

    def test_propensity_is_never_zero(self) -> None:
        bandit = SpeculationBandit()
        for step in range(200):
            bandit.observe("sparse", 0, 60.0)
        for action in ACTIONS:
            self.assertGreater(bandit.propensity("sparse", action, seed=3, draws=64), 0.0)

    def test_propensities_sum_to_one(self) -> None:
        bandit = SpeculationBandit()
        total = sum(bandit.distribution("sparse", seed=5).values())
        self.assertAlmostEqual(total, 1.0)

    def test_selection_is_reproducible_from_its_seed(self) -> None:
        bandit = SpeculationBandit()
        self.assertEqual(bandit.select("sparse", seed=42), bandit.select("sparse", seed=42))
        for seed in (-1, "x", True):
            with self.assertRaises(BanditError):
                bandit.select("sparse", seed=seed)

    def test_a_bad_rate_is_refused_rather_than_learned(self) -> None:
        bandit = SpeculationBandit()
        for rate in (0.0, -3.0, float("nan"), float("inf"), True, "fast"):
            with self.assertRaises(BanditError):
                bandit.observe("sparse", 0, rate)

    def test_the_calibration_curve_seeds_the_prior_but_is_overruled(self) -> None:
        # The curve says width 4 is best; the requests say otherwise.
        bandit = SpeculationBandit.seeded({0: 60.0, 1: 62.0, 2: 64.0, 3: 68.0, 4: 76.0})
        self.assertGreater(bandit.distribution("sparse")[4], 0.2)
        rates = {0: 60.0, 1: 55.0, 2: 52.0, 3: 50.0, 4: 45.0}
        for step in range(400):
            action, _propensity = bandit.select("sparse", seed=step)
            bandit.observe("sparse", action, rates[action])
        self.assertGreater(bandit.distribution("sparse")[0], 0.6)

    def test_an_empty_curve_is_a_valid_start(self) -> None:
        self.assertEqual(SpeculationBandit.seeded(None).posteriors, {})
        self.assertEqual(SpeculationBandit.seeded({}).posteriors, {})
        self.assertEqual(SpeculationBandit.seeded({0: 0.0}).posteriors, {})


if __name__ == "__main__":
    unittest.main()
