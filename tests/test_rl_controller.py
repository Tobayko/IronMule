"""Unit tests for the AdaptiveRLController."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from friday_serve.rl_controller import (
    ACTIONS,
    ACTION_TO_KNOBS,
    AdaptiveRLController,
    LinUCBModel,
)


class TestLinUCBModel(unittest.TestCase):
    def test_initialization(self) -> None:
        model = LinUCBModel.create(dim=6)
        self.assertEqual(model.A.shape, (6, 6))
        self.assertEqual(model.b.shape, (6, 1))
        np.testing.assert_array_equal(model.A, np.identity(6))

    def test_update_and_ucb(self) -> None:
        model = LinUCBModel.create(dim=6)
        x = np.ones((6, 1))
        ucb_before = model.ucb(x, alpha=1.0)
        model.update(x, reward=0.25)
        theta = model.theta()
        self.assertGreater(float((theta.T @ x).item()), 0.0)


class TestAdaptiveRLController(unittest.TestCase):
    def test_feature_extraction(self) -> None:
        feat = AdaptiveRLController.extract_features(
            model_id="mlx-community/gemma-3-4b-it-4bit",
            prompt_tokens=512,
            output_tokens=64,
        )
        self.assertEqual(feat.shape, (6, 1))
        # is_1b=0, is_4b=1, is_12b=0
        self.assertEqual(feat[0, 0], 0.0)
        self.assertEqual(feat[1, 0], 1.0)
        self.assertEqual(feat[2, 0], 0.0)
        self.assertAlmostEqual(feat[3, 0], 0.5)  # 512 / 1024

    def test_select_action_and_learn(self) -> None:
        ctrl = AdaptiveRLController(alpha=0.1)
        # Initially all actions have equal/neutral prior
        action, knobs, score = ctrl.select_action(
            "mlx-community/gemma-3-4b-it-4bit", 100, 32
        )
        self.assertIn(action, ACTIONS)

        # Train: full_optimized gets high reward, baseline gets 0
        for _ in range(20):
            ctrl.observe_reward("full_optimized", "mlx-community/gemma-3-4b-it-4bit", 100, 32, reward=0.15)
            ctrl.observe_reward("baseline", "mlx-community/gemma-3-4b-it-4bit", 100, 32, reward=0.0)

        # Now full_optimized must dominate
        chosen, knobs, score = ctrl.select_action(
            "mlx-community/gemma-3-4b-it-4bit", 100, 32
        )
        self.assertEqual(chosen, "full_optimized")
        self.assertTrue(knobs.get("head_skip_prefill"))
        self.assertEqual(knobs.get("readback_every"), 8)

    def test_save_and_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            path = Path(tmp.name)
            ctrl = AdaptiveRLController(alpha=0.5, save_path=path)
            ctrl.observe_reward("full_optimized", "mlx-community/gemma-3-1b-it-4bit", 200, 32, 0.30)
            ctrl.save()

            loaded = AdaptiveRLController.load(path)
            self.assertEqual(len(loaded.history), 0)  # history in separate list, weights restored
            self.assertEqual(loaded.alpha, 0.5)
            # Verify learned weights match
            np.testing.assert_allclose(loaded.models["full_optimized"].b, ctrl.models["full_optimized"].b)


if __name__ == "__main__":
    unittest.main()
