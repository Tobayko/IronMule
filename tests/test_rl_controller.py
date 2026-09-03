"""Unit tests for the AdaptiveRLController with extended 9-dimensional features and action space."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from friday_serve.rl_controller import (
    ACTIONS,
    ACTION_TO_KNOBS,
    FEATURE_DIM,
    AdaptiveRLController,
    LinUCBModel,
)


class TestLinUCBModel(unittest.TestCase):
    def test_initialization(self) -> None:
        model = LinUCBModel.create(dim=FEATURE_DIM)
        self.assertEqual(model.A.shape, (FEATURE_DIM, FEATURE_DIM))
        self.assertEqual(model.b.shape, (FEATURE_DIM, 1))
        np.testing.assert_array_equal(model.A, np.identity(FEATURE_DIM))

    def test_update_and_ucb(self) -> None:
        model = LinUCBModel.create(dim=FEATURE_DIM)
        x = np.ones((FEATURE_DIM, 1))
        ucb_before = model.ucb(x, alpha=1.0)
        model.update(x, reward=0.35)
        theta = model.theta()
        self.assertGreater(float((theta.T @ x).item()), 0.0)


class TestAdaptiveRLController(unittest.TestCase):
    def test_feature_dim(self) -> None:
        self.assertEqual(FEATURE_DIM, 9)
        self.assertEqual(len(ACTIONS), 6)
        for act in (
            "baseline",
            "full_optimized",
            "deep_bundled_long",
            "speculative_draft",
            "prefix_cached",
            "throughput_grouped",
        ):
            self.assertIn(act, ACTIONS)
            self.assertIn(act, ACTION_TO_KNOBS)

    def test_feature_extraction(self) -> None:
        feat = AdaptiveRLController.extract_features(
            model_id="mlx-community/gemma-3-4b-it-4bit",
            prompt_tokens=512,
            output_tokens=64,
            has_prefix_cache=True,
            is_concurrent=False,
            has_ngram_overlap=True,
        )
        self.assertEqual(feat.shape, (9, 1))
        # [is_1b, is_4b, is_12b, norm_p, norm_o, ratio, has_prefix, is_concurrent, has_ngram]
        self.assertEqual(feat[0, 0], 0.0)  # 1b
        self.assertEqual(feat[1, 0], 1.0)  # 4b
        self.assertEqual(feat[2, 0], 0.0)  # 12b
        self.assertAlmostEqual(feat[3, 0], 0.5)  # 512 / 1024
        self.assertAlmostEqual(feat[4, 0], 0.25)  # 64 / 256
        self.assertEqual(feat[6, 0], 1.0)  # has_prefix_cache
        self.assertEqual(feat[7, 0], 0.0)  # is_concurrent
        self.assertEqual(feat[8, 0], 1.0)  # has_ngram_overlap

    def test_select_action_and_learn_prefix_cached(self) -> None:
        ctrl = AdaptiveRLController(alpha=0.1)

        # Train: when has_prefix_cache=True, prefix_cached gets huge reward (0.88)
        for _ in range(25):
            ctrl.observe_reward(
                "prefix_cached",
                "mlx-community/gemma-3-4b-it-4bit",
                350,
                32,
                reward=0.88,
                has_prefix_cache=True,
            )
            ctrl.observe_reward(
                "baseline",
                "mlx-community/gemma-3-4b-it-4bit",
                350,
                32,
                reward=0.0,
                has_prefix_cache=True,
            )

        # Should select prefix_cached when has_prefix_cache=True
        chosen, knobs, score = ctrl.select_action(
            "mlx-community/gemma-3-4b-it-4bit", 350, 32, has_prefix_cache=True
        )
        self.assertEqual(chosen, "prefix_cached")
        self.assertTrue(knobs.get("prefix_cached"))
        self.assertTrue(knobs.get("compiled_fixed_cache"))

    def test_select_action_and_learn_throughput_grouped(self) -> None:
        ctrl = AdaptiveRLController(alpha=0.1)

        # Train: when is_concurrent=True, throughput_grouped gets high reward
        for _ in range(25):
            ctrl.observe_reward(
                "throughput_grouped",
                "mlx-community/gemma-3-1b-it-4bit",
                100,
                64,
                reward=1.05,
                is_concurrent=True,
            )
            ctrl.observe_reward(
                "baseline",
                "mlx-community/gemma-3-1b-it-4bit",
                100,
                64,
                reward=0.0,
                is_concurrent=True,
            )

        chosen, knobs, score = ctrl.select_action(
            "mlx-community/gemma-3-1b-it-4bit", 100, 64, is_concurrent=True
        )
        self.assertEqual(chosen, "throughput_grouped")
        self.assertEqual(knobs.get("mode"), "throughput")
        self.assertEqual(knobs.get("max_width"), 4)

    def test_select_action_and_learn_speculative(self) -> None:
        ctrl = AdaptiveRLController(alpha=0.1)

        # Train: when has_ngram_overlap=True, speculative_draft gets high reward
        for _ in range(25):
            ctrl.observe_reward(
                "speculative_draft",
                "mlx-community/gemma-3-4b-it-4bit",
                200,
                80,
                reward=0.40,
                has_ngram_overlap=True,
            )
            ctrl.observe_reward(
                "baseline",
                "mlx-community/gemma-3-4b-it-4bit",
                200,
                80,
                reward=0.0,
                has_ngram_overlap=True,
            )

        chosen, knobs, score = ctrl.select_action(
            "mlx-community/gemma-3-4b-it-4bit", 200, 80, has_ngram_overlap=True
        )
        self.assertEqual(chosen, "speculative_draft")
        self.assertEqual(knobs.get("speculate_k"), ACTION_TO_KNOBS["speculative_draft"]["speculate_k"])

    def test_save_and_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            path = Path(tmp.name)
            ctrl = AdaptiveRLController(alpha=0.5, save_path=path)
            ctrl.observe_reward(
                "deep_bundled_long",
                "mlx-community/gemma-3-12b-it-4bit",
                500,
                256,
                reward=0.35,
            )
            ctrl.save()

            loaded = AdaptiveRLController.load(path)
            self.assertEqual(len(loaded.history), 0)
            self.assertEqual(loaded.alpha, 0.5)
            self.assertEqual(len(loaded.models), 6)
            np.testing.assert_allclose(
                loaded.models["deep_bundled_long"].b,
                ctrl.models["deep_bundled_long"].b,
            )
            self.assertEqual(loaded.models["deep_bundled_long"].A.shape, (9, 9))

    def test_shadow_log_records_without_touching_weights(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            shadow = Path(d) / "shadow.jsonl"
            ctrl = AdaptiveRLController(shadow_log_path=shadow)
            a_before = ctrl.models["full_optimized"].A.copy()
            ctrl.log_decision(
                model_id="mlx-community/gemma-3-4b-it-4bit",
                prompt_tokens=200,
                output_tokens=32,
                shadow_action="full_optimized",
                shadow_knobs=ACTION_TO_KNOBS["full_optimized"],
                shadow_score=1.2,
                applied_knobs={"head_skip_prefill": True},
                applied_plan="device_profile_dispatch",
            )
            import json as _json

            record = _json.loads(shadow.read_text().splitlines()[0])
            self.assertEqual(record["shadow_action"], "full_optimized")
            self.assertEqual(record["applied_knobs"], {"head_skip_prefill": True})
            self.assertEqual(ctrl.history, [])
            np.testing.assert_array_equal(ctrl.models["full_optimized"].A, a_before)


if __name__ == "__main__":
    unittest.main()
