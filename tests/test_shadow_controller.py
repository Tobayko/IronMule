"""Test Adaptive Learning Controller v0.1 in Shadow Mode (Backlog L1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from friday_serve.rl_controller import (
    ACTIONS,
    ACTION_TO_KNOBS,
    FEATURE_DIM,
    AdaptiveRLController,
    LinUCBModel,
)


def test_linucb_rank1_update(tmp_path: Path):
    model = LinUCBModel.create(dim=FEATURE_DIM)
    x = np.ones((FEATURE_DIM, 1), dtype=np.float64) / np.sqrt(FEATURE_DIM)

    # Initial score
    score_0 = model.ucb(x, alpha=1.0)
    assert score_0 > 0.0

    # Observe high reward
    model.update(x, reward=1.0)
    score_1 = model.ucb(x, alpha=1.0)
    assert score_1 != score_0

    # Theta should reflect positive reward
    theta = model.theta()
    assert np.all(theta > 0.0)


def test_adaptive_rl_controller_shadow_mode(tmp_path: Path):
    save_file = tmp_path / "rl-controller.json"
    ctrl = AdaptiveRLController.load(save_file)
    assert len(ctrl.models) == len(ACTIONS)

    # Extract features for a 4B model request
    x = ctrl.extract_features(
        model_id="mlx-community/gemma-3-4b-it-4bit",
        prompt_tokens=256,
        output_tokens=64,
        has_prefix_cache=False,
    )
    assert x.shape == (FEATURE_DIM, 1)

    # Action selection
    action, knobs, score = ctrl.select_action(
        model_id="mlx-community/gemma-3-4b-it-4bit",
        prompt_tokens=256,
        output_tokens=64,
    )
    assert action in ACTIONS
    assert isinstance(knobs, dict)

    # Update reward in shadow mode
    ctrl.observe_reward(
        action=action,
        model_id="mlx-community/gemma-3-4b-it-4bit",
        prompt_tokens=256,
        output_tokens=64,
        reward=0.18,  # +18% speedup reward
    )

    assert save_file.exists()
    saved_data = json.loads(save_file.read_text())
    assert saved_data["history_count"] == 1
    assert action in saved_data["actions"]

    # Reload controller and verify persistent weights
    reloaded = AdaptiveRLController.load(save_file)
    assert reloaded.models[action].A.shape == (FEATURE_DIM, FEATURE_DIM)
    assert np.allclose(reloaded.models[action].A, ctrl.models[action].A)
    assert np.allclose(reloaded.models[action].b, ctrl.models[action].b)
