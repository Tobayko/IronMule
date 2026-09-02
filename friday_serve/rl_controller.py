"""Reinforcement Learning Contextual Bandit Controller for Friday LLM Runtime.

Selects the optimal hardware optimization knobs dynamically based on request features
(model size, prompt length, output length, phase balance) using LinUCB (Upper Confidence Bound).
Logs all decisions and rewards for offline replay evaluation (OPE).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTIONS: tuple[str, ...] = (
    "baseline",
    "head_skip",
    "fixed_compiled",
    "readback_bundled",
    "full_optimized",
)

ACTION_TO_KNOBS: Mapping[str, Mapping[str, Any]] = {
    "baseline": {},
    "head_skip": {"head_skip_prefill": True},
    "fixed_compiled": {"compiled_fixed_cache": True},
    "readback_bundled": {"readback_every": 8},
    "full_optimized": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "readback_every": 8,
    },
}

FEATURE_DIM = 6


@dataclass
class LinUCBModel:
    """LinUCB parameters for one action."""

    A: np.ndarray  # (d, d)
    b: np.ndarray  # (d, 1)

    @classmethod
    def create(cls, dim: int = FEATURE_DIM) -> "LinUCBModel":
        return cls(A=np.identity(dim, dtype=np.float64), b=np.zeros((dim, 1), dtype=np.float64))

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def ucb(self, x: np.ndarray, alpha: float = 1.0) -> float:
        """Compute Upper Confidence Bound score."""
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean = float((theta.T @ x).item())
        variance = max(0.0, float((x.T @ A_inv @ x).item()))
        uncertainty = alpha * math.sqrt(variance)
        return mean + uncertainty

    def update(self, x: np.ndarray, reward: float) -> None:
        """Rank-1 Bayesian update."""
        self.A += x @ x.T
        self.b += reward * x


class AdaptiveRLController:
    """Contextual Bandit Controller mapping request context to optimal knob configurations."""

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        save_path: Path | None = None,
    ) -> None:
        self.alpha = alpha
        self.save_path = save_path
        self.models: dict[str, LinUCBModel] = {
            action: LinUCBModel.create(FEATURE_DIM) for action in ACTIONS
        }
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def extract_features(
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
    ) -> np.ndarray:
        """Extract a normalized 6-dimensional feature vector:

        [is_1b, is_4b, is_12b, norm_prompt_len, norm_output_len, prompt_to_output_ratio]
        """
        is_1b = 1.0 if "1b" in model_id.lower() else 0.0
        is_4b = 1.0 if "4b" in model_id.lower() else 0.0
        is_12b = 1.0 if "12b" in model_id.lower() else 0.0
        norm_p = min(prompt_tokens / 1024.0, 2.0)
        norm_o = min(output_tokens / 256.0, 2.0)
        ratio = min(prompt_tokens / max(1, output_tokens), 10.0) / 10.0

        vec = np.array([is_1b, is_4b, is_12b, norm_p, norm_o, ratio], dtype=np.float64)
        return vec.reshape((FEATURE_DIM, 1))

    def select_action(
        self,
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
        *,
        allowed_actions: Sequence[str] = ACTIONS,
    ) -> tuple[str, Mapping[str, Any], float]:
        """Select best action using LinUCB exploration/exploitation.

        Returns (action_name, knob_dict, ucb_score).
        """
        x = self.extract_features(model_id, prompt_tokens, output_tokens)
        best_action = "baseline"
        best_score = -float("inf")

        for action in allowed_actions:
            if action not in self.models:
                continue
            score = self.models[action].ucb(x, alpha=self.alpha)
            if score > best_score:
                best_score = score
                best_action = action

        knobs = ACTION_TO_KNOBS[best_action]
        return best_action, knobs, best_score

    def observe_reward(
        self,
        action: str,
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
        reward: float,
    ) -> None:
        """Update the RL model with the observed empirical reward (gain = 1 - ratio)."""
        if action not in self.models:
            return
        x = self.extract_features(model_id, prompt_tokens, output_tokens)
        self.models[action].update(x, reward)
        self.history.append({
            "action": action,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "reward": reward,
        })
        if self.save_path:
            self.save()

    def save(self) -> None:
        if not self.save_path:
            return
        data = {
            "alpha": self.alpha,
            "actions": {
                action: {
                    "A": model.A.tolist(),
                    "b": model.b.tolist(),
                }
                for action, model in self.models.items()
            },
            "history_count": len(self.history),
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "AdaptiveRLController":
        if not path.exists():
            return cls(save_path=path)
        data = json.loads(path.read_text())
        controller = cls(alpha=data.get("alpha", 0.5), save_path=path)
        for action, params in data.get("actions", {}).items():
            if action in controller.models:
                controller.models[action].A = np.array(params["A"], dtype=np.float64)
                controller.models[action].b = np.array(params["b"], dtype=np.float64)
        return controller


__all__ = [
    "ACTIONS",
    "ACTION_TO_KNOBS",
    "AdaptiveRLController",
    "LinUCBModel",
]
