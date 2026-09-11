"""LinUCB contextual bandit for Friday — shadow only.

Given a request's features (model size, prompt/output length, phase balance,
n-gram overlap) the controller *would* pick a knob configuration. In the serving
path it never does: :meth:`log_decision` records the choice it would have made
next to the knobs the device profile actually authorised, and nothing else. No
weights are updated from a serving request (there is no measured reward there),
and ``speculate_k`` is never applied — RL stays NO-GO until R2 (BACKLOG L1),
speculation stays out of the delivery path (GEMINI_SELF_LEARNING_SYSTEM E01).

``select_action`` / ``observe_reward`` remain for the offline replay/OPE
harnesses in ``friday_optimizer``; only ``log_decision`` runs live.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ACTIONS: tuple[str, ...] = (
    "baseline",
    "full_optimized",
    "deep_bundled_long",
    "speculative_draft",
    "prefix_cached",
    "throughput_grouped",
)

ACTION_TO_KNOBS: Mapping[str, Mapping[str, Any]] = {
    "baseline": {},
    "full_optimized": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "readback_every": 8,
    },
    "deep_bundled_long": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "readback_every": 16,
        "wired_fraction": 0.6,
    },
    "speculative_draft": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "speculate_k": 3,
        "speculate_ngram": 3,
    },
    "prefix_cached": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "readback_every": 8,
        "prefix_cached": True,
    },
    "throughput_grouped": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "mode": "throughput",
        "max_width": 4,
    },
}

FEATURE_DIM = 9


def detect_ngram_overlap(tokens: Sequence[int], ngram: int = 3) -> bool:
    """Detect if token sequence contains repeated n-grams indicating context reuse."""
    if len(tokens) < 32:
        return False
    seen = set()
    for i in range(len(tokens) - ngram + 1):
        g = tuple(tokens[i : i + ngram])
        if g in seen:
            return True
        seen.add(g)
    return False


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
        shadow_log_path: Path | None = None,
    ) -> None:
        self.alpha = alpha
        self.save_path = save_path
        self.shadow_log_path = shadow_log_path
        self.models: dict[str, LinUCBModel] = {
            action: LinUCBModel.create(FEATURE_DIM) for action in ACTIONS
        }
        self.history: list[dict[str, Any]] = []

    def log_decision(
        self,
        *,
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
        shadow_action: str,
        shadow_knobs: Mapping[str, Any],
        shadow_score: float,
        applied_knobs: Mapping[str, Any],
        applied_plan: str,
        has_ngram_overlap: bool = False,
    ) -> None:
        """Append one shadow decision. Never touches the model weights.

        This is the propensity stream the offline replay harness needs; the
        serving path applies ``applied_knobs`` (the device-profile set), not
        ``shadow_knobs``.
        """

        if self.shadow_log_path is None:
            return
        record = {
            "ts": time.time(),
            "model_id": model_id,
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "shadow_action": shadow_action,
            "shadow_knobs": dict(shadow_knobs),
            "shadow_score": float(shadow_score),
            "applied_knobs": dict(applied_knobs),
            "applied_plan": applied_plan,
            "has_ngram_overlap": bool(has_ngram_overlap),
        }
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.shadow_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @staticmethod
    def extract_features(
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
        *,
        has_prefix_cache: bool = False,
        is_concurrent: bool = False,
        has_ngram_overlap: bool = False,
    ) -> np.ndarray:
        """Extract a normalized 9-dimensional feature vector:

        [is_1b, is_4b, is_12b, norm_prompt_len, norm_output_len, prompt_to_output_ratio,
         has_prefix_cache, is_concurrent, has_ngram_overlap]
        """
        is_1b = 1.0 if "1b" in model_id.lower() else 0.0
        is_4b = 1.0 if "4b" in model_id.lower() else 0.0
        is_12b = 1.0 if "12b" in model_id.lower() else 0.0
        norm_p = min(prompt_tokens / 1024.0, 2.0)
        norm_o = min(output_tokens / 256.0, 2.0)
        ratio = min(prompt_tokens / max(1, output_tokens), 10.0) / 10.0
        f_prefix = 1.0 if has_prefix_cache else 0.0
        f_concurrent = 1.0 if is_concurrent else 0.0
        f_ngram = 1.0 if has_ngram_overlap else 0.0

        vec = np.array(
            [is_1b, is_4b, is_12b, norm_p, norm_o, ratio, f_prefix, f_concurrent, f_ngram],
            dtype=np.float64,
        )
        return vec.reshape((FEATURE_DIM, 1))

    def select_action(
        self,
        model_id: str,
        prompt_tokens: int,
        output_tokens: int,
        *,
        has_prefix_cache: bool = False,
        is_concurrent: bool = False,
        has_ngram_overlap: bool = False,
        allowed_actions: Sequence[str] = ACTIONS,
    ) -> tuple[str, Mapping[str, Any], float]:
        """Select best action using LinUCB exploration/exploitation.

        Returns (action_name, knob_dict, ucb_score).
        """
        x = self.extract_features(
            model_id,
            prompt_tokens,
            output_tokens,
            has_prefix_cache=has_prefix_cache,
            is_concurrent=is_concurrent,
            has_ngram_overlap=has_ngram_overlap,
        )
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
        *,
        has_prefix_cache: bool = False,
        is_concurrent: bool = False,
        has_ngram_overlap: bool = False,
    ) -> None:
        """Update the RL model with the observed empirical reward (gain = 1 - ratio)."""
        if action not in self.models:
            return
        x = self.extract_features(
            model_id,
            prompt_tokens,
            output_tokens,
            has_prefix_cache=has_prefix_cache,
            is_concurrent=is_concurrent,
            has_ngram_overlap=has_ngram_overlap,
        )
        self.models[action].update(x, reward)
        self.history.append({
            "action": action,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "reward": reward,
            "has_prefix_cache": has_prefix_cache,
            "is_concurrent": is_concurrent,
            "has_ngram_overlap": has_ngram_overlap,
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
    def load(cls, path: Path, *, shadow_log_path: Path | None = None) -> "AdaptiveRLController":
        if not path.exists():
            return cls(save_path=path, shadow_log_path=shadow_log_path)
        data = json.loads(path.read_text())
        controller = cls(
            alpha=data.get("alpha", 0.5), save_path=path, shadow_log_path=shadow_log_path
        )
        for action, params in data.get("actions", {}).items():
            if action in controller.models:
                arr_a = np.array(params["A"], dtype=np.float64)
                arr_b = np.array(params["b"], dtype=np.float64)
                if arr_a.shape == (FEATURE_DIM, FEATURE_DIM) and arr_b.shape == (FEATURE_DIM, 1):
                    controller.models[action].A = arr_a
                    controller.models[action].b = arr_b
        return controller


__all__ = [
    "ACTIONS",
    "ACTION_TO_KNOBS",
    "AdaptiveRLController",
    "LinUCBModel",
]
