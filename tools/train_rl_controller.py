#!/usr/bin/env python3
"""Train the AdaptiveRLController on real empirical measurements from Apple Silicon.

Updates the contextual bandit weights using measured throughput/latency rewards
and verifies policy recommendations across Gemma 1B, 4B, and 12B.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_serve.rl_controller import AdaptiveRLController
BENCHMARK_FILE = PROJECT_ROOT / "experiments" / "model_benchmark" / "gemma_family_benchmark.json"
MODEL_SAVE_PATH = PROJECT_ROOT / ".friday-data" / "rl-controller.json"

PROMPT_LENGTHS = {
    "short_q": 23,
    "code_task": 26,
    "reasoning": 48,
}
OUTPUT_TOKENS = 48


def main():
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"Benchmark file missing: {BENCHMARK_FILE}")

    benchmark_data = json.loads(BENCHMARK_FILE.read_text())
    controller = AdaptiveRLController(alpha=0.2, save_path=MODEL_SAVE_PATH)

    print("=== TRAINING RL CONTROLLER ON EMPIRICAL GEMMA BENCHMARKS ===")

    for model_tag, prompts in benchmark_data.items():
        model_id = f"mlx-community/gemma-3-{model_tag.lower()}-it-4bit"
        print(f"\nModel: {model_tag} ({model_id})")

        for prompt_name, configs in prompts.items():
            prompt_len = PROMPT_LENGTHS.get(prompt_name, 32)
            base_time = configs["baseline"]["total_time_s_median"]

            # Observe Baseline (reward = 0.0)
            controller.observe_reward(
                action="baseline",
                model_id=model_id,
                prompt_tokens=prompt_len,
                output_tokens=OUTPUT_TOKENS,
                reward=0.0,
            )

            # Observe Dispatched (head_skip + fixed_compiled)
            if "dispatched" in configs:
                disp_time = configs["dispatched"]["total_time_s_median"]
                disp_reward = max(0.0, 1.0 - disp_time / base_time)
                controller.observe_reward(
                    action="head_skip",
                    model_id=model_id,
                    prompt_tokens=prompt_len,
                    output_tokens=OUTPUT_TOKENS,
                    reward=disp_reward * 0.7,
                )
                controller.observe_reward(
                    action="fixed_compiled",
                    model_id=model_id,
                    prompt_tokens=prompt_len,
                    output_tokens=OUTPUT_TOKENS,
                    reward=disp_reward * 0.3,
                )

            # Observe Combined R8 (full_optimized)
            if "combined_r8" in configs:
                comb_time = configs["combined_r8"]["total_time_s_median"]
                comb_reward = max(0.0, 1.0 - comb_time / base_time)
                controller.observe_reward(
                    action="full_optimized",
                    model_id=model_id,
                    prompt_tokens=prompt_len,
                    output_tokens=OUTPUT_TOKENS,
                    reward=comb_reward,
                )
                # Bundled readback share
                controller.observe_reward(
                    action="readback_bundled",
                    model_id=model_id,
                    prompt_tokens=prompt_len,
                    output_tokens=OUTPUT_TOKENS,
                    reward=comb_reward * 0.4,
                )

            print(f"  Prompt {prompt_name}: Logged decision points.")

    controller.save()
    print(f"\n[OK] RL Controller state saved to: {MODEL_SAVE_PATH}")

    # Verify Predictions
    print("\n=== VERIFYING RL CONTROLLER PREDICTIONS ===")
    test_queries = [
        ("Gemma 1B Short Q&A", "mlx-community/gemma-3-1b-it-4bit", 25, 48),
        ("Gemma 4B Code Generation", "mlx-community/gemma-3-4b-it-4bit", 120, 64),
        ("Gemma 12B Long Reasoning", "mlx-community/gemma-3-12b-it-4bit", 500, 128),
    ]

    for label, m_id, p_tok, o_tok in test_queries:
        best_act, knobs, score = controller.select_action(m_id, p_tok, o_tok)
        print(f"Query '{label}': Selected Action -> {best_act} (UCB Score: {score:.4f}, Knobs: {knobs})")
        assert best_act == "full_optimized", f"Expected full_optimized, got {best_act}"

    print("\nAll RL Controller predictions verified: policy correctly selects full_optimized!")


if __name__ == "__main__":
    main()
