#!/usr/bin/env python3
"""Empirical benchmark of Prompt-Lookup Speculative Decoding on Apple Silicon M1 Max GPU.

Measures:
1. Baseline (speculate_k=0) vs Speculative (speculate_k=2, speculate_k=3).
2. Decode TPS (tokens/s), Wall-Time (s), Acceptance Rate.
3. Strict 100% Token-Identity Gate against greedy baseline.
4. Models: Gemma 3 4B and Gemma 3 12B across Document Extraction & Code Refactoring.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
if str(IRONMULE) not in sys.path:
    sys.path.insert(0, str(IRONMULE))

from friday_serve.ironmule_backend import IronMuleBackend
from ironmule.hw import static_facts

MODELS = [
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-3-12b-it-4bit",
]

BENCHMARK_PROMPTS = [
    {
        "type": "document_extraction",
        "name": "Document Extraction / Summarization",
        "prompt": (
            "Extract and reproduce the exact 'Architecture Overview' paragraph from the document below without modifying any words:\n\n"
            "Document Start:\n"
            "Section 1: Architecture Overview\n"
            "The Apple Silicon unified memory subsystem couples a high-performance system-on-chip with wide memory channels. "
            "By sharing a single physical DRAM pool across all execution clusters, the processor eliminates redundant data copies between CPU and GPU. "
            "During large language model inference, prompt prefill executes across parallel GPU threads, whereas autoregressive token decode is strictly constrained by memory bandwidth. "
            "Hardware-aware inference runtimes leverage stateful prefix caching, compiled graph execution, and speculative prompt lookup to maximize end-to-end token generation throughput.\n\n"
            "Section 2: Benchmarking Methodology\n"
            "Measurements are performed using paired trials with warmups and medians across repeated executions.\n"
            "Document End\n\n"
            "Extracted Architecture Overview:\n"
            "Section 1: Architecture Overview\n"
            "The Apple Silicon unified memory subsystem"
        ),
        "max_tokens": 90,
    },
    {
        "type": "code_refactoring",
        "name": "Python Code Refactoring / Implementation",
        "prompt": (
            "Here is the reference implementation of the inference telemetry collector:\n\n"
            "```python\n"
            "class InferenceTelemetry:\n"
            "    def __init__(self, model_name: str, batch_size: int, prefill_latency_ns: int, decode_latency_ns: int) -> None:\n"
            "        self.model_name = model_name\n"
            "        self.batch_size = batch_size\n"
            "        self.prefill_latency_ns = prefill_latency_ns\n"
            "        self.decode_latency_ns = decode_latency_ns\n"
            "        self.total_latency_ns = prefill_latency_ns + decode_latency_ns\n"
            "\n"
            "    def summary(self) -> dict[str, object]:\n"
            "        return {\n"
            "            \"model_name\": self.model_name,\n"
            "            \"batch_size\": self.batch_size,\n"
            "            \"prefill_ms\": self.prefill_latency_ns / 1_000_000.0,\n"
            "            \"decode_ms\": self.decode_latency_ns / 1_000_000.0,\n"
            "            \"total_ms\": self.total_latency_ns / 1_000_000.0,\n"
            "        }\n"
            "```\n\n"
            "Task: Provide the identical code block wrapped in an export module, keeping every line exactly unchanged:\n\n"
            "```python\n"
        ),
        "max_tokens": 80,
    },
]

CONFIGURATIONS = [
    {
        "name": "baseline_k0",
        "k": 0,
        "knobs": {"compiled_fixed_cache": True, "head_skip_prefill": True, "speculate_k": 0},
    },
    {
        "name": "speculative_k2",
        "k": 2,
        "knobs": {"compiled_fixed_cache": True, "head_skip_prefill": True, "speculate_k": 2, "speculate_ngram": 3},
    },
    {
        "name": "speculative_k3",
        "k": 3,
        "knobs": {"compiled_fixed_cache": True, "head_skip_prefill": True, "speculate_k": 3, "speculate_ngram": 3},
    },
]

N_REPETITIONS = 3


def benchmark_model_speculative(model_id: str) -> dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Benchmarking Speculative Decoding: {model_id}")
    print(f"=======================================================")

    backend = IronMuleBackend.load(model_id)
    # Ensure prefix cache is unset so decode behavior is isolated
    backend.set_prefix_cache(None)

    prompt_results: list[dict[str, Any]] = []

    for p_info in BENCHMARK_PROMPTS:
        p_type = p_info["type"]
        p_name = p_info["name"]
        prompt = p_info["prompt"]
        max_tokens = p_info["max_tokens"]

        token_ids = backend.encode(prompt)
        print(f"\n--- Task: {p_name} (Prompt Tokens: {len(token_ids)}, Target Decode: {max_tokens}) ---")

        # 1. Warmup for each configuration (width k+1)
        print("Warming up execution graphs for each knob setting...")
        for cfg in CONFIGURATIONS:
            backend.generate(token_ids, max_tokens=12, knobs=cfg["knobs"])

        cfg_measurements: dict[str, Any] = {}
        baseline_tokens: list[int] | None = None

        for cfg in CONFIGURATIONS:
            cfg_name = cfg["name"]
            k_val = cfg["k"]
            knobs = cfg["knobs"]

            runs_decode_s = []
            runs_tps = []
            runs_wall_s = []
            runs_acc = []
            captured_tokens = None
            captured_text = None

            for rep in range(N_REPETITIONS):
                t0 = time.perf_counter()
                res = backend.generate(token_ids, max_tokens=max_tokens, knobs=knobs)
                wall_s = time.perf_counter() - t0

                tokens = res["logical_tokens"]
                decode_s = res["decode_ns"] / 1e9
                tps = len(tokens) / decode_s if decode_s > 0 else 0.0
                acc = res.get("acceptance", 0.0)

                runs_decode_s.append(decode_s)
                runs_tps.append(tps)
                runs_wall_s.append(wall_s)
                runs_acc.append(acc)

                if captured_tokens is None:
                    captured_tokens = tokens
                    captured_text = res.get("text")

            med_decode_s = statistics.median(runs_decode_s)
            med_tps = statistics.median(runs_tps)
            med_wall_s = statistics.median(runs_wall_s)
            med_acc = statistics.median(runs_acc)

            if cfg_name == "baseline_k0":
                baseline_tokens = captured_tokens
                token_match = True
            else:
                token_match = (captured_tokens == baseline_tokens)

            print(f"  [{cfg_name} (k={k_val})] Decode TPS: {med_tps:.2f} tok/s | Decode Time: {med_decode_s:.3f} s | Acceptance: {med_acc:.2%} | Match: {token_match}")

            cfg_measurements[cfg_name] = {
                "k": k_val,
                "median_decode_tps": round(med_tps, 2),
                "median_decode_s": round(med_decode_s, 4),
                "median_wall_s": round(med_wall_s, 4),
                "median_acceptance_rate": round(med_acc, 4),
                "token_match": token_match,
                "tokens_generated": len(captured_tokens),
                "tokens": captured_tokens,
                "text": captured_text,
            }

        # Compare speculative configs against baseline
        base_tps = cfg_measurements["baseline_k0"]["median_decode_tps"]
        k2_tps = cfg_measurements["speculative_k2"]["median_decode_tps"]
        k3_tps = cfg_measurements["speculative_k3"]["median_decode_tps"]

        k2_speedup = k2_tps / base_tps if base_tps > 0 else 1.0
        k3_speedup = k3_tps / base_tps if base_tps > 0 else 1.0
        k2_tps_change_pct = (k2_speedup - 1.0) * 100.0
        k3_tps_change_pct = (k3_speedup - 1.0) * 100.0

        all_match = (
            cfg_measurements["speculative_k2"]["token_match"] and
            cfg_measurements["speculative_k3"]["token_match"]
        )

        print(f"Summary for {p_name}:")
        print(f"  Baseline TPS: {base_tps:.2f} tok/s")
        print(f"  Speculative k=2: {k2_tps:.2f} tok/s ({k2_tps_change_pct:+.2f}%, speedup {k2_speedup:.2f}x, acc {cfg_measurements['speculative_k2']['median_acceptance_rate']:.2%})")
        print(f"  Speculative k=3: {k3_tps:.2f} tok/s ({k3_tps_change_pct:+.2f}%, speedup {k3_speedup:.2f}x, acc {cfg_measurements['speculative_k3']['median_acceptance_rate']:.2%})")
        print(f"  100% Token-Identity Gate: {'PASSED' if all_match else 'FAILED'}")

        prompt_results.append({
            "task_type": p_type,
            "task_name": p_name,
            "prompt_tokens": len(token_ids),
            "max_tokens": max_tokens,
            "configurations": cfg_measurements,
            "comparison": {
                "k2_speedup": round(k2_speedup, 3),
                "k2_tps_change_pct": round(k2_tps_change_pct, 2),
                "k2_acceptance_rate": cfg_measurements["speculative_k2"]["median_acceptance_rate"],
                "k3_speedup": round(k3_speedup, 3),
                "k3_tps_change_pct": round(k3_tps_change_pct, 2),
                "k3_acceptance_rate": cfg_measurements["speculative_k3"]["median_acceptance_rate"],
                "token_identity_passed": all_match,
            }
        })

    return {
        "model_id": model_id,
        "model_revision": backend.model_revision,
        "tasks": prompt_results,
    }


def main() -> None:
    facts = static_facts()
    print("=" * 60)
    print("FRIDAY PROMPT-LOOKUP SPECULATIVE BENCHMARK (M1 MAX)")
    print(f"Hardware: {facts.get('chip')} | GPU Cores: {facts.get('gpu_cores')} | Memory: {facts.get('memory_bytes', 0)//(1024**3)} GB")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    results: dict[str, Any] = {
        "benchmark": "speculative_decoding_empirical",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": facts,
        "models": {},
    }

    for model_id in MODELS:
        model_res = benchmark_model_speculative(model_id)
        results["models"][model_id] = model_res

    out_file = PROJECT_ROOT / "experiments" / "model_benchmark" / "speculative_empirical_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print(f"Raw results saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
