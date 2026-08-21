#!/usr/bin/env python3
"""Where does inference time actually go: moving bytes, or doing arithmetic?

This answers a question that decides which optimizations can possibly help.  If
a machine is memory-bound, then making the code "closer to the metal" optimizes
the part that was never the bottleneck; only moving fewer bytes helps.

Two independent readings, so neither has to be trusted alone:

  1. **Utilization.** Autoregressive generation reads every weight once per
     token.  Dividing weight bytes by measured time gives effective bandwidth;
     comparing 2*parameters per token against the same time gives effective
     compute.  Both are then expressed as a share of the device peak.
  2. **Prefill versus generation.** Prefill processes many tokens against one
     pass over the weights, generation one token per pass.  If the machine is
     memory-bound, prefill must be dramatically faster per token -- and by how
     much is a measurement, not an assumption.

Peak figures are published vendor numbers and are marked as such: they bound the
ratio, they are not themselves measured here.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import BudgetGuard, release_gate, require_ac_power, run_persisted  # noqa: E402

# Published Apple M1 Max figures.  Not measured here; used only to form ratios.
PEAK_BANDWIDTH_GB_S = 400.0
PEAK_FP16_TFLOPS = 21.0

MODELS = (
    ("mlx-community/gemma-3-1b-it-4bit", "gemma-3-1b"),
    ("mlx-community/gemma-3-4b-it-4bit", "gemma-3-4b"),
)
PROMPT_REPEATS = 60
GENERATE_TOKENS = 24
REPETITIONS = 5
WARMUP = 2
# 4-bit weights: roughly two parameters per stored byte.
PARAMS_PER_BYTE = 2


def utilization(weight_bytes: int, seconds_per_token: float) -> dict[str, float]:
    """Effective bandwidth and compute for one generated token, as peak shares.

    Generation reads the full weight set per token, which makes weight_bytes the
    honest numerator for bandwidth.  The compute figure uses the standard
    2*parameters estimate for a forward pass.
    """

    if seconds_per_token <= 0:
        raise ValueError("time per token must be positive")
    bandwidth_gb_s = weight_bytes / seconds_per_token / 1e9
    tflops = 2 * weight_bytes * PARAMS_PER_BYTE / seconds_per_token / 1e12
    return {
        "bandwidth_gb_s": round(bandwidth_gb_s, 1),
        "bandwidth_share": round(bandwidth_gb_s / PEAK_BANDWIDTH_GB_S, 4),
        "compute_tflops": round(tflops, 4),
        "compute_share": round(tflops / PEAK_FP16_TFLOPS, 4),
    }


def classify(shares: dict[str, float], factor: float = 3.0) -> str:
    """Name the bottleneck, or refuse to when the readings are close.

    The threshold is deliberately blunt: a small gap between the two shares means
    the measurement cannot separate them, and saying so is better than picking a
    winner from noise.
    """

    bandwidth = shares["bandwidth_share"]
    compute = shares["compute_share"]
    if bandwidth >= compute * factor:
        return "memory_bound"
    if compute >= bandwidth * factor:
        return "compute_bound"
    return "inconclusive"


def _self_check() -> int:
    """Offline checks of the arithmetic and the verdict rule; no GPU, no model."""

    # 1 GB of weights at 10 ms per token is 100 GB/s, a quarter of peak.
    shares = utilization(1_000_000_000, 0.01)
    assert shares["bandwidth_gb_s"] == 100.0, shares
    assert abs(shares["bandwidth_share"] - 0.25) < 1e-6, shares

    assert classify({"bandwidth_share": 0.52, "compute_share": 0.04}) == "memory_bound"
    assert classify({"bandwidth_share": 0.03, "compute_share": 0.60}) == "compute_bound"
    # Close readings must not be forced into a verdict.
    assert classify({"bandwidth_share": 0.30, "compute_share": 0.25}) == "inconclusive"
    # Clearly past the factor and clearly short of it.  The exact boundary is
    # deliberately not asserted: 0.10 * 3.0 is 0.30000000000000004 in binary
    # floating point, so testing equality there would pin down an artefact.
    assert classify({"bandwidth_share": 0.31, "compute_share": 0.10}) == "memory_bound"
    assert classify({"bandwidth_share": 0.28, "compute_share": 0.10}) == "inconclusive"

    try:
        utilization(1000, 0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("zero time per token must be refused")
    print(json.dumps({"self_check": "pass", "checks": 8}))
    return 0


def measure_model(model_id: str, label: str, guard: BudgetGuard) -> dict[str, object]:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    model, tokenizer = load(model_id)
    weight_bytes = sum(p.size * p.dtype.size for _, p in tree_flatten(model.parameters()))

    prompt = "Explain neural networks in detail. " * PROMPT_REPEATS
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    prompt_tokens = len(text) if isinstance(text, list) else len(tokenizer.encode(text))

    for _ in range(WARMUP):
        gpu_started = time.perf_counter()
        for _ in stream_generate(model, tokenizer, text, max_tokens=4):
            pass
        guard.record_gpu(time.perf_counter() - gpu_started)

    prefill_s: list[float] = []
    per_token_s: list[float] = []
    for _ in range(REPETITIONS):
        started = time.perf_counter_ns()
        first = None
        produced = 0
        last = started
        for _ in stream_generate(model, tokenizer, text, max_tokens=GENERATE_TOKENS):
            produced += 1
            last = time.perf_counter_ns()
            if first is None:
                first = last - started
        guard.record_gpu((last - started) / 1e9)
        if produced < 2 or first is None:
            raise SystemExit("generation produced too few tokens to measure")
        prefill_s.append(first / 1e9)
        per_token_s.append(((last - started) - first) / 1e9 / (produced - 1))

    prefill = statistics.median(prefill_s)
    per_token = statistics.median(per_token_s)
    shares = utilization(weight_bytes, per_token)
    result = {
        "model": label,
        "weight_bytes": weight_bytes,
        "prompt_tokens": prompt_tokens,
        "prefill_seconds": round(prefill, 4),
        "prefill_tokens_per_second": round(prompt_tokens / prefill, 1),
        "generation_ms_per_token": round(per_token * 1000, 3),
        "generation_tokens_per_second": round(1 / per_token, 1),
        "prefill_speedup_per_token": round((prompt_tokens / prefill) * per_token, 1),
        "prefill_seconds_samples": prefill_s,
        "generation_seconds_per_token_samples": per_token_s,
        **shares,
        "verdict": classify(shares),
    }
    del model, tokenizer
    mx.clear_cache()
    return result


def run() -> dict[str, object]:
    guard = BudgetGuard()
    results = []
    for index, (model_id, label) in enumerate(MODELS):
        if index:
            guard.required_break()
        results.append(measure_model(model_id, label, guard))
    verdicts = {entry["verdict"] for entry in results}
    budget = guard.summary()
    return {
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
        "peak_fp16_tflops": PEAK_FP16_TFLOPS,
        "peak_source": "published vendor figures, not measured here",
        "models": results,
        "verdict": verdicts.pop() if len(verdicts) == 1 else "mixed",
        "wall_seconds": budget["wall_seconds"],
        "gpu_work_seconds": budget["gpu_work_seconds"],
        "budget": budget,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_roofline", allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    gated = release_gate(args, _self_check)
    if gated is not None:
        return gated

    power = require_ac_power()

    def operation() -> dict[str, object]:
        report = run()
        report["power_source"] = power
        return report

    report = run_persisted("roofline", operation)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
