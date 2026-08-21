#!/usr/bin/env python3
"""A layer between an unmodified model and the hardware, measured.

The model is not retrained, not quantized, not edited.  Its forward pass is
wrapped in ``mx.compile``, which fuses adjacent operations so intermediate
results stay in fast memory instead of travelling through main memory.  Then the
wrapped and unwrapped versions are compared, paired, against a threshold fixed
before the run.

Why this is the one lever left, and not a guess: the roofline measurement in this
project found inference to be memory-bound by roughly a factor of 13 -- 2-4% of
peak compute against 32-52% of peak bandwidth.  Fusion is precisely the technique
that removes memory traffic without changing arithmetic, so it is where a gain
should appear if one exists anywhere.

Two regimes are measured separately because they behave differently:

  - **prefill**: the whole prompt in one pass, matmul-dominated
  - **single token**: one autoregressive step, where elementwise work weighs more
    relative to the matmuls -- and where real generation spends its time

Correctness is checked before timing: the wrapped forward pass must produce
bit-identical logits, or the result is discarded.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import (  # noqa: E402
    BudgetGuard,
    release_gate,
    require_ac_power,
    resolve_local_model_snapshot,
    run_persisted,
)

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "measure_dispatch_plan", Path(__file__).resolve().parent / "measure_dispatch_plan.py"
)
_PLAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLAN)

MODELS = (
    ("mlx-community/gemma-3-1b-it-4bit", "gemma-3-1b"),
    ("mlx-community/gemma-3-4b-it-4bit", "gemma-3-4b"),
)
PROMPT = "Explain neural networks."
BLOCKS = 20
REPLICATES = 3
WARMUP = 4
MDE = _PLAN.MDE
BOOTSTRAP_SEED = 0x40092_2026


def summarize(log_ratios: list[float]) -> dict[str, float]:
    summary = _PLAN.paired_ratio(log_ratios)
    summary["clears_mde"] = _PLAN.clears_threshold(summary, MDE)
    return summary


def _self_check() -> int:
    """Offline checks of the statistics; no GPU, no model."""

    clean = summarize([math.log(0.86)] * 20)
    assert abs(clean["ratio"] - 0.86) < 1e-9
    assert clean["clears_mde"]
    assert not summarize([0.0] * 20)["clears_mde"]
    # A 3% gain is real but must not clear a 5% threshold.
    assert not summarize([math.log(0.97)] * 20)["clears_mde"]
    # A regression is detected too; the gate is two-sided.
    assert summarize([math.log(1.25)] * 20)["clears_mde"]
    # One wild block must not manufacture a finding.
    assert not summarize([0.0] * 19 + [math.log(0.2)])["clears_mde"]
    assert MDE == _PLAN.MDE, "the layer must not use a friendlier threshold"
    print(json.dumps({"self_check": "pass", "checks": 7}))
    return 0


def measure_model(model_id: str, label: str, guard: BudgetGuard) -> dict[str, object]:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    snapshot = resolve_local_model_snapshot(model_id)
    model, tokenizer = load(str(snapshot.path))
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    tokens = mx.array(
        encoded if isinstance(encoded, list) else tokenizer.encode(encoded)
    )[None]
    gpu_started = time.perf_counter()
    mx.eval(tokens)
    mx.synchronize()
    guard.record_gpu(time.perf_counter() - gpu_started)

    def eager(batch):
        return model(batch)

    compiled = mx.compile(eager)

    regimes = {"prefill": tokens, "single_token": tokens[:, -1:]}
    results = []
    for regime, batch in regimes.items():
        guard.required_break()
        # Correctness before timing: the wrapper may fuse, never alter.
        gpu_started = time.perf_counter()
        plain = eager(batch)
        fused = compiled(batch)
        mx.eval(plain, fused)
        mx.synchronize()
        guard.record_gpu(time.perf_counter() - gpu_started)
        # Convert through MLX: numpy has no bfloat16, which some models use.
        plain_host = np.array(plain.astype(mx.float32), copy=False)
        fused_host = np.array(fused.astype(mx.float32), copy=False)
        scale = float(np.abs(plain_host).max())
        deviation = float(np.abs(plain_host - fused_host).max()) / max(scale, 1e-9)
        if deviation != 0.0:
            results.append(
                {"regime": regime, "rejected": "fused forward pass changed the logits",
                 "deviation": deviation}
            )
            continue

        for _ in range(WARMUP):
            gpu_started = time.perf_counter()
            mx.eval(eager(batch))
            mx.eval(compiled(batch))
            mx.synchronize()
            guard.record_gpu(time.perf_counter() - gpu_started)

        replicates: list[list[float]] = []
        eager_ms: list[float] = []
        eager_ns_samples: list[list[int]] = []
        fused_ns_samples: list[list[int]] = []
        for _ in range(REPLICATES):
            log_ratios = []
            replicate_eager_ns: list[int] = []
            replicate_fused_ns: list[int] = []
            for _ in range(BLOCKS):
                start = time.perf_counter_ns()
                mx.eval(eager(batch))
                mx.synchronize()
                plain_ns = time.perf_counter_ns() - start
                start = time.perf_counter_ns()
                mx.eval(compiled(batch))
                mx.synchronize()
                fused_ns = time.perf_counter_ns() - start
                guard.record_gpu((plain_ns + fused_ns) / 1e9)
                log_ratios.append(math.log(fused_ns / plain_ns))
                eager_ms.append(plain_ns / 1e6)
                replicate_eager_ns.append(plain_ns)
                replicate_fused_ns.append(fused_ns)
            replicates.append(log_ratios)
            eager_ns_samples.append(replicate_eager_ns)
            fused_ns_samples.append(replicate_fused_ns)

        aggregate = _PLAN.hierarchical_bootstrap(replicates, seed=BOOTSTRAP_SEED)
        aggregate["clears_mde"] = _PLAN.clears_threshold(aggregate, MDE)
        results.append(
            {
                "regime": regime,
                "rejected": None,
                "eager_ms": round(statistics.median(eager_ms), 3),
                "ratio": round(aggregate["ratio"], 4),
                "ci_low": round(aggregate["ci_low"], 4),
                "ci_high": round(aggregate["ci_high"], 4),
                "effect_percent": round(100.0 * (aggregate["ratio"] - 1.0), 2),
                "clears_mde": aggregate["clears_mde"],
                "replicate_ratios": [
                    round(math.exp(statistics.median(logs)), 4) for logs in replicates
                ],
                "replicate_log_ratios": replicates,
                "eager_ns_samples": eager_ns_samples,
                "fused_ns_samples": fused_ns_samples,
                "correctness": "bit_identical",
            }
        )

    del model, tokenizer
    mx.clear_cache()
    return {"model": label, **snapshot.report_identity(), "regimes": results}


def run() -> dict[str, object]:
    guard = BudgetGuard()
    measured = [measure_model(model_id, label, guard) for model_id, label in MODELS]
    generation = [
        regime
        for entry in measured
        for regime in entry["regimes"]
        if regime["regime"] == "single_token" and not regime.get("rejected")
    ]
    confirmed = [r for r in generation if r["clears_mde"]]
    budget = guard.summary()
    return {
        "models": measured,
        "mde": MDE,
        "generation_confirmed": len(confirmed),
        "generation_measured": len(generation),
        "verdict": (
            "layer_confirmed"
            if confirmed and len(confirmed) == len(generation)
            else "not_confirmed"
        ),
        "wall_seconds": budget["wall_seconds"],
        "gpu_work_seconds": budget["gpu_work_seconds"],
        "budget": budget,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_fusion_layer", allow_abbrev=False)
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

    report = run_persisted("fusion", operation)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1))
    return 0 if report["verdict"] == "layer_confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
