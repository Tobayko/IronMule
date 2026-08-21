#!/usr/bin/env python3
"""Measure one execution-plan optimization: batched dispatch vs. per-op sync.

Both arms perform *identical* arithmetic and produce byte-identical results.
Only the synchronization plan differs:

    serial   -- dispatch one matmul, wait for the GPU, repeat
    batched  -- dispatch N matmuls, wait once at the end

The comparison is **paired inside each block**, following the A/A design in
``docs/PHASE1_MATMUL_SPEC.md`` 5.3.1.  That matters more than it looks: an
unpaired measurement of this same machine carries roughly 20% between-run
spread, which is larger than most real effects.  Pairing cancels the shared
disturbance because both arms meet it inside the same block.

Hardware budgets from ``docs/H1_VORREGISTRIERUNG_ENTWURF.md`` section 5 are
enforced: mains power is required and the GPU work is capped.

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

FIXTURE_SEED = 4051312678
CANDIDATE_POOL_SEED = 20260821
SHAPE = 2048
DEFAULT_BLOCKS = 25
DEFAULT_REPLICATES = 3
# Frozen before measuring: an effect must clear this to count as real.
MDE = 0.05
BOOTSTRAP_SEED = 0xB0025_2026


# tools/ is loaded as loose scripts, not a package, so make the directory
# importable before pulling in the shared preconditions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import BudgetGuard, release_gate, require_ac_power, run_persisted  # noqa: E402


def paired_ratio(log_ratios: list[float]) -> dict[str, float]:
    """Session ratio and its interval, using the registered paired estimator.

    ``R = exp(median_b(log(t_candidate / t_baseline)))`` -- the median keeps a
    single wild block from carrying the result, which matters here because
    individual blocks on this device do go wild.
    """

    if len(log_ratios) < 2:
        raise ValueError("a paired ratio needs at least two blocks")
    centre = statistics.median(log_ratios)
    half = 1.96 * statistics.stdev(log_ratios) / math.sqrt(len(log_ratios))
    return {
        "ratio": math.exp(centre),
        "ci_low": math.exp(centre - half),
        "ci_high": math.exp(centre + half),
        "sd_log": statistics.stdev(log_ratios),
        "blocks": len(log_ratios),
    }


def clears_threshold(summary: dict[str, float], mde: float = MDE) -> bool:
    """True only when the whole interval sits beyond the frozen threshold."""

    return summary["ci_high"] < 1.0 - mde or summary["ci_low"] > 1.0 + mde


def _uniform_stream(seed: int):
    """Deterministic SHA-256 counter stream; no library RNG, no seed drift."""

    import hashlib

    counter = 0
    while True:
        digest = hashlib.sha256(seed.to_bytes(8, "big") + counter.to_bytes(8, "big")).digest()
        counter += 1
        for offset in range(0, 32, 4):
            yield int.from_bytes(digest[offset : offset + 4], "big") / 2**32


def hierarchical_bootstrap(
    per_replicate_log_ratios: list[list[float]], *, seed: int, draws: int = 10_000
) -> dict[str, float]:
    """Aggregate replicates the way the A/A contract does.

    Resamples replicates, then blocks within each drawn replicate, and rebuilds
    the whole statistic each time.  A single noisy replicate therefore widens
    the interval instead of deciding the outcome -- which is exactly what a
    per-replicate verdict gets wrong.
    """

    if len(per_replicate_log_ratios) < 2:
        raise ValueError("aggregation needs at least two replicates")
    stream = _uniform_stream(seed)
    point = math.exp(
        statistics.median([statistics.median(block) for block in per_replicate_log_ratios])
    )
    replays: list[float] = []
    for _ in range(draws):
        drawn = [
            per_replicate_log_ratios[int(next(stream) * len(per_replicate_log_ratios))]
            for _ in range(len(per_replicate_log_ratios))
        ]
        centres = [
            statistics.median([block[int(next(stream) * len(block))] for _ in range(len(block))])
            for block in drawn
        ]
        replays.append(math.exp(statistics.median(centres)))
    replays.sort()
    return {
        "ratio": point,
        "ci_low": replays[int(0.025 * len(replays))],
        "ci_high": replays[int(0.975 * len(replays))],
        "draws": draws,
    }


def _self_check() -> int:
    """Offline check of the statistics; no GPU, no MLX."""

    flat = [math.log(0.80)] * 20
    summary = paired_ratio(flat)
    assert abs(summary["ratio"] - 0.80) < 1e-9, summary
    assert clears_threshold(summary), "a clean 20% gain must clear a 5% threshold"

    none = [0.0] * 20
    assert not clears_threshold(paired_ratio(none)), "no effect must not clear"

    # One wild block must not flip a null result into a finding.
    noisy = [0.0] * 19 + [math.log(0.2)]
    assert not clears_threshold(paired_ratio(noisy)), "median must absorb one outlier"

    # A real effect must survive one wild block in the other direction.
    real = [math.log(0.80)] * 19 + [math.log(3.0)]
    assert paired_ratio(real)["ratio"] < 0.9, "median must hold against one spike"
    print(json.dumps({"self_check": "pass", "checks": 4}))
    return 0


def run(n: int, blocks: int, replicates: int) -> dict[str, object]:
    guard = BudgetGuard()
    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT))
    from friday_h0.benchmark import _generate_fixture

    guard.before_candidate()
    fixture = _generate_fixture(np, FIXTURE_SEED, shape=SHAPE)
    a = mx.array(fixture.a)
    gpu_started = time.perf_counter()
    mx.eval(a)
    mx.synchronize()
    guard.record_gpu(time.perf_counter() - gpu_started)
    rng = np.random.Generator(np.random.PCG64(CANDIDATE_POOL_SEED))
    operands = [
        mx.array(rng.uniform(-1.0, 1.0, (SHAPE, SHAPE)).astype(np.float16)) for _ in range(n)
    ]
    gpu_started = time.perf_counter()
    mx.eval(*operands)
    mx.synchronize()
    guard.record_gpu(time.perf_counter() - gpu_started)
    references = []
    for operand in operands:
        gpu_started = time.perf_counter()
        references.append(
            np.array(mx.matmul(a, operand), copy=False).astype(np.float32)
        )
        guard.record_gpu(time.perf_counter() - gpu_started)

    def serial() -> list[object]:
        produced = []
        for operand in operands:
            value = mx.matmul(a, operand)
            mx.eval(value)
            mx.synchronize()
            produced.append(value)
        return produced

    def batched() -> list[object]:
        produced = [mx.matmul(a, operand) for operand in operands]
        mx.eval(produced)
        mx.synchronize()
        return produced

    # Correctness gate: an execution-plan change may not alter one bit.
    for name, plan in (("serial", serial), ("batched", batched)):
        gpu_started = time.perf_counter()
        produced = plan()
        guard.record_gpu(time.perf_counter() - gpu_started)
        worst = max(
            float(np.abs(np.array(value, copy=False).astype(np.float32) - reference).max())
            for value, reference in zip(produced, references)
        )
        if worst != 0.0:
            raise SystemExit(f"{name} plan changed the result by {worst}; refusing to report timing")

    results = []
    collected: list[list[float]] = []
    for replicate in range(replicates):
        if replicate:
            guard.required_break()
        gpu_started = time.perf_counter()
        serial()
        batched()
        guard.record_gpu(time.perf_counter() - gpu_started)
        log_ratios: list[float] = []
        serial_ns_values: list[int] = []
        batched_ns_values: list[int] = []
        for _ in range(blocks):
            start = time.perf_counter_ns()
            serial()
            serial_ns = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            batched()
            batched_ns = time.perf_counter_ns() - start
            guard.record_gpu((serial_ns + batched_ns) / 1e9)
            log_ratios.append(math.log(batched_ns / serial_ns))
            serial_ns_values.append(serial_ns)
            batched_ns_values.append(batched_ns)
        summary = paired_ratio(log_ratios)
        summary.update(
            {
                "replicate": replicate,
                "serial_ms_per_op": statistics.median(serial_ns_values) / 1e6 / n,
                "batched_ms_per_op": statistics.median(batched_ns_values) / 1e6 / n,
                "clears_mde": clears_threshold(summary),
                "log_ratios": log_ratios,
                "serial_ns": serial_ns_values,
                "batched_ns": batched_ns_values,
            }
        )
        results.append(summary)
        collected.append(log_ratios)

    aggregate = hierarchical_bootstrap(collected, seed=BOOTSTRAP_SEED)
    aggregate["clears_mde"] = clears_threshold(aggregate)
    serial_per_op = statistics.median([entry["serial_ms_per_op"] for entry in results])
    batched_per_op = statistics.median([entry["batched_ms_per_op"] for entry in results])
    guard.finish_candidate()
    budget = guard.summary()
    return {
        "n_matmuls": n,
        "blocks_per_replicate": blocks,
        "replicates": results,
        "aggregate": aggregate,
        "effect_percent": 100.0 * (aggregate["ratio"] - 1.0),
        "saved_ms_per_matmul": serial_per_op - batched_per_op,
        "serial_ms_per_op": serial_per_op,
        "batched_ms_per_op": batched_per_op,
        "verdict": "effect_confirmed" if aggregate["clears_mde"] else "not_beyond_threshold",
        "mde": MDE,
        "gpu_work_seconds": budget["gpu_work_seconds"],
        "wall_seconds": budget["wall_seconds"],
        "budget": budget,
        "correctness": "byte_identical",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_dispatch_plan", allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args(argv)

    gated = release_gate(args, _self_check)
    if gated is not None:
        return gated
    if not 2 <= args.n <= 16:
        raise SystemExit("n must be between 2 and 16")
    if not 2 <= args.blocks <= 100:
        raise SystemExit("blocks must be between 2 and 100")
    if not 2 <= args.replicates <= 10:
        raise SystemExit("replicates must be between 2 and 10")

    power = require_ac_power()

    def operation() -> dict[str, object]:
        report = run(args.n, args.blocks, args.replicates)
        report["power_source"] = power
        return report

    report = run_persisted("dispatch", operation)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1))
    return 0 if report["aggregate"]["clears_mde"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
