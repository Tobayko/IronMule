#!/usr/bin/env python3
"""How much does one forward pass cost as a function of how many token positions
it carries -- and how much of that does the generation loop actually realise?

This is the measurement that decides which optimisations can pay on this device.
The roofline run established that generation is memory-bound: every weight is
read once per token.  The natural inference is that carrying k+1 positions
through one pass costs about what carrying one costs, since the weights are read
once either way.  Every multi-token scheme -- speculative decoding, tree
verification, batch serving -- rests on that inference.

It is only partly true here, and the shape of where it fails is the finding.
Cost is flat across one wide plateau and steps sharply between plateaus, so the
per-position cost depends on *which* width is chosen far more than on how many
positions are needed.  Widths in the pathological zone cost nearly a full pass
each; widths on the plateau are close to free.

Two arms, because headroom and delivery are different questions:

  1. **Forward pass.** Direct cost of one pass at each width, with a warm cache.
     This is the ceiling: no sampling, no detokenisation, no scheduling.
  2. **Generation loop.** ``batch_generate`` at each batch size, which is the
     only mechanism that fills the width with work that is correct by
     construction -- no draft model, no acceptance rate, nothing to diverge.

The gap between the two arms is unrealised headroom sitting in the software
layer, and reporting it is the point: a ceiling nobody reaches is not a result.

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
)

MODELS = {
    "4b": "mlx-community/gemma-3-4b-it-4bit",
    "1b": "mlx-community/gemma-3-1b-it-4bit",
}
PASS_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 32, 48, 64)
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
CONTEXT_TOKENS = 256
PASS_REPETITIONS = 12
# Two token counts, not one. Timing a single generation and dividing by its token
# count charges the prompt prefill to every step: at batch 32 that prefill is over
# two seconds, which over 24 tokens fabricates ~86 ms of per-step "overhead" that
# does not exist. The slope between two counts cancels it; the intercept reports it.
BATCH_TOKEN_COUNTS = (6, 24)
BATCH_REPETITIONS = 2

# The guard enforces a duty cycle over a rolling 60 s window, not per event. Those
# are not the same constraint: resting 4x after each event holds 20% only when every
# event is the same size. A short event earns a short rest, and a long event that
# follows it then lands in a window that is still full. Measured failure: events of
# 3.88, 3.98, 2.98 and 5.31 s each met the per-event target, yet summed to 16.0 s
# inside one 60 s window against a 15 s ceiling. Targeting 0.15 leaves room for that
# variance instead of pacing to the edge of a limit that is measured differently.
DUTY_TARGET = 0.15
BREAK_SECONDS = 4.0

TOPICS = (
    "a CPU cache line", "branch prediction", "a TLB", "memory prefetching",
    "SIMD vectorisation", "out-of-order execution", "store buffers", "NUMA locality",
)
PROMPT = (
    "Explain in a few sentences how {} works and why it matters for "
    "performance on modern hardware."
)


def breaks_for(seconds: float, duty_target: float = DUTY_TARGET) -> int:
    """Fixed guard breaks needed to hold the duty cycle at or under the target."""

    if seconds < 0:
        raise ValueError("worked seconds must be non-negative")
    if not 0.0 < duty_target < 1.0:
        raise ValueError("duty target must lie strictly between 0 and 1")
    needed = seconds * (1.0 - duty_target) / duty_target
    # Discards a floating-point remainder far below any real timing effect; see
    # the same rounding note in measure_speculative.py.
    return math.ceil(needed / BREAK_SECONDS - 1e-9)


def account(guard: BudgetGuard, seconds: float) -> None:
    """Charge one continuous block of GPU work, then rest in proportion to it.

    Each call is charged separately on purpose.  Timing a warmup and its
    repetitions as a single block reports a continuous load that never happened
    and trips the guard's 6 s ceiling on work that was actually well inside it.
    """

    guard.record_gpu(seconds)
    for _ in range(breaks_for(seconds)):
        guard.required_break()


def derive_policy(ms_by_width: dict[int, float], tolerance: float = 0.05) -> dict[str, object]:
    """Turn a measured cost curve into statements a dispatcher can act on.

    "Use the widest width" is true of any curve that keeps improving and tells a
    dispatcher nothing it did not already assume.  Two facts are worth reporting
    because neither is predictable from the shape of the model:

      * **regressions** -- widths where going *wider* made cost per position
        strictly worse than some narrower width already achieved.  These are the
        pathological zone: a dispatcher that lands there is paying more per token
        than it would with fewer positions, which no cost model would predict.
      * **free upgrades** -- the widest width whose *absolute* cost is within
        tolerance of this one.  Landing on width 8 and paying 5% more to get 32
        positions is the single cheapest win available, and it only exists because
        the curve has plateaus.

    The tolerance keeps near-ties out of both sets: widths within a few percent
    are not separable across thermal drift, and carving those up would be reading
    noise as structure.
    """

    if len(ms_by_width) < 2:
        raise ValueError("need at least two widths to derive a policy")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must lie in [0, 1)")

    per_position = {w: ms / w for w, ms in ms_by_width.items()}
    widths = sorted(ms_by_width)
    best = min(widths, key=lambda w: (per_position[w], w))

    regressions = []
    for index, width in enumerate(widths[1:], start=1):
        narrower = min(per_position[w] for w in widths[:index])
        if narrower < per_position[width] * (1.0 - tolerance):
            regressions.append(width)

    upgrades = {}
    for width in widths:
        budget = ms_by_width[width] * (1.0 + tolerance)
        reachable = [w for w in widths if w > width and ms_by_width[w] <= budget]
        if reachable:
            upgrades[width] = max(reachable)

    return {
        "best_width": best,
        "best_ms_per_position": round(per_position[best], 4),
        "speedup_vs_width_1": round(per_position[widths[0]] / per_position[best], 3),
        "regression_widths": regressions,
        "free_upgrades": upgrades,
        "tolerance": tolerance,
    }


def marginal_costs(ms_by_width: dict[int, float]) -> dict[str, float]:
    """Cost of each additional position between consecutive measured widths.

    A plateau shows up here as a marginal cost near zero, which is exactly the
    regime a multi-token scheme wants and the pathological zone is not.
    """

    widths = sorted(ms_by_width)
    if len(widths) < 2:
        raise ValueError("need at least two widths to form a marginal cost")
    out = {}
    for lo, hi in zip(widths, widths[1:]):
        out[f"{lo}->{hi}"] = round((ms_by_width[hi] - ms_by_width[lo]) / (hi - lo), 4)
    return out


def _self_check() -> int:
    checks = 0

    assert breaks_for(3.0) == 5, breaks_for(3.0)
    checks += 1
    assert breaks_for(3.1) == 5
    checks += 1
    assert breaks_for(0.0) == 0
    checks += 1
    for d in (0.5, 1.0, 2.7, 3.0, 9.0):
        rested = breaks_for(d) * BREAK_SECONDS
        assert d / (d + rested) <= DUTY_TARGET + 1e-9, d
    checks += 1
    for bad in ((-1.0, 0.2), (1.0, 0.0), (1.0, 1.0)):
        try:
            breaks_for(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid pacing input must be refused: {bad}")

    # A curve that keeps improving has no regression anywhere.
    improving = derive_policy({1: 10.0, 2: 12.0, 4: 16.0})
    assert improving["best_width"] == 4 and improving["regression_widths"] == [], improving
    checks += 1
    # Measured 4B numbers: width 6 costs 31.8 ms more than width 5 for one extra
    # position, so 6 is a regression even though it is wider.
    cliff = derive_policy({1: 13.015, 5: 36.94, 6: 68.771, 32: 76.593})
    assert cliff["best_width"] == 32, cliff
    assert cliff["regression_widths"] == [6], cliff
    checks += 1
    # Width 6 does not reach width 32 inside 5%: 76.593 / 68.771 is 1.114, and a
    # tolerance that swallowed an 11% jump would stop separating plateaus at all.
    assert 6 not in cliff["free_upgrades"], cliff["free_upgrades"]
    checks += 1
    # Measured widths that do sit on one plateau: 86.24 ms at width 14 reaches
    # 86.517 ms at width 32, so 14 buys 18 more positions for 0.3%.
    plateau = derive_policy({5: 42.004, 14: 86.24, 24: 88.839, 32: 86.517})
    assert plateau["free_upgrades"].get(14) == 32, plateau["free_upgrades"]
    assert plateau["free_upgrades"].get(24) == 32, plateau["free_upgrades"]
    assert 5 not in plateau["free_upgrades"], "a genuine step is not an upgrade"
    checks += 1
    # A curve with strictly proportional cost has no free upgrade at all.
    proportional = derive_policy({8: 80.0, 16: 160.0, 32: 320.0})
    assert proportional["regression_widths"] == [], proportional
    assert proportional["free_upgrades"] == {}, proportional
    checks += 1
    for bad in (({1: 1.0}, 0.05), ({1: 1.0, 2: 2.0}, 1.0), ({1: 1.0, 2: 2.0}, -0.1)):
        try:
            derive_policy(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid policy input must be refused: {bad}")

    # A perfectly flat plateau costs nothing per extra position.
    flat = marginal_costs({8: 80.0, 16: 80.0, 32: 80.0})
    assert flat == {"8->16": 0.0, "16->32": 0.0}, flat
    checks += 1
    # A linear region reports its slope.
    assert marginal_costs({1: 10.0, 3: 20.0}) == {"1->3": 5.0}
    checks += 1
    try:
        marginal_costs({1: 10.0})
    except ValueError:
        checks += 1
    else:  # pragma: no cover
        raise AssertionError("a single width cannot yield a marginal cost")

    # The slope must recover a known per-step cost even when a large fixed cost is
    # present: 2.0 s of prefill plus 0.1 s per step, sampled at 6 and 24 tokens.
    low_s, high_s = 2.0 + 0.1 * 6, 2.0 + 0.1 * 24
    recovered = (high_s - low_s) / (24 - 6)
    assert abs(recovered - 0.1) < 1e-12, recovered
    assert abs((low_s - recovered * 6) - 2.0) < 1e-12
    checks += 1

    class FakeGuard:
        breaks = 0
        charged: list[float] = []

        def record_gpu(self, seconds: float) -> None:
            self.charged.append(seconds)

        def required_break(self) -> None:
            self.breaks += 1

    fake = FakeGuard()
    account(fake, 3.0)
    assert fake.charged == [3.0] and fake.breaks == 5, (fake.charged, fake.breaks)
    checks += 1

    print(json.dumps({"self_check": "pass", "checks": checks}))
    return 0


def measure_forward_pass(model, guard: BudgetGuard) -> dict[str, object]:
    """Cost of a single warm-cache forward pass at each width."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    context = mx.array([[1] * CONTEXT_TOKENS])
    ms_by_width: dict[int, float] = {}
    for width in PASS_WIDTHS:
        cache = make_prompt_cache(model)
        primed = model(context, cache=cache)
        mx.eval(primed)
        mx.synchronize()
        step = mx.array([[7] * width])
        for _ in range(3):
            warm = model(step, cache=cache)
            mx.eval(warm)
        mx.synchronize()

        samples = []
        started = time.perf_counter()
        for _ in range(PASS_REPETITIONS):
            at = time.perf_counter_ns()
            out = model(step, cache=cache)
            mx.eval(out)
            mx.synchronize()
            samples.append(time.perf_counter_ns() - at)
        account(guard, time.perf_counter() - started)
        ms_by_width[width] = statistics.median(samples) / 1e6

    single = ms_by_width[PASS_WIDTHS[0]]
    rows = [
        {
            "width": width,
            "ms": round(ms, 3),
            "ms_per_position": round(ms / width, 4),
            "throughput_vs_width_1": round(single / (ms / width), 3),
        }
        for width, ms in ms_by_width.items()
    ]
    best = max(rows, key=lambda r: r["throughput_vs_width_1"])
    return {
        "widths": rows,
        "marginal_ms_per_position": marginal_costs(ms_by_width),
        "policy": derive_policy(ms_by_width),
        "best_width": best["width"],
        "best_throughput_vs_width_1": best["throughput_vs_width_1"],
    }


def measure_generation(model, tokenizer, guard: BudgetGuard) -> dict[str, object]:
    """Throughput actually delivered by the batched generation loop."""

    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)

    def prompts_for(count: int) -> list[list[int]]:
        built = []
        for index in range(count):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(TOPICS[index % len(TOPICS)])}],
                add_generation_prompt=True,
            )
            built.append(list(text if isinstance(text, list) else tokenizer.encode(text)))
        return built

    rows = []
    for size in BATCH_SIZES:
        prompts = prompts_for(size)

        def run(max_tokens: int) -> float:
            at = time.perf_counter()
            batch_generate(model, tokenizer, prompts, max_tokens=max_tokens, sampler=sampler)
            worked = time.perf_counter() - at
            account(guard, worked)
            return worked

        run(4)
        low, high = BATCH_TOKEN_COUNTS
        low_seconds = min(run(low) for _ in range(BATCH_REPETITIONS))
        high_seconds = min(run(high) for _ in range(BATCH_REPETITIONS))
        per_step = (high_seconds - low_seconds) / (high - low)
        if per_step <= 0:
            raise SystemExit("generation slope must be positive; measurement is unusable")
        rows.append(
            {
                "batch_size": size,
                "seconds_per_step": round(per_step, 6),
                "fixed_seconds": round(low_seconds - per_step * low, 4),
                "tokens_per_second": round(size / per_step, 2),
                "naive_tokens_per_second": round(size * high / high_seconds, 2),
            }
        )

    single = rows[0]["tokens_per_second"]
    for row in rows:
        row["throughput_vs_batch_1"] = round(row["tokens_per_second"] / single, 3)
    best = max(rows, key=lambda r: r["tokens_per_second"])
    return {
        "batches": rows,
        "token_counts": list(BATCH_TOKEN_COUNTS),
        "best_batch_size": best["batch_size"],
        "best_tokens_per_second": best["tokens_per_second"],
        "best_throughput_vs_batch_1": best["throughput_vs_batch_1"],
    }


def measure(guard: BudgetGuard, *, model_key: str = "4b") -> dict[str, object]:
    import mlx.core as mx
    from mlx_lm import load

    snapshot = resolve_local_model_snapshot(MODELS[model_key])
    started = time.perf_counter()
    model, tokenizer = load(str(snapshot.path))
    load_seconds = time.perf_counter() - started

    forward = measure_forward_pass(model, guard)
    generation = measure_generation(model, tokenizer, guard)

    # The single-stream path is the honest baseline for the delivered figure:
    # batch_generate at size 1 carries scheduling overhead a plain stream does
    # not, so comparing a batch only against itself flatters it.
    realised = generation["best_throughput_vs_batch_1"]
    headroom = forward["best_throughput_vs_width_1"]
    result = {
        "model": model_key,
        **snapshot.report_identity(),
        "load_seconds": round(load_seconds, 3),
        "context_tokens": CONTEXT_TOKENS,
        "forward_pass": forward,
        "generation": generation,
        "headroom_vs_realised": {
            "forward_pass_ceiling": headroom,
            "generation_loop_realised": realised,
            "unrealised_factor": round(headroom / realised, 3) if realised else None,
        },
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
    }
    del model, tokenizer
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually run the measurement")
    parser.add_argument("--self-check", action="store_true", help="offline checks only")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--model", choices=sorted(MODELS), default="4b",
                        help="the width curve is model-specific; each needs its own policy")
    args = parser.parse_args()

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate

    power = require_ac_power()
    guard = BudgetGuard()
    report = measure(guard, model_key=args.model)
    report["power_source"] = power
    report["budget"] = guard.summary()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
