#!/usr/bin/env python3
"""Use the efficient batch width without exceeding the continuous-load limit.

The width measurement found that a batch of 32 costs 2.61 ms per sample-step while
a batch of 2 costs 9.68 ms -- a factor of 3.71.  The guard's 6 s ceiling on
continuous GPU work then takes most of that back: at batch 32 a single generation
call may run only 71 steps before it trips, and a useful answer needs closer to
288.  Two runs in the previous session failed exactly there, correctly.

The conflict is not physical.  A KV cache is just state; pausing between steps
changes nothing about what the model computes next.  So the generation can be cut
into segments, each its own continuous block with a guard break after it, and the
efficient width kept throughout.

That claim has to be tested, not asserted, and the correctness arm is the reason
this file exists at all.  Under greedy sampling a segmented run must produce
*byte-identical* tokens to an unsegmented one; if pausing changed the output, the
speed comparison would be meaningless because the two arms would not be doing the
same thing.  The gate runs at a token count small enough that the unsegmented arm
still fits inside the guard's ceiling, so both arms can be measured honestly.

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
WIDE, NARROW = 32, 2
GENERATE_TOKENS = 240
CORRECTNESS_TOKENS = 48
CORRECTNESS_SEGMENT = 16
# Cost per prefill position at the efficient width, plus headroom. Prefill is
# charged in positions, not in tokens: a chunk of 98 tokens at batch 32 is 3136
# positions, and splitting only the sequence axis let exactly that overrun the
# continuous ceiling on a longer prompt.
PREFILL_MS_PER_POSITION = 2.8

# Measured per-step cost at the wide width, plus headroom.  The segment length is
# derived from this and the guard's own ceiling rather than being a constant: a
# hard-coded segment length would silently stop being safe on a different model.
ASSUMED_MS_PER_STEP = 90.0
SAFETY_FRACTION = 0.75

# The guard enforces a duty cycle over a rolling 60 s window, not per event. Those
# are not the same constraint: resting 4x after each event holds 20% only when every
# event is the same size. A short event earns a short rest, and a long event that
# follows it then lands in a window that is still full. Measured failure: events of
# 3.88, 3.98, 2.98 and 5.31 s each met the per-event target, yet summed to 16.0 s
# inside one 60 s window against a 15 s ceiling. Targeting 0.15 leaves room for that
# variance instead of pacing to the edge of a limit that is measured differently.
DUTY_TARGET = 0.15
BREAK_SECONDS = 4.0

PROMPT = "Explain in a few sentences how a CPU cache line works and why it matters."

# Widths past 32 were unreachable before the segmented loop existed: a single
# generate call at that width overran the continuous ceiling, so the earlier width
# policy stopped where the measurement stopped, not where the hardware did.
BATCH_SWEEP = (8, 16, 32, 48, 64, 96, 128, 192, 256)
SWEEP_TOKENS = 64
# Refuse a batch whose peak footprint would crowd the machine. Unified memory is
# shared with everything else running; filling it is not a measurement, it is an
# outage.
PEAK_MEMORY_CEILING_GB = 16.0


def segment_steps(
    ms_per_step: float, continuous_limit_s: float, safety: float = SAFETY_FRACTION
) -> int:
    """How many decode steps fit in one continuous block, with headroom.

    The headroom is not decoration.  Step cost varies with thermal state and with
    KV length, so a segment sized to exactly fill the ceiling would trip whenever
    the machine ran a little slower than when it was calibrated -- and tripping is
    a failed run, not a slow one.
    """

    if ms_per_step <= 0:
        raise ValueError("step cost must be positive")
    if continuous_limit_s <= 0:
        raise ValueError("continuous limit must be positive")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety fraction must lie in (0, 1]")
    steps = int(continuous_limit_s * safety * 1000.0 / ms_per_step)
    return max(1, steps)


def all_finished(columns, stop_tokens: frozenset[int]) -> bool:
    """Has every sequence emitted a stop token somewhere in what it has produced?

    Takes the token columns rather than a running flag so that the check can be
    made at a segment boundary without having tracked state through the inner loop
    -- the inner loop must stay free of host reads.
    """

    if not stop_tokens:
        return False
    rows = [list(row) for row in zip(*(column.tolist() for column in columns))]
    return all(any(int(t[0]) in stop_tokens for t in row) for row in rows)


def calibrate_step_ms(model, cache, y, sampler, guard, probes: int = 4) -> float:
    """Measure what one decode step costs right now, at this batch and cache length.

    The segment length has to come from this rather than from a constant. A constant
    calibrated at batch 32 (about 90 ms per step) would size a 50-step segment at
    batch 128, where a step costs closer to 300 ms -- a 15 s block against a 6 s
    ceiling. The probe itself is short enough to stay inside the ceiling at any batch
    that fits in memory at all.
    """

    import mlx.core as mx

    if probes < 1:
        raise ValueError("need at least one probe step")

    # Warm first, then time. The first steps after a prefill carry one-time work --
    # allocation and kernel setup at this shape -- and folding that into a four-step
    # average overstated the cost 3.8x at batch 96, which sized the segments far
    # smaller than they needed to be. Conservative, but wrong, and reported.
    at = time.perf_counter()
    probe = y
    for _ in range(2):
        logits = model(probe, cache=cache)
        probe = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(probe)
    mx.synchronize()
    warm_seconds = time.perf_counter() - at

    at = time.perf_counter()
    for _ in range(probes):
        logits = model(probe, cache=cache)
        probe = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(probe)
    mx.synchronize()
    worked = time.perf_counter() - at
    account(guard, warm_seconds + worked)
    return worked / probes * 1000.0


def prefill_chunk(
    batch: int,
    ms_per_position: float,
    continuous_limit_s: float,
    safety: float = SAFETY_FRACTION,
) -> int:
    """Prompt tokens per prefill block, given how many rows are being filled at once.

    The block the GPU actually runs is ``batch * chunk`` positions wide, so the chunk
    has to shrink as the batch grows. Deriving it from batch and measured cost is the
    same discipline the decode segment uses, and for the same reason: a constant that
    is safe at batch 2 silently is not at batch 32.
    """

    if batch < 1:
        raise ValueError("batch must be at least one")
    if ms_per_position <= 0:
        raise ValueError("position cost must be positive")
    if continuous_limit_s <= 0:
        raise ValueError("continuous limit must be positive")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety fraction must lie in (0, 1]")
    positions = continuous_limit_s * safety * 1000.0 / ms_per_position
    return max(1, int(positions / batch))


def breaks_for(seconds: float, duty_target: float = DUTY_TARGET) -> int:
    if seconds < 0:
        raise ValueError("worked seconds must be non-negative")
    if not 0.0 < duty_target < 1.0:
        raise ValueError("duty target must lie strictly between 0 and 1")
    needed = seconds * (1.0 - duty_target) / duty_target
    return math.ceil(needed / BREAK_SECONDS - 1e-9)


def account(guard: BudgetGuard, seconds: float) -> None:
    guard.record_gpu(seconds)
    for _ in range(breaks_for(seconds)):
        guard.required_break()


def _self_check() -> int:
    checks = 0

    # 90 ms per step against a 6 s ceiling at 75% headroom: 4500 ms of budget, so
    # 50 steps.  The whole point is that this is derived, not assumed.
    assert segment_steps(90.0, 6.0) == 50, segment_steps(90.0, 6.0)
    checks += 1
    # A cheaper step buys proportionally more steps per segment.
    assert segment_steps(19.36, 6.0) == 232, segment_steps(19.36, 6.0)
    checks += 1
    # A step costing more than a whole segment still yields one step, never zero:
    # a segment of zero steps would never terminate.
    assert segment_steps(9000.0, 6.0) == 1
    checks += 1
    # Every derived segment stays inside the ceiling it was derived from.
    for cost in (12.0, 19.36, 83.46, 90.0, 165.0):
        assert segment_steps(cost, 6.0) * cost / 1000.0 <= 6.0, cost
    checks += 1
    for bad in ((0.0, 6.0, 0.75), (90.0, 0.0, 0.75), (90.0, 6.0, 0.0), (90.0, 6.0, 1.5)):
        try:
            segment_steps(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid segment input must be refused: {bad}")

    class Col:
        def __init__(self, values): self._v = values
        def tolist(self): return [[v] for v in self._v]

    # Two sequences, three steps. Sequence 0 stops at step 1, sequence 1 at step 2.
    cols = [Col([5, 5]), Col([9, 5]), Col([5, 9])]
    assert all_finished(cols, frozenset({9})) is True
    checks += 1
    # Sequence 1 never stops, so the batch is not finished.
    assert all_finished([Col([9, 5]), Col([9, 5])], frozenset({9})) is False
    checks += 1
    # No stop tokens configured means the loop must never end early by accident.
    assert all_finished(cols, frozenset()) is False
    checks += 1

    # Uneven event sizes are the case that broke a real run, so the invariant is
    # checked against a rolling window, not against each event on its own.
    events = [3.884, 3.980, 2.978, 5.314]
    window, gpu = 0.0, 0.0
    for e in events:
        gpu += e
        window += e + breaks_for(e) * BREAK_SECONDS
    assert gpu / window <= 0.25, (gpu, window)
    checks += 1
    # The same sequence paced at the old 0.20 target does exceed a 15 s ceiling in
    # a 60 s window; keeping the case here stops the target drifting back up.
    old = sum(e + math.ceil(e * 4.0 / BREAK_SECONDS - 1e-9) * BREAK_SECONDS for e in events)
    assert sum(events) / old > 0.15, (sum(events), old)
    checks += 1

    # 6 s at 75% headroom and 2.8 ms per position is 1607 positions: 50 tokens at
    # batch 32, 803 at batch 2. The constant it replaced was 256 for both.
    # A step cost measured at batch 128 must not be sized with a batch-32 constant:
    # 50 steps of 300 ms is 15 s against a 6 s ceiling, 15 steps of 300 ms is 4.5 s.
    assert segment_steps(300.0, 6.0) == 15, segment_steps(300.0, 6.0)
    checks += 1

    assert prefill_chunk(32, 2.8, 6.0) == 50, prefill_chunk(32, 2.8, 6.0)
    checks += 1
    assert prefill_chunk(2, 2.8, 6.0) == 803, prefill_chunk(2, 2.8, 6.0)
    checks += 1
    # Every derived chunk keeps its block inside the ceiling it came from.
    for b in (1, 2, 8, 32, 64, 256):
        block = prefill_chunk(b, 2.8, 6.0) * b * 2.8 / 1000.0
        assert block <= 6.0, (b, block)
    checks += 1
    # A batch so wide that even one token overruns still yields one token, never
    # zero: a chunk of zero would not advance.
    assert prefill_chunk(100000, 2.8, 6.0) == 1
    checks += 1
    for bad in ((0, 2.8, 6.0), (32, 0.0, 6.0), (32, 2.8, 0.0)):
        try:
            prefill_chunk(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid prefill input must be refused: {bad}")

    assert breaks_for(3.0) == 5
    checks += 1
    for d in (0.5, 2.7, 3.0, 9.0):
        assert d / (d + breaks_for(d) * BREAK_SECONDS) <= DUTY_TARGET + 1e-9
    checks += 1

    print(json.dumps({"self_check": "pass", "checks": checks}))
    return 0


def decode(
    model,
    prompt_ids: list[int],
    batch: int,
    max_tokens: int,
    sampler,
    guard: BudgetGuard,
    *,
    steps_per_segment: int | None = None,
    stop_tokens: frozenset[int] = frozenset(),
) -> dict[str, object]:
    """Batched greedy decode, charged to the guard one segment at a time.

    ``steps_per_segment`` equal to ``max_tokens`` reproduces an ordinary
    unsegmented run, which is what makes the correctness arm a fair comparison.

    Stopping is checked once per segment rather than once per step.  Checking every
    step would need a host synchronisation every step, which is the one thing that
    breaks the pipelining this whole loop exists to exploit; the segment boundary
    already pays for a synchronisation, so the check there is free.  The cost is
    overrunning the true stop by at most one segment.
    """

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    prompt = mx.array([prompt_ids] * batch)

    # The prefill obeys the same ceiling as the decode; a long prompt at a wide
    # batch is exactly as capable of overrunning it as a long generation is.
    generated_seconds = 0.0
    logits = None
    chunk_len = prefill_chunk(
        batch, PREFILL_MS_PER_POSITION, guard.policy.continuous_gpu_limit_s
    )
    for start in range(0, prompt.shape[1], chunk_len):
        piece = prompt[:, start : start + chunk_len]
        at = time.perf_counter()
        logits = model(piece, cache=cache)
        mx.eval(logits)
        mx.synchronize()
        worked = time.perf_counter() - at
        account(guard, worked)

    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
    mx.eval(y)
    columns = [y]

    # A caller that names a segment length gets it -- the correctness arm needs to
    # pin one. Otherwise it is derived from what a step actually costs here.
    if steps_per_segment is None:
        measured = calibrate_step_ms(model, cache, y, sampler, guard)
        steps_per_segment = segment_steps(measured, guard.policy.continuous_gpu_limit_s)
    else:
        measured = None

    produced = 1
    segments = 0
    stopped_early = False
    while produced < max_tokens:
        count = min(steps_per_segment, max_tokens - produced)
        at = time.perf_counter()
        for _ in range(count):
            logits = model(y, cache=cache)
            y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
            columns.append(y)
        mx.eval(columns[-1])
        mx.synchronize()
        worked = time.perf_counter() - at
        generated_seconds += worked
        account(guard, worked)
        produced += count
        segments += 1
        if stop_tokens and all_finished(columns, stop_tokens):
            stopped_early = True
            break

    tokens = mx.concatenate(columns, axis=1)
    mx.eval(tokens)
    return {
        "tokens": tokens.tolist(),
        "generated": produced,
        "segments": segments,
        "generation_seconds": generated_seconds,
        "steps_per_segment": steps_per_segment,
        "calibrated_ms_per_step": None if measured is None else round(measured, 3),
        "stopped_early": stopped_early,
    }


def sweep(model, tokenizer, ids, sampler, guard: BudgetGuard) -> dict[str, object]:
    """Sample-token throughput across batch widths, each paced by its own measurement."""

    import mlx.core as mx

    stops = frozenset(int(t) for t in tokenizer.eos_token_ids)
    rows = []
    for batch in BATCH_SWEEP:
        mx.reset_peak_memory()
        result = decode(
            model, ids, batch, SWEEP_TOKENS, sampler, guard, stop_tokens=stops
        )
        peak = mx.get_peak_memory() / 1e9
        produced = result["generated"]
        rate = batch * produced / result["generation_seconds"]
        rows.append(
            {
                "batch": batch,
                "generated": produced,
                "segments": result["segments"],
                "steps_per_segment": result["steps_per_segment"],
                "calibrated_ms_per_step": result["calibrated_ms_per_step"],
                "ms_per_sample_token": round(
                    result["generation_seconds"] / batch / produced * 1000.0, 4
                ),
                "sample_tokens_per_second": round(rate, 2),
                "peak_memory_gb": round(peak, 3),
            }
        )
        if peak > PEAK_MEMORY_CEILING_GB:
            rows[-1]["stopped_sweep"] = "peak memory ceiling reached"
            break

    base = rows[0]["sample_tokens_per_second"]
    for row in rows:
        row["vs_smallest_batch"] = round(row["sample_tokens_per_second"] / base, 3)
    best = max(rows, key=lambda r: r["sample_tokens_per_second"])
    return {
        "rows": rows,
        "tokens_per_row": SWEEP_TOKENS,
        "best_batch": best["batch"],
        "best_sample_tokens_per_second": best["sample_tokens_per_second"],
        "peak_memory_ceiling_gb": PEAK_MEMORY_CEILING_GB,
    }


def measure(guard: BudgetGuard, *, do_sweep: bool = False,
            model_key: str = "4b") -> dict[str, object]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    snapshot = resolve_local_model_snapshot(MODELS[model_key])
    model, tokenizer = load(str(snapshot.path))
    sampler = make_sampler(temp=0.0)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    ids = list(text if isinstance(text, list) else tokenizer.encode(text))

    # Correctness first.  A speed result from a loop that changes the answer is
    # not a speed result, so nothing else runs until this passes.
    whole = decode(
        model, ids, WIDE, CORRECTNESS_TOKENS, sampler, guard,
        steps_per_segment=CORRECTNESS_TOKENS,
    )
    split = decode(
        model, ids, WIDE, CORRECTNESS_TOKENS, sampler, guard,
        steps_per_segment=CORRECTNESS_SEGMENT,
    )
    identical = whole["tokens"] == split["tokens"]
    if not identical:
        raise SystemExit("segmented decode changed the output; refusing to report timings")

    limit = guard.policy.continuous_gpu_limit_s
    per_segment = segment_steps(ASSUMED_MS_PER_STEP, limit)
    if do_sweep:
        swept = sweep(model, tokenizer, ids, sampler, guard)
        result = {
            "model": model_key,
            **snapshot.report_identity(),
            "correctness": {
                "segmented_matches_unsegmented": identical,
                "tokens_compared": CORRECTNESS_TOKENS,
                "segment_size": CORRECTNESS_SEGMENT,
            },
            "sweep": swept,
        }
        del model, tokenizer
        mx.clear_cache()
        return result

    # Both timed arms generate the same content from the same prompt under greedy
    # sampling, so enabling early stop for both keeps the comparison fair while
    # measuring the loop as it would actually be used.
    stops = frozenset(int(t) for t in tokenizer.eos_token_ids)
    wide = decode(
        model, ids, WIDE, GENERATE_TOKENS, sampler, guard,
        steps_per_segment=per_segment, stop_tokens=stops,
    )
    narrow = decode(
        model, ids, NARROW, GENERATE_TOKENS, sampler, guard,
        steps_per_segment=GENERATE_TOKENS, stop_tokens=stops,
    )

    # Normalise by tokens actually produced: early stop can end the two arms at
    # different segment boundaries, and dividing by the cap would credit whichever
    # arm happened to overrun less.
    wide_per_sample = wide["generation_seconds"] / WIDE / wide["generated"]
    narrow_per_sample = narrow["generation_seconds"] / NARROW / narrow["generated"]
    result = {
        **snapshot.report_identity(),
        "correctness": {
            "segmented_matches_unsegmented": identical,
            "tokens_compared": CORRECTNESS_TOKENS,
            "segment_size": CORRECTNESS_SEGMENT,
            "segments_used": split["segments"],
        },
        "continuous_limit_seconds": limit,
        "steps_per_segment": per_segment,
        "stop_tokens": sorted(stops),
        "wide": {
            "batch": WIDE,
            "segments": wide["segments"],
            "generated": wide["generated"],
            "stopped_early": wide["stopped_early"],
            "generation_seconds": round(wide["generation_seconds"], 3),
            "seconds_per_sample_token": round(wide_per_sample, 6),
            "sample_tokens_per_second": round(
                WIDE * wide["generated"] / wide["generation_seconds"], 2),
        },
        "narrow": {
            "batch": NARROW,
            "segments": narrow["segments"],
            "generated": narrow["generated"],
            "stopped_early": narrow["stopped_early"],
            "generation_seconds": round(narrow["generation_seconds"], 3),
            "seconds_per_sample_token": round(narrow_per_sample, 6),
            "sample_tokens_per_second": round(
                NARROW * narrow["generated"] / narrow["generation_seconds"], 2),
        },
        "wide_advantage": round(narrow_per_sample / wide_per_sample, 3),
        "generate_tokens": GENERATE_TOKENS,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
    }
    del model, tokenizer
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", choices=sorted(MODELS), default="4b")
    parser.add_argument("--sweep", action="store_true",
                        help="scan batch widths instead of the wide/narrow comparison")
    args = parser.parse_args()

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate

    power = require_ac_power()
    guard = BudgetGuard()
    report = measure(guard, do_sweep=args.sweep, model_key=args.model)
    report["power_source"] = power
    report["budget"] = guard.summary()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
