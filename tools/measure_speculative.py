#!/usr/bin/env python3
"""Can a small draft model break the memory-bandwidth wall the roofline found?

The roofline run classified both local Gemma models as memory-bound: generation
reads every weight once per token, so the clock is set by bytes moved, not by
arithmetic.  Every optimisation this project has measured so far -- batched
dispatch (-12%), a fused Residual+RMSNorm Metal kernel (+1.9%, below its gate) --
attacks the arithmetic side.  Amdahl's law caps that whole family well under 2x.

Speculative decoding attacks the bytes instead.  A cheap draft model proposes k
tokens; the target model verifies all k+1 positions in **one** forward pass.  The
target's weights are therefore read once per k+1 tokens rather than once per
token, which is the only lever that moves the binding constraint.

Two things are measured, and the second matters more than the first:

  1. **Speed.** Paired, interleaved blocks against the same target model, so
     thermal drift hits both arms equally.
  2. **Identity.** Under greedy sampling the accepted-token rule makes
     speculative decoding produce the *same token sequence* as the target model
     alone.  That is a testable claim, not a hope: the arms must agree token for
     token, or the run is invalid.  A speedup that changes the answer would be a
     different model, not a faster one.

The acceptance rate is recorded because it, not the token count, is what governs
the achievable speedup -- see `predict_speedup`.

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

TARGET_MODEL = "mlx-community/gemma-3-4b-it-4bit"
DRAFT_MODEL = "mlx-community/gemma-3-1b-it-4bit"

# Draft lengths to sweep.  0 is the baseline arm: target model, no draft.
DRAFT_LENGTHS = (0, 2, 3, 4, 6)
GENERATE_TOKENS = 64
BLOCKS = 3
WARMUP_TOKENS = 4

# Rest is derived from the generation that was just paid for, not from a guessed
# constant.  A fixed two-break pause was tried first and tripped the guard's
# rolling duty-cycle limit: the arms differ in length by design, so no single
# constant can hold for all of them.  Resting in proportion to measured work is
# the only pacing that stays correct when the work changes.
# The guard enforces a duty cycle over a rolling 60 s window, not per event. Those
# are not the same constraint: resting 4x after each event holds 20% only when every
# event is the same size. A short event earns a short rest, and a long event that
# follows it then lands in a window that is still full. Measured failure: events of
# 3.88, 3.98, 2.98 and 5.31 s each met the per-event target, yet summed to 16.0 s
# inside one 60 s window against a 15 s ceiling. Targeting 0.15 leaves room for that
# variance instead of pacing to the edge of a limit that is measured differently.
DUTY_TARGET = 0.15
BREAK_SECONDS = 4.0

PROMPT = (
    "Write a short technical explanation of how a CPU cache line works, "
    "why false sharing hurts multithreaded code, and how to avoid it."
)


def predict_speedup(alpha: float, draft_length: int, draft_cost_ratio: float) -> float:
    """Expected speedup of speculative decoding over plain autoregressive decoding.

    With per-token acceptance probability ``alpha`` and ``k`` drafted tokens, a
    verification step emits ``i`` drafted tokens with probability
    ``alpha**i * (1 - alpha)`` plus one token the target always contributes, so
    the expected yield is the geometric sum ``sum(alpha**i for i in 0..k)``.

    The cost of that step is one target pass plus ``k`` draft passes.  The target
    pass is charged as 1 regardless of ``k``: verifying k+1 positions is a single
    read of the target weights, and under a memory-bound regime that read is the
    cost.  That approximation is exactly what the roofline measurement licenses,
    and it is the reason the whole technique works.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("acceptance rate must lie in [0, 1]")
    if draft_length < 0:
        raise ValueError("draft length must be non-negative")
    if draft_cost_ratio < 0.0:
        raise ValueError("draft cost ratio must be non-negative")
    if draft_length == 0:
        return 1.0
    expected_tokens = sum(alpha**i for i in range(draft_length + 1))
    return expected_tokens / (1.0 + draft_length * draft_cost_ratio)


def best_draft_length(alpha: float, draft_cost_ratio: float, limit: int = 16) -> int:
    """The k that maximises predicted speedup, or 0 when drafting cannot pay."""

    scored = [
        (predict_speedup(alpha, k, draft_cost_ratio), -k) for k in range(0, limit + 1)
    ]
    return -max(scored)[1]


def breaks_for(seconds: float, duty_target: float = DUTY_TARGET) -> int:
    """How many fixed guard breaks keep the duty cycle at or under the target.

    Working for ``d`` seconds at a duty cycle of ``t`` requires ``d * (1 - t) / t``
    seconds of rest.  The guard only offers a fixed 4 s verified break, so the
    requirement is rounded up to whole breaks -- rounding down would defeat the
    purpose of asking.
    """

    if seconds < 0:
        raise ValueError("worked seconds must be non-negative")
    if not 0.0 < duty_target < 1.0:
        raise ValueError("duty target must lie strictly between 0 and 1")
    needed = seconds * (1.0 - duty_target) / duty_target
    # The ratio can land a hair above a whole number in binary floating point,
    # which would buy an extra break to cover a few femtoseconds.  The tolerance is far
    # below any real timing effect, and the 0.20 target already sits well inside
    # the guard's own 0.25 limit, so the discarded remainder cannot matter.
    return math.ceil(needed / BREAK_SECONDS - 1e-9)


def rest(guard: BudgetGuard, seconds: float) -> None:
    """Rest in proportion to the GPU work just recorded."""

    for _ in range(breaks_for(seconds)):
        guard.required_break()


def summarise(samples: list[float]) -> dict[str, float]:
    """Median plus spread; the median is the project's established estimator."""

    if not samples:
        raise ValueError("no samples to summarise")
    median = statistics.median(samples)
    return {
        "median": median,
        "min": min(samples),
        "max": max(samples),
        "samples": list(samples),
    }


def _self_check() -> int:
    """Offline checks of the speedup model and pacing; no GPU, no model."""

    checks = 0

    # A draft that is never accepted is pure overhead: strictly slower.
    assert predict_speedup(0.0, 4, 0.5) == 1.0 / 3.0, predict_speedup(0.0, 4, 0.5)
    checks += 1
    # A draft that is always accepted yields k+1 tokens for 1 + k*c cost.
    assert abs(predict_speedup(1.0, 4, 0.0) - 5.0) < 1e-12
    checks += 1
    # A free draft (cost ratio 0) makes speedup equal to the expected yield.
    assert abs(predict_speedup(0.5, 2, 0.0) - 1.75) < 1e-12
    checks += 1
    # No draft is the baseline by definition.
    assert predict_speedup(0.9, 0, 0.5) == 1.0
    checks += 1
    # Speedup rises with acceptance, holding k and cost fixed.
    rising = [predict_speedup(a / 10, 4, 0.4) for a in range(0, 11)]
    assert all(x < y for x, y in zip(rising, rising[1:])), rising
    checks += 1
    # An expensive draft cannot pay off at low acceptance: optimum is "don't".
    assert best_draft_length(0.3, 0.5) == 0, best_draft_length(0.3, 0.5)
    checks += 1
    # A free draft always pays, and wants to draft as far as it is allowed.
    assert best_draft_length(0.9, 0.0, limit=8) == 8
    checks += 1
    # A high-acceptance, cheap draft has an interior optimum.
    interior = best_draft_length(0.8, 0.1, limit=32)
    assert 0 < interior < 32, interior
    checks += 1

    for bad in ((-0.1, 2, 0.5), (1.1, 2, 0.5), (0.5, -1, 0.5), (0.5, 2, -0.1)):
        try:
            predict_speedup(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid input must be refused: {bad}")

    class FakeGuard:
        breaks = 0

        def required_break(self) -> None:
            self.breaks += 1

    # 3 s of work at a 15% duty cycle needs 17 s of rest: five breaks.
    assert breaks_for(3.0) == 5, breaks_for(3.0)
    checks += 1
    # Rounding is upward: a longer job never gets fewer breaks than a shorter one.
    assert breaks_for(3.1) == 5, breaks_for(3.1)
    checks += 1
    assert breaks_for(0.0) == 0
    checks += 1
    # The invariant that actually matters: the resulting duty cycle stays inside
    # the guard's policy limit, with the same tolerance the rounding uses.
    for d in (0.5, 1.0, 2.7, 3.0, 9.0):
        rested = breaks_for(d) * BREAK_SECONDS
        assert d / (d + rested) <= DUTY_TARGET + 1e-9, d
    checks += 1

    fake = FakeGuard()
    rest(fake, 3.0)
    assert fake.breaks == 5, fake.breaks
    checks += 1
    for bad in ((-1.0, 0.2), (1.0, 0.0), (1.0, 1.0)):
        try:
            breaks_for(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid pacing input must be refused: {bad}")

    assert summarise([3.0, 1.0, 2.0])["median"] == 2.0
    checks += 1

    print(json.dumps({"self_check": "pass", "checks": checks}))
    return 0


def generate_once(
    model,
    draft,
    tokenizer,
    text,
    *,
    draft_length: int,
    max_tokens: int,
    sampler,
):
    """One generation; returns tokens, draft-accepted count and generation time.

    Prefill is excluded from the timing: it is one shared pass over the prompt in
    both arms and would dilute the per-token effect this run is measuring.
    """

    from mlx_lm.generate import stream_generate

    kwargs = {}
    if draft_length:
        kwargs["draft_model"] = draft
        kwargs["num_draft_tokens"] = draft_length

    tokens: list[int] = []
    accepted = 0
    first_ns = None
    last_ns = None
    started = time.perf_counter_ns()
    for response in stream_generate(
        model, tokenizer, text, max_tokens=max_tokens, sampler=sampler, **kwargs
    ):
        last_ns = time.perf_counter_ns()
        if first_ns is None:
            first_ns = last_ns
        tokens.append(int(response.token))
        accepted += bool(response.from_draft)
    if first_ns is None or last_ns is None or len(tokens) < 2:
        raise SystemExit("generation produced too few tokens to measure")
    return {
        "tokens": tokens,
        "draft_accepted": accepted,
        "generation_seconds": (last_ns - first_ns) / 1e9,
        "generated": len(tokens),
        "total_seconds": (last_ns - started) / 1e9,
    }


def measure(guard: BudgetGuard) -> dict[str, object]:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    target_snapshot = resolve_local_model_snapshot(TARGET_MODEL)
    draft_snapshot = resolve_local_model_snapshot(DRAFT_MODEL)

    load_started = time.perf_counter()
    model, tokenizer = load(str(target_snapshot.path))
    draft, draft_tokenizer = load(str(draft_snapshot.path))
    load_seconds = time.perf_counter() - load_started

    # Speculative decoding is only valid when both models score the same token
    # ids.  A silent mismatch would produce fluent nonsense, so it is refused.
    if tokenizer.vocab_size != draft_tokenizer.vocab_size:
        raise SystemExit("draft and target tokenizers disagree; refusing to speculate")

    target_bytes = sum(p.size * p.dtype.size for _, p in tree_flatten(model.parameters()))
    draft_bytes = sum(p.size * p.dtype.size for _, p in tree_flatten(draft.parameters()))

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.0)

    for _ in range(2):
        started = time.perf_counter()
        generate_once(
            model, draft, tokenizer, text,
            draft_length=0, max_tokens=WARMUP_TOKENS, sampler=sampler,
        )
        worked = time.perf_counter() - started
        guard.record_gpu(worked)
        rest(guard, worked)

    arms: dict[int, list[dict[str, object]]] = {k: [] for k in DRAFT_LENGTHS}
    for block in range(BLOCKS):
        # Alternate arm order between blocks so that position in the block cannot
        # be confounded with the arm itself.
        order = DRAFT_LENGTHS if block % 2 == 0 else tuple(reversed(DRAFT_LENGTHS))
        for draft_length in order:
            started = time.perf_counter()
            result = generate_once(
                model, draft, tokenizer, text,
                draft_length=draft_length,
                max_tokens=GENERATE_TOKENS,
                sampler=sampler,
            )
            worked = time.perf_counter() - started
            guard.record_gpu(worked)
            rest(guard, worked)
            arms[draft_length].append(result)

    baseline_tokens = arms[0][0]["tokens"]
    report_arms = []
    identical = True
    for draft_length in DRAFT_LENGTHS:
        runs = arms[draft_length]
        per_token = [r["generation_seconds"] / (r["generated"] - 1) for r in runs]
        matches = all(r["tokens"] == baseline_tokens for r in runs)
        identical = identical and matches
        accepted = sum(int(r["draft_accepted"]) for r in runs)
        produced = sum(int(r["generated"]) for r in runs)
        steps = produced - accepted
        report_arms.append(
            {
                "draft_length": draft_length,
                "seconds_per_token": summarise(per_token),
                "tokens_per_second": round(1.0 / statistics.median(per_token), 2),
                "draft_accepted": accepted,
                "generated": produced,
                "target_forward_passes": steps,
                "tokens_per_target_pass": round(produced / steps, 4) if steps else None,
                "acceptance_rate": (
                    round(accepted / (steps * draft_length), 4)
                    if draft_length and steps
                    else None
                ),
                "token_identical_to_baseline": matches,
            }
        )

    baseline_per_token = statistics.median(
        [r["generation_seconds"] / (r["generated"] - 1) for r in arms[0]]
    )
    for arm in report_arms:
        arm["measured_speedup"] = round(
            baseline_per_token / arm["seconds_per_token"]["median"], 4
        )

    best = max(report_arms, key=lambda a: a["measured_speedup"])
    result = {
        "target": target_snapshot.report_identity(),
        "draft": draft_snapshot.report_identity(),
        "target_weight_bytes": target_bytes,
        "draft_weight_bytes": draft_bytes,
        "load_seconds": round(load_seconds, 3),
        "prompt": PROMPT,
        "generate_tokens": GENERATE_TOKENS,
        "blocks": BLOCKS,
        "sampler": "greedy_temp_0",
        "arms": report_arms,
        "all_arms_token_identical": identical,
        "best_draft_length": best["draft_length"],
        "best_measured_speedup": best["measured_speedup"],
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
    }
    del model, draft, tokenizer, draft_tokenizer
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually run the measurement")
    parser.add_argument("--self-check", action="store_true", help="offline checks only")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args()

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate

    power = require_ac_power()
    guard = BudgetGuard()
    report = measure(guard)
    report["power_source"] = power
    report["budget"] = guard.summary()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
