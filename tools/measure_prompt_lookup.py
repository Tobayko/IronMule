#!/usr/bin/env python3
"""Speculation without a draft model, and without touching the weights.

Draft-model speculative decoding was measured at 0.560x here: the 1B costs 0.46 of
a 4B step, so it has to be right almost every time to pay, and it was right 39% of
the time. That killed the technique with that draft. It does not kill the technique.

The cost of a drafted token is what decides this, and a draft does not have to come
from a model. Much of what a model writes it has already read -- an identifier, a
path, a quoted line, a repeated structure -- so the last few tokens can be looked up
in the context and whatever followed them there proposed as the continuation. The
draft is a string search on the CPU and costs nothing measurable.

Against the measured width curve that changes the break-even acceptance from
impossible to ordinary: one drafted token needs 0.47 acceptance to pay, three need
0.72. And when no match exists the step falls back to an ordinary one, so the
downside is bounded at zero rather than at the 0.44 the draft model lost.

The output is byte-identical to greedy decoding by construction -- a drafted token
is kept only where it equals what the model would have produced -- and the run
verifies that rather than assuming it. Nothing is quantised, nothing is approximated,
the model keeps everything it knew.

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
DRAFT_LENGTHS = (0, 1, 2, 3, 4)
NGRAM = 3
GENERATE_TOKENS = 96
DUTY_TARGET = 0.15
BREAK_SECONDS = 4.0


def find_continuation(tokens: list[int], ngram: int, draft_length: int) -> list[int]:
    """Longest-recent match: what followed the last `ngram` tokens the last time.

    Searched from the end backwards, because the most recent occurrence is the one
    most likely to continue the same way -- an identifier repeated three lines ago
    beats the same identifier in a different function forty lines up.

    Returns fewer than `draft_length` tokens near the end of the context, and none
    at all when nothing matches. Proposing nothing is a normal decode step, which is
    why a miss costs nothing.
    """

    if ngram < 1 or draft_length < 0:
        raise ValueError("ngram must be positive and draft length non-negative")
    if draft_length == 0 or len(tokens) <= ngram:
        return []
    needle = tokens[-ngram:]
    # Stop before the trailing occurrence, which is the needle itself.
    for start in range(len(tokens) - ngram - 1, -1, -1):
        if tokens[start : start + ngram] == needle:
            proposal = tokens[start + ngram : start + ngram + draft_length]
            if proposal:
                return list(proposal)
    return []


def accepted_prefix(drafted: list[int], produced: list[int]) -> int:
    """How many drafted tokens the model would have produced anyway.

    Acceptance stops at the first disagreement. Keeping a later match after an
    earlier miss would splice a continuation onto a different prefix and silently
    change the answer, which is the one thing this must not do.
    """

    count = 0
    for want, got in zip(drafted, produced):
        if want != got:
            break
        count += 1
    return count


def expected_speedup(acceptance: float, draft_length: int, verify_cost: float) -> float:
    """Tokens per pass over the cost of that pass, both relative to a plain step."""

    if not 0.0 <= acceptance <= 1.0:
        raise ValueError("acceptance must lie in [0, 1]")
    if draft_length < 0 or verify_cost <= 0:
        raise ValueError("draft length must be non-negative and cost positive")
    if draft_length == 0:
        return 1.0
    yielded = sum(acceptance**i for i in range(draft_length + 1))
    return yielded / verify_cost


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

    # The continuation of the most recent match wins, not the first one.
    assert find_continuation([1, 2, 3, 9, 9, 1, 2, 3], 3, 2) == [9, 9], \
        find_continuation([1, 2, 3, 9, 9, 1, 2, 3], 3, 2)
    checks += 1
    # Two candidate matches: take the later one.
    assert find_continuation([1, 2, 3, 7, 0, 0, 1, 2, 3, 8, 0, 1, 2, 3], 3, 1) == [8]
    checks += 1
    # No match at all proposes nothing, which is an ordinary decode step.
    assert find_continuation([5, 6, 7, 8], 3, 3) == []
    checks += 1
    # A match close to the end yields a short proposal rather than a padded one:
    # only three tokens follow it, so asking for five returns three.
    assert find_continuation([1, 2, 3, 1, 2, 3], 3, 5) == [1, 2, 3], \
        find_continuation([1, 2, 3, 1, 2, 3], 3, 5)
    checks += 1
    # Asking for no draft, or having too little history, proposes nothing.
    assert find_continuation([1, 2, 3, 4], 3, 0) == []
    assert find_continuation([1, 2], 3, 2) == []
    checks += 1
    for bad in ((0, 2), (3, -1)):
        try:
            find_continuation([1, 2, 3, 4], *bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid lookup must be refused: {bad}")

    assert accepted_prefix([1, 2, 3], [1, 2, 3]) == 3
    assert accepted_prefix([1, 2, 3], [1, 9, 3]) == 1, "a later match must not count"
    assert accepted_prefix([1, 2, 3], [9, 2, 3]) == 0
    assert accepted_prefix([], [1]) == 0
    checks += 1

    # No draft is the baseline; a perfect draft yields k+1 tokens for the pass cost.
    assert expected_speedup(0.9, 0, 1.0) == 1.0
    assert abs(expected_speedup(1.0, 3, 2.59) - 4 / 2.59) < 1e-9
    checks += 1
    # Measured break-even: three drafted tokens cost 2.59 passes, so acceptance has
    # to reach about 0.72 before speculation pays at all.
    assert expected_speedup(0.72, 3, 2.59) > 1.0
    assert expected_speedup(0.68, 3, 2.59) < 1.0
    checks += 1
    # One drafted token is a far lower bar, which is the point of a free draft.
    assert expected_speedup(0.50, 1, 1.47) > 1.0
    checks += 1
    for bad in ((-0.1, 1, 1.5), (1.1, 1, 1.5), (0.5, -1, 1.5), (0.5, 1, 0.0)):
        try:
            expected_speedup(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid speedup input must be refused: {bad}")

    assert breaks_for(3.0) == 5
    checks += 1

    print(json.dumps({"self_check": "pass", "checks": checks}))
    return 0


def generate(model, sampler, prompt_ids, *, draft_length, max_tokens, guard, ngram=NGRAM):
    """Greedy decode, drafting from the context when a match exists."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    cache = make_prompt_cache(model)
    context = list(prompt_ids)

    started = time.perf_counter()
    logits = model(mx.array([context]), cache=cache)
    token = int(sampler(logits[:, -1, :].astype(mx.float32))[0])
    mx.eval(token)
    context.append(token)
    generated = [token]

    steps = drafted_total = accepted_total = 0
    while len(generated) < max_tokens:
        drafted = find_continuation(context, ngram, draft_length)
        window = [context[-1]] + drafted
        logits = model(mx.array([window]), cache=cache)
        picks = sampler(logits[0].astype(mx.float32)).tolist()
        mx.eval(logits)

        keep = accepted_prefix(drafted, picks[:-1]) if drafted else 0
        # picks[i] is what the model produces after position i, so the accepted run
        # plus one bonus token is exactly picks[:keep + 1].
        new = picks[: keep + 1]
        steps += 1
        drafted_total += len(drafted)
        accepted_total += keep

        # The pass wrote len(window) positions into the cache; everything past the
        # accepted run plus its bonus token never happened and has to be rolled back.
        surplus = len(window) - (keep + 1)
        if surplus > 0:
            trim_prompt_cache(cache, surplus)

        for t in new:
            context.append(int(t))
            generated.append(int(t))
            if len(generated) >= max_tokens:
                break

    mx.eval(mx.array(generated))
    mx.synchronize()
    worked = time.perf_counter() - started
    account(guard, worked)
    return {
        "tokens": generated[:max_tokens],
        "seconds": worked,
        "steps": steps,
        "drafted": drafted_total,
        "accepted": accepted_total,
        "acceptance": (accepted_total / drafted_total) if drafted_total else None,
        "tokens_per_step": len(generated[:max_tokens]) / steps if steps else None,
    }


def measure(model_key: str, prompt: str, guard: BudgetGuard, *,
            ngram: int = NGRAM) -> dict[str, object]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    snapshot = resolve_local_model_snapshot(MODELS[model_key])
    model, tokenizer = load(str(snapshot.path))
    sampler = make_sampler(temp=0.0)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    ids = list(text if isinstance(text, list) else tokenizer.encode(text))

    generate(model, sampler, ids, draft_length=0, max_tokens=8, guard=guard, ngram=ngram)

    arms = []
    baseline_tokens = None
    baseline_seconds = None
    for draft_length in DRAFT_LENGTHS:
        run = generate(
            model, sampler, ids,
            draft_length=draft_length, max_tokens=GENERATE_TOKENS, guard=guard,
            ngram=ngram,
        )
        if baseline_tokens is None:
            baseline_tokens, baseline_seconds = run["tokens"], run["seconds"]
        identical = run["tokens"] == baseline_tokens
        arms.append({
            "draft_length": draft_length,
            "seconds": round(run["seconds"], 4),
            "steps": run["steps"],
            "tokens_per_step": round(run["tokens_per_step"], 4) if run["tokens_per_step"] else None,
            "acceptance": None if run["acceptance"] is None else round(run["acceptance"], 4),
            "drafted": run["drafted"],
            "accepted": run["accepted"],
            "speedup": round(baseline_seconds / run["seconds"], 4),
            "identical_to_greedy": identical,
        })

    best = max(arms, key=lambda a: a["speedup"])
    result = {
        "model": model_key,
        **snapshot.report_identity(),
        "ngram": ngram,
        "generate_tokens": GENERATE_TOKENS,
        "prompt_tokens": len(ids),
        "arms": arms,
        "all_identical_to_greedy": all(a["identical_to_greedy"] for a in arms),
        "best_draft_length": best["draft_length"],
        "best_speedup": best["speedup"],
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
    }
    del model, tokenizer
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--model", choices=sorted(MODELS), default="4b")
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--ngram", type=int, default=NGRAM,
                        help="lookup window: longer matches less often but more exactly")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    else:
        raise SystemExit("a prompt is required: pass --prompt or --prompt-file")

    power = require_ac_power()
    guard = BudgetGuard()
    if args.ngram < 1:
        raise SystemExit("the lookup window must be at least one token")
    report = measure(args.model, prompt, guard, ngram=args.ngram)
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
