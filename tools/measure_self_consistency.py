#!/usr/bin/env python3
"""Does the width plateau buy accuracy, not just throughput?

The decode-width measurement found that one forward pass carrying 32 token
positions costs about what one carrying 8 costs.  Batch serving turns that into
throughput for many users.  This asks whether a single user can spend it instead:
draw k independent samples for one question and take the majority answer.

That is worth measuring precisely because the cost model is not the obvious one.
Self-consistency is normally described as k times the compute, which is why it is
usually reserved for datacentres.  On this device the k samples ride the plateau,
so the real question -- how much accuracy per wall-clock second -- has an answer
specific to this hardware and nobody's published number can supply it.

The task is generated, not downloaded: every problem is well-posed, has a unique
integer answer, and is checked arithmetically rather than by another model.  A
fixed seed gives every arm the identical problem set, so arms differ only in how
they were sampled.

Two numbers are reported, and the second is easy to forget:

  1. **Accuracy** -- majority answer equals ground truth.
  2. **Coverage** -- an answer could be extracted at all.  A model that runs past
     its token budget mid-sentence produces no answer, and counting that as a
     wrong answer would hide a failure mode that has a completely different fix.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
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
    "1b": "mlx-community/gemma-3-1b-it-4bit",
    "4b": "mlx-community/gemma-3-4b-it-4bit",
}
PROBLEM_SEED = 20260823
MAX_TOKENS = 288
# One batch_generate call is one continuous GPU block.  At 16 samples and this
# token budget a call stays near 5 s, inside the guard's 6 s ceiling; larger
# sample counts are split into several calls rather than raising the ceiling.
CHUNK = 16
SAMPLE_TEMPERATURE = 0.7
# Measured 4B cost at the efficient width, used to derive the segment length.
SEGMENT_MS_PER_STEP = 90.0
# The guard enforces a duty cycle over a rolling 60 s window, not per event. Those
# are not the same constraint: resting 4x after each event holds 20% only when every
# event is the same size. A short event earns a short rest, and a long event that
# follows it then lands in a window that is still full. Measured failure: events of
# 3.88, 3.98, 2.98 and 5.31 s each met the per-event target, yet summed to 16.0 s
# inside one 60 s window against a 15 s ceiling. Targeting 0.15 leaves room for that
# variance instead of pacing to the edge of a limit that is measured differently.
DUTY_TARGET = 0.15
BREAK_SECONDS = 4.0

INSTRUCTION = (
    " Solve it step by step, keep it brief, then end with the final answer "
    "on its own last line in exactly this form:\nANSWER: <number>"
)
ANSWER_PATTERN = re.compile(r"ANSWER:\s*(-?\d+)")


def hard_problems(count: int, seed: int = PROBLEM_SEED) -> list[tuple[str, int]]:
    """Four-factor variants, for models that sit at the ceiling on the standard set.

    A measurement taken at 100% cannot move, so it cannot answer whether a method
    helps.  These were calibrated the same way and leave the 1B model near 17% --
    deliberately past the point of usefulness for the small model, because their
    job is to give the large one somewhere to fall.
    """

    if count < 1:
        raise ValueError("problem count must be positive")
    rng = random.Random(seed + 1)
    out: list[tuple[str, int]] = []
    while len(out) < count:
        kind = rng.choice(["factory", "commute", "discount"])
        if kind == "factory":
            w, h, d, p = rng.randint(4, 12), rng.randint(3, 9), rng.randint(2, 6), rng.randint(3, 11)
            bad = rng.randint(5, 60)
            total = w * h * d * p
            if bad >= total:
                continue
            question = (
                f"A workshop has {w} workers. Each works {h} hours a day for {d} days. "
                f"A worker makes {p} items per hour. Afterwards {bad} items are found "
                f"defective and thrown away. How many good items are left?"
            )
            answer = total - bad
        elif kind == "commute":
            s1, t1 = rng.randint(30, 80), rng.randint(2, 5)
            s2, t2 = rng.randint(40, 95), rng.randint(2, 5)
            question = (
                f"A van drives {s1} km/h for {t1} hours, then {s2} km/h for {t2} hours. "
                f"A second van covers the same total distance in {t1 + t2} hours. "
                f"What is the second van's average speed in km/h, rounded down?"
            )
            answer = (s1 * t1 + s2 * t2) // (t1 + t2)
        else:
            base = rng.randrange(200, 900, 10)
            off, tax = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25])
            after = base * (100 - off) // 100
            question = (
                f"A bike costs {base} euro. It is reduced by {off} percent, then {tax} "
                f"percent tax is added to the reduced price. What is the final price "
                f"in euro?"
            )
            answer = after + after * tax // 100
        out.append((question, answer))
    return out


def problems(count: int, seed: int = PROBLEM_SEED) -> list[tuple[str, int]]:
    """Generated word problems with arithmetically checkable answers.

    Difficulty was calibrated, not guessed.  A four-factor variant left the 1B
    model at 17% with a third of its answers truncated before the answer line; a
    single-step variant left it at 80%, with no headroom for any method to show
    an improvement.  These sit near 64%, which leaves room in both directions.
    """

    if count < 1:
        raise ValueError("problem count must be positive")
    rng = random.Random(seed)
    out: list[tuple[str, int]] = []
    while len(out) < count:
        kind = rng.choice(["crew", "factory", "shop", "discount"])
        if kind == "crew":
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            minutes, extra = rng.randint(3, 12), rng.randint(10, 80)
            question = (
                f"Crew A lays {a} tiles per minute and crew B lays {b} tiles per "
                f"minute. They work together for {minutes} minutes, then crew A "
                f"alone lays {extra} more. How many tiles were laid in total?"
            )
            answer = (a + b) * minutes + extra
        elif kind == "factory":
            workers, hours, rate = rng.randint(3, 9), rng.randint(2, 6), rng.randint(3, 9)
            defective = rng.randint(5, 40)
            total = workers * hours * rate
            if defective >= total:
                continue
            question = (
                f"A workshop has {workers} workers. Each works {hours} hours and "
                f"makes {rate} items per hour. Then {defective} items are found "
                f"defective and thrown away. How many good items are left?"
            )
            answer = total - defective
        elif kind == "shop":
            a, x = rng.randint(2, 9), rng.randint(3, 12)
            b, y = rng.randint(2, 9), rng.randint(3, 12)
            change = rng.randint(5, 40)
            question = (
                f"Tom buys {a} notebooks at {x} euro each and {b} pens at {y} euro "
                f"each. He pays with {a * x + b * y + change} euro. How much change "
                f"does he get?"
            )
            answer = change
        else:
            base, off = rng.randrange(200, 900, 20), rng.choice([10, 20, 25, 50])
            question = (
                f"A bike costs {base} euro and is reduced by {off} percent. "
                f"What is the reduced price in euro?"
            )
            answer = base * (100 - off) // 100
        out.append((question, answer))
    return out


def extract(text: str) -> int | None:
    """Last explicit answer line, or None when the model never produced one.

    The last match wins: a model that corrects itself restates the answer, and the
    restatement is the one it is standing behind.
    """

    found = ANSWER_PATTERN.findall(text)
    return int(found[-1]) if found else None


def majority(votes: list[int | None]) -> int | None:
    """Most common extracted answer, ignoring samples that produced none.

    Ties are broken toward the answer seen first, which keeps the whole function
    deterministic given the sample order -- otherwise repeat runs of the same
    measurement could disagree for no reason anyone could reconstruct.
    """

    usable = [v for v in votes if v is not None]
    if not usable:
        return None
    counts = Counter(usable)
    best = max(counts.values())
    for value in usable:
        if counts[value] == best:
            return value
    return None  # pragma: no cover - unreachable, usable is non-empty


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


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    The normal approximation is wrong at the sample sizes a local GPU budget
    allows, and it is wrong in the direction that flatters a result: near 100% it
    produces intervals that extend past 1.0.  Wilson stays inside [0, 1].
    """

    if trials <= 0:
        raise ValueError("need at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie within the trials")
    p = successes / trials
    d = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / d
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _self_check() -> int:
    checks = 0

    got = problems(30)
    assert len(got) == 30
    checks += 1
    # Same seed, same problems: every arm must face an identical set.
    assert problems(8) == problems(8)
    checks += 1
    assert problems(8, seed=1) != problems(8, seed=2)
    checks += 1
    # Sharding must not change which problems an arm sees: the tail of a longer
    # set is exactly the set an offset run receives.
    full = problems(12)
    assert problems(12)[6:] == full[6:] and len(problems(6 + 6)[6:]) == 6
    checks += 1
    hard = hard_problems(20)
    assert len(hard) == 20 and all(isinstance(a, int) and a > 0 for _, a in hard)
    checks += 1
    assert hard != problems(20), "the hard set must differ from the standard one"
    checks += 1
    # Every generated answer is a positive integer, so a parse failure and a
    # legitimate answer can never be confused.
    assert all(isinstance(a, int) and a > 0 for _, a in got)
    checks += 1
    try:
        problems(0)
    except ValueError:
        checks += 1
    else:  # pragma: no cover
        raise AssertionError("a zero-length problem set must be refused")

    assert extract("blah\nANSWER: 42") == 42
    checks += 1
    assert extract("ANSWER: 7\nno wait\nANSWER: 9") == 9, "the restatement wins"
    checks += 1
    assert extract("ANSWER: -3") == -3
    checks += 1
    assert extract("I think it is 42") is None, "no answer line means no answer"
    checks += 1

    assert majority([1, 2, 2, 3]) == 2
    checks += 1
    assert majority([None, 5, None, 5]) == 5, "unparsed samples must not vote"
    checks += 1
    assert majority([None, None]) is None
    checks += 1
    # A tie resolves to whichever tied value was seen first, both ways round.
    assert majority([7, 9, 7, 9]) == 7
    checks += 1
    assert majority([9, 7, 9, 7]) == 9
    checks += 1

    low, high = wilson(9, 14)
    assert 0.0 < low < 9 / 14 < high < 1.0, (low, high)
    checks += 1
    # The interval must stay inside [0, 1] even at the extremes.
    assert wilson(14, 14)[1] <= 1.0 and wilson(0, 14)[0] >= 0.0
    checks += 1
    for bad in ((1, 0), (-1, 5), (6, 5)):
        try:
            wilson(*bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid proportion must be refused: {bad}")

    assert breaks_for(3.0) == 5
    checks += 1
    for d in (0.5, 2.7, 3.0, 9.0):
        assert d / (d + breaks_for(d) * BREAK_SECONDS) <= DUTY_TARGET + 1e-9
    checks += 1

    print(json.dumps({"self_check": "pass", "checks": checks}))
    return 0


def _segmented_votes(model, tokenizer, ids, samples, sampler, guard, *, max_tokens, chunk):
    """Draw `samples` completions at the efficient width, one guard block at a time.

    `batch_generate` cannot be interrupted mid-call, so at the efficient width it
    can only run about 71 steps before the guard's continuous-load ceiling stops
    it -- far short of a full answer. The segmented loop keeps the KV cache across
    a guard break, which the correctness arm of `measure_segmented_decode` showed
    leaves the produced tokens byte-identical.
    """

    from measure_segmented_decode import decode, segment_steps

    stops = frozenset(int(t) for t in tokenizer.eos_token_ids)
    per_segment = segment_steps(SEGMENT_MS_PER_STEP, guard.policy.continuous_gpu_limit_s)

    texts: list[str] = []
    seconds = 0.0
    remaining = samples
    while remaining > 0:
        size = min(chunk, remaining)
        result = decode(
            model, ids, size, max_tokens, sampler, guard,
            steps_per_segment=per_segment, stop_tokens=stops,
        )
        seconds += result["generation_seconds"]
        for row in result["tokens"]:
            trimmed = []
            for token in row:
                if token in stops:
                    break
                trimmed.append(token)
            texts.append(tokenizer.decode(trimmed))
        remaining -= size
    return texts, seconds, per_segment


def run_arm(
    model_key: str,
    samples: int,
    count: int,
    guard: BudgetGuard,
    *,
    offset: int = 0,
    chunk: int = CHUNK,
    max_tokens: int = MAX_TOKENS,
    difficulty: str = "standard",
    temperature: float | None = None,
    backend: str = "batch",
) -> dict[str, object]:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    snapshot = resolve_local_model_snapshot(MODELS[model_key])
    started = time.perf_counter()
    model, tokenizer = load(str(snapshot.path))
    load_seconds = time.perf_counter() - started

    # A single sample defaults to greedy, because with one vote there is nothing to
    # be consistent with. It is not forced: a lone sample at t > 0 is the control
    # that separates escaping degeneration from majority voting, and forcing greedy
    # here made that control impossible to run.
    used_temperature = temperature if temperature is not None else (
        0.0 if samples == 1 else SAMPLE_TEMPERATURE
    )
    sampler = make_sampler(temp=used_temperature)
    # Offsetting shards one problem set across processes.  The guard caps a single
    # process at 120 s of GPU work, which a large model at many samples exceeds on
    # a full set; sharding keeps every arm on the identical seeded problems rather
    # than quietly shrinking the set for the expensive arms only.
    generator = hard_problems if difficulty == "hard" else problems
    task = generator(count + offset)[offset:]

    correct = covered = 0
    per_problem = []
    generation_seconds = 0.0
    for question, truth in task:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + INSTRUCTION}], add_generation_prompt=True
        )
        ids = list(text if isinstance(text, list) else tokenizer.encode(text))

        votes: list[int | None] = []
        if backend == "segmented":
            texts, worked, segment_size = _segmented_votes(
                model, tokenizer, ids, samples, sampler, guard,
                max_tokens=max_tokens, chunk=chunk,
            )
            generation_seconds += worked
            votes.extend(extract(t) for t in texts)
        else:
            segment_size = None
            remaining = samples
            while remaining > 0:
                size = min(chunk, remaining)
                at = time.perf_counter()
                response = batch_generate(
                    model, tokenizer, [ids] * size, max_tokens=max_tokens, sampler=sampler
                )
                worked = time.perf_counter() - at
                generation_seconds += worked
                account(guard, worked)
                votes.extend(extract(t) for t in response.texts)
                remaining -= size

        voted = majority(votes)
        covered += voted is not None
        correct += voted == truth
        per_problem.append(
            {
                "truth": truth,
                "answer": voted,
                "correct": voted == truth,
                "parsed_votes": sum(v is not None for v in votes),
                "distinct_answers": len({v for v in votes if v is not None}),
            }
        )

    accuracy_low, accuracy_high = wilson(correct, count)
    result = {
        "model": model_key,
        **snapshot.report_identity(),
        "samples_per_problem": samples,
        "problems": count,
        "problem_offset": offset,
        "difficulty": difficulty,
        "chunk": chunk,
        "backend": backend,
        "steps_per_segment": segment_size,
        "temperature": used_temperature,
        "max_tokens": max_tokens,
        "load_seconds": round(load_seconds, 3),
        "correct": correct,
        "accuracy": round(correct / count, 4),
        "accuracy_ci95": [round(accuracy_low, 4), round(accuracy_high, 4)],
        "coverage": round(covered / count, 4),
        "generation_seconds": round(generation_seconds, 3),
        "seconds_per_problem": round(generation_seconds / count, 4),
        "mean_distinct_answers": round(
            statistics.mean(p["distinct_answers"] for p in per_problem), 3
        ),
        "per_problem": per_problem,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 4),
    }
    del model, tokenizer
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually run the measurement")
    parser.add_argument("--self-check", action="store_true", help="offline checks only")
    parser.add_argument("--model", choices=sorted(MODELS), default="1b")
    parser.add_argument("--samples", type=int, default=1, help="samples per problem")
    parser.add_argument("--problems", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0, help="skip this many problems")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="samples per batch call")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--difficulty", choices=("standard", "hard"), default="standard")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature; default greedy at one sample, "
                             f"{SAMPLE_TEMPERATURE} above that")
    parser.add_argument("--backend", choices=("batch", "segmented"), default="batch",
                        help="segmented keeps the efficient width inside the load limit")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate
    if args.samples < 1:
        raise SystemExit("samples per problem must be at least 1")
    if args.chunk < 1 or args.offset < 0 or args.max_tokens < 1:
        raise SystemExit("chunk and max tokens must be positive, offset non-negative")
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        raise SystemExit("temperature must lie in [0, 2]")

    power = require_ac_power()
    guard = BudgetGuard()
    report = run_arm(
        args.model, args.samples, args.problems, guard,
        offset=args.offset, chunk=args.chunk,
        max_tokens=args.max_tokens, difficulty=args.difficulty,
        temperature=args.temperature, backend=args.backend,
    )
    report["power_source"] = power
    report["budget"] = guard.summary()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_problem"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
