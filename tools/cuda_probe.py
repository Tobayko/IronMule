"""HW3 and TB1 on one CUDA card: what a decode step costs part by part, which of the engine's optimisation
knobs and their combinations keep every token, and whether true tensor batching is exact (BACKLOG HW3, TB1).

Fixed before data (BACKLOG HW3 and TB1, 2026-10-29):

  model      mlx-community/Qwen3-1.7B-bf16 at revision 9cd66928, greedy, the engine without a tuned profile.
  factorial  the engine's knobs `fuse_projections`, `compiled_fixed_cache`, `fused_argmax`, `readback_every`
             (1 or 4) and `speculate_k` (0 or 3), all 32 combinations, through `Engine.generate`, on six
             prompts (three short questions, three behind about 1,000 tokens of context; chat template), 64
             tokens each. One untimed pass, then five timed rounds; in each round the baseline runs first and
             last (an A/A pair) and the other 31 in an order rotated by the round. Fused combinations run on a
             second copy of the model whose projections are fused once.
  exact      a combination's logical tokens equal the baseline's on every prompt, in every pass.
  cost       decode milliseconds per token; per (prompt, round) the ratio to that round's mean baseline; the
             geometric mean, with a 2,000-resample bootstrap (seed 0) over the pairs.
  noise      the largest distance from 1 of the rounds' A/A ratios (geometric mean over prompts).
  combos     main effects and two-factor interactions of the five knobs on log decode time, so a
             combination's value is measured rather than multiplied from single gains.
  anatomy    mlx-lm's own decode loop after a short and a long prompt, synchronous against pipelined (the
             next step built while the card runs this one); attention at the model's shapes, masked as
             `FixedKVCache` masks it against unmasked on keys cut to the context, at 1 and 8 rows and 64,
             1,024 and 2,048 tokens; every linear layer of block 0 and the output projection at one row, for
             the bandwidth floor.
  TB1        kernel level: those layers at M = 1, 2, 3, 4, 8, 16, 32 rows, each row compared bit for bit with
             the same row alone; model level: 16 distinct rows of 64 and of 1,024 tokens, each prefilled alone
             with mlx-lm's cache, then decoded 32 steps alone and stacked at B = 2, 4, 8, 16.
  decisions  `analyse`. Every gain counts, wherever it shows and however small, once the measurement resolves
             it (maintainer decision, 2026-10-29): no minimum effect size, only the noise or a confidence bound.

HW4 (`--plan hw4`, BACKLOG HW4, fixed 2026-10-29 before its data): every combination of the knobs that change the
arithmetic, plus `head_skip_prefill`, compared bit for bit with the baseline (prefill logits and state, every decode
step fed the baseline's tokens, the final state, speculation's verify windows); `head_skip_prefill` x
`fuse_projections` x HW3's decode winner, plus `prefill_into_fixed`; every class-A decode set at widths 1, 2 and 4
on the grouped path; compile cost per new body; the prefill's compute floor. Its decisions and RL7's first two kill
rules are in `analyse_hw4`.

The report keeps digests of token ids and arrays, counts, times and logit differences, never prompt or answer
text. To run it on Kaggle (T4 x2; it uses the first card):

    python tools/cuda_probe.py --write-kernel kernel --owner <user> --commit <sha>
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/ironmule-cuda-probe -p results
    python tools/cuda_probe.py --analyse results/cuda_probe_report.json
    python tools/cuda_probe.py --write-kernel kernel4 --owner <user> --commit <sha> --plan hw4 \
        --decode-winner '{"compiled_fixed_cache": true}'
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

MODEL = "mlx-community/Qwen3-1.7B-bf16"
REVISION = "9cd6692855d3e06772228e9a962b2606359b2d24"  # the revision RL3-H to LD1-V served
PROMPTS = ("Explain how a refrigerator keeps food cold.", "What causes the seasons on Earth?",
           "Describe how a bicycle gear system works.", "Why is the sky blue during the day?",
           "Summarize the water cycle in a few sentences.", "How does a compiler differ from an interpreter?",
           "What is the difference between weather and climate?", "Explain what a hash table is.",
           "How do vaccines train the immune system?", "Why do metals conduct electricity?",
           "Describe the life cycle of a star like the Sun.", "What is photosynthesis?",
           "How does a binary search work?", "Explain the greenhouse effect.",
           "What does a CPU cache do?", "How do airplanes stay in the air?")
CONTEXT = "Background notes, not needed for the answer. " + " ".join(PROMPTS * 8)  # as in rl2_headroom
FACTORS = ("fuse_projections", "compiled_fixed_cache", "fused_argmax", "readback_every", "speculate_k")
LEVELS = {"fuse_projections": (False, True), "compiled_fixed_cache": (False, True),
          "fused_argmax": (False, True), "readback_every": (1, 4), "speculate_k": (0, 3)}
PROMPT_SET = ((0, False), (1, False), (2, False), (3, True), (4, True), (5, True))  # (question, long context)
TOKENS = 64
ROUNDS = 5
RESAMPLES = 2000
ROWS = (1, 2, 3, 4, 8, 16, 32)
BATCHES = (1, 2, 4, 8, 16)
BATCH_LENGTHS = (64, 1024)
BATCH_STEPS = 32
ATTENTION_LENGTHS = (64, 1024, 2048)
ATTENTION_ROWS = (1, 8)
REPEATS = 20
OVERLAP_PAIRS = 5      # (16): synchronous and pipelined loops alternate this often per context
REPOSITORY = "https://github.com/Tobayko/IronMule"
# HW4 (BACKLOG HW4, fixed 2026-10-29 before its data)
CAPTURE_FACTORS = ("fuse_projections", "compiled_fixed_cache", "fused_argmax", "speculate_k", "head_skip_prefill")
CAPTURE_LEVELS = {**LEVELS, "head_skip_prefill": (False, True)}
SPECULATION_AT = (0, 16, 32)       # decode positions whose k+1 verify window is compared with sequential steps
WIDTHS_HW4 = (1, 2, 4)
PREFILL_ROWS = (256, 1024)         # prefill matmul rows for the compute floor
# KB1-0 (BACKLOG KB1, fixed 2026-10-29 before its data)
KB1_ROWS = (1, 2, 4, 8)
KB1_SAMPLES = 64                   # real activation rows per layer, from one 64-token forward
GROUPED_LEVER = 8 / (19.8 + 7 * 13.4) * 19.8  # RL6's fitted grouped batch-1: a width-8 round is 1.39x cheaper per token
# KB1-1 (BACKLOG KB1, fixed 2026-10-29 before its data)
KB1M_PROMPTS = 8                   # distinct requests, alternately short and behind ~1,000 tokens of context
KB1M_STEPS = 64
KB1M_STACKS = (3, 8)               # an uneven group and a full one
LONG_PREFIX, LONG_SCORED, LONG_CHUNKS = 1024, 128, 8  # the gate's long-offset variant, so FA2 is scored far out


# ---------------------------------------------------------------- the plan

def combos() -> list[tuple[int, ...]]:
    return list(itertools.product((0, 1), repeat=len(FACTORS)))


def combo_name(bits: tuple[int, ...]) -> str:
    return "+".join(f for f, b in zip(FACTORS, bits) if b) or "baseline"


def knobs_for(bits: tuple[int, ...]) -> dict[str, Any]:
    return {f: LEVELS[f][b] for f, b in zip(FACTORS, bits)}


def round_order(round_: int) -> list[tuple[int, ...]]:
    """The baseline, the other 31 rotated by the round, the baseline again."""
    others = [bits for bits in combos() if any(bits)]
    shift = (round_ * 7) % len(others)
    return [combos()[0], *others[shift:], *others[:shift], combos()[0]]


def _digest(tokens: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()


def _first_difference(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _median_time(fn: Callable[[], Any], repeats: int) -> float:
    import mlx.core as mx

    mx.eval(fn())  # warm: compile and allocate outside the clock
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def _paired_times(first: Callable[[], Any], second: Callable[[], Any], repeats: int) -> tuple[list[float], list[float]]:
    """Both calls timed alternately, so drift lands on both."""
    import mlx.core as mx

    mx.eval(first(), second())
    a, b = [], []
    for _ in range(repeats):
        for fn, times in ((first, a), (second, b)):
            started = time.perf_counter()
            mx.eval(fn())
            times.append(time.perf_counter() - started)
    return a, b


def chat_ids(tokenizer: Any, question: int, long: bool, limit: int | None = None) -> list[int]:
    prompt = (CONTEXT + "\n\n" if long else "") + PROMPTS[question]
    ids = list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                             tokenize=True))
    return ids if limit is None else ids[-limit:]


def row_ids(tokenizer: Any, row: int, length: int) -> list[int]:
    """Row `row` of a stack: its own question first, then the shared context, cut to `length` tokens."""
    text = PROMPTS[row % len(PROMPTS)] + " " + CONTEXT
    ids = list(tokenizer.encode(text))
    while len(ids) < length:
        ids += ids
    return ids[:length]


# ---------------------------------------------------------------- measurements

def linear_layers(model: Any) -> list[tuple[str, Callable[[Any], Any], int, int]]:
    """(name, call, input width, weight bytes) for each linear layer of block 0 and the output projection."""
    from ironmule.runtime import _project, _trunk

    trunk = _trunk(model)
    block = trunk.layers[0]
    out = []
    for owner, names in ((block.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                         (block.mlp, ("gate_proj", "up_proj", "down_proj"))):
        for name in names:
            module = getattr(owner, name, None)
            if module is not None:
                out.append((name, module, int(module.weight.shape[1]), int(module.weight.nbytes)))
    embed = trunk.embed_tokens.weight
    head = getattr(model, "lm_head", None)
    weight = head.weight if head is not None and not getattr(model.args, "tie_word_embeddings", False) else embed
    out.append(("output", lambda x: _project(model, x), int(weight.shape[1]), int(weight.nbytes)))
    return out


def probe_linear(model: Any, rows: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
    """TB1 kernel level and the bandwidth floor: is each row of an M-row product bit-identical to the row alone?"""
    import mlx.core as mx

    results = []
    for name, call, width, weight_bytes in linear_layers(model):
        x = mx.random.normal((max(rows), 1, width), key=mx.random.key(0)).astype(mx.bfloat16)
        alone = [call(x[i:i + 1]) for i in range(max(rows))]
        mx.eval(*alone)
        entry: dict[str, Any] = {"layer": name, "input": width, "weight_bytes": weight_bytes, "rows": {}}
        for m in rows:
            y = call(x[:m])
            mx.eval(y)
            equal = [bool(mx.array_equal(y[i:i + 1], alone[i]).item()) for i in range(m)]
            difference = max(float(mx.max(mx.abs(y[i:i + 1].astype(mx.float32) - alone[i].astype(mx.float32))).item())
                             for i in range(m))
            seconds = _median_time(lambda: call(x[:m]), repeats)
            entry["rows"][str(m)] = {"identical_rows": sum(equal), "max_difference": difference, "seconds": seconds,
                                     "gb_per_s": weight_bytes / seconds / 1e9}
        results.append(entry)
    return results


def probe_attention(model: Any, lengths: tuple[int, ...], rows: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
    """(6): one decode step's attention, masked as `FixedKVCache` masks it, against unmasked on keys cut to the
    context (the fused vector path X1 found disabled by the mask)."""
    import mlx.core as mx

    args = model.args
    heads, kv_heads = args.num_attention_heads, args.num_key_value_heads
    dim = getattr(args, "head_dim", None) or args.hidden_size // heads
    scale = dim ** -0.5
    results = []
    for length, batch in itertools.product(lengths, rows):
        capacity = ((length + TOKENS + 63) // 64) * 64
        key = mx.random.key(length * 31 + batch)
        q = mx.random.normal((batch, heads, 1, dim), key=key).astype(mx.bfloat16)
        k = mx.random.normal((batch, kv_heads, capacity, dim), key=mx.random.key(1)).astype(mx.bfloat16)
        v = mx.random.normal((batch, kv_heads, capacity, dim), key=mx.random.key(2)).astype(mx.bfloat16)
        mask = (mx.arange(capacity) < length)[None, None, None, :]

        def masked():
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        def cut():
            return mx.fast.scaled_dot_product_attention(q, k[:, :, :length], v[:, :, :length], scale=scale)

        a, b = masked(), cut()
        mx.eval(a, b)
        masked_times, cut_times = _paired_times(masked, cut, repeats)
        results.append({"length": length, "rows": batch, "capacity": capacity,
                        "masked_seconds": statistics.median(masked_times), "cut_seconds": statistics.median(cut_times),
                        "saved_seconds": [m - c for m, c in zip(masked_times, cut_times)],
                        "identical": bool(mx.array_equal(a, b).item()),
                        "max_difference": float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())})
    return results


def _prefill(model: Any, ids: list[int]) -> tuple[list[tuple[Any, Any]], Any]:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=cache)
    first = mx.argmax(logits[:, -1, :], axis=-1)
    states = [c.state for c in cache]
    mx.eval(first, states)
    return states, first


def _restore(model: Any, snapshots: list[list[tuple[Any, Any]]]) -> list[Any]:
    """A cache holding the given rows' prefilled states, stacked along the batch axis. The arrays are only
    read: the first update reallocates, so a snapshot can be restored again."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    caches = make_prompt_cache(model)
    for layer, cache in enumerate(caches):
        if len(snapshots) == 1:
            cache.state = snapshots[0][layer]
        else:
            cache.state = (mx.concatenate([s[layer][0] for s in snapshots], axis=0),
                           mx.concatenate([s[layer][1] for s in snapshots], axis=0))
    return caches


def _decode_rows(model: Any, caches: list[Any], first: Any, steps: int) -> tuple[list[list[int]], Any, list[float]]:
    import mlx.core as mx

    y, tokens, times, first_logits = first, [first], [], None
    for step in range(steps):
        started = time.perf_counter()
        logits = model(y[:, None], cache=caches)[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        mx.eval(y, logits) if step == 0 else mx.eval(y)
        times.append(time.perf_counter() - started)
        if step == 0:
            first_logits = logits.astype(mx.float32)
        tokens.append(y)
    rows = mx.stack(tokens, axis=1).tolist()
    return rows, first_logits, times


def probe_batching(model: Any, tokenizer: Any, lengths: tuple[int, ...], batches: tuple[int, ...],
                   steps: int) -> list[dict[str, Any]]:
    """TB1 model level: rows of equal length decoded alone and stacked; only the row count differs."""
    import mlx.core as mx

    results = []
    for length in lengths:
        snapshots, firsts = zip(*(_prefill(model, row_ids(tokenizer, row, length)) for row in range(max(batches))))
        alone = [_decode_rows(model, _restore(model, [snapshots[r]]), firsts[r], steps) for r in range(max(batches))]
        step_alone = statistics.median(t for _, _, times in alone for t in times[2:] or times)
        for batch in batches:
            if batch == 1:
                results.append({"length": length, "batch": 1, "seconds_per_step": step_alone,
                                "equal_tokens": max(batches), "identical_logits": max(batches),
                                "max_logit_difference": 0.0, "first_difference": []})
                continue
            rows, logits, times = _decode_rows(model, _restore(model, list(snapshots[:batch])),
                                               mx.concatenate(list(firsts[:batch])), steps)
            equal = identical = 0
            difference, first_differences = 0.0, []
            for r in range(batch):
                own_rows, own_logits, _ = alone[r]
                where = _first_difference(rows[r], own_rows[0])
                equal += where is None
                if where is not None:
                    first_differences.append(where)
                identical += bool(mx.array_equal(logits[r:r + 1], own_logits).item())
                difference = max(difference, float(mx.max(mx.abs(logits[r:r + 1] - own_logits)).item()))
            results.append({"length": length, "batch": batch,
                            "seconds_per_step": statistics.median(times[2:] or times),
                            "equal_tokens": equal, "identical_logits": identical,
                            "max_logit_difference": difference, "first_difference": first_differences})
    return results


def probe_overlap(model: Any, tokenizer: Any, steps: int, limit: int | None = None,
                  pairs: int = OVERLAP_PAIRS) -> list[dict[str, Any]]:
    """(16): mlx-lm's decode loop, waiting for each token before building the next step, against building the
    next step while the card runs this one, as `mlx_lm.generate` does."""
    import mlx.core as mx

    def loop(snapshot: Any, first: Any, pipelined: bool) -> tuple[float, list[int]]:
        caches, y, out = _restore(model, [snapshot]), first, []

        def step(token: Any) -> Any:
            return mx.argmax(model(token[:, None], cache=caches)[:, -1, :], axis=-1)

        started = time.perf_counter()
        if pipelined:
            mx.async_eval(y)
            for _ in range(steps):
                following = step(y)
                mx.async_eval(following)
                out.append(int(y.item()))
                y = following
            out.append(int(y.item()))
        else:
            out.append(int(y.item()))
            for _ in range(steps):
                y = step(y)
                mx.eval(y)
                out.append(int(y.item()))
        return (time.perf_counter() - started) / steps, out

    results = []
    for question, long in (PROMPT_SET[0], PROMPT_SET[3]):
        snapshot, first = _prefill(model, chat_ids(tokenizer, question, long, limit))
        loop(snapshot, first, False)  # warm
        sync, piped, same = [], [], True
        for _ in range(pairs):
            seconds, sync_tokens = loop(snapshot, first, False)
            sync.append(seconds)
            seconds, piped_tokens = loop(snapshot, first, True)
            piped.append(seconds)
            same = same and sync_tokens == piped_tokens
        results.append({"long": long, "context": int(snapshot[0][0].shape[2]),
                        "sync_seconds_per_step": statistics.median(sync),
                        "pipelined_seconds_per_step": statistics.median(piped),
                        "saved": [1 - b / a for a, b in zip(sync, piped)], "same_tokens": same})
    return results


def probe_factorial(models: tuple[Any, Any], tokenizer: Any, rounds: int, tokens: int,
                    limit: int | None = None, on_run: Callable[[dict[str, Any]], None] | None = None
                    ) -> dict[str, Any]:
    """The 32 knob combinations through the product engine: one untimed pass (round -1), then `rounds`."""
    from dataclasses import replace

    from ironmule import fast, runtime

    # One compiled body per capacity and width for every prompt, so no timed run recompiles one.
    runtime._BODY_CACHE = max(runtime._BODY_CACHE, 4 * len(PROMPT_SET))
    plain, fused = models
    engines: dict[tuple[int, ...], Any] = {}
    errors: dict[str, str] = {}
    try:
        fast.fuse_projections(fused)
    except Exception as exc:  # noqa: BLE001 - recorded; the fused half of the factorial is then missing
        errors["fuse_projections"] = repr(exc)
    for bits in combos():
        if bits[0] and "fuse_projections" in errors:
            continue
        knobs = runtime.Knobs(**knobs_for(bits))
        # The fused copy is already fused once; an engine asked to fuse it again would find nothing to rewrite.
        engines[bits] = runtime.Engine(fused if bits[0] else plain, tokenizer, replace(knobs, fuse_projections=False))
    eos = tuple(sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]))
    prompts = [chat_ids(tokenizer, q, long, limit) for q, long in PROMPT_SET]
    reference: dict[int, list[int]] = {}
    runs: list[dict[str, Any]] = []
    for round_ in range(-1, rounds):
        order = [bits for bits in round_order(max(round_, 0)) if bits in engines]
        if round_ < 0:
            order = order[:-1]  # the untimed pass runs each combination once
        for position, bits in enumerate(order):
            for p, ids in enumerate(prompts):
                row: dict[str, Any] = {"combo": combo_name(bits), "round": round_, "prompt": p,
                                       "position": position}
                try:
                    result = engines[bits].generate(ids, tokens, eos)
                except Exception as exc:  # noqa: BLE001 - one combination's failure does not end the probe
                    row["error"] = repr(exc)
                else:
                    logical = list(result["logical_tokens"])
                    if not any(bits) and p not in reference:
                        reference[p] = logical
                    decoded = max(1, len(result["physical_tokens"]) - 1)
                    row.update({"ms_per_token": result["decode_ns"] / 1e6 / decoded,
                                "prefill_ms": result["prefill_ns"] / 1e6, "tokens": len(logical),
                                "digest": _digest(logical), "capacity": result["capacity"],
                                "acceptance": result["acceptance"],
                                "first_difference": _first_difference(logical, reference[p])})
                runs.append(row)
                if on_run is not None:
                    on_run(row)
    for engine in engines.values():
        engine.close()
    return {"runs": runs, "errors": errors, "prompt_tokens": [len(ids) for ids in prompts]}


# ---------------------------------------------------------------- HW4: class A, prefill, widths, compile

def _bits(value: Any) -> str:
    """A digest of an array's exact bits (bfloat16 read as its 16-bit pattern)."""
    import mlx.core as mx
    import numpy as np

    if value.dtype in (mx.bfloat16, mx.float16):
        value = mx.view(value, mx.uint16)
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _state_bits(state: Any) -> str:
    from ironmule.runtime import _leaves

    return hashlib.sha256("".join(_bits(leaf) for leaf in _leaves(state)).encode()).hexdigest()


def capacity_for(prompt_len: int, tokens: int) -> int:
    """The product's capacity (`ironmule.service.MLXBackend.capacity_for`), which HW4 checks at, not the
    engine's own `_capacity`: on CUDA the capacity moves tokens (X1)."""
    return ((prompt_len + tokens + 8 + 63) // 64) * 64


def capture_set(engine: Any, ids: list[int], tokens: int, feed: list[int] | None,
                speculate_k: int = 0) -> dict[str, Any]:
    """One knob set on one prompt, bit for bit: the prefill's last logits (through `_project`, looked up at
    call time, so a wrapper sees them) and its state; then `tokens - 1` decode steps through the engine's
    own body, fed the reference tokens `feed` (its own greedy picks when None) so one flipped token does not
    end the comparison; the logits (or, with `fused_argmax`, the picks) of every step; the final state; and
    with speculation, the k + 1 verify window's logits at `SPECULATION_AT` against the sequential steps."""
    import mlx.core as mx
    from ironmule import runtime

    capacity = capacity_for(len(ids), tokens)
    captured: list[Any] = []
    original = runtime._project

    def capture(model: Any, hidden: Any) -> Any:
        out = original(model, hidden)
        captured.append(out)
        return out

    runtime._project = capture
    try:
        state, token = engine._prefill(list(ids), capacity)
    finally:
        runtime._project = original
    last = captured[-1][:, -1, :]
    mx.eval(last)
    result: dict[str, Any] = {"capacity": capacity, "prefill_logits": _bits(last),
                              "prefill_state": _state_bits(state), "first_token": int(token.reshape((-1,)).item())}
    fused = engine.knobs.fused_argmax
    body = engine._body(capacity, 1)
    picks = [result["first_token"]]
    steps: list[str] = []
    states: dict[int, Any] = {}
    current = picks[0] if feed is None else feed[0]
    for step in range(tokens - 1):
        if step in SPECULATION_AT:
            states[step] = state
        out = body(mx.array([[current]]), state)
        mx.eval(out[0], *runtime._leaves(out[1]))
        state = out[1]
        if fused:
            pick = int(out[0].reshape((-1,))[-1].item())
            steps.append(str(pick))
        else:
            logits = out[0][:, -1, :]
            steps.append(_bits(logits))
            pick = int(mx.argmax(logits.astype(mx.float32), axis=-1).item())
        picks.append(pick)
        current = pick if feed is None else feed[step + 1]
    result.update({"steps": steps, "picks": picks, "final_state": _state_bits(state)})
    if speculate_k and not fused:
        window_body = engine._body(capacity, speculate_k + 1)
        reference = feed if feed is not None else picks
        windows = {}
        for at, saved in states.items():
            window = reference[at:at + speculate_k + 1]
            if len(window) < speculate_k + 1:
                continue
            out = window_body(mx.array([window]), saved)
            mx.eval(out[0])
            windows[str(at)] = [_bits(out[0][:, j, :]) for j in range(speculate_k + 1)]
        result["verify_windows"] = windows
    return result


def capture_sets() -> list[tuple[int, ...]]:
    """Readback only moves the host's reads, never the arithmetic, so it is left out of the capture; it is
    class A exactly when its twin without it is."""
    return list(itertools.product((0, 1), repeat=len(CAPTURE_FACTORS)))


def capture_name(bits: tuple[int, ...]) -> str:
    return "+".join(f for f, b in zip(CAPTURE_FACTORS, bits) if b) or "baseline"


def probe_class_a(models: tuple[Any, Any], tokenizer: Any, tokens: int, limit: int | None = None) -> dict[str, Any]:
    from dataclasses import replace

    from ironmule import runtime

    plain, fused = models
    prompts = [chat_ids(tokenizer, q, long, limit) for q, long in PROMPT_SET]
    results: dict[str, Any] = {}
    reference: dict[int, dict[str, Any]] = {}
    for bits in capture_sets():
        knobs = {f: CAPTURE_LEVELS[f][b] for f, b in zip(CAPTURE_FACTORS, bits)}
        engine = runtime.Engine(fused if knobs["fuse_projections"] else plain, tokenizer,
                                replace(runtime.Knobs(**knobs), fuse_projections=False))
        rows = []
        try:
            for p, ids in enumerate(prompts):
                feed = None if not any(bits) else reference[p]["picks"]
                row = capture_set(engine, ids, tokens, feed, knobs["speculate_k"])
                if not any(bits):
                    reference[p] = row
                rows.append(row)
        except Exception as exc:  # noqa: BLE001 - recorded; the other sets still run
            results[capture_name(bits)] = {"error": repr(exc)}
            continue
        finally:
            engine.close()
        results[capture_name(bits)] = {"rows": rows}
    return {"sets": results, "prompt_tokens": [len(ids) for ids in prompts]}


def probe_separability(models: tuple[Any, Any], tokenizer: Any, decode_winner: dict[str, Any], rounds: int,
                       tokens: int, limit: int | None = None) -> dict[str, Any]:
    """`head_skip_prefill` x `fuse_projections` x HW3's decode winner (2 x 2 x 2) through `Engine.generate`,
    plus `prefill_into_fixed` alone (E1's kill on Metal, re-measured on CUDA), in HW3's round design."""
    from dataclasses import replace

    from ironmule import runtime

    plain, fused = models
    arms: dict[str, dict[str, Any]] = {}
    for hs, fp, dw in itertools.product((0, 1), repeat=3):
        knobs = {"head_skip_prefill": bool(hs), "fuse_projections": bool(fp), **(decode_winner if dw else {})}
        name = "+".join(n for n, b in (("head_skip_prefill", hs), ("fuse_projections", fp), ("decode_winner", dw))
                        if b) or "baseline"
        arms[name] = knobs
    arms["prefill_into_fixed"] = {"prefill_into_fixed": True}
    engines = {name: runtime.Engine(fused if knobs.get("fuse_projections") else plain, tokenizer,
                                    replace(runtime.Knobs(**knobs), fuse_projections=False))
               for name, knobs in arms.items()}
    eos = tuple(sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]))
    prompts = [chat_ids(tokenizer, q, long, limit) for q, long in PROMPT_SET]
    names = list(arms)
    others = [n for n in names if n != "baseline"]
    reference: dict[int, list[int]] = {}
    runs = []
    for round_ in range(-1, rounds):
        shift = max(round_, 0) % len(others)
        order = ["baseline", *others[shift:], *others[:shift]] + ([] if round_ < 0 else ["baseline"])
        for position, name in enumerate(order):
            for p, ids in enumerate(prompts):
                row: dict[str, Any] = {"arm": name, "round": round_, "prompt": p, "position": position}
                try:
                    result = engines[name].generate(ids, tokens, eos)
                except Exception as exc:  # noqa: BLE001
                    row["error"] = repr(exc)
                else:
                    logical = list(result["logical_tokens"])
                    if name == "baseline":
                        reference.setdefault(p, logical)
                    row.update({"prefill_ms": result["prefill_ns"] / 1e6,
                                "ms_per_token": result["decode_ns"] / 1e6 / max(1, len(result["physical_tokens"]) - 1),
                                "first_difference": _first_difference(logical, reference[p])})
                runs.append(row)
    for engine in engines.values():
        engine.close()
    return {"arms": {name: {k: v for k, v in knobs.items()} for name, knobs in arms.items()}, "runs": runs,
            "prompt_tokens": [len(ids) for ids in prompts]}


def probe_widths(models: tuple[Any, Any], tokenizer: Any, decode_sets: dict[str, dict[str, Any]], rounds: int,
                 tokens: int, limit: int | None = None) -> list[dict[str, Any]]:
    """Each decode set on the product's grouped path (`Runtime.serve`, `ThroughputMode`) at widths 1, 2 and 4,
    short prompts: wall seconds per decode round, from which a further session's cost follows. At width 1
    the sequential path applies readback and speculation; grouped rounds step one token at a time."""
    from dataclasses import replace

    from ironmule import runtime
    from ironmule.service import Request, Runtime, ThroughputMode

    plain, fused = models
    prompts = [chat_ids(tokenizer, q, False, limit) for q in range(max(WIDTHS_HW4))]
    rows = []
    for round_ in range(rounds):
        names = list(decode_sets)
        shift = round_ % len(names)
        for name in names[shift:] + names[:shift]:
            knobs = decode_sets[name]
            for width in WIDTHS_HW4:
                engine = runtime.Engine(fused if knobs.get("fuse_projections") else plain, tokenizer,
                                        replace(runtime.Knobs(**knobs), fuse_projections=False))
                served = Runtime(engine, tokenizer, mode=ThroughputMode(max_width=max(width, 1)))
                try:
                    group = [Request(prompt_ids=prompts[i], max_tokens=tokens) for i in range(width)]
                    served.serve(group)  # warm: compile this capacity and width outside the clock
                    started = time.perf_counter()
                    results = served.serve([Request(prompt_ids=prompts[i], max_tokens=tokens) for i in range(width)])
                    seconds = time.perf_counter() - started
                    rows.append({"set": name, "width": width, "round": round_, "seconds": seconds,
                                 "tokens": [len(r.tokens) for r in results],
                                 "digests": [_digest(r.tokens) for r in results]})
                except Exception as exc:  # noqa: BLE001
                    rows.append({"set": name, "width": width, "round": round_, "error": repr(exc)})
                finally:
                    served.close()
    return rows


def probe_compile(models: tuple[Any, Any], tokenizer: Any, decode_sets: dict[str, dict[str, Any]],
                  limit: int | None = None) -> list[dict[str, Any]]:
    """What a new compiled decode body costs: the first request at a capacity the engine has not seen, minus
    the same request again, at three capacities."""
    from dataclasses import replace

    from ironmule import runtime

    plain, fused = models
    eos = tuple(sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]))
    ids = chat_ids(tokenizer, 0, False, limit)
    rows = []
    for name, knobs in decode_sets.items():
        engine = runtime.Engine(fused if knobs.get("fuse_projections") else plain, tokenizer,
                                replace(runtime.Knobs(**knobs), fuse_projections=False))
        try:
            for tokens in (16, 80, 144):  # three capacities 64 apart
                started = time.perf_counter()
                engine.generate(ids, tokens, eos)
                first = time.perf_counter() - started
                started = time.perf_counter()
                engine.generate(ids, tokens, eos)
                rows.append({"set": name, "tokens": tokens, "first_seconds": first,
                             "again_seconds": time.perf_counter() - started})
        finally:
            engine.close()
    return rows


def probe_prefill_floor(model: Any, rows: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
    """The prefill's compute floor: each linear layer at M = prompt rows, achieved TFLOP/s."""
    import mlx.core as mx

    out = []
    for name, call, width, weight_bytes in linear_layers(model):
        outputs = weight_bytes // 2 // width
        for m in rows:
            x = mx.random.normal((1, m, width), key=mx.random.key(m)).astype(mx.bfloat16)
            seconds = _median_time(lambda: call(x), repeats)
            out.append({"layer": name, "rows": m, "seconds": seconds,
                        "tflop_per_s": 2 * m * width * outputs / seconds / 1e12})
    return out


def class_a_sets(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each capture set's verdict against the baseline, bit for bit (HW4's class A)."""
    sets = capture.get("sets") or {}
    base = (sets.get("baseline") or {}).get("rows")
    verdicts: dict[str, dict[str, Any]] = {}
    if not base:
        return verdicts
    for name, entry in sets.items():
        rows = entry.get("rows")
        if not rows:
            verdicts[name] = {"class_a": False, "reason": entry.get("error", "no rows")}
            continue
        fused = "fused_argmax" in name.split("+")
        prefill = all(r["prefill_logits"] == b["prefill_logits"] and r["prefill_state"] == b["prefill_state"]
                      for r, b in zip(rows, base))
        if fused:
            steps = all(r["picks"] == b["picks"] for r, b in zip(rows, base))
        else:
            steps = all(r["steps"] == b["steps"] for r, b in zip(rows, base))
        final = all(r["final_state"] == b["final_state"] for r, b in zip(rows, base))
        first_step = min((i for r, b in zip(rows, base) for i, (x, y) in enumerate(zip(r["steps"], b["steps"]))
                          if x != y and not fused), default=None)
        windows = None
        if any("verify_windows" in r for r in rows):
            windows = all(digests == b["steps"][int(at):int(at) + len(digests)]
                          for r, b in zip(rows, base) for at, digests in (r.get("verify_windows") or {}).items())
        verdicts[name] = {"prefill": prefill, "steps": steps, "final_state": final, "verify_windows": windows,
                          "first_differing_step": first_step,
                          "class_a": prefill and steps and final and windows is not False}
    for name, verdict in verdicts.items():  # fused argmax hides its logits: class A only if its twin is
        parts = name.split("+")
        if "fused_argmax" in parts and verdict.get("class_a"):
            twin = "+".join(x for x in parts if x != "fused_argmax") or "baseline"
            verdict["class_a"] = bool(verdicts.get(twin, {}).get("class_a"))
            verdict["assumes"] = "argmax of bit-identical logits picks the same token"
    return verdicts


def decode_sets_from(verdicts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The decode knob sets HW4's width and compile parts measure: every class-A capture set without the
    prefill knob, each with readback 1 and 4 (readback moves only the host's reads)."""
    out: dict[str, dict[str, Any]] = {}
    for name, verdict in verdicts.items():
        parts = [x for x in name.split("+") if x != "baseline"]
        if not verdict.get("class_a") or "head_skip_prefill" in parts:
            continue
        knobs = {f: CAPTURE_LEVELS[f][1] for f in parts}
        for readback in (1, 4):
            label = "+".join(parts + (["readback_every"] if readback == 4 else [])) or "baseline"
            out[label] = {**knobs, "readback_every": readback}
    return out or {"baseline": {}}


def probe_hw4(model_id: str, out: Path | None, decode_winner: dict[str, Any], *, smoke: bool = False,
              loaded: tuple[Any, Any, Any] | None = None) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    report: dict[str, Any] = {"schema": "ironmule.cuda_probe.hw4.v1", "model": model_id, "smoke": smoke,
                              "decode_winner": decode_winner, "started_unix": time.time(), "env": environment()}

    def save() -> None:
        if out is not None:
            out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")

    if loaded is None:
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        path = snapshot_download(model_id, revision=REVISION if model_id == MODEL else None,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                 "tokenizer*"])
        plain, tokenizer = load(path)
        fused, _ = load(path)
        report["revision"] = Path(path).name
    else:
        plain, fused, tokenizer = loaded
    from ironmule import fast

    try:
        fast.fuse_projections(fused)
    except Exception as exc:  # noqa: BLE001 - the fused sets then fail and are recorded
        report["fuse_error"] = repr(exc)
    report["weights_bytes"] = sum(v.nbytes for _, v in tree_flatten(plain.parameters()))
    models = (plain, fused)
    tokens, rounds, limit, repeats = (4, 1, 24, 2) if smoke else (TOKENS, ROUNDS, None, REPEATS)
    save()
    parts: list[tuple[str, Callable[[], Any]]] = [
        ("prefill_floor", lambda: probe_prefill_floor(plain, (8,) if smoke else PREFILL_ROWS, repeats)),
        ("class_a", lambda: probe_class_a(models, tokenizer, tokens, limit)),
        ("separability", lambda: probe_separability(models, tokenizer, decode_winner, rounds, tokens, limit)),
    ]
    for name, part in parts:
        started = time.time()
        try:
            report[name] = part()
        except Exception as exc:  # noqa: BLE001 - a failed part is recorded and the next one still runs
            report[name] = {"error": repr(exc)}
        report.setdefault("seconds", {})[name] = time.time() - started
        save()
    decode_sets = decode_sets_from(class_a_sets(report["class_a"]) if "sets" in report["class_a"] else {})
    report["decode_sets"] = decode_sets
    for name, part in (("widths", lambda: probe_widths(models, tokenizer, decode_sets, rounds, tokens, limit)),
                       ("compile", lambda: probe_compile(models, tokenizer, decode_sets, limit))):
        started = time.time()
        try:
            report[name] = part()
        except Exception as exc:  # noqa: BLE001
            report[name] = {"error": repr(exc)}
        report.setdefault("seconds", {})[name] = time.time() - started
        save()
    report["finished_unix"] = time.time()
    report["analysis"] = analyse_hw4(report)
    save()
    return report


# ---------------------------------------------------------------- KB1-0: a batch-invariant dense row kernel

DENSE_HEADER = r"""
__device__ __forceinline__ float lo_bf(unsigned int u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_bf(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ unsigned short to_bf(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
"""

# `ironmule/cuda_native.py`'s 4-bit row kernel without the dequantisation: one warp per R output rows and
# M <= 8 activation rows of a bfloat16 weight; each lane walks K in 8-element chunks, every (row, activation
# row) pair keeps its own float32 chain with pinned roundings, then one shuffle tree. A row's arithmetic does
# not depend on M, so the kernel is batch-invariant by construction; the weight is read once for all M rows.
DENSE_SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int row0 = (gid >> 5) * R;
  const unsigned int lane = threadIdx.x & 31;
  if (row0 >= N) return;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  float acc[R * M];
#pragma unroll
  for (int i = 0; i < R * M; ++i) acc[i] = 0.f;
  for (unsigned int c = lane; c < K / 8; c += 32) {
    float xf[M * 8];
#pragma unroll
    for (int mo = 0; mo < M; ++mo) {
      const uint4 xa = xv[(size_t)mo * (K / 8) + c];
      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        xf[mo * 8 + 2 * e] = lo_bf(xw[e]);
        xf[mo * 8 + 2 * e + 1] = hi_bf(xw[e]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const unsigned int row = row0 + r;
      if (row < N) {
        const uint4 q = w4[(size_t)row * (K / 8) + c];
        const unsigned int qw[4] = {q.x, q.y, q.z, q.w};
        float wf[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          wf[2 * e] = lo_bf(qw[e]);
          wf[2 * e + 1] = hi_bf(qw[e]);
        }
#pragma unroll
        for (int mo = 0; mo < M; ++mo) {
          float d = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) d = __fmaf_rn(wf[i], xf[mo * 8 + i], d);
          acc[r * M + mo] = __fadd_rn(acc[r * M + mo], d);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R * M; ++i) {
    float a = acc[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_down_sync(0xffffffffu, a, o));
    const unsigned int row = row0 + i / M;
    if (lane == 0 && row < N) out[(i % M) * N + row] = to_bf(a);
  }
"""
_dense_kernel: Any = None


def dense_rows(x: Any, w: Any, rows: int | None = None) -> Any:
    """`x @ w.T` for a bfloat16 weight and up to 8 rows of bfloat16 activations, through the row kernel.
    `rows` per warp only changes how output rows are spread over warps, never a row's arithmetic."""
    import mlx.core as mx

    global _dense_kernel
    if _dense_kernel is None:
        _dense_kernel = mx.fast.cuda_kernel(name="ironmule_dense_rows", input_names=["x", "w"],
                                            output_names=["out"], source=DENSE_SOURCE, header=DENSE_HEADER)
    n, k = w.shape
    m = x.size // k
    rows = rows or (4 if n >= 8192 else 2)
    threads = -(-(-(-n // rows) * 32) // 256) * 256
    out = _dense_kernel(inputs=[mx.contiguous(x.reshape(m, k)), w],
                        template=[("N", n), ("K", k), ("R", rows), ("M", m)],
                        grid=(threads, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(m, n)],
                        output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16).reshape(*x.shape[:-1], n)


def kb1_layers(model: Any) -> list[tuple[str, Any, Callable[[Any], Any], int]]:
    """(name, weight, MLX's own call, layers per step) for block 0's linear layers and the output head."""
    from ironmule.runtime import _project, _trunk

    trunk = _trunk(model)
    block = trunk.layers[0]
    out = [(name, getattr(owner, name).weight, getattr(owner, name), len(trunk.layers))
           for owner, names in ((block.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                                (block.mlp, ("gate_proj", "up_proj", "down_proj")))
           for name in names]
    head = getattr(model, "lm_head", None)
    weight = head.weight if head is not None and not getattr(model.args, "tie_word_embeddings", False) \
        else trunk.embed_tokens.weight
    out.append(("output", weight, lambda x: _project(model, x), 1))
    return out


def kb1_activations(model: Any, tokenizer: Any, samples: int) -> dict[str, Any]:
    """Real inputs of every KB1 layer: one forward over `samples` prompt tokens, each layer's input captured
    (the output head's input is the trunk's normed output)."""
    import mlx.core as mx
    from ironmule.runtime import _trunk

    trunk = _trunk(model)
    block = trunk.layers[0]
    captured: dict[str, Any] = {}
    swapped = []
    for owner, names in ((block.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                         (block.mlp, ("gate_proj", "up_proj", "down_proj"))):
        for name in names:
            module = getattr(owner, name)
            base = type(module)

            def call(self: Any, x: Any, *args: Any, _name: str = name, _base: Any = base, **kwargs: Any) -> Any:
                captured.setdefault(_name, x)
                return _base.__call__(self, x, *args, **kwargs)

            module.__class__ = type(f"Capture_{name}", (base,), {"__call__": call})
            swapped.append((module, base))
    try:
        hidden = trunk(mx.array(row_ids(tokenizer, 0, samples))[None])
    finally:
        for module, base in swapped:
            module.__class__ = base
    captured["output"] = hidden
    rows = {name: mx.contiguous(value.reshape(-1, value.shape[-1])[:samples]) for name, value in captured.items()}
    mx.eval(*rows.values())
    return rows


def probe_kb1(model: Any, tokenizer: Any, repeats: int, samples: int = KB1_SAMPLES,
              kernel: Callable[[Any, Any], Any] | None = None) -> list[dict[str, Any]]:
    """KB1-0: per layer, the kernel at one row against MLX's own product bit for bit (every sample row), each
    row at M = 2, 4, 8 against the same kernel at one row, and both timed alternately at every M."""
    import mlx.core as mx

    kernel = kernel or dense_rows
    activations = kb1_activations(model, tokenizer, samples)
    results = []
    for name, weight, call, per_step in kb1_layers(model):
        x = activations[name]
        mine = [kernel(x[i:i + 1], weight) for i in range(x.shape[0])]
        theirs = [call(x[i:i + 1]) for i in range(x.shape[0])]
        mx.eval(*mine, *theirs)
        identical = sum(bool(mx.array_equal(a, b).item()) for a, b in zip(mine, theirs))
        difference = max(float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
                         for a, b in zip(mine, theirs))
        entry: dict[str, Any] = {"layer": name, "shape": list(weight.shape), "per_step": per_step,
                                 "weight_bytes": int(weight.nbytes), "identical_to_mlx_at_1": identical,
                                 "rows_compared": x.shape[0], "max_difference_to_mlx": difference, "rows": {}}
        for m in KB1_ROWS:
            stacked = kernel(x[:m], weight)
            mx.eval(stacked)
            invariant = sum(bool(mx.array_equal(stacked[i:i + 1], mine[i]).item()) for i in range(m))
            kernel_times, mlx_times = _paired_times(lambda: kernel(x[:m], weight), lambda: call(x[:m]), repeats)
            entry["rows"][str(m)] = {"invariant_rows": invariant, "kernel_seconds": kernel_times,
                                     "mlx_seconds": mlx_times}
        results.append(entry)
    return results


def analyse_kb1(layers: list[dict[str, Any]], resamples: int = RESAMPLES) -> dict[str, Any]:
    """KB1-0's three kill rules, as fixed in BACKLOG before data."""
    if not layers:
        return {"valid": False}
    exact_at_1 = all(e["identical_to_mlx_at_1"] == e["rows_compared"] for e in layers)
    invariant = all(r["invariant_rows"] == int(m) for e in layers for m, r in e["rows"].items())
    repeats = min(len(r["kernel_seconds"]) for e in layers for r in e["rows"].values())

    def step(m: str, key: str, i: int) -> float:
        return sum(e["per_step"] * e["rows"][m][key][i] for e in layers)

    ratios = [step("8", "kernel_seconds", i) / step("1", "mlx_seconds", i) for i in range(repeats)]
    ci = _bootstrap_mean(ratios, resamples)
    limit = 8 / GROUPED_LEVER
    per_layer = {e["layer"]: {m: {"kernel_ms": statistics.median(r["kernel_seconds"]) * 1000,
                                  "mlx_ms": statistics.median(r["mlx_seconds"]) * 1000,
                                  "kernel_gb_per_s": e["weight_bytes"] / statistics.median(r["kernel_seconds"]) / 1e9}
                              for m, r in e["rows"].items()} for e in layers}
    return {"valid": True,
            "kill (i) bit-identical to MLX at one row": exact_at_1,
            "kill (ii) batch-invariant": invariant,
            "linear step at 8 rows over MLX at one row": statistics.fmean(ratios), "ratio_ci95": ci,
            "kill (iii) limit": limit, "lever_holds": ci[1] < limit,
            "verdict": ("bug: a row changes with M, fix the kernel first" if not invariant else
                        "no lever over grouped batch-1: stop" if ci[1] >= limit else
                        "class A route: build the stacked step on today's answers" if exact_at_1 else
                        "lever holds but not bit-identical to MLX: only inside a plan version with FA2"),
            "layers": per_layer,
            "identical_to_mlx_at_1": {e["layer"]: f"{e['identical_to_mlx_at_1']}/{e['rows_compared']}" for e in layers},
            "max_difference_to_mlx": {e["layer"]: e["max_difference_to_mlx"] for e in layers}}


def probe_kb1_report(model_id: str, out: Path | None, *, smoke: bool = False,
                     loaded: tuple[Any, Any] | None = None, kernel: Callable[[Any, Any], Any] | None = None,
                     model_level: bool = False, plan: "DensePlan | None" = None) -> dict[str, Any]:
    report: dict[str, Any] = {"schema": "ironmule.cuda_probe.kb1.v1", "model": model_id, "smoke": smoke,
                              "started_unix": time.time(), "env": environment()}
    if loaded is None:
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        path = snapshot_download(model_id, revision=REVISION if model_id == MODEL else None,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                 "tokenizer*"])
        model, tokenizer = load(path)
        report["revision"] = Path(path).name
    else:
        model, tokenizer = loaded
    try:
        report["kb1"] = probe_kb1(model, tokenizer, 2 if smoke else REPEATS, 8 if smoke else KB1_SAMPLES, kernel)
        report["analysis"] = analyse_kb1(report["kb1"])
    except Exception as exc:  # noqa: BLE001 - recorded, so a failed kernel still leaves a report
        report["kb1"] = {"error": repr(exc)}
        report["analysis"] = {"valid": False, "error": repr(exc)}
    if model_level:
        if out is not None:
            out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
        try:
            report["kb1_model"] = probe_kb1_model(
                model, tokenizer, steps=4 if smoke else KB1M_STEPS, rounds=1 if smoke else ROUNDS,
                limit=24 if smoke else None, plan=plan,
                gate=gate_items(tokenizer, 4, 12, (16, 6, 4)) if smoke else None)
            report["analysis_model"] = analyse_kb1_model(report["kb1_model"])
        except Exception as exc:  # noqa: BLE001
            report["kb1_model"] = {"error": repr(exc)}
    report["finished_unix"] = time.time()
    if out is not None:
        out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    return report


# ---------------------------------------------------------------- KB1-1: the dense plan, stacked rows, FA2

FA2_CHUNK = 64  # keys per chunk, at absolute positions: the split depends only on a row's own length

FA2_HEADER = DENSE_HEADER + r"""
__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
#define NEG_INF __int_as_float(0xff800000)
"""

# Pass 1: one warp per (row, KV head, 64-key chunk); each lane scores two keys with the whole head-dim dot in
# one fixed order, the chunk's max and sum through fixed trees, then P.V with lanes over the head dim.
FA2_PARTIAL = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * HKV * NC) return;
  const unsigned int c = warp % NC;
  const unsigned int b = (warp / NC) / HKV;
  const unsigned int kh = (warp / NC) % HKV;
  const int len = lengths[b];
  const int start = c * 64;
  if (start >= len) return;
  const unsigned short* qs = reinterpret_cast<const unsigned short*>(q);
  const unsigned short* ks = reinterpret_cast<const unsigned short*>(k);
  const unsigned short* vs = reinterpret_cast<const unsigned short*>(v);
  const size_t base = ((size_t)b * HKV + kh) * C * D;
  const int count = min(64, len - start);
  for (int g = 0; g < G; ++g) {
    const size_t qo = ((size_t)b * H + kh * G + g) * D;
    float s[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int t = start + j * 32 + (int)lane;
      if (t < len) {
        const unsigned short* kr = ks + base + (size_t)t * D;
        float d = 0.f;
        for (int i = 0; i < D; ++i) d = __fmaf_rn(one_bf(qs[qo + i]), one_bf(kr[i]), d);
        s[j] = __fmul_rn(d, SCALE);
      } else {
        s[j] = NEG_INF;
      }
    }
    float m = fmaxf(s[0], s[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float pr[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) pr[j] = (s[j] == NEG_INF) ? 0.f : expf(s[j] - m);
    float sum = __fadd_rn(pr[0], pr[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, o));
    sum = __shfl_sync(0xffffffffu, sum, 0);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = 0; t < count; ++t) {
      const float pt = __shfl_sync(0xffffffffu, pr[t >> 5], t & 31);
      const unsigned short* vr = vs + base + (size_t)(start + t) * D + lane * 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(pt, one_bf(vr[e]), acc[e]);
    }
    const size_t po = ((size_t)b * H + kh * G + g) * NC + c;
#pragma unroll
    for (int e = 0; e < 4; ++e) pacc[po * D + lane * 4 + e] = acc[e];
    if (lane == 0) {
      pml[po * 2] = m;
      pml[po * 2 + 1] = sum;
    }
  }
"""

# Pass 2: one warp per (row, query head) combines that row's chunks in index order.
FA2_COMBINE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * H) return;
  const unsigned int b = warp / H;
  const unsigned int h = warp % H;
  const int nc = (lengths[b] + 63) / 64;
  float M = NEG_INF, L = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < nc; ++c) {
    const size_t po = ((size_t)b * H + h) * NC + c;
    const float mc = pml[po * 2];
    const float lc = pml[po * 2 + 1];
    const float mn = fmaxf(M, mc);
    const float a = expf(M - mn);
    const float bb = expf(mc - mn);
    L = __fmaf_rn(L, a, __fmul_rn(lc, bb));
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(acc[e], a, __fmul_rn(pacc[po * D + lane * 4 + e], bb));
    M = mn;
  }
#pragma unroll
  for (int e = 0; e < 4; ++e) out[((size_t)b * H + h) * D + lane * 4 + e] = to_bf(__fdiv_rn(acc[e], L));
"""
_fa2_kernels: dict[str, Any] = {}


def fa2(q: Any, k: Any, v: Any, lengths: Any) -> Any:
    """FA2: one decode query per row against a fixed-capacity cache, reading only the first `lengths[b]`
    positions, split into 64-key chunks at absolute positions and combined in index order. The result depends
    on neither the capacity nor the other rows."""
    import mlx.core as mx

    b, h, d = q.shape
    _, hkv, c, _ = k.shape
    if d != 128 or h % hkv:
        raise ValueError("FA2 serves head dim 128 and grouped query heads only")
    if not _fa2_kernels:
        scale = f"{d ** -0.5!r}f"
        _fa2_kernels["partial"] = mx.fast.cuda_kernel(
            name="ironmule_fa2_partial", input_names=["q", "k", "v", "lengths"], output_names=["pacc", "pml"],
            source=FA2_PARTIAL.replace("SCALE", scale), header=FA2_HEADER)
        _fa2_kernels["combine"] = mx.fast.cuda_kernel(
            name="ironmule_fa2_combine", input_names=["pacc", "pml", "lengths"], output_names=["out"],
            source=FA2_COMBINE, header=FA2_HEADER)
    nc = -(-c // FA2_CHUNK)
    lengths = lengths.astype(mx.int32)
    template = [("B", b), ("H", h), ("HKV", hkv), ("G", h // hkv), ("D", d), ("C", c), ("NC", nc)]
    pacc, pml = _fa2_kernels["partial"](
        inputs=[mx.contiguous(q), k, v, lengths], template=template,
        grid=(-(-(b * hkv * nc * 32) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b, h, nc, d), (b, h, nc, 2)], output_dtypes=[mx.float32, mx.float32])
    out = _fa2_kernels["combine"](
        inputs=[pacc, pml, lengths], template=template, grid=(-(-(b * h * 32) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(b, h, d)], output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16)


class DensePlan:
    """KB1-1: one Qwen 3 decode step for up to eight stacked rows at their own positions, the same forward at
    every B, so stacking is exact within the plan. Linears through the row kernel, attention through FA2 over
    a fixed-capacity cache, MLX's own RoPE with one position per row, `rms_norm` and `silu`. It changes
    answers against today's engine, so it is a numeric plan: opt-in, behind the quality gate."""

    def __init__(self, model: Any, linear: Callable[[Any, Any], Any] | None = None,
                 attention: Callable[[Any, Any, Any, Any], Any] | None = None):
        from ironmule.runtime import _trunk

        trunk = _trunk(model)
        args = model.args
        self.model, self.trunk, self.args = model, trunk, args
        self.linear, self.attention = linear or dense_rows, attention or fa2
        self.heads, self.kv_heads = args.num_attention_heads, args.num_key_value_heads
        self.dim = getattr(args, "head_dim", None) or args.hidden_size // self.heads
        self.eps = args.rms_norm_eps
        if getattr(args, "rope_scaling", None):
            raise ValueError("the dense plan's RoPE covers unscaled rotary embeddings only")
        head = getattr(model, "lm_head", None)
        self.head = head.weight if head is not None and not getattr(args, "tie_word_embeddings", False) \
            else trunk.embed_tokens.weight

    def rope(self, x: Any, positions: Any) -> Any:
        """MLX's own RoPE with one position per row (bit-identical to a single row's call on the CPU)."""
        import mlx.core as mx

        rows, heads, dim = x.shape
        return mx.fast.rope(x.reshape(rows, heads, 1, dim), dim, traditional=False, base=self.args.rope_theta,
                            scale=1.0, offset=positions.astype(mx.int32)).reshape(rows, heads, dim)

    def new_cache(self, rows: int, capacity: int) -> list[tuple[Any, Any]]:
        import mlx.core as mx

        shape = (rows, self.kv_heads, capacity, self.dim)
        return [(mx.zeros(shape, mx.bfloat16), mx.zeros(shape, mx.bfloat16)) for _ in self.trunk.layers]

    def step(self, tokens: Any, positions: Any, cache: list[tuple[Any, Any]]) -> tuple[Any, list[tuple[Any, Any]]]:
        """Tokens (B,) at positions (B,) into `cache`; returns float32 logits (B, V) and the updated cache."""
        import mlx.core as mx
        from mlx_lm.models.activations import swiglu

        rows = tokens.shape[0]
        h = self.trunk.embed_tokens(tokens)
        index = mx.broadcast_to(positions.astype(mx.int32)[:, None, None, None], (rows, self.kv_heads, 1, self.dim))
        updated = []
        for layer, (keys, values) in zip(self.trunk.layers, cache):
            attn, mlp = layer.self_attn, layer.mlp
            x = mx.fast.rms_norm(h, layer.input_layernorm.weight, self.eps)
            q = mx.fast.rms_norm(self.linear(x, attn.q_proj.weight).reshape(rows, self.heads, self.dim),
                                 attn.q_norm.weight, self.eps)
            k = mx.fast.rms_norm(self.linear(x, attn.k_proj.weight).reshape(rows, self.kv_heads, self.dim),
                                 attn.k_norm.weight, self.eps)
            v = self.linear(x, attn.v_proj.weight).reshape(rows, self.kv_heads, self.dim)
            q, k = self.rope(q, positions), self.rope(k, positions)
            keys = mx.put_along_axis(keys, index, k[:, :, None, :], axis=2)
            values = mx.put_along_axis(values, index, v[:, :, None, :], axis=2)
            o = self.attention(q, keys, values, positions + 1)
            h = h + self.linear(o.reshape(rows, self.heads * self.dim), attn.o_proj.weight)
            x = mx.fast.rms_norm(h, layer.post_attention_layernorm.weight, self.eps)
            h = h + self.linear(swiglu(self.linear(x, mlp.gate_proj.weight), self.linear(x, mlp.up_proj.weight)),
                                mlp.down_proj.weight)
            updated.append((keys, values))
        h = mx.fast.rms_norm(h, self.trunk.norm.weight, self.eps)
        return self.linear(h, self.head).astype(mx.float32), updated

    def load_prefill(self, cache: list[tuple[Any, Any]], row: int, states: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
        """Copy one request's prefilled keys and values (mlx-lm's cache state, (1, Hkv, L, D)) into row `row`."""
        out = []
        for (keys, values), (k, v) in zip(cache, states):
            n = k.shape[2]
            keys[row:row + 1, :, :n, :] = k.astype(keys.dtype)
            values[row:row + 1, :, :n, :] = v.astype(values.dtype)
            out.append((keys, values))
        return out


def gate_items(tokenizer: Any, chunks: int | None = None, size: int | None = None,
               long: tuple[int, int, int] = (LONG_PREFIX, LONG_SCORED, LONG_CHUNKS)) -> dict[str, list[tuple[list[int], list[int]]]]:
    """The numeric plans' quality text (`ironmule.numeric_choice`), as (prefix, scored) pairs: the gate's own
    16 x 512-token chunks with no prefix, and a long variant whose scored tokens follow a 1,024-token prefix."""
    from ironmule.numeric_choice import CHUNK_TOKENS, CHUNKS, _text_ids

    chunks, size = chunks or CHUNKS, size or CHUNK_TOKENS
    ids = _text_ids(tokenizer)
    main = [([], ids[i * size:(i + 1) * size + 1]) for i in range(chunks)]
    prefix, scored, count = long
    step = prefix + scored + 1
    far = [(ids[i * step:i * step + prefix], ids[i * step + prefix:(i + 1) * step]) for i in range(count)
           if (i + 1) * step <= len(ids)]
    return {"main": [m for m in main if len(m[1]) == size + 1], "long": far}


def _nll(logits: Any, target: int) -> float:
    import mlx.core as mx

    row = logits.reshape(-1).astype(mx.float32)
    return float((mx.logsumexp(row) - row[target]).item())


def reference_nll(engine: Any, prefix: list[int], scored: list[int]) -> float:
    """Today's engine on its decode path, teacher-forced: the prefix and the first scored token prefilled,
    then every next token's negative log-likelihood from a decode step; the mean over the scored tokens."""
    import mlx.core as mx
    from ironmule import runtime

    capacity = capacity_for(len(prefix) + 1, len(scored))
    captured: list[Any] = []
    original = runtime._project

    def capture(model: Any, hidden: Any) -> Any:
        out = original(model, hidden)
        captured.append(out)
        return out

    runtime._project = capture
    try:
        state, _ = engine._prefill(prefix + scored[:1], capacity)
    finally:
        runtime._project = original
    total = _nll(captured[-1][:, -1, :], scored[1])
    body = engine._body(capacity, 1)
    for t in range(1, len(scored) - 1):
        out = body(mx.array([[scored[t]]]), state)
        state = out[1]
        total += _nll(out[0][:, -1, :], scored[t + 1])
    return total / (len(scored) - 1)


def plan_nll(plan: "DensePlan", model: Any, items: list[tuple[list[int], list[int]]], rows: int,
             step: Callable[..., Any] | None = None) -> list[float]:
    """The plan on its decode path, teacher-forced, `rows` items stacked (exact within the plan)."""
    import mlx.core as mx

    step = step or plan.step

    means = []
    for start in range(0, len(items), rows):
        group = items[start:start + rows]
        width = max(len(prefix) + len(scored) for prefix, scored in group)
        cache = plan.new_cache(len(group), capacity_for(width, 0))
        sums = []
        for r, (prefix, scored) in enumerate(group):
            states, _ = _prefill(model, prefix + scored[:1])
            cache = plan.load_prefill(cache, r, states)
            logits = model(mx.array(prefix + scored[:1])[None])  # the first prediction is the unchanged prefill's
            sums.append(_nll(logits[:, -1, :], scored[1]))
        for t in range(1, len(group[0][1]) - 1):
            tokens = mx.array([scored[t] for _, scored in group])
            positions = mx.array([len(prefix) + t for prefix, _ in group])
            logits, cache = step(tokens, positions, cache)
            mx.eval(logits)
            for r, (_, scored) in enumerate(group):
                sums[r] += _nll(logits[r], scored[t + 1])
        means.extend(total / (len(scored) - 1) for total, (_, scored) in zip(sums, group))
    return means


def probe_kb1_model(model: Any, tokenizer: Any, *, steps: int, rounds: int, limit: int | None = None,
                    gate: dict[str, Any] | None = None, plan: "DensePlan | None" = None) -> dict[str, Any]:
    """KB1-1: (a) stacked rows against the same plan at one row, bit for bit; (b) a stacked step of eight
    against a width-8 grouped round of today's engine; (c) the quality gate on the plan's decode path."""
    import mlx.core as mx
    from ironmule import runtime

    plan = plan or DensePlan(model)
    prompts = [chat_ids(tokenizer, i, i % 2 == 1, limit) for i in range(KB1M_PROMPTS)]
    prefills = [_prefill(model, ids) for ids in prompts]
    out: dict[str, Any] = {"prompt_tokens": [len(ids) for ids in prompts]}

    def alone(r: int) -> tuple[list[int], list[str], list[float]]:
        states, first = prefills[r]
        cache = plan.load_prefill(plan.new_cache(1, capacity_for(len(prompts[r]), steps)), 0, states)
        tokens, digests, times = [int(first.item())], [], []
        for t in range(steps):
            started = time.perf_counter()
            logits, cache = plan.step(mx.array([tokens[-1]]), mx.array([len(prompts[r]) + t]), cache)
            mx.eval(logits)
            times.append(time.perf_counter() - started)
            digests.append(_bits(logits[0]))
            tokens.append(int(mx.argmax(logits[0]).item()))
        return tokens, digests, times

    singles = [alone(r) for r in range(len(prompts))]
    out["alone_seconds_per_step"] = statistics.median(t for _, _, times in singles for t in times[2:] or times)

    def stacked(rows: int, stretch: int = 1) -> tuple[list[list[str]], list[float]]:
        capacity = max(capacity_for(len(prompts[r]), steps) for r in range(rows)) * stretch
        cache = plan.new_cache(rows, capacity)
        for r in range(rows):
            cache = plan.load_prefill(cache, r, prefills[r][0])
        digests, times = [[] for _ in range(rows)], []
        for t in range(steps):
            tokens = mx.array([singles[r][0][t] for r in range(rows)])  # each row fed its own tokens alone
            positions = mx.array([len(prompts[r]) + t for r in range(rows)])
            started = time.perf_counter()
            logits, cache = plan.step(tokens, positions, cache)
            mx.eval(logits)
            times.append(time.perf_counter() - started)
            for r in range(rows):
                digests[r].append(_bits(logits[r]))
        return digests, times

    out["stacks"] = {}
    for rows in KB1M_STACKS:
        digests, times = stacked(rows)
        out["stacks"][str(rows)] = {"identical_steps": sum(d == e for r in range(rows)
                                                           for d, e in zip(digests[r], singles[r][1])),
                                    "steps": rows * steps, "seconds_per_step": statistics.median(times[2:] or times)}
    digests, times = stacked(KB1M_PROMPTS, stretch=2)  # the KV write must not copy the whole cache
    out["double_capacity"] = {"seconds_per_step": statistics.median(times[2:] or times), "steps": KB1M_PROMPTS * steps,
                              "identical_steps": sum(d == e for r in range(KB1M_PROMPTS)
                                                     for d, e in zip(digests[r], singles[r][1]))}
    eos = tuple(sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]))
    engines = {"baseline": runtime.Engine(model, tokenizer, runtime.Knobs()),
               "compiled": runtime.Engine(model, tokenizer, runtime.Knobs(compiled_fixed_cache=True))}

    def grouped_round(engine: Any) -> list[float]:
        """Today's grouped batch-1 at width 8, as the executor runs a round: every member's step dispatched at
        its own capacity, one wait for all of them."""
        states, tokens, bodies = [], [], []
        for ids in prompts:
            capacity = capacity_for(len(ids), steps)
            state, token = engine._prefill(list(ids), capacity)
            states.append(state)
            tokens.append(token)
            bodies.append(engine._body(capacity, 1))
        times = []
        for _ in range(steps):
            started = time.perf_counter()
            outs = [body(token, state) for body, token, state in zip(bodies, tokens, states)]
            picks = [mx.argmax(out[0][:, -1:, :], axis=-1) for out in outs]
            mx.eval(*picks, *[runtime._leaves(out[1]) for out in outs])
            times.append(time.perf_counter() - started)
            states, tokens = [out[1] for out in outs], picks
        return times

    try:
        for engine in engines.values():
            grouped_round(engine)  # warm every capacity
        rounds_out = []
        for _ in range(rounds):
            _, times = stacked(KB1M_PROMPTS)
            today = {name: statistics.median(grouped_round(engine)[2:]) for name, engine in engines.items()}
            rounds_out.append({"stacked": statistics.median(times[2:] or times), **today})
        out["speed"] = {"rounds": rounds_out}
        engine = engines["baseline"]
        out["agreement"] = [_first_difference(singles[r][0], engine.generate(prompts[r], steps + 1, eos)["physical_tokens"])
                            for r in range(len(prompts))]
        tuned = runtime.Engine(model, tokenizer, runtime.Knobs(compiled_fixed_cache=True, readback_every=4))
        try:  # (d): today's best exact decode, one request at a time
            rows = []
            for ids in prompts:
                tuned.generate(ids, steps + 1, eos)
                result = tuned.generate(ids, steps + 1, eos)
                rows.append(result["decode_ns"] / 1e9 / max(1, len(result["physical_tokens"]) - 1))
            out["today_tuned_seconds_per_token"] = statistics.median(rows)
        finally:
            tuned.close()
        items = gate or gate_items(tokenizer)
        out["gate"] = {name: {"reference": [reference_nll(engine, prefix, scored) for prefix, scored in group],
                              "plan": plan_nll(plan, model, group, KB1M_PROMPTS)}
                       for name, group in items.items() if group}
    finally:
        for engine in engines.values():
            engine.close()
    return out


def analyse_kb1_model(result: dict[str, Any]) -> dict[str, Any]:
    """KB1-1's three decisions, as fixed in BACKLOG before data."""
    out: dict[str, Any] = {}
    stacks = result.get("stacks") or {}
    checks = [*stacks.values(), *([result["double_capacity"]] if result.get("double_capacity") else [])]
    out["(a) exact within the plan"] = bool(stacks) and all(v["identical_steps"] == v["steps"] for v in checks)
    rounds = (result.get("speed") or {}).get("rounds") or []
    if rounds:
        ratios = [r["stacked"] / min(r["baseline"], r["compiled"]) for r in rounds]
        ci = _bootstrap_mean(ratios, 2000)
        out["(b) stacked step over grouped round"] = {
            "ratio": statistics.fmean(ratios), "ci95": ci, "passes": ci[1] < 1,
            "stacked_ms": statistics.median(r["stacked"] for r in rounds) * 1000,
            "grouped_round_ms": {k: statistics.median(r[k] for r in rounds) * 1000 for k in ("baseline", "compiled")},
            "alone_ms": (result.get("alone_seconds_per_step") or 0) * 1000}
    gate = result.get("gate") or {}
    if gate:
        from ironmule.numeric_choice import ratio_interval

        verdicts = {name: ratio_interval(g["reference"], g["plan"]) for name, g in gate.items()
                    if len(g.get("reference") or []) >= 4}
        out["(c) quality gate"] = {"paths": verdicts, "passes": bool(verdicts) and all(v["passes"]
                                                                                    for v in verdicts.values())}
    if result.get("today_tuned_seconds_per_token") and result.get("alone_seconds_per_step"):
        out["(d) one request, plan over today's tuned"] = (result["alone_seconds_per_step"]
                                                          / result["today_tuned_seconds_per_token"])
    if result.get("double_capacity") and (result.get("stacks") or {}).get(str(KB1M_PROMPTS)):
        out["double capacity step ratio"] = (result["double_capacity"]["seconds_per_step"]
                                             / result["stacks"][str(KB1M_PROMPTS)]["seconds_per_step"])
    out["agreement_with_today"] = result.get("agreement")
    out["verdict"] = ("all three pass: build the plan's stacked executor, opt-in" if out["(a) exact within the plan"]
                      and (out.get("(b) stacked step over grouped round") or {}).get("passes")
                      and (out.get("(c) quality gate") or {}).get("passes") else "does not pass as a whole")
    return out


# ---------------------------------------------------------------- KB1-2: the plan's single-row speed and its cost curve

KB12_WIDTHS = tuple(range(1, 9))
KB12_STEPS = 32
KB12_READBACK = 4


def plan_variants(model: Any) -> dict[str, tuple["DensePlan", Callable[..., Any]]]:
    """(i) KB1-1's plan; (ii) one output row per warp at every M (each row's arithmetic unchanged); (iii) (ii)
    with the whole step compiled per shape."""
    import functools

    import mlx.core as mx

    base = DensePlan(model)
    spread = DensePlan(model, linear=functools.partial(dense_rows, rows=1))
    return {"kb1_1": (base, base.step), "rows1": (spread, spread.step),
            "rows1_compiled": (spread, mx.compile(spread.step))}


def plan_decode(step: Callable[..., Any], first: Any, positions: Any, cache: Any, steps: int,
                readback: int = KB12_READBACK) -> tuple[float, list[list[int]]]:
    """The plan's decode loop as a server would run it: the pick stays on the card and feeds the next step,
    which is built before the host reads; the host reads every `readback` steps. Seconds per step, tokens."""
    import mlx.core as mx

    tokens, pending, out = first, [], []
    started = time.perf_counter()
    for t in range(steps):
        logits, cache = step(tokens, positions + t, cache)
        tokens = mx.argmax(logits, axis=-1).astype(mx.uint32)
        pending.append(tokens)
        if len(pending) == readback or t == steps - 1:
            mx.eval(*pending)
            out.extend(p.tolist() for p in pending)
            pending = []
        else:
            mx.async_eval(tokens)
    seconds = (time.perf_counter() - started) / steps
    return seconds, [list(col) for col in zip(*out)]


def probe_kb1_2(model: Any, tokenizer: Any, *, steps: int = KB12_STEPS, rounds: int = ROUNDS,
                limit: int | None = None, gate: dict[str, Any] | None = None,
                variants: dict[str, tuple["DensePlan", Callable[..., Any]]] | None = None) -> dict[str, Any]:
    import mlx.core as mx
    from ironmule import runtime

    variants = variants or plan_variants(model)
    prompts = [chat_ids(tokenizer, i, i % 2 == 1, limit) for i in range(max(KB12_WIDTHS))]
    prefills = [_prefill(model, ids) for ids in prompts]
    capacity = max(capacity_for(len(ids), steps) for ids in prompts)

    def stack(plan: "DensePlan", rows: int) -> tuple[Any, Any, Any]:
        cache = plan.new_cache(rows, capacity)
        for r in range(rows):
            cache = plan.load_prefill(cache, r, prefills[r][0])
        first = mx.concatenate([prefills[r][1] for r in range(rows)]).astype(mx.uint32)
        return first, mx.array([len(prompts[r]) for r in range(rows)]), cache

    def digests(plan: "DensePlan", step: Callable[..., Any], rows: int, feed: list[list[int]]) -> list[list[str]]:
        first, positions, cache = stack(plan, rows)
        out = [[] for _ in range(rows)]
        tokens = first
        for t in range(steps):
            logits, cache = step(tokens, positions + t, cache)
            mx.eval(logits)
            for r in range(rows):
                out[r].append(_bits(logits[r]))
            tokens = mx.array([feed[r][t] for r in range(rows)], mx.uint32)
        return out

    result: dict[str, Any] = {"prompt_tokens": [len(ids) for ids in prompts], "variants": {}}
    reference_digests = None
    for name, (plan, step) in variants.items():
        alone = []  # each row alone at its own capacity, fed its own greedy tokens
        for r in range(max(KB12_WIDTHS)):
            first = prefills[r][1].astype(mx.uint32)
            cache = plan.load_prefill(plan.new_cache(1, capacity_for(len(prompts[r]), steps)), 0, prefills[r][0])
            _, picks = plan_decode(step, first, mx.array([len(prompts[r])]), cache, steps)
            alone.append([int(first.item())] + picks[0])
        single = []
        for r in range(max(KB12_WIDTHS)):
            first = prefills[r][1].astype(mx.uint32)
            cache = plan.load_prefill(plan.new_cache(1, capacity_for(len(prompts[r]), steps)), 0, prefills[r][0])
            row, tokens = [], first
            for t in range(steps):
                logits, cache = step(tokens, mx.array([len(prompts[r]) + t]), cache)
                mx.eval(logits)
                row.append(_bits(logits[0]))
                tokens = mx.array([alone[r][t + 1]], mx.uint32)
            single.append(row)
        exact = {}
        for rows in (3, max(KB12_WIDTHS)):
            got = digests(plan, step, rows, [a[1:] for a in alone])
            exact[str(rows)] = sum(x == y for r in range(rows) for x, y in zip(got[r], single[r]))
        entry = {"identical_steps": exact, "steps_per_row": steps}
        if reference_digests is None:
            reference_digests = single
        else:
            entry["identical_to_kb1_1"] = sum(x == y for a, b in zip(single, reference_digests) for x, y in zip(a, b))
        result["variants"][name] = entry
    eos = tuple(sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id]))
    tuned = runtime.Engine(model, tokenizer, runtime.Knobs(compiled_fixed_cache=True, readback_every=4))
    timing: list[dict[str, Any]] = []
    try:
        for ids in prompts:
            tuned.generate(ids, steps + 1, eos)
        for name, (plan, step) in variants.items():  # warm every shape
            for rows in KB12_WIDTHS:
                plan_decode(step, *stack(plan, rows), 4)
        for round_ in range(rounds):
            names = list(variants)
            shift = round_ % len(names)
            row: dict[str, Any] = {"round": round_}
            for name in names[shift:] + names[:shift]:
                plan, step = variants[name]
                row[name] = {str(rows): plan_decode(step, *stack(plan, rows), steps)[0] for rows in KB12_WIDTHS}
            per = []
            for ids in prompts:
                out = tuned.generate(ids, steps + 1, eos)
                per.append(out["decode_ns"] / 1e9 / max(1, len(out["physical_tokens"]) - 1))
            row["today_tuned"] = statistics.fmean(per)
            timing.append(row)
    finally:
        tuned.close()
    result["timing"] = timing
    differs = [n for n, e in result["variants"].items() if e.get("identical_to_kb1_1") not in (None, max(KB12_WIDTHS) * steps)]
    if differs:  # a variant with other bits needs its own gate
        engine = runtime.Engine(model, tokenizer, runtime.Knobs())
        try:
            items = gate or gate_items(tokenizer)
            reference = {k: [reference_nll(engine, prefix, scored) for prefix, scored in group]
                         for k, group in items.items() if group}
            result["gate"] = {n: {k: {"reference": reference[k],
                                      "plan": plan_nll(variants[n][0], model, group, max(KB12_WIDTHS), variants[n][1])}
                                  for k, group in items.items() if group} for n in differs}
        finally:
            engine.close()
    return result


def analyse_kb1_2(result: dict[str, Any]) -> dict[str, Any]:
    """KB1-2's decisions, as fixed in BACKLOG before data."""
    out: dict[str, Any] = {"variants": {}}
    widths = [str(w) for w in KB12_WIDTHS]
    timing = result.get("timing") or []
    for name, entry in (result.get("variants") or {}).items():
        exact = all(v == int(rows) * entry["steps_per_row"] for rows, v in entry["identical_steps"].items())
        curve = {w: statistics.median(t[name][w] for t in timing) * 1000 for w in widths} if timing else {}
        ratios = [t[name]["1"] / t["today_tuned"] for t in timing]
        ci = _bootstrap_mean(ratios, 2000) if ratios else [None, None]
        gate = (result.get("gate") or {}).get(name)
        gate_ok = None
        if gate:
            from ironmule.numeric_choice import ratio_interval
            verdicts = {k: ratio_interval(g["reference"], g["plan"]) for k, g in gate.items() if len(g["reference"]) >= 4}
            gate_ok = all(v["passes"] for v in verdicts.values())
            out.setdefault("gates", {})[name] = verdicts
        out["variants"][name] = {"exact_across_widths": exact, "same_bits_as_kb1_1": entry.get("identical_to_kb1_1"),
                                 "ms_per_step_by_width": curve,
                                 "ms_per_token_by_width": {w: curve[w] / int(w) for w in curve},
                                 "one_row_over_today_tuned": statistics.fmean(ratios) if ratios else None,
                                 "ci95": ci, "gate_passes": gate_ok,
                                 "eligible": exact and gate_ok is not False}
    eligible = {n: v for n, v in out["variants"].items() if v["eligible"] and v["ci95"][0] is not None}
    best = min(eligible, key=lambda n: eligible[n]["one_row_over_today_tuned"]) if eligible else None
    out["best"] = best
    out["today_tuned_ms"] = statistics.median(t["today_tuned"] for t in timing) * 1000 if timing else None
    out["(d) single-row regression"] = (None if best is None else
                                        "closed" if eligible[best]["ci95"][1] < 1 else
                                        f"stands: {eligible[best]['one_row_over_today_tuned']:.3f}x today's tuned")
    return out


def environment() -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for name in ("ironmule", "mlx", "mlx-lm", "mlx-cuda"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        gpus = ""
    return {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "versions": versions, "gpus": gpus.strip().splitlines(), "nsys": shutil.which("nsys") is not None}


def probe(model_id: str, out: Path | None, *, smoke: bool = False, loaded: tuple[Any, Any, Any] | None = None
          ) -> dict[str, Any]:
    """Every part, in order of risk: the factorial (where a CUDA graph once crashed, CUDA1) comes last, and the
    report is saved after each part."""
    from mlx.utils import tree_flatten

    report: dict[str, Any] = {"schema": "ironmule.cuda_probe.v1", "model": model_id, "smoke": smoke,
                              "started_unix": time.time(), "env": environment()}

    def save() -> None:
        if out is not None:
            out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")

    if loaded is None:
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        path = snapshot_download(model_id, revision=REVISION if model_id == MODEL else None,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                 "tokenizer*"])
        plain, tokenizer = load(path)
        fused, _ = load(path)
        report["revision"] = Path(path).name
    else:
        plain, fused, tokenizer = loaded
    report["weights_bytes"] = sum(v.nbytes for _, v in tree_flatten(plain.parameters()))
    report["layers"] = len(plain.model.layers)
    repeats, limit = (2, 24) if smoke else (REPEATS, None)
    save()
    partial: list[dict[str, Any]] = []

    def progress(row: dict[str, Any]) -> None:  # a crash mid-factorial (CUDA1) keeps the rows so far
        partial.append(row)
        if len(partial) % len(PROMPT_SET) == 0:
            report["factorial"] = {"runs": partial, "partial": True}
            save()

    parts: list[tuple[str, Callable[[], Any]]] = [
        ("linear", lambda: probe_linear(plain, (1, 2) if smoke else ROWS, repeats)),
        ("attention", lambda: probe_attention(plain, (16,) if smoke else ATTENTION_LENGTHS,
                                              (1, 2) if smoke else ATTENTION_ROWS, repeats)),
        ("overlap", lambda: probe_overlap(plain, tokenizer, 4 if smoke else TOKENS, limit, 2 if smoke else OVERLAP_PAIRS)),
        ("batching", lambda: probe_batching(plain, tokenizer, (8, 12) if smoke else BATCH_LENGTHS,
                                            (1, 2) if smoke else BATCHES, 3 if smoke else BATCH_STEPS)),
        ("factorial", lambda: probe_factorial((plain, fused), tokenizer, 1 if smoke else ROUNDS,
                                              4 if smoke else TOKENS, limit, progress)),
    ]
    for name, part in parts:
        started = time.time()
        try:
            report[name] = part()
        except Exception as exc:  # noqa: BLE001 - a failed part is recorded and the next one still runs
            report[name] = {"error": repr(exc)}
        report.setdefault("seconds", {})[name] = time.time() - started
        save()
    report["finished_unix"] = time.time()
    report["analysis"] = analyse(report)
    save()
    return report


# ---------------------------------------------------------------- decisions, fixed before data

def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(v) for v in values))


def _bootstrap(logs: list[float], resamples: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(math.exp(statistics.fmean(rng.choice(logs) for _ in logs)) for _ in range(resamples))
    return [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]]


def _bootstrap_mean(values: list[float], resamples: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(resamples))
    return [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]]


def analyse_factorial(factorial: dict[str, Any], resamples: int = RESAMPLES) -> dict[str, Any]:
    runs = factorial.get("runs") or []
    if not runs:
        return {"valid": False, "problems": ["no factorial runs"]}
    names = [combo_name(bits) for bits in combos()]
    by: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by.setdefault(row["combo"], []).append(row)
    exact = {name: bool(rows) and all("error" not in r and r.get("first_difference") is None for r in rows)
             for name, rows in by.items()}
    timed = [r for r in runs if r["round"] >= 0 and "error" not in r]
    rounds = sorted({r["round"] for r in timed})
    base: dict[tuple[int, int], float] = {}
    aa: list[float] = []
    for round_ in rounds:
        pairs = []
        for p in sorted({r["prompt"] for r in timed}):
            first_last = sorted((r for r in timed if r["combo"] == "baseline" and r["round"] == round_
                                 and r["prompt"] == p), key=lambda r: r["position"])
            if len(first_last) == 2:
                base[round_, p] = _geomean([first_last[0]["ms_per_token"], first_last[1]["ms_per_token"]])
                pairs.append(first_last[1]["ms_per_token"] / first_last[0]["ms_per_token"])
        if pairs:
            aa.append(_geomean(pairs))
    noise = max((abs(1 - x) for x in aa), default=float("nan"))
    table: dict[str, Any] = {}
    for name in names:
        logs = [math.log(r["ms_per_token"] / base[r["round"], r["prompt"]]) for r in timed
                if r["combo"] == name and (r["round"], r["prompt"]) in base and name != "baseline"]
        if name == "baseline" or not logs:
            continue
        ratio, ci = math.exp(statistics.fmean(logs)), _bootstrap(logs, resamples)
        accepted = [r["acceptance"] for r in by.get(name, []) if r.get("acceptance")]
        table[name] = {"exact": exact.get(name, False), "ratio": ratio, "ratio_ci95": ci,
                       "qualifies": exact.get(name, False) and ci[1] < 1 - noise,
                       "first_differences": sorted({r["first_difference"] for r in by.get(name, [])
                                                    if r.get("first_difference") is not None}),
                       "acceptance": statistics.fmean(accepted) if accepted else None}
    log_ratio = {bits: math.log(table[combo_name(bits)]["ratio"]) if any(bits) else 0.0
                 for bits in combos() if not any(bits) or combo_name(bits) in table}
    effects = interactions = None
    if len(log_ratio) == len(combos()):
        def mean_where(condition: Callable[[tuple[int, ...]], bool]) -> float:
            return statistics.fmean(v for bits, v in log_ratio.items() if condition(bits))
        effects = {f: math.exp(mean_where(lambda b, i=i: b[i] == 1) - mean_where(lambda b, i=i: b[i] == 0))
                   for i, f in enumerate(FACTORS)}
        interactions = {}
        for i, j in itertools.combinations(range(len(FACTORS)), 2):
            cells = {(x, y): mean_where(lambda b, x=x, y=y: b[i] == x and b[j] == y) for x in (0, 1) for y in (0, 1)}
            interactions[f"{FACTORS[i]}*{FACTORS[j]}"] = math.exp(
                (cells[1, 1] - cells[1, 0] - cells[0, 1] + cells[0, 0]) / 2)
    qualifying = sorted((v["ratio"], k) for k, v in table.items() if v["qualifies"])
    baseline_rows = [r for r in timed if r["combo"] == "baseline"]
    prompts = factorial.get("prompt_tokens") or []
    long_prompts = {p for p, n in enumerate(prompts) if n >= 512}
    return {"valid": bool(table), "problems": [] if exact.get("baseline", False) else ["baseline not reproducible"],
            "aa": aa, "noise": noise, "combos": table, "main_effects": effects, "interactions": interactions,
            "best_qualifying": qualifying[0][1] if qualifying else None,
            "baseline_ms_short": statistics.fmean(r["ms_per_token"] for r in baseline_rows
                                                  if r["prompt"] not in long_prompts) if baseline_rows else None,
            "baseline_ms_long": statistics.fmean(r["ms_per_token"] for r in baseline_rows
                                                 if r["prompt"] in long_prompts)
            if any(r["prompt"] in long_prompts for r in baseline_rows) else None,
            "errors": factorial.get("errors") or {}}


PHASE_TOKENS = (16, 64, 256, 1024)  # output lengths the phase table is scored at; beyond 64 a model, not a measurement


def analyse_phases(factorial: dict[str, Any], resamples: int = RESAMPLES) -> dict[str, Any]:
    """Post hoc, added 2026-10-29 while HW3 ran and before its data was read (BACKLOG RL7): it opens RL7 and
    decides nothing in HW3. Prefill and decode separately for every exact combination, short and long prompts
    apart; the cost of a request with n output tokens as 2 x prefill + n x decode per token (the product's
    time in system plus time to first content when nothing queues); and the headroom of a table that picks the
    cheapest exact combination per context (prompt class, n) over the best single combination, held out by
    leaving one prompt out, against the same cost's A/A noise."""
    runs = [r for r in factorial.get("runs") or [] if r["round"] >= 0 and "error" not in r]
    prompts = factorial.get("prompt_tokens") or []
    if not runs or not prompts:
        return {"valid": False, "problems": ["no factorial runs"]}
    exact = {r["combo"] for r in factorial["runs"]} - {r["combo"] for r in factorial["runs"]
                                                       if "error" in r or r.get("first_difference") is not None}
    klass = {p: ("long" if n >= 512 else "short") for p, n in enumerate(prompts)}
    cell: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
    for r in runs:
        if r["combo"] in exact:
            cell.setdefault((r["combo"], r["prompt"], r["round"]), []).append((r["prefill_ms"], r["ms_per_token"]))
    mean = {key: (statistics.fmean(v[0] for v in values), statistics.fmean(v[1] for v in values))
            for key, values in cell.items()}  # the baseline's two runs per round averaged

    def cost(prefill: float, decode: float, n: int) -> float:
        return 2 * prefill + n * decode

    # A/A: the baseline's last run of a round against its first, per prompt, in the same composite cost
    aa = []
    for round_ in sorted({r["round"] for r in runs}):
        for n in PHASE_TOKENS:
            pairs = []
            for p in range(len(prompts)):
                rows = sorted((r for r in runs if r["combo"] == "baseline" and r["round"] == round_
                               and r["prompt"] == p), key=lambda r: r["position"])
                if len(rows) == 2:
                    pairs.append(cost(rows[1]["prefill_ms"], rows[1]["ms_per_token"], n)
                                 / cost(rows[0]["prefill_ms"], rows[0]["ms_per_token"], n))
            if pairs:
                aa.append(_geomean(pairs))
    noise = max((abs(1 - x) for x in aa), default=float("nan"))
    combos_ = sorted({key[0] for key in mean})
    rounds = sorted({key[2] for key in mean})
    phases: dict[str, dict[str, dict[str, float]]] = {}
    for c in combos_:
        for k in ("short", "long"):
            ratios = [(mean[c, p, r][0] / mean["baseline", p, r][0], mean[c, p, r][1] / mean["baseline", p, r][1])
                      for p in range(len(prompts)) if klass[p] == k for r in rounds
                      if (c, p, r) in mean and ("baseline", p, r) in mean]
            if ratios:
                phases.setdefault(c, {})[k] = {"prefill_ratio": _geomean([a for a, _ in ratios]),
                                               "decode_ratio": _geomean([b for _, b in ratios])}

    def arm_cost(c: str, p: int, n: int, rounds_: list[int]) -> float | None:
        values = [cost(*mean[c, p, r], n) for r in rounds_ if (c, p, r) in mean]
        return statistics.fmean(values) if values else None

    winners: dict[str, str] = {}
    held: list[tuple[float, float]] = []  # (table cost, single-arm cost) per held-out (prompt, round, n)
    for p in range(len(prompts)):
        others = [q for q in range(len(prompts)) if q != p]
        for n in PHASE_TOKENS:
            same = [q for q in others if klass[q] == klass[p]]

            def total(c: str, group: list[int]) -> float:
                values = [arm_cost(c, q, n, rounds) for q in group]
                return sum(values) if all(v is not None for v in values) else float("inf")

            table = min(combos_, key=lambda c: (total(c, same), c))
            single = min(combos_, key=lambda c: (sum(total(c, [q]) for q in others), c))
            for r in rounds:
                if (table, p, r) in mean and (single, p, r) in mean:
                    held.append((cost(*mean[table, p, r], n), cost(*mean[single, p, r], n)))
    for k in ("short", "long"):
        for n in PHASE_TOKENS:
            group = [p for p in range(len(prompts)) if klass[p] == k]
            if group:
                winners[f"{k}/{n}"] = min(combos_, key=lambda c: (sum(arm_cost(c, q, n, rounds) or float("inf")
                                                                      for q in group), c))
    headroom = 1 - sum(t for t, _ in held) / sum(s for _, s in held) if held else float("nan")
    rng = random.Random(0)
    draws = sorted(1 - sum(t for t, _ in sample) / sum(s for _, s in sample)
                   for sample in ([rng.choice(held) for _ in held] for _ in range(resamples))) if held else []
    ci = [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]] if draws else [None, None]
    one_winner = len(set(winners.values())) == 1
    return {"label": "post hoc: opens RL7, decides nothing in HW3", "noise": noise, "exact_combos": len(exact),
            "phases": phases, "winners": winners, "headroom": headroom, "headroom_ci95": ci,
            "verdict": ("one combination for every context: a single profile, no table and no learner"
                        if one_winner or (ci[0] is not None and ci[0] <= noise)
                        else "a per-context table beats one profile: RL7 continues")}


def analyse(report: dict[str, Any], resamples: int = RESAMPLES) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "ironmule.cuda_probe.analysis.v1"}
    factorial = report.get("factorial") or {}
    f = analyse_factorial(factorial, resamples) if "runs" in factorial else {"valid": False,
                                                                             "problems": [str(factorial.get("error"))]}
    out["factorial"] = f
    if "runs" in factorial:
        out["phases"] = analyse_phases(factorial, resamples)
    single = {name: (f.get("combos") or {}).get(name) for name in FACTORS}
    out["decisions"] = decisions = {}
    compiled = single.get("compiled_fixed_cache")
    decisions["compile (8)"] = None if compiled is None else (
        "qualifies" if compiled["qualifies"] else
        "closed on this card: not exact" if not compiled["exact"] else "closed on this card: not faster than noise")
    speculation = single.get("speculate_k")
    decisions["speculation (10)"] = None if speculation is None else (
        ("exact" if speculation["exact"] else "not exact: verifying drafts changes tokens on this card")
        + f", acceptance {speculation['acceptance']}")
    overlap = report.get("overlap")
    if isinstance(overlap, list) and overlap:
        # Resolved: every alternating pair saved time at that context (a sign test, p = 1/32 with five pairs).
        resolved = [o for o in overlap if o.get("saved") and min(o["saved"]) > 0]
        decisions["overlap (16)"] = {"saved": {("long" if o["long"] else "short"): statistics.median(o["saved"])
                                               for o in overlap if o.get("saved")},
                                     "verdict": "worth building" if resolved else "closed on this card: not resolved"}
    attention = report.get("attention")
    step_long = f.get("baseline_ms_long")
    if isinstance(attention, list) and attention and step_long:
        row = next((a for a in attention if a["length"] == 1024 and a["rows"] == 1), attention[0])
        share = report.get("layers", 0) * (row["masked_seconds"] - row["cut_seconds"]) * 1000 / step_long
        saved = row.get("saved_seconds") or []
        low = _bootstrap_mean(saved, resamples)[0] if saved else float("nan")
        resolved = [a for a in attention if a.get("saved_seconds") and _bootstrap_mean(a["saved_seconds"], resamples)[0] > 0]
        decisions["fused attention (6)"] = {"share_of_long_step": share, "identical": row["identical"],
                                            "saved_seconds_ci95_low": low,
                                            "resolved_at": [[a["length"], a["rows"]] for a in resolved],
                                            "verdict": "worth building" if resolved
                                            else "closed for decode on this card: not resolved"}
    linear = report.get("linear")
    step_short = f.get("baseline_ms_short")
    if isinstance(linear, list) and linear and report.get("weights_bytes"):
        best = max(e["rows"]["1"]["gb_per_s"] for e in linear)
        floor_ms = report["weights_bytes"] / best / 1e9 * 1000
        decisions["bandwidth floor (5, 7)"] = {"best_gb_per_s": best, "floor_ms": floor_ms,
                                               "share_of_short_step": floor_ms / step_short if step_short else None}
        out["batch_invariant_rows"] = {e["layer"]: [int(m) for m, r in e["rows"].items()
                                                    if r["identical_rows"] == int(m)] for e in linear}
    batching = report.get("batching")
    if isinstance(batching, list) and batching:
        stacked = [b for b in batching if b["batch"] > 1]
        if all(b["identical_logits"] == b["batch"] for b in stacked):
            verdict = "batch-invariant: a batched path keeps every answer (class A)"
        elif all(b["equal_tokens"] == b["batch"] for b in stacked):
            verdict = "token-exact here but not invariant (class B)"
        else:
            verdict = "not exact: stacking rows changes tokens"
        step = {(b["length"], b["batch"]): b["seconds_per_step"] for b in batching}
        lever = {str(length): (8 * step[length, 1] / step[length, 8]) if (length, 8) in step else None
                 for length in sorted({b["length"] for b in batching})}
        gains = [v for v in lever.values() if v is not None]
        decisions["true batching (1)"] = {"exactness": verdict, "lever_at_8": lever,
                                          "verdict": None if not gains else
                                          ("lever holds" if max(gains) > 1 else
                                           "closed on this card: stacking makes no token cheaper")}
    return out


def analyse_hw4(report: dict[str, Any], resamples: int = RESAMPLES) -> dict[str, Any]:
    """HW4's decisions and RL7's first two kill rules, as fixed in BACKLOG before data."""
    out: dict[str, Any] = {"schema": "ironmule.cuda_probe.hw4.analysis.v1"}
    capture = report.get("class_a") or {}
    verdicts = class_a_sets(capture) if "sets" in capture else {}
    class_a = sorted(n for n, v in verdicts.items() if v.get("class_a") and n != "baseline")
    out["class_a"] = {"verdicts": verdicts, "sets": class_a}
    decisions: dict[str, Any] = {}
    decisions["RL7 kill 1 (class A)"] = ("knob axis empty: fewer than two class-A sets besides the baseline"
                                         if len(class_a) < 2 else f"{len(class_a)} class-A sets: continue")
    # separability: 2 x 2 x 2 on log total cost, per (prompt, round) cell against that round's baseline
    sep = report.get("separability") or {}
    runs = [r for r in sep.get("runs") or [] if r["round"] >= 0 and "error" not in r]
    if runs:
        exact = {r["arm"] for r in sep["runs"]} - {r["arm"] for r in sep["runs"]
                                                   if "error" in r or r.get("first_difference") is not None}
        cells: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for r in runs:
            cells.setdefault((r["arm"], r["prompt"], r["round"]), []).append(r)

        def value(arm: str, p: int, r: int, n: int) -> float | None:
            rows = cells.get((arm, p, r))
            if not rows:
                return None
            return statistics.fmean(2 * x["prefill_ms"] + n * x["ms_per_token"] for x in rows)

        keys = sorted({(p, r) for _, p, r in cells})
        factors = ("head_skip_prefill", "fuse_projections", "decode_winner")

        def arm_of(bits: tuple[int, ...]) -> str:
            return "+".join(f for f, b in zip(factors, bits) if b) or "baseline"

        separability: dict[str, Any] = {"exact_arms": sorted(exact)}
        for n in (16, 64, 256):
            contrasts: dict[str, list[float]] = {}
            for p, r in keys:
                logs = {bits: value(arm_of(bits), p, r, n) for bits in itertools.product((0, 1), repeat=3)}
                if any(v is None for v in logs.values()):
                    continue
                logs = {b: math.log(v) for b, v in logs.items()}
                for i, j in ((0, 2), (1, 2), (0, 1)):
                    def at(x: int, y: int) -> float:
                        return statistics.fmean(v for b, v in logs.items() if b[i] == x and b[j] == y)
                    contrasts.setdefault(f"{factors[i]}*{factors[j]}", []).append(
                        (at(1, 1) - at(1, 0) - at(0, 1) + at(0, 0)) / 2)
            separability[str(n)] = {name: {"interaction": math.exp(statistics.fmean(v)),
                                           "ci95": [math.exp(x) for x in _bootstrap_mean(v, resamples)]}
                                    for name, v in contrasts.items()}
        fixed = [(value("prefill_into_fixed", p, r, 0) or 0) / value("baseline", p, r, 0) for p, r in keys
                 if value("prefill_into_fixed", p, r, 0) and value("baseline", p, r, 0)]
        separability["prefill_into_fixed"] = {"exact": "prefill_into_fixed" in exact,
                                              "prefill_ratio": _geomean(fixed) if fixed else None}
        coupled = [f"{n}:{k}" for n in ("16", "64", "256") for k, v in separability.get(n, {}).items()
                   if "decode_winner" in k and not (v["ci95"][0] <= 1 <= v["ci95"][1])]
        separability["verdict"] = ("prefill and decode knobs couple: " + ", ".join(coupled)) if coupled \
            else "separable: the best combination is the best prefill set plus the best decode set"
        out["separability"] = separability
    # widths: seconds per decode round, a further session's cost, and RL7 kill 2 (dominance)
    widths = [w for w in report.get("widths") or [] if isinstance(w, dict) and "error" not in w]
    if widths:
        step: dict[tuple[str, int, int], float] = {(w["set"], w["width"], w["round"]): w["seconds"] / max(w["tokens"])
                                                   for w in widths}
        sets_ = sorted({w["set"] for w in widths})
        rounds = sorted({w["round"] for w in widths})
        per_set = {}
        for name in sets_:
            slopes = []
            for r in rounds:
                points = [(wd, step[name, wd, r]) for wd in WIDTHS_HW4 if (name, wd, r) in step]
                if len(points) >= 2:
                    mx_ = statistics.fmean(x for x, _ in points)
                    my = statistics.fmean(y for _, y in points)
                    den = sum((x - mx_) ** 2 for x, _ in points)
                    slopes.append(sum((x - mx_) * (y - my) for x, y in points) / den if den else 0.0)
            per_set[name] = {"seconds_per_round": {str(wd): statistics.median(step[name, wd, r] for r in rounds
                                                                              if (name, wd, r) in step)
                                                   for wd in WIDTHS_HW4},
                             "seconds_per_extra_session": statistics.median(slopes) if slopes else None}
        flips = []
        for a, b in itertools.combinations(sets_, 2):
            signs = {}
            for wd in WIDTHS_HW4:
                diffs = [step[a, wd, r] - step[b, wd, r] for r in rounds if (a, wd, r) in step and (b, wd, r) in step]
                if diffs and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
                    signs[wd] = 1 if diffs[0] > 0 else -1
            if len(set(signs.values())) > 1:
                flips.append([a, b, signs])
        digests = {}
        for w in widths:
            digests.setdefault(w["width"], {}).setdefault(w["set"], set()).add(tuple(w["digests"]))
        same = all(len({d for ds in by_set.values() for d in ds}) == 1 for by_set in digests.values())
        out["widths"] = {"sets": per_set, "flips": flips, "identical_answers_across_sets": same}
        decisions["RL7 kill 2 (dominance)"] = (
            "knob sets and scheduling separate: refit the pool model per set, re-rank the fixed schedules, no learner"
            if not flips else f"{len(flips)} pair(s) change order with the width: continue")
    compile_rows = [c for c in report.get("compile") or [] if isinstance(c, dict)]
    if compile_rows:
        out["compile_seconds"] = {name: statistics.median(c["first_seconds"] - c["again_seconds"]
                                                          for c in compile_rows if c["set"] == name)
                                  for name in sorted({c["set"] for c in compile_rows})}
    floor = [f for f in report.get("prefill_floor") or [] if isinstance(f, dict)]
    if floor and report.get("weights_bytes") and runs:
        best = max(f["tflop_per_s"] for f in floor)
        long_prompts = [p for p, n in enumerate(sep.get("prompt_tokens") or []) if n >= 512]
        measured = [x["prefill_ms"] for x in runs if x["arm"] == "baseline" and x["prompt"] in long_prompts]
        if measured and long_prompts:
            tokens = statistics.fmean(sep["prompt_tokens"][p] for p in long_prompts)
            floor_ms = 2 * (report["weights_bytes"] / 2) * tokens / (best * 1e12) * 1000
            out["gap"] = {"prefill_long_ms": statistics.fmean(measured), "prefill_floor_ms": floor_ms,
                          "best_tflop_per_s": best,
                          "prefill_gap_share": 1 - floor_ms / statistics.fmean(measured),
                          "remedy_classes": {"head_skip_prefill / fusion": "class A if HW4 says so",
                                             "float16 tensor cores (U2)": "class B: a quality-gated plan"}}
    out["decisions"] = decisions
    return out


def probe_kb12_report(model_id: str, out: Path | None, *, smoke: bool = False, loaded: tuple[Any, Any] | None = None,
                      variants: dict[str, tuple["DensePlan", Callable[..., Any]]] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"schema": "ironmule.cuda_probe.kb12.v1", "model": model_id, "smoke": smoke,
                              "started_unix": time.time(), "env": environment()}
    if loaded is None:
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        path = snapshot_download(model_id, revision=REVISION if model_id == MODEL else None,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
                                                 "tokenizer*"])
        model, tokenizer = load(path)
        report["revision"] = Path(path).name
    else:
        model, tokenizer = loaded
    try:
        report["kb12"] = probe_kb1_2(model, tokenizer, steps=4 if smoke else KB12_STEPS, rounds=1 if smoke else ROUNDS,
                                     limit=24 if smoke else None, variants=variants,
                                     gate=gate_items(tokenizer, 4, 12, (16, 6, 4)) if smoke else None)
        report["analysis"] = analyse_kb1_2(report["kb12"])
    except Exception as exc:  # noqa: BLE001 - recorded
        report["kb12"] = {"error": repr(exc)}
        report["analysis"] = {"error": repr(exc)}
    report["finished_unix"] = time.time()
    if out is not None:
        out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    return report


# ---------------------------------------------------------------- Kaggle

_BOOTSTRAP = '''"""IronMule HW3 and TB1 on this Kaggle session; written by tools/cuda_probe.py."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ironmule[cuda] @ git+{repository}@{commit}"], check=True)
'''


def write_kernel(directory: Path, owner: str, commit: str, plan: str = "hw3",
                 decode_winner: dict[str, Any] | None = None) -> Path:
    """A private script kernel that installs IronMule at `commit` and runs this probe's `plan`; the report lands
    in the kernel's output as `cuda_probe_report.json` (HW3) or `cuda_probe_hw4_report.json` (HW4)."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).read_text().replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    if plan == "hw4":
        if decode_winner is None:
            raise ValueError("HW4 needs HW3's decode winner")
        arguments = ["--out", "/kaggle/working/cuda_probe_hw4_report.json", "--plan", "hw4",
                     "--decode-winner", json.dumps(decode_winner, sort_keys=True)]
    elif plan in ("kb1", "kb1m", "kb12"):
        arguments = ["--out", f"/kaggle/working/cuda_probe_{plan}_report.json", "--plan", plan]
    else:
        arguments = ["--out", "/kaggle/working/cuda_probe_report.json"]
    entry = f"\nsys.exit(main({arguments!r}))\n"
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit) + body + entry)
    name = "ironmule-cuda-probe" + ("" if plan == "hw3" else f"-{plan}")
    metadata = {"id": f"{owner}/{name}", "title": name, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    return directory


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the report here")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes: checks this tool, measures nothing")
    parser.add_argument("--analyse", type=Path, metavar="REPORT", help="print the analysis of a finished report")
    parser.add_argument("--write-kernel", type=Path, metavar="DIR", help="write a Kaggle kernel folder instead")
    parser.add_argument("--owner")
    parser.add_argument("--commit")
    parser.add_argument("--plan", choices=("hw3", "hw4", "kb1", "kb1m", "kb12"), default="hw3")
    parser.add_argument("--decode-winner", help="HW4: HW3's winning decode knobs as JSON, fixed before HW4's data")
    args = parser.parse_args(None if argv is None else list(argv))
    winner = None if args.decode_winner is None else json.loads(args.decode_winner)
    if args.plan == "hw4" and not isinstance(winner, dict):
        parser.error("--plan hw4 needs --decode-winner")
    if args.write_kernel is not None:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        print(json.dumps({"kernel": str(write_kernel(args.write_kernel, args.owner, args.commit, args.plan, winner))}))
        return 0
    if args.analyse is not None:
        report = json.loads(args.analyse.read_text())
        schema = report.get("schema")
        result = (analyse_hw4(report) if schema == "ironmule.cuda_probe.hw4.v1" else
                  {"kb1_0": analyse_kb1(report.get("kb1") or []),
                   "kb1_1": analyse_kb1_model(report["kb1_model"]) if "stacks" in (report.get("kb1_model") or {})
                   else report.get("kb1_model")} if schema == "ironmule.cuda_probe.kb1.v1" else
                  analyse_kb1_2(report.get("kb12") or {}) if schema == "ironmule.cuda_probe.kb12.v1" else analyse(report))
        print(json.dumps(result, indent=1, sort_keys=True, default=str))
        return 0
    if args.out is None:
        parser.error("--out is required to run")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one card; set before MLX loads
    if args.plan == "kb12":
        report = probe_kb12_report(args.model, args.out, smoke=args.smoke)
        print(json.dumps({"report": str(args.out), "analysis": report["analysis"]}, default=str), flush=True)
        return 0
    if args.plan in ("kb1", "kb1m"):
        report = probe_kb1_report(args.model, args.out, smoke=args.smoke, model_level=args.plan == "kb1m")
        print(json.dumps({"report": str(args.out), "analysis": report.get("analysis_model") or
                          {k: v for k, v in report["analysis"].items() if k != "layers"}}, default=str), flush=True)
        return 0
    if args.plan == "hw4":
        report = probe_hw4(args.model, args.out, winner, smoke=args.smoke)
        print(json.dumps({"report": str(args.out), "decisions": report["analysis"]["decisions"]}, default=str),
              flush=True)
        return 0
    report = probe(args.model, args.out, smoke=args.smoke)
    print(json.dumps({"report": str(args.out), "decisions": report["analysis"]["decisions"]}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
