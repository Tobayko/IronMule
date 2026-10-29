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

The report keeps digests of token ids, counts, times and logit differences, never prompt or answer text.
To run it on Kaggle (T4 x2; it uses the first card):

    python tools/cuda_probe.py --write-kernel kernel --owner <user> --commit <sha>
    kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
    kaggle kernels output <user>/ironmule-cuda-probe -p results
    python tools/cuda_probe.py --analyse results/cuda_probe_report.json
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


def analyse(report: dict[str, Any], resamples: int = RESAMPLES) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "ironmule.cuda_probe.analysis.v1"}
    factorial = report.get("factorial") or {}
    f = analyse_factorial(factorial, resamples) if "runs" in factorial else {"valid": False,
                                                                             "problems": [str(factorial.get("error"))]}
    out["factorial"] = f
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


# ---------------------------------------------------------------- Kaggle

_BOOTSTRAP = '''"""IronMule HW3 and TB1 on this Kaggle session; written by tools/cuda_probe.py."""
from __future__ import annotations
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ironmule[cuda] @ git+{repository}@{commit}"], check=True)
'''


def write_kernel(directory: Path, owner: str, commit: str) -> Path:
    """A private script kernel that installs IronMule at `commit` and runs this probe; the report lands in the
    kernel's output as `cuda_probe_report.json`."""
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("commit must be a full 40-character lowercase SHA, so the kernel runs exactly it")
    directory.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).read_text().replace('\nif __name__ == "__main__":\n    sys.exit(main())\n', "")
            .replace("from __future__ import annotations\n", "", 1))
    entry = f"\nsys.exit(main({['--out', '/kaggle/working/cuda_probe_report.json']!r}))\n"
    (directory / "run.py").write_text(_BOOTSTRAP.format(repository=REPOSITORY, commit=commit) + body + entry)
    metadata = {"id": f"{owner}/ironmule-cuda-probe", "title": "ironmule-cuda-probe", "code_file": "run.py",
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
    args = parser.parse_args(None if argv is None else list(argv))
    if args.write_kernel is not None:
        if not args.owner or not args.commit:
            parser.error("--write-kernel needs --owner and --commit")
        print(json.dumps({"kernel": str(write_kernel(args.write_kernel, args.owner, args.commit))}))
        return 0
    if args.analyse is not None:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--out is required to run")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one card; set before MLX loads
    report = probe(args.model, args.out, smoke=args.smoke)
    print(json.dumps({"report": str(args.out), "decisions": report["analysis"]["decisions"]}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
