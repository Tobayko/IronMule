#!/usr/bin/env python3
"""Two independent decode requests through one Gemma 12B, sharing only weight loads.

Prefill is untouched and runs per request through the ordinary model call. Decoding then
steps both requests together through the real submodules: each keeps its own KV cache,
its own attention, its own residuals and its own output. Only the five `K=3840`
projections per layer are executed once for both.

Three arms, so scheduling and weight reuse can be told apart:

* `separate` runs each request through the unmodified model call, one after the other.
* `paired_separate` runs the same two-request loop but calls the projections
  individually, so it carries the loop's scheduling without sharing anything.
* `paired_shared` is the same loop with the shared-weight kernel.

Anything that differs by one bit in logits, tokens or KV state locks the variant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.gemma3_text import clip_residual

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b45_shared_weight_kernel import TARGET_K, run_shared  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE = 0.90
PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
)


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def _project(module: nn.QuantizedLinear, vectors: list[mx.array], *, shared: bool) -> list:
    """One call for both requests, or one call each. Same kernel either way."""

    out_features = int(module.weight.shape[0])
    flat = [v.reshape(1, TARGET_K) for v in vectors]
    if shared:
        results = run_shared(module.weight, module.scales, module.biases, flat, out_features)
    else:
        results = [
            run_shared(module.weight, module.scales, module.biases, [v], out_features)[0]
            for v in flat
        ]
    return [r.reshape(1, 1, out_features) for r in results]


def _attention(layer, xs: list[mx.array], masks: list, caches: list, *, shared: bool):
    attn = layer.self_attn
    queries = _project(attn.q_proj, xs, shared=shared)
    keys = _project(attn.k_proj, xs, shared=shared)
    values = _project(attn.v_proj, xs, shared=shared)
    outputs = []
    for index in range(len(xs)):
        B, L, _ = xs[index].shape
        q = queries[index].reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = keys[index].reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = values[index].reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        cache = caches[index]
        q = attn.rope(q, offset=cache.offset)
        k = attn.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=attn.scale, mask=masks[index]
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        outputs.append(attn.o_proj(out))
    return outputs


def _block(layer, hs: list[mx.array], masks: list, caches: list, *, shared: bool):
    normed = [layer.input_layernorm(h) for h in hs]
    attended = _attention(layer, normed, masks, caches, shared=shared)
    hs = [clip_residual(h, layer.post_attention_layernorm(r)) for h, r in zip(hs, attended)]
    pre = [layer.pre_feedforward_layernorm(h) for h in hs]
    gate = _project(layer.mlp.gate_proj, pre, shared=shared)
    up = _project(layer.mlp.up_proj, pre, shared=shared)
    fed = [layer.mlp.down_proj(nn.gelu_approx(g) * u) for g, u in zip(gate, up)]
    return [clip_residual(h, layer.post_feedforward_layernorm(r)) for h, r in zip(hs, fed)]


def _paired_step(language, tokens: list[mx.array], caches: list, *, shared: bool):
    inner = language.model
    hs = []
    for token in tokens:
        h = inner.embed_tokens(token)
        h = h * mx.array(inner.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        hs.append(h)
    masks = []
    for cache in caches:
        global_mask = create_attention_mask(hs[0], cache[inner.sliding_window_pattern - 1])
        sliding = (
            create_attention_mask(hs[0], cache[0], window_size=inner.window_size)
            if inner.sliding_window_pattern > 1 else None
        )
        masks.append((global_mask, sliding))
    for index, layer in enumerate(inner.layers):
        is_global = index % inner.sliding_window_pattern == inner.sliding_window_pattern - 1
        step_masks = [pair[0] if is_global else pair[1] for pair in masks]
        hs = _block(layer, hs, step_masks, [cache[index] for cache in caches], shared=shared)
    return [language.lm_head(inner.norm(h)[:, -1, :]) for h in hs]


def _prefill(language, tokenizer, prompt: str):
    cache = language.make_cache()
    ids = mx.array(tokenizer.encode(prompt))[None]
    out = language.model(ids, cache=cache)
    logits = language.lm_head(out[:, -1, :])
    mx.eval(logits)
    return cache, logits


def _run_separate(language, tokenizer, steps: int) -> dict:
    """Both requests one after the other, entirely on the unmodified path."""

    results = []
    began = time.perf_counter_ns()
    for prompt in PROMPTS:
        cache, logits = _prefill(language, tokenizer, prompt)
        tokens, digest = [], hashlib.sha256()
        for _ in range(steps):
            token = int(mx.argmax(logits, axis=-1).item())
            tokens.append(token)
            digest.update(bytes(memoryview(logits)))
            out = language.model(mx.array([[token]]), cache=cache)
            logits = language.lm_head(out[:, -1, :])
            mx.eval(logits)
        results.append({"tokens": tokens, "logits_sha256": digest.hexdigest(),
                        "cache_sha256": _cache_digest(cache)})
    return {"wall_ns": time.perf_counter_ns() - began, "requests": results}


def _run_paired(language, tokenizer, steps: int, *, shared: bool) -> dict:
    began = time.perf_counter_ns()
    caches, logits = [], []
    for prompt in PROMPTS:
        cache, first = _prefill(language, tokenizer, prompt)
        caches.append(cache)
        logits.append(first)
    tokens = [[] for _ in PROMPTS]
    digests = [hashlib.sha256() for _ in PROMPTS]
    for _ in range(steps):
        chosen = []
        for index, value in enumerate(logits):
            token = int(mx.argmax(value, axis=-1).item())
            tokens[index].append(token)
            digests[index].update(bytes(memoryview(value)))
            chosen.append(mx.array([[token]]))
        logits = _paired_step(language, chosen, caches, shared=shared)
        mx.eval(logits)
    return {
        "wall_ns": time.perf_counter_ns() - began,
        "requests": [
            {"tokens": tokens[i], "logits_sha256": digests[i].hexdigest(),
             "cache_sha256": _cache_digest(caches[i])}
            for i in range(len(PROMPTS))
        ],
    }


def _cache_digest(cache) -> str:
    state = hashlib.sha256()
    for layer in cache:
        for part in (layer.keys, layer.values):
            if part is not None:
                mx.eval(part)
                state.update(bytes(memoryview(part[..., : layer.offset, :])))
    return state.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    language = getattr(model, "language_model", model)

    arms = {
        "separate": lambda: _run_separate(language, tokenizer, args.steps),
        "paired_separate": lambda: _run_paired(language, tokenizer, args.steps, shared=False),
        "paired_shared": lambda: _run_paired(language, tokenizer, args.steps, shared=True),
    }

    # Gate one, outside the timed region.
    truth = arms["separate"]()["requests"]
    identity = {}
    for name in ("paired_separate", "paired_shared"):
        got = arms[name]()["requests"]
        identity[name] = {
            "tokens": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
            "logits": [a["logits_sha256"] == b["logits_sha256"] for a, b in zip(truth, got)],
            "kv_cache": [a["cache_sha256"] == b["cache_sha256"] for a, b in zip(truth, got)],
        }
    clean = all(all(all(values) for values in row.values()) for row in identity.values())
    print(json.dumps({"bit_identical": clean, "identity": identity}, sort_keys=True))
    if not clean:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"schema": "ironmule.two_request_model.v1",
             "experiment_id": args.out.stem, "agent": "claude", "kind": "paired_ab",
             "status": "locked", "decision": "MODEL NO-GO",
             "reason": "the paired loop is not bit-identical to the separate path",
             "identity": identity,
             "observed_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True), encoding="utf-8")
        return 1

    for _ in range(args.warmup):
        for run in arms.values():
            run()

    before = _vm()
    timings = {name: [] for name in arms}
    order = list(arms)
    for block in range(args.blocks):
        rotation = order[block % len(order):] + order[: block % len(order)]
        for name in rotation:
            timings[name].append(arms[name]()["wall_ns"])
    after = _vm()
    paging = {key: after.get(key, 0) - value for key, value in before.items()}

    def interval(numerator, denominator):
        ratios = [n / d for n, d in zip(timings[numerator], timings[denominator])]
        rng = random.Random(20260909)
        draws = sorted(median([ratios[rng.randrange(len(ratios))] for _ in ratios])
                       for _ in range(10000))
        return {"median": median(ratios), "min": min(ratios), "max": max(ratios),
                "ci95_low": draws[int(0.025 * len(draws))],
                "ci95_high": draws[int(0.975 * len(draws)) - 1],
                "blocks_below_gate": sum(1 for value in ratios if value <= GATE)}

    shared = interval("paired_shared", "separate")
    scheduling = interval("paired_separate", "separate")
    sharing_only = interval("paired_shared", "paired_separate")
    meets = shared["ci95_high"] <= GATE
    decision = ("MODEL GO" if meets and paging.get("Swapouts", 0) == 0
                else ("BLOCKED" if paging.get("Swapouts", 0) else "MODEL NO-GO"))

    record = {
        "schema": "ironmule.two_request_model.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "paired_ab",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "do two requests sharing weight loads finish 10 per cent sooner together",
        "decision": decision,
        "model": args.model,
        "gate": GATE,
        "preregistered": {
            "arms": list(arms), "blocks": args.blocks, "steps_per_request": args.steps,
            "order": "rotated so every arm meets every position",
            "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
            "success_rule": "paired_shared / separate upper bound at or below 0.90",
            "abort_rule": "swapouts during the run make it not evaluable",
            "budget": "one run, no extension",
        },
        "identity": identity,
        "bit_identical": clean,
        "arm_median_ns": {name: median(values) for name, values in timings.items()},
        "arm_block_ns": timings,
        "shared_over_separate": shared,
        "scheduling_only_paired_separate_over_separate": scheduling,
        "sharing_effect_shared_over_paired_separate": sharing_only,
        "meets_gate": meets,
        "paging_during_run": paging,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "limits": [
            "two fixed prompts, greedy, one machine, one MLX build",
            "prefill runs per request on the unmodified path and is inside the wall time",
            "no artificial wait is used to form the pair; both requests are present",
        ],
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key in ("decision", "meets_gate", "arm_median_ns",
                                 "shared_over_separate",
                                 "scheduling_only_paired_separate_over_separate",
                                 "sharing_effect_shared_over_paired_separate",
                                 "paging_during_run", "peak_memory_bytes")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
