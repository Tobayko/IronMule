#!/usr/bin/env python3
"""Pad the reduction dimension to 512 so the fast quantised kernel runs, and pay for it.

`B24S` measured that 99.3 to 99.8% of a decode step's GPU time is the quantised
matrix-vector kernel, and that MLX runs its fast variant only when the reduction
dimension is a multiple of 512. `B41K` measured the price on the device: 362 GB/s on the
boundary against 307 either side, a 17.6% difference, while padding 3840 to 4096 costs
6.7% more bytes.

The padding is exact, not approximate. Appending zero packed weights with zero scales
and zero biases makes `dequantize` return exactly 0.0 for the new columns, and the input
is zero there too, so every added product is `0 * 0`. No weight is requantised and no
existing byte is touched. What can still change is the order the kernel reduces in, so
this compares tokens and logits, not only time.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = 512
PROMPT = "Explain in three sentences why unified memory changes how a laptop runs a language model."


class PadCache:
    """One padded copy per width, reused while the same input is still the live one.

    Seven matmuls per layer read only three distinct vectors, so padding once per vector
    instead of once per call removes more than half of the added kernels.
    """

    def __init__(self) -> None:
        self.entries: dict[int, tuple[mx.array, mx.array]] = {}

    def padded(self, x: mx.array, width: int) -> mx.array:
        cached = self.entries.get(width)
        if cached is not None and cached[0] is x:
            return cached[1]
        padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, width - x.shape[-1])])
        self.entries[width] = (x, padded)
        return padded


class PaddedQuantizedLinear(nn.Module):
    """A `QuantizedLinear` whose reduction dimension is padded up to the boundary."""

    def __init__(self, source: nn.QuantizedLinear, target: int, cache: PadCache) -> None:
        super().__init__()
        self._cache = cache
        self.group_size = source.group_size
        self.bits = source.bits
        rows, packed = source.weight.shape
        columns = packed * 32 // self.bits
        self.original_columns = columns
        self.padded_columns = target
        extra = target - columns
        packed_extra = extra * self.bits // 32
        groups_extra = extra // self.group_size
        self.weight = mx.concatenate(
            [source.weight, mx.zeros((rows, packed_extra), dtype=source.weight.dtype)], axis=1
        )
        self.scales = mx.concatenate(
            [source.scales, mx.zeros((rows, groups_extra), dtype=source.scales.dtype)], axis=1
        )
        self.biases = mx.concatenate(
            [source.biases, mx.zeros((rows, groups_extra), dtype=source.biases.dtype)], axis=1
        )

    def __call__(self, x: mx.array) -> mx.array:
        padded = self._cache.padded(x, self.padded_columns)
        return mx.quantized_matmul(
            padded, self.weight, self.scales, self.biases,
            transpose=True, group_size=self.group_size, bits=self.bits,
        )


def _pad_model(module: nn.Module, changed: list[str], cache: PadCache, prefix: str = "") -> int:
    """Replace every off-boundary quantised linear in place; returns how many."""

    count = 0
    for name, child in list(module.children().items()):
        path = f"{prefix}.{name}" if prefix else name
        # `nn.Module` subclasses `dict`, so test for Module before any mapping check.
        if isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, nn.Module):
                    count += _pad_model(item, changed, cache, f"{path}.{index}")
            continue
        if isinstance(child, nn.QuantizedLinear):
            columns = child.weight.shape[1] * 32 // child.bits
            if columns % BOUNDARY:
                target = ((columns + BOUNDARY - 1) // BOUNDARY) * BOUNDARY
                setattr(module, name, PaddedQuantizedLinear(child, target, cache))
                changed.append(f"{path}:{columns}->{target}")
                count += 1
        elif isinstance(child, nn.Module):
            count += _pad_model(child, changed, cache, path)
    return count


def _decode(language, tokenizer, steps: int) -> tuple[list[int], list[float], mx.array]:
    cache = language.make_cache()
    ids = mx.array(tokenizer.encode(PROMPT))[None]
    tokens, times, logits = [], [], None
    for index in range(steps):
        began = time.perf_counter_ns()
        out = language.model(ids, cache=cache)
        logits = language.lm_head(out[:, -1, :])
        token = int(mx.argmax(logits, axis=-1).item())
        times.append(time.perf_counter_ns() - began)
        tokens.append(token)
        if index == 0:
            first_logits = logits
        ids = mx.array([[token]])
    return tokens, times, first_logits


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference, tokenizer = load(args.model)
    reference_language = getattr(reference, "language_model", reference)
    candidate, _ = load(args.model)
    candidate_language = getattr(candidate, "language_model", candidate)
    changed: list[str] = []
    padded = _pad_model(candidate_language, changed, PadCache())
    mx.eval(candidate_language.parameters())
    print(f"padded {padded} quantised linears")

    for _ in range(args.warmup):
        _decode(reference_language, tokenizer, 2)
        _decode(candidate_language, tokenizer, 2)

    # Interleaved and order-balanced: the machine drifts more than the effect.
    arms: dict[str, list[float]] = {"reference": [], "candidate": []}
    tokens: dict[str, list[int]] = {}
    logits: dict[str, mx.array] = {}
    for block in range(args.blocks):
        order = ("reference", "candidate") if block % 2 == 0 else ("candidate", "reference")
        for arm in order:
            language = reference_language if arm == "reference" else candidate_language
            step_tokens, step_times, step_logits = _decode(language, tokenizer, args.steps)
            arms[arm].append(median(step_times))
            tokens[arm] = step_tokens
            logits[arm] = step_logits

    reference_ns = median(arms["reference"])
    candidate_ns = median(arms["candidate"])
    difference = mx.max(mx.abs(logits["candidate"] - logits["reference"])).item()
    identical = tokens["reference"] == tokens["candidate"]

    record = {
        "schema": "ironmule.padded_qmv_ab.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "paired_ab",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does padding the reduction dimension to 512 make a real decode step faster, exactly",
        "model": args.model,
        "padded_linears": padded,
        "padded_shapes": sorted({entry.split(":")[1] for entry in changed}),
        "steps_per_arm": args.steps,
        "blocks": args.blocks,
        "reference_step_ns_median": reference_ns,
        "candidate_step_ns_median": candidate_ns,
        "ratio_candidate_over_reference": candidate_ns / reference_ns,
        "speedup_percent": (1.0 - candidate_ns / reference_ns) * 100.0,
        "reference_block_medians_ns": arms["reference"],
        "candidate_block_medians_ns": arms["candidate"],
        "tokens_identical": identical,
        "max_absolute_logit_difference": difference,
        "reference_tokens": tokens["reference"],
        "candidate_tokens": tokens["candidate"],
        "limits": [
            "one prompt, one machine, one MLX build, 4-bit weights",
            "padding costs memory: the padded columns are stored and read",
            "the input pad is an extra kernel per call, and it is inside the measured time",
        ],
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("reference_tokens", "candidate_tokens", "padded_shapes")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
