#!/usr/bin/env python3
"""B24 workload: a plain greedy decode loop, small enough to trace and nothing else.

Run under `xcrun xctrace record --template 'Metal System Trace'`. The point is not this
script's own numbers but the GPU timeline recorded around it, so it keeps the host side
minimal: load, warm up, then a fixed number of steps between two markers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

PROMPT = "Explain in three sentences why unified memory changes how a laptop runs a language model."


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-4b-it-4bit")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--gap-s", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    model, tokenizer = load(args.model)
    language = getattr(model, "language_model", model)
    cache = language.make_cache()
    ids = mx.array(tokenizer.encode(PROMPT))[None]

    for _ in range(args.warmup):
        out = language.model(ids, cache=cache)
        logits = language.lm_head(out[:, -1, :])
        token = int(mx.argmax(logits, axis=-1).item())
        ids = mx.array([[token]])
    # A quiet gap the trace can be split on: everything after it is the measured phase.
    time.sleep(args.gap_s)

    steps = []
    for index in range(args.steps):
        began = time.perf_counter_ns()
        out = language.model(ids, cache=cache)
        logits = language.lm_head(out[:, -1, :])
        submitted = time.perf_counter_ns()
        token = int(mx.argmax(logits, axis=-1).item())
        finished = time.perf_counter_ns()
        ids = mx.array([[token]])
        steps.append(
            {
                "step": index,
                "submit_ns": submitted - began,
                "wait_ns": finished - submitted,
                "total_ns": finished - began,
            }
        )

    totals = {
        "model": args.model,
        "steps": len(steps),
        "warmup": args.warmup,
        "gap_s": args.gap_s,
        "submit_ns_mean": sum(step["submit_ns"] for step in steps) / len(steps),
        "wait_ns_mean": sum(step["wait_ns"] for step in steps) / len(steps),
        "total_ns_mean": sum(step["total_ns"] for step in steps) / len(steps),
        "total_ns_min": min(step["total_ns"] for step in steps),
        "per_step": steps,
    }
    print(json.dumps({key: value for key, value in totals.items() if key != "per_step"}))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(totals, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
