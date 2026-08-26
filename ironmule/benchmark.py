"""Reproducible local benchmark of the two service modes.

    python -m ironmule.benchmark                 # both modes, default workload
    python -m ironmule.benchmark --requests 8 --max-tokens 48
    python -m ironmule.benchmark --plan reusable # shared-document session

Reports throughput and per-request latency separately. Group wall time is never
divided by group width, because that quotient is not a caller latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DOCUMENT = (
    "Apple silicon uses a unified memory architecture in which the CPU and the GPU "
    "address the same physical memory. MLX evaluates arrays lazily and executes them "
    "on the GPU, so a timer must synchronise before it can be trusted. Quantised "
    "weights are stored in four bits with a scale and a bias for every group of 64 "
    "values. Decoding one token reads the whole weight set, which is why decode is "
    "bound by memory bandwidth rather than by arithmetic. "
) * 5

QUESTIONS = [
    "What does unified memory avoid?",
    "How are quantised weights stored?",
    "Why must a timer synchronise?",
    "What bounds decoding?",
    "What does MLX do lazily?",
    "Which two units share the memory?",
    "How large is a quantisation group?",
    "What is read when decoding one token?",
]


def _run(rt, ironmule, mode, requests):
    rt.mode = mode
    started = time.perf_counter_ns()
    results = rt.serve(requests)
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    snap = rt.telemetry.snapshot()
    snap["outer_wall_ms"] = wall_ms
    return results, snap


def main(argv=None) -> int:
    import ironmule

    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--plan", choices=["strict", "reusable"], default="strict")
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    rt = ironmule.Runtime.load(model_id=args.model)
    plan = (ironmule.StrictOneShotPlan() if args.plan == "strict"
            else rt.session_plan(DOCUMENT, name="benchmark"))
    questions = (QUESTIONS * 4)[:args.requests]

    def build():
        prompts = [(DOCUMENT + "\n\nQuestion: " + q) if args.plan == "reusable" else q
                   for q in questions]
        return [ironmule.Request(prompt_ids=rt.encode(p), max_tokens=args.max_tokens,
                              plan=plan, rid=f"q{i}") for i, p in enumerate(prompts)]

    print(f"model {rt.model_id}")
    print(f"plan {args.plan}   requests {args.requests}   max_tokens {args.max_tokens}\n")

    sequential, seq = _run(rt, ironmule, ironmule.InteractiveMode(), build())
    grouped, grp = _run(rt, ironmule, ironmule.ThroughputMode(), build())

    identical = all(a.tokens == b.tokens for a, b in zip(sequential, grouped))
    row = "{:<14}{:>11}{:>13}{:>13}{:>13}{:>13}"
    print(row.format("mode", "wall ms", "tok/s", "svcTTFT p50", "lat p50", "lat p95"))
    for name, snap in (("interactive", seq), ("throughput", grp)):
        print(row.format(name, f"{snap['wall_ms']:.0f}",
                         f"{snap['aggregate_tokens_per_second']:.1f}",
                         f"{snap['service_ttft_p50_ms']:.1f}",
                         f"{snap['latency_p50_ms']:.1f}",
                         f"{snap['latency_p95_ms']:.1f}"))

    gain = 1 - grp["wall_ms"] / seq["wall_ms"]
    print(f"\nthroughput gain {gain*100:+.2f}%   "
          f"mean realised width {grp['mean_realised_width']:.2f}")
    print(f"service TTFT p50 {seq['service_ttft_p50_ms']:.1f} -> "
          f"{grp['service_ttft_p50_ms']:.1f} ms   "
          f"median latency {seq['latency_p50_ms']:.1f} -> {grp['latency_p50_ms']:.1f} ms")
    print(f"identical answers in both modes: {identical}")
    print(f"fallbacks {grp['fallbacks']}   correctness errors {grp['correctness_errors']}   "
          f"peak {grp['peak_memory_bytes']/1e9:.2f} GB")

    if args.json:
        args.json.write_text(json.dumps(
            {"interactive": seq, "throughput": grp, "gain": gain,
             "identical": identical, "plan": args.plan,
             "fingerprint": rt.fingerprint(plan, {"prompt_tokens": st.median(
                 len(r.prompt_ids) for r in build()), "max_tokens": args.max_tokens,
                 "concurrency": args.requests})},
            indent=1, sort_keys=True, default=str))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
