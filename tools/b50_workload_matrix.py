#!/usr/bin/env python3
"""Where does the paired path help, where is it neutral, where does it hurt?

Eight load cases against the unchanged `ThroughputMode`, decided one by one. No average
is reported, because an average would hide a case that regresses.

The prompts are new: none of them was used to build or tune any kernel in this line of
work, so the answers are not fitted to the thing being measured. Correctness is checked
per case before anything is timed, and the bit-level check runs separately at step level
because equal tokens are a weaker statement than equal logits.

Preregistered below in `MATRIX`, `BLOCKS`, `THRESHOLDS` and the arm order. Selection and
confirmation are separate runs of this tool.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.service import (  # noqa: E402
    PairedThroughputMode,
    Request,
    Runtime,
    ThroughputMode,
    paired_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHORT_IN = "What is a cache line?"
LONG_IN = (
    "Consider a laptop with unified memory running a quantised language model locally. "
    "The model weights live in the same physical memory as the operating system, the "
    "window server and every other application. During decoding the runtime reads the "
    "entire weight set once per generated token, while the attention cache grows with "
    "every step and competes for the same bandwidth. Scheduling decisions, memory "
    "pressure and thermal limits all interact. Given that setting, describe what "
    "determines the speed of generation and which of those factors an application "
    "author can actually influence from outside the runtime."
)
SECOND_SHORT_IN = "Define instruction level parallelism."
SECOND_LONG_IN = (
    "A service receives several independent chat requests at once, each with its own "
    "conversation history and its own stopping rule. The requests do not share text and "
    "must not influence one another's output. The service holds one model in memory and "
    "must decide, at every step, which requests to advance together and which to serve "
    "alone. Explain the trade-offs that decision involves and what a caller would "
    "notice in latency and throughput under each choice."
)

# name: (prompts, max_tokens per request, arrival_ms per request)
MATRIX = {
    "single_short_in_short_out": ([SHORT_IN], [8], [0.0]),
    "single_short_in_long_out": ([SHORT_IN], [48], [0.0]),
    "pair_short_in_short_out": ([SHORT_IN, SECOND_SHORT_IN], [8, 8], [0.0, 0.0]),
    "pair_short_in_long_out": ([SHORT_IN, SECOND_SHORT_IN], [48, 48], [0.0, 0.0]),
    "pair_long_in_short_out": ([LONG_IN, SECOND_LONG_IN], [8, 8], [0.0, 0.0]),
    "pair_long_in_long_out": ([LONG_IN, SECOND_LONG_IN], [48, 48], [0.0, 0.0]),
    "pair_staggered": ([SHORT_IN, SECOND_SHORT_IN], [24, 24], [0.0, 80.0]),
    "quad_simultaneous": ([SHORT_IN, SECOND_SHORT_IN, LONG_IN, SECOND_LONG_IN],
                          [24, 24, 24, 24], [0.0, 0.0, 0.0, 0.0]),
}

BLOCKS = 6
THRESHOLDS = {
    "advantage": "group completion CI95 entirely below 1.0 and no request's latency above 1.05x",
    "harm": "group completion CI95 entirely above 1.0, or a latency above 1.05x",
    "no_advantage": "CI95 contains 1.0",
    "unclear": "the A/A control fails, or the CI95 is wider than 0.10",
}
ARMS = ("A_throughput", "A_throughput_aa", "C_paired")


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def _requests(runtime, case):
    prompts, tokens, arrivals = MATRIX[case]
    return [Request(prompt_ids=runtime.encode(p), max_tokens=t,
                    plan=StrictOneShotPlan(), arrival_ms=a)
            for p, t, a in zip(prompts, tokens, arrivals)]


def _fingerprint(results):
    return [{"tokens": list(r.tokens), "stop_reason": r.stop_reason,
             "text_sha256": hashlib.sha256(r.text.encode()).hexdigest()} for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--phase", default="selection", choices=("selection", "confirmation"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    load_began = time.perf_counter_ns()
    runtime = Runtime.load(args.model, mode=ThroughputMode(), use_tuned_profile=False)
    load_ns = time.perf_counter_ns() - load_began
    try:
        modes = {"A_throughput": ThroughputMode(), "A_throughput_aa": ThroughputMode(),
                 "C_paired": PairedThroughputMode()}

        # Correctness per case, outside every timed region.
        correctness = {}
        for case in MATRIX:
            requests = _requests(runtime, case)
            runtime.mode = modes["A_throughput"]
            truth = _fingerprint(runtime.serve(requests))
            runtime.mode = modes["C_paired"]
            got = _fingerprint(runtime.serve(requests))
            correctness[case] = {
                "tokens": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
                "stop": [a["stop_reason"] == b["stop_reason"] for a, b in zip(truth, got)],
                "text": [a["text_sha256"] == b["text_sha256"] for a, b in zip(truth, got)],
                "stop_reasons": [r["stop_reason"] for r in truth],
                "generated": [len(r["tokens"]) for r in truth],
            }
        clean = all(all(row[key]) for row in correctness.values()
                    for key in ("tokens", "stop", "text"))
        print(json.dumps({"correctness_clean": clean}, sort_keys=True))
        if not clean:
            args.out.write_text(json.dumps({"state": "LOCKED", "correctness": correctness},
                                           indent=2, sort_keys=True))
            return 1

        # Warm every arm on every case before timing anything.
        for case in MATRIX:
            requests = _requests(runtime, case)
            for name in ARMS:
                runtime.mode = modes[name]
                runtime.serve(requests)

        before = _vm()
        samples = {case: {name: [] for name in ARMS} for case in MATRIX}
        status_after = {}
        for block in range(args.blocks):
            rotation = ARMS[block % len(ARMS):] + ARMS[: block % len(ARMS)]
            for case in MATRIX:
                requests = _requests(runtime, case)
                for name in rotation:
                    runtime.mode = modes[name]
                    began = time.perf_counter_ns()
                    results = runtime.serve(requests)
                    samples[case][name].append({
                        "group_ns": time.perf_counter_ns() - began,
                        "latency_ms": [r.metrics["latency_ms"] for r in results],
                        "ttft_ms": [r.metrics["service_ttft_ms"] for r in results],
                        "fell_back": any(r.metrics["fell_back"] for r in results),
                    })
                    if name == "C_paired":
                        status_after[case] = paired_status(modes["C_paired"]).copy()
        after = _vm()
        paging = {key: after.get(key, 0) - value for key, value in before.items()}

        def interval(values_a, values_b):
            ratios = [a / b for a, b in zip(values_a, values_b)]
            rng = random.Random(20260909)
            draws = sorted(median([ratios[rng.randrange(len(ratios))] for _ in ratios])
                           for _ in range(10000))
            return {"median": median(ratios), "min": min(ratios), "max": max(ratios),
                    "ci95_low": draws[int(0.025 * len(draws))],
                    "ci95_high": draws[int(0.975 * len(draws)) - 1]}

        verdicts = {}
        for case in MATRIX:
            rows = samples[case]
            group = interval([r["group_ns"] for r in rows["C_paired"]],
                             [r["group_ns"] for r in rows["A_throughput"]])
            aa = interval([r["group_ns"] for r in rows["A_throughput_aa"]],
                          [r["group_ns"] for r in rows["A_throughput"]])
            aa_ok = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
            count = len(rows["A_throughput"][0]["latency_ms"])
            lat = []
            for index in range(count):
                a = median([r["latency_ms"][index] for r in rows["A_throughput"]])
                c = median([r["latency_ms"][index] for r in rows["C_paired"]])
                lat.append(c / a)
            ttft = []
            for index in range(count):
                a = median([r["ttft_ms"][index] for r in rows["A_throughput"]])
                c = median([r["ttft_ms"][index] for r in rows["C_paired"]])
                ttft.append(c / a)
            width = group["ci95_high"] - group["ci95_low"]
            latency_ok = all(v <= 1.05 for v in lat)
            if not aa_ok or width > 0.10:
                verdict = "UNCLEAR"
            elif group["ci95_high"] < 1.0 and latency_ok:
                verdict = "ADVANTAGE"
            elif group["ci95_low"] > 1.0 or not latency_ok:
                verdict = "HARM"
            else:
                verdict = "NO ADVANTAGE"
            verdicts[case] = {
                "verdict": verdict,
                "group_completion": group,
                "aa_null_control": aa, "aa_passes": aa_ok,
                "latency_ratio_per_request": lat,
                "ttft_ratio_per_request": ttft,
                "latency_within_limit": latency_ok,
                "ci_width": width,
                "paired_steps": status_after[case]["paired_steps"],
                "solo_steps_no_partner": status_after[case]["solo_steps_no_partner"],
                "requests": count,
                "generated": correctness[case]["generated"],
                "stop_reasons": correctness[case]["stop_reasons"],
                "any_fallback": any(r["fell_back"] for r in rows["C_paired"]),
                "group_ns_median": {name: median(r["group_ns"] for r in rows[name])
                                    for name in ARMS},
            }

        record = {
            "schema": "ironmule.paired_load_matrix.v1",
            "experiment_id": args.out.stem,
            "agent": "claude",
            "kind": "load_matrix",
            "status": "measured",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase,
            "question": "under which load conditions does the paired path help, stay neutral or hurt",
            "reference_arm": "unchanged ThroughputMode",
            "preregistered": {
                "cases": list(MATRIX), "blocks": args.blocks, "arms": list(ARMS),
                "order": "rotated so every arm meets every position",
                "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
                "thresholds": THRESHOLDS,
                "prompts_are_new": "none of these prompts was used to build or tune a kernel",
                "budget": "one selection run and one confirmation run, no extension",
            },
            "correctness": correctness,
            "correctness_clean": clean,
            "verdicts": verdicts,
            "load_ns_including_compile": load_ns,
            "paging_during_run": paging,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "limits": [
                "one machine, one MLX build, greedy, this Gemma 12B revision",
                "service-level identity here; bit-level logit and KV identity is covered "
                "separately by the step-level check in B46 and B47",
                "no average across cases is reported: it would hide a regression",
            ],
            "mlx_version": mx.__version__,
            "device": {"platform": platform.platform(), "machine": platform.machine()},
            "git_revision": subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False).stdout.strip(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        for case, row in verdicts.items():
            print(f"  {row['verdict']:<13} {case:<28} "
                  f"{row['group_completion']['median']:.4f} "
                  f"[{row['group_completion']['ci95_low']:.4f}, {row['group_completion']['ci95_high']:.4f}] "
                  f"paired={row['paired_steps']:>3} solo={row['solo_steps_no_partner']:>3}")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
