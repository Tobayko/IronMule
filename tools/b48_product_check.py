#!/usr/bin/env python3
"""Does extending the sharing to o_proj and down_proj buy 5 per cent more?

The reference is the current opt-in path, `PairedThroughputMode()` with `K=3840` sharing
already on, not `ThroughputMode`. The candidate adds the aligned families. Both families
cleared their kernel gate, so the combination is measured: it is the upper bound, and if
it misses, neither single family can reach it.

Correctness first and outside the timed region, including the operational regressions
the paired path already owed: a genuine EOS on one side, unequal lengths, a late partner
and a single request that never waits.
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
GATE = 0.95
LATENCY_LIMIT = 1.05
PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
)
EOS_PROMPT = "Write one sentence about the sea."
LONG_PROMPT = "Explain, step by step, how a laptop runs a large language model locally."


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--session", default="1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runtime = Runtime.load(args.model, mode=ThroughputMode(), use_tuned_profile=False)
    try:
        library = ThroughputMode()
        modes = {
            "A_paired_k3840": PairedThroughputMode(),
            "A_paired_k3840_aa": PairedThroughputMode(),
            "C_plus_aligned": PairedThroughputMode(share_aligned=True),
        }

        def serve(mode, requests):
            runtime.mode = mode
            results = runtime.serve(requests)
            return [{"tokens": list(r.tokens), "stop_reason": r.stop_reason,
                     "text_sha256": hashlib.sha256(r.text.encode()).hexdigest(),
                     "latency_ms": r.metrics["latency_ms"],
                     "ttft_ms": r.metrics["service_ttft_ms"],
                     "fell_back": r.metrics["fell_back"]} for r in results]

        def pair(a, at, b, bt, arrival=0.0):
            return [
                Request(prompt_ids=runtime.encode(a), max_tokens=at, plan=StrictOneShotPlan()),
                Request(prompt_ids=runtime.encode(b), max_tokens=bt,
                        plan=StrictOneShotPlan(), arrival_ms=arrival),
            ]

        # Correctness and the operational regressions, against the library path.
        cases = {
            "simultaneous": pair(PROMPTS[0], 12, PROMPTS[1], 12),
            "genuine_eos": pair(EOS_PROMPT, 64, LONG_PROMPT, 64),
            "unequal_lengths": pair(PROMPTS[1], 6, LONG_PROMPT, 20),
            "late_partner": pair(PROMPTS[1], 16, LONG_PROMPT, 16, arrival=60.0),
            "single": [Request(prompt_ids=runtime.encode(PROMPTS[0]), max_tokens=8,
                               plan=StrictOneShotPlan())],
        }
        regressions = {}
        for name, requests in cases.items():
            truth = serve(library, requests)
            got = serve(modes["C_plus_aligned"], requests)
            regressions[name] = {
                "tokens": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
                "stop": [a["stop_reason"] == b["stop_reason"] for a, b in zip(truth, got)],
                "text": [a["text_sha256"] == b["text_sha256"] for a, b in zip(truth, got)],
                "stop_reasons": [r["stop_reason"] for r in truth],
                "lengths": [len(r["tokens"]) for r in truth],
                "no_fallback": not any(r["fell_back"] for r in got),
            }
        clean = all(all(row[key]) for row in regressions.values()
                    for key in ("tokens", "stop", "text"))
        eos_seen = "eos" in regressions["genuine_eos"]["stop_reasons"]
        print(json.dumps({"identity_clean": clean, "genuine_eos_observed": eos_seen},
                         sort_keys=True))

        def one_round(mode):
            requests = [Request(prompt_ids=runtime.encode(p), max_tokens=args.max_tokens,
                                plan=StrictOneShotPlan()) for p in PROMPTS]
            runtime.mode = mode
            began = time.perf_counter_ns()
            results = runtime.serve(requests)
            return {"wall_ns": time.perf_counter_ns() - began,
                    "latency_ms": [r.metrics["latency_ms"] for r in results],
                    "ttft_ms": [r.metrics["service_ttft_ms"] for r in results]}

        for mode in modes.values():
            one_round(mode)

        before = _vm()
        arms = {name: [] for name in modes}
        order = list(modes)
        for block in range(args.blocks):
            rotation = order[block % len(order):] + order[: block % len(order)]
            for name in rotation:
                arms[name].append(one_round(modes[name]))
        after = _vm()
        paging = {key: after.get(key, 0) - value for key, value in before.items()}

        def wall(name):
            return [row["wall_ns"] for row in arms[name]]

        def interval(numerator, denominator):
            ratios = [n / d for n, d in zip(wall(numerator), wall(denominator))]
            rng = random.Random(20260909)
            draws = sorted(median([ratios[rng.randrange(len(ratios))] for _ in ratios])
                           for _ in range(10000))
            return {"median": median(ratios), "min": min(ratios), "max": max(ratios),
                    "ci95_low": draws[int(0.025 * len(draws))],
                    "ci95_high": draws[int(0.975 * len(draws)) - 1],
                    "blocks_under_gate": sum(1 for v in ratios if v < GATE)}

        def per_request(name, field):
            return [median([row[field][i] for row in arms[name]]) for i in range(len(PROMPTS))]

        candidate = interval("C_plus_aligned", "A_paired_k3840")
        aa = interval("A_paired_k3840_aa", "A_paired_k3840")
        aa_passes = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
        a_lat = per_request("A_paired_k3840", "latency_ms")
        c_lat = per_request("C_plus_aligned", "latency_ms")
        lat_ratios = [c / a for c, a in zip(c_lat, a_lat)]
        lat_ok = all(v <= LATENCY_LIMIT for v in lat_ratios)
        meets = candidate["ci95_high"] < GATE
        decision = ("GOAL GO" if meets and aa_passes and clean and lat_ok
                    and paging.get("Swapouts", 0) == 0
                    else ("BLOCKED" if paging.get("Swapouts", 0) else
                          ("NO-GO, identity" if not clean else "GOAL NOT MET")))

        record = {
            "schema": "ironmule.aligned_sharing_product.v1",
            "experiment_id": args.out.stem,
            "agent": "claude",
            "kind": "product_ab",
            "status": "measured",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "session": args.session,
            "question": "does sharing o_proj and down_proj too buy 5 per cent more pair time",
            "decision": decision,
            "reference_arm": "PairedThroughputMode() with K=3840 sharing, the current opt-in path",
            "gate": GATE,
            "latency_limit": LATENCY_LIMIT,
            "preregistered": {
                "blocks": args.blocks, "max_tokens": args.max_tokens, "arms": list(modes),
                "order": "rotated so every arm meets every position",
                "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
                "success_rule": (
                    "candidate upper bound below 0.95 in both sessions, A/A containing 1.0, "
                    "identity clean, no request above 1.05x the reference's completion"
                ),
                "budget": "two sessions, no extension",
            },
            "regressions": regressions,
            "identity_clean": clean,
            "genuine_eos_observed": eos_seen,
            "arm_wall_ns_median": {name: median(wall(name)) for name in modes},
            "arm_wall_ns": {name: wall(name) for name in modes},
            "candidate_over_reference": candidate,
            "aa_null_control": aa,
            "aa_null_control_passes": aa_passes,
            "meets_gate": meets,
            "latency_ms_median": {name: per_request(name, "latency_ms") for name in modes},
            "ttft_ms_median": {name: per_request(name, "ttft_ms") for name in modes},
            "latency_ratio_candidate_over_reference": lat_ratios,
            "latency_within_limit": lat_ok,
            "paging_during_run": paging,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "limits": [
                "two fixed prompts, greedy, one machine, one MLX build",
                "the combination is measured as the upper bound of the two families",
                "load and compile time are outside these numbers",
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
                          if key in ("decision", "meets_gate", "candidate_over_reference",
                                     "aa_null_control_passes", "identity_clean",
                                     "genuine_eos_observed", "arm_wall_ns_median",
                                     "latency_ratio_candidate_over_reference",
                                     "paging_during_run")}, indent=2, sort_keys=True))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
