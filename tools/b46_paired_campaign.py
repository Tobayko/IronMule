#!/usr/bin/env python3
"""Does pairing beat IronMule's shipped throughput path, on real hardware?

Three arms over the same requests, the same runtime, one resident model:

* `A_throughput` is the shipped `ThroughputMode`, the product reference.
* `B_paired_separate` is the new pairing with every projection still separate.
* `C_paired_shared` is the same pairing with one weight sweep for both.

`B` against `C` is what the sharing is worth; only `C` against `A` decides the product
question. Correctness runs first and outside every timed region: logit bit patterns and
KV digests at step level against the shipped decode body, then tokens, stop reasons and
lengths at service level, including a single request, staggered arrival, unequal output
lengths and an early stop.

Latencies are read from the runtime's own per-request telemetry, never derived from a
pair's wall time.
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

from ironmule import paired_research as pr  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.service import Request, Runtime, ThroughputMode  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE = 0.90
LATENCY_LIMIT = 1.30  # preregistered: no request may exceed A's completion by more than this
PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
)


class PairedMode:
    def __init__(self, share: bool) -> None:
        self.share = share
        self.name = "paired_shared" if share else "paired_separate"

    def executor(self, backend, telemetry):
        return pr.PairedGroupedExecutor(backend, telemetry, share=self.share)


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def _step_level_identity(runtime) -> dict:
    """The strict gate: same logits bytes and same KV state as the shipped body."""

    backend = runtime.backend
    engine = runtime.engine
    prompts = [runtime.encode(p) for p in PROMPTS]
    capacity = backend.capacity_for([len(p) for p in prompts], 8)
    states, tokens = [], []
    for ids in prompts:
        state, token = backend.prefill(ids, StrictOneShotPlan(), capacity)
        states.append(state)
        tokens.append(token)

    reference = []
    for state, token in zip(states, tokens):
        body = engine._body(capacity, 1)
        out, new_state = body(mx.array([[token]]), state)
        mx.eval(out, *[v for layer in new_state["layers"] for v in layer.values()])
        reference.append((bytes(memoryview(out[:, -1, :])),
                          backend.kv_hash(new_state, int(new_state["position"]["offset"].item()))))

    rows = {}
    for share in (False, True):
        paired = pr.PairedBackend(backend, share=share)
        handles = paired.step_pair(states, tokens, capacity)
        got = []
        for (out, new_state), _ in handles:
            mx.eval(out, *[v for layer in new_state["layers"] for v in layer.values()])
            got.append((bytes(memoryview(out[:, -1, :])),
                        paired.kv_hash(new_state, int(new_state["position"]["offset"].item()))))
        rows[f"share={share}"] = {
            "logits_identical": [a[0] == b[0] for a, b in zip(reference, got)],
            "kv_identical": [a[1] == b[1] for a, b in zip(reference, got)],
        }
    return rows


def _service_cases(runtime, modes: dict) -> dict:
    """Single, staggered, unequal lengths and an early stop, per arm against A."""

    def serve(mode, requests):
        runtime.mode = mode
        results = runtime.serve(requests)
        return [{"tokens": list(r.tokens), "stop_reason": r.stop_reason,
                 "text_sha256": hashlib.sha256(r.text.encode()).hexdigest()} for r in results]

    cases = {
        "single": [Request(prompt_ids=runtime.encode(PROMPTS[0]), max_tokens=8,
                           plan=StrictOneShotPlan())],
        "simultaneous": [Request(prompt_ids=runtime.encode(p), max_tokens=8,
                                 plan=StrictOneShotPlan()) for p in PROMPTS],
        "staggered": [
            Request(prompt_ids=runtime.encode(PROMPTS[0]), max_tokens=8,
                    plan=StrictOneShotPlan()),
            Request(prompt_ids=runtime.encode(PROMPTS[1]), max_tokens=8,
                    plan=StrictOneShotPlan(), arrival_ms=40.0),
        ],
        "unequal_lengths": [
            Request(prompt_ids=runtime.encode(PROMPTS[0]), max_tokens=4,
                    plan=StrictOneShotPlan()),
            Request(prompt_ids=runtime.encode(PROMPTS[1]), max_tokens=12,
                    plan=StrictOneShotPlan()),
        ],
        "early_stop": [Request(prompt_ids=runtime.encode(p), max_tokens=64,
                               plan=StrictOneShotPlan()) for p in PROMPTS],
    }
    outcome = {}
    for name, requests in cases.items():
        truth = serve(modes["A_throughput"], requests)
        row = {"stop_reasons": [r["stop_reason"] for r in truth],
               "lengths": [len(r["tokens"]) for r in truth]}
        for arm in ("B_paired_separate", "C_paired_shared"):
            got = serve(modes[arm], requests)
            row[arm] = {
                "tokens": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
                "stop": [a["stop_reason"] == b["stop_reason"] for a, b in zip(truth, got)],
                "text": [a["text_sha256"] == b["text_sha256"] for a, b in zip(truth, got)],
            }
        outcome[name] = row
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--session", default="1")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runtime = Runtime.load(args.model, mode=ThroughputMode(), use_tuned_profile=False)
    try:
        admission = pr.admit(runtime.engine.model, runtime.model_identity)
        modes = {
            "A_throughput": ThroughputMode(),
            "A_throughput_aa": ThroughputMode(),
            "B_paired_separate": PairedMode(False),
            "C_paired_shared": PairedMode(True),
        }

        step_identity = _step_level_identity(runtime)
        service_identity = _service_cases(runtime, modes)
        step_clean = all(all(values) for row in step_identity.values()
                         for values in row.values())
        service_clean = all(
            all(all(values) for values in row[arm].values())
            for row in service_identity.values()
            for arm in ("B_paired_separate", "C_paired_shared")
        )
        clean = step_clean and service_clean
        print(json.dumps({"step_identity_clean": step_clean,
                          "service_identity_clean": service_clean}, sort_keys=True))
        if not clean or args.correctness_only:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({
                "schema": "ironmule.paired_campaign.v1",
                "experiment_id": args.out.stem, "agent": "claude",
                "kind": "integration_correctness",
                "status": "measured" if clean else "locked",
                "decision": "CORRECTNESS ONLY" if clean else "NO-GO",
                "step_identity": step_identity, "service_identity": service_identity,
                "admission": admission,
                "observed_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True), encoding="utf-8")
            return 0 if clean else 1

        def one_round(mode):
            runtime.mode = mode
            requests = [Request(prompt_ids=runtime.encode(p), max_tokens=args.max_tokens,
                                plan=StrictOneShotPlan()) for p in PROMPTS]
            began = time.perf_counter_ns()
            results = runtime.serve(requests)
            wall = time.perf_counter_ns() - began
            return {
                "wall_ns": wall,
                "latency_ms": [r.metrics["latency_ms"] for r in results],
                "service_ttft_ms": [r.metrics["service_ttft_ms"] for r in results],
                "queue_wait_ms": [r.metrics["queue_wait_ms"] for r in results],
                "generated": [r.metrics["generated_tokens"] for r in results],
                "fell_back": [r.metrics["fell_back"] for r in results],
            }

        for name in modes:
            for _ in range(args.warmup):
                one_round(modes[name])

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
                    "blocks_at_or_under_gate": sum(1 for v in ratios if v <= GATE)}

        def per_request(name, field):
            return [median([row[field][index] for row in arms[name]]) for index in range(len(PROMPTS))]

        c_over_a = interval("C_paired_shared", "A_throughput")
        b_over_a = interval("B_paired_separate", "A_throughput")
        c_over_b = interval("C_paired_shared", "B_paired_separate")
        aa = interval("A_throughput_aa", "A_throughput")
        aa_passes = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
        a_latency = per_request("A_throughput", "latency_ms")
        c_latency = per_request("C_paired_shared", "latency_ms")
        latency_ratios = [c / a for c, a in zip(c_latency, a_latency)]
        latency_ok = all(value <= LATENCY_LIMIT for value in latency_ratios)
        meets = c_over_a["ci95_high"] <= GATE
        decision = ("PRODUCT GO" if meets and aa_passes and latency_ok
                    and paging.get("Swapouts", 0) == 0
                    else ("BLOCKED" if paging.get("Swapouts", 0) else "PRODUCT NO-GO"))

        record = {
            "schema": "ironmule.paired_campaign.v1",
            "experiment_id": args.out.stem,
            "agent": "claude",
            "kind": "product_ab",
            "status": "measured",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "session": args.session,
            "question": "does pairing with shared weight loads beat the shipped throughput path",
            "decision": decision,
            "model": args.model,
            "gate": GATE,
            "latency_limit": LATENCY_LIMIT,
            "preregistered": {
                "sessions": 2, "blocks": args.blocks, "max_tokens": args.max_tokens,
                "arms": list(modes),
                "order": "rotated so every arm meets every position",
                "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
                "success_rule": (
                    "C over A upper bound at or below 0.90 in both sessions, A/A control "
                    "containing 1.0, no request's completion above 1.30x A's, identity clean"
                ),
                "abort_rule": "swapouts during a run make it not evaluable",
                "budget": "two sessions, no extension",
            },
            "admission": admission,
            "step_identity": step_identity,
            "service_identity": service_identity,
            "identity_clean": clean,
            "arm_wall_ns_median": {name: median(wall(name)) for name in modes},
            "arm_wall_ns": {name: wall(name) for name in modes},
            "C_over_A": c_over_a,
            "B_over_A": b_over_a,
            "sharing_effect_C_over_B": c_over_b,
            "aa_null_control": aa,
            "aa_null_control_passes": aa_passes,
            "meets_gate": meets,
            "latency_ms_median": {name: per_request(name, "latency_ms") for name in modes},
            "service_ttft_ms_median": {name: per_request(name, "service_ttft_ms") for name in modes},
            "queue_wait_ms_median": {name: per_request(name, "queue_wait_ms") for name in modes},
            "latency_ratio_C_over_A": latency_ratios,
            "latency_within_limit": latency_ok,
            "fallbacks": {name: any(any(row["fell_back"]) for row in arms[name]) for name in modes},
            "paging_during_run": paging,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "limits": [
                "two fixed prompts, greedy, one machine, one MLX build",
                "load and compile time are outside these numbers",
                "results against B are not a gain against A",
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
                          if key in ("decision", "meets_gate", "C_over_A", "B_over_A",
                                     "sharing_effect_C_over_B", "aa_null_control_passes",
                                     "latency_ms_median", "latency_ratio_C_over_A",
                                     "latency_within_limit", "arm_wall_ns_median",
                                     "paging_during_run", "fallbacks")},
                         indent=2, sort_keys=True))
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
