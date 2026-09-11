#!/usr/bin/env python3
"""Does routing reach the better fixed mode on every workload, and cost nothing?

One process, one model load, three arms over the same loaded weights:

    I_interactive   `InteractiveMode`, named by hand, for every workload
    T_throughput    `ThroughputMode`, named by hand, for every workload
    R_routed        `AppleRuntime`, objective "throughput", which chooses per dispatch
    L_routed_latency`AppleRuntime`, objective "latency", which must stay sequential

The workload is a fixed mixed sequence — one request, then two, then four — because
that is the case a fixed mode cannot serve well: `E15`/`E16` measured grouping buying
throughput at 26-31% median latency, so the right mode depends on the dispatch and not
on the machine. An arm that is best at both ends of that sequence is the only thing the
router claims.

Arm order rotates by block, so each arm occupies each position equally often. `B27e`
measured an order/temporal drift large enough (5.8-7.9%) to invent a result on its own,
so a fixed order is not acceptable here. Mirroring is not enough either: with four arms
it puts the same arm on both sides of a block boundary, and attempt 3 showed exactly
that pattern in the ratios it produced.

Sixteen blocks, not eight. Attempt 3 measured the per-dispatch spread on the two-request
workload at roughly plus or minus five per cent, which cannot resolve a two per cent
equivalence margin from eight paired ratios. That is a power problem in the design, not
a result, and it is fixed before the run rather than argued about after it.

Correctness gates the timing and is not derived from it: every arm must produce
identical token IDs, counts and stop reasons for every request. The knob set's own gain
and the paired path's own gain are not re-derived here; they are the autotuner
confirmation and `B46`/`B50`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

from ironmule.hw import fingerprint, swap_used_bytes  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.router import AppleRuntime, ExecutionRouter  # noqa: E402
from ironmule.service import InteractiveMode, Request, Runtime, ThroughputMode  # noqa: E402
from ironmule.tune import DEFAULT_MODEL, gpu_busy, load_profile, resolve_local_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURED_SOURCES = (
    "ironmule/router.py", "ironmule/service.py", "ironmule/service_strategy.py",
    "ironmule/executor.py", "ironmule/runtime.py", "ironmule/telemetry.py",
    "ironmule/tune.py", "ironmule/hw.py", "tools/b55_execution_router.py",
)

PREREGISTRATION = {
    "experiment": "B55_execution_router",
    "question": (
        "over a mixed dispatch sequence, does the router reach the faster fixed mode on "
        "each workload, and does the wrapper cost anything against naming that mode by hand"
    ),
    "arms": ["I_interactive", "T_throughput", "R_routed", "L_routed_latency"],
    "workloads": [
        {"name": "solo", "requests": 1},
        {"name": "pair", "requests": 2},
        {"name": "quad", "requests": 4},
    ],
    "blocks": 16,
    "arm_order": (
        "rotated by block index so each arm occupies each position four times "
        "(B27e: order drift is real; attempt 3: mirroring repeats an arm across a "
        "block boundary)"
    ),
    "power": (
        "attempt 3 measured a per-dispatch spread near 5 per cent on the pair workload; "
        "sixteen blocks are preregistered here because eight cannot resolve the 2 per "
        "cent margin. Attempts 1 to 3 are kept and reported, not replaced."
    ),
    "statistic": (
        "per-block ratio of arm wall time, median, 95 per cent bootstrap over 10000 "
        "resamples; per workload and over the mixed total"
    ),
    "equivalence_margin": 0.02,
    "decision_rule": (
        "ROUTER_REACHES_BEST only if, per workload, the 95 per cent interval of R over "
        "the workload's better fixed arm lies inside 0.98 to 1.02, and the interval of R "
        "over the worse fixed arm has an upper bound below 1.0 wherever the two fixed "
        "arms actually differ. An interval that merely contains 1.0 is not equivalence."
    ),
    "objective_rule": (
        "L_routed_latency must take the sequential route on every workload and be "
        "equivalent to I_interactive within the same margin; a caller who asked not to "
        "pay median latency for aggregate throughput must not be grouped"
    ),
    "correctness_gate": (
        "token IDs, physical token counts and stop reasons identical across all four "
        "arms for every request of every workload; checked before any timing is read"
    ),
    "blocking": (
        "any correctness difference, any fallback, any swap growth above 64 MiB, or a "
        "concurrent model process blocks the verdict"
    ),
    "not_measured": (
        "the tuned knob set's own gain (the autotuner's paired confirmation) and the "
        "paired path's own gain (B46, B50); neither is re-derived here"
    ),
}

PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
    "Summarise what a tokenizer does before a model sees any text.",
    "State one reason a quantised model reads fewer bytes per token.",
)
MAX_TOKENS = 48
SWAP_GROWTH_LIMIT = 64 * 1024 ** 2


def source_binding() -> dict[str, object]:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def bootstrap(ratios: list[float], resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios),
            "ratios": ratios}


def build_requests(runtime, count: int) -> list[Request]:
    """The same token IDs for every arm; only the mode differs between them."""
    return [Request(prompt_ids=runtime.encode(PROMPTS[i % len(PROMPTS)]),
                    max_tokens=MAX_TOKENS, plan=StrictOneShotPlan())
            for i in range(count)]


def run_arm(arm, requests: list[Request]) -> dict:
    """One dispatch. Wall time is measured around `serve` and nothing else."""
    fresh = [Request(prompt_ids=list(r.prompt_ids), max_tokens=r.max_tokens,
                     plan=StrictOneShotPlan()) for r in requests]
    started = time.perf_counter_ns()
    results = arm.serve(fresh)
    wall_ns = time.perf_counter_ns() - started
    telemetry = arm.telemetry.snapshot()
    return {
        "wall_ns": wall_ns,
        "tokens": [list(r.tokens) for r in results],
        "stop_reasons": [r.stop_reason for r in results],
        "token_counts": [len(r.tokens) for r in results],
        "fallbacks": telemetry["fallbacks"],
        "fallback_reasons": telemetry["fallback_reasons"],
        "latency_p50_ms": telemetry["latency_p50_ms"],
        "service_ttft_p50_ms": telemetry["service_ttft_p50_ms"],
        "aggregate_tokens_per_second": telemetry["aggregate_tokens_per_second"],
        "peak_memory_bytes": telemetry["peak_memory_bytes"],
        "routing": telemetry["routing"],
        "mode": telemetry["mode"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION,
             "source_binding": source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    resolved = resolve_local_model(args.model)
    profile = load_profile(args.model, model_identity=resolved.identity)
    if profile is None:
        raise SystemExit("no tuned profile for this machine/model: the router would "
                         "correctly refuse to route, and there is nothing to measure")

    swap_start = swap_used_bytes()
    runtime = Runtime.load(args.model, mode=InteractiveMode())
    router = ExecutionRouter(profile,
                             identity_sha256=resolved.identity.identity_sha256,
                             fingerprint=fingerprint(),
                             mlx=mx.__version__, mlx_lm=mlx_lm.__version__)
    routed = AppleRuntime(runtime, router)

    class Fixed:
        """A fixed mode over the same loaded engine, so weights load exactly once."""

        def __init__(self, mode):
            self.mode = mode

        def serve(self, requests):
            runtime.mode = self.mode
            return runtime.serve(requests)

        @property
        def telemetry(self):
            return runtime.telemetry

    latency_router = ExecutionRouter(profile,
                                     identity_sha256=resolved.identity.identity_sha256,
                                     fingerprint=fingerprint(),
                                     mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                                     objective="latency")
    arms = {"I_interactive": Fixed(InteractiveMode()),
            "T_throughput": Fixed(ThroughputMode()),
            "R_routed": routed,
            "L_routed_latency": AppleRuntime(runtime, latency_router)}
    order = ["I_interactive", "T_throughput", "R_routed", "L_routed_latency"]

    workloads = {w["name"]: build_requests(runtime, w["requests"])
                 for w in PREREGISTRATION["workloads"]}

    # One warm-up dispatch per arm, discarded: the first grouped run pays for compilation.
    for name in order:
        run_arm(arms[name], workloads["pair"])

    blocks: list[dict] = []
    try:
        for block in range(args.blocks):
            shift = block % len(order)
            sequence = order[shift:] + order[:shift]
            record: dict = {"block": block, "arm_order": list(sequence), "runs": {}}
            for arm_name in sequence:
                per_workload = {}
                for workload_name, requests in workloads.items():
                    per_workload[workload_name] = run_arm(arms[arm_name], requests)
                record["runs"][arm_name] = per_workload
            blocks.append(record)
            print(f"block {block} done ({'->'.join(sequence)})")
    finally:
        runtime.close()

    swap_end = swap_used_bytes()

    # -- correctness first, and it gates everything below ---------------------
    differences = []
    for record in blocks:
        for workload_name in workloads:
            reference = record["runs"]["I_interactive"][workload_name]
            for arm_name in order[1:]:
                other = record["runs"][arm_name][workload_name]
                for field in ("tokens", "token_counts", "stop_reasons"):
                    if other[field] != reference[field]:
                        differences.append({"block": record["block"],
                                            "workload": workload_name,
                                            "arm": arm_name, "field": field})
    fallbacks = sum(run["fallbacks"] for record in blocks
                    for arm in record["runs"].values() for run in arm.values())
    swap_growth = (None if swap_start is None or swap_end is None
                   else swap_end - swap_start)

    # -- timing ---------------------------------------------------------------
    comparisons = {}
    for workload_name in list(workloads) + ["mixed_total"]:
        def wall(record, arm_name):
            if workload_name == "mixed_total":
                return sum(run["wall_ns"] for run in record["runs"][arm_name].values())
            return record["runs"][arm_name][workload_name]["wall_ns"]

        pairs = {}
        for numerator, denominator in (("R_routed", "I_interactive"),
                                       ("R_routed", "T_throughput"),
                                       ("T_throughput", "I_interactive"),
                                       ("L_routed_latency", "I_interactive")):
            pairs[f"{numerator}_over_{denominator}"] = bootstrap(
                [wall(record, numerator) / wall(record, denominator) for record in blocks])
        comparisons[workload_name] = pairs

    routes = {}
    for arm_name in ("R_routed", "L_routed_latency"):
        routes[arm_name] = {}
        for workload_name in workloads:
            seen = {record["runs"][arm_name][workload_name]["routing"].get("route")
                    for record in blocks}
            realised = {record["runs"][arm_name][workload_name]["mode"] for record in blocks}
            routes[arm_name][workload_name] = {
                "routes": sorted(x for x in seen if x),
                "modes": sorted(x for x in realised if x)}

    latency_medians = {}
    for workload_name in workloads:
        latency_medians[workload_name] = {
            arm_name: median([record["runs"][arm_name][workload_name]["latency_p50_ms"]
                              for record in blocks])
            for arm_name in order}

    margin = PREREGISTRATION["equivalence_margin"]
    verdict_parts = {}
    for workload_name in workloads:
        pairs = comparisons[workload_name]
        fixed_ratio = pairs["T_throughput_over_I_interactive"]["median"]
        better = "T_throughput" if fixed_ratio < 1.0 else "I_interactive"
        worse = "I_interactive" if better == "T_throughput" else "T_throughput"
        against_better = pairs[f"R_routed_over_{better}"]
        against_worse = pairs[f"R_routed_over_{worse}"]
        fixed_arms_differ = not (1 - margin <= fixed_ratio <= 1 + margin)
        equivalent = (against_better["ci_low"] >= 1 - margin
                      and against_better["ci_high"] <= 1 + margin)
        beats_worse = against_worse["ci_high"] < 1.0
        verdict_parts[workload_name] = {
            "better_fixed_arm": better, "worse_fixed_arm": worse,
            "fixed_arms_differ_beyond_margin": fixed_arms_differ,
            "equivalent_to_better": equivalent,
            "beats_worse": beats_worse,
            "passes": equivalent and (beats_worse or not fixed_arms_differ),
        }

    objective_holds = all(
        routes["L_routed_latency"][workload_name]["routes"] == ["interactive"]
        for workload_name in workloads)
    latency_arm_equivalent = all(
        comparisons[w]["L_routed_latency_over_I_interactive"]["ci_low"] >= 1 - margin
        and comparisons[w]["L_routed_latency_over_I_interactive"]["ci_high"] <= 1 + margin
        for w in workloads)

    blocked = bool(differences) or fallbacks or (
        swap_growth is not None and swap_growth > SWAP_GROWTH_LIMIT)
    verdict = ("BLOCKED" if blocked
               else "ROUTER_REACHES_BEST"
               if all(p["passes"] for p in verdict_parts.values())
               and objective_holds and latency_arm_equivalent
               else "ROUTER_DOES_NOT_REACH_BEST")

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "hardware_fingerprint": fingerprint(),
            "model_id": resolved.identity.model_id,
            "model_revision": resolved.identity.revision,
            "model_identity_sha256": resolved.identity.identity_sha256,
            "tuned_knobs": profile["knobs"],
            "service_strategy_record": profile.get("service_strategy"),
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "blocks": len(blocks),
        "correctness": {
            "differences": differences,
            "identical_across_arms": not differences,
            "requests_compared": len(blocks) * sum(
                w["requests"] for w in PREREGISTRATION["workloads"]) * 2,
        },
        "resources": {"swap_used_bytes_start": swap_start, "swap_used_bytes_end": swap_end,
                      "swap_growth_bytes": swap_growth, "fallbacks": fallbacks},
        "routes_taken": routes,
        "comparisons": comparisons,
        "per_request_latency_p50_ms": latency_medians,
        "workload_verdicts": verdict_parts,
        "objective": {"latency_arm_stays_sequential": objective_holds,
                      "latency_arm_equivalent_to_interactive": latency_arm_equivalent},
        "verdict": verdict,
        "raw_blocks": blocks,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "routes_taken": routes,
                      "workload_verdicts": verdict_parts,
                      "per_request_latency_p50_ms": latency_medians,
                      "medians": {w: {k: v["median"] for k, v in p.items()}
                                  for w, p in comparisons.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
