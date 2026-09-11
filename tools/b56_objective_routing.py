#!/usr/bin/env python3
"""Is the objective a request carries enough to route it, and what does that cost?

One process, one model load, two arms over the same loaded weights:

    M_manual      the qualified strategy named by hand. `InteractiveMode` for a
                  latency workload, `ThroughputMode` for a throughput one, and for a
                  mixed dispatch the same two-cohort split written out by the caller.
    R_objective   `AppleRuntime.serve`, which reads `Request.objective` and splits the
                  dispatch itself under `ironmule.dispatch.objective_cohorts.v1`.

The question is not whether routing beats the hand-written version. It is whether it
reaches it without a relevant extra cost, and whether a `latency` request keeps its
latency when a foreign `throughput` request shares the dispatch. `B55` measured why that
matters: grouping bought `+7.7%` aggregate tokens per second for `+74.9%` median
per-request latency at two ready requests.

Correctness gates the timing and is not derived from it. Every arm is compared against a
pure `InteractiveMode` reference over the same prompts, so a token difference is caught
against a path that has no grouping in it at all.

Sixteen blocks with rotated arm order, because `B55` measured a per-dispatch spread near
five per cent and eight mirrored blocks could not resolve a two per cent margin.
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
import os
from statistics import median

import mlx.core as mx
import mlx_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

from ironmule.hw import fingerprint, swap_used_bytes  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.router import SCHEDULING_RULE, AppleRuntime, ExecutionRouter  # noqa: E402
from ironmule.service import InteractiveMode, Request, Runtime, ThroughputMode  # noqa: E402
from ironmule.tune import DEFAULT_MODEL, gpu_busy, load_profile, resolve_local_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURED_SOURCES = (
    "ironmule/router.py", "ironmule/service.py", "ironmule/service_strategy.py",
    "ironmule/executor.py", "ironmule/runtime.py", "ironmule/telemetry.py",
    "ironmule/tune.py", "ironmule/hw.py", "tools/b56_objective_routing.py",
)

MAX_TOKENS = 32
SWAP_GROWTH_LIMIT = 64 * 1024 ** 2
MARGIN = 0.02
# Attempt 3 was contaminated: an unrelated editor language server started mid-run and the
# hand-written arm's wall time doubled from ~810 ms to ~1640 ms over blocks 9 to 14.
# `gpu_busy()` did not catch it because it only recognises a loaded model. These two
# gates are stability gates: they can turn a verdict into BLOCKED and never into GO.
LOAD_CEILING = 4.0            # 1-minute load average before the run may start
BLOCK_DRIFT_LIMIT = 0.25      # a block this far off the median is a disturbed block

# (name, [(objective, arrival_ms, prompt_index), ...]) -- fixed before the run.
#
# The `*c` entries are the latency-only controls: the same prompts, in the same order,
# with the same declared arrivals, and no foreign throughput request anywhere. Attempt 1
# compared a mixed workload's latency requests against a different workload's, which
# confounded the prompt mix and the arrival schedule with the thing being asked about.
# The threshold is unchanged; what it is applied to is now matched.
WORKLOADS = (
    ("W1_latency_1", (("latency", 0.0, 0),)),
    ("W2_latency_2", (("latency", 0.0, 0), ("latency", 0.0, 1))),
    ("W3_throughput_2", (("throughput", 0.0, 0), ("throughput", 0.0, 1))),
    ("W4_mixed_1_1", (("latency", 0.0, 0), ("throughput", 0.0, 1))),
    ("W5_mixed_4", (("latency", 0.0, 0), ("throughput", 0.0, 1),
                    ("latency", 0.0, 2), ("throughput", 0.0, 3))),
    ("W5c_latency_control", (("latency", 0.0, 0), ("latency", 0.0, 2))),
    ("W6_staggered", (("throughput", 0.0, 0), ("latency", 60.0, 1),
                      ("throughput", 0.0, 2), ("latency", 60.0, 3))),
    ("W6c_staggered_control", (("latency", 60.0, 1), ("latency", 60.0, 3))),
)
# Which latency-only workload each mixed one is judged against. Same prompts, same
# positions, same declared arrivals.
CONTROL_PAIRS = (
    ("W4_mixed_1_1", "W1_latency_1"),
    ("W5_mixed_4", "W5c_latency_control"),
    ("W6_staggered", "W6c_staggered_control"),
)

PREREGISTRATION = {
    "experiment": "B56_objective_routing",
    "question": (
        "does per-request objective routing reach the hand-named qualified strategy "
        "without a relevant extra cost, and does a latency request keep its latency "
        "when a foreign throughput request shares the dispatch"
    ),
    "arms": ["M_manual", "R_objective"],
    "reference": "pure InteractiveMode over the same prompts, for token identity only",
    "scheduling_rule": SCHEDULING_RULE,
    "scheduling_rule_text": (
        "resolve each request's objective (request, else dispatch, else runtime); "
        "partition into one cohort per objective preserving caller order; order cohorts "
        "by (earliest arrival_ms, latency before throughput); serve each cohort with the "
        "route the router picks, sharing one dispatch timestamp; return results in "
        "caller order"
    ),
    "workloads": [{"name": name,
                   "requests": [{"objective": o, "arrival_ms": a, "prompt_index": i}
                                for o, a, i in spec]}
                  for name, spec in WORKLOADS],
    "control_pairs": [{"mixed": mixed, "control": control}
                      for mixed, control in CONTROL_PAIRS],
    "max_tokens": MAX_TOKENS,
    "blocks": 16,
    "arm_order": "rotated by block index so each arm leads eight times",
    "statistic": (
        "per-block ratio, median, 95 per cent bootstrap over 10000 resamples"
    ),
    "equivalence_margin": MARGIN,
    "criteria": {
        "C1_correctness": (
            "token IDs, physical token counts and stop reasons identical between both "
            "arms and the InteractiveMode reference, for every request of every block"
        ),
        "C2_routing_overhead": (
            "per workload, the 95 per cent interval of R_objective over M_manual group "
            "completion time lies inside 0.98 to 1.02"
        ),
        "C3_latency_protection": (
            "the completion latency of a mixed workload's latency requests over the same "
            "requests in their matched latency-only control -- same prompts, same "
            "positions, same declared arrivals -- has a 95 per cent upper bound below "
            "1.02; a foreign throughput request must not cost a latency request "
            "measurable latency"
        ),
        "C4_throughput_reach": (
            "R_objective over a hand-named ThroughputMode on W3 lies inside 0.98 to "
            "1.02, and every throughput request is routed 'throughput' when its cohort "
            "holds at least two members and 'interactive' when it holds one"
        ),
        "C5_determinism": (
            "for each workload the map request index -> (objective, route) is identical "
            "in every block"
        ),
        "C6_no_starvation": (
            "every request of every block completes with a stop reason, and no request "
            "is served twice"
        ),
    },
    "decision_rule": (
        "B56_GO only if C1 to C6 all hold. Any failure names the criterion and the "
        "verdict is B56_NO_GO. Thresholds are fixed here and are not revisited after "
        "the run."
    ),
    "blocking": (
        "any fallback, any swap growth above 64 MiB, a concurrent model process, a "
        "1-minute load average above 4.0 at the start, or any block whose hand-written "
        "arm deviates more than 25 per cent from the median block blocks the verdict"
    ),
    "not_measured": (
        "the paired path (B58), any new kernel, and any learned policy; none is touched "
        "here"
    ),
}

PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
    "Summarise what a tokenizer does before a model sees any text.",
    "State one reason a quantised model reads fewer bytes per token.",
)


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
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0, "ratios": []}
    draws = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios),
            "ratios": ratios}


def make_requests(prompt_ids, spec) -> list[Request]:
    """One fresh request set. Same token IDs and same order for every arm."""
    return [Request(prompt_ids=list(prompt_ids[prompt_index]),
                    max_tokens=MAX_TOKENS, plan=StrictOneShotPlan(),
                    objective=objective, arrival_ms=arrival)
            for objective, arrival, prompt_index in spec]


def _collect(results, telemetry, wall_ns, routes, objectives) -> dict:
    per_request = {row["rid"]: row for row in telemetry["per_request"]}
    return {
        "wall_ns": wall_ns,
        "order": [result.rid for result in results],
        "tokens": {r.rid: list(r.tokens) for r in results},
        "stop_reasons": {r.rid: r.stop_reason for r in results},
        "token_counts": {r.rid: len(r.tokens) for r in results},
        "routes": dict(routes),
        "objectives": dict(objectives),
        "per_request": {rid: {
            "service_ttft_ms": row["service_ttft_ms"],
            "engine_ttft_ms": row["engine_ttft_ms"],
            "queue_wait_ms": row["queue_wait_ms"],
            "latency_ms": row["latency_ms"],
            "decode_ms": row["decode_ms"],
            "decode_tokens_per_second": row["decode_tokens_per_second"],
        } for rid, row in per_request.items()},
        "aggregate_tokens_per_second": telemetry["aggregate_tokens_per_second"],
        "fallbacks": telemetry["fallbacks"],
        "fallback_reasons": telemetry["fallback_reasons"],
        "peak_memory_bytes": telemetry["peak_memory_bytes"],
    }


def run_routed(runtime: AppleRuntime, requests: list[Request]) -> dict:
    started = time.perf_counter_ns()
    results = runtime.serve(requests)
    wall = time.perf_counter_ns() - started
    decision = runtime.last_decision
    return _collect(results, decision["telemetry"], wall,
                    decision["route_by_request"], decision["objective_by_request"])


def run_manual(runtime: Runtime, requests: list[Request]) -> dict:
    """The same cohort split, written out by the caller instead of routed.

    This is what a person does today: hold the latency requests apart, run them on the
    sequential path, then run the rest grouped. It is deliberately the same schedule, so
    the comparison measures routing and not a different plan.
    """
    from ironmule.telemetry import Telemetry

    latency = [r for r in requests if r.objective == "latency"]
    throughput = [r for r in requests if r.objective == "throughput"]
    # A lone request has no partner to group with, so a careful caller puts it on the
    # sequential path whatever it asked for. Anything else would compare two schedules.
    def cohort(members, objective):
        grouped = objective == "throughput" and len(members) > 1
        return (ThroughputMode() if grouped else InteractiveMode()), members

    cohorts = []
    if latency and throughput:
        first_latency = min(r.arrival_ms for r in latency)
        first_throughput = min(r.arrival_ms for r in throughput)
        cohorts = ([cohort(latency, "latency"), cohort(throughput, "throughput")]
                   if first_latency <= first_throughput
                   else [cohort(throughput, "throughput"), cohort(latency, "latency")])
    elif latency:
        cohorts = [cohort(latency, "latency")]
    else:
        cohorts = [cohort(throughput, "throughput")]

    started = time.perf_counter_ns()
    results, routes, parts = [], {}, []
    for mode, members in cohorts:
        runtime.mode = mode
        results.extend(runtime.serve(members, dispatch_ns=started))
        route = "throughput" if mode.name == "throughput" else "interactive"
        routes.update({r.rid: route for r in members})
        parts.append(runtime.telemetry)
    wall = time.perf_counter_ns() - started

    merged = Telemetry(mode="manual")
    for part in parts:
        merged.plan_kinds.extend(part.plan_kinds)
        merged.requests.extend(part.requests)
        merged.realised_widths.extend(part.realised_widths)
        merged.wall_ns += part.wall_ns
        merged.fallbacks += part.fallbacks
        merged.fallback_reasons.extend(part.fallback_reasons)
        merged.peak_memory_bytes = max(merged.peak_memory_bytes, part.peak_memory_bytes)
    by_rid = {result.rid: result for result in results}
    ordered = [by_rid[request.rid] for request in requests]
    return _collect(ordered, merged.snapshot(), wall, routes,
                    {r.rid: r.objective for r in requests})


def run_reference(runtime: Runtime, requests: list[Request]) -> dict:
    """Pure sequential, one request at a time. Token reference only, never timed."""
    runtime.mode = InteractiveMode()
    tokens, stops = {}, {}
    for request in requests:
        result = runtime.serve([request])[0]
        tokens[request.rid] = list(result.tokens)
        stops[request.rid] = result.stop_reason
    return {"tokens": tokens, "stop_reasons": stops}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")
    load_start = os.getloadavg()
    if load_start[0] > LOAD_CEILING:
        raise SystemExit(f"1-minute load average is {load_start[0]:.2f}, above the "
                         f"{LOAD_CEILING} ceiling; refusing to measure on a busy machine")

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
    prompt_ids = [runtime.encode(prompt) for prompt in PROMPTS]

    reference = {}
    blocks: list[dict] = []
    try:
        # Token reference first, on the path with no grouping in it at all.
        for name, spec in WORKLOADS:
            reference[name] = run_reference(runtime, make_requests(prompt_ids, spec))

        for _ in range(2):                      # warm-up, discarded
            run_manual(runtime, make_requests(prompt_ids, WORKLOADS[4][1]))
            run_routed(routed, make_requests(prompt_ids, WORKLOADS[4][1]))

        order = ["M_manual", "R_objective"]
        for block in range(args.blocks):
            shift = block % len(order)
            sequence = order[shift:] + order[:shift]
            record = {"block": block, "arm_order": list(sequence), "runs": {}}
            for arm in sequence:
                record["runs"][arm] = {}
                for name, spec in WORKLOADS:
                    requests = make_requests(prompt_ids, spec)
                    run = (run_routed(routed, requests) if arm == "R_objective"
                           else run_manual(runtime, requests))
                    run["index_by_rid"] = {r.rid: i for i, r in enumerate(requests)}
                    record["runs"][arm][name] = run
            blocks.append(record)
            print(f"block {block} done ({'->'.join(sequence)})")
    finally:
        runtime.close()

    swap_end = swap_used_bytes()
    load_end = os.getloadavg()
    arms = ("M_manual", "R_objective")

    # -- stability: was the machine the same machine throughout? ---------------
    block_walls = [sum(record["runs"]["M_manual"][name]["wall_ns"] / 1e6
                       for name, _spec in WORKLOADS) for record in blocks]
    wall_median = median(block_walls)
    disturbed = [{"block": index, "wall_ms": wall,
                  "deviation": wall / wall_median - 1.0}
                 for index, wall in enumerate(block_walls)
                 if abs(wall / wall_median - 1.0) > BLOCK_DRIFT_LIMIT]

    # -- C1 correctness, and it gates everything below ------------------------
    differences = []
    for record in blocks:
        for name, _spec in WORKLOADS:
            expected = reference[name]
            for arm in arms:
                run = record["runs"][arm][name]
                index = run["index_by_rid"]
                by_index = {index[rid]: rid for rid in index}
                ref_index = {i: rid for i, rid in enumerate(expected["tokens"])}
                for position, rid in by_index.items():
                    ref_rid = ref_index[position]
                    if run["tokens"][rid] != expected["tokens"][ref_rid]:
                        differences.append({"block": record["block"], "workload": name,
                                            "arm": arm, "position": position,
                                            "field": "tokens"})
                    if run["stop_reasons"][rid] != expected["stop_reasons"][ref_rid]:
                        differences.append({"block": record["block"], "workload": name,
                                            "arm": arm, "position": position,
                                            "field": "stop_reason"})

    fallbacks = sum(run["fallbacks"] for record in blocks
                    for arm in record["runs"].values() for run in arm.values())
    swap_growth = (None if swap_start is None or swap_end is None
                   else swap_end - swap_start)

    # -- C2 routing overhead --------------------------------------------------
    overhead = {}
    for name, _spec in WORKLOADS:
        overhead[name] = bootstrap([
            record["runs"]["R_objective"][name]["wall_ns"]
            / record["runs"]["M_manual"][name]["wall_ns"] for record in blocks])
    c2 = all(row["ci_low"] >= 1 - MARGIN and row["ci_high"] <= 1 + MARGIN
             for row in overhead.values())

    # -- C3 latency protection ------------------------------------------------
    arrival_by_position = {name: [arrival for _objective, arrival, _index in spec]
                           for name, spec in WORKLOADS}

    def latency_median(record, arm, workload, *, from_declared_arrival=False):
        run = record["runs"][arm][workload]
        arrivals = arrival_by_position[workload]
        index = run["index_by_rid"]
        values = []
        for rid, objective in run["objectives"].items():
            if objective != "latency":
                continue
            value = run["per_request"][rid]["latency_ms"]
            if from_declared_arrival:
                value -= arrivals[index[rid]]
            values.append(value)
        return median(values) if values else None

    PAIRS = CONTROL_PAIRS
    protection = {}
    for mixed, alone in PAIRS:
        protection[f"{mixed}_over_{alone}"] = bootstrap([
            latency_median(record, "R_objective", mixed)
            / latency_median(record, "R_objective", alone) for record in blocks])
    # The criterion is judged on the ratios above, exactly as preregistered. The two
    # series below are context and decide nothing: the first charges a staggered request
    # only the time after its own declared arrival, the second asks whether a
    # hand-written caller does anything different.
    protection_context = {}
    for mixed, alone in PAIRS:
        protection_context[f"{mixed}_over_{alone}_from_declared_arrival"] = bootstrap([
            latency_median(record, "R_objective", mixed, from_declared_arrival=True)
            / latency_median(record, "R_objective", alone, from_declared_arrival=True)
            for record in blocks])
        protection_context[f"{mixed}_over_{alone}_manual_arm"] = bootstrap([
            latency_median(record, "M_manual", mixed)
            / latency_median(record, "M_manual", alone) for record in blocks])
    c3 = all(row["ci_high"] < 1 + MARGIN for row in protection.values())

    # -- C4 throughput reach --------------------------------------------------
    route_errors = []
    for record in blocks:
        for name, _spec in WORKLOADS:
            run = record["runs"]["R_objective"][name]
            cohort_size = {}
            for rid, objective in run["objectives"].items():
                cohort_size[objective] = cohort_size.get(objective, 0) + 1
            for rid, objective in run["objectives"].items():
                expected = ("throughput" if objective == "throughput"
                            and cohort_size[objective] > 1 else "interactive")
                if run["routes"][rid] != expected:
                    route_errors.append({"block": record["block"], "workload": name,
                                         "objective": objective,
                                         "expected": expected,
                                         "observed": run["routes"][rid]})
    c4 = overhead["W3_throughput_2"]["ci_low"] >= 1 - MARGIN and \
        overhead["W3_throughput_2"]["ci_high"] <= 1 + MARGIN and not route_errors

    # -- C5 determinism -------------------------------------------------------
    signatures = {}
    determinism_errors = []
    for record in blocks:
        for name, _spec in WORKLOADS:
            run = record["runs"]["R_objective"][name]
            index = run["index_by_rid"]
            signature = tuple(sorted(
                (index[rid], run["objectives"][rid], run["routes"][rid])
                for rid in index))
            first = signatures.setdefault(name, signature)
            if signature != first:
                determinism_errors.append({"block": record["block"], "workload": name})
    c5 = not determinism_errors

    # -- C6 no starvation -----------------------------------------------------
    starvation = []
    for record in blocks:
        for name, spec in WORKLOADS:
            for arm in arms:
                run = record["runs"][arm][name]
                if len(run["order"]) != len(spec) or len(set(run["order"])) != len(spec):
                    starvation.append({"block": record["block"], "workload": name,
                                       "arm": arm, "reason": "request count"})
                if any(not reason for reason in run["stop_reasons"].values()):
                    starvation.append({"block": record["block"], "workload": name,
                                       "arm": arm, "reason": "unfinished request"})
    c6 = not starvation

    c1 = not differences
    blocked = (fallbacks
               or (swap_growth is not None and swap_growth > SWAP_GROWTH_LIMIT)
               or bool(disturbed))
    criteria = {"C1_correctness": c1, "C2_routing_overhead": c2,
                "C3_latency_protection": c3, "C4_throughput_reach": c4,
                "C5_determinism": c5, "C6_no_starvation": c6}
    verdict = ("BLOCKED_UNSTABLE" if disturbed
               else "BLOCKED" if blocked or not c1
               else "B56_GO" if all(criteria.values()) else "B56_NO_GO")

    observed = {}
    for name, _spec in WORKLOADS:
        rows = {}
        for arm in arms:
            rows[arm] = {
                "wall_ms": median([record["runs"][arm][name]["wall_ns"] / 1e6
                                   for record in blocks]),
                "aggregate_tokens_per_second": median([
                    record["runs"][arm][name]["aggregate_tokens_per_second"]
                    for record in blocks]),
            }
            for objective in ("latency", "throughput"):
                values = []
                for record in blocks:
                    run = record["runs"][arm][name]
                    picked = [run["per_request"][rid]["latency_ms"]
                              for rid in run["objectives"]
                              if run["objectives"][rid] == objective]
                    if picked:
                        values.append(median(picked))
                if values:
                    rows[arm][f"{objective}_latency_p50_ms"] = median(values)
                ttfts = []
                for record in blocks:
                    run = record["runs"][arm][name]
                    picked = [run["per_request"][rid]["service_ttft_ms"]
                              for rid in run["objectives"]
                              if run["objectives"][rid] == objective]
                    if picked:
                        ttfts.append(median(picked))
                if ttfts:
                    rows[arm][f"{objective}_service_ttft_p50_ms"] = median(ttfts)
        observed[name] = rows

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
        "criteria": criteria,
        "correctness": {"differences": differences, "identical": c1,
                        "requests_compared": sum(len(spec) for _n, spec in WORKLOADS)
                        * len(blocks) * len(arms)},
        "resources": {"swap_used_bytes_start": swap_start, "swap_used_bytes_end": swap_end,
                      "swap_growth_bytes": swap_growth, "fallbacks": fallbacks,
                      "load_average_start": list(load_start),
                      "load_average_end": list(load_end)},
        "stability": {"block_wall_ms": block_walls, "median_block_wall_ms": wall_median,
                      "disturbed_blocks": disturbed, "stable": not disturbed},
        "routing_overhead": overhead,
        "latency_protection": protection,
        "latency_protection_context": protection_context,
        "route_errors": route_errors,
        "determinism_errors": determinism_errors,
        "starvation": starvation,
        "route_signatures": {name: [list(row) for row in signature]
                             for name, signature in signatures.items()},
        "observed": observed,
        "verdict": verdict,
        "raw_blocks": blocks,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "criteria": criteria,
                      "routing_overhead": {k: {"median": v["median"],
                                               "ci": [v["ci_low"], v["ci_high"]]}
                                           for k, v in overhead.items()},
                      "latency_protection": {k: {"median": v["median"],
                                                 "ci": [v["ci_low"], v["ci_high"]]}
                                             for k, v in protection.items()},
                      "latency_protection_context": {
                          k: {"median": v["median"], "ci": [v["ci_low"], v["ci_high"]]}
                          for k, v in protection_context.items()},
                      "observed": observed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
