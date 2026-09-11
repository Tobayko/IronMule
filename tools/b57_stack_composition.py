#!/usr/bin/env python3
"""The best combination of the levers this machine has already earned, measured as stacks.

Every optimisation here was measured against its own reference, in its own harness, with
its own denominator. Adding those percentages produces a number that describes nothing.
So each combination is executed as one execution candidate and measured whole, and the
only thing that counts is the stack that actually ran.

`ironmule/stacks.py` holds the table this reads: what each lever needs, how far its
evidence reaches, and which levers the source itself keeps apart. The candidate set is
reduced by that table before anything runs, and every dropped candidate is dropped for a
reason written into the record.

**Plans are not comparable across plan kinds.** `E9` measured `StrictOneShotPlan` and
`ReusableSessionPlan` producing tokens up to 4.31 logits apart. A stack that reuses a
prefix is therefore never compared against a strict-plan stack: its reference is the same
plan with the cache dropped between requests, which is the same chunked prefill, the same
tokens, and no reuse. That isolates what reuse is worth without pretending two plans are
one.

Selection and confirmation are separate runs of this file. Selection names one candidate
per workload class; confirmation measures only that one against its reference, with the
blocks, the interval and the gates fixed before it starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import resource
import subprocess
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

from ironmule.hw import (MEMORY_PRESSURE_NAMES, MEMORY_PRESSURE_NORMAL,  # noqa: E402
                        fingerprint, memory_pressure_level, static_facts,
                        swap_used_bytes, vm_counters)
from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan  # noqa: E402
from ironmule.router import AppleRuntime, ExecutionRouter  # noqa: E402
from ironmule.service import (InteractiveMode, PairedThroughputMode, Request,  # noqa: E402
                              Runtime, ThroughputMode)
from ironmule.stacks import (CANDIDATES, STACK_MODEL_VERSION, Context,  # noqa: E402
                             OPTIMISATIONS, reduce)
from ironmule.telemetry import Telemetry  # noqa: E402
from ironmule.tune import DEFAULT_MODEL, gpu_busy, load_profile, resolve_local_model  # noqa: E402
from ironmule.qmv_k3840 import VERIFIED_IDENTITY_SHA256  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURED_SOURCES = (
    "ironmule/stacks.py", "ironmule/router.py", "ironmule/service.py",
    "ironmule/executor.py", "ironmule/runtime.py", "ironmule/plans.py",
    "ironmule/paired_research.py", "ironmule/telemetry.py",
    "tools/b57_stack_composition.py",
)

MARGIN = 0.02
LOAD_CEILING = 4.0
BLOCK_DRIFT_LIMIT = 0.25
RSS_LIMIT_FRACTION = 0.60
MIN_FREE_PERCENT = 10.0
_FREE_PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_VM_STAT_SWAPOUTS = re.compile(r"Swapouts:\s*([0-9]+)")

DOCUMENT = (
    "Apple Silicon shares one pool of memory between the CPU and the GPU. A language "
    "model reads its whole weight set once per token, so the rate at which those bytes "
    "arrive sets the speed of decoding, and the arithmetic units wait. Quantisation "
    "reduces the bytes per weight and therefore the bytes per token. A key value cache "
    "grows with every token produced, and its own reads join the weight traffic. "
    "Grouping several requests into one submission amortises host work across them "
    "without changing any tensor shape, which is why the tokens do not move."
) * 3

QUESTIONS = (
    "Summarise the paragraph in one sentence.",
    "Which resource sets the speed of decoding?",
    "What does quantisation reduce?",
    "Why do grouped tokens not move?",
)

# name: (requests, plan_kind, max_tokens, arrivals, objective, stacks)
# Reduced on purpose: a latency stack is not run against a two-request class, and a
# prefix stack is not run against a class that declares no shared prefix.
WORKLOAD_CLASSES = (
    ("single_short", 1, "strict", 32, (0.0,), "latency", ("A", "E")),
    ("single_long", 1, "strict", 128, (0.0,), "latency", ("A", "E")),
    ("session_warm", 3, "session_warm", 32, (0.0, 0.0, 0.0), "latency", ("B", "E")),
    ("session_new", 3, "session_new", 32, (0.0, 0.0, 0.0), "latency", ("B",)),
    ("pair_short", 2, "strict", 32, (0.0, 0.0), "throughput", ("A_t", "C", "E")),
    ("pair_long", 2, "strict", 128, (0.0, 0.0), "throughput", ("A_t", "C", "E")),
    ("pair_staggered", 2, "strict", 32, (0.0, 60.0), "throughput", ("A_t", "C", "E")),
    ("pair_session", 2, "session_warm", 32, (0.0, 0.0), "throughput", ("B_t", "C", "E")),
)

PREREGISTRATION = {
    "experiment": "B57_stack_composition",
    "question": (
        "which combination of the already qualified levers is fastest per workload "
        "class and objective, measured as whole executed stacks rather than as a sum "
        "of separately measured percentages"
    ),
    "stack_model": STACK_MODEL_VERSION,
    "candidates": [{"id": s.id, "members": list(s.members), "objective": s.objective,
                    "reference": s.reference} for s in CANDIDATES],
    "workload_classes": [
        {"name": name, "requests": count, "plan": plan, "max_tokens": tokens,
         "arrivals_ms": list(arrivals), "objective": objective, "stacks": list(stacks)}
        for name, count, plan, tokens, arrivals, objective, stacks in WORKLOAD_CLASSES],
    "reduction_rule": (
        "the candidate set is reduced by ironmule/stacks.py before anything runs: an "
        "internally contradictory stack, one outside its fingerprint, and one whose "
        "member is held or still being measured never execute. Every drop is recorded "
        "with its reason"
    ),
    "phases": {
        "selection": "names one candidate per workload class from its own blocks",
        "confirmation": (
            "a separate run measuring only the selected candidate against its declared "
            "reference, with blocks, interval and gates fixed before it starts"
        ),
    },
    "blocks": 12,
    "arm_order": "rotated by block index",
    "statistic": "per-block ratio, median, 95 per cent bootstrap over 10000 resamples",
    "equivalence_margin": MARGIN,
    "adoption_rule": (
        "a stack enters the composition profile only if its 95 per cent interval against "
        "its declared reference lies entirely below 1.0 in confirmation. An interval that "
        "merely contains a number below 1.0 is not a win, and the reference is kept"
    ),
    "correctness_rule": (
        "token IDs, physical counts and stop reasons identical within a plan kind, "
        "checked before any timing is read. Across plan kinds no identity is claimed or "
        "required: E9 measured the two plans up to 4.31 logits apart, so a prefix stack "
        "is compared against the same plan with reuse dropped, never against a strict "
        "one. The paired path carries its own B45/B51 logit and KV bit contract and is "
        "not re-derived here; no tolerance replaces any of it"
    ),
    "measured_directly": [
        "end to end wall time", "service and engine TTFT", "prefill time", "decode time",
        "tokens per second", "per-request latency", "group completion",
        "paired and solo steps", "peak RSS", "MLX peak memory", "swapout counter",
    ],
    "forbidden": (
        "prefix, autotuner and paired gains are never added arithmetically; only a fully "
        "executed stack counts. No candidate is activated because it once won alone"
    ),
    "blocking": (
        "any fallback, any correctness difference within a plan kind, free memory below 10 "
        "per cent, combined peak RSS above 60 per cent of installed memory, a concurrent "
        "model process at start, a 1-minute load average above 4.0 at start, any block "
        "deviating more than 25 per cent from the median block, and the resource gate "
        "named in the record's `resource_gate.applied`: `legacy` blocks on ANY system-wide "
        "swapout; `pressure` is the gate B65 qualified and blocks when swap in use ends "
        "above where it started, when macOS's own memory pressure level is anything but "
        "normal at any probe, or when that level cannot be read. `legacy` is the default"
    ),
    "not_measured": (
        "B56b's own dispatch, which is deciding separately and may not be composed until "
        "it does; and silicon characterisation, which is not started here"
    ),
}


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


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def system_sample(label: str) -> dict:
    """Between blocks only. A subprocess probe inside a dispatch changes what it reports."""
    swapouts = _VM_STAT_SWAPOUTS.search(_command(["vm_stat"]))
    free = _FREE_PERCENT.search(_command(["memory_pressure", "-Q"]))
    # `B65` added the native readings beside the legacy ones. They cost no subprocess, and
    # the legacy fields are unchanged so a legacy-gated run scores exactly as it did.
    counters = vm_counters()
    level = memory_pressure_level()
    return {"label": label, "swapouts": int(swapouts.group(1)) if swapouts else None,
            "memory_free_percent": float(free.group(1)) if free else None,
            "swap_used_bytes": swap_used_bytes(),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "memory_pressure_level": level,
            "memory_pressure_name": MEMORY_PRESSURE_NAMES.get(level),
            "native_swapouts": counters["swapouts"] if counters else None,
            "native_swapins": counters["swapins"] if counters else None,
            "native_pageouts": counters["pageouts"] if counters else None,
            "native_compressor_pages": counters["compressor_page_count"] if counters else None,
            "native_free_pages": counters["free_count"] if counters else None,
            "native_wired_pages": counters["wire_count"] if counters else None}


def evaluate_resource_gate(samples, *, gate: str, memory_total: int, peak_rss: int) -> dict:
    """Did the machine stay out of the way? One implementation, used by the run and its test.

    `legacy` blocks on any system-wide swapout. `pressure` is what `B65` qualified: the
    swapout counter is recorded but does not decide, swap in use may not end above where it
    started, macOS's own pressure level must read normal at every probe, and a level that
    cannot be read blocks. Both keep the free-memory floor and the RSS ceiling unchanged.
    """
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    swapouts = [s["swapouts"] for s in samples if s["swapouts"] is not None]
    levels = [s.get("memory_pressure_level") for s in samples]

    swapout_delta = max(swapouts) - min(swapouts) if len(swapouts) >= 2 else None
    swap_growth = (max(swaps) - swaps[0]) if swaps else None
    readable = all(level is not None for level in levels)
    abnormal = any(level != MEMORY_PRESSURE_NORMAL for level in levels)

    reasons = []
    if not frees:
        reasons.append("no free-memory probe")
    elif min(frees) < MIN_FREE_PERCENT:
        reasons.append(f"free memory reached {min(frees):.0f} per cent")
    if not memory_total:
        reasons.append("installed memory unknown")
    elif peak_rss > memory_total * RSS_LIMIT_FRACTION:
        reasons.append("peak RSS above the ceiling")
    if gate == "legacy":
        if swapout_delta is None:
            reasons.append("no swapout probe")
        elif swapout_delta != 0:
            reasons.append(f"{swapout_delta} system-wide swapouts")
    else:
        if swap_growth is None:
            reasons.append("no swap probe")
        elif swap_growth > 0:
            reasons.append(f"swap in use rose {swap_growth / 1e9:.2f} GB above its start")
        if not readable:
            reasons.append("memory pressure level unreadable at some probe")
        elif abnormal:
            reasons.append("macOS reported memory pressure above normal")
    return {"gate": gate, "passed": not reasons, "reasons": reasons,
            "swapout_counter_delta": swapout_delta,
            "swap_growth_above_start_bytes": swap_growth,
            "memory_pressure_levels": levels,
            "memory_pressure_readable_at_every_probe": readable,
            "memory_pressure_ever_abnormal": abnormal}


class Bench:
    """One loaded model, and one way to execute any admitted stack over it."""

    def __init__(self, runtime: Runtime, router: ExecutionRouter):
        self.runtime = runtime
        self.routed = AppleRuntime(runtime, router)
        self.document_ids = runtime.encode(DOCUMENT)
        self.session_plan = runtime.session_plan(DOCUMENT, name="composition")
        self.question_ids = [runtime.encode(DOCUMENT + "\n\n" + question)
                             for question in QUESTIONS]
        self.plain_ids = [runtime.encode(question) for question in QUESTIONS]

    def _requests(self, spec, stack_id: str, fresh_plan: bool) -> list[Request]:
        _name, count, plan_kind, max_tokens, arrivals, objective, _stacks = spec
        requests = []
        for index in range(count):
            if plan_kind == "strict":
                plan = StrictOneShotPlan()
                ids = self.plain_ids[index % len(self.plain_ids)]
            else:
                # `session_new` gives every request its own plan, so the cache is always
                # cold; `session_warm` shares one plan. `fresh_plan` is the reference arm:
                # the same chunked prefill with reuse dropped.
                own = plan_kind == "session_new" or fresh_plan
                plan = (ReusableSessionPlan(self.session_plan.prefix_ids, name="cold")
                        if own else self.session_plan)
                ids = self.question_ids[index % len(self.question_ids)]
            requests.append(Request(prompt_ids=list(ids), max_tokens=max_tokens,
                                    plan=plan, arrival_ms=arrivals[index],
                                    objective=objective))
        return requests

    def run(self, spec, stack_id: str, *, fresh_plan: bool = False) -> dict:
        requests = self._requests(spec, stack_id, fresh_plan)
        started = time.perf_counter_ns()
        if stack_id == "E":
            results = self.routed.serve(requests)
            telemetry = self.routed.runtime.telemetry
            paired = self.routed.last_decision["cohorts"][0]["mode_status"]
            routes = self.routed.last_decision["route_by_request"]
        else:
            mode = {"A": InteractiveMode, "B": InteractiveMode,
                    "A_t": ThroughputMode, "B_t": ThroughputMode,
                    "C": PairedThroughputMode}[stack_id]()
            self.runtime.mode = mode
            results = self.runtime.serve(requests)
            telemetry = self.runtime.telemetry
            from ironmule.service import paired_status
            paired = paired_status(mode)
            routes = {r.rid: mode.name for r in requests}
        wall = time.perf_counter_ns() - started
        snapshot = telemetry.snapshot()
        return {
            "wall_ns": wall,
            "order": [r.rid for r in results],
            "tokens": {r.rid: list(r.tokens) for r in results},
            "stop_reasons": {r.rid: r.stop_reason for r in results},
            "plan_kinds": snapshot["plan_kinds"],
            "routes": routes,
            "paired_steps": paired.get("paired_steps", 0),
            "solo_steps_no_partner": paired.get("solo_steps_no_partner", 0),
            "aggregate_tokens_per_second": snapshot["aggregate_tokens_per_second"],
            "latency_p50_ms": snapshot["latency_p50_ms"],
            "service_ttft_p50_ms": snapshot["service_ttft_p50_ms"],
            "engine_ttft_p50_ms": snapshot["engine_ttft_p50_ms"],
            "peak_memory_bytes": snapshot["peak_memory_bytes"],
            "fallbacks": snapshot["fallbacks"],
            "per_request": snapshot["per_request"],
        }


def build_context(identity, profile) -> Context:
    admits = identity.identity_sha256 == VERIFIED_IDENTITY_SHA256
    return Context(hardware_fingerprint=fingerprint(),
                   model_identity_sha256=identity.identity_sha256,
                   architecture=identity.architecture,
                   mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                   profile_present=profile is not None,
                   kernel_admits=admits, paired_admits=admits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--phase", choices=("selection", "confirmation"),
                        default="selection")
    parser.add_argument("--resource-gate", choices=("legacy", "pressure"), default="legacy",
                        help="legacy blocks on any system-wide swapout; pressure is the "
                             "gate B65 qualified. Default stays legacy so an existing "
                             "command reproduces exactly what it did before")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    parser.add_argument("--plan-only", action="store_true",
                        help="print the reduction for this machine and model, run nothing")
    args = parser.parse_args(argv)

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    resolved = resolve_local_model(args.model)
    profile = load_profile(args.model, model_identity=resolved.identity)
    context = build_context(resolved.identity, profile)
    admitted, excluded = reduce(CANDIDATES, context)
    admitted_ids = {stack.id for stack in admitted}

    if args.plan_only:
        print(json.dumps({
            "model": resolved.identity.model_id,
            "identity_sha256": resolved.identity.identity_sha256,
            "profile_present": profile is not None,
            "admitted": sorted(admitted_ids),
            "excluded": [{"stack": row.stack_id, "reason": row.reason,
                          "detail": row.detail} for row in excluded],
            "runnable_classes": {
                name: [s for s in stacks if s in admitted_ids]
                for name, _c, _p, _t, _a, _o, stacks in WORKLOAD_CLASSES},
        }, indent=2))
        return 0

    if profile is None:
        raise SystemExit(f"no confirmed profile for {args.model}; tune it before "
                         "composing anything on it")
    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")
    load_start = os.getloadavg()
    if load_start[0] > LOAD_CEILING:
        raise SystemExit(f"1-minute load average is {load_start[0]:.2f}, above the "
                         f"{LOAD_CEILING} ceiling; refusing to measure on a busy machine")

    swap_start = swap_used_bytes()
    runtime = Runtime.load(args.model, mode=InteractiveMode())
    router = ExecutionRouter(profile, identity_sha256=resolved.identity.identity_sha256,
                             fingerprint=fingerprint(), mlx=mx.__version__,
                             mlx_lm=mlx_lm.__version__)
    bench = Bench(runtime, router)

    plan = [(spec, [s for s in spec[6] if s in admitted_ids]) for spec in WORKLOAD_CLASSES]
    plan = [(spec, stacks) for spec, stacks in plan if stacks]
    samples = [system_sample("start")]
    blocks: list[dict] = []
    try:
        for spec, stacks in plan:                        # warm-up, discarded
            for stack_id in stacks:
                bench.run(spec, stack_id)
            if spec[2] != "strict":
                bench.run(spec, stacks[0], fresh_plan=True)

        for block in range(args.blocks):
            record: dict = {"block": block, "runs": {}}
            for spec, stacks in plan:
                name = spec[0]
                shift = block % max(1, len(stacks))
                sequence = stacks[shift:] + stacks[:shift]
                record["runs"][name] = {}
                for stack_id in sequence:
                    record["runs"][name][stack_id] = bench.run(spec, stack_id)
                if spec[2] != "strict":
                    # The prefix reference: same plan, reuse dropped, same tokens.
                    record["runs"][name]["reference_cold_prefix"] = bench.run(
                        spec, stacks[0], fresh_plan=True)
            blocks.append(record)
            samples.append(system_sample(f"after_block_{block}"))
            print(f"block {block} done")
    finally:
        runtime.close()

    samples.append(system_sample("end"))
    swap_end = swap_used_bytes()

    # -- correctness, within a plan kind, before any timing -------------------
    differences = []
    for record in blocks:
        for name, runs in record["runs"].items():
            by_plan: dict[str, list] = {}
            for arm, run in runs.items():
                key = tuple(sorted(run["plan_kinds"]))
                by_plan.setdefault(str(key), []).append((arm, run))
            for key, group in by_plan.items():
                first_arm, first = group[0]
                for arm, run in group[1:]:
                    ordered_first = [first["tokens"][rid] for rid in first["order"]]
                    ordered = [run["tokens"][rid] for rid in run["order"]]
                    if ordered != ordered_first:
                        differences.append({"block": record["block"], "class": name,
                                            "plan": key, "arms": [first_arm, arm],
                                            "field": "tokens"})
                    if ([run["stop_reasons"][rid] for rid in run["order"]]
                            != [first["stop_reasons"][rid] for rid in first["order"]]):
                        differences.append({"block": record["block"], "class": name,
                                            "plan": key, "arms": [first_arm, arm],
                                            "field": "stop_reason"})

    # -- comparisons ----------------------------------------------------------
    comparisons: dict[str, dict] = {}
    for spec, stacks in plan:
        name = spec[0]
        reference = ("reference_cold_prefix" if spec[2] != "strict"
                     else ("A_t" if spec[5] == "throughput" else "A"))
        if reference not in blocks[0]["runs"][name]:
            reference = stacks[0]
        comparisons[name] = {"reference": reference, "arms": {}}
        for stack_id in stacks:
            if stack_id == reference:
                continue
            comparisons[name]["arms"][stack_id] = bootstrap([
                record["runs"][name][stack_id]["wall_ns"]
                / record["runs"][name][reference]["wall_ns"] for record in blocks])

    # -- resources ------------------------------------------------------------
    memory_total = int(static_facts().get("memory_bytes") or 0)
    swapouts = [s["swapouts"] for s in samples if s["swapouts"] is not None]
    swapout_delta = max(swapouts) - min(swapouts) if len(swapouts) >= 2 else None
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    peak_rss = max(s["peak_rss_bytes"] for s in samples)
    block_walls = [sum(run["wall_ns"] / 1e6 for runs in record["runs"].values()
                       for run in runs.values()) for record in blocks]
    wall_median = median(block_walls)
    disturbed = [{"block": i, "wall_ms": w, "deviation": w / wall_median - 1.0}
                 for i, w in enumerate(block_walls)
                 if abs(w / wall_median - 1.0) > BLOCK_DRIFT_LIMIT]
    fallbacks = sum(run["fallbacks"] for record in blocks
                    for runs in record["runs"].values() for run in runs.values())
    gate = evaluate_resource_gate(samples, gate=args.resource_gate,
                                  memory_total=memory_total, peak_rss=peak_rss)
    resources_ok = gate["passed"]

    blocked = (bool(differences) or fallbacks or bool(disturbed) or not resources_ok)

    # -- composition profile: only what beat its reference outright ------------
    composition = []
    for spec, stacks in plan:
        name = spec[0]
        reference = comparisons[name]["reference"]
        winners = [(stack_id, row) for stack_id, row in comparisons[name]["arms"].items()
                   if row["ci_high"] is not None and row["ci_high"] < 1.0]
        best = min(winners, key=lambda item: item[1]["median"]) if winners else None
        composition.append({
            "hardware_fingerprint": fingerprint(),
            "model_id": resolved.identity.model_id,
            "model_identity_sha256": resolved.identity.identity_sha256,
            "workload_class": name,
            "objective": spec[5],
            "reference": reference,
            "best_stack": best[0] if best else reference,
            "ratio_median": best[1]["median"] if best else 1.0,
            "ratio_ci": [best[1]["ci_low"], best[1]["ci_high"]] if best else [1.0, 1.0],
            "members": list(next(s for s in CANDIDATES if s.id == best[0]).members)
            if best else None,
            "adopted": bool(best),
            "reason": ("interval entirely below 1.0 against its reference" if best
                       else "no candidate beat the reference outright; reference kept"),
        })

    verdict = ("BLOCKED" if blocked
               else "SELECTION_COMPLETE" if args.phase == "selection"
               else "CONFIRMED")

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "phase": args.phase,
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "stack_model": {
            "version": STACK_MODEL_VERSION,
            "optimisations": {row.id: {
                "phase": row.phase, "objective": row.objective,
                "activation": row.activation,
                "fingerprint_scope": list(row.fingerprint_scope),
                "incompatible_with": list(row.incompatible_with),
                "correctness_contract": row.correctness_contract,
                "evidence_ids": list(row.evidence_ids)} for row in OPTIMISATIONS.values()},
            "admitted": sorted(admitted_ids),
            "excluded": [{"stack": row.stack_id, "reason": row.reason,
                          "detail": row.detail} for row in excluded],
        },
        "environment": {
            "platform": platform.platform(), "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__, "hardware_fingerprint": fingerprint(),
            "model_id": resolved.identity.model_id,
            "model_revision": resolved.identity.revision,
            "model_identity_sha256": resolved.identity.identity_sha256,
            "tuned_knobs": profile["knobs"],
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "blocks": len(blocks),
        "correctness": {"differences": differences, "identical_within_plan": not differences},
        "comparisons": comparisons,
        "composition_profile": composition,
        "resource_gate": dict(gate, applied=args.resource_gate),
        "resources": {"swap_used_bytes_start": swap_start, "swap_used_bytes_end": swap_end,
                      "swapout_counter_delta": swapout_delta,
                      "min_memory_free_percent": min(frees) if frees else None,
                      "peak_rss_bytes": peak_rss, "memory_total_bytes": memory_total,
                      "fallbacks": fallbacks, "samples": samples,
                      "load_average_start": list(load_start),
                      "load_average_end": list(os.getloadavg())},
        "stability": {"block_wall_ms": block_walls, "median_block_wall_ms": wall_median,
                      "disturbed_blocks": disturbed, "stable": not disturbed},
        "verdict": verdict,
        "raw_blocks": blocks,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "admitted": sorted(admitted_ids),
                      "composition_profile": composition,
                      "comparisons": {k: {"reference": v["reference"],
                                          "arms": {a: {"median": r["median"],
                                                       "ci": [r["ci_low"], r["ci_high"]]}
                                                   for a, r in v["arms"].items()}}
                                      for k, v in comparisons.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
