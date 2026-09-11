#!/usr/bin/env python3
"""Can a late latency request be protected by its own dispatch while a group is running?

`B56` left exactly one case open. A `latency` request that becomes servable after a
`throughput` cohort has already started waits for it and finishes at `2.10x` its own
solo latency. Letting it join the group instead gave `2.05x`. A third number, `1.00x`,
was recorded for "its own dispatch" -- but that measurement served the request with *no
throughput work running at all*, so it measured the empty machine and answered nothing.
This experiment measures the case that was actually asked about.

Four arms, one throughput preload, one latency request, two OS processes:

    A_child     the latency request alone on the child's engine. The floor.
    A_control   the same again, in the same repetition. The A/A control.
    A_parent    the same request alone on the parent's engine. Says whether the two
                engines are interchangeable, because `B` runs on one and `C` on the other.
    B_router    today's router: one dispatch holding the cohort at arrival 0 and the
                latency request at arrival 60 ms. The cohort leads and the request waits.
    C_own       the parent serves the cohort; at the same 60 ms the child serves the
                latency request from its own process, on its own engine.

`C` is only a comparison if it really ran while the cohort was running. That is a hard
gate, not an assumption: every repetition must show the child's first token stamped
before the parent cohort's last token. `time.perf_counter_ns` is system-wide on macOS, so
those two stamps are on one timeline and the comparison is direct.

**No preemption is claimed.** Two processes submitting to one GPU is an observed
scheduling outcome and nothing here inspects the device's scheduler. Protection that is
bought by starving the cohort is not protection, so the cohort's own completion is a
gated measurement too.
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

from ironmule.hw import fingerprint, swap_used_bytes  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.router import AppleRuntime, ExecutionRouter  # noqa: E402
from ironmule.service import InteractiveMode, Request, Runtime, ThroughputMode  # noqa: E402
from ironmule.tune import DEFAULT_MODEL, gpu_busy, load_profile, resolve_local_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURED_SOURCES = (
    "ironmule/router.py", "ironmule/service.py", "ironmule/executor.py",
    "ironmule/runtime.py", "ironmule/telemetry.py",
    "tools/b56b_own_dispatch.py", "tools/b56b_latency_worker.py",
)

PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
    "Summarise what a tokenizer does before a model sees any text.",
    "State one reason a quantised model reads fewer bytes per token.",
)
COHORT_PROMPTS = (0, 2)          # the throughput preload
LATENCY_PROMPT = 1               # the request whose latency is the question
ARRIVAL_MS = 60.0
MAX_TOKENS = 32
REPETITIONS = 12
LOAD_CEILING = 4.0
REPETITION_DRIFT_LIMIT = 0.25
SWAP_GROWTH_LIMIT = 64 * 1024 ** 2
# G7 reuses the project's own safe limits rather than inventing new ones:
# `ironmule_product.memory.RSS_LIMIT_FRACTION` and
# `ironmule_product.readiness.SystemLimits.min_free_percent`.
RSS_LIMIT_FRACTION = 0.60
MIN_FREE_PERCENT = 10.0
_FREE_PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
_VM_STAT_SWAPOUTS = re.compile(r"Swapouts:\s*([0-9]+)")

# Fixed here, before any measurement, and not revisited afterwards.
GATES = {
    "G1_protection": {
        "statistic": "C_own over A_child, completion latency of the latency request",
        "requirement": "95 per cent upper bound below 1.30",
        "limit": 1.30,
        "why": (
            "waiting costs 2.10x. Protection means the request keeps most of its solo "
            "behaviour, so the limit is set at less than a third of what waiting costs. "
            "It is not 1.00: one device shared with a running group cannot be free, and "
            "a limit that could only be met by an idle machine would test nothing."
        ),
    },
    "G2_beats_waiting": {
        "statistic": "C_own over B_router, completion latency of the latency request",
        "requirement": "95 per cent upper bound below 0.75",
        "limit": 0.75,
        "why": "a strategy that does not clearly beat waiting is not worth a second path",
    },
    "G3_cohort_not_starved": {
        "statistic": "C_own over B_router, completion of the throughput cohort",
        "requirement": "95 per cent upper bound below 1.25",
        "limit": 1.25,
        "why": (
            "protection bought by starving the running group is not protection. The "
            "cohort shares a device in C and owns it in B, so some cost is expected; "
            "a quarter of its own runtime is the most that can be called sharing"
        ),
    },
    "G4_aa_control": {
        "statistic": "A_control over A_child",
        "requirement": "95 per cent interval contains 1.00",
        "why": "a control that cannot find its own arm cannot support the others",
    },
    "G4b_engine_control": {
        "statistic": "A_parent over A_child",
        "requirement": "95 per cent interval inside 0.95 to 1.05",
        "why": (
            "B measures the latency request on the parent engine and C on the child's. "
            "If the two engines are not interchangeable, C over B is not a comparison"
        ),
    },
    "G5_correctness": {
        "requirement": (
            "token IDs and stop reasons of the latency request identical across every "
            "arm and against a solo reference; the cohort identical between B and C. "
            "Model arithmetic is untouched, so the B51 logit/KV identity contracts are "
            "carried, not re-derived"
        ),
    },
    "G6_concurrency": {
        "requirement": (
            "in every repetition, the child's first token is stamped strictly before "
            "the parent cohort's last token. A C that ran after the cohort finished is "
            "not an own dispatch and voids the run"
        ),
    },
    "G7_resources": {
        "requirement": (
            "zero swapouts over the whole session (vm_stat Swapouts counter delta), "
            "system free memory at or above 10 per cent at every sample, and combined "
            "peak RSS of both processes at or below 60 per cent of installed memory"
        ),
        "limits": {"swapout_delta": 0, "min_free_percent": MIN_FREE_PERCENT,
                   "rss_limit_fraction": RSS_LIMIT_FRACTION},
        "why": (
            "C keeps two models resident. That is part of what C costs and it is not "
            "allowed to disappear from the result, so peak RSS of both processes, the "
            "MLX peak per process and the model's own weight bytes are reported whether "
            "the gate passes or not. The limits are the project's existing ones "
            "(ironmule_product.memory.RSS_LIMIT_FRACTION, readiness min_free_percent), "
            "not new numbers chosen for this run"
        ),
        "sampling": (
            "subprocess probes (vm_stat, memory_pressure) run only between repetitions, "
            "never inside a dispatch: B55 measured a sysctl subprocess costing 17.4 ms "
            "and changing the number it was reporting on. ru_maxrss and the MLX counters "
            "need no subprocess and are read at every repetition boundary"
        ),
    },
}

PREREGISTRATION = {
    "experiment": "B56b_own_dispatch",
    "question": (
        "can a latency request that becomes servable while a throughput cohort is "
        "already running be protected by dispatching it separately"
    ),
    "arms": ["A_child", "A_control", "A_parent", "B_router", "C_own"],
    "workload": {
        "throughput_cohort_prompts": list(COHORT_PROMPTS),
        "latency_prompt": LATENCY_PROMPT,
        "latency_arrival_ms": ARRIVAL_MS,
        "max_tokens": MAX_TOKENS,
    },
    "repetitions": REPETITIONS,
    "arm_order": "rotated by repetition index",
    "statistic": "per-repetition ratio, median, 95 per cent bootstrap over 10000 resamples",
    "gates": GATES,
    "decision_rule": (
        "B56b_GO only if G1 to G7 all hold in two separate confirmation sessions, each "
        "its own process. A single passing session is not a result. Thresholds are fixed "
        "here and are not revisited after the run, and no session is extended to reach "
        "significance."
    ),
    "blocking": (
        "any fallback, any correctness difference, any swap growth above 64 MiB, any "
        "swapout during the session, a concurrent model process at start, a 1-minute "
        "load average above 4.0 at start, or any repetition whose A_child deviates more "
        "than 25 per cent from the median yields BLOCKED and never a verdict"
    ),
    "semantics": (
        "no GPU preemption is claimed or measured. A separate dispatch is named as an "
        "observed scheduling strategy only. Nothing here inspects the device scheduler"
    ),
    "not_measured": (
        "B56a is unchanged and not re-decided here; no kernel, no model arithmetic, and "
        "no paired path (B58) is touched"
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


def _command(argv: list[str]) -> str:
    """A subprocess probe. Only ever called between repetitions, never inside one."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def system_sample(label: str) -> dict:
    """What the machine's memory is doing, sampled at a repetition boundary."""
    swapouts = _VM_STAT_SWAPOUTS.search(_command(["vm_stat"]))
    free = _FREE_PERCENT.search(_command(["memory_pressure", "-Q"]))
    return {
        "label": label,
        "at_ns": time.perf_counter_ns(),
        "swapouts": int(swapouts.group(1)) if swapouts else None,
        "memory_free_percent": float(free.group(1)) if free else None,
        "swap_used_bytes": swap_used_bytes(),
        "parent_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "parent_mlx_peak_bytes": int(mx.get_peak_memory()),
        "parent_mlx_active_bytes": int(mx.get_active_memory()),
    }


def bootstrap(ratios: list[float], resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0, "ratios": []}
    draws = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios),
            "ratios": ratios}


class Child:
    """The other process. Owns its own engine and its own submission to the device."""

    def __init__(self, model_id: str):
        self.process = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "tools" / "b56b_latency_worker.py"),
             model_id, json.dumps(list(PROMPTS))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = json.loads(self.process.stdout.readline())
        if not ready.get("ready"):
            raise SystemExit("latency worker did not come up")
        self.pid = ready["pid"]
        self.load_ns = ready["load_ns"]
        self.load_resources = ready["resources"]

    def stats(self) -> dict:
        self.process.stdin.write(json.dumps({"cmd": "stats"}) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())["resources"]

    def ask(self, *, at_ns: int | None, prompt_index: int, max_tokens: int) -> dict:
        self.process.stdin.write(json.dumps(
            {"cmd": "serve", "at_ns": at_ns, "prompt_index": prompt_index,
             "max_tokens": max_tokens}) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def send(self, *, at_ns: int, prompt_index: int, max_tokens: int) -> None:
        """Hand over the command and return at once, so the parent can start its own work."""
        self.process.stdin.write(json.dumps(
            {"cmd": "serve", "at_ns": at_ns, "prompt_index": prompt_index,
             "max_tokens": max_tokens}) + "\n")
        self.process.stdin.flush()

    def collect(self) -> dict:
        return json.loads(self.process.stdout.readline())

    def close(self) -> None:
        try:
            self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait(timeout=30)


def cohort_requests(prompt_ids) -> list[Request]:
    return [Request(prompt_ids=list(prompt_ids[index]), max_tokens=MAX_TOKENS,
                    plan=StrictOneShotPlan(), objective="throughput", arrival_ms=0.0)
            for index in COHORT_PROMPTS]


def parent_cohort_only(runtime: Runtime, prompt_ids) -> dict:
    """The throughput cohort on the parent's engine, on its qualified path."""
    runtime.mode = ThroughputMode()
    requests = cohort_requests(prompt_ids)
    results = runtime.serve(requests)
    rows = {m.rid: m for m in runtime.telemetry.requests}
    return {
        "tokens": {r.rid: list(r.tokens) for r in results},
        "stop_reasons": {r.rid: r.stop_reason for r in results},
        "order": [r.rid for r in results],
        "last_token_ns": max(rows[r.rid].finished_ns for r in results),
        "first_token_ns": min(rows[r.rid].first_token_ns for r in results),
        "arrival_ns": min(rows[r.rid].arrival_ns for r in results),
        "per_request": {r.rid: {"arrival_ns": rows[r.rid].arrival_ns,
                                "engine_start_ns": rows[r.rid].engine_start_ns,
                                "first_token_ns": rows[r.rid].first_token_ns,
                                "finished_ns": rows[r.rid].finished_ns,
                                "generated_tokens": rows[r.rid].generated_tokens}
                        for r in results},
        "fallbacks": runtime.telemetry.fallbacks,
    }


def parent_solo_latency(runtime: Runtime, prompt_ids) -> dict:
    """The latency request alone on the parent's engine. The cross-engine control."""
    runtime.mode = InteractiveMode()
    request = Request(prompt_ids=list(prompt_ids[LATENCY_PROMPT]),
                      max_tokens=MAX_TOKENS, plan=StrictOneShotPlan())
    result = runtime.serve([request])[0]
    metrics = runtime.telemetry.requests[0]
    return {"tokens": list(result.tokens), "stop_reason": result.stop_reason,
            "arrival_ns": metrics.arrival_ns, "engine_start_ns": metrics.engine_start_ns,
            "first_token_ns": metrics.first_token_ns,
            "finished_ns": metrics.finished_ns,
            "generated_tokens": metrics.generated_tokens,
            "fallbacks": runtime.telemetry.fallbacks}


def parent_router_mixed(routed: AppleRuntime, prompt_ids) -> dict:
    """Today's router: cohort at arrival 0, latency request at arrival 60 ms, one dispatch."""
    requests = [
        Request(prompt_ids=list(prompt_ids[COHORT_PROMPTS[0]]), max_tokens=MAX_TOKENS,
                plan=StrictOneShotPlan(), objective="throughput", arrival_ms=0.0),
        Request(prompt_ids=list(prompt_ids[LATENCY_PROMPT]), max_tokens=MAX_TOKENS,
                plan=StrictOneShotPlan(), objective="latency", arrival_ms=ARRIVAL_MS),
        Request(prompt_ids=list(prompt_ids[COHORT_PROMPTS[1]]), max_tokens=MAX_TOKENS,
                plan=StrictOneShotPlan(), objective="throughput", arrival_ms=0.0),
    ]
    results = routed.serve(requests)
    decision = routed.last_decision
    rows = {row["rid"]: row for row in decision["telemetry"]["per_request"]}
    by_rid = {r.rid: r for r in results}
    latency_rid = requests[1].rid
    cohort_rids = [requests[0].rid, requests[2].rid]
    metrics = {m.rid: m for m in routed.runtime.telemetry.requests}
    return {
        "latency": {
            "tokens": list(by_rid[latency_rid].tokens),
            "stop_reason": by_rid[latency_rid].stop_reason,
            "arrival_ns": metrics[latency_rid].arrival_ns,
            "engine_start_ns": metrics[latency_rid].engine_start_ns,
            "first_token_ns": metrics[latency_rid].first_token_ns,
            "finished_ns": metrics[latency_rid].finished_ns,
            "generated_tokens": metrics[latency_rid].generated_tokens,
        },
        "cohort": {
            "tokens": {rid: list(by_rid[rid].tokens) for rid in cohort_rids},
            "stop_reasons": {rid: by_rid[rid].stop_reason for rid in cohort_rids},
            "order": cohort_rids,
            "arrival_ns": min(metrics[rid].arrival_ns for rid in cohort_rids),
            "first_token_ns": min(metrics[rid].first_token_ns for rid in cohort_rids),
            "last_token_ns": max(metrics[rid].finished_ns for rid in cohort_rids),
        },
        "latency_protected": decision["latency_protected"],
        "cohort_order": decision["cohort_order"],
        "fallbacks": decision["fallbacks"],
        "unused": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help=("functional check only: skips the load ceiling and marks "
                              "the record so it can never be read as evidence"))
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
    if load_start[0] > LOAD_CEILING and not args.smoke:
        raise SystemExit(f"1-minute load average is {load_start[0]:.2f}, above the "
                         f"{LOAD_CEILING} ceiling; refusing to measure on a busy machine")

    resolved = resolve_local_model(args.model)
    profile = load_profile(args.model, model_identity=resolved.identity)
    if profile is None:
        raise SystemExit("no tuned profile for this machine and model")

    swap_start = swap_used_bytes()
    runtime = Runtime.load(args.model, mode=InteractiveMode())
    router = ExecutionRouter(profile, identity_sha256=resolved.identity.identity_sha256,
                             fingerprint=fingerprint(), mlx=mx.__version__,
                             mlx_lm=mlx_lm.__version__)
    routed = AppleRuntime(runtime, router)
    prompt_ids = [runtime.encode(prompt) for prompt in PROMPTS]
    child = Child(args.model)

    samples: list[dict] = [dict(system_sample("session_start"),
                                child=child.load_resources)]
    repetitions: list[dict] = []
    try:
        reference = parent_solo_latency(runtime, prompt_ids)
        for _ in range(2):                                # warm-up, discarded
            child.ask(at_ns=None, prompt_index=LATENCY_PROMPT, max_tokens=MAX_TOKENS)
            parent_cohort_only(runtime, prompt_ids)
            parent_router_mixed(routed, prompt_ids)

        order = ["A_child", "A_control", "A_parent", "B_router", "C_own"]
        for index in range(args.repetitions):
            shift = index % len(order)
            sequence = order[shift:] + order[:shift]
            record: dict = {"repetition": index, "arm_order": list(sequence), "runs": {}}
            for arm in sequence:
                if arm in ("A_child", "A_control"):
                    record["runs"][arm] = {"latency": child.ask(
                        at_ns=None, prompt_index=LATENCY_PROMPT, max_tokens=MAX_TOKENS)}
                elif arm == "A_parent":
                    record["runs"][arm] = {
                        "latency": parent_solo_latency(runtime, prompt_ids)}
                elif arm == "B_router":
                    record["runs"][arm] = parent_router_mixed(routed, prompt_ids)
                else:
                    # Hand the child an absolute deadline, then start the cohort here.
                    # Both processes then run without either waiting on the other.
                    start = time.perf_counter_ns() + 20_000_000
                    child.send(at_ns=start + int(ARRIVAL_MS * 1e6),
                               prompt_index=LATENCY_PROMPT, max_tokens=MAX_TOKENS)
                    while time.perf_counter_ns() < start:
                        pass
                    cohort = parent_cohort_only(runtime, prompt_ids)
                    record["runs"][arm] = {"latency": child.collect(), "cohort": cohort,
                                           "cohort_started_ns": start}
            repetitions.append(record)
            samples.append(dict(system_sample(f"after_repetition_{index}"),
                                child=child.stats()))
            print(f"repetition {index} done ({'->'.join(sequence)})")
    finally:
        child.close()
        runtime.close()

    samples.append(system_sample("session_end"))
    swap_end = swap_used_bytes()
    load_end = os.getloadavg()

    def latency_ms(record, arm):
        row = record["runs"][arm]["latency"]
        return (row["finished_ns"] - row["arrival_ns"]) / 1e6

    def cohort_ms(record, arm):
        block = record["runs"][arm]["cohort"]
        return (block["last_token_ns"] - block["arrival_ns"]) / 1e6

    # -- G6 concurrency: was C actually running while the cohort was? ----------
    concurrency = []
    for record in repetitions:
        run = record["runs"]["C_own"]
        latency, cohort = run["latency"], run["cohort"]
        concurrency.append({
            "repetition": record["repetition"],
            "child_dispatch_ns": latency["dispatch_ns"],
            "child_first_token_ns": latency["first_token_ns"],
            "child_finished_ns": latency["finished_ns"],
            "cohort_first_token_ns": cohort["first_token_ns"],
            "cohort_last_token_ns": cohort["last_token_ns"],
            "child_dispatched_before_cohort_end":
                latency["dispatch_ns"] < cohort["last_token_ns"],
            "child_first_token_before_cohort_end":
                latency["first_token_ns"] < cohort["last_token_ns"],
            "overlap_ms": (min(latency["finished_ns"], cohort["last_token_ns"])
                           - max(latency["dispatch_ns"], cohort["first_token_ns"])) / 1e6,
        })
    g6 = all(row["child_first_token_before_cohort_end"] for row in concurrency)

    # -- G5 correctness -------------------------------------------------------
    differences = []
    for record in repetitions:
        for arm in ("A_child", "A_control", "A_parent", "C_own"):
            row = record["runs"][arm]["latency"]
            if row["tokens"] != reference["tokens"]:
                differences.append({"repetition": record["repetition"], "arm": arm,
                                    "field": "tokens"})
            if row["stop_reason"] != reference["stop_reason"]:
                differences.append({"repetition": record["repetition"], "arm": arm,
                                    "field": "stop_reason"})
        router_latency = record["runs"]["B_router"]["latency"]
        if router_latency["tokens"] != reference["tokens"]:
            differences.append({"repetition": record["repetition"], "arm": "B_router",
                                "field": "tokens"})
        if router_latency["stop_reason"] != reference["stop_reason"]:
            differences.append({"repetition": record["repetition"], "arm": "B_router",
                                "field": "stop_reason"})
        b_cohort = [record["runs"]["B_router"]["cohort"]["tokens"][rid]
                    for rid in record["runs"]["B_router"]["cohort"]["order"]]
        c_cohort = [record["runs"]["C_own"]["cohort"]["tokens"][rid]
                    for rid in record["runs"]["C_own"]["cohort"]["order"]]
        if b_cohort != c_cohort:
            differences.append({"repetition": record["repetition"], "arm": "cohort",
                                "field": "tokens"})
    g5 = not differences

    # -- ratios ---------------------------------------------------------------
    comparisons = {
        "G1_C_over_A": bootstrap([latency_ms(r, "C_own") / latency_ms(r, "A_child")
                                  for r in repetitions]),
        "G2_C_over_B": bootstrap([latency_ms(r, "C_own") / latency_ms(r, "B_router")
                                  for r in repetitions]),
        "G3_cohort_C_over_B": bootstrap([cohort_ms(r, "C_own") / cohort_ms(r, "B_router")
                                         for r in repetitions]),
        "G4_A_control_over_A": bootstrap([latency_ms(r, "A_control")
                                          / latency_ms(r, "A_child") for r in repetitions]),
        "G4b_A_parent_over_A_child": bootstrap([latency_ms(r, "A_parent")
                                                / latency_ms(r, "A_child")
                                                for r in repetitions]),
    }
    g1 = comparisons["G1_C_over_A"]["ci_high"] < GATES["G1_protection"]["limit"]
    g2 = comparisons["G2_C_over_B"]["ci_high"] < GATES["G2_beats_waiting"]["limit"]
    g3 = comparisons["G3_cohort_C_over_B"]["ci_high"] < GATES["G3_cohort_not_starved"]["limit"]
    g4 = (comparisons["G4_A_control_over_A"]["ci_low"] <= 1.0
          <= comparisons["G4_A_control_over_A"]["ci_high"])
    g4b = (comparisons["G4b_A_parent_over_A_child"]["ci_low"] >= 0.95
           and comparisons["G4b_A_parent_over_A_child"]["ci_high"] <= 1.05)

    # -- stability and blocking ----------------------------------------------
    floors = [latency_ms(record, "A_child") for record in repetitions]
    floor_median = median(floors)
    disturbed = [{"repetition": index, "latency_ms": value,
                  "deviation": value / floor_median - 1.0}
                 for index, value in enumerate(floors)
                 if abs(value / floor_median - 1.0) > REPETITION_DRIFT_LIMIT]
    fallbacks = sum(
        run.get("latency", {}).get("fallbacks", 0) or 0
        for record in repetitions for run in record["runs"].values()) + sum(
        run.get("cohort", {}).get("fallbacks", 0) or 0
        for record in repetitions for run in record["runs"].values())
    swap_growth = (None if swap_start is None or swap_end is None
                   else swap_end - swap_start)

    # -- G7 resources: two resident models are part of what C costs ------------
    from ironmule.hw import static_facts
    memory_total = int(static_facts().get("memory_bytes") or 0)
    swapout_counts = [s["swapouts"] for s in samples if s["swapouts"] is not None]
    swapout_delta = (max(swapout_counts) - min(swapout_counts)
                     if len(swapout_counts) >= 2 else None)
    free_percents = [s["memory_free_percent"] for s in samples
                     if s["memory_free_percent"] is not None]
    parent_peak_rss = max(s["parent_peak_rss_bytes"] for s in samples)
    child_samples = [s["child"] for s in samples if s.get("child")]
    child_peak_rss = max(row["peak_rss_bytes"] for row in child_samples)
    combined_peak_rss = parent_peak_rss + child_peak_rss
    rss_limit = int(memory_total * RSS_LIMIT_FRACTION) if memory_total else None
    resources = {
        "memory_total_bytes": memory_total,
        "model_weight_bytes": int(getattr(resolved.identity, "manifest_bytes", 0) or 0),
        "resident_models": 2,
        "parent_peak_rss_bytes": parent_peak_rss,
        "child_peak_rss_bytes": child_peak_rss,
        "combined_peak_rss_bytes": combined_peak_rss,
        "combined_peak_rss_fraction": (combined_peak_rss / memory_total
                                       if memory_total else None),
        "rss_limit_bytes": rss_limit,
        "parent_mlx_peak_bytes": max(s["parent_mlx_peak_bytes"] for s in samples),
        "child_mlx_peak_bytes": max(row["mlx_peak_memory_bytes"] for row in child_samples),
        "swapout_counter_delta": swapout_delta,
        "min_memory_free_percent": min(free_percents) if free_percents else None,
        "samples": samples,
    }
    g7 = (swapout_delta == 0
          and bool(free_percents) and min(free_percents) >= MIN_FREE_PERCENT
          and rss_limit is not None and combined_peak_rss <= rss_limit)

    gates = {"G1_protection": g1, "G2_beats_waiting": g2, "G3_cohort_not_starved": g3,
             "G4_aa_control": g4, "G4b_engine_control": g4b, "G5_correctness": g5,
             "G6_concurrency": g6, "G7_resources": g7}
    blocked = (bool(disturbed) or fallbacks or not g5 or not g6
               or swapout_delta != 0
               or (swap_growth is not None and swap_growth > SWAP_GROWTH_LIMIT))
    verdict = ("SMOKE_NOT_EVIDENCE" if args.smoke
               else "BLOCKED_UNSTABLE" if disturbed
               else "BLOCKED" if blocked
               else "B56b_SESSION_GO" if all(gates.values())
               else "B56b_SESSION_NO_GO")

    observed = {
        "latency_ms": {arm: median([latency_ms(r, arm) for r in repetitions])
                       for arm in ("A_child", "A_control", "A_parent", "B_router", "C_own")},
        "cohort_ms": {arm: median([cohort_ms(r, arm) for r in repetitions])
                      for arm in ("B_router", "C_own")},
        "child_queue_ms": median([
            (r["runs"]["C_own"]["latency"]["engine_start_ns"]
             - r["runs"]["C_own"]["latency"]["dispatch_ns"]) / 1e6 for r in repetitions]),
        "child_ttft_ms": {
            arm: median([(r["runs"][arm]["latency"]["first_token_ns"]
                          - r["runs"][arm]["latency"]["arrival_ns"]) / 1e6
                         for r in repetitions])
            for arm in ("A_child", "C_own")},
        "overlap_ms": median([row["overlap_ms"] for row in concurrency]),
        "cohort_tokens_per_second": {
            arm: median([
                sum(len(r["runs"][arm]["cohort"]["tokens"][rid])
                    for rid in r["runs"][arm]["cohort"]["order"])
                / (cohort_ms(r, arm) / 1e3) for r in repetitions])
            for arm in ("B_router", "C_own")},
        "latency_tokens_per_second": {
            arm: median([r["runs"][arm]["latency"]["generated_tokens"]
                         / (latency_ms(r, arm) / 1e3) for r in repetitions])
            for arm in ("A_child", "B_router", "C_own")},
    }

    record = {
        "experiment": PREREGISTRATION["experiment"],
        # A smoke record checks that the machinery runs. It is not evidence, it is never
        # compared with a session, and the flag stays in the file so it cannot be.
        "smoke_functional_check_only": bool(args.smoke),
        "session": None if args.smoke else args.session,
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
            "parent_pid": os.getpid(), "child_pid": child.pid,
            "child_load_ns": child.load_ns,
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "repetitions": len(repetitions),
        "gates": gates,
        "comparisons": comparisons,
        "concurrency": concurrency,
        "correctness": {"differences": differences, "identical": g5,
                        "reference_tokens": len(reference["tokens"])},
        "stability": {"a_child_latency_ms": floors,
                      "median_latency_ms": floor_median,
                      "disturbed_repetitions": disturbed, "stable": not disturbed},
        "resources": {"swap_used_bytes_start": swap_start, "swap_used_bytes_end": swap_end,
                      "swap_growth_bytes": swap_growth, "fallbacks": fallbacks,
                      "load_average_start": list(load_start),
                      "load_average_end": list(load_end),
                      **resources},
        "observed": observed,
        "verdict": verdict,
        "raw_repetitions": repetitions,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    summary_resources = {k: v for k, v in resources.items() if k != "samples"}
    print(json.dumps({"verdict": verdict, "gates": gates, "observed": observed,
                      "resources": summary_resources,
                      "comparisons": {k: {"median": v["median"],
                                          "ci": [v["ci_low"], v["ci_high"]]}
                                      for k, v in comparisons.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
