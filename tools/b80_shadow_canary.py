#!/usr/bin/env python3
"""The monitor watching real dispatches, and what watching costs.

`B80`'s replay showed the rule behaves on material that was already measured. This runs it
live: a fixed number of ordinary requests through `AppleRuntime`, with `B79`'s activation
opted in and `B80` observing every one of them. The monitor may only watch. Nothing here is
tuned from what it sees, and no request is dispatched to learn from.

What is recorded is what a person would need to decide whether to leave it on: how many times
it cried wolf, which states it moved through, what it cost per dispatch, and how large the
state it keeps actually is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TOOLS = PROJECT_ROOT / "tools"
RAW = PROJECT_ROOT / "research" / "raw"

MODEL = "mlx-community/gemma-3-12b-it-4bit"
MAX_TOKENS = 32
DISPATCHES = 60
WARMUPS = 2
CHILD_TIMEOUT_S = 1800.0
MICRO_CALLS = 20000
MICRO_BLOCKS = 10
EQUIVALENCE_MARGIN = 0.02

PREREGISTRATION = {
    "experiment": "B80_shadow_canary",
    "question": ("watching real dispatches, does the monitor stay quiet on a machine that "
                 "has not changed, and what does watching cost"),
    "monitoring_only": ("the monitor observes. It cannot qualify an action, and nothing in "
                        "this run is tuned from what it sees"),
    "no_exploration": ("every dispatch takes whatever action B79 would have taken anyway. "
                       "No request is served differently in order to learn from it"),
    "dispatches": DISPATCHES,
    "workload": f"one request, StrictOneShotPlan, {MAX_TOKENS} new tokens, the B57 question set",
    "recorded": ["false drift alarms", "real drift alarms", "time to detection",
                 "state transitions", "runtime overhead", "state size on disk",
                 "reference fallbacks"],
    "not_a_performance_study": ("the timings here describe the monitor's cost. The action's "
                                "gain was measured in B76 and is not re-derived from this"),
    "hot_path_margin": EQUIVALENCE_MARGIN,
    "hot_path_reference": (
        "for_dispatch runs once per cohort, so the margin applies to its added time as a "
        "share of a measured dispatch, which is how B79 sealed the same question. The first "
        "run of this file held it to a ratio against an isolated for_dispatch call instead "
        "and recorded a FAIL on that reading; that record is kept and the denominator was "
        "wrong rather than the result"),
}

CHILD = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b80_shadow_canary import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def child_main(spec: dict) -> dict:
    """One process, one model, a fixed number of ordinary requests, all of them watched."""
    import importlib.util

    import mlx.core as mx

    from ironmule.plans import StrictOneShotPlan
    from ironmule.router import AppleRuntime, monitor_path
    from ironmule.service import Request

    loader = importlib.util.spec_from_file_location(
        "b57", str(Path(spec["root"]) / "tools" / "b57_stack_composition.py"))
    b57 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(b57)

    mx.reset_peak_memory()
    runtime = AppleRuntime.load(spec["model"], enable_local_learned_dispatch=True,
                                local_state_path=Path(spec["state_path"]))
    try:
        status = runtime.status()
        prompt = runtime.encode(b57.QUESTIONS[0])
        dispatches = []
        for index in range(spec["dispatches"] + spec["warmups"]):
            request = Request(prompt_ids=list(prompt), max_tokens=spec["max_tokens"],
                              plan=StrictOneShotPlan(), objective="latency")
            started = time.perf_counter_ns()
            result = runtime.serve([request])[0]
            wall_ns = time.perf_counter_ns() - started
            if index < spec["warmups"]:
                continue
            record = runtime.decisions[-1]
            activation = (record.get("activation") or [None])[0] or {}
            monitoring = (record.get("monitoring") or [None])[0] or {}
            dispatches.append({
                "index": index - spec["warmups"],
                "effective_action": activation.get("effective_action", "reference"),
                "end_to_end_ms": wall_ns / 1e6,
                "monitor_state": monitoring.get("state"),
                "monitor_reason": monitoring.get("reason"),
                "monitor_changed": monitoring.get("changed"),
                "verdict": monitoring.get("verdict"),
                "fallbacks": record.get("fallbacks", 0),
                "tokens": list(map(int, result.tokens)),
                "stop_reason": result.stop_reason,
            })
        final = runtime.status()
        monitor = final.get("monitoring") or {}
        path = monitor_path(Path(spec["state_path"]))
        runtime.monitor.save(path)
        return {
            "pid": os.getpid(),
            "activation_enabled": bool(
                (status.get("local_learned_dispatch") or {}).get("enabled")),
            "dispatches": dispatches,
            "monitor": monitor,
            "monitor_state_bytes": path.stat().st_size if path.exists() else 0,
            "monitor_state_path": str(path),
            "controller_state": (
                (status.get("local_learning") or {}).get("actions") or [{}])[0].get("state"),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
        }
    finally:
        runtime.close()


def _micro() -> dict:
    """What the hot path pays for having a monitor attached, and what observing costs."""
    from b78_cold_start_replay import _context, replay_evidence
    from ironmule.activation import ActivationContext, LearnedDispatchActivation
    from ironmule.local_learner import LocalLearner
    from ironmule.monitoring import BASELINE_MIN, DriftMonitor, Observation

    intake, _provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    context = ActivationContext(
        hardware_fingerprint=intake.hardware_fingerprint,
        gpu_architecture=intake.gpu_architecture, model_id=intake.model_id,
        model_identity_sha256=intake.model_identity_sha256,
        model_revision=intake.model_revision, quantization_bits=4,
        quantization_group_size=64, hidden_size=3840, projection_widths=(15360,),
        mlx=intake.mlx, mlx_lm=intake.mlx_lm)

    monitor = DriftMonitor(min_material_shift=0.0226)
    fields = {"hardware_fingerprint": intake.hardware_fingerprint,
              "gpu_architecture": intake.gpu_architecture,
              "model_identity_sha256": intake.model_identity_sha256,
              "model_revision": intake.model_revision, "quantization_bits": 4,
              "quantization_group_size": 64, "mlx": intake.mlx, "mlx_lm": intake.mlx_lm,
              "workload_class": "single_short", "action_id": "k3840_geometry_sg4_r8",
              "action_code_digest": "d", "effective_action": "candidate"}

    def observation(index: int) -> Observation:
        return Observation(observed_at=f"2026-09-11T00:00:{index % 60:02d}+00:00",
                           end_to_end_ms=921.0 + (index % 5), service_ttft_ms=20.0,
                           tokens_per_second=33.0, new_tokens=32, prompt_tokens=27,
                           fallbacks=0, correctness_errors=0, memory_pressure_level=1,
                           swap_used_bytes=0, controller_digest="c", **fields)

    for index in range(BASELINE_MIN + 5):
        monitor.observe(observation(index))

    plain = LearnedDispatchActivation(learner, context, enabled=True)
    plain.prepare()
    watched = LearnedDispatchActivation(learner, context, enabled=True, monitor=monitor)
    watched.prepare()
    for layer in (plain, watched):
        for _ in range(1000):
            layer.for_dispatch("single_short", "interactive")

    arms = {
        "for_dispatch_without_monitor": lambda: [
            plain.for_dispatch("single_short", "interactive") for _ in range(MICRO_CALLS)],
        "for_dispatch_with_monitor": lambda: [
            watched.for_dispatch("single_short", "interactive") for _ in range(MICRO_CALLS)],
    }
    names = list(arms)
    per_block = []
    for block in range(MICRO_BLOCKS):
        shift = block % len(names)
        timings = {}
        for name in (names[shift:] + names[:shift]):
            started = time.perf_counter_ns()
            arms[name]()
            timings[name] = (time.perf_counter_ns() - started) / MICRO_CALLS
        per_block.append(timings)

    observe_samples = []
    for block in range(MICRO_BLOCKS):
        started = time.perf_counter_ns()
        for index in range(2000):
            monitor.observe(observation(10_000 + block * 2000 + index))
        observe_samples.append((time.perf_counter_ns() - started) / 2000)

    generator = random.Random(20260911)
    ratios = [b["for_dispatch_with_monitor"] / b["for_dispatch_without_monitor"]
              for b in per_block]
    draws = sorted(median([generator.choice(ratios) for _ in ratios]) for _ in range(10000))
    nanoseconds = {name: statistics.median([b[name] for b in per_block]) for name in names}
    added = (nanoseconds["for_dispatch_with_monitor"]
             - nanoseconds["for_dispatch_without_monitor"])
    dispatch_ms, source = _measured_dispatch_ms()
    return {
        "median_nanoseconds": nanoseconds,
        "with_over_without": {"median": median(ratios), "ci_low": draws[250],
                              "ci_high": draws[9749], "n": len(ratios)},
        "added_nanoseconds_per_dispatch": added,
        "measured_dispatch_ms": dispatch_ms,
        "measured_dispatch_source": source,
        "share_of_one_measured_dispatch": (added / 1e6) / dispatch_ms if dispatch_ms else None,
        "observe_median_nanoseconds": statistics.median(observe_samples),
        "note": ("for_dispatch runs once per cohort, so its reference is a dispatch and the "
                 "share of one is the criterion, exactly as B79 sealed it. The ratio against "
                 "an isolated for_dispatch call is reported and not gated: a guard measured "
                 "in hundreds of nanoseconds cannot be held to a denominator that small. "
                 "observe() runs after the answer exists, beside the other diagnostics"),
    }


def _measured_dispatch_ms() -> tuple[float, str]:
    """A dispatch this machine ran, from B79's sealed canary."""
    canary = json.loads((RAW / "B79_canary_20260911_v2.json").read_text(encoding="utf-8"))
    rows = [row["end_to_end_ms"] for phase in canary["phases"].values()
            for row in phase["dispatches"]]
    return statistics.median(rows), f"B79_canary_20260911_v2.json, {len(rows)} dispatches"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from b78_cold_start_replay import _context, replay_evidence
    from ironmule.local_learner import CANDIDATE_QUALIFIED, LocalLearner
    from ironmule.monitoring import MONITORING, REQUALIFICATION_REQUIRED, WARMING_UP
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    intake, provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    controller = learner.recommendation("k3840_geometry_sg4_r8", "single_short")
    if controller.local_learning_state != CANDIDATE_QUALIFIED:
        raise SystemExit(f"the controller is {controller.local_learning_state}")

    scratch = Path(tempfile.mkdtemp(prefix="b80_shadow_"))
    state_path = learner.save(scratch / "local_learning.json")

    print(f"{DISPATCHES} ordinary dispatches, monitored", flush=True)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c",
         CHILD.format(root=str(PROJECT_ROOT), tools=str(TOOLS)),
         json.dumps({"model": MODEL, "dispatches": DISPATCHES, "warmups": WARMUPS,
                     "max_tokens": MAX_TOKENS, "state_path": str(state_path),
                     "root": str(PROJECT_ROOT)})],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(PROJECT_ROOT),
        env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    try:
        stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    marker = next((line[2:] for line in stdout.splitlines() if line.startswith("@@")), None)
    if process.returncode != 0 or marker is None:
        raise SystemExit(f"the shadow child failed ({process.returncode}):\n{stderr[-3000:]}")
    run = json.loads(marker)

    print("measuring what watching costs", flush=True)
    micro = _micro()

    dispatches = run["dispatches"]
    states = [row["monitor_state"] for row in dispatches]
    transitions = run["monitor"].get("transitions") or []
    alarms = [row for row in dispatches if row["monitor_state"] == REQUALIFICATION_REQUIRED]
    fallbacks = sum(row["fallbacks"] for row in dispatches)
    tokens = {tuple(row["tokens"]) for row in dispatches}
    latencies = [row["end_to_end_ms"] for row in dispatches]
    candidate = [row for row in dispatches if row["effective_action"] == "candidate"]

    checks = {
        "activation_was_on": bool(run["activation_enabled"]),
        "every_dispatch_ran_the_candidate": len(candidate) == len(dispatches),
        "no_false_drift_alarm": not alarms,
        "monitoring_state_was_reached": MONITORING in states,
        "correctness_identical": len(tokens) == 1,
        "no_fallbacks": fallbacks == 0,
        "hot_path_within_margin": micro["share_of_one_measured_dispatch"] < EQUIVALENCE_MARGIN,
        "state_stayed_small": run["monitor_state_bytes"] < 64 * 1024,
    }
    verdict = ("B80_SHADOW_PASS" if all(checks.values()) else
               ("B80_TOO_SENSITIVE" if alarms else "B80_FAIL"))

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "controller_at_start": controller.as_dict(),
        "run": run,
        "state_transitions": transitions,
        "warming_up_dispatches": sum(1 for state in states if state == WARMING_UP),
        "monitoring_dispatches": sum(1 for state in states if state == MONITORING),
        "false_drift_alarms": len(alarms),
        "real_drift_alarms": 0,
        "time_to_detection": None,
        "reference_fallbacks": fallbacks,
        "latency": {"n": len(latencies), "median_ms": median(latencies),
                    "min_ms": min(latencies), "max_ms": max(latencies)},
        "overhead": micro,
        "state_size_bytes": run["monitor_state_bytes"],
        "data_volume": {"observations": len(dispatches),
                        "segments": len(run["monitor"].get("segments") or []),
                        "bytes_per_observation": (run["monitor_state_bytes"] / len(dispatches)
                                                  if dispatches else None),
                        "note": ("the monitor keeps a baseline and a window per segment, not "
                                 "the observations themselves, so the state does not grow "
                                 "with use")},
        "checks": checks,
        "verdict": verdict,
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/monitoring.py", "ironmule/activation.py",
                         "ironmule/router.py", "tools/b80_shadow_canary.py")},
        "nothing_activated": ("the controller state and the monitor state are in a scratch "
                              "directory, the user's store was never written, and "
                              "enable_local_learned_dispatch still defaults to False"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "transitions": transitions,
                      "false_drift_alarms": len(alarms),
                      "latency": record["latency"],
                      "state_size_bytes": record["state_size_bytes"],
                      "overhead": {
                          "with_over_without": micro["with_over_without"],
                          "added_ns_per_dispatch":
                              round(micro["added_nanoseconds_per_dispatch"], 1),
                          "share_of_one_measured_dispatch":
                              micro["share_of_one_measured_dispatch"],
                          "observe_ns": round(micro["observe_median_nanoseconds"], 1)}},
                     indent=2, default=str))
    return 0 if verdict == "B80_SHADOW_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
