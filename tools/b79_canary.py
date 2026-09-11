#!/usr/bin/env python3
"""The first real dispatches this project has ever let a learned preference change.

Everything up to here was allowed to be wrong without consequence: `B78`'s controller
annotated a decision nobody read, and `B79`'s preflight exercised a decision path with nothing
installed behind it. This runs the candidate on the real `12B`, through `AppleRuntime`, with
`enable_local_learned_dispatch=True`, on a fixed number of realistic `single_short` requests
chosen before the run and never after it.

**The shape is fixed in advance.** A reference control, then the canary, then the reference
control again, each in its own process with its own model image. The controls exist to catch a
gross regression that the canary alone could not see, and they are the same requests.

**Correctness is identity, not a summary.** Every dispatch in every phase asks the same
question and must return the same token ids and the same stop reason. One difference anywhere
is a `B79_ACTIVATION_FAIL`, not a note.

**The kill switch is exercised, not assumed.** After the canary a kill record is written and a
further process is started with activation opted in. It must come up disabled.

**Nothing is left switched on.** The controller state lives in a scratch directory for the
duration, the user's own store is never written, and the default stays `False`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TOOLS = PROJECT_ROOT / "tools"
RAW = PROJECT_ROOT / "research" / "raw"

MODEL = "mlx-community/gemma-3-12b-it-4bit"
MAX_TOKENS = 32
CONTROL_DISPATCHES = 5
CANARY_DISPATCHES = 30
CHILD_TIMEOUT_S = 1800.0

PREREGISTRATION = {
    "experiment": "B79_canary_learned_dispatch",
    "question": ("does the locally qualified candidate run correctly and without a fallback "
                 "on real dispatches, and does every safety path behave as designed"),
    "opt_in": ("enable_local_learned_dispatch=True is passed explicitly. The default is "
               "False and is not changed"),
    "phases": [
        {"name": "control_before", "dispatches": CONTROL_DISPATCHES, "activation": False},
        {"name": "canary", "dispatches": CANARY_DISPATCHES, "activation": True},
        {"name": "control_after", "dispatches": CONTROL_DISPATCHES, "activation": False},
        {"name": "after_kill", "dispatches": CONTROL_DISPATCHES, "activation": True,
         "note": "a kill record is written first; this process must come up disabled"},
    ],
    "workload": (f"one request, StrictOneShotPlan, {MAX_TOKENS} new tokens, the B57 question "
                 f"set. That is the single_short class B76 qualified and no other"),
    "no_exploration": ("every dispatch inside a phase uses that phase's action. There is no "
                       "A/B alternation in the user path and no exploration of any kind"),
    "fixed_before_the_run": ("the dispatch counts, the phase order and the question set are "
                             "fixed here. Nothing is selected after a result is seen"),
    "correctness": ("token ids and stop reasons must be identical across every dispatch of "
                    "every phase. One difference is a failure of the whole study"),
    "kill_criteria": ["any correctness difference", "any fallback in the candidate path",
                      "two runtime errors", "a controller or state inconsistency",
                      "an interval that stops clearing 1.0"],
    "no_threshold_changes": "no gate or threshold is altered during B79",
    "state_is_scratch": ("the controller state is written to a temporary directory. The "
                         "user's own store is never written"),
}

CHILD = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b79_canary import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def child_main(spec: dict) -> dict:
    """One phase, one process, one model image, one action throughout."""
    import importlib.util
    import time

    import mlx.core as mx

    from ironmule.hw import memory_pressure_level, swap_used_bytes, vm_counters
    from ironmule.plans import StrictOneShotPlan
    from ironmule.router import AppleRuntime
    from ironmule.service import Request

    loader = importlib.util.spec_from_file_location(
        "b57", str(Path(spec["root"]) / "tools" / "b57_stack_composition.py"))
    b57 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(b57)

    def probe(label):
        counters = vm_counters() or {}
        return {"label": label, "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "mlx_peak_bytes": int(mx.get_peak_memory()),
                "swapouts": counters.get("swapouts")}

    mx.reset_peak_memory()
    samples = [probe("start")]
    runtime = AppleRuntime.load(spec["model"],
                                enable_local_learned_dispatch=spec["activation"],
                                local_state_path=Path(spec["state_path"]))
    try:
        status = runtime.status()
        prompt = runtime.encode(b57.QUESTIONS[0])
        dispatches = []
        errors = []
        for index in range(spec["dispatches"] + spec["warmups"]):
            request = Request(prompt_ids=list(prompt), max_tokens=spec["max_tokens"],
                              plan=StrictOneShotPlan(), objective="latency")
            started = time.perf_counter_ns()
            try:
                result = runtime.serve([request])[0]
            except Exception as error:  # noqa: BLE001 - recorded, never swallowed silently
                errors.append({"index": index, "error": f"{type(error).__name__}: {error}"})
                continue
            wall_ns = time.perf_counter_ns() - started
            if index < spec["warmups"]:
                continue
            record = runtime.decisions[-1]
            snapshot = runtime.telemetry.snapshot()
            activation = (record.get("activation") or [None])[0] or {}
            dispatches.append({
                "index": index - spec["warmups"],
                "effective_action": activation.get("effective_action", "reference"),
                "base_action": activation.get("base_action", "reference"),
                "local_learning_state": activation.get("local_learning_state"),
                "evidence_count": activation.get("evidence_count"),
                "controller_digest": activation.get("controller_digest"),
                "activation_reason": activation.get("activation_reason"),
                "route": record.get("route"),
                "workload_class": activation.get("workload_class"),
                "end_to_end_ms": wall_ns / 1e6,
                "service_ttft_p50_ms": snapshot["service_ttft_p50_ms"],
                "engine_ttft_p50_ms": snapshot["engine_ttft_p50_ms"],
                "aggregate_tokens_per_second": snapshot["aggregate_tokens_per_second"],
                "fallbacks": snapshot["fallbacks"],
                "fallback_reasons": list(snapshot.get("fallback_reasons") or [])[:5],
                "correctness_errors": snapshot.get("correctness_errors", 0),
                "tokens": list(map(int, result.tokens)),
                "stop_reason": result.stop_reason,
            })
        samples.append(probe("end"))
        activation_status = status.get("local_learned_dispatch") or {}
        return {
            "phase": spec["phase"], "pid": os.getpid(),
            "activation_requested": spec["activation"],
            "activation_enabled": bool(activation_status.get("enabled")),
            "activation_disabled_reason": activation_status.get("disabled_reason"),
            "activation_installed": activation_status.get("installed"),
            "activation_status_after": (runtime.status().get("local_learned_dispatch") or {}),
            "controller_state": (
                (status.get("local_learning") or {}).get("actions") or [{}])[0].get("state"),
            "dispatches": dispatches, "errors": errors,
            "resource_samples": samples,
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    finally:
        runtime.close()


def _run_child(spec: dict) -> dict:
    process = subprocess.Popen(
        [sys.executable, "-u", "-c",
         CHILD.format(root=str(PROJECT_ROOT), tools=str(TOOLS)), json.dumps(spec)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(PROJECT_ROOT),
        env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    try:
        stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    marker = next((line[2:] for line in stdout.splitlines() if line.startswith("@@")), None)
    if process.returncode != 0 or marker is None:
        return {"phase": spec["phase"], "failed": True, "returncode": process.returncode,
                "stderr_tail": stderr[-4000:], "dispatches": [], "errors": []}
    return json.loads(marker)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from b78_cold_start_replay import _context, replay_evidence
    from ironmule.activation import kill_path, write_kill
    from ironmule.local_learner import CANDIDATE_QUALIFIED, LocalLearner

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION,
             "written_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    intake, provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    controller = learner.recommendation("k3840_geometry_sg4_r8", "single_short")
    if controller.local_learning_state != CANDIDATE_QUALIFIED:
        raise SystemExit(f"the controller is {controller.local_learning_state}; a canary "
                         f"without a qualified action would measure nothing")

    scratch = Path(tempfile.mkdtemp(prefix="b79_canary_"))
    state_path = learner.save(scratch / "local_learning.json")

    def spec(phase: str, activation: bool, dispatches: int) -> dict:
        return {"phase": phase, "activation": activation, "dispatches": dispatches,
                "warmups": 2, "model": MODEL, "max_tokens": MAX_TOKENS,
                "state_path": str(state_path), "root": str(PROJECT_ROOT)}

    phases = {}
    for name, activation, count in (("control_before", False, CONTROL_DISPATCHES),
                                    ("canary", True, CANARY_DISPATCHES),
                                    ("control_after", False, CONTROL_DISPATCHES)):
        print(f"{name}: {count} dispatches, activation={activation}", flush=True)
        phases[name] = _run_child(spec(name, activation, count))
        print(f"  {name} done", flush=True)

    # The kill switch, exercised across a process boundary rather than asserted.
    kill_record = write_kill(state_path, "B79 canary: the kill switch is exercised here "
                                         "deliberately, after the canary completed")
    print("after_kill: a kill record is written; the next process must come up disabled",
          flush=True)
    phases["after_kill"] = _run_child(spec("after_kill", True, CONTROL_DISPATCHES))

    # -- checks ---------------------------------------------------------------
    everything = [row for phase in phases.values() for row in phase["dispatches"]]
    reference_tokens = next((row["tokens"] for row in everything), None)
    reference_stop = next((row["stop_reason"] for row in everything), None)
    differences = [{"phase": name, "index": row["index"]}
                   for name, phase in phases.items() for row in phase["dispatches"]
                   if row["tokens"] != reference_tokens or row["stop_reason"] != reference_stop]
    candidate_rows = [row for row in phases["canary"]["dispatches"]
                      if row["effective_action"] == "candidate"]
    fallbacks = sum(row["fallbacks"] for row in everything)
    runtime_errors = sum(len(phase["errors"]) for phase in phases.values())
    failed_children = [name for name, phase in phases.items() if phase.get("failed")]

    swaps = [s["swap_used_bytes"] for phase in phases.values()
             for s in phase.get("resource_samples", []) if s["swap_used_bytes"] is not None]
    levels = [s["memory_pressure_level"] for phase in phases.values()
              for s in phase.get("resource_samples", [])]
    peak_rss = max((phase.get("peak_rss_bytes", 0) for phase in phases.values()), default=0)
    gate_reasons = []
    if not swaps or max(swaps) > swaps[0]:
        gate_reasons.append("swap in use rose above its value at the start")
    if not all(level == 1 for level in levels):
        gate_reasons.append("macOS memory pressure was not normal at every probe")

    def latencies(name):
        return [row["end_to_end_ms"] for row in phases[name]["dispatches"]]

    timing = {name: {"n": len(latencies(name)),
                     "median_ms": statistics.median(latencies(name)) if latencies(name) else None,
                     "min_ms": min(latencies(name)) if latencies(name) else None,
                     "max_ms": max(latencies(name)) if latencies(name) else None}
              for name in phases}
    controls = latencies("control_before") + latencies("control_after")
    regression = None
    if controls and latencies("canary"):
        regression = statistics.median(latencies("canary")) / statistics.median(controls)

    checks = {
        "no_child_failed": not failed_children,
        "correctness_identical_everywhere": not differences,
        "no_fallbacks": fallbacks == 0,
        "fewer_than_two_runtime_errors": runtime_errors < 2,
        "resource_gate_passed": not gate_reasons,
        "canary_actually_ran_the_candidate": len(candidate_rows) == CANARY_DISPATCHES,
        "controls_stayed_on_the_reference": all(
            row["effective_action"] == "reference"
            for name in ("control_before", "control_after")
            for row in phases[name]["dispatches"]),
        "opting_out_never_enabled_activation": all(
            phases[name]["activation_enabled"] is False
            for name in ("control_before", "control_after")),
        "the_kill_record_disabled_a_later_process": (
            phases["after_kill"]["activation_enabled"] is False
            and all(row["effective_action"] == "reference"
                    for row in phases["after_kill"]["dispatches"])),
        "no_kill_fired_during_the_canary": not (
            (phases["canary"].get("activation_status_after") or {}).get("events")),
    }
    passed = all(checks.values())
    verdict = ("B79_CANARY_PASS" if passed else
               ("B79_ACTIVATION_FAIL" if (differences or fallbacks or failed_children)
                else "B79_SAFE_FALLBACK"))

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "controller_at_start": controller.as_dict(),
        "state_path": str(state_path),
        "kill_record": str(kill_record),
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/activation.py", "ironmule/qmv_variant.py",
                         "ironmule/local_learner.py", "ironmule/router.py",
                         "tools/b79_canary.py")},
        "phases": phases,
        "timing": timing,
        "canary_over_controls": regression,
        "correctness_differences": differences,
        "fallbacks": fallbacks,
        "runtime_errors": runtime_errors,
        "resource_gate": {"passed": not gate_reasons, "reasons": gate_reasons,
                          "peak_rss_bytes": peak_rss},
        "checks": checks,
        "verdict": verdict,
        "nothing_left_on": ("the state and the kill record are in a scratch directory, the "
                            "user's store was never written, and the default stays False"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks, "timing": timing,
                      "canary_over_controls": regression,
                      "candidate_dispatches": len(candidate_rows),
                      "fallbacks": fallbacks, "runtime_errors": runtime_errors},
                     indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
