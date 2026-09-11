#!/usr/bin/env python3
"""The whole recovery, end to end, on this machine, with a real comparison in the middle.

`B80` proved a monitor can take a qualified action away. `B81` has to prove the only way back
actually works, and that every step of it survives a restart. So the sequence is run for real:
a qualified controller, a drift state created *only* through the test harness, a fresh runtime
that must serve the reference, a requalification that a person starts, a genuine comparative
measurement with fresh processes and an A/A control, and a final fresh runtime to see which
action is now in charge.

**Nothing in the product path is slowed down to make drift happen.** The drift state is
written by feeding the monitor a preregistered observation sequence, which is the harness's
job. No dispatch is made worse and no timing is faked.

**No performance number here is invented.** Step six is the real thing: twenty-seven children,
each loading one model and running one arm, decided by its own intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TOOLS = PROJECT_ROOT / "tools"
RAW = PROJECT_ROOT / "research" / "raw"

from ironmule.monitoring import (BASELINE_MIN, DriftMonitor, Observation,
                                 REQUALIFICATION_REQUIRED, WINDOW, requalification_path)
from ironmule.qmv_variant import QUALIFIED_ACTION_ID
from ironmule.requalification import BLOCKS, SESSIONS, lineage_path

MODEL = "mlx-community/gemma-3-12b-it-4bit"
CLASS = "single_short"
MAX_TOKENS = 32
VERIFY_DISPATCHES = 3
CHILD_TIMEOUT_S = 3600.0

#: Fixed before the run. The shift is the one B80 preregistered as the case worth catching,
#: sustained for a full window, and the observations that follow it are deliberately
#: candidate-friendly so that "more good news does not clear it" is actually tested.
DRIFT_SHIFT = 1.25
FRIENDLY_AFTER_DRIFT = 50
FRIENDLY_SHIFT = 0.85

PREREGISTRATION = {
    "experiment": "B81_live_recovery",
    "question": ("does the only permitted way out of REQUALIFICATION_REQUIRED work end to "
                 "end on this machine, and does every step survive a restart"),
    "steps": [
        "load a valid controller state qualified from B75 and B76",
        "create the drift state through the harness only, never by slowing a product path",
        "restart the runtime and confirm the reference serves",
        "feed further candidate-friendly observations and confirm the state does not clear",
        "start an explicit requalification",
        "measure the candidate against the reference for real",
        "restart the runtime again",
        "record which action is now effective",
    ],
    "drift_sequence": {"shift": DRIFT_SHIFT, "window": WINDOW,
                       "why": "the case B80 preregistered as worth catching"},
    "friendly_after_drift": {"observations": FRIENDLY_AFTER_DRIFT, "shift": FRIENDLY_SHIFT,
                             "why": ("good news must not clear a requalification. Only "
                                     "comparative evidence may")},
    "comparison": {"sessions": SESSIONS, "blocks_per_session": BLOCKS,
                   "children": SESSIONS * BLOCKS * 3,
                   "no_invented_numbers": "the measurement is real and decides itself"},
    "default_stays_off": "enable_local_learned_dispatch is passed explicitly and defaults to False",
    "proves_the_mechanism_not_the_frequency": (
        "B81 shows recovery works. It says nothing about how often real drift happens"),
}

CHILD = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b81_recovery import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


def child_main(spec: dict) -> dict:
    """A fresh runtime, a few ordinary requests, and what action they actually took."""
    from ironmule.plans import StrictOneShotPlan
    from ironmule.router import AppleRuntime
    from ironmule.service import Request

    runtime = AppleRuntime.load(spec["model"], enable_local_learned_dispatch=True,
                                local_state_path=Path(spec["state_path"]))
    try:
        status = runtime.status()
        prompt = runtime.encode(spec["prompt"])
        actions, tokens = [], []
        for _ in range(spec["dispatches"]):
            result = runtime.serve([Request(prompt_ids=list(prompt),
                                            max_tokens=spec["max_tokens"],
                                            plan=StrictOneShotPlan(),
                                            objective="latency")])[0]
            record = runtime.decisions[-1]
            activation = (record.get("activation") or [None])[0] or {}
            actions.append(activation.get("effective_action", "reference"))
            tokens.append(list(map(int, result.tokens)))
        activation_status = status.get("local_learned_dispatch") or {}
        controller = ((status.get("local_learning") or {}).get("actions") or [{}])[0]
        return {
            "phase": spec["phase"],
            "effective_actions": actions,
            "unique_actions": sorted(set(actions)),
            "tokens_identical": len({tuple(row) for row in tokens}) == 1,
            "tokens": tokens[0] if tokens else [],
            "activation_enabled": bool(activation_status.get("enabled")),
            "activation_disabled_reason": activation_status.get("disabled_reason"),
            "requalification_required": activation_status.get("requalification_required"),
            "controller_state": controller.get("state"),
            "controller_evidence_ids": controller.get("evidence_ids", [])[:4],
            "monitor_state": ((status.get("monitoring") or {}).get("segments") or [{}])[0]
                .get("state"),
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
                "stderr_tail": stderr[-3000:], "effective_actions": []}
    return json.loads(marker)


def _inject_drift(state_path: Path, context, digest: str, latencies: list[float]) -> dict:
    """The harness's job, and only the harness's: write a drifted monitor state."""
    from ironmule.router import monitor_path

    floor = 0.0226
    monitor = DriftMonitor(state_path, min_material_shift=floor)
    fields = {"hardware_fingerprint": context.hardware_fingerprint,
              "gpu_architecture": context.gpu_architecture,
              "model_identity_sha256": context.model_identity_sha256,
              "model_revision": context.model_revision, "quantization_bits": 4,
              "quantization_group_size": 64, "mlx": context.mlx, "mlx_lm": context.mlx_lm,
              "workload_class": CLASS, "action_id": QUALIFIED_ACTION_ID,
              "action_code_digest": digest, "effective_action": "candidate"}

    def observation(index: int, milliseconds: float) -> Observation:
        moment = datetime(2026, 9, 11, tzinfo=timezone.utc) + timedelta(seconds=index)
        return Observation(observed_at=moment.isoformat(), end_to_end_ms=milliseconds,
                           service_ttft_ms=None, tokens_per_second=None, new_tokens=32,
                           prompt_tokens=27, fallbacks=0, correctness_errors=0,
                           memory_pressure_level=1, swap_used_bytes=0,
                           controller_digest="harness", **fields)

    centre = statistics.median(latencies)
    for index in range(BASELINE_MIN + 5):
        monitor.observe(observation(index, latencies[index % len(latencies)]))
    baseline_state = monitor.state_for(CLASS, QUALIFIED_ACTION_ID)
    for index in range(WINDOW):
        monitor.observe(observation(100 + index, centre * DRIFT_SHIFT))
    drifted = monitor.requalification_required
    friendly_kept = None
    for index in range(FRIENDLY_AFTER_DRIFT):
        monitor.observe(observation(200 + index, centre * FRIENDLY_SHIFT))
    friendly_kept = monitor.requalification_required
    monitor.save(monitor_path(state_path))
    return {"baseline_state": baseline_state, "drift_detected": drifted,
            "state_after_friendly_observations": friendly_kept,
            "requalification_record": str(requalification_path(state_path)),
            "record_exists": requalification_path(state_path).exists(),
            "transitions": list(monitor.transitions),
            "observations_injected": BASELINE_MIN + 5 + WINDOW + FRIENDLY_AFTER_DRIFT}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION,
             "written_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    from b78_cold_start_replay import _context, replay_evidence
    from ironmule.local_learner import CANDIDATE_QUALIFIED, LocalLearner
    from ironmule.monitoring import default_action_code_digest
    from ironmule.requalification import PROBE_PROMPT, requalify
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    intake, provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    if learner.recommendation(QUALIFIED_ACTION_ID, CLASS).local_learning_state \
            != CANDIDATE_QUALIFIED:
        raise SystemExit("the controller is not qualified; there is nothing to recover")

    scratch = Path(tempfile.mkdtemp(prefix="b81_recovery_"))
    state_path = learner.save(scratch / "local_learning.json")
    original_state = state_path.read_bytes()

    def spec(phase: str) -> dict:
        return {"phase": phase, "model": MODEL, "dispatches": VERIFY_DISPATCHES,
                "max_tokens": MAX_TOKENS, "prompt": PROBE_PROMPT,
                "state_path": str(state_path), "root": str(PROJECT_ROOT)}

    print("1-2. a qualified controller, and a runtime that should use it", flush=True)
    before = _run_child(spec("before_drift"))

    print("3. injecting the preregistered drift sequence through the harness", flush=True)
    canary = json.loads((RAW / "B79_canary_20260911_v2.json").read_text())
    latencies = [row["end_to_end_ms"] for row in canary["phases"]["canary"]["dispatches"]]
    injection = _inject_drift(state_path, intake, default_action_code_digest(), latencies)

    print("4. restarting the runtime; the reference must serve", flush=True)
    after_drift = _run_child(spec("after_drift"))

    print("5-6. the explicit requalification, with a real comparison", flush=True)

    def progress(session, block, arm):
        print(f"    session {session} block {block} {arm} done", flush=True)

    run = requalify(MODEL, state_path=state_path, on_progress=progress)

    print("7-8. restarting the runtime; which action is in charge now", flush=True)
    after_requalification = _run_child(spec("after_requalification"))

    lineage = json.loads(lineage_path(state_path).read_text())
    archived = sorted(str(p.name) for p in scratch.glob("*.epoch.json"))

    outcome = run["outcome"]
    expected_after = "candidate" if outcome == "PASS" else "reference"
    checks = {
        "no_child_failed": not any(phase.get("failed") for phase in
                                   (before, after_drift, after_requalification)),
        "the_candidate_ran_before_the_drift": before.get("unique_actions") == ["candidate"],
        "the_harness_produced_the_drift": bool(injection["drift_detected"]),
        "good_news_did_not_clear_it": bool(injection["state_after_friendly_observations"]),
        "the_state_survived_a_restart": bool(after_drift.get("requalification_required")),
        "the_reference_served_after_the_drift":
            after_drift.get("unique_actions") == ["reference"],
        "the_requalification_started_only_when_required": run["outcome"] != "REFUSED",
        "the_run_decided_on_its_own_numbers": run.get("decision", {}).get("reason", "") != "",
        "the_action_after_matches_the_outcome":
            after_requalification.get("unique_actions") == [expected_after],
        "the_old_epoch_was_archived": bool(archived),
        "the_lineage_reads_as_a_story": len(lineage["events"]) >= 2,
        "correctness_held_throughout": all(
            phase.get("tokens_identical", False) for phase in
            (before, after_drift, after_requalification)),
    }
    passed = all(checks.values())
    if not passed:
        verdict = "B81_FAIL"
    elif outcome == "PASS":
        verdict = "B81_REQUALIFICATION_PASS"
    else:
        verdict = "B81_SAFE_REFERENCE"

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "scratch": str(scratch),
        "state_path": str(state_path),
        "phases": {"before_drift": before, "injection": injection,
                   "after_drift": after_drift,
                   "after_requalification": after_requalification},
        "requalification": run,
        "lineage": lineage,
        "archived_epochs": archived,
        "checks": checks,
        "outcome": outcome,
        "verdict": verdict,
        "cost": run.get("cost"),
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/requalification.py", "ironmule/monitoring.py",
                         "ironmule/activation.py", "ironmule/router.py",
                         "tools/b81_recovery.py")},
        "original_state_sha256": hashlib.sha256(original_state).hexdigest(),
        "nothing_activated": ("everything is in a scratch directory, the user's store was "
                              "never written, and enable_local_learned_dispatch still "
                              "defaults to False"),
        "what_this_does_not_show": ("how often real drift happens. B81 proves the recovery "
                                    "mechanism and nothing about its frequency"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "outcome": outcome, "checks": checks,
                      "actions": {name: phase.get("unique_actions") for name, phase in
                                  (("before_drift", before), ("after_drift", after_drift),
                                   ("after_requalification", after_requalification))},
                      "session_ratios": [row["candidate"]["median"] for row in
                                         run.get("comparison", {}).get("sessions", [])],
                      "cost": run.get("cost"),
                      "lineage": [event["event"] for event in lineage["events"]]},
                     indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
