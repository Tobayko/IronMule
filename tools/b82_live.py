#!/usr/bin/env python3
"""One attempt at the live pass `B81` could not reach, with a cheap check in front of it.

`B81` spent twenty-six minutes on two comparative requalifications and threw both away,
correctly: the A/A controls ran at half widths of `0.20` and `0.23`, and the candidate looked up
to twenty per cent faster precisely because the reference arm was being disturbed. The gate
worked. It just worked after the expensive part.

`B82` puts a reference-against-reference probe in front. If the machine's own control cannot be
read now, nothing expensive starts and `B82_NOT_READY` is the answer -- a result, not a failure.
If it can be read, exactly one full `B81` requalification runs, unchanged, and whatever it says
is what this study reports.

**No favourable moment is waited for.** The probe runs once, when this is invoked, and its
answer stands. There is no retry, no loop, and no second full run.

**No rule from `B81` is altered.** The comparison, its gates and its decision are untouched.
The readiness limits are `requalification`'s own; none was derived from the runs it filters.
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

from b81_recovery import CLASS, MAX_TOKENS, MODEL, VERIFY_DISPATCHES, _inject_drift, _run_child
from ironmule.qmv_variant import QUALIFIED_ACTION_ID
from ironmule.readiness import PROTOCOL as READINESS_PROTOCOL
from ironmule.readiness import evaluate_recorded_aa
from ironmule.readiness import probe as readiness_probe
from ironmule.requalification import BLOCKS, SESSIONS, lineage_path

PREREGISTRATION = {
    "experiment": "B82_live_requalification",
    "question": ("with a cheap readiness probe in front of it, does one full requalification "
                 "reach the live PASS branch B81 could not"),
    "readiness": READINESS_PROTOCOL,
    "exactly_one_full_run": ("if readiness passes, one comparison runs. If it fails, none "
                             "does. Either way this executes once and reports what it got"),
    "no_chasing": ("the probe runs when this is invoked. No favourable machine state is "
                   "waited for, no retry happens, and no second full run is started"),
    "b81_rules_unchanged": ("the comparison, its gates and its decision rule are exactly "
                            "B81's. The readiness limits are requalification's own"),
    "verdicts": {
        "B82_LIVE_REQUALIFICATION_PASS": ("readiness passed, one real comparison passed, and "
                                          "a restarted runtime dispatched the candidate"),
        "B82_NOT_READY": "readiness failed and no comparison was started",
        "B82_SAFE_REFERENCE": ("readiness passed and the comparison did not confirm the "
                               "candidate, or was invalid"),
        "B82_FAIL": "a state, safety or dispatch error",
    },
    "default_stays_off": "enable_local_learned_dispatch defaults to False and is not changed",
}


def _retrospective() -> dict:
    """Would this probe have stopped `B81`'s two expensive runs? Only from what they recorded.

    A requalification session's `reference_aa` arm *is* reference against reference over three
    blocks, which is the shape this probe uses, so the two are directly comparable. What is not
    reconstructible is timing: a probe runs once, before the comparison, so the honest
    counterfactual is the *first* session's control of each attempt, not the worst one.
    """
    outcomes = []
    for name in ("B81_recovery_20260911.json", "B81_recovery_20260911_attempt2.json"):
        record = json.loads((RAW / name).read_text())
        sessions = record["requalification"]["comparison"]["sessions"]
        first = sessions[0]
        ready, why = evaluate_recorded_aa(first["reference_aa"]["median"],
                                          first["aa_half_width"])
        per_session = [evaluate_recorded_aa(row["reference_aa"]["median"],
                                            row["aa_half_width"])[0] for row in sessions]
        outcomes.append({
            "attempt": name,
            "outcome_that_happened": record["outcome"],
            "first_session_aa_median": first["reference_aa"]["median"],
            "first_session_aa_half_width": first["aa_half_width"],
            "probe_would_have_passed": ready,
            "why": why,
            "would_have_been_prevented": not ready,
            "every_session_control_readable": per_session,
            "wall_seconds_that_would_have_been_saved": (
                record["cost"]["wall_seconds"] if not ready else 0.0),
        })
    prevented = [row for row in outcomes if row["would_have_been_prevented"]]
    return {
        "method": ("a requalification session's reference_aa arm is reference against "
                   "reference over three blocks, the same shape this probe uses, so the "
                   "recorded controls can be judged by the same rule"),
        "honest_limit": ("the probe runs before the comparison, so the counterfactual uses "
                         "each attempt's first session control. A disturbance that arrived "
                         "later is not something a probe taken earlier could have seen"),
        "not_retrofitted": ("no preflight existed during B81 and this does not pretend one "
                            "did. B81's records are unchanged"),
        "attempts": outcomes,
        "would_have_prevented": len(prevented),
        "of": len(outcomes),
        "wall_seconds_that_would_have_been_saved": sum(
            row["wall_seconds_that_would_have_been_saved"] for row in outcomes),
    }


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

    retrospective = _retrospective()

    intake, provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    if learner.recommendation(QUALIFIED_ACTION_ID, CLASS).local_learning_state \
            != CANDIDATE_QUALIFIED:
        raise SystemExit("the controller is not qualified; there is nothing to recover")

    scratch = Path(tempfile.mkdtemp(prefix="b82_live_"))
    state_path = learner.save(scratch / "local_learning.json")

    def spec(phase: str) -> dict:
        return {"phase": phase, "model": MODEL, "dispatches": VERIFY_DISPATCHES,
                "max_tokens": MAX_TOKENS, "prompt": PROBE_PROMPT,
                "state_path": str(state_path), "root": str(PROJECT_ROOT)}

    print("1. a qualified controller, and the drift written by the harness", flush=True)
    before = _run_child(spec("before_drift"))
    canary = json.loads((RAW / "B79_canary_20260911_v2.json").read_text())
    latencies = [row["end_to_end_ms"] for row in canary["phases"]["canary"]["dispatches"]]
    injection = _inject_drift(state_path, intake, default_action_code_digest(), latencies)

    print("2. restarting the runtime; the reference must serve", flush=True)
    after_drift = _run_child(spec("after_drift"))

    print("3. readiness: reference against reference, once", flush=True)

    def readiness_progress(block, arm):
        print(f"    readiness block {block} {arm} done", flush=True)

    readiness = readiness_probe(MODEL, on_progress=readiness_progress)
    print(f"   readiness: {'READY' if readiness.ready else 'NOT READY'} -- {readiness.reason}",
          flush=True)

    run = None
    after_requalification = None
    if readiness.ready:
        print("4. exactly one full requalification", flush=True)

        def progress(session, block, arm):
            print(f"    session {session} block {block} {arm} done", flush=True)

        run = requalify(MODEL, state_path=state_path, readiness=readiness.as_dict(),
                        on_progress=progress)
        print("5. restarting the runtime; which action is in charge now", flush=True)
        after_requalification = _run_child(spec("after_requalification"))

    lineage = json.loads(lineage_path(state_path).read_text())
    archived = sorted(str(p.name) for p in scratch.glob("*.epoch.json"))

    outcome = run["outcome"] if run else "NOT_READY"
    expected_after = "candidate" if outcome == "PASS" else "reference"
    checks = {
        "no_child_failed": not any(
            phase.get("failed") for phase in
            (before, after_drift, after_requalification) if phase),
        "the_candidate_ran_before_the_drift": before.get("unique_actions") == ["candidate"],
        "the_harness_produced_the_drift": bool(injection["drift_detected"]),
        "the_reference_served_after_the_drift":
            after_drift.get("unique_actions") == ["reference"],
        "the_state_survived_a_restart": bool(after_drift.get("requalification_required")),
        "readiness_decided_whether_to_spend_anything": (
            readiness.ready == (run is not None)),
        "exactly_one_full_run_at_most": (run is None) or (
            len(run.get("comparison", {}).get("sessions", [])) <= SESSIONS),
        "the_action_after_matches_the_outcome": (
            after_requalification is None
            or after_requalification.get("unique_actions") == [expected_after]),
        "correctness_held_throughout": all(
            phase.get("tokens_identical", False) for phase in
            (before, after_drift, after_requalification) if phase),
        "b81_rules_unchanged": bool(run is None or run["protocol"]["sessions"] == SESSIONS),
    }
    passed = all(checks.values())
    if not passed:
        verdict = "B82_FAIL"
    elif not readiness.ready:
        verdict = "B82_NOT_READY"
    elif outcome == "PASS":
        verdict = "B82_LIVE_REQUALIFICATION_PASS"
    else:
        verdict = "B82_SAFE_REFERENCE"

    readiness_cost = readiness.record["wall_seconds"]
    full_cost = run["cost"]["wall_seconds"] if run else None
    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "scratch": str(scratch),
        "phases": {"before_drift": before, "injection": injection,
                   "after_drift": after_drift,
                   "readiness": readiness.as_dict(),
                   "after_requalification": after_requalification},
        "readiness_ready": readiness.ready,
        "readiness_reason": readiness.reason,
        "requalification": run,
        "outcome": outcome,
        "lineage": lineage,
        "archived_epochs": archived,
        "checks": checks,
        "verdict": verdict,
        "retrospective_on_b81": retrospective,
        "cost": {
            "readiness_wall_seconds": readiness_cost,
            "readiness_model_loads": readiness.record["children_run"],
            "full_requalification_wall_seconds": full_cost,
            "full_requalification_model_loads": (run["cost"]["model_loads"] if run else 0),
            "readiness_share_of_a_full_run": (
                readiness_cost / full_cost if full_cost else None),
            "b81_wall_seconds_spent_on_invalid_runs": sum(
                json.loads((RAW / name).read_text())["cost"]["wall_seconds"]
                for name in ("B81_recovery_20260911.json",
                             "B81_recovery_20260911_attempt2.json")),
            "goal": ("avoid paying for a comparison a machine cannot support. Not to raise "
                     "the chance of a PASS, and not to make the candidate look better"),
        },
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/readiness.py", "ironmule/requalification.py",
                         "ironmule/monitoring.py", "ironmule/router.py",
                         "tools/b82_live.py")},
        "nothing_activated": ("everything is in a scratch directory, the user's store was "
                              "never written, and enable_local_learned_dispatch still "
                              "defaults to False"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "outcome": outcome,
                      "readiness": {"ready": readiness.ready, "reason": readiness.reason,
                                    "aa_median": readiness.record["aa"]["median"],
                                    "aa_half_width": readiness.record["aa_half_width"],
                                    "wall_seconds": round(readiness_cost)},
                      "checks": checks,
                      "actions": {name: phase.get("unique_actions") for name, phase in
                                  (("before_drift", before), ("after_drift", after_drift),
                                   ("after_requalification", after_requalification))
                                  if phase},
                      "cost": {k: record["cost"][k] for k in
                               ("readiness_wall_seconds",
                                "full_requalification_wall_seconds",
                                "readiness_share_of_a_full_run",
                                "b81_wall_seconds_spent_on_invalid_runs")},
                      "retrospective": {"would_have_prevented":
                                            retrospective["would_have_prevented"],
                                        "of": retrospective["of"],
                                        "seconds_saved":
                                            retrospective["wall_seconds_that_would_have_been_saved"]}},
                     indent=2, default=str))
    return 0 if verdict != "B82_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
