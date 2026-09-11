#!/usr/bin/env python3
"""`B81`'s verdict, from both live attempts and the code that makes the rules structural.

Two full recovery runs were made. Neither requalification cleared its own gates, so neither
restored the candidate, and both left the machine on the reference with the requalification
still required. That is the fail-closed branch working, and it is reported as what it is: the
`PASS` branch is proven by tests and was not reached live.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = PROJECT_ROOT / "research" / "raw"

GATES = (
    "drift leads persistently to the reference",
    "passive data never clears the state",
    "only an explicit requalification may decide",
    "a passing requalification restores the candidate correctly",
    "negative and invalid runs stay fail-closed",
    "restarts preserve the state correctly",
    "the default stays off",
)


def _aa_spread(record: dict) -> dict:
    rows = record["requalification"]["comparison"]["sessions"]
    controls = [row["reference_aa"]["median"] for row in rows]
    widths = [row["aa_half_width"] for row in rows if row["aa_half_width"] is not None]
    return {"sessions": len(rows), "aa_medians": controls,
            "aa_half_widths": widths,
            "max_half_width": max(widths) if widths else None,
            "sessions_passing_the_gate": sum(1 for row in rows if row["aa_gate_passed"]),
            "disturbed_blocks": sum(len(row["disturbed_blocks"]) for row in rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, nargs="+",
                        default=[RAW / "B81_recovery_20260911.json",
                                 RAW / "B81_recovery_20260911_attempt2.json"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule import requalification as requalification_module
    from ironmule import router as router_module
    from ironmule.router import AppleRuntime

    attempts = [json.loads(path.read_text()) for path in args.attempts]

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_b81_requalification.py",
         "tests/test_b80_monitoring.py", "tests/test_b79_activation.py",
         "tests/test_b78_local_learner.py", "tests/engine/test_router.py", "-n", "0", "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    # The dispatch path never imports the requalification module, so recovery adds nothing
    # to the cost of serving a request. Checked at the import level: the first pass of this
    # tool grepped the source for the word and matched a comment, which is a check of prose
    # rather than of structure. That record is kept.
    import ast

    router_tree = ast.parse(Path(router_module.__file__).read_text())
    imported = {node.module for node in ast.walk(router_tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(router_tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    dispatch_untouched = not any("requalification" in str(name) for name in imported)

    default_off = inspect.signature(AppleRuntime.load).parameters[
        "enable_local_learned_dispatch"].default is False

    outcomes = [record["outcome"] for record in attempts]
    restored = [record for record in attempts
                if record["phases"]["after_requalification"]["unique_actions"] == ["candidate"]]
    checks = {
        "drift_leads_to_the_reference": all(
            record["checks"]["the_reference_served_after_the_drift"] for record in attempts),
        "state_survives_a_restart": all(
            record["checks"]["the_state_survived_a_restart"] for record in attempts),
        "good_news_never_clears_it": all(
            record["checks"]["good_news_did_not_clear_it"] for record in attempts),
        "only_an_explicit_run_may_decide": all(
            record["checks"]["the_requalification_started_only_when_required"]
            for record in attempts) and tests.returncode == 0,
        "invalid_runs_stay_fail_closed": all(
            record["requalification"]["applied"]["state"] == "REQUALIFICATION_REQUIRED"
            for record in attempts if record["outcome"] == "INVALID"),
        "no_run_ever_restored_the_candidate_without_passing": not restored,
        "the_action_after_matched_the_outcome": all(
            record["checks"]["the_action_after_matches_the_outcome"] for record in attempts),
        "correctness_held_throughout": all(
            record["checks"]["correctness_held_throughout"] for record in attempts),
        "the_pass_branch_is_covered_by_tests": tests.returncode == 0,
        "dispatch_path_untouched_by_recovery": bool(dispatch_untouched),
        "default_is_off": bool(default_off),
    }
    if not all(checks.values()):
        verdict = "B81_FAIL"
    elif "PASS" in outcomes:
        verdict = "B81_REQUALIFICATION_PASS"
    else:
        verdict = "B81_SAFE_REFERENCE"

    record = {
        "experiment": "B81_requalification",
        "verdict": verdict,
        "gates": list(GATES),
        "checks": checks,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "attempts": [{"file": path.name, "outcome": attempt["outcome"],
                      "verdict": attempt["verdict"],
                      "decision": attempt["requalification"]["decision"],
                      "resulting_state": attempt["requalification"]["applied"]["state"],
                      "actions": {name: phase.get("unique_actions")
                                  for name, phase in attempt["phases"].items()
                                  if isinstance(phase, dict) and "unique_actions" in phase},
                      "session_ratios": [row["candidate"]["median"] for row in
                                         attempt["requalification"]["comparison"]["sessions"]],
                      "machine_noise": _aa_spread(attempt),
                      "cost": attempt["cost"]}
                     for path, attempt in zip(args.attempts, attempts)],
        "the_pass_branch": {
            "reached_live": False,
            "covered_by": ["tests/test_b81_requalification.py::"
                           "test_a_pass_restores_the_candidate_on_new_evidence",
                           "tests/test_b81_requalification.py::"
                           "test_the_previous_epoch_is_archived_and_never_deleted"],
            "why_not_live": ("both runs met a machine too noisy to read. In each, at least "
                             "one block deviated past the drift gate and at least one A/A "
                             "control was far too wide. The gate refused rather than "
                             "believing the numbers, which is the behaviour being tested"),
            "not_retried_further": ("a third attempt would be running until the answer is "
                                    "the one wanted. Two are recorded and both stand"),
        },
        "cost": {
            "wall_seconds": [attempt["cost"]["wall_seconds"] for attempt in attempts],
            "model_loads": [attempt["cost"]["model_loads"] for attempt in attempts],
            "comparative_requests": [attempt["cost"]["comparative_requests"]
                                     for attempt in attempts],
            "runtime_overhead_outside_requalification": (
                "none. The dispatch path does not import the requalification module at all, "
                "so recovery costs a served request nothing. B80's per-dispatch monitor "
                "check of 8 ns is unchanged by B81"),
            "how_long_a_user_stays_on_the_reference": (
                "from the moment monitoring takes the action away until a requalification "
                "passes. On this machine, across both attempts, that is still ongoing: a "
                "machine too noisy to measure keeps its user on the reference, which is the "
                "safe end of the trade"),
        },
        "state_machine": {
            "permitted": ["CANDIDATE_QUALIFIED -> REQUALIFICATION_REQUIRED (drift)",
                          "REQUALIFICATION_REQUIRED -> CANDIDATE_QUALIFIED (explicit PASS)",
                          "REQUALIFICATION_REQUIRED -> REFERENCE_ONLY (explicit NO_GAIN or WORSE)",
                          "REQUALIFICATION_REQUIRED -> REQUALIFICATION_REQUIRED (INVALID)"],
            "forbidden": ["anything observational clearing REQUALIFICATION_REQUIRED",
                          "a requalification starting without the state requiring it",
                          "a kill record being cleared by a measurement"],
        },
        "protocol": requalification_module.PROTOCOL,
        "test_returncode": tests.returncode,
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/requalification.py", "ironmule/monitoring.py",
                         "ironmule/activation.py", "ironmule/router.py", "ironmule_cli.py",
                         "tests/test_b81_requalification.py")},
        "what_this_does_not_show": ("how often real drift happens. B81 proves the recovery "
                                    "mechanism, and on this machine it proved the half of it "
                                    "that refuses"),
        "nothing_activated": ("both runs used scratch directories, the user's store was never "
                              "written, and enable_local_learned_dispatch still defaults to "
                              "False. Nothing is committed or pushed"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "attempts": [{"outcome": row["outcome"],
                                    "reason": row["decision"]["reason"],
                                    "state": row["resulting_state"],
                                    "aa_max_half_width": row["machine_noise"]["max_half_width"],
                                    "aa_sessions_passing":
                                        row["machine_noise"]["sessions_passing_the_gate"],
                                    "disturbed_blocks":
                                        row["machine_noise"]["disturbed_blocks"],
                                    "wall_seconds": round(row["cost"]["wall_seconds"])}
                                   for row in record["attempts"]],
                      "pass_branch_reached_live": False}, indent=2, default=str))
    return 0 if verdict != "B81_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
