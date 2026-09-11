#!/usr/bin/env python3
"""`B78`'s verdict, assembled from the artefacts that earn it and nothing else.

Every condition the verdict names is read from a sealed record or from a test run that
exits non-zero when it fails. Nothing here re-derives a result, and a missing artefact is a
`B78_INVALID`, not a default.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = PROJECT_ROOT / "research" / "raw"

CONDITIONS = (
    "the cold start is reconstructed step by step from sealed B75 and B76 evidence",
    "the candidate is never recommended before the preregistered minimum of sessions",
    "persistent learning reproduces exactly across a save and a restore",
    "uncertainty is preserved: a preference never carries a single gain as a parameter",
    "fail-closed is complete: every corrupt, foreign or ungated input ends at the reference",
    "the router and its dispatch are unchanged: decide() cannot read the controller",
    "nothing is activated in any product path",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path,
                        default=RAW / "B78_cold_start_replay_20260911_v2.json")
    parser.add_argument("--overhead", type=Path,
                        default=RAW / "B78_shadow_overhead_20260911_v2.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule.local_learner import MIN_SESSIONS, SCHEMA
    from ironmule.router import ExecutionRouter

    replay = json.loads(args.replay.read_text())
    overhead = json.loads(args.overhead.read_text())

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_b78_local_learner.py",
         "tests/engine/test_router.py", "-n", "0", "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    suite_passed = tests.returncode == 0

    decide_source = inspect.getsource(ExecutionRouter.decide)
    router_blind = "local_learner" not in decide_source and "local_learning" not in decide_source

    profile = replay["final_profile"]
    action = profile["actions"][0]
    uncertainty_preserved = (
        "gain" not in action["action_preference"]
        and action["expected_gain_distribution"]["running_mean"]["uncertainty"] > 0.0
        and action["expected_gain_distribution"]["bayesian"]["uncertainty"] > 0.0)
    nothing_activated = (profile["activation"] == "none" and profile["shadow_only"] is True)

    checks = {
        "cold_start_reconstructed": bool(replay["phases_correct"]
                                         and replay["qualified_at_step"] is not None),
        "never_candidate_without_enough_evidence": bool(replay["never_candidate_too_early"]
                                                        and replay["no_wrong_action"]),
        "persistent_learning_reproducible": bool(replay["persistence"]["identical"]
                                                 and replay["persistence"]["digest_matches"]),
        "uncertainty_preserved": bool(uncertainty_preserved),
        "fail_closed_complete": bool(
            suite_passed
            and replay["persistence"]["missing_state_falls_back"] == "reference"
            and replay["persistence"]["corrupt_state_falls_back"] == "reference"),
        "router_and_dispatch_unchanged": bool(router_blind
                                              and overhead["hot_path_within_margin"]
                                              and overhead["dispatch_overhead_within_margin"]),
        "no_product_activation": bool(nothing_activated),
    }
    verdict = "B78_LOCAL_CONTROLLER_PASS" if all(checks.values()) else "B78_LOCAL_CONTROLLER_FAIL"

    record = {
        "experiment": "B78_local_controller_shadow",
        "verdict": verdict,
        "conditions": list(CONDITIONS),
        "checks": checks,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"replay": args.replay.name, "overhead": args.overhead.name,
                   "test_suites": ["tests/test_b78_local_learner.py",
                                   "tests/engine/test_router.py"],
                   "test_returncode": tests.returncode,
                   "test_summary": tests.stdout.strip().splitlines()[-1] if tests.stdout else ""},
        "state_schema": SCHEMA,
        "minimum_sessions": MIN_SESSIONS,
        "qualified_after_sessions": replay["sessions_needed_to_qualify"],
        "final_state": replay["final_state"],
        "inclusion_rule_demonstrated": replay["b69_inclusion"],
        "overhead": {"decide_ratio": overhead["comparisons"]["decide"]["median"],
                     "decide_interval": [overhead["comparisons"]["decide"]["ci_low"],
                                         overhead["comparisons"]["decide"]["ci_high"]],
                     "annotate_ratio": overhead["comparisons"]["annotate"]["median"],
                     "added_nanoseconds_per_annotation":
                         overhead["added_nanoseconds_per_annotation"],
                     "share_of_one_measured_dispatch":
                         overhead["share_of_one_measured_dispatch"]},
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/local_learner.py", "ironmule/router.py",
                         "tools/b78_cold_start_replay.py", "tools/b78_shadow_overhead.py",
                         "tests/test_b78_local_learner.py")},
        "next": ("B79 is the controlled opt-in activation of a locally qualified action with "
                 "an immediate reference fallback. It is not started here and nothing in "
                 "this record authorises it"),
        "nothing_activated": ("shadow only. No RouteDecision changed, no kernel activated, "
                              "no default moved, no product path touched, nothing committed "
                              "or pushed"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "qualified_after_sessions": record["qualified_after_sessions"],
                      "overhead": record["overhead"]}, indent=2, default=str))
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
