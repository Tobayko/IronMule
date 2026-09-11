#!/usr/bin/env python3
"""`B79`'s verdict, from the artefacts that earn it.

Every condition is read from a sealed record or from a test run that exits non-zero when it
fails. A missing artefact is `B79_ACTIVATION_FAIL`, never a default.
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
    "the preflight is exactly correct at every step of the measured history",
    "the candidate runs only where local evidence qualifies it",
    "every mismatch, on every axis, ends at the reference",
    "the canary carried no correctness or safety failure",
    "no fallback occurred in the candidate path",
    "the persistent kill switch disables a later process",
    "the default is still off",
    "no hardware or workload class is extrapolated",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path,
                        default=RAW / "B79_preflight_replay_20260911_v2.json")
    parser.add_argument("--canary", type=Path, default=RAW / "B79_canary_20260911_v2.json")
    parser.add_argument("--overhead", type=Path,
                        default=RAW / "B79_overhead_20260911_v4.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule import activation as activation_module
    from ironmule.qmv_variant import QUALIFIED_ACTION_ID, QUALIFIED_GEOMETRY
    from ironmule.router import AppleRuntime, ExecutionRouter

    preflight = json.loads(args.preflight.read_text())
    canary = json.loads(args.canary.read_text())
    overhead = json.loads(args.overhead.read_text())

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_b79_activation.py",
         "tests/test_b78_local_learner.py", "tests/engine/test_router.py", "-n", "0", "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    default_off = inspect.signature(AppleRuntime.load).parameters[
        "enable_local_learned_dispatch"].default is False
    decide_blind = ("local_learn" not in inspect.getsource(ExecutionRouter.decide)
                    and "activation" not in inspect.getsource(ExecutionRouter.decide))
    no_magnitude = all(name not in inspect.getsource(activation_module)
                       for name in ("estimated_ratio", "prediction_interval",
                                    "predicted_ratio"))
    classes_activated = sorted({row.get("workload_class")
                                for phase in canary["phases"].values()
                                for row in phase["dispatches"]
                                if row.get("effective_action") == "candidate"})

    checks = {
        "preflight_exactly_correct": bool(preflight["preflight_passed"]),
        "candidate_only_on_qualified_evidence": bool(
            preflight["checks"]["candidate_never_before_qualification"]
            and canary["checks"]["canary_actually_ran_the_candidate"]),
        "every_mismatch_ends_at_the_reference": bool(
            preflight["checks"]["no_other_route_or_class_ever_activates"]
            and preflight["checks"]["opting_out_is_always_reference"]
            and tests.returncode == 0),
        "canary_clean": bool(canary["checks"]["correctness_identical_everywhere"]
                             and canary["checks"]["resource_gate_passed"]
                             and canary["checks"]["no_child_failed"]
                             and canary["checks"]["fewer_than_two_runtime_errors"]),
        "no_fallback_in_the_candidate_path": bool(canary["checks"]["no_fallbacks"]),
        "persistent_kill_switch_works": bool(
            canary["checks"]["the_kill_record_disabled_a_later_process"]),
        "default_is_off": bool(default_off),
        "nothing_extrapolated": bool(classes_activated == ["single_short"]),
        "hot_path_within_margin": bool(overhead["verdict"] == "B79_OVERHEAD_PASS"),
        "decide_untouched": bool(decide_blind),
        "no_magnitude_decides_activation": bool(no_magnitude),
    }
    safety_broken = not (checks["canary_clean"]
                         and checks["no_fallback_in_the_candidate_path"]
                         and checks["candidate_only_on_qualified_evidence"]
                         and checks["every_mismatch_ends_at_the_reference"])
    if all(checks.values()):
        verdict = "B79_LEARNED_DISPATCH_CONFIRMED"
    elif safety_broken:
        verdict = "B79_ACTIVATION_FAIL"
    else:
        verdict = "B79_SAFE_FALLBACK"

    record = {
        "experiment": "B79_learned_dispatch",
        "verdict": verdict,
        "conditions": list(CONDITIONS),
        "checks": checks,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"preflight": args.preflight.name, "canary": args.canary.name,
                   "overhead": args.overhead.name,
                   "test_suites": ["tests/test_b79_activation.py",
                                   "tests/test_b78_local_learner.py",
                                   "tests/engine/test_router.py"],
                   "test_returncode": tests.returncode},
        "action": {"action_id": QUALIFIED_ACTION_ID,
                   "geometry": {"num_simdgroups": QUALIFIED_GEOMETRY[0],
                                "results_per_simdgroup": QUALIFIED_GEOMETRY[1]}},
        "workload_classes_activated": classes_activated,
        "canary": {"candidate_dispatches": sum(
                       1 for row in canary["phases"]["canary"]["dispatches"]
                       if row["effective_action"] == "candidate"),
                   "timing": canary["timing"],
                   "canary_over_controls": canary["canary_over_controls"],
                   "note": ("the before/after controls detect a gross regression. They are "
                            "not a paired AB design and this ratio is not a qualification "
                            "of the gain; B76 measured that")},
        "overhead": {"projection_B_over_A":
                         overhead["comparisons"]["projection_B_over_A"]["median"],
                     "projection_B_interval": [
                         overhead["comparisons"]["projection_B_over_A"]["ci_low"],
                         overhead["comparisons"]["projection_B_over_A"]["ci_high"]],
                     "projection_C_over_A":
                         overhead["comparisons"]["projection_C_over_A"]["median"],
                     "decision_B_over_A_reported_not_gated":
                         overhead["comparisons"]["decision_B_over_A"]["median"],
                     "added_nanoseconds_per_dispatch":
                         overhead["added_nanoseconds_per_dispatch"],
                     "share_of_one_measured_dispatch":
                         overhead["share_of_one_measured_dispatch"]},
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/activation.py", "ironmule/qmv_variant.py",
                         "ironmule/local_learner.py", "ironmule/router.py",
                         "tests/test_b79_activation.py")},
        "what_is_now_shown": (
            "unknown local machine state -> collect evidence -> learn a preference -> "
            "persist it -> use the learned action in a real dispatch. That is not "
            "reinforcement learning and not cross-hardware learning, and neither is claimed"),
        "next": ("B80 is controlled continual learning during ordinary use, without "
                 "exploration. It is not started here"),
        "nothing_activated_by_default": (
            "enable_local_learned_dispatch defaults to False, the canary wrote its state to "
            "a scratch directory, no product path is switched on, and nothing is committed "
            "or pushed"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "workload_classes_activated": classes_activated,
                      "overhead": record["overhead"]}, indent=2, default=str))
    return 0 if verdict.endswith("CONFIRMED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
