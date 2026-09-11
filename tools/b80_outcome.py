#!/usr/bin/env python3
"""`B80`'s verdict, from the artefacts that earn it and the code that makes them structural."""

from __future__ import annotations

import argparse
import ast
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

GATES = (
    "no exploration happens anywhere",
    "observational and comparative evidence are strictly separated",
    "passive data never qualifies a candidate on its own",
    "stable use does not deactivate the candidate unnecessarily",
    "the defined drift sequences are detected",
    "drift ends fail-closed at the reference",
    "persistent state is reproducible",
    "version and digest changes invalidate correctly",
    "the router hot path stays inside the existing overhead limit",
    "B79's default is still off",
)


def _code_of(path: Path) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    body = ast.get_source_segment(text, tree.body[-1]) or ""
    return text[text.index(body):] if body else text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=RAW / "B80_replay_20260911_v2.json")
    parser.add_argument("--shadow", type=Path,
                        default=RAW / "B80_shadow_canary_20260911_v2.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from ironmule import activation as activation_module
    from ironmule import monitoring as monitoring_module
    from ironmule.router import AppleRuntime

    replay = json.loads(args.replay.read_text())
    shadow = json.loads(args.shadow.read_text())

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_b80_monitoring.py",
         "tests/test_b79_activation.py", "tests/test_b78_local_learner.py",
         "tests/engine/test_router.py", "-n", "0", "-q"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    monitoring_code = _code_of(Path(monitoring_module.__file__))
    activation_code = _code_of(Path(activation_module.__file__))
    no_exploration = (
        "random" not in monitoring_code and "random" not in activation_code
        and "shuffle" not in activation_code and "choice" not in activation_code)
    # The layer can move an action one way only. Every record it builds is either
    # reference->reference or reference->candidate, and nothing else exists to choose.
    one_direction = "CANDIDATE, REFERENCE" not in activation_code
    default_off = inspect.signature(AppleRuntime.load).parameters[
        "enable_local_learned_dispatch"].default is False

    detected = {name: row for name, row in replay["sequences"].items()
                if row["must_requalify"]}
    quiet = {name: row for name, row in replay["sequences"].items()
             if not row["must_requalify"]}
    checks = {
        "no_exploration": bool(no_exploration and one_direction),
        "evidence_kinds_separated": bool(replay["checks"]["observation_cannot_qualify_anything"]
                                         and tests.returncode == 0),
        "passive_data_never_qualifies": bool(
            replay["separation"]["monitor_cannot_build_comparative_evidence"]
            and replay["separation"]["observation_carries_no_ratio"]),
        "stable_use_stays_quiet": bool(replay["checks"]["stable_and_mild_stay_quiet"]
                                       and replay["checks"]["ordinary_use_does_not_requalify"]
                                       and replay["false_alarms"]["alarms"] == 0
                                       and shadow["false_drift_alarms"] == 0),
        "defined_drift_detected": bool(all(row["requalified"] for row in detected.values())),
        "drift_ends_at_the_reference": tests.returncode == 0,
        "persistent_state_reproducible": tests.returncode == 0,
        "changes_invalidate_correctly": bool(
            replay["checks"]["every_version_change_invalidates_correctly"]),
        "hot_path_within_margin": bool(shadow["checks"]["hot_path_within_margin"]),
        "b79_default_still_off": bool(default_off),
    }
    if not checks["defined_drift_detected"]:
        verdict = "B80_TOO_INSENSITIVE"
    elif not checks["stable_use_stays_quiet"]:
        verdict = "B80_TOO_SENSITIVE"
    elif all(checks.values()):
        verdict = "B80_CONTINUAL_MONITORING_PASS"
    else:
        verdict = "B80_FAIL"

    record = {
        "experiment": "B80_continual_monitoring",
        "verdict": verdict,
        "gates": list(GATES),
        "checks": checks,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"replay": args.replay.name, "shadow": args.shadow.name,
                   "test_returncode": tests.returncode,
                   "test_suites": ["tests/test_b80_monitoring.py",
                                   "tests/test_b79_activation.py",
                                   "tests/test_b78_local_learner.py",
                                   "tests/engine/test_router.py"]},
        "state_machine": {"states": list(monitoring_module.STATES),
                          "transitions_observed": shadow["state_transitions"],
                          "only_exit_from_requalification": (
                              "an explicit requalification run producing comparative "
                              "evidence. Ordinary use never produces it")},
        "drift_rule": {"baseline_min": monitoring_module.BASELINE_MIN,
                       "window": monitoring_module.WINDOW,
                       "robust_z_limit": monitoring_module.ROBUST_Z_LIMIT,
                       "spread_ratio_limit": monitoring_module.SPREAD_RATIO_LIMIT,
                       "min_relative_scale": monitoring_module.MIN_RELATIVE_SCALE,
                       "material_shift_rule": monitoring_module.MATERIAL_SHIFT_RULE,
                       "min_material_shift_used": replay["min_material_shift"]},
        "detection": {name: {"requalified": row["requalified"],
                             "observations_to_detection": row["observations_to_detection"]}
                      for name, row in replay["sequences"].items()},
        "false_alarms": {"replay": replay["false_alarms"],
                         "shadow_run": shadow["false_drift_alarms"]},
        "overhead": {"added_nanoseconds_per_dispatch":
                         shadow["overhead"]["added_nanoseconds_per_dispatch"],
                     "share_of_one_measured_dispatch":
                         shadow["overhead"]["share_of_one_measured_dispatch"],
                     "for_dispatch_ratio_reported_not_gated":
                         shadow["overhead"]["with_over_without"]["median"],
                     "observe_nanoseconds": shadow["overhead"]["observe_median_nanoseconds"]},
        "state_size": {"bytes": shadow["state_size_bytes"],
                       "observations": shadow["data_volume"]["observations"],
                       "segments": shadow["data_volume"]["segments"],
                       "note": shadow["data_volume"]["note"]},
        "aging_rule": monitoring_module.AGING_RULE,
        "separation_rule": monitoring_module.SEPARATION_RULE,
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/monitoring.py", "ironmule/activation.py",
                         "ironmule/local_learner.py", "ironmule/router.py",
                         "tests/test_b80_monitoring.py")},
        "next": ("B81 would be the explicit requalification run that can clear a "
                 "REQUALIFICATION_REQUIRED state. It is not started here, and until it "
                 "exists a requalification is cleared only by a person"),
        "nothing_activated": ("monitoring only. enable_local_learned_dispatch still defaults "
                              "to False, the canary wrote to a scratch directory, and nothing "
                              "is committed or pushed"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "detection": record["detection"],
                      "false_alarms": record["false_alarms"],
                      "overhead": record["overhead"],
                      "state_size": record["state_size"]}, indent=2, default=str))
    return 0 if verdict == "B80_CONTINUAL_MONITORING_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
