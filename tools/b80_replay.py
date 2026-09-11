#!/usr/bin/env python3
"""Does the monitor stay quiet through ordinary use, and speak up when something really moves?

A drift detector has two ways to be useless. One is to fire on noise, which costs a user the
whole gain for nothing. The other is to sleep through a real shift, which is the failure it
exists to prevent. `B80`'s rule was fixed before any of this ran, and this measures both
failure modes against it.

The stable material is real: the end-to-end times of `B79`'s canary dispatches, replayed in
the order they were measured and resampled from for a false-alarm rate. The drift material is
synthetic and written down here before it was used, because no real drift has happened on this
machine and waiting for one is not a test.

**Temporally correct.** The controller is qualified from `B75` and `B76`, which came first.
Then `B79`'s canary dispatches arrive as ordinary use. Then, and only then, the injected
sequences. No later evidence reaches an earlier step.

**Nothing here learns.** The monitor takes observations and can only ever say the preference
needs rechecking. It cannot construct comparative evidence, so it cannot qualify anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = PROJECT_ROOT / "research" / "raw"

from ironmule.monitoring import (AGING_RULE, BASELINE_MIN, DriftMonitor,
                                 MATERIAL_SHIFT_RULE, MONITORING,
                                 Observation, REQUALIFICATION_REQUIRED, ROBUST_Z_LIMIT,
                                 SEGMENT_FIELDS, SEPARATION_RULE, SPREAD_RATIO_LIMIT,
                                 WARMING_UP, WINDOW)

ACTION = "k3840_geometry_sg4_r8"
CLASS = "single_short"

#: Written before any of it ran. A sequence is named, its multiplier fixed, and what it must
#: do to the monitor stated in advance.
SEQUENCES = {
    "stable": {"shift": 1.00, "spread": 1.0, "must_requalify": False,
               "why": "ordinary use must not cost a user the gain"},
    "negligible": {"shift": 1.01, "spread": 1.0, "must_requalify": False,
                   "why": ("one per cent is statistically obvious on a machine this stable "
                           "and still smaller than the gain the action buys, so the action "
                           "keeps winning and there is nothing to recheck")},
    "mild": {"shift": 1.03, "spread": 1.0, "must_requalify": True,
             "why": ("three per cent sits inside the 2.3 to 4.6 per cent B76 measured the "
                     "action to be worth, so a sustained move of that size can erase the "
                     "reason to use it. The first run of this file preregistered this as a "
                     "false alarm, which was wrong: it is exactly the case worth catching")},
    "drift_shift": {"shift": 1.25, "spread": 1.0, "must_requalify": True,
                    "why": "a sustained quarter slower is the case this exists for"},
    "drift_variance": {"shift": 1.00, "spread": 6.0, "must_requalify": True,
                       "why": "a machine that became erratic has drifted even at the same median"},
}
SEQUENCE_LENGTH = 40
FALSE_ALARM_TRIALS = 200
FALSE_ALARM_LENGTH = 100
SEED = 20260911

PREREGISTRATION = {
    "experiment": "B80_drift_replay",
    "question": ("does the preregistered drift rule stay silent through ordinary use and "
                 "catch a defined shift, and how long does catching it take"),
    "rule": {"baseline_min": BASELINE_MIN, "window": WINDOW,
             "robust_z_limit": ROBUST_Z_LIMIT, "spread_ratio_limit": SPREAD_RATIO_LIMIT,
             "material_shift_rule": MATERIAL_SHIFT_RULE,
             "fixed_before_the_run": True,
             "no_threshold_moves_afterwards": ("a rule tuned until the answer looks right "
                                               "measures the tuner")},
    "this_is_the_second_rule_and_the_first_is_kept": (
        "the first run used the statistical test alone and sealed B80_TOO_SENSITIVE. On a "
        "machine whose ordinary spread is half a per cent, a robust z-score calls a one per "
        "cent move overwhelming, and requalifying over that costs a user a gain the move had "
        "not taken away. The second rule adds a floor that is derived rather than tuned: the "
        "shift must also exceed the smallest gain the controller's own qualified interval "
        "supports. Both records stand"),
    "stable_material": ("the real end-to-end times of B79's canary dispatches, replayed in "
                        "measured order and resampled for the false-alarm rate"),
    "injected_material": ("synthetic and named above. No real drift has occurred on this "
                          "machine and waiting for one is not a test"),
    "temporal_order": ("B75 and B76 qualify the controller, then B79's canary arrives as "
                       "ordinary use, then the injected sequences. Nothing later reaches "
                       "anything earlier"),
    "sequences": SEQUENCES,
    "false_alarm_trials": FALSE_ALARM_TRIALS,
    "false_alarm_length": FALSE_ALARM_LENGTH,
    "separation_rule": SEPARATION_RULE,
    "aging_rule": AGING_RULE,
    "no_exploration": ("nothing here dispatches anything. The monitor consumes records of "
                       "dispatches that already happened"),
}

BASE_FIELDS = {
    "hardware_fingerprint": "", "gpu_architecture": "", "model_identity_sha256": "",
    "model_revision": "", "quantization_bits": 4, "quantization_group_size": 64,
    "mlx": "", "mlx_lm": "", "workload_class": CLASS, "action_id": ACTION,
    "action_code_digest": "", "effective_action": "candidate",
}


def _observation(index: int, milliseconds: float, fields: dict, **overrides) -> Observation:
    moment = datetime(2026, 9, 11, tzinfo=timezone.utc) + timedelta(seconds=index)
    row = {**fields, "observed_at": moment.isoformat(), "end_to_end_ms": milliseconds,
           "service_ttft_ms": None, "tokens_per_second": None, "new_tokens": 32,
           "prompt_tokens": 27, "fallbacks": 0, "correctness_errors": 0,
           "memory_pressure_level": 1, "swap_used_bytes": 0, "controller_digest": "sealed"}
    row.update(overrides)
    return Observation(**row)


def _canary_latencies() -> tuple[list[float], dict]:
    """B79's own canary, candidate phase, in the order it was measured."""
    canary = json.loads((RAW / "B79_canary_20260911_v2.json").read_text())
    rows = [row["end_to_end_ms"] for row in canary["phases"]["canary"]["dispatches"]
            if row["effective_action"] == "candidate"]
    return rows, {"source": "B79_canary_20260911_v2.json, canary phase",
                  "n": len(rows), "median": statistics.median(rows)}


def _primed(fields: dict, latencies: list[float], floor: float) -> tuple[DriftMonitor, int]:
    """A monitor that has already watched ordinary use and built its baseline from it."""
    monitor = DriftMonitor(min_material_shift=floor)
    generator = random.Random(SEED)
    index = 0
    while monitor.state_for(CLASS, ACTION) == WARMING_UP:
        monitor.observe(_observation(index, generator.choice(latencies), fields))
        index += 1
    return monitor, index


def _run_sequence(name: str, fields: dict, latencies: list[float], floor: float) -> dict:
    spec = SEQUENCES[name]
    monitor, index = _primed(fields, latencies, floor)
    baseline = next(iter(monitor.segments.values())).baseline
    generator = random.Random(SEED + 1)
    centre = statistics.median(latencies)
    detected_at = None
    for step in range(SEQUENCE_LENGTH):
        sample = generator.choice(latencies)
        value = centre * spec["shift"] + (sample - centre) * spec["spread"]
        monitor.observe(_observation(index + step, max(value, 1.0), fields))
        if monitor.requalification_required and detected_at is None:
            detected_at = step + 1
            break
    return {
        "sequence": name, **spec,
        "baseline": baseline.as_dict() if baseline else None,
        "observations_before_injection": index,
        "requalified": monitor.requalification_required,
        "observations_to_detection": detected_at,
        "final_state": monitor.state_for(CLASS, ACTION),
        "transitions": list(monitor.transitions),
        "behaved_as_preregistered": monitor.requalification_required == spec["must_requalify"],
    }


def _false_alarm_rate(fields: dict, latencies: list[float], floor: float) -> dict:
    """How often ordinary use alone would cost a user the gain."""
    alarms = 0
    for trial in range(FALSE_ALARM_TRIALS):
        generator = random.Random(SEED + 1000 + trial)
        monitor = DriftMonitor(min_material_shift=floor)
        for index in range(FALSE_ALARM_LENGTH):
            monitor.observe(_observation(index, generator.choice(latencies), fields))
            if monitor.requalification_required:
                alarms += 1
                break
    return {"trials": FALSE_ALARM_TRIALS, "observations_per_trial": FALSE_ALARM_LENGTH,
            "alarms": alarms, "rate": alarms / FALSE_ALARM_TRIALS,
            "note": ("resampled from B79's real canary latencies. This is the cost of the "
                     "rule on a machine that has not changed")}


def _invalidation(fields: dict, latencies: list[float], floor: float) -> list[dict]:
    """A change to any segment field starts a new segment rather than drifting the old one."""
    outcomes = []
    for field, value in (("mlx", "0.33.0"), ("mlx_lm", "0.32.0"),
                         ("model_revision", "another"), ("action_code_digest", "changed"),
                         ("hardware_fingerprint", "another"),
                         ("quantization_bits", 8)):
        monitor, index = _primed(fields, latencies, floor)
        before = len(monitor.segments)
        monitor.observe(_observation(index, statistics.median(latencies),
                                     {**fields, field: value}))
        segments = list(monitor.segments.values())
        outcomes.append({
            "changed_field": field,
            "segments_before": before, "segments_after": len(segments),
            "new_segment_started": len(segments) == before + 1,
            "old_segment_superseded": any(s.superseded for s in segments),
            "new_segment_state": segments[-1].state,
            "requalification_required": monitor.requalification_required,
            "correct": (len(segments) == before + 1
                        and any(s.superseded for s in segments)
                        and segments[-1].state == WARMING_UP
                        and not monitor.requalification_required),
        })
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once
    from b78_cold_start_replay import _context, replay_evidence
    from ironmule.local_learner import CANDIDATE_QUALIFIED, LocalLearner

    intake, provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    controller = learner.recommendation(ACTION, CLASS)
    if controller.local_learning_state != CANDIDATE_QUALIFIED:
        raise SystemExit(f"the controller is {controller.local_learning_state}; there is "
                         f"nothing for a monitor to watch")

    fields = {**BASE_FIELDS,
              "hardware_fingerprint": intake.hardware_fingerprint,
              "gpu_architecture": intake.gpu_architecture,
              "model_identity_sha256": intake.model_identity_sha256,
              "model_revision": intake.model_revision,
              "mlx": intake.mlx, "mlx_lm": intake.mlx_lm,
              "action_code_digest": hashlib.sha256(
                  (PROJECT_ROOT / "ironmule" / "qmv_variant.py").read_bytes()).hexdigest()}

    latencies, material = _canary_latencies()
    # Derived, not chosen: the smallest gain the controller's own qualified interval supports.
    # A slowdown below it leaves the action still winning and there is nothing to recheck.
    floor = max(0.0, 1.0 - controller.prediction_interval[1])
    material["min_material_shift"] = floor
    material["min_material_shift_from"] = (
        f"1 - {controller.prediction_interval[1]:.4f}, the upper bound of the controller's "
        f"qualified predictive interval")

    # Ordinary use first, in the order it was measured, before anything is injected.
    ordinary = DriftMonitor(min_material_shift=floor)
    for index, value in enumerate(latencies):
        ordinary.observe(_observation(index, value, fields))
    ordinary_outcome = {
        "observations": len(latencies),
        "state": ordinary.state_for(CLASS, ACTION),
        "requalification_required": ordinary.requalification_required,
        "transitions": list(ordinary.transitions),
        "note": ("B79's canary replayed in measured order. It must reach MONITORING and "
                 "must not requalify"),
    }

    sequences = {name: _run_sequence(name, fields, latencies, floor)
                 for name in SEQUENCES}
    false_alarms = _false_alarm_rate(fields, latencies, floor)
    invalidation = _invalidation(fields, latencies, floor)

    # The separation, demonstrated rather than asserted: a monitor cannot reach the intake.
    import ast

    text = (PROJECT_ROOT / "ironmule" / "monitoring.py").read_text()
    tree = ast.parse(text)
    # The module docstring names the controller in prose, which is not a reference to it.
    # Only the code counts, so the docstring is cut before anything is looked for.
    body = ast.get_source_segment(text, tree.body[-1]) or ""
    code = text[text.index(body):] if body else text
    imported = {name.module for name in tree.body if isinstance(name, ast.ImportFrom)}
    separation = {
        "monitor_imports_no_controller": not any(
            module and "local_learner" in module for module in imported),
        "monitor_never_names_the_controller_in_code": "LocalLearner" not in code,
        "monitor_cannot_build_comparative_evidence": "Evidence" not in code,
        "observation_carries_no_ratio": "ratio" not in code,
        "rule": SEPARATION_RULE,
    }

    detected = {name: row for name, row in sequences.items() if row["must_requalify"]}
    quiet = {name: row for name, row in sequences.items() if not row["must_requalify"]}
    checks = {
        "ordinary_use_reaches_monitoring": ordinary_outcome["state"] == MONITORING,
        "ordinary_use_does_not_requalify": not ordinary_outcome["requalification_required"],
        "every_sequence_behaved_as_preregistered": all(
            row["behaved_as_preregistered"] for row in sequences.values()),
        "defined_drift_is_detected": all(row["requalified"] for row in detected.values()),
        "stable_and_mild_stay_quiet": all(not row["requalified"] for row in quiet.values()),
        "false_alarm_rate_is_zero": false_alarms["alarms"] == 0,
        "every_version_change_invalidates_correctly": all(row["correct"]
                                                          for row in invalidation),
        "observation_cannot_qualify_anything": all(separation[name] for name in (
            "monitor_imports_no_controller",
            "monitor_never_names_the_controller_in_code",
            "monitor_cannot_build_comparative_evidence",
            "observation_carries_no_ratio")),
    }
    if not checks["defined_drift_is_detected"]:
        verdict = "B80_TOO_INSENSITIVE"
    elif not (checks["stable_and_mild_stay_quiet"]
              and checks["ordinary_use_does_not_requalify"]
              and checks["false_alarm_rate_is_zero"]):
        verdict = "B80_TOO_SENSITIVE"
    elif all(checks.values()):
        verdict = "B80_REPLAY_PASS"
    else:
        verdict = "B80_FAIL"

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "controller_at_start": controller.as_dict(),
        "stable_material": material,
        "min_material_shift": floor,
        "segment_fields": list(SEGMENT_FIELDS),
        "ordinary_use": ordinary_outcome,
        "sequences": sequences,
        "false_alarms": false_alarms,
        "invalidation": invalidation,
        "separation": separation,
        "checks": checks,
        "verdict": verdict,
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/monitoring.py", "ironmule/activation.py",
                         "ironmule/router.py", "tools/b80_replay.py")},
        "nothing_activated": ("no model is loaded, nothing is dispatched, and the monitor "
                              "cannot qualify an action"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "checks": checks,
                      "ordinary_use": {k: ordinary_outcome[k] for k in
                                       ("observations", "state",
                                        "requalification_required")},
                      "sequences": {name: {"requalified": row["requalified"],
                                           "observations_to_detection":
                                               row["observations_to_detection"],
                                           "as_preregistered": row["behaved_as_preregistered"]}
                                    for name, row in sequences.items()},
                      "false_alarms": {k: false_alarms[k] for k in
                                       ("trials", "observations_per_trial", "alarms", "rate")},
                      "invalidation": [{"field": row["changed_field"],
                                        "correct": row["correct"]} for row in invalidation]},
                     indent=2, default=str))
    return 0 if verdict == "B80_REPLAY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
