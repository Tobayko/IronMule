#!/usr/bin/env python3
"""The local learner, walked forward through evidence it has already been given once.

`B78`'s question is not whether the `(4, 8)` geometry is faster here. `B69`, `B75` and `B76`
settled the sign and `B77` settled how much of the magnitude is trustworthy. The question is
whether a persistent controller, starting from nothing, would have said the right thing *at
every point on the way*, and in particular would never have recommended a candidate before
the evidence entitled it to.

So the controller is replayed against `B75`'s cold-start qualification and `B76`'s fourteen
sessions, in the order they were actually measured, and after every step what it would have
recommended is sealed `write_once`. Nothing is measured here. No model is loaded, no GPU is
touched, and the controller never acts on anything: `ExecutionRouter.decide` does not read it.

`B69` is not replay evidence. It is offered to the controller once at the end to demonstrate
the inclusion rule from `B77`: valid historical evidence whose own A/A control is too wide to
set a scale enters the preference and never the gain distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = PROJECT_ROOT / "research" / "raw"

from ironmule.local_learner import (CANDIDATE, CANDIDATE_QUALIFIED, COLLECTING, Evidence,
                                    INCLUSION_RULE, IntakeContext, LocalLearner,
                                    MIN_SESSIONS, REFERENCE, UNKNOWN, VALID, evidence_from)

ACTION = "k3840_geometry_sg4_r8"
CLASS = "single_short"
REFERENCE_STACK = "A"
MODEL = "mlx-community/gemma-3-12b-it-4bit"

PREREGISTRATION = {
    "experiment": "B78_local_controller_shadow",
    "question": ("starting from no state, would the persistent local controller have "
                 "recommended the right action at every point of B75's and B76's measured "
                 "history, and never a candidate before the evidence entitled it to"),
    "no_new_measurement": ("every row is read from sealed B75 and B76 records. No model is "
                           "loaded, no GPU touched and nothing is timed"),
    "shadow_only": ("ExecutionRouter.decide never reads the controller. A recommendation is "
                    "attached in annotate(), once per dispatch, after the route exists"),
    "not_cross_hardware": "one machine, one model. B73 remains the cross-hardware test",
    "no_context_features": ("load, free memory and swap are not inputs. B77 measured a "
                            "ridge over them as worse than a constant model on prediction "
                            "error, coverage and regret"),
    "minimum_sessions": MIN_SESSIONS,
    "inclusion_rule": INCLUSION_RULE,
    "b69_is_not_replay_evidence": ("it is offered once at the end to demonstrate the "
                                   "inclusion rule, and is not part of the walk forward"),
    "checks": [
        "the controller never recommends the candidate before MIN_SESSIONS accepted sessions",
        "an empty controller is UNKNOWN and serves the reference",
        "every step before qualification is COLLECTING and serves the reference",
        "no step ever recommends an action the sealed evidence measured as worse",
        "the recommendation survives a save and a restore unchanged",
    ],
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context() -> tuple[IntakeContext, dict]:
    """Machine and libraries from the sealed records; model identity from the local snapshot.

    The sealed records carry the fingerprint and the library builds but not the model
    identity, so that one fact is resolved here and the provenance is recorded rather than
    implied.
    """
    from ironmule.tune import resolve_local_model

    resolved = resolve_local_model(MODEL)
    facts = json.loads((RAW / "B75_cold_start_20260910_v2.json").read_text())["static_facts"]
    context = IntakeContext(
        hardware_fingerprint=facts["hardware_fingerprint"],
        gpu_architecture=facts["gpu_architecture"],
        mlx=facts["mlx"], mlx_lm=facts["mlx_lm"], model_id=MODEL,
        model_identity_sha256=resolved.identity.identity_sha256,
        model_revision=resolved.identity.revision)
    provenance = {
        "machine_and_libraries": "read from B75's sealed static_facts",
        "model_identity": ("resolved from the local snapshot, because the sealed records "
                           "carry no model identity. Every study used this model id"),
        "model_identity_sha256": resolved.identity.identity_sha256,
        "model_revision": resolved.identity.revision,
    }
    return context, provenance


def _row(evidence_id: str, measured_at: str, candidate: dict, aa: dict,
         aa_half_width: float | None, within_se: float | None,
         context: IntakeContext) -> Evidence:
    return evidence_from({
        "evidence_id": evidence_id, "action_id": ACTION, "workload_class": CLASS,
        "reference_stack": REFERENCE_STACK,
        "hardware_fingerprint": context.hardware_fingerprint,
        "gpu_architecture": context.gpu_architecture,
        "mlx": context.mlx, "mlx_lm": context.mlx_lm, "model_id": context.model_id,
        "model_identity_sha256": context.model_identity_sha256,
        "model_revision": context.model_revision,
        "ratio": candidate["median"], "ci_low": candidate["ci_low"],
        "ci_high": candidate["ci_high"], "within_session_se": within_se,
        "aa_median": aa["median"], "aa_half_width": aa_half_width,
        "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
        "measured_at": measured_at})


def _standard_error(ratios) -> float | None:
    if len(ratios) < 2:
        return None
    return statistics.stdev(ratios) / (len(ratios) ** 0.5)


def replay_evidence(context: IntakeContext) -> list[Evidence]:
    """B75 first, then B76's fourteen sessions, in the order they were measured."""
    b75 = json.loads((RAW / "B75_cold_start_20260910_v2.json").read_text())
    qualification = next(h for h in b75["history"] if h.get("probe") == "qualification")
    adoption = qualification["adoption"][CLASS]
    if qualification["blocked"] or not qualification["correctness_identical"] \
            or qualification["fallbacks"] or not qualification["resource_gate_passed"]:
        raise SystemExit("B75's qualification did not pass its own gates; it is not evidence")
    rows = [_row("B75_cold_start_single_short", b75["sealed_at"], adoption["candidate"],
                 adoption["reference_aa"], adoption["aa_half_width"],
                 _standard_error(adoption["candidate"]["ratios"]), context)]

    sessions = sorted((RAW / "B76_sessions_20260910").glob("session_*_result.json"))
    for path in sessions:
        result = json.loads(path.read_text())
        if not result["valid_for_learning"]:
            continue
        rows.append(_row(f"B76_session_{result['session']:02d}",
                         result["state_before"]["captured_at"], result["primary"],
                         result["aa"], result["aa_half_width"],
                         result["within_session_se"], context))
    return rows


def b69_row(context: IntakeContext) -> Evidence:
    """Valid historical evidence with a control too wide to set a scale."""
    b69 = json.loads((RAW / "B69_stack_proof_20260910.json").read_text())
    candidate = b69["comparisons"][CLASS]["candidate"]
    aa = b69["comparisons"][CLASS]["reference_aa"]
    return _row("B69_stack_proof_single_short", b69["measured_at"], candidate, aa,
                (aa["ci_high"] - aa["ci_low"]) / 2,
                _standard_error(candidate["ratios"]), context)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal-dir", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    seal_dir = args.seal_dir or args.out.parent / (args.out.stem + "_steps")
    seal_dir.mkdir(parents=True, exist_ok=True)

    context, provenance = _context()
    rows = replay_evidence(context)
    learner = LocalLearner(context)

    steps = []
    before = learner.recommendation(ACTION, CLASS)
    steps.append({"step": 0, "evidence_id": None, "accepted": None,
                  "evidence_count": 0, **before.as_dict()})
    write_once(seal_dir / "step_00_no_evidence.json", json.dumps(
        {"step": 0, "note": "before any evidence exists", **before.as_dict()},
        indent=2, sort_keys=True))

    for index, row in enumerate(rows, start=1):
        intake = learner.observe(row)
        recommendation = learner.recommendation(ACTION, CLASS)
        step = {"step": index, "evidence_id": row.evidence_id,
                "measured_at": row.measured_at,
                "accepted": intake["accepted"],
                "used_for_gain_distribution": intake.get("used_for_gain_distribution"),
                "evidence_ratio": row.ratio,
                "evidence_interval": [row.ci_low, row.ci_high],
                "aa_median": row.aa_median, "aa_half_width": row.aa_half_width,
                **recommendation.as_dict()}
        steps.append(step)
        write_once(seal_dir / f"step_{index:02d}_{row.evidence_id}.json",
                   json.dumps({**step, "note": ("sealed after this evidence and before the "
                                                "next was offered")},
                              indent=2, sort_keys=True, default=str))

    # -- the checks the verdict is made of ------------------------------------
    qualified_at = next((s["step"] for s in steps
                         if s["local_learning_state"] == CANDIDATE_QUALIFIED), None)
    too_early = [s for s in steps
                 if s["recommended_action"] == CANDIDATE and s["evidence_count"] < MIN_SESSIONS]
    wrong_action = [s for s in steps
                    if s["recommended_action"] == CANDIDATE
                    and s.get("evidence_interval") is not None
                    and s["evidence_interval"][0] > 1.0]
    before_qualification = [s for s in steps
                            if qualified_at is not None and s["step"] < qualified_at]
    phases_correct = (steps[0]["local_learning_state"] == UNKNOWN
                      and all(s["local_learning_state"] in (UNKNOWN, COLLECTING)
                              and s["recommended_action"] == REFERENCE
                              for s in before_qualification))

    # -- persistence round trip ----------------------------------------------
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "local_learning.json"
        learner.save(state_path)
        restored = LocalLearner.restore(state_path, context, rows)
        round_trip = {
            "saved": True,
            "restored_state": restored.recommendation(ACTION, CLASS).local_learning_state,
            "identical": (restored.recommendation(ACTION, CLASS).as_dict()
                          == learner.recommendation(ACTION, CLASS).as_dict()),
            "digest_matches": (restored.as_dict()["digest"] == learner.as_dict()["digest"]),
            "missing_state_falls_back": (
                LocalLearner.restore(Path(directory) / "absent.json", context)
                .recommendation(ACTION, CLASS).recommended_action),
        }
        corrupt = Path(directory) / "corrupt.json"
        corrupt.write_text("{not json", encoding="utf-8")
        round_trip["corrupt_state_falls_back"] = (
            LocalLearner.restore(corrupt, context).recommendation(ACTION, CLASS)
            .recommended_action)

    # -- B69, the inclusion rule, demonstrated rather than asserted -----------
    b69 = b69_row(context)
    with_b69 = LocalLearner(context)
    with_b69.observe_all([*rows, b69])
    b69_intake = next(r for r in with_b69.as_dict()["actions"]
                      if r["action_id"] == ACTION)
    inclusion = {
        "b69_ratio": b69.ratio,
        "b69_aa_median": b69.aa_median, "b69_aa_half_width": b69.aa_half_width,
        "b69_control_is_readable": b69.aa_is_readable,
        "accepted_for_preference": b69.evidence_id in b69_intake["evidence_ids"],
        "excluded_from_gain_distribution": (
            b69.evidence_id in b69_intake["expected_gain_distribution"]
            ["excluded_for_a_noisy_control"]),
        "gain_distribution_unchanged": (
            b69_intake["expected_gain_distribution"]["running_mean"]
            == next(r for r in learner.as_dict()["actions"]
                    if r["action_id"] == ACTION)["expected_gain_distribution"]["running_mean"]),
        "rule": INCLUSION_RULE,
    }

    final = learner.recommendation(ACTION, CLASS)
    passed = (not too_early and not wrong_action and phases_correct
              and qualified_at is not None
              and round_trip["identical"] and round_trip["digest_matches"]
              and round_trip["missing_state_falls_back"] == REFERENCE
              and round_trip["corrupt_state_falls_back"] == REFERENCE
              and inclusion["accepted_for_preference"]
              and inclusion["excluded_from_gain_distribution"]
              and inclusion["gain_distribution_unchanged"])

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "source_binding": {
            name: _digest(PROJECT_ROOT / name)
            for name in ("ironmule/local_learner.py", "ironmule/router.py",
                         "tools/b78_cold_start_replay.py")},
        "evidence_sources": ["B75_cold_start_20260910_v2.json",
                             "B76_sessions_20260910/session_*_result.json"],
        "evidence_rows": len(rows),
        "steps": steps,
        "qualified_at_step": qualified_at,
        "sessions_needed_to_qualify": (steps[qualified_at]["evidence_count"]
                                       if qualified_at is not None else None),
        "never_candidate_too_early": not too_early,
        "no_wrong_action": not wrong_action,
        "phases_correct": phases_correct,
        "persistence": round_trip,
        "b69_inclusion": inclusion,
        "final_state": final.as_dict(),
        "final_profile": learner.as_dict(),
        "replay_passed": bool(passed),
        "nothing_activated": ("shadow only. No RouteDecision is changed, no kernel "
                              "activated, no default moved, nothing committed or pushed"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({
        "replay_passed": passed,
        "evidence_rows": len(rows),
        "qualified_at_step": qualified_at,
        "sessions_needed_to_qualify": record["sessions_needed_to_qualify"],
        "never_candidate_too_early": not too_early,
        "phases_correct": phases_correct,
        "persistence": round_trip,
        "b69_inclusion": {k: inclusion[k] for k in
                          ("b69_aa_half_width", "b69_control_is_readable",
                           "accepted_for_preference", "excluded_from_gain_distribution",
                           "gain_distribution_unchanged")},
        "final": final.as_dict(),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
