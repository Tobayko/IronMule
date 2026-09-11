#!/usr/bin/env python3
"""What would actually have run, at every point of this machine's measured history.

`B78` replayed the *controller* through `B75`'s and `B76`'s evidence and checked what it
would have recommended. `B79` asks the harder version of the same question: with activation
opted in, what would the dispatch have *done*? A recommendation that nobody reads cannot hurt
anyone. An activation that fires one session too early can.

So the whole history is walked forward again, and at each step a fresh activation layer is
built from the controller as it stood at that moment -- the same thing a runtime restarted at
that instant would have had -- and asked for its effective action. No future evidence reaches
an earlier step, which is what makes this a test rather than a reconstruction.

Nothing is measured and no model is loaded. `prepare()` admits without installing, so the
decision path runs with nothing behind it to execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = PROJECT_ROOT / "research" / "raw"

from b78_cold_start_replay import ACTION, CLASS, _context, replay_evidence
from ironmule.activation import (ActivationContext, CANDIDATE, LearnedDispatchActivation,
                                 REFERENCE)
from ironmule.local_learner import CANDIDATE_QUALIFIED, COLLECTING, LocalLearner, UNKNOWN

ROUTE = "interactive"
OTHER_ROUTES = ("throughput", "reference")
OTHER_CLASSES = ("single_long", "session_warm", "")

PREREGISTRATION = {
    "experiment": "B79_preflight_replay",
    "question": ("with activation opted in, would a runtime started at any point of B75's "
                 "and B76's measured history have dispatched the candidate exactly when the "
                 "controller qualified it, and the reference at every other point"),
    "no_new_measurement": "no model is loaded, no GPU touched, nothing is timed",
    "no_future_evidence": ("each step sees only the evidence that existed at that moment. A "
                           "fresh activation layer is built per step, which is what a "
                           "restart at that instant would have had"),
    "expectation": {"UNKNOWN": REFERENCE, "COLLECTING": REFERENCE,
                    "CANDIDATE_QUALIFIED": CANDIDATE, "REFERENCE_ONLY": REFERENCE},
    "also_checked": ("every other route and every workload class the controller has not "
                     "qualified must stay on the reference at every step, including after "
                     "qualification"),
}


def _activation(learner: LocalLearner, context: ActivationContext, enabled: bool):
    activation = LearnedDispatchActivation(learner, context, enabled=enabled)
    activation.prepare()
    return activation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    intake, provenance = _context()
    rows = replay_evidence(intake)
    context = ActivationContext(
        hardware_fingerprint=intake.hardware_fingerprint,
        gpu_architecture=intake.gpu_architecture, model_id=intake.model_id,
        model_identity_sha256=intake.model_identity_sha256,
        model_revision=intake.model_revision,
        quantization_bits=4, quantization_group_size=64, hidden_size=3840,
        projection_widths=(15360,), mlx=intake.mlx, mlx_lm=intake.mlx_lm)

    learner = LocalLearner(intake)
    steps = []
    for index in range(len(rows) + 1):
        if index:
            learner.observe(rows[index - 1])
        state = learner.recommendation(ACTION, CLASS)
        opted_in = _activation(learner, context, enabled=True)
        opted_out = _activation(learner, context, enabled=False)
        record = opted_in.for_dispatch(CLASS, ROUTE)
        expected = PREREGISTRATION["expectation"][state.local_learning_state]
        elsewhere = {
            f"{workload}|{route}": _activation(learner, context, enabled=True)
            .for_dispatch(workload, route).effective_action
            for route in (ROUTE, *OTHER_ROUTES) for workload in (CLASS, *OTHER_CLASSES)
            if not (workload == CLASS and route == ROUTE)}
        steps.append({
            "step": index,
            "evidence_id": rows[index - 1].evidence_id if index else None,
            "measured_at": rows[index - 1].measured_at if index else None,
            "controller_state": state.local_learning_state,
            "evidence_count": state.evidence_count,
            "effective_action": record.effective_action,
            "expected_action": expected,
            "correct": record.effective_action == expected,
            "opted_out_effective_action":
                opted_out.for_dispatch(CLASS, ROUTE).effective_action,
            "activation_record": record.as_dict(),
            "other_contexts": elsewhere,
            "other_contexts_all_reference": all(value == REFERENCE
                                                for value in elsewhere.values()),
        })

    first_candidate = next((s["step"] for s in steps
                            if s["effective_action"] == CANDIDATE), None)
    first_qualified = next((s["step"] for s in steps
                            if s["controller_state"] == CANDIDATE_QUALIFIED), None)
    checks = {
        "every_step_matches_its_expectation": all(s["correct"] for s in steps),
        "candidate_never_before_qualification": (
            first_candidate is not None and first_qualified is not None
            and first_candidate == first_qualified),
        "opting_out_is_always_reference": all(
            s["opted_out_effective_action"] == REFERENCE for s in steps),
        "no_other_route_or_class_ever_activates": all(
            s["other_contexts_all_reference"] for s in steps),
        "the_first_step_has_no_evidence": (steps[0]["controller_state"] == UNKNOWN
                                           and steps[0]["effective_action"] == REFERENCE),
    }
    passed = all(checks.values())

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "context": provenance,
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/activation.py", "ironmule/qmv_variant.py",
                         "ironmule/local_learner.py", "ironmule/router.py",
                         "tools/b79_preflight_replay.py")},
        "evidence_rows": len(rows),
        "steps": steps,
        "first_qualified_step": first_qualified,
        "first_candidate_step": first_candidate,
        "checks": checks,
        "preflight_passed": bool(passed),
        "nothing_activated": ("this tool loads no model and installs no kernel. It exercises "
                              "the decision path only"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"preflight_passed": passed, "checks": checks,
                      "first_qualified_step": first_qualified,
                      "first_candidate_step": first_candidate,
                      "actions": [f"{s['step']}:{s['controller_state']}"
                                  f"->{s['effective_action']}" for s in steps]},
                     indent=2, default=str))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
