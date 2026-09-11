#!/usr/bin/env python3
"""What the cold start decided, against what this machine already knew but was not told.

`tools/b75_cold_start.py` produced its decision without opening a single historical record.
This opens them, and only now. The comparison is not a check on whether the geometry is
faster -- `B69` settled that here months of measurement ago in project time. It is a check on
whether a system starting from nothing reached the same conclusion, how long that took, and
whether it was ever confidently wrong on the way.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    sealed = json.loads(args.sealed.read_text())
    history = json.loads((RAW / "B69_stack_proof_20260910.json").read_text())
    first_attempt = json.loads((RAW / "B75_cold_start_20260910.json").read_text())

    qualification = next(h for h in sealed["history"] if h.get("probe") == "qualification")
    deep = next(h for h in sealed["history"] if h.get("probe") == "deep_geometry")
    local = qualification["adoption"]["single_short"]["candidate"]
    historical = history["comparisons"]["single_short"]["candidate"]

    overlap = not (local["ci_high"] < historical["ci_low"]
                   or historical["ci_high"] < local["ci_low"])
    decision = sealed["final_decision"]["decision"]
    historical_best_is_candidate = historical["ci_high"] < 1.0

    wrong_turns = [h for h in sealed["history"]
                   if h.get("step") == "decision" and h.get("decision") == "REFERENCE"]
    blocked = (qualification["blocked"] or not qualification["resource_gate_passed"]
               or not qualification["correctness_identical"])

    if blocked:
        verdict = "B75_INVALID"
    elif decision == "CANDIDATE" and historical_best_is_candidate:
        verdict = "B75_LOCAL_LEARNING_CONFIRMED"
    elif decision == "REFERENCE" and not historical_best_is_candidate:
        verdict = "B75_SAFE_REFERENCE"
    elif decision == "REFERENCE":
        verdict = "B75_SAFE_REFERENCE"
    else:
        verdict = "B75_LOCAL_LEARNING_FAIL"

    record = {
        "experiment": "B75_ground_truth",
        "verdict": verdict,
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "order_held": ("the sealed decision was written before this file opened any "
                       "historical record. Its digest is below and it is not modified here"),
        "sealed_decision": {"path": "research/raw/" + args.sealed.name,
                            "digest": hashlib.sha256(args.sealed.read_bytes()).hexdigest(),
                            "decision": decision,
                            "sealed_at": sealed["sealed_at"]},
        "correct_action": {
            "cold_start_chose": decision,
            "history_says_candidate_is_better": historical_best_is_candidate,
            "agrees": (decision == "CANDIDATE") == historical_best_is_candidate,
        },
        "magnitude": {
            "cold_start_deep_probe_isolated": {"median": deep["ratios"]["candidate"]["median"],
                                               "ci": [deep["ratios"]["candidate"]["ci_low"],
                                                      deep["ratios"]["candidate"]["ci_high"]]},
            "cold_start_local_qualification_stack": {"median": local["median"],
                                                     "ci": [local["ci_low"], local["ci_high"]],
                                                     "blocks": local["n"]},
            "historical_B69_stack": {"median": historical["median"],
                                     "ci": [historical["ci_low"], historical["ci_high"]],
                                     "blocks": historical["n"]},
            "intervals_overlap": overlap,
            "reading": (
                "both qualifications ran on this machine, both passed every gate, and their "
                "95 per cent intervals do not overlap: the cold start measured "
                f"{local['median']:.4f} [{local['ci_low']:.4f}; {local['ci_high']:.4f}] over "
                f"{local['n']} blocks and B69 measured {historical['median']:.4f} "
                f"[{historical['ci_low']:.4f}; {historical['ci_high']:.4f}] over "
                f"{historical['n']}. The SIGN agrees and the MAGNITUDE does not. That is a "
                "caution about how much any single confirmation's magnitude is worth on a "
                "machine whose load moves, and it is the more useful finding of the two"
                if not overlap else
                "the two independent qualifications agree within their intervals"),
        },
        "regret": {
            "value": 0.0 if (decision == "CANDIDATE" and historical_best_is_candidate) else None,
            "note": ("the cold start chose the action history also measures as better, so "
                     "its regret against the measured optimum is zero"),
        },
        "cost": sealed["cost"],
        "wrong_decisions_during_learning": {
            "in_the_sealed_run": len(wrong_turns),
            "in_the_first_attempt": {
                "record": "research/raw/B75_cold_start_20260910.json",
                "decision": first_attempt["final_decision"]["decision"],
                "what_went_wrong": (
                    "a defect in the learner, not in the machine: a fast-probe point "
                    "estimate carries no interval, and the decision logic read `no interval "
                    "below 1.0` as `no evidence for the candidate` and answered REFERENCE. "
                    "It reached a safe answer for a wrong reason and skipped the deep probe "
                    "the design exists to trigger. Fixed, and the first run is kept"),
                "was_it_dangerous": ("no. It fell back to the reference, which is the safe "
                                     "direction. A user would have lost the gain and nothing "
                                     "else"),
            },
        },
        "would_the_reference_fallback_have_protected_a_user": (
            "yes, in every branch this run could have taken. The reference is what a fresh "
            "install serves until it has earned something else, every failure mode observed "
            "here ends there, and the only cost of that safety is the gain itself"),
        "what_this_does_not_show": (
            "anything about another Mac. B75 is a cold start on hardware IronMule has "
            "measured before, in a state that was denied access to those measurements. "
            "Whether the knowledge it builds transfers is B73, and B73 has not run"),
        "historical_record_used": {
            "path": "research/raw/B69_stack_proof_20260910.json",
            "digest": hashlib.sha256(
                (RAW / "B69_stack_proof_20260910.json").read_bytes()).hexdigest(),
            "verdict": history["verdict"],
        },
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict,
                      "agrees": record["correct_action"]["agrees"],
                      "intervals_overlap": overlap,
                      "cold_start": [local["median"], local["ci_low"], local["ci_high"]],
                      "historical": [historical["median"], historical["ci_low"],
                                     historical["ci_high"]],
                      "time_to_useful_knowledge_s":
                          sealed["cost"]["time_to_useful_hardware_knowledge_seconds"]},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
