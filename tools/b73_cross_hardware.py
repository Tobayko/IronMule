#!/usr/bin/env python3
"""The first cross-hardware test, and the gate that refuses to let it run here.

Everything `B71`, `B72` and `B74` built rests on one machine. `B71` said so, `B72` measured
that its hardware block never varies and therefore teaches nothing, and `B74` showed the
action-cost task is well posed while showing nothing about hardware. The question none of
them could ask is the only one that matters for a self-characterising runtime: on a Mac
nobody has tuned, does a few seconds of probing predict what hours of stack measurement will
find, or does the system honestly say it does not know?

**This file refuses to answer that on a machine it was trained on.** The first thing it does
is compare the running fingerprint against the frozen model's training fingerprints, and it
stops if they match. That is not a formality: a prediction made on the training machine
would look excellent and would mean nothing.

**The prediction is sealed before the ground truth exists.** `predict` writes its answer
write-once and then stops. `ground-truth` runs `B69`'s harness. `verdict` compares them. The
order is enforced by the files: the verdict step refuses if the prediction file was written
after the ground-truth file.

**Nothing is retrained before the verdict.** The model is read from the freeze, coefficients
and all. Learning from the new machine happens afterwards, in its own step, and never
overwrites the prediction that was made without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = PROJECT_ROOT / "research" / "raw"
FROZEN = RAW / "B73_frozen_model_20260910.json"

#: The one question. Reference against the intervention `B69` confirmed here, on the shape
#: it was confirmed on, in the classes whose confirmed stack runs ungrouped decode.
DECISION = {
    "actions": ["reference_stack", "k3840_geometry_sg4_r8"],
    "metric": "complete_stack_wall_time_ratio",
    "model_id": "mlx-community/gemma-3-12b-it-4bit",
    "workload_classes": ["single_short", "single_long", "session_warm"],
    "objective": "latency",
    "shape": {"k": 3840, "decode_width": 1},
}
#: Which measured relations the decision needs. Missing one of these is what a deep probe is
#: for, and it is the only thing a deep probe may be run for.
DECISION_RELEVANT_FEATURES = ("geometry_4_8_ratio", "cache_to_dram_ratio",
                              "m16_cost_per_row_vs_m1")


class NotEligible(SystemExit):
    pass


def frozen_model() -> dict:
    return json.loads(FROZEN.read_text())


def eligibility() -> dict:
    """Is this machine actually unknown? Everything else depends on the answer being yes."""
    from ironmule.hw import fingerprint, installed_memory_bytes, static_facts
    import mlx.core as mx
    import mlx_lm

    facts = static_facts()
    here = fingerprint()
    trained = frozen_model()["training_fingerprints"]
    return {
        "hardware_fingerprint": here,
        "soc": facts.get("chip"),
        "gpu_architecture": mx.device_info().get("architecture"),
        "gpu_cores": facts.get("gpu_cores"),
        "unified_memory_bytes": installed_memory_bytes(),
        "os_release": facts.get("os_release"),
        "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
        "training_fingerprints": trained,
        "is_unknown_machine": here not in trained,
        "refusal": ("" if here not in trained else
                    f"fingerprint {here} is a training fingerprint of the frozen model. A "
                    "prediction made here would be a memory, not a forecast, and would look "
                    "excellent for that reason. B73 needs a machine this model has never "
                    "seen"),
    }


def require_eligible() -> dict:
    row = eligibility()
    if not row["is_unknown_machine"]:
        print(json.dumps({"verdict": "B73_NOT_STARTED", "reason": row["refusal"]}, indent=2))
        raise NotEligible(3)
    return row


def predict(vector: dict, machine: dict) -> dict:
    """The frozen model's answer for the one decision, with its own fail-closed policy."""
    model = frozen_model()
    relations = {name: row["value"] for name, row in vector["relations"].items()}
    missing = [name for name in DECISION_RELEVANT_FEATURES if name not in relations]

    beta = np.asarray(model["coefficients"])
    mean = np.asarray(model["feature_mean"])
    scale = np.asarray(model["feature_scale"])
    names = model["feature_names"]

    def encode(action: str, workload_class: str) -> np.ndarray:
        row = np.zeros(len(names))
        for index, name in enumerate(names):
            if name == f"action={action}" or name == f"class={workload_class}":
                row[index] = 1.0
            elif name == f"model={DECISION['model_id']}":
                row[index] = 1.0
            elif name == f"metric={DECISION['metric']}":
                row[index] = 1.0
            elif name == "workload.k":
                row[index] = float(DECISION["shape"]["k"])
            elif name == "workload.decode_width":
                row[index] = float(DECISION["shape"]["decode_width"])
            elif name.endswith(".missing"):
                base = name[: -len(".missing")]
                if base.startswith("hardware."):
                    row[index] = 0.0 if base[len("hardware."):] in relations else 1.0
                elif base in ("workload.k", "workload.decode_width"):
                    row[index] = 0.0
                else:
                    row[index] = 1.0
            elif name.startswith("hardware."):
                key = name[len("hardware."):]
                row[index] = float(relations.get(key, 0.0))
        return row

    answers = {}
    for workload_class in DECISION["workload_classes"]:
        costs = {}
        for action in DECISION["actions"]:
            encoded = encode(action, workload_class)
            standard = np.append((encoded - mean) / scale, 1.0)
            costs[action] = float(standard @ beta)
        # Uncertainty: the frozen residual, widened by how far this machine's encoded row
        # sits from anything the model was fitted on. Unknown hardware is far by definition.
        widened = model["residual_spread"] * (1.0 + len(missing))
        best = min(costs, key=costs.get)
        reasons = []
        if missing:
            reasons.append(f"decision-relevant features missing: {missing}")
        if widened > model["policy_thresholds"]["max_usable_half_width"]:
            reasons.append(f"interval half width {widened:.4f} exceeds "
                           f"{model['policy_thresholds']['max_usable_half_width']}")
        if not machine["is_unknown_machine"]:
            reasons.append("this is a training machine")
        # Out of distribution by construction: no training row carries this fingerprint.
        reasons.append("the hardware block never varied in training, so no coefficient on "
                       "it was fitted against variation and it cannot be trusted on an "
                       "unseen machine")
        answers[workload_class] = {
            "predicted_cost": {a: c for a, c in costs.items()},
            "predicted_best_action": best,
            "uncertainty_half_width": widened,
            "abstained": bool(reasons),
            "reason": "; ".join(reasons),
        }
    return {"answers": answers, "missing_decision_relevant_features": missing,
            "relations_used": relations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("check", "predict", "ground-truth", "verdict"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--vector", type=Path, help="a B71 quick characterization record")
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--ground-truth-record", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.step == "check":
        row = eligibility()
        print(json.dumps(row, indent=2))
        if args.out:
            write_once(args.out, json.dumps(
                {"experiment": "B73_eligibility", "checked_at":
                 datetime.now(timezone.utc).isoformat(), "machine": row,
                 "frozen_model": {"path": "research/raw/" + FROZEN.name,
                                  "digest": hashlib.sha256(FROZEN.read_bytes()).hexdigest()},
                 "verdict": "ELIGIBLE" if row["is_unknown_machine"] else "B73_NOT_STARTED"},
                indent=2, sort_keys=True))
        return 0 if row["is_unknown_machine"] else 3

    if args.step == "predict":
        machine = require_eligible()
        if not args.vector or not args.out:
            raise SystemExit("predict needs --vector and --out")
        record = json.loads(args.vector.read_text())
        answer = predict(record["vector"], machine)
        write_once(args.out, json.dumps({
            "experiment": "B73_sealed_prediction",
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "machine": machine,
            "decision": DECISION,
            "frozen_model_digest": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
            "vector_source": args.vector.name,
            "vector": record["vector"],
            "prediction": answer,
            "sealed_before_ground_truth": True,
            "no_retraining_happened": ("the coefficients come from the freeze and were not "
                                       "touched. Learning from this machine is a later step "
                                       "and never overwrites this file"),
            "environment": {"platform": platform.platform()},
        }, indent=2, sort_keys=True, default=str))
        print(json.dumps({"sealed": str(args.out),
                          "abstained": all(a["abstained"] for a in answer["answers"].values()),
                          "missing": answer["missing_decision_relevant_features"]}, indent=2))
        return 0

    if args.step == "ground-truth":
        require_eligible()
        print("run tools/b69_stack_proof.py on this machine, unchanged, and pass its record "
              "to the verdict step. It carries its own AB/BA blocks, A/A arm, per-projection "
              "byte check and B65 resource gate, and nothing here may alter them.")
        return 0

    machine = eligibility()
    if not args.prediction or not args.ground_truth_record or not args.out:
        raise SystemExit("verdict needs --prediction, --ground-truth-record and --out")
    prediction = json.loads(args.prediction.read_text())
    truth = json.loads(args.ground_truth_record.read_text())
    if args.prediction.stat().st_mtime > args.ground_truth_record.stat().st_mtime:
        raise SystemExit("the prediction is newer than the ground truth; it was not sealed "
                         "first and no verdict may be drawn from it")
    rows = {}
    for name, row in truth.get("comparisons", {}).items():
        answer = prediction["prediction"]["answers"].get(name)
        if answer is None:
            continue
        measured = row["candidate"]["median"]
        predicted = answer["predicted_cost"]["k3840_geometry_sg4_r8"]
        rows[name] = {
            "abstained": answer["abstained"],
            "predicted_ratio": predicted, "measured_ratio": measured,
            "sign_correct": (predicted < 1.0) == (measured < 1.0),
            "absolute_error": abs(predicted - measured),
            "interval_contains_truth":
                abs(predicted - measured) <= answer["uncertainty_half_width"],
            "best_action_correct":
                (answer["predicted_best_action"] == "k3840_geometry_sg4_r8") == (measured < 1.0),
            "regret": max(0.0, measured - min(measured, 1.0)) if not answer["abstained"] else None,
        }
    abstained = all(a["abstained"] for a in prediction["prediction"]["answers"].values())
    gate_ok = truth.get("resource_gate", {}).get("passed") and truth.get(
        "correctness", {}).get("identical")
    if not gate_ok:
        verdict = "B73_INVALID"
    elif abstained:
        verdict = "B73_SAFE_ABSTAIN"
    elif all(r["sign_correct"] and r["best_action_correct"] for r in rows.values()):
        verdict = "B73_TRANSFER_SIGNAL"
    else:
        verdict = "B73_TRANSFER_FAIL"
    write_once(args.out, json.dumps({
        "experiment": "B73_verdict", "verdict": verdict,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "machine": machine, "per_class": rows,
        "prediction_was_sealed_first": True,
        "h1": ("two machines settle a sign, not a magnitude, and no generalisation claim "
               "follows from either outcome"),
    }, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "per_class": rows}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
