#!/usr/bin/env python3
"""Rows that may be ranked against each other, and rows that may not, said out loud.

`B72` tried to score `B71`'s export and found two ways it could not be scored. Some action
families compare unequal work — the grouped-width rows report total time across `M = 1` to
`16`, so ranking them picks the smallest batch and means nothing — and every study owns its
own action names, so a study-level split leaves the held-out actions unseen by construction.

This repairs the data contract and nothing else. No model is trained, no hardware is
measured, no route changes. Every row gains the four things a ranking needs: which comparison
it belongs to, which actions it is allowed to compete with, how much work it did, and the
cost normalised by that work — or an explicit refusal where the evidence does not carry the
work unit.

**Two rules do most of the work.** An action may only compete with actions that fulfil the
same semantic order, deliver the same output work, carry the same correctness contract and
sit on the same model, shape, class and objective. And a normalisation is only applied where
the source record already measured the unit it needs. Where it did not, the row is
`NOT_COMPARABLE` and stays in the dataset as context rather than being quietly deleted.

**Canonical action names.** `B66` measured the `(4, 8)` geometry against a library call and
`B69` measured the same intervention against a confirmed stack. Those are two costs of one
action, and naming them differently hid that. They now share a name, and the two costs stay
apart because they carry different metrics and different references.
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
SOURCE = RAW / "B72_cost_dataset_20260910.json"
SCHEMA = "ironmule.cost_dataset.v2"
EXCLUDED_NOTE = [
    {"record": "the grouped-width family", "verdict": "NOT_A_DECISION_SET",
     "why": ("cost per row is a valid unit, but a dispatch cannot choose a width "
             "larger than the number of ready requests, so width is not a free "
             "action and is never ranked")},
]

#: One intervention, one name, whatever study measured it. `B66` timed the `(4, 8)` geometry
#: against a library call and `B69` timed it inside a stack; hiding that behind two names is
#: what let `B72`'s holdout action look unseen when it had in fact been measured before.
CANONICAL_ACTIONS = {
    "kernel_geometry_sg4_r8": "k3840_geometry_sg4_r8",
    "k3840_geometry_4_8": "k3840_geometry_sg4_r8",
    "kernel_geometry_library_aa": "reference_repeated",
    "reference_repeated": "reference_repeated",
}

#: Which metric families rank unequal work, and what unit the evidence offers instead.
WORK_UNITS = {
    "complete_stack_wall_time_ratio": {
        "work_units": "one dispatch of the class, token identical across actions",
        "equal_work": True,
        "normalization_method": "none needed: every action serves the same requests and "
                                "produces the same tokens, so the ratio is already per unit "
                                "of work",
    },
    "kernel_wall_time_ratio_vs_library": {
        "work_units": "one matvec at M=1 over the same buffers",
        "equal_work": True,
        "normalization_method": "none needed: every action computes the same output from the "
                                "same inputs and is byte identical to the library",
    },
    "kernel_wall_time_ratio_vs_width_4": {
        "work_units": "M rows, and M differs between the actions",
        "equal_work": False,
        "normalization_method": "cost per row, taken from the source record's own per_row "
                                "block rather than derived here",
    },
}


def canonical(action: str) -> str:
    return CANONICAL_ACTIONS.get(action, action)


def per_row_costs() -> dict:
    """The width family's own per-row numbers, read from where `B66` recorded them."""
    out = {}
    for label, path in (("12B", "B66_axes_12b_20260910.json"),
                        ("4B", "B66_axes_4b_20260910.json")):
        record = json.loads((RAW / path).read_text())
        for arm, row in record["axes"]["width"]["per_row"].items():
            out[(path.replace(".json", ""), f"grouped_width_{arm}")] = row["cost_per_row_vs_M4"]
    return out


def build(rows) -> tuple[list, list]:
    """Every row, annotated. Nothing is deleted; what cannot be ranked says so."""
    per_row = per_row_costs()
    annotated, excluded = [], []
    for row in rows:
        metric = row["measured_cost"]["metric"]
        rule = WORK_UNITS.get(metric)
        workload = row["workload_features"]
        action = canonical(row["action"])
        context = "|".join(str(x) for x in (
            row["evidence_id"], workload.get("model_id", ""),
            workload.get("workload_class", ""), metric,
            workload.get("k", ""), workload.get("n", "")))
        entry = dict(row)
        entry["action"] = action
        entry["original_action"] = row["action"]
        entry["comparison_context_id"] = context
        entry["work_units"] = rule["work_units"] if rule else None

        if rule is None:
            entry["normalized_cost"] = None
            entry["normalization_method"] = None
            entry["comparability"] = "NOT_COMPARABLE"
            entry["comparability_reason"] = f"no work unit is defined for metric {metric!r}"
        elif rule["equal_work"]:
            entry["normalized_cost"] = row["measured_cost"]["value"]
            entry["normalization_method"] = rule["normalization_method"]
            entry["comparability"] = "COMPARABLE"
            entry["comparability_reason"] = ""
        else:
            key = (row["evidence_id"], row["action"])
            cost = per_row.get(key)
            if cost is None:
                entry["normalized_cost"] = None
                entry["normalization_method"] = None
                entry["comparability"] = "NOT_COMPARABLE"
                entry["comparability_reason"] = (
                    "the actions do unequal work and the source record carries no per-row "
                    "cost for this arm")
            else:
                entry["normalized_cost"] = cost
                entry["normalization_method"] = rule["normalization_method"]
                # Per-row cost is a legitimate unit and still not a free choice: a dispatch
                # cannot pick width 16 unless sixteen requests are ready. The width family
                # therefore carries a normalised cost as a FEATURE and is not a decision set.
                entry["comparability"] = "NOT_A_DECISION_SET"
                entry["comparability_reason"] = (
                    "cost per row is a valid unit, but grouped width is not a free action: "
                    "a dispatch cannot choose a width larger than the number of ready "
                    "requests. B66 measured these as isolated kernel inputs, not as service "
                    "decisions, so they are kept as context and never ranked")
        annotated.append(entry)
        if entry["comparability"] != "COMPARABLE":
            excluded.append({"action": action, "context": context,
                             "comparability": entry["comparability"],
                             "reason": entry["comparability_reason"]})
    return annotated, excluded


def reference_rows(annotated) -> list:
    """The reference each set was measured against is itself an available action, at 1.0.

    Leaving it out would make every set a choice between alternatives to a baseline nobody
    could pick, which is not the decision anyone faces.
    """
    references = {
        "complete_stack_wall_time_ratio": "reference_stack",
        "kernel_wall_time_ratio_vs_library": "library_default",
    }
    out, seen = [], set()
    for row in annotated:
        if row["comparability"] != "COMPARABLE":
            continue
        context = row["comparison_context_id"]
        if context in seen:
            continue
        seen.add(context)
        name = references.get(row["measured_cost"]["metric"])
        if name is None:
            continue
        out.append({
            **{k: row[k] for k in ("hardware_features", "workload_features")},
            "action": name, "original_action": name,
            "comparison_context_id": context,
            "work_units": row["work_units"],
            "measured_cost": {"metric": row["measured_cost"]["metric"], "value": 1.0,
                              "lower_is_better": True},
            "normalized_cost": 1.0,
            "normalization_method": "the reference is 1.0 by definition of the ratio",
            "uncertainty": {"kind": "definitional", "ci_low": 1.0, "ci_high": 1.0,
                            "blocks": None},
            "evidence_id": row["evidence_id"], "validity": row["validity"],
            "comparability": "COMPARABLE", "comparability_reason": "",
            "is_reference_action": True,
        })
    return out


def coverage(sets, train, validation, holdout) -> dict:
    def actions(rows):
        return {r["action"] for r in rows}
    train_actions = actions(train)
    out = {}
    for context, rows in sets.items():
        where = ("holdout" if any(r in holdout for r in rows)
                 else "validation" if any(r in validation for r in rows) else "train")
        names = sorted({r["action"] for r in rows})
        unseen = sorted(set(names) - train_actions) if where != "train" else []
        known = sorted(set(names) & train_actions) if where != "train" else names
        out[context] = {
            "split": where, "actions": names, "n_actions": len(names),
            "known_action": known, "unseen_action": unseen,
            "insufficient_action_coverage": bool(unseen) or len(names) < 2,
            "usable_as_a_top1_test": where != "train" and not unseen and len(names) >= 2,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    source = json.loads(SOURCE.read_text())
    assert source["schema"] == "ironmule.cost_dataset.v1", source["schema"]
    rows = source["rows"]

    checks = {}
    checks["no_blocked_or_invalid"] = all(
        r["validity"] not in ("BLOCKED", "EXPLORATORY_INVALID", "NOT_STARTED") for r in rows)
    fingerprints = {json.dumps(r["hardware_features"], sort_keys=True) for r in rows}
    checks["single_hardware_fingerprint"] = len(fingerprints) == 1
    seen = set()
    duplicates = []
    for r in rows:
        key = (r["evidence_id"], r["action"],
               json.dumps(r["workload_features"], sort_keys=True))
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    checks["no_duplicate_evidence"] = not duplicates

    annotated, excluded = build(rows)
    annotated += reference_rows(annotated)

    sets: dict[str, list] = {}
    for row in annotated:
        if row["comparability"] == "COMPARABLE":
            sets.setdefault(row["comparison_context_id"], []).append(row)
    sets = {k: v for k, v in sets.items() if len(v) >= 2}

    checks["every_ranked_row_is_in_one_set"] = all(
        len({r["comparison_context_id"] for r in rows_}) == 1 for rows_ in sets.values())
    checks["every_set_shares_one_metric_and_work_unit"] = all(
        len({(r["measured_cost"]["metric"], r["work_units"]) for r in rows_}) == 1
        for rows_ in sets.values())
    checks["no_unequal_work_is_ranked"] = all(
        r["comparability"] != "COMPARABLE"
        for r in annotated
        if r["measured_cost"]["metric"] == "kernel_wall_time_ratio_vs_width_4")
    checks["normalization_reproducible"] = all(
        r["normalized_cost"] is not None or r["comparability"] != "COMPARABLE"
        for r in annotated)

    # The split. Study groups stay whole; the holdout is the study whose actions have the
    # best chance of having been seen elsewhere, which is the only thing that could make a
    # top-1 test possible at all here.
    holdout_studies = {"B69_stack_proof_20260910"}
    validation_studies = {"B66_axes_4b_20260910"}
    train = [r for r in annotated if r["evidence_id"] not in holdout_studies | validation_studies]
    validation = [r for r in annotated if r["evidence_id"] in validation_studies]
    holdout = [r for r in annotated if r["evidence_id"] in holdout_studies]
    checks["no_train_holdout_leak"] = not (
        {r["evidence_id"] for r in train} & {r["evidence_id"] for r in holdout})

    cover = coverage(sets, train, validation, holdout)
    usable = [c for c, row in cover.items() if row["usable_as_a_top1_test"]]

    if not all(checks.values()):
        verdict = "B74_FAIL"
        why = f"a quality check failed: {[k for k, v in checks.items() if not v]}"
    elif usable:
        verdict = "B74_PASS"
        why = (f"{len(usable)} comparison set(s) outside training rank only actions that "
               "were seen in training, on one metric and one work unit, with no study on "
               "both sides of the split")
    else:
        verdict = "B74_DATA_INSUFFICIENT"
        why = ("no comparison set outside training ranks only actions seen in training. "
               "Canonical naming put the (4, 8) geometry in both B66 and B69, but the "
               "reference action of every set is study-local, so each holdout set still "
               "contains at least one action training never saw. Study groups cannot be "
               "split further without leakage, and no synthetic rows were created")

    record = {
        "schema": SCHEMA,
        "experiment": "B74_comparable_action_sets",
        "verdict": verdict,
        "why": why,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(SOURCE.relative_to(PROJECT_ROOT)),
                   "schema": source["schema"], "rows": len(rows),
                   "digest": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "canonical_action_map": CANONICAL_ACTIONS,
        "work_units": WORK_UNITS,
        "quality_checks": checks,
        "comparable_sets": {
            context: {"actions": sorted(r["action"] for r in rows_),
                      "metric": rows_[0]["measured_cost"]["metric"],
                      "work_units": rows_[0]["work_units"],
                      "normalization_method": rows_[0]["normalization_method"],
                      "best_action": min(rows_, key=lambda r: r["normalized_cost"])["action"],
                      "best_normalized_cost": min(r["normalized_cost"] for r in rows_)}
            for context, rows_ in sets.items()},
        "n_comparable_sets": len(sets),
        "coverage": cover,
        "usable_top1_sets": usable,
        "excluded_rows": excluded,
        "split": {"train_rows": len(train), "validation_rows": len(validation),
                  "holdout_rows": len(holdout),
                  "train_studies": sorted({r["evidence_id"] for r in train}),
                  "validation_studies": sorted({r["evidence_id"] for r in validation}),
                  "holdout_studies": sorted({r["evidence_id"] for r in holdout})},
        "hardware_learning_limit": (
            "one fingerprint. Even at B74_PASS this shows only that the action-cost task is "
            "correctly defined. A cross-hardware test needs B73 on a machine nobody has "
            "tuned, and nothing here brings that closer"),
        "may_b72_be_replayed": bool(usable),
        "rows": annotated,
        "source_binding": hashlib.sha256(
            (PROJECT_ROOT / "tools" / "b74_comparable_dataset.py").read_bytes()).hexdigest(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))

    # A companion file in the shape B72 already reads, so the replay changes B72's input
    # path and nothing else: same models, same split rule, same abstain policy. Only rows
    # that may be ranked are in it, and their cost is the normalised one.
    if usable:
        companion = args.out.with_name(args.out.stem + "_v1shape.json")
        ranked = [r for r in annotated if r["comparability"] == "COMPARABLE"]
        write_once(companion, json.dumps({
            "schema": "ironmule.cost_dataset.v1",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "purpose": ("the B74 comparable rows, written in the v1 shape so B72 can be "
                        "replayed with its models, split rule and policy untouched"),
            "derived_from": str(args.out.name),
            "row_fields": ["hardware_features", "workload_features", "action",
                           "measured_cost", "uncertainty", "evidence_id", "validity"],
            "normalisation": ("measured_cost.value is B74's normalized_cost. Where the "
                              "actions did equal work that is the original ratio; nothing "
                              "with unequal work is in this file at all"),
            "excluded_sources": EXCLUDED_NOTE,
            "one_machine": ("every row carries one fingerprint, so no hardware feature "
                            "varies in training and nothing fitted on this can separate a "
                            "hardware effect from a constant. No hardware generalisation is "
                            "claimed and B71's H1 is not reported as tested"),
            "counts": {"rows": len(ranked)},
            "rows": [{"hardware_features": r["hardware_features"],
                      "workload_features": r["workload_features"],
                      "action": r["action"],
                      "measured_cost": {"metric": r["measured_cost"]["metric"],
                                        "value": r["normalized_cost"],
                                        "lower_is_better": True},
                      "uncertainty": r["uncertainty"],
                      "evidence_id": r["evidence_id"],
                      "validity": r["validity"]} for r in ranked],
        }, indent=2, sort_keys=True, default=str))
        print(f"companion written to {companion}")
    print(json.dumps({
        "verdict": verdict, "why": why[:200],
        "comparable_sets": len(sets),
        "usable_top1_sets": len(usable),
        "quality_checks": checks,
        "excluded": len(excluded),
        "may_replay_b72": bool(usable),
    }, indent=2))
    for context, row in sorted(cover.items()):
        print(f'  [{row["split"]:10}] {len(row["actions"]):2} actions  usable={row["usable_as_a_top1_test"]}'
              f'  unseen={row["unseen_action"][:2]}  {context[:70]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
