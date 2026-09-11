#!/usr/bin/env python3
"""Feature rows for a later cost model. Data only: no decision, no policy, no training.

`B72` is meant to learn `hardware + workload + action -> expected cost` offline. This
exports the rows it would learn from, out of measurements this project already qualified,
and it exports nothing else. There is no model here, nothing is fitted, and no row carries a
recommendation.

**Only qualified evidence.** A run whose verdict was `BLOCKED`, `EXPLORATORY_INVALID` or
`NOT_STARTED` contributes nothing. `B57`'s two blocked `12B` confirmations, `B66`'s forced
exploratory run and its confounded first cache arm are all excluded by that rule, and each
exclusion is listed rather than silently dropped.

**Every row is traceable.** `evidence_id` names the raw record, `validity` names the verdict
that record carried, and `uncertainty` is whatever that record measured -- an interval where
one exists, a half range where one exists, and `None` where the source reported neither. No
row invents an uncertainty it does not have.
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
SCHEMA = "ironmule.cost_dataset.v1"
ROW_FIELDS = ("hardware_features", "workload_features", "action", "measured_cost",
              "uncertainty", "evidence_id", "validity")

EXCLUDED = [
    {"record": "B57_stack_composition_20260910_12b_confirmation.json", "verdict": "BLOCKED",
     "why": "the resource gate blocked it; a blocked run is not a measurement"},
    {"record": "B57_stack_composition_20260910_12b_confirmation_b63.json", "verdict": "BLOCKED",
     "why": "the same gate, the one authorised repetition"},
    {"record": "B66_FORCED_LOAD_EXPLORATORY_20260910.json", "verdict": "EXPLORATORY_INVALID",
     "why": "swap grew 3.40 GB during its first child"},
    {"record": "B66_stack_proof_20260910_not_started.json", "verdict": "NOT_STARTED",
     "why": "no data to export"},
    {"record": "B66_axes_*_20260910.json cache axis", "verdict": "confounded",
     "why": ("the first cache arm compared one kernel behind its own eval with twelve "
             "behind one, so it measured a submission boundary. Its replacement is used")},
]


def _hardware(vector: dict) -> dict:
    """The hardware side of a row: static facts plus the relations that were measurable."""
    static = vector["static"]
    relations = {name: row["value"] for name, row in vector["relations"].items()}
    return {"hardware_fingerprint": static["hardware_fingerprint"],
            "chip": static["chip"], "gpu_architecture": static["gpu_architecture"],
            "gpu_cores": static["gpu_cores"],
            "unified_memory_bytes": static["unified_memory_bytes"],
            "mlx": static["mlx"], "mlx_lm": static["mlx_lm"],
            "relations": relations,
            "relations_missing": sorted(set(
                ["cache_to_dram_ratio", "k_unaligned_to_aligned_ratio",
                 "m16_cost_per_row_vs_m1", "m8_cost_per_row_vs_m4",
                 "eval_fixed_over_one_kernel", "geometry_4_8_ratio"]) - set(relations))}


def rows_from_b69(hardware: dict) -> list[dict]:
    """The one action measured against a confirmed complete stack."""
    record = json.loads((RAW / "B69_stack_proof_20260910.json").read_text())
    if record["verdict"] != "STACK_CONFIRMED":
        return []
    references = {"single_short": "A", "single_long": "A", "session_warm": "B"}
    out = []
    for name, row in record["comparisons"].items():
        for arm, action in (("candidate", "k3840_geometry_4_8"),
                            ("reference_aa", "reference_repeated")):
            entry = row[arm]
            out.append({
                "hardware_features": hardware,
                "workload_features": {"model_id": "mlx-community/gemma-3-12b-it-4bit",
                                      "workload_class": name,
                                      "reference_stack": references[name],
                                      "objective": "latency", "decode_width": 1},
                "action": action,
                "measured_cost": {"metric": "complete_stack_wall_time_ratio",
                                  "value": entry["median"],
                                  "lower_is_better": True},
                "uncertainty": {"kind": "bootstrap_95_ci",
                                "ci_low": entry["ci_low"], "ci_high": entry["ci_high"],
                                "blocks": entry["n"]},
                "evidence_id": "B69_stack_proof_20260910",
                "validity": "STACK_CONFIRMED",
            })
    return out


def rows_from_b66_axes(hardware: dict) -> list[dict]:
    """Isolated kernel responses. Labelled as isolated, because E5 measured that they travel badly."""
    out = []
    for label, path in (("12B", "B66_axes_12b_20260910.json"),
                        ("4B", "B66_axes_4b_20260910.json")):
        record = json.loads((RAW / path).read_text())
        axes = record["axes"]
        model = {"12B": "mlx-community/gemma-3-12b-it-4bit",
                 "4B": "mlx-community/gemma-3-4b-it-4bit"}[label]
        shape = axes["width"]["shape"]
        for arm, entry in axes["width"]["result"]["ratios"].items():
            out.append({
                "hardware_features": hardware,
                "workload_features": {"model_id": model, "workload_class": "isolated_kernel",
                                      "k": shape["k"], "n": shape["n"],
                                      "reference_width": 4},
                "action": f"grouped_width_{arm}",
                "measured_cost": {"metric": "kernel_wall_time_ratio_vs_width_4",
                                  "value": entry["median"], "lower_is_better": True},
                "uncertainty": {"kind": "bootstrap_95_ci", "ci_low": entry["ci_low"],
                                "ci_high": entry["ci_high"], "blocks": entry["n"]},
                "evidence_id": path.replace(".json", ""),
                "validity": "MEASURED_ISOLATED",
            })
        geometry = axes.get("geometry")
        if geometry and geometry.get("result"):
            for arm, entry in geometry["result"]["ratios"].items():
                out.append({
                    "hardware_features": hardware,
                    "workload_features": {"model_id": model,
                                          "workload_class": "isolated_kernel",
                                          "k": geometry["shape"]["k"],
                                          "n": geometry["shape"]["n"], "decode_width": 1},
                    "action": f"kernel_geometry_{arm}",
                    "measured_cost": {"metric": "kernel_wall_time_ratio_vs_library",
                                      "value": entry["median"], "lower_is_better": True},
                    "uncertainty": {"kind": "bootstrap_95_ci", "ci_low": entry["ci_low"],
                                    "ci_high": entry["ci_high"], "blocks": entry["n"]},
                    "evidence_id": path.replace(".json", ""),
                    "validity": "MEASURED_ISOLATED",
                })
    return out


def rows_from_b68(hardware: dict) -> list[dict]:
    """A closed axis is still data: the rows say the actions did not pay."""
    record = json.loads((RAW / "B68_command_buffer_sweep_20260910.json").read_text())
    if not record["resource_gate"]["passed"]:
        return []
    out = []
    for label, classes in record["analysis"].items():
        model = {"12B": "mlx-community/gemma-3-12b-it-4bit",
                 "4B": "mlx-community/gemma-3-4b-it-4bit"}[label]
        for name, row in classes.items():
            for configuration, entry in row["arms"].items():
                out.append({
                    "hardware_features": hardware,
                    "workload_features": {"model_id": model, "workload_class": name,
                                          "reference_stack": "confirmed",
                                          "objective": "latency"},
                    "action": f"mlx_command_buffer_{configuration}",
                    "measured_cost": {"metric": "complete_stack_wall_time_ratio",
                                      "value": entry["median"], "lower_is_better": True},
                    "uncertainty": {"kind": "bootstrap_95_ci", "ci_low": entry["ci_low"],
                                    "ci_high": entry["ci_high"], "blocks": entry["n"]},
                    "evidence_id": "B68_command_buffer_sweep_20260910",
                    "validity": "MEASURED_DEFAULT_WINS",
                })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vector", type=Path,
                        default=RAW / "B71_vector_m1max_20260910_v3.json")
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    vector_record = json.loads(args.vector.read_text())
    hardware = _hardware(vector_record["vector"])
    rows = (rows_from_b69(hardware) + rows_from_b66_axes(hardware) + rows_from_b68(hardware))
    for row in rows:
        assert set(row) == set(ROW_FIELDS), sorted(set(row) ^ set(ROW_FIELDS))

    record = {
        "schema": SCHEMA,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("rows for an offline cost model: hardware plus workload plus action to "
                    "expected cost. Nothing is trained here and no row is a recommendation"),
        "row_fields": list(ROW_FIELDS),
        "hardware_vector_source": str(args.vector.relative_to(PROJECT_ROOT)),
        "hardware_vector_verdict": vector_record["verdict"],
        "one_machine": ("every row carries the same hardware_features, because there is one "
                        "machine. A model fitted on this alone can only learn the workload "
                        "and action axes, and would be unable to tell a hardware effect "
                        "from a constant. That is a property of the data, not a flaw in the "
                        "schema, and it is why a second machine is the next thing needed"),
        "excluded_sources": EXCLUDED,
        "counts": {"rows": len(rows),
                   "by_validity": {v: sum(1 for r in rows if r["validity"] == v)
                                   for v in sorted({r["validity"] for r in rows})},
                   "by_action_kind": {k: sum(1 for r in rows if r["action"].startswith(k))
                                      for k in sorted({r["action"].split("_")[0] for r in rows})}},
        "rows": rows,
        "source_binding": hashlib.sha256(
            Path("tools/b71_dataset.py").read_bytes()).hexdigest(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"rows": len(rows), "by_validity": record["counts"]["by_validity"],
                      "excluded": len(EXCLUDED)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
