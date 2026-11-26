"""Compare the frozen cost head and two constant rules on complete, held-out projection geometries."""

import argparse
import json
from pathlib import Path
from statistics import median

from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save


def replay(policy):
    fit = json.loads((policy / "fit.json").read_bytes())
    if checksum(policy / "pilot.npz") != fit["weights_sha256"]:
        raise ValueError("weights and fit refer to different training runs")
    rows = [row for row in fit["rows"] if row["set"] == "test"]
    train_shapes = {tuple(r["shape"][1:3]) for r in fit["rows"] if r["set"] != "test"}
    if not rows or any(tuple(r["shape"][1:3]) in train_shapes for r in rows):
        raise ValueError("held-out projections overlap the training or calibration set")
    strategies = {
        "stock": lambda r: "stock",
        "fp8": lambda r: "fp8",
        "random_initialization": lambda r: r["random_profile"],
        "cpu_pilot": lambda r: r["trained_profile"],
    }
    result = {}
    for name, select in strategies.items():
        totals = [r["median_us"][select(r)] for r in rows]
        optimum = [min(r["median_us"].values()) for r in rows]
        result[name] = {
            "sum_operator_us": sum(totals),
            "sum_best_observed_us": sum(optimum),
            "relative_to_best_measured": sum(totals) / sum(optimum),
            "fastest_choices": sum(
                a == b for a, b in zip(totals, optimum, strict=True)
            ),
            "choices": len(rows),
        }
    return {
        "weights": fit["weights_sha256"],
        "strategies": result,
        "switched_after_training": sum(
            r["trained_profile"] != r["random_profile"] for r in rows
        ),
        "median_cpu_decision_us": median(r["decision_us"] for r in rows),
        "scope": "Each held-out geometry/width is weighted once. The comparison is of operators before system qualification.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    save(a.out, replay(a.policy))
