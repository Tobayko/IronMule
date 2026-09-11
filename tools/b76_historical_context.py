#!/usr/bin/env python3
"""`B69` and `B75` placed beside `B76`'s temporal distribution, and never pooled with it.

`B69` qualified the `(4, 8)` geometry at `0.8469` `[0.7580; 0.9488]` and `B75` qualified the
same intervention on the same machine at `0.9624` `[0.9563; 0.9630]`. The intervals do not
overlap. Two readings were possible and neither could be chosen from two runs: either single
confirmations here carry more between-session variation than their own intervals admit, or
one of the two studies met a machine state the other did not.

`B76` measured that between-session distribution directly. This file asks the only question
that needed it: are those two numbers ordinary draws from it, or is one of them an outlier
that points at a state neither run recorded?

**They are not pooled.** `B69` and `B75` used different block counts, different designs and a
different qualification scope. Averaging them into `B76` would manufacture a number no study
measured. Each is compared to `B76`'s distribution and reported on its own.

**This runs after `B76`'s verdict is sealed**, and `B76`'s own harness has no path to the
files opened here.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = PROJECT_ROOT / "research" / "raw"

PRIMARY_CLASS = "single_short"


def _position(value: float, ratios: list[float], mean: float, predictive_sd: float) -> dict:
    """Where one historical median falls in the temporal distribution B76 measured."""
    below = sum(1 for r in ratios if r < value)
    z = (value - mean) / predictive_sd if predictive_sd > 0 else None
    interval = [mean - 1.96 * predictive_sd, mean + 1.96 * predictive_sd]
    return {
        "median": value,
        "b76_sessions_below_it": below,
        "b76_sessions_total": len(ratios),
        "empirical_quantile": below / len(ratios) if ratios else None,
        "inside_b76_observed_range": bool(ratios and min(ratios) <= value <= max(ratios)),
        "b76_predictive_interval": interval,
        "inside_b76_predictive_interval": bool(interval[0] <= value <= interval[1]),
        "z_against_b76_predictive": z,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True,
                        help="the sealed B76 record; it must already carry a verdict")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    b76 = json.loads(args.sealed.read_text())
    if "verdict" not in b76:
        raise SystemExit("the B76 record carries no verdict; this tool runs after it is sealed")

    b69 = json.loads((RAW / "B69_stack_proof_20260910.json").read_text())
    b75 = json.loads((RAW / "B75_cold_start_20260910_v2.json").read_text())

    ratios = [row["ratio"] for row in b76["learning_history"]]
    if len(ratios) < 2:
        raise SystemExit("B76 produced too few valid sessions to place anything beside")
    decomposition = b76["variance_decomposition"]
    mean = statistics.fmean(ratios)
    tau_squared = decomposition.get("tau_squared_method_of_moments") or 0.0
    within = decomposition.get("mean_within_session_variance") or 0.0
    predictive_sd = math.sqrt(tau_squared + within)

    historical = {
        "B69_stack_proof": {
            "design": (f"{b69['complete_blocks']} blocks, three children each, three classes, "
                       "one arm per process"),
            **b69["comparisons"][PRIMARY_CLASS]["candidate"],
            "aa_median": b69["comparisons"][PRIMARY_CLASS]["reference_aa"]["median"],
        },
        "B75_cold_start": {},
    }
    qualification = next(h for h in b75["history"] if h.get("probe") == "qualification")
    adoption = qualification["adoption"][PRIMARY_CLASS]
    historical["B75_cold_start"] = {
        "design": f"{qualification['blocks']} blocks, three children each, one class",
        **adoption["candidate"],
        "aa_median": adoption["reference_aa"]["median"],
    }

    placed = {}
    for name, row in historical.items():
        position = _position(row["median"], ratios, mean, predictive_sd)
        overlap = not (row["ci_high"] < min(ratios) or row["ci_low"] > max(ratios))
        placed[name] = {
            **row, **position,
            "own_interval": [row["ci_low"], row["ci_high"]],
            "own_interval_overlaps_b76_observed_range": bool(overlap),
            "own_interval_half_width": (row["ci_high"] - row["ci_low"]) / 2,
        }

    # The question is not which study was right. It is whether B76's measured between-session
    # spread is large enough to contain both, or whether one sits outside it.
    outside = [name for name, row in placed.items()
               if not row["inside_b76_predictive_interval"]]
    both_inside = not outside
    if both_inside:
        reading = ("both historical medians are ordinary draws from the between-session "
                   "distribution B76 measured. Their disagreement is temporal variation that "
                   "neither single confirmation could see, and it is not evidence that either "
                   "study met a state the other did not")
    elif len(outside) == len(placed):
        reading = ("neither historical median lies inside B76's between-session distribution. "
                   "B76's own sessions do not explain either of them, which points at a state "
                   "difference between those studies and this one rather than at variation "
                   "within a single machine state")
    else:
        reading = (f"{outside[0]} lies outside the between-session distribution B76 measured "
                   f"and the other does not. That study met a machine state B76's sessions did "
                   f"not reproduce, and its magnitude should not be read as this machine's "
                   f"typical effect")

    record = {
        "experiment": "B76_historical_context",
        "runs_after": "the B76 verdict, which was sealed before this file was opened",
        "not_pooled": ("B69, B75 and B76 used different designs and scopes. No mean is taken "
                       "across them and none is offered"),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "b76": {
            "verdict": b76["verdict"],
            "valid_sessions": len(ratios),
            "session_ratios": ratios,
            "mean": mean,
            "observed_range": [min(ratios), max(ratios)],
            "between_session_sd_total": decomposition.get("between_session_sd_total"),
            "tau_sd": decomposition.get("tau_sd"),
            "mean_within_session_sd": decomposition.get("mean_within_session_sd"),
            "predictive_sd_used_here": predictive_sd,
            "aa_distribution": b76["aa_distribution"],
        },
        "placed": placed,
        "outside_b76_predictive_interval": outside,
        "reading": reading,
        "what_this_does_not_say": (
            "nothing about another Mac, and nothing causal. B76 randomised no machine state, "
            "so a study landing outside the distribution names an unexplained difference, not "
            "its cause"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"reading": reading, "outside": outside,
                      "b76_range": record["b76"]["observed_range"],
                      "b76_predictive_interval": placed[next(iter(placed))]
                      ["b76_predictive_interval"],
                      "placed": {k: {"median": v["median"],
                                     "quantile": v["empirical_quantile"],
                                     "z": v["z_against_b76_predictive"],
                                     "inside": v["inside_b76_predictive_interval"]}
                                 for k, v in placed.items()}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
