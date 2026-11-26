"""Regrade prospective calibration predictions and public runtime outputs on a CPU."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import median

from lab.pk1 import restart_trial
from lab.pk1.post_restart_verify import check_selection
from training.calibration_stability import CONTRACT, heldout, sha
from training.code_author import digest, write_once
from training.code_trial import read
from training.trial_release import require


def measured_return(field, raw, arm):
    prices = arm["startup_prices"][field]
    if field == "padding":
        values = {
            int(k): v for k, v in raw["direct_field_values"]["pad_prices"].items()
        }
        require(
            set(values) == set(range(1, arm["binding"]["max_rows"] + 1))
            and all(math.isfinite(v) and v > 0 for v in values.values()),
            "missing complete measured padding prices",
        )
        configured = {}
        for width in values:
            best = min(
                (larger for larger in values if larger >= width),
                key=lambda n: (values[n], n),
            )
            if best > width and values[best] < 0.95 * values[width]:
                configured[str(width)] = best
        require(
            configured == prices == raw["return_value"],
            "full price curve yields a different padding choice",
        )
    elif field == "passes":
        require(
            raw["return_value"]
            == {"object_type": "PassCosts", "fields": {"points": prices}}
            and len(prices) >= 2
            and all(
                type(n) is int and n > 0 and math.isfinite(t) and t > 0
                for n, t in prices
            )
            and sorted({n for n, _ in prices}) == [n for n, _ in prices],
            "incomplete pass-cost object",
        )
    elif field == "pilot":
        require(
            set(prices) == {"step_one", "row", "prefill_1k", "kv_1k"}
            and all(math.isfinite(t) and t > 0 for t in prices.values())
            and prices == raw["return_value"],
            "incomplete Pilot costs",
        )
    else:
        raise ValueError("unknown recorded constructor output")
    return prices


def replay(data, release, checkout):
    limits = read(CONTRACT)
    forecast_path = data / "prediction.json"
    forecast = read(forecast_path)
    sources = [data / f"{s}.json" for s in limits["seeds"]]
    control = release / "control.json"
    weights = release / "startup-head.npz"
    judged = heldout(forecast_path, sources, control, weights, checkout)
    require(
        judged == read(data / "holdout.json"), "archived holdout does not reproduce"
    )
    receipt = read(control)
    bound = receipt["qualification"]["binding"]
    historical = []
    for item in forecast["historical_rows"]:
        src = release / item["name"]
        require(sha(src) == item["sha256"], "historical calibration row differs")
        historical.append(read(src))
    observed = [read(p) for p in sources]
    live = []
    for row in observed:
        arm = row["arm"]
        restart_trial.validate(
            arm, row["seed"], expected_source=limits["source_sha256"]
        )
        restart_trial.verify_binding(arm, receipt["order"])
        check_selection(row["selection"], weights, receipt, limits)
        require(
            arm["pilot_sha256"] == bound["pilot_sha256"]
            and arm["mask_bits"] == bound["mask_bits"]
            and row["base_commit"] == receipt["publication"]["candidate"]
            and row["arm"]["factual_first"] == [],
            "different parent release, Pilot or work",
        )
        live.append({r["method"]: r for r in row["calibrations"]})
    summaries = []
    for spec in forecast["alternatives"]:
        method, field = spec["method"], spec.get("archive_field")
        calibration = [entry[method] for entry in live if method in entry]
        old = [r["startup_prices"][field] for r in historical] if field else []
        require(
            Counter(map(digest, old)) == spec["return_hash_counts"],
            "historical prediction counts differ",
        )
        values = []
        for capture, item in zip(observed, live, strict=True):
            if method in item:
                values.append(
                    measured_return(field, item[method], capture["arm"])
                    if field
                    else item[method]["return_value"]
                )
        predicted = len(old) >= 6 and all(v == old[0] for v in old)
        require(
            predicted == spec["predicts_exact_match"]
            and (old[0] if predicted else None) == spec["prediction"],
            "post-hoc prediction or wrong original values",
        )
        hits = sum(v == old[0] for v in values) if predicted else None
        state = [c["direct_field_values"] for c in calibration]
        full = [
            digest([c["return_value"], c["direct_field_values"]]) for c in calibration
        ]
        summaries.append(
            {
                "method": method,
                "historical_starts_with_full_return": len(old),
                "heldout_starts": len(values),
                "median_profiled_calibration_s": median(
                    c["elapsed_s"] for c in calibration
                )
                if calibration
                else None,
                "predicted_return_matches": hits,
                "next_exact_match_posterior_mean": (
                    (1 + len(old) + hits) / (2 + len(old) + len(values))
                )
                if predicted
                else None,
                "complete_return_and_direct_state_sha256": full,
                "same_direct_python_fields": (state[0] == state[1])
                if len(state) == 2
                else None,
                "same_recorded_state_and_return": (full[0] == full[1])
                if len(full) == 2
                else None,
            }
        )
    return {
        "schema": "ironmule.calibration-stability-replay.v1",
        "forecast_sha256": sha(forecast_path),
        "heldout_sha256": sha(data / "holdout.json"),
        "observations": [sha(p) for p in sources],
        "seeds": limits["seeds"],
        "control": sha(control),
        "weights": sha(weights),
        "profiles": summaries,
        "full_workers": [
            {
                "cold_completed": r["arm"]["cold"]["completed"],
                "warm_completed": r["arm"]["warm"]["completed"],
                "facts_correct": sum(r["arm"]["facts"]),
                "pilot_decisions": r["arm"]["pilot_decisions"],
                "broker_elapsed_s": r["broker_elapsed_s"],
            }
            for r in observed
        ],
        "confidence_scope": "Predictive means assume exchangeable exact return values within the frozen scope and Beta(1,1). No rate was estimated for options without a predeclared constant forecast. Native side effects and behavior under new load/hardware are unresolved.",
        "qualification": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data", "release", "checkout", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    args = p.parse_args()
    write_once(args.out, replay(args.data, args.release, args.checkout))
