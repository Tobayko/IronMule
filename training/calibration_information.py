"""Reconstruct recorded median decisions and regrade the prospective three-sample abstention rule."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median

from training.calibration_stability import CONTRACT, SOURCE, heldout, sha
from training.code_author import digest, write_once
from training.code_trial import read
from training.trial_release import require


def dispatch(prices):
    result = {}
    for t in sorted(prices):
        k = min((i for i in prices if i >= t), key=lambda i: (prices[i], i))
        if k > t and prices[k] < .95 * prices[t]:
            result[str(t)] = k
    return result


def bounded_dispatch(timings):
    """Exact decision invariance at every corner of the independent empirical price intervals."""
    middle = {t: median(values[1:4]) for t, values in timings.items()}
    lo = {t: .99 * min(values[1:4]) for t, values in timings.items()}
    hi = {t: 1.01 * max(values[1:4]) for t, values in timings.items()}
    proposed = dispatch(middle)
    uncertain = {}
    for t in sorted(middle):
        k = proposed.get(str(t))
        if k is None:
            # A larger width at its lower bound and t at its upper bound can change the decision.
            possible = [m for m in sorted(middle) if m > t and lo[m] < .95 * hi[t]]
            if possible:
                uncertain[str(t)] = possible
        else:
            # The proposed winner stays cheaper than t and wins against every competing width
            # throughout the entire box. Equal prices select the smaller width.
            rivals = [m for m in sorted(middle) if m >= t and m != k and
                      (hi[k] > lo[m] or hi[k] == lo[m] and m < k)]
            if rivals or hi[k] >= .95 * lo[t]:
                uncertain[str(t)] = sorted(set(rivals + ([t] if hi[k] >= .95 * lo[t] else [])))
    return {"dispatch": proposed, "uncertain": uncertain, "intervals_s": {str(t): [lo[t], hi[t]] for t in sorted(lo)}}


def decision(forecast, observations, control, weights, checkout):
    evidence = heldout(forecast, observations, control, weights, checkout)
    f = read(forecast)
    selected = [x for x in f["alternatives"] if x["information_eligible"]]
    candidates = []
    for x in selected:
        method, candidate = x["method"], {"method": x["method"]}
        models = []
        for row in sorted((read(p) for p in observations), key=lambda r: r["seed"]):
            sample = next(a for a in row["calibrations"] if a["method"] == method)
            timings = {int(t): vals for t, vals in sample["timing_samples_by_width"].items()}
            # This adapter handles the specific observed, source-defined 5% discrete dispatch.
            # All unknown return semantics remain outside its prediction authority.
            require(set(sample["direct_field_values"]) == {"pad_prices"} and isinstance(sample["return_value"], dict)
                    and set(timings) == set(range(1, row["arm"]["binding"]["max_rows"] + 1))
                    and all(len(v) == 6 and all(math.isfinite(s) and s > 0 for s in v) for v in timings.values()),
                    "unrecognized reduction, missing original work or timing samples")
            prices = {t: median(a[1:]) for t, a in timings.items()}
            require({str(t): v for t, v in prices.items()} == sample["direct_field_values"]["pad_prices"]
                    and dispatch(prices) == sample["return_value"], "recorded raw medians or discrete decision differ")
            reduced = bounded_dispatch(timings)
            useful = sum(sum(v[4:]) for v in timings.values())
            actual = sample["return_value"]
            models.append({**reduced, "seed": row["seed"], "actual": actual, "exact": reduced["dispatch"] == actual,
                           "measured_full_method_s": sample["elapsed_s"], "counterfactual_tail_sample_s": useful,
                           "passes": not reduced["uncertain"] and reduced["dispatch"] == actual and useful > f["required_cold_saving_s"]})
        candidate.update(holdouts=models, passes=all(m["passes"] for m in models),
                         conservative_payback_starts_if_savings_real=math.ceil((evidence["measured_s"] + f["estimated_qualification_s"])
                            / min(m["counterfactual_tail_sample_s"] for m in models)))
        candidates.append(candidate)
    return {"schema": "ironmule.calibration-stability-information-decision.v1", "forecast_sha256": sha(forecast),
            "contract_sha256": sha(CONTRACT), "source_sha256": sha(checkout / SOURCE), "verified_holdout_digest": digest(evidence),
            "holdout_sha256": evidence["holdout_sha256"], "scenarios": candidates, "stage": "simulated partial sampling",
            "recommended": "prepare independently frozen target experiment" if any(c["passes"] for c in candidates)
            else "retain full real-time measurement for this registered sampling rule",
            "authoring_eligible": False, "activation": None,
            "limitations": "Empirical price intervals have no statistical coverage guarantee. Tail samples are observed work; actual net source speed, future workload and driver state remain untested."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", type=Path, action="append", required=True)
    for k in ("forecast", "control", "weights", "checkout", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    args = p.parse_args()
    write_once(args.out, decision(args.forecast, args.observation, args.control, args.weights, args.checkout))
