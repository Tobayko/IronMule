"""Select a fresh-price measurement plan, then seek a source variant only after separate blind observations."""
from __future__ import annotations

import argparse
import math
from itertools import product
from pathlib import Path
from statistics import median

from lab.pk1.followup_contract import contract_digest
from lab.pk1.restart_order import order_digest
from training import calibration_information as prior
from training import calibration_stability as stability
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require

CONTRACT = stability.CONTRACT.with_name("calibration_sampling_contract.json")


def replay(row, method, w, n, floor):
    cal = next(v for v in row["calibrations"] if v["method"] == method)
    samples = {int(t): data for t, data in cal["timing_samples_by_width"].items()}
    require(set(samples) == set(range(1, row["arm"]["binding"]["max_rows"] + 1))
            and all(len(v) == 6 and all(math.isfinite(t) and t > 0 for t in v) for v in samples.values())
            and set(cal["direct_field_values"]) == {"pad_prices"}, "unknown, incomplete or incompatible calibration estimator")
    reference = {k: median(v[1:]) for k, v in samples.items()}
    require({str(k): t for k, t in reference.items()} == cal["direct_field_values"]["pad_prices"]
            and prior.dispatch(reference) == cal["return_value"], "fresh prices or original dispatch differ")
    running = {k: v[w:w + n] for k, v in samples.items()}
    prices = {k: median(v) for k, v in running.items()}
    intervals = {k: [.99 * min(v), 1.01 * max(v)] for k, v in running.items()}
    proposed = prior.dispatch(prices)
    uncertain = {}
    for k in sorted(prices):
        best = proposed.get(str(k))
        if best is None:
            rivals = [m for m in sorted(prices) if m > k and intervals[m][0] < .95 * intervals[k][1]]
        else:
            upper = intervals[best][1]
            rivals = [m for m in sorted(prices) if m >= k and m != best and
                      (upper > intervals[m][0] or upper == intervals[m][0] and m < best)]
            if upper >= .95 * intervals[k][0]:
                rivals = [k, *rivals]
        if rivals:
            uncertain[str(k)] = sorted(set(rivals))
    error = max(abs(prices[k] / val - 1) for k, val in reference.items())
    covered = all(lo <= reference[k] <= hi for k, (lo, hi) in intervals.items())
    retained = sum(sum(v[:w + n]) for v in samples.values())
    omitted = sum(sum(v[w+n:]) for v in samples.values())
    return {"seed": row["seed"], "correct_discrete": proposed == cal["return_value"],
            "reference_dispatch": cal["return_value"], "uncertain": uncertain, "max_price_relative_error": error,
            "reference_in_all_intervals": covered, "inclusive_measured_method_s": cal["elapsed_s"],
            "retained_warm_and_samples_s": retained, "modeled_tail_saved_s": omitted,
            "qualifies_for_experiment": not uncertain and covered and proposed == cal["return_value"] and error <= .02 and omitted > floor}


def forecast(prior_dir, control, weights, checkout):
    original = prior_dir / "prediction.json"
    samples = [prior_dir / f"{s}.json" for s in read(CONTRACT)["training_seeds"]]
    last = prior.decision(original, samples, control, weights, checkout)
    require(read(prior_dir / "information-decision.json") == last, "prior evidence differs")
    orig, limits = read(original), read(CONTRACT)
    ranks = []
    for candidate in last["scenarios"]:
        if candidate["passes"]:
            continue
        method = candidate["method"]
        for w, n in product(limits["warms"], limits["samples"]):
            if w + n >= limits["ref_warms"] + limits["ref_samples"]:
                continue
            rows = [replay(read(p), method, w, n, orig["required_cold_saving_s"]) for p in samples]
            ranks.append({"method": method, "warmups": w, "samples": n, "rows": rows,
                          "minimum_modeled_saving_s": min(r["modeled_tail_saved_s"] for r in rows),
                          "max_price_relative_error": max(r["max_price_relative_error"] for r in rows),
                          "passed": all(r["qualifies_for_experiment"] for r in rows)})
    ranks.sort(key=lambda r: (-int(r["passed"]), -r["minimum_modeled_saving_s"], r["max_price_relative_error"], r["samples"], r["method"]))
    selected = {k:ranks[0][k] for k in ("method", "warmups", "samples")} if ranks and ranks[0]["passed"] else None
    return {"schema": "ironmule.calibration-sampling-forecast.v1", "contract_sha256": stability.sha(CONTRACT),
            "base_commit": orig["base_commit"], "source_sha256": orig["source_sha256"], "control_receipt_sha256": stability.sha(control),
            "previous_decision_sha256": stability.sha(prior_dir / "information-decision.json"),
            "training_observations": [stability.sha(p) for p in samples], "holdout_seeds": limits["seeds"],
            "alternatives": orig["alternatives"], "candidate_plans": ranks, "selected_plan": selected,
            "select_information_probe": selected is not None, "full_qualification_lower_bound_s": orig["estimated_qualification_s"],
            "required_cold_saving_s": orig["required_cold_saving_s"], "hourly_usd": orig["hourly_usd"],
            "limits": "Prefix replay retains and charges every warm step. Actual checking, context transitions and fallback will be included in the proposed independent hardware trial."}


def decide(forecast_path, samples, control, weights, checkout):
    f, c = read(forecast_path), read(CONTRACT)
    proof = stability.heldout(forecast_path, samples, control, weights, checkout, contract=CONTRACT)
    require(f["schema"] == "ironmule.calibration-sampling-forecast.v1" and f["select_information_probe"],
            "missing prospectively selected sampling plan")
    plan = f["selected_plan"]
    method = plan["method"]
    rows = [replay(r, method, plan["warmups"], plan["samples"], f["required_cold_saving_s"])
            for r in sorted((read(p) for p in samples),key=lambda r:r["seed"])]
    full_experiment = f["full_qualification_lower_bound_s"]
    minimum = min(r["modeled_tail_saved_s"] for r in rows)
    observed_cost = proof["measured_s"]
    meta = {"sampling_contract_sha256": stability.sha(CONTRACT), "contract": contract_digest(),
            "base_commit": c["base_commit"], "parent_order": c["parent_order"], "control_receipt_sha256": c["control_receipt_sha256"],
            "weights_sha256": stability.sha(weights), "source_sha256": c["source_sha256"], "forecast_sha256": stability.sha(forecast_path),
            "parent_observations": f["training_observations"], "observations": proof["holdout_sha256"], "selected_plan": {
                **plan, "reference_warmups": c["ref_warms"], "reference_samples": c["ref_samples"], "uncertainty_slack": c["uncertainty_slack"]},
            "blind_results": rows, "simulation": {"kind": "simulated, using complete original measured work",
                "qualification_lower_bound_s": full_experiment, "new_discovery_s": observed_cost,
                "new_discovery_usd": proof["attributable_lease_usd"],
                "estimated_qualification_usd": full_experiment * f["hourly_usd"] / 3600,
                "future_starts_for_qualification_cost_only": math.ceil(full_experiment / minimum),
                "modeled_net_s_after_qualification": {str(k): k*minimum - full_experiment for k in (1,32,256)}},
            "expected_work": "Current qualified 32-client/18-fact PR60 Qwen startup on matching hardware with trained startup head. New mechanism measures fresh local prices. No reusable measured price values, test outcomes, numerical checks or workload text may be embedded."}
    if not all(r["qualifies_for_experiment"] for r in rows):
        return {**meta, "schema":"ironmule.calibration-sampling-abstain.v1", "reason": "The registered strategy failed at least one independent full-work holdout.", "order_id": None}
    (line,) = [x["line"] for x in f["alternatives"] if x["method"] == method]
    target = {"path": stability.SOURCE, "class": "GgmlEngine", "function": method, "line": line}
    order = {**meta, "schema": "ironmule.qwen-calibration-sampling-order.v1", "target": target, "allowed_paths":[target["path"]],
             "hypothesis": "Implement and test a lower-cost, freshly measured calibration within the bound scope. Preserve the qualified original route as fallback. The registered prefix plan saved measurement work with bounded price error and unchanged discrete decisions in all blind source traces.",
             "requires": ["Preserve every original weight, mask binding, correctness check, fixed user workload, stale/foreign fallback and complete profile.",
                          "Unknown or ambiguous intervals, incompatible optional flags or startup regime must use the complete original calibration. Never cache a constant decision or stale price.",
                          "Report the actually measured fresh per-width prices with sample count, measurement source, duration and fallback reason. Preserve legitimate warmup, KV and graph state and check the full startup and warm serving clocks.",
                          "Independently compare complete candidate dispatch to a full original fresh measurement on target. Require <=2% per-width price discrepancy under bound conditions, A/A noise checked independently; include miss/ambiguity paths.",
                          "Packaging, a simulation or matching prices alone cannot activate. The unchanged independent end-to-end hardware gates and budget limits apply."]}
    order["profile_interface"] = {
        "opt_in": ["IRONMULE_CALIBRATION_SAMPLING_PROFILE", "IRONMULE_CALIBRATION_SAMPLING_SHA256"],
        "schema": "ironmule.qwen-sampling-profile.v1",
        "fields": ["order_id", "source_sha256", "method", "selected_plan", "parent_cache_sha256", "runtime", "created_s", "expires_s"],
        "runtime": "exact prior qualified W4 receipt.binding; no price labels",
        "telemetry": "self.calibration_sampling contains profile_sha256, mode adaptive/full, reason, counts by width [warm_count,measured_count], prices by width in fresh measured seconds; missing, expired, foreign or uncertain profiles retain complete measurement"}
    order["order_id"] = order_digest(order)
    return order


def reconstruct(prior_dir, forecast_path, observations, control, weights, checkout):
    """Untrusted exports must reproduce the preceding prospective forecast before a new order can be trusted."""
    expected = forecast(prior_dir, control, weights, checkout)
    require(read(forecast_path) == expected, "selected forecast differs from the complete historical experiment")
    return decide(forecast_path, observations, control, weights, checkout)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("forecast", "decide"))
    p.add_argument("--prior-dir", type=Path)
    p.add_argument("--forecast", type=Path)
    p.add_argument("--observation", type=Path, action="append")
    for k in ("control", "weights", "checkout", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    result = forecast(a.prior_dir, a.control, a.weights, a.checkout) if a.mode == "forecast" else decide(a.forecast, a.observation, a.control, a.weights, a.checkout)
    write_once(a.out, result)
