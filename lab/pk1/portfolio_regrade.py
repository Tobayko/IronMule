"""Reconstruct all prospective portfolio price comparisons on a CPU from the original receipts."""
import argparse
from pathlib import Path
from statistics import median

from lab.pk1.followup_contract import PARENT
from lab.pk1.native_trial import checksum
from lab.pk1.post_restart_verify import check_selection
from lab.pk1.restart_trial import validate, verify_binding
from training.code_trial import read
from training.portfolio_shadow import CONTRACT
from training.trial_release import require


def vector(doc):
    methods = {row["method"]: row for row in doc["calibrations"]}
    values = {"padding": methods["measure_padding"]["direct_field_values"]["pad_prices"],
              "passes": dict((str(k), v) for k, v in doc["arm"]["startup_prices"]["passes"])}
    for family, method in (("passes", "measure_pass_costs"), ("padding", "measure_padding")):
        row = methods[method]
        require({w: median(ts[1:]) for w, ts in row["timing_samples_by_width"].items()} == values[family],
                "recorded price differs from the original native samples")
    dispatch = methods["measure_padding"]["return_value"]
    require(dispatch == doc["arm"]["startup_prices"]["padding"]
            and methods["measure_pass_costs"]["return_value"]["fields"]["points"] == doc["arm"]["startup_prices"]["passes"],
            "different effective calibration")
    return values, dispatch, methods


def comparison(forecast, context, seed, sentinels, full, dispatch):
    components, donors = forecast["components"], forecast["donors"]
    require(all(full[k].keys() == donors[0]["complete"][k].keys() for k in components), "different portfolio widths")
    distances = {d["seed"]: max(abs(d["complete"][k]["1"] / sentinels[k] - 1) for k in components) for d in donors}
    donor = min(donors, key=lambda d: (distances[d["seed"]], -d["seed"]))
    def errors(d):
        return {k: {w: abs(d["complete"][k][w] / price - 1) for w, price in full[k].items()} for k in components}
    def worst(e):
        return max(value for row in e.values() for value in row.values())
    error = errors(donor)
    max_error = worst(error)
    drift = distances[donor["seed"]]
    within = drift <= forecast["sentinel_drift_limit"]
    same = dispatch == donor["dispatch"]
    return {"seed": seed, "context": context, "donor": donor["seed"], "donor_distances": {str(s): v for s, v in distances.items()},
            "drift": drift, "within": within, "cold_admissible": context == "startup" and within, "dispatch_match": same,
            "within_guard": max_error <= forecast["max_complete_vector_relative_error"] and same, "complete_error": max_error,
            "component_error": {k: {"worst_width": max(v, key=v.get), "max": max(v.values())} for k, v in error.items()},
            "fixed_donor_error": worst(errors(donors[-1])), "fixed_empty_action_passes": not dispatch,
            "actual": full, "predicted": {k: donor["complete"][k] for k in components}}


def verify(directory):
    report = read(directory / "independent-replay.json")
    grade = read(directory / "result.json")
    forecast_path = directory / "prediction.json"
    forecast = read(forecast_path)
    control_path = directory.parent / "a100-restart" / PARENT / "control.json"
    control, contract = read(control_path), read(CONTRACT)
    binding = control["qualification"]["binding"]
    require(report["schema"] == "ironmule.portfolio-shadow-independent-replay.v1"
            and report["forecast_sha256"] == checksum(forecast_path) and checksum(CONTRACT) == forecast["contract_sha256"]
            and report["recorded_grade_sha256"] == checksum(directory / "result.json") and forecast["parent_control_sha256"] == checksum(control_path)
            and report["source_order"] is grade["source_order"] is None and report["activation"] is grade["activation"] is None,
            "unbound shadow report")
    for donor in forecast["donors"]:
        file = directory.parent / "a100-calibration-stability" / f"{donor['seed']}.json"
        raw = read(file)
        actual, dispatch, _ = vector(raw)
        require(checksum(file) == donor["sha256"] and all(donor["complete"][k] == actual[k] for k in forecast["components"]) and donor["dispatch"] == dispatch
                and raw["arm"]["binding"] == binding["runtime"], "different donor or historical prices")
    predictions = []
    workers = []
    for seed in contract["seeds"]:
        file = directory / f"{seed}.json"
        doc = read(file)
        arm = doc["arm"]
        validate(arm, seed, expected_source=contract["source_sha256"])
        verify_binding(arm, control["order"])
        check_selection(doc["selection"], control_path.parent / "startup-head.npz", control, contract)
        before, after = doc["resources_before"], doc["post_work"]["after_work"]
        require(doc["base_commit"] == contract["base_commit"] and doc["source_sha256"] == doc["engine_sha256"] == contract["source_sha256"]
                and doc["forecast_sha256"] == checksum(forecast_path) and doc["contract_sha256"] == checksum(CONTRACT)
                and doc["parent_order"] == PARENT and doc["weights_sha256"] == contract["weights_sha256"]
                and doc["selection_before"] == doc["selection_after"] == "candidate" and arm["binding"] == binding["runtime"]
                and arm["cuda_vendor"] == binding["vendor"] and arm["mask_bits"] == binding["mask_bits"] and arm["reused"]
                and arm["facts"] == [True] * 18 and not arm["factual_first"] and len(arm["cold"]["answers"]) == len(arm["warm"]["answers"]) == 32
                and set(before["cpu_affinity"]) == set(arm["binding"]["affinity"])
                and forecast["parent_cache_created_s"] <= before["utc_s"] <= after["utc_s"] <= forecast["parent_cache_created_s"] + 86400,
                "different serving worker or incomplete original work")
        full, dispatch, methods = vector(doc)
        predictions.append(comparison(forecast, "startup", seed, {k:v["1"] for k, v in full.items()}, full, dispatch))
        worker = {"seed": seed, "raw_sha256": checksum(file), "broker_s": doc["broker_elapsed_s"], "initialized_s": arm["initialized_s"],
                  "fresh_calibrations_s": methods["measure_padding"]["elapsed_s"] + methods["measure_pass_costs"]["elapsed_s"],
                  "cold_s": arm["cold_s"], "independent_cold_sentinel": False, "initial_binding_matches_parent_release": True,
                  "warm_in_startup_scope": False, "startup_threads": arm["binding"]["threads"], "post_delivery_threads": doc["post_work"]["post_work_runtime"]["threads"],
                  "age_upper_s": before["utc_s"] - forecast["parent_cache_created_s"], "load_1min_before_after": [r["load_1_5_15"][0] for r in (before, after)],
                  "major_faults_before_after": [r["major_faults"] for r in (before, after)], "nvidia_snapshot_before_after": [r["nvml_snapshot"] for r in (before, after)],
                  "first_serving_reads": {
                      phase: {field: {key: readout[key] for key in ("count", "first_caller", "line")}
                              for field, readout in fields.items()}
                      for phase, fields in doc["serving_field_reads"].items()
                  }, "measured_method_seconds": {k: methods[k]["elapsed_s"] for k in ("measure_costs","measure_padding","measure_pass_costs")}}
        shadow = doc["post_work"].get("shadow")
        if shadow:
            probe = shadow["standalone"]
            for k, n in (("passes",4),("padding",6)):
                times = probe[k]["timing_samples_s"]
                require(len(times) == n and median(times[1:]) == probe[k]["price_s"], "different post-delivery native sentinel")
            predictions.append(comparison(forecast,"post-delivery",seed, {k:v["price_s"] for k,v in probe.items()},shadow["complete"],shadow["dispatch"]))
            worker.update(standalone_s=sum(t["elapsed_s"] for t in probe.values()), full_methods_s=sum(shadow["full_method_s"].values()),
                          registered_shadow_s=shadow["elapsed_s"], width1_fresh_vs_subsequent_original_seconds={k:[v["price_s"],shadow["complete"][k]["1"]] for k,v in probe.items()})
        workers.append(worker)
    require(report["observations"] == predictions and report["observation_sha256"] == [w["raw_sha256"] for w in workers] and report["workers"] == workers,
            "predicted or actual price, abstention, or physical worker differs")
    cost = report["economic_comparison"]
    seconds = sum(w["broker_s"] for w in workers)
    require(cost["broker_elapsed_s"] == grade["broker_elapsed_s"] == seconds and cost["rate_usd_per_hour"] == forecast["hourly_usd"]
            and cost["marginal_lease_estimate_usd"] == grade["attributable_usd"] == seconds * forecast["hourly_usd"] / 3600
            and cost["actual_customer_serving_savings_s"] == 0 and cost["net_measured_serving_advantage"] is None
            and cost["cold_sentinel_alone_actually_measured"] is False, "different attributed cost or unmeasured benefit")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report = verify(parser.parse_args().directory)
    print(f"Verified {len(report['observations'])} original numerical comparisons from {len(report['workers'])} serving workers.")
