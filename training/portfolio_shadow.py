"""Freeze a prospective numeric prediction and independently grade new original and post-work observations."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from lab.pk1 import portfolio_shadow_worker as worker
from lab.pk1 import post_restart_verify, restart_trial
from training import calibration_portfolio as portfolio
from training.calibration_stability import sha
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require

CONTRACT = Path(__file__).with_name("portfolio_shadow_contract.json")


def forecast(data, control):
    baseline = data / "a100-calibration-stability"
    path = data / "a100-calibration-sampling/portfolio-research.json"
    prior = portfolio.assess(data, control)
    require(prior == read(path), "different historical portfolio study")
    question = prior["new_information_question"]
    require(question is not None, "no empirical information target")
    selected = next(
        p
        for p in prior["portfolios"]
        if p["profile_components"] == question["components"]
    )
    original = read(control)
    donors = [
        portfolio.original(baseline / f"{s}.json", original)
        for s in prior["cohorts"]["donors"]
    ]
    limits = read(CONTRACT)
    cache_path = next(
        (data / "a100-calibration-sampling").glob(
            "trial-"
            + read(data / "a100-calibration-sampling/decision.json")["order_id"]
            + "/sampling-parent-receipt.json"
        )
    )
    require(
        sha(cache_path) == original["qualification"]["binding"]["cache_receipt_sha256"],
        "different baseline cache",
    )
    return {
        "schema": "ironmule.portfolio-shadow-forecast.v1",
        "contract_sha256": sha(CONTRACT),
        "parent_control_sha256": sha(control),
        "original_cache_sha256": sha(cache_path),
        "parent_cache_created_s": read(cache_path)["created_s"],
        "historical_portfolio_study_sha256": sha(path),
        "components": selected["profile_components"],
        "donors": donors,
        "registered_contexts": limits["contexts"],
        "max_complete_vector_relative_error": 0.02,
        "sentinel_drift_limit": selected["calibrated_sentinel_drift_limit"],
        "historical_error_envelope": selected["calibrated_complete_error_envelope"],
        "selection": "Minimize largest relative width-1 difference to one original donor. Ties select later donor. Abstain if outside empirical calibration drift; the post-delivery context always remains shadow.",
        "operational_clock": "Exact donor timestamps are unavailable. The validated parent cache creation gives an upper bound on age; previous model release traffic does not establish a serving cache TTL.",
        "budget_usd": 0.05,
        "maximum_batch_s": 81,
        "hourly_usd": read(data / "a100-calibration-sampling/prediction.json")[
            "hourly_usd"
        ],
        "can_issue_source_order": False,
    }


def grade(data, control, weights, forecast_path, samples):
    q = read(control)
    f = forecast(data, control)
    limits = read(CONTRACT)
    require(
        f == read(forecast_path) and sha(weights) == limits["weights_sha256"],
        "unbound prediction or startup head",
    )
    rows = sorted((read(p) for p in samples), key=lambda r: r["seed"])
    require(
        [r["seed"] for r in rows] == limits["seeds"], "incomplete prospective sample"
    )
    proof = []
    for r in rows:
        a = r["arm"]
        seed = r["seed"]
        require(
            r["schema"] == limits["observation_schema"]
            and r["forecast_sha256"] == sha(forecast_path)
            and r["portfolio_context"] == f["registered_contexts"][str(seed)]
            and r["profiler_sha256"] == sha(Path(worker.__file__))
            and r["original_probe_sha256"]
            == sha(Path(worker.__file__).with_name("calibration_stability_probe.py"))
            and r["selection_before"] == r["selection_after"] == "candidate"
            and r["contract_sha256"] == sha(CONTRACT)
            and r["parent_order"] == limits["parent_order"]
            and r["engine_sha256"] == r["source_sha256"] == limits["source_sha256"],
            "foreign context or original measurement",
        )
        post_restart_verify.check_selection(r["selection"], weights, q, limits)
        restart_trial.validate(a, seed, expected_source=limits["source_sha256"])
        restart_trial.verify_binding(a, q["order"])
        bound = q["qualification"]["binding"]
        require(
            a["binding"] == bound["runtime"]
            and a["cuda_vendor"] == bound["vendor"]
            and a["mask_bits"] == bound["mask_bits"]
            and a["reused"],
            "different full native profile",
        )
        start = r["resources_before"]
        post = r["post_work"]
        require(
            set(start["cpu_affinity"]) == set(a["binding"]["affinity"])
            and start["utc_s"] >= f["parent_cache_created_s"]
            and post["after_work"]["utc_s"] <= f["parent_cache_created_s"] + 86400,
            "foreign affinity or expired cache time",
        )
        measures = {v["method"]: v for v in r["calibrations"]}
        pad = measures["measure_padding"]
        pas = measures["measure_pass_costs"]
        full = {
            "padding": pad["direct_field_values"]["pad_prices"],
            "passes": dict((str(w), p) for w, p in a["startup_prices"]["passes"]),
        }
        require(
            pad["return_value"] == a["startup_prices"]["padding"]
            and pas["return_value"]["fields"]["points"]
            == a["startup_prices"]["passes"],
            "incomplete original state",
        )
        for k, obj in (("padding", pad), ("passes", pas)):
            require(
                all(
                    median(v[1:]) == full[k][str(w)]
                    for w, v in obj["timing_samples_by_width"].items()
                )
                and set(obj["timing_samples_by_width"]) == set(full[k]),
                "unbound full calibration prices",
            )

        def observed(context, complete, sentinels, dispatch, eligible):
            current = {
                "seed": seed,
                "family": context,
                "complete": complete,
                "dispatch": dispatch,
            }
            def sentinel_drift(donor):
                return max(
                    abs(donor["complete"][k]["1"] / sentinels[k] - 1.0)
                    for k in f["components"]
                )

            x = min(f["donors"], key=lambda b: (sentinel_drift(b), -b["seed"]))
            drift = sentinel_drift(x)
            error, per = portfolio.distance(x, current, f["components"])
            numerical = (
                error <= f["max_complete_vector_relative_error"]
                and x["dispatch"] == dispatch
            )
            return {
                "seed": seed,
                "context": context,
                "donor_seed": x["seed"],
                "sentinel_drift": drift,
                "within_sentinel_threshold": drift <= f["sentinel_drift_limit"],
                "context_eligible_for_research": eligible,
                "max_relative_error": error,
                "per_component_max_relative_error": per,
                "complete_vector_guard_passed": numerical,
                "inside_historical_error_envelope": error
                <= f["historical_error_envelope"],
                "fixed_latest_donor_error": portfolio.distance(
                    f["donors"][-1], current, f["components"]
                )[0],
                "unchanged_empty_action_rule_matches": not dispatch,
                "actual_padding_action": dispatch,
            }

        proof.append(
            observed(
                "full-original-initialization",
                full,
                {k: v["1"] for k, v in full.items()},
                pad["return_value"],
                True,
            )
        )
        if r["portfolio_context"] == "post-delivery-shadow":
            shadow = post["shadow"]
            fresh = shadow["standalone"]
            for k, num in (("passes", 3), ("padding", 5)):
                require(
                    len(fresh[k]["timing_samples_s"]) == num + 1
                    and fresh[k]["price_s"] == median(fresh[k]["timing_samples_s"][1:]),
                    "incomplete standalone sentinel",
                )
            witness = observed(
                "after-serving-outside-startup-qualification",
                shadow["complete"],
                {k: v["price_s"] for k, v in fresh.items()},
                shadow["dispatch"],
                False,
            )
            witness.update(
                standalone_sentinel_s=sum(v["elapsed_s"] for v in fresh.values()),
                measured_full_s=sum(shadow["full_method_s"].values()),
                post_work_runtime=post["post_work_runtime"],
            )
            proof.append(witness)
    charged = sum(r["broker_elapsed_s"] for r in rows)
    require(
        charged <= f["maximum_batch_s"]
        and charged * f["hourly_usd"] / 3600 <= f["budget_usd"],
        "registered diagnostic price exceeded",
    )
    admitted = [
        p
        for p in proof
        if p["within_sentinel_threshold"] and p["context_eligible_for_research"]
    ]
    failed = [
        p
        for p in admitted
        if not p["complete_vector_guard_passed"]
    ]
    result = (
        "numerical prediction refuted"
        if failed
        else "no admitted cold observations"
        if not admitted
        else "numerical shadow only"
    )
    return {
        "schema": "ironmule.portfolio-shadow-result.v1",
        "contract_sha256": sha(CONTRACT),
        "forecast_sha256": sha(forecast_path),
        "raw_sha256": [sha(p) for p in samples],
        "verdict": result,
        "observations": proof,
        "broker_elapsed_s": charged,
        "attributable_usd": charged * f["hourly_usd"] / 3600,
        "source_order": None,
        "activation": None,
        "resources": [
            {
                "seed": r["seed"],
                "before": r["resources_before"],
                "after": r["post_work"]["after_work"],
                "maximum_possible_donor_age_s": r["resources_before"]["utc_s"]
                - f["parent_cache_created_s"],
            }
            for r in rows
        ],
        "field_read_diagnostics": [
            {"seed": r["seed"], "per_actual_delivered_wave": r["serving_field_reads"]}
            for r in rows
        ],
        "limitation": "All responses used freshly calibrated original parameters. No cached profile controlled an output, and the shadow changes its disposable native context. Trace-level attribute access is not a causal performance gain.",
    }


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("mode", choices=("forecast", "grade"))
    for k in ("data", "control", "out"):
        cmd.add_argument("--" + k, type=Path, required=True)
    cmd.add_argument("--weights", type=Path)
    cmd.add_argument("--forecast", type=Path)
    cmd.add_argument("--observation", type=Path, action="append")
    a = cmd.parse_args()
    out = (
        forecast(a.data, a.control)
        if a.mode == "forecast"
        else grade(a.data, a.control, a.weights, a.forecast, a.observation)
    )
    write_once(a.out, out)
