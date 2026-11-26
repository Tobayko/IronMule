"""Rank the exact repeatability of observed startup calibration outputs and test registered predictions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import math
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

from lab.pk1 import calibration_stability_probe as probe
from lab.pk1 import restart_trial
from lab.pk1.post_restart_verify import check_selection
from training.code_author import digest, write_once
from training.code_trial import read
from training.trial_release import require

SOURCE = "ironmule/ggml_engine.py"
CONTRACT = Path(__file__).resolve().with_name("calibration_stability_contract.json")
# These serializations are the three *fields recorded by restart_arm*. Resolve each producer from the
# actual merged constructor's assignments below. New or missing producers stay in the inventory as unknown.
DESTINATIONS = {"pilot": ("pilot", None), "padding": ("pad_to", None), "passes": ("pass_costs", "points")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_source(source):
    info = probe.calibration_methods(source)
    cls = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == "GgmlEngine")
    init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    phases = []
    for n in ast.walk(init):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "trace_note" and n.args:
            msg = n.args[0]
            if isinstance(msg, ast.Constant) and isinstance(msg.value, str) and (match := re.match(r"init (\w+) begin$", msg.value)):
                phases.append((n.lineno, match[1]))
    for assignment in ast.walk(init):
        if not isinstance(assignment, (ast.Assign, ast.AnnAssign)) or assignment.value is None:
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        for call in ast.walk(assignment.value):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "self" and call.func.attr in info):
                continue
            method = info[call.func.attr]
            before = [(line, phase) for line, phase in phases if line < call.lineno]
            method["historical_phase"] = max(before)[1] if before else None
            for name, (owner, _) in DESTINATIONS.items():
                if any((isinstance(t, ast.Attribute) and t.attr == owner) or (isinstance(t, ast.Name) and t.id == owner) for t in targets):
                    method["archive_field"] = name
    return info


def historical_phases(path):
    events = []
    for line in path.open():
        if m := re.match(r"^# init (\w+)(?: begin)? (\d+(?:\.\d+)?)$", line.rstrip()):
            events.append((m[1], float(m[2])))
    result = {}
    for (name, start), (_, end) in pairwise(events):
        if end >= start:
            result.setdefault(name, []).append(end - start)
    return result


def archived_arms(history, ctrl, contract):
    names = [*(f"cand-{i}.json" for i in range(6)), *(f"facts-cand-{i}.json" for i in range(2))]
    arms = []
    for name in names:
        row = read(history / name)
        binding = ctrl["qualification"]["binding"]
        restart_trial.validate(row, row["seed"], expected_source=contract["source_sha256"])
        restart_trial.verify_binding(row, ctrl["order"])
        require(row["reused"] and row["binding"] == binding["runtime"] and row["mask_bits"] == binding["mask_bits"]
                and row["cuda_vendor"] == binding["vendor"] and bool(row["factual_first"]) == name.startswith("facts-")
                and set(row["startup_prices"]) == set(DESTINATIONS), "missing or mismatched current historical calibration arm")
        arms.append((name, row))
    require(len({r["seed"] for _, r in arms}) == len(names), "near-duplicate training starts")
    return arms


def forecast(history, control, checkout, phase_priors, hourly):
    c = read(CONTRACT)
    source = checkout / SOURCE
    ctrl = read(control)
    require(sha(control) == c["control_receipt_sha256"] and sha(source) == c["source_sha256"]
            and ctrl["publication"]["candidate"] == c["base_commit"] and math.isfinite(hourly) and hourly > 0,
            "unsupported history, source, control or cost")
    methods = production_source(source)
    arms = archived_arms(history, ctrl, c)
    n = len(arms)
    wall = max(row["broker_elapsed_s"] for _, row in arms)
    cold_floor = max(row["cold_s"] for _, row in arms) * (1 - 1 / c["positive_gain_ratio"])
    prior_values = [historical_phases(p) for p in phase_priors]
    alternatives = []
    for method, item in sorted(methods.items()):
        field = item.get("archive_field")
        values = [r["startup_prices"][field] for _, r in arms] if field else []
        counts = Counter(digest(v) for v in values)
        phase = item.get("historical_phase")
        prior_s = [min(p[phase]) for p in prior_values if phase in p]
        stable = len(counts) == 1 and n >= 6
        # Posterior predictive Beta(success+1, miss+1): P(at least one mismatch in two new starts).
        p_miss = 1 - (n + 1) / (n + 3) if stable else None
        gross = p_miss * c["qualification_workers"] * wall if stable else 0.
        select = bool(stable and len(prior_s) >= 2 and min(prior_s) > cold_floor and gross > 2 * wall)
        contrasting = []
        if stable:
            for p in sorted(history.glob("*.json")):
                r = read(p)
                if r.get("schema") == "ironmule.qwen-restart.arm.v1" and not r["reused"] and field in r["startup_prices"]:
                    other = r["startup_prices"][field]
                    if digest(other) not in counts:
                        contrasting.append({"file": p.name, "sha256": sha(p), "value": other,
                                            "different_source_or_cache_route": True})
        alternatives.append({**item, "method": method, "measured_starts": len(values),
                             "distinct_complete_returns": len(counts), "return_hash_counts": dict(sorted(counts.items())),
                             "prediction": values[0] if stable else None, "predicts_exact_match": stable,
                             "separate_historical_phase_s": prior_s, "counterexamples_other_regimes": contrasting,
                             "unrecorded_direct_fields_in_history": item["direct_python_writes"],
                             "observed_class": "identical complete recorded return" if stable else
                             "varying complete returns" if values else "no complete optional return in these archive rows",
                             "probability_any_miss_in_two": p_miss, "conditional_larger_trial_avoidance_s": gross,
                             "information_eligible": select,
                             "optimistic_payback_starts_at_entire_prior_phase_removed": math.ceil((2 + c["qualification_workers"]) * wall / min(prior_s))
                             if prior_s and min(prior_s) > 0 else None})
    alternatives.sort(key=lambda r: (-int(r["information_eligible"]), r["method"]))
    return {"schema": "ironmule.calibration-stability-forecast.v1", "contract_sha256": sha(CONTRACT),
            "control_receipt_sha256": sha(control), "source_sha256": sha(source), "base_commit": c["base_commit"],
            "method_provenance": c["search_family_provenance"], "cohorts": {"mixed": 6, "factual_first": 2},
            "historical_rows": [{"name": n, "seed": r["seed"], "sha256": sha(history / n)} for n, r in arms],
            "separate_regime_phase_priors": [{"name": p.name, "sha256": sha(p)} for p in phase_priors],
            "holdout_seeds": c["seeds"], "alternatives": alternatives, "required_cold_saving_s": cold_floor,
            "select_information_probe": any(r["information_eligible"] for r in alternatives),
            "estimated_holdout_s": 2 * wall, "estimated_holdout_usd": 2 * wall * hourly / 3600,
            "estimated_qualification_s": c["qualification_workers"] * wall, "hourly_usd": hourly,
            "authoring_eligible": False, "training_label": False,
            "price_scope": "Older stage intervals are priors from a different regime. The value of learning a rejection assumes a larger future trial would otherwise be considered. Actual future start count and attainable saving are unknown."}


def heldout(forecast_path, samples, control, weights, checkout):
    f = read(forecast_path)
    roots, ctrl = read(CONTRACT), read(control)
    require(f["contract_sha256"] == sha(CONTRACT) and f["source_sha256"] == sha(checkout / SOURCE)
            and f["control_receipt_sha256"] == sha(control) and sha(weights) == roots["weights_sha256"]
            and f["select_information_probe"], "unbound prospective prediction")
    ordered = sorted(((p, read(p)) for p in samples), key=lambda t: t[1]["seed"])
    observations = [r for _, r in ordered]
    require([r["seed"] for r in observations] == roots["seeds"], "unregistered prospective seeds")
    methods = probe.calibration_methods(checkout / SOURCE)
    by_seed = []
    for r in observations:
        require(r["schema"] == roots["observation_schema"] and r["inventory"] == methods and r["forecast_sha256"] == sha(forecast_path)
                and r["contract_sha256"] == sha(CONTRACT) and r["weights_sha256"] == sha(weights)
                and r["source_sha256"] == roots["source_sha256"] and r["parent_order"] == roots["parent_order"]
                and r["selection_before"] == r["selection_after"] == "candidate" and r["control_receipt_sha256"] == sha(control)
                and r["profiler_sha256"] == sha(Path(probe.__file__)), "foreign holdout capture")
        check_selection(r["selection"], weights, ctrl, roots)
        arm, bind = r["arm"], ctrl["qualification"]["binding"]
        restart_trial.validate(arm, r["seed"], expected_source=roots["source_sha256"])
        restart_trial.verify_binding(arm, ctrl["order"])
        require(arm["binding"] == bind["runtime"] and arm["reused"] and arm["cuda_vendor"] == bind["vendor"]
                and arm["mask_bits"] == bind["mask_bits"], "foreign workload or target")
        values = {}
        for cal in r["calibrations"]:
            name = cal["method"]
            require(name in methods and name not in values and cal["method_sha256"] == methods[name]["method_sha256"]
                    and set(cal["direct_field_values"]) == set(methods[name]["direct_python_writes"]) and
                    0 < cal["elapsed_s"] <= arm["initialized_s"], "incomplete optional calibration")
            values[name] = cal
        by_seed.append(values)
    outcome = []
    for item in f["alternatives"]:
        name, field = item["method"], item.get("archive_field")
        captured = [v[name] for v in by_seed if name in v]
        outputs, side = [], []
        for obs, values in zip(observations, by_seed, strict=True):
            if name not in values:
                continue
            cal = values[name]
            ret = cal["return_value"]
            payload = DESTINATIONS[field][1] if field else None
            if payload:
                ret = ret["fields"][payload]
            if field:
                require(ret == obs["arm"]["startup_prices"][field], "observed return differs from the delivered runtime")
            outputs.append(ret)
            side.append(cal["direct_field_values"])
        if field:
            require(len(captured) == 2, "measured calibration absent in held-out data")
        prediction = item["predicts_exact_match"]
        matches = [v == item["prediction"] for v in outputs] if prediction else []
        outcome.append({"method": name, "historical_forecast": item["prediction"], "prediction_applicable": prediction,
                        "exact_matches": matches, "decision": "confirmed two observed return values" if matches == [True, True]
                        else "prediction refuted" if matches else "no exact-return forecast",
                        "return_sha256": [digest(v) for v in outputs], "direct_state_sha256": [digest(v) for v in side],
                        "direct_state_identical": len(side) == 2 and side[0] == side[1],
                        "elapsed_s": [x["elapsed_s"] for x in captured], "unobserved_native_state": True})
    elapsed = sum(r["broker_elapsed_s"] for r in observations)
    return {"schema": "ironmule.calibration-stability-holdout.v1", "forecast_sha256": sha(forecast_path),
            "holdout_sha256": [sha(p) for p, _ in ordered], "control_sha256": sha(control), "weights_sha256": sha(weights),
            "seeds": roots["seeds"], "outcomes": outcome, "measured_s": elapsed, "attributable_lease_usd": elapsed * f["hourly_usd"] / 3600,
            "authoring_eligible": False, "training_label": False, "activation": None,
            "conclusion": "The registered observations test exact return-value prediction. Dynamic prices, indirect/native state and target release effects require independent evidence."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("forecast", "holdout"))
    for k in ("control", "checkout", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("history", "forecast", "weights"):
        p.add_argument("--" + k, type=Path)
    for k in ("observation", "phase-prior"):
        p.add_argument("--" + k, type=Path, action="append")
    p.add_argument("--hourly-usd", type=float)
    a = p.parse_args()
    if a.mode == "forecast":
        result = forecast(a.history, a.control, a.checkout, a.phase_prior, a.hourly_usd)
    else:
        result = heldout(a.forecast, a.observation, a.control, a.weights, a.checkout)
    write_once(a.out, result)


if __name__ == "__main__":
    main()
