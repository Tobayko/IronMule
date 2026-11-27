"""Cross-study replay of original PR60 calibration vectors and low-cost sentinel observations."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from statistics import median

from lab.pk1 import followup_contract as rules
from lab.pk1 import followup_trial as trial
from training.calibration_stability import archived_arms, sha
from training.code_author import digest, write_once
from training.code_trial import read
from training.trial_release import require

FAMILIES = ("padding", "passes", "pilot")
METHODS = {
    "padding": "measure_padding",
    "passes": "measure_pass_costs",
    "pilot": "measure_costs",
}


def original(path, control, ctx=None):
    raw = read(path)
    r = raw.get("arm", raw)
    if ctx:
        require(
            r["source_label"] == "reference" and not r["sampling"]["screen"],
            "only uninstrumented original reference arms",
        )
        trial.validate(
            r, "reference", r["seed"], ctx, control, bool(r["factual_first"])
        )
        pad = r["sampling"]["observed"]["startup"]["prices"]
    else:
        require(
            r["source_sha256"] == control["qualification"]["candidate_sha256"]
            and r["reused"]
            and raw["selection_before"] == "candidate"
            and raw["selection_after"] == "candidate"
            and r["facts"] == [True] * 18,
            "original observed PR60 worker required",
        )
        pad = next(a for a in raw["calibrations"] if a["method"] == METHODS["padding"])[
            "direct_field_values"
        ]["pad_prices"]
    require(
        r["binding"] == control["qualification"]["binding"]["runtime"],
        "different complete runtime",
    )
    prices = {
        "padding": pad,
        "passes": {str(k): v for k, v in r["startup_prices"]["passes"]},
        "pilot": r["startup_prices"]["pilot"],
    }
    return {
        "seed": r["seed"],
        "sha256": sha(path),
        "family": "factual-first" if r["factual_first"] else "mixed-first",
        "complete": prices,
        "dispatch": r["startup_prices"]["padding"],
        "binding": digest(r["binding"]),
        "startup_s": r["initialized_s"],
        "worker_s": r["broker_elapsed_s"],
    }


def costs(paths):
    estimates = {family: [] for family in FAMILIES}
    for path in paths:
        x = read(path)
        for family, method in METHODS.items():
            row = next(v for v in x["calibrations"] if v["method"] == method)
            elapsed = row["elapsed_s"]
            raw = row["timing_samples_by_width"]
            # Retain every originally observed warm and measurement call for the minimal width.
            # Charge the entire measured setup/cleanup and all other non-timed overhead again.
            # The Pilot has no separately observed one-field probe, so charge its full method.
            if raw:
                first = min(raw, key=int)
                budget = elapsed - sum(sum(v) for v in raw.values()) + sum(raw[first])
            else:
                budget = elapsed
            require(0 < budget <= elapsed, "incomplete source timing")
            estimates[family].append((elapsed, budget))
    return {
        k: {
            "min_full_s": min(x for x, _ in v),
            "max_sentinel_setup_and_samples_s": max(y for _, y in v),
            "lower_historical_opportunity_s": min(x - y for x, y in v),
            "sentinel": "step_one (entire method charged)"
            if k == "pilot"
            else "width 1; real full observation used only in offline replay",
        }
        for k, v in estimates.items()
    }


def distance(reference, target, subset, *, sentinels=False):
    errors = {}
    for family in subset:
        a, b = reference["complete"][family], target["complete"][family]
        require(a.keys() == b.keys(), "incompatible calibration shape")
        keys = [("step_one" if family == "pilot" else "1")] if sentinels else b.keys()
        errors[family] = max(abs(a[k] / b[k] - 1.0) for k in keys)
    return max(errors.values()), errors


def prediction(bank, row, subset):
    ref = min(
        bank, key=lambda x: (distance(x, row, subset, sentinels=True)[0], -x["seed"])
    )
    sentinel, _ = distance(ref, row, subset, sentinels=True)
    err, details = distance(ref, row, subset)
    return {
        "seed": row["seed"],
        "family": row["family"],
        "single_original_profile_seed": ref["seed"],
        "sentinel_relative_drift": sentinel,
        "complete_max_relative_error": err,
        "per_component_relative_error": details,
        "within_two_percent": err <= 0.02,
        "padding_action_matches": ref["dispatch"] == row["dispatch"]
        if "padding" in subset
        else None,
    }


def assess(data, parent_path):
    q = read(parent_path)
    base = data / "a100-calibration-stability"
    follow = data / "a100-calibration-sampling"
    d = follow / ("trial-" + read(follow / "decision.json")["order_id"])
    history = [base / f"{s}.json" for s in (97251, 97252)]
    calibration = [follow / f"{s}.json" for s in (97351, 97352)]
    ctrl_root = parent_path.parent
    # These two original study captures and the source-specific held-out qualification are independently reproducible.
    require(
        sha(parent_path) == rules.CONTROL
        and set(read(base / "holdout.json")["holdout_sha256"])
        == {sha(p) for p in history}
        and set(read(follow / "decision.json")["observations"])
        == {sha(p) for p in calibration}
        and sha(ctrl_root / "startup-head.npz") == rules.HEAD,
        "different two source studies or trained controller",
    )
    verdict = read(d / "result.json")
    require(
        verdict["passed"] is False
        and verdict["context"]["order"] == read(follow / "decision.json")["order_id"],
        "unrelated or incomplete source study",
    )
    ctx = read(d / "context.json")
    files = [
        d / f"{group}-reference-{s}.json"
        for group, seeds in (("mixed", rules.MIXED), ("facts", rules.FACTS))
        for s in seeds
    ]
    require(
        all(sha(f) == verdict["artifacts"][f.name] for f in files),
        "changed held-out reference arm",
    )
    train, cal, test = (
        [original(p, q) for p in history],
        [original(p, q) for p in calibration],
        [original(p, q, ctx) for p in files],
    )
    price = costs([*history, *calibration])
    trials = []
    for size in range(1, len(FAMILIES) + 1):
        for subset in itertools.combinations(FAMILIES, size):
            exam = [prediction(train, r, subset) for r in cal]
            # The calibration study alone fixes limits. The third study is never in the donor bank.
            drift = max(s["sentinel_relative_drift"] for s in exam)
            envelope = max(s["complete_max_relative_error"] for s in exam)
            out = [prediction(train, r, subset) for r in test]
            for raw, row in zip(out, test, strict=True):
                raw["within_empirical_calibration_envelope"] = (
                    raw["complete_max_relative_error"] <= envelope
                )
                raw["sentinel_accepts_observed_price_regime"] = (
                    raw["sentinel_relative_drift"] <= drift
                )
                raw["fixed_latest_training_profile_error"] = distance(
                    train[-1], row, subset
                )[0]
                raw["fixed_latest_pretrial_profile_error"] = distance(
                    cal[-1], row, subset
                )[0]
            admitted = [r for r in out if r["sentinel_accepts_observed_price_regime"]]
            net = sum(price[k]["lower_historical_opportunity_s"] for k in subset)
            trials.append(
                {
                    "profile_components": list(subset),
                    "calibration_study": exam,
                    "held_out_study": out,
                    "calibrated_sentinel_drift_limit": drift,
                    "calibrated_complete_error_envelope": envelope,
                    "empirical_2pct_hits_all": sum(
                        r["within_two_percent"] for r in out
                    ),
                    "empirical_2pct_fixed_training_hits_all": sum(
                        r["fixed_latest_training_profile_error"] <= 0.02 for r in out
                    ),
                    "empirical_2pct_fixed_pretrial_hits_all": sum(
                        r["fixed_latest_pretrial_profile_error"] <= 0.02 for r in out
                    ),
                    "empirical_envelope_coverage": sum(
                        r["within_empirical_calibration_envelope"] for r in out
                    ),
                    "sentinel_accepted": len(admitted),
                    "sentinel_accepted_2pct_failures": sum(
                        not r["within_two_percent"] for r in admitted
                    ),
                    "max_full_vector_relative_error": max(
                        r["complete_max_relative_error"] for r in out
                    ),
                    "historical_full_method_minus_sentinel_opportunity_s": net,
                }
            )
    viability = [
        t
        for t in trials
        if t["empirical_2pct_hits_all"] == len(test)
        and t["historical_full_method_minus_sentinel_opportunity_s"] > 0
    ]
    best = (
        max(
            viability,
            key=lambda t: t["historical_full_method_minus_sentinel_opportunity_s"],
        )
        if viability
        else None
    )
    typical = median(r["worker_s"] for r in test)
    rate = read(follow / "prediction.json")["hourly_usd"]
    # A separate, earlier PR60 study captured complete pass/pilot vectors, but omitted the
    # padding prices. Never fill this missing component from another worker or infer its age.
    reverse_stress = {}
    older = archived_arms(
        parent_path.parent, q, {"source_sha256": q["qualification"]["candidate_sha256"]}
    )
    for family in ("passes", "pilot"):
        group = next(p for p in trials if p["profile_components"] == [family])
        rows = []
        for name, raw in older:
            observed = raw["startup_prices"][family]
            values = (
                {str(k): v for k, v in observed} if family == "passes" else observed
            )
            row = {
                "seed": raw["seed"],
                "family": "factual-first" if raw["factual_first"] else "mixed-first",
                "complete": {family: values},
                "dispatch": raw["startup_prices"]["padding"],
            }
            choice = prediction(train, row, (family,))
            rows.append(
                {
                    "file": name,
                    "sha256": sha(parent_path.parent / name),
                    **choice,
                    "sentinel_accepts_observed_price_regime": choice[
                        "sentinel_relative_drift"
                    ]
                    <= group["calibrated_sentinel_drift_limit"],
                }
            )
        reverse_stress[family] = rows
    net_after_fallback = None
    if best:
        n = len(best["held_out_study"])
        admitted = best["sentinel_accepted"]
        sentinel = sum(
            price[k]["max_sentinel_setup_and_samples_s"]
            for k in best["profile_components"]
        )
        net_after_fallback = (
            admitted * best["historical_full_method_minus_sentinel_opportunity_s"]
            - (n - admitted) * sentinel
        ) / n
    return {
        "schema": "ironmule.calibration-portfolio-research.v1",
        "control_sha256": sha(parent_path),
        "negative_source_result_sha256": sha(d / "result.json"),
        "base_commit": rules.BASE,
        "cohorts": {
            "donors": [r["seed"] for r in train],
            "calibrate_intervals": [r["seed"] for r in cal],
            "single_held_out_physical_study": {
                "mixed": list(rules.MIXED),
                "factual_first": list(rules.FACTS),
            },
        },
        "observations": [
            {k: r[k] for k in ("seed", "sha256", "family", "binding")}
            for r in [*train, *cal, *test]
        ],
        "historical_measured_costs": price,
        "portfolios": trials,
        "older_separate_regime_stress": reverse_stress,
        "references": {
            "fresh_current_PR60": "All three calibrations run, full observed output and measured native state.",
            "always_empty_padding_correct": sum(not r["dispatch"] for r in test),
            "fixed_complete_row": "A single unmodified prior worker, no component averaging. Reported for the last donor and the most recent pretrial worker.",
        },
        "new_information_question": None
        if not best
        else {
            "components": best["profile_components"],
            "replay_net_s_per_start_after_paying_rejected_sentinels": net_after_fallback,
            "qualification_only_break_even_with_observed_abstentions": math.ceil(
                22 * typical / net_after_fallback
            )
            if net_after_fallback and net_after_fallback > 0
            else None,
            "question": "Can a fresh bounded sentinel identify a transferable full original profile in a prospectively registered change of operating conditions, while retaining the incumbent's warmed native state and identical downstream work?",
            "max_observed_component_error": best["max_full_vector_relative_error"],
            "conditional_phase_opportunity_s": best[
                "historical_full_method_minus_sentinel_opportunity_s"
            ],
            "lower_bound_crossover_for_22_workers": math.ceil(
                22
                * typical
                / best["historical_full_method_minus_sentinel_opportunity_s"]
            ),
            "price_for_two_full_original_shadow_workers_s": 2 * typical,
            "price_for_two_full_original_shadow_workers_usd": 2 * typical * rate / 3600,
            "reference_plus_counterfactual_pair_needed_for_end_to_end": True,
        },
        "source_order": None,
        "activation": None,
        "new_paid_action": None,
        "limitations": [
            "Retrospective study-level holdout, inspected after the scientific rejection. Eight test workers share one physical acquisition; they are not eight independent drift regimes.",
            "The 2% line is the earlier pad-price guard and a descriptive sensitivity for other price vectors, not an established dispatch tolerance for Pilot or PassCosts.",
            "Unseen loads, clock state, exact donor age and telemetry first-read times are unavailable. A production selector abstains when those conditions are unknown. Factual-first was absent from training.",
            "One observed calibration study does not yield a 95% distribution-free prediction interval. The empirical error envelope and acceptance coverage are descriptive.",
            "Sentinels are read from full measurements here; standalone runtime cost, context/graph warming, KV, interaction and honest aged telemetry would require a distinct real test.",
            "No packed-wave width choices, Pilot logits or per-decision features establish causality. Constant zero padding matches the observed action. No validated gain over a fixed prior rule or full current PR60 serving.",
        ],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("data", "control", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    write_once(a.out, assess(a.data, a.control))
