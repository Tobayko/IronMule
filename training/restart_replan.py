"""Plan one information experiment from measured serving after a qualified startup release."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from lab.pk1 import prefill_verdict, restart_order, restart_release
from lab.pk1.native_trial import checksum
from training import scoped_release
from training.code_author import digest, write_once
from training.code_trial import read


def past_attempts(archive, order):
    family = order["binding"]
    # The canonical CUDA observation was the parent of both the W4 candidate and Lt screen.
    dense = archive / "a100-prefill-lt"
    plan, raw = read(dense / "plan.json"), read(dense / "measurement.json")
    negative = prefill_verdict.decide(plan, raw)
    if negative != read(dense / "decision.json") or any(
        plan["binding"][k] != family[k]
        for k in ("fp", "gguf", "native", "w4_sha256", "table_sha256")
    ):
        raise ValueError("unrelated or changed prefill history")
    folder = archive / "a100-native" / plan["parent_order"]
    kernel, screen = read(folder / "order.json"), read(folder / "screen.json")
    if (
        kernel["order_id"] != restart_order.order_digest(kernel)
        or kernel["order_id"] not in order["evidence"]["local_profile_ids"]
        or checksum(folder / "order.profile.json") != kernel["profile_sha256"]
        or read(folder / "result.json")["passed"] is not False
        or screen["model"] != family["gguf"]
        or screen["table"] != family["table_sha256"]
        or screen["reference"] != family["w4_sha256"]
        or screen["all_elements_identical"] is not True
    ):
        raise ValueError("incompatible native negative experience")
    worst = max(
        screen["measurements"],
        key=lambda r: r["arms"][1]["microseconds"] / r["arms"][0]["microseconds"],
    )
    return {
        "native_order": kernel["order_id"],
        "native_screen_sha256": checksum(folder / "screen.json"),
        "worst_width32_shape": worst["shape"],
        "variant": worst["variant"],
        "matched_microseconds": [r["microseconds"] for r in worst["arms"]],
        "native_note": "Local numerical screen only. Paired serving did not qualify this candidate.",
        "prefill_receipt_sha256": checksum(dense / "measurement.json"),
        "prefill_history": negative,
    }


def rank(data, historic, history, cost):
    """Model assumptions price diagnostic candidates; they never qualify code changes."""
    cold = median(data["http_cold_s"])
    required = cold * (1 - 1 / 1.02)
    warm_required = median(data["http_mixed_warm_s"]) * (1 - 1 / 1.02)
    # A conservative threshold from the existing native planner. Keep it fixed across phases.
    fraction = 0.10
    worker_residual = min(
        median(data["native_init_s"]), median(data["http_request_to_worker_ready_s"])
    )
    times = {
        k: median(v)
        for k, v in historic.items()
        if k in ("check_argmax", "warm_up", "measure_passes", "measure_padding")
    }
    diag_s = max(data["native_factual_worker_elapsed_s"])
    rate = cost["quoted_compute_usd_per_h"] + cost["disk_rate_estimate_usd_per_h"]
    estimate_usd = diag_s * rate / 3600
    rows = []
    for phase, old_s in times.items():
        conditional = min(old_s, worker_residual) * fraction
        rows.append(
            {
                "experiment": "time-active-startup-phase",
                "phase": phase,
                "historical_other_run_s": old_s,
                "active_phase_time_s": None,
                "hypothetical_10pct_saving_s": conditional,
                "would_cross_2pct_cold": conditional >= required,
                "decision_value_per_probe_s": conditional / diag_s,
            }
        )
    rows.sort(
        key=lambda v: (v["would_cross_2pct_cold"], v["decision_value_per_probe_s"]),
        reverse=True,
    )
    chooser_s = median(data["cpu_decision_ms"]) / 1000
    best = rows[0] if rows else None
    unresolved = bool(best and best["would_cross_2pct_cold"])
    next_test = {
        "decision": "collect one passive startup trace"
        if unresolved
        else "defer further experimentation",
        "selected_phase": best["phase"] if unresolved else None,
        "book_additional_resources": False,
        "maximum_new_paid_leases": 0,
        "maximum_extra_author_calls": 0,
        "when": "next already planned maintenance startup of this exact active profile"
        if unresolved
        else None,
        "estimated_full_worker_seconds": diag_s,
        "estimated_equivalent_lease_usd": estimate_usd,
        "collection": "In a separate diagnostic worker use IRONMULE_GGML_TRACE in its private writable directory. Time the marked phase, capture cold/warm completed work and keep diagnostic timing outside qualification."
        if unresolved
        else None,
        "acceptance_for_new_order": "Need an actual trace of the active source on this GPU, a reproducible phase cost, and a separate quality-preserving treatment with headroom above the full-serving gate. One exploratory trace cannot promote code.",
    }
    return {
        "next": next_test,
        "ranked_information_experiments": rows,
        "current_correct_cold_s": cold,
        "unattributed_worker_init_s": worker_residual,
        "current_2pct_cold_s": required,
        "current_2pct_mixed_warm_s": warm_required,
        "realized_startup_choice_ms": 1000 * chooser_s,
        "sensitivity_fraction": fraction,
        "retired": [
            {
                "target": "identical W4 calibration-reuse change",
                "reason": "already served, merged and exercised on this binding",
            },
            {
                "target": "previous W4 CUDA variant",
                "width32_microseconds": history["matched_microseconds"],
                "reason": "recorded local slowdown and no qualified whole-serving benefit",
            },
            {
                "target": "re-run same isolated Lt screen",
                "best_projected_s": history["prefill_history"]["optimistic_best_sum_ms"]
                / 1000,
                "required_s": warm_required,
                "reason": "screen already completed; a new proposal requires new measured evidence",
            },
            {
                "target": "remove startup CPU selection",
                "upper_bound_s": chooser_s,
                "required_s": required,
                "could_meet_threshold": chooser_s >= required,
            },
        ],
        "uncertainty": "Phase durations come from historical K1/K2. New residual initialization and HTTP latencies are observed in different processes and are not subtracted. Predicted savings are sensitivity estimates.",
    }


def completed_phase_probe(archive, order, control, head, receipt_sha):
    """Recover completed source attribution, preserving the external GPU-load exclusion."""
    folder = archive / "a100-post-restart-native-60"
    if not folder.exists():
        return None
    decision = read(folder / "decision.json")
    provenance = decision["verification"]
    incident = folder / "product-nested-diagnostic.json"
    if (
        decision["schema"] != "ironmule.post-restart-native-deferred.v1"
        or decision["parent_order"] != order["order_id"]
        or decision["weights_sha256"] != head
        or decision["control_receipt_sha256"] != receipt_sha
        or decision["source_sha256"] != control["qualification"]["candidate_sha256"]
        or decision["base_commit"] != control["publication"]["candidate"]
        or decision["selected"] is not None
        or any(
            decision[key] is not False
            for key in ("timing_eligible", "training_eligible", "authoring_eligible")
        )
        or decision["confounder"]["sha256"] != checksum(incident)
    ):
        raise ValueError("invalid diagnostics or conflict provenance")
    hashes = [checksum(folder / f"{seed}.json") for seed in provenance["seeds"]]
    if (
        len(hashes) != 2
        or hashes != provenance["observation_sha256"]
        or hashes != decision["observations"]
    ):
        raise ValueError("partial active-source diagnosis")
    summary = {
        "decision_sha256": checksum(folder / "decision.json"),
        "observation_sha256": hashes,
        "incident_sha256": checksum(incident),
        "observed_cells": [v["cell"] for v in provenance["selected_cells"]],
        "timing_eligible": False,
        "training_eligible": False,
        "reason": decision["confounder"]["reason"],
    }
    # The later sampling experiment used prospectively held-out, uncontended source
    # observations and developed its own program. Never transfer its timing to the
    # conflicting native traces above.
    study = archive / "a100-calibration-sampling"
    closed = study / "post-terminal-replan.json"
    if closed.is_file():
        from training.calibration_replan import plan as regrade

        source_order = read(study / "decision.json")
        raw = study / ("trial-" + source_order["order_id"])
        parent = archive / "a100-restart" / order["order_id"] / "control.json"
        if checksum(parent) != receipt_sha or source_order["weights_sha256"] != head:
            raise ValueError("different physical experiment or deployment")
        graded = regrade(raw, source_order, parent)
        if (
            graded != read(closed)
            or graded["activate"] is not False
            or graded["retry_order"] is not False
        ):
            raise ValueError("unverified calibration trial disposition")
        summary["closed_sampling_experiment"] = {
            "order_id": source_order["order_id"],
            "source_sha256": graded["source_sha256"],
            "result_sha256": checksum(raw / "result.json"),
            "replan_sha256": checksum(closed),
            "decision": graded["development_decision"],
            "information": graded["information_decisions"],
            "regular_pairs": len(graded["validated_regular_pairs"]),
            "actual_adaptive": graded["actual_adaptive_regimes"],
            "full_fallback": graded["actual_full_fallback_regimes"],
        }
    return summary


def verified_release(state, checkout):
    deployed, trial = state / "release", state / "trial"
    published = read(deployed / "published.json")
    control = scoped_release.pinned(deployed, published["receipt_sha256"])
    order = read(state / "order.json")
    if control["order"] != order or control[
        "qualification"
    ] != restart_release.approved(
        SimpleNamespace(
            trial=trial,
            checkout=checkout,
            order=state / "order.json",
            candidate_source=state / "authoring/candidate.py",
        )
    ):
        raise ValueError("source release differs from qualified trial")
    return published, control, order


def declined(state, *, checkout, archive=None, product_dir=None):
    from lab.pk1 import restart_http

    state = Path(state)
    product = Path(product_dir) if product_dir is not None else state / "product"
    archive = (
        Path(archive)
        if archive
        else Path(__file__).resolve().parents[1] / "lab/autonomous/results"
    )
    published, control, order = verified_release(state, checkout)
    observed = read(product / "product-qualify.json")
    if observed.get("passed") is not False or observed != read(
        product / "decision.json"
    ):
        raise ValueError("need a saved rejected product qualification")
    history = past_attempts(archive, order)
    names = [
        f"{name}-{seed}.json"
        for seed in restart_http.SEEDS
        for name in ("reference", "candidate")
    ]
    present = {
        name: checksum(product / name) for name in names if (product / name).is_file()
    }
    complete = len(present) == len(names)
    rows = []
    if complete and "pairs" in observed:
        graded = restart_http.assess(product, control)
        if graded != observed:
            raise ValueError(
                "product rejection differs from independently graded pairs"
            )
        rows = observed["pairs"]
    next_test = {
        "decision": "retain reference product route and diagnose rejected gate",
        "book_additional_resources": False,
        "maximum_new_paid_leases": 0,
        "maximum_extra_author_calls": 0,
        "evidence_needed": "Inspect the failed check against its fixed workload and runtime. Capture one instrumented session only on the next already planned maintenance window; no candidate activation or repeat of prior code variants.",
    }
    metrics = {k: median(x[k] for x in rows) for k in rows[0]} if rows else {}
    result = {
        "schema": "ironmule.restart-replan-declined.v1",
        "provenance": "planning from a rejected product gate",
        "order": order["order_id"],
        "source_receipt": published["receipt_sha256"],
        "native_result_sha256": checksum(state / "trial/result.json"),
        "cost_observation_sha256": checksum(state / "cost-after-verdict.json"),
        "product_verdict_sha256": checksum(product / "product-qualify.json"),
        "pair_count": len(rows),
        "completed_raw_files": present,
        "raw_medians": metrics,
        "reason": observed.get(
            "reason", "completed HTTP pairs outside predeclared bounds"
        ),
        "retained_measured_history": history,
        "next": next_test,
        "new_serving_improvement": None,
    }
    result["plan_id"] = digest(result)
    write_once(
        (state if product == state / "product" else product)
        / "replan-after-product-declined.json",
        result,
    )
    return result


def replan(state, *, checkout, archive=None, product_dir=None):
    from lab.pk1 import restart_http

    state = Path(state)
    archive = (
        Path(archive)
        if archive
        else Path(__file__).resolve().parents[1] / "lab/autonomous/results"
    )
    trial = state / "trial"
    product = Path(product_dir) if product_dir is not None else state / "product"
    published, control, order = verified_release(state, checkout)
    verdict = restart_http.assess(product, control)
    served, rollback = (
        read(product / "product-serve.json"),
        read(product / "product-exercise.json"),
    )
    if (
        verdict != read(product / "product-qualify.json")
        or verdict["passed"] is not True
        or served["arm"] != "candidate"
        or rollback["passed"] is not True
        or rollback["before"]["arm"] != "candidate"
        or rollback["after"]["arm"] != "reference"
        or rollback["before"]["work_sha256"] != rollback["after"]["work_sha256"]
        or served["completions"][-1]["graded_correct"] != 18
    ):
        raise ValueError(
            "need actual passed product, serving and rollback measurements"
        )
    rows = [read(product / f"candidate-{seed}.json") for seed in restart_http.SEEDS]
    native = [read(trial / f"facts-cand-{i}.json") for i in range(2)]
    head = checksum(state / "startup-head.npz")
    if not all(
        a["startup_decision"]["profile"] == "candidate"
        and a["startup_decision"]["weights_sha256"] == head
        for a in rows
    ):
        raise ValueError("product observations came from a different startup policy")
    data = {
        "http_cold_s": [a["cold_completed_s"] for a in rows],
        "http_request_to_worker_ready_s": [a["startup_s"] for a in rows],
        "http_mixed_warm_s": [a["perf_wall_s"] for a in rows],
        "cpu_decision_ms": [a["startup_decision"]["cpu_ms"] for a in rows],
        "native_init_s": [a["initialized_s"] for a in native],
        "native_factual_worker_elapsed_s": [a["broker_elapsed_s"] for a in native],
    }
    if not all(all(math.isfinite(v) and v > 0 for v in vs) for vs in data.values()):
        raise ValueError("missing measured positive duration")
    history = past_attempts(archive, order)
    # If final billing is pending always use this immutable comparable already-metered baseline;
    # a newer after-product record belongs to actual accounting, never to the forecast.
    cost_path = state / "cost-after-verdict.json"
    cost = read(cost_path)
    if cost["order_id"] != order["order_id"] or cost["phase"] != "after-verdict":
        raise ValueError("cost observation belongs to another experiment")
    result = {
        "schema": "ironmule.restart-replan.v1",
        "provenance": "planning after measured deployment",
        "order": order["order_id"],
        "source_receipt": published["receipt_sha256"],
        "cpu_head": head,
        "grade_sha256": checksum(product / "product-qualify.json"),
        "served_sha256": checksum(product / "product-serve.json"),
        "rollback_sha256": checksum(product / "product-exercise.json"),
        "counterbalanced_native_sha256": checksum(trial / "result.json"),
        "cost_observation_sha256": checksum(cost_path),
        "cost_phase": cost["phase"],
        "lease_estimate_at_source_verdict_usd": cost["estimated_lease_usd_before_tail"],
        "provider_delta_at_source_verdict_usd": cost[
            "provider_account_delta_since_lease_start_usd"
        ],
        "retained_campaign_reservation_usd": cost["campaign_reserved_usd"],
        "hardware": order["binding"]["fp"]["devices"],
        "observed": data,
        "negative_history": history,
    }
    result.update(
        rank(data, order["evidence"]["historical_startup_seconds"], history, cost)
    )
    probe = completed_phase_probe(
        archive, order, control, head, published["receipt_sha256"]
    )
    if probe is not None:
        result["completed_phase_probe"] = probe
        result["retired"].append(
            {
                "target": "repeat nested startup call and device-symbol inventory",
                "reason": "Source calls and device symbols have already been identified.",
                "observation_sha256": probe["observation_sha256"],
            }
        )
        result["next"] = {
            "decision": "inspect archived complete calibration results and bound withheld starts",
            "selected_phase": None,
            "book_additional_resources": False,
            "maximum_new_paid_leases": 0,
            "maximum_extra_author_calls": 0,
            "estimated_extra_lease_usd": 0,
            "evidence_needed": "A fresh local complete calibration result, all observable side effects, repeatability "
            "on independent held-out starts, measured full-work headroom and a qualified fallback "
            "are required before any further candidate. The concurrent diagnostic supplies source "
            "ownership and device symbols only.",
        }
        if experiment := probe.get("closed_sampling_experiment"):
            result["retired"].append(
                {
                    "target": "repeat rejected adaptive sampling source",
                    "order": experiment["order_id"],
                    "result_sha256": experiment["result_sha256"],
                }
            )
            result["next"] = {
                "decision": experiment["decision"],
                "book_additional_resources": False,
                "maximum_new_paid_leases": 0,
                "maximum_extra_author_calls": 0,
                "estimated_extra_lease_usd": 0,
                "evidence": experiment["information"],
            }
    result["plan_id"] = digest(result)
    write_once(
        (state if product == state / "product" else product)
        / "replan-after-product.json",
        result,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--checkout", required=True, type=Path)
    args = p.parse_args()
    verdict = read(args.state / "product/product-qualify.json")
    planner = replan if verdict["passed"] else declined
    print(planner(args.state, checkout=args.checkout)["next"])
