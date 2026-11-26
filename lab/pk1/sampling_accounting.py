"""Account for a completed sampling trial on CPU from its exported, independently checkable target records."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import median

from lab.pk1 import followup_contract as protocol
from lab.pk1 import followup_trial as trial
from lab.pk1 import restart_trial
from training import sampling_evidence
from training.code_author import write_once
from training.code_trial import read
from training.trial_release import require


def work(row):
    first = row["factual_first"]
    return {
        "initial_factual_deliveries": sum(len(b["answers"]) for b in first),
        "initial_factual_correct": sum(sum(b["correct"]) for b in first),
        "cold_mixed_deliveries": row["cold"]["completed"],
        "cold_mixed_generated_tokens": row["cold"]["tokens"],
        "warm_mixed_deliveries": row["warm"]["completed"],
        "warm_mixed_generated_tokens": row["warm"]["tokens"],
        "factual_deliveries": len(row["factual"]),
        "factual_correct": sum(row["facts"]),
    }


def arm(row):
    prefix = row["broker_to_process_s"]
    batch = row["factual_first"][0] if row["factual_first"] else row["cold"]
    elapsed = prefix + (batch["completed_s"] if row["factual_first"] else row["cold_s"])
    warm, facts = row["warm"], row["factual_timing"]
    first_warm = row["factual_first"][1] if row["factual_first"] else None
    return {
        "seed": row["seed"],
        "source": row["source_label"],
        "work": work(row),
        "first_complete_wave_s": elapsed,
        "mixed_complete_s": prefix + row["cold_s"],
        "broker_to_worker_start_s": prefix,
        "startup_profile_cpu_ms": row["startup_selection"]["cpu_ms"],
        "startup_profile_inference_cpu_us": row["startup_selection"]["cpu_us"],
        "first_wave_absolute_p95_ttft_s": elapsed
        - batch["wall_s"]
        + sorted(batch["ttfts_s"])[math.ceil(0.95 * len(batch["ttfts_s"])) - 1],
        "warm_mixed_wall_s": warm["wall_s"],
        "warm_mixed_p95_ttft_s": sorted(warm["ttfts_s"])[30],
        "warm_factual_wall_s": facts["wall_s"],
        "warm_factual_p95_ttft_s": restart_trial.fact_timing(facts),
        **(
            {
                "second_factual_wave_s": first_warm["wall_s"],
                "second_factual_wave_p95_s": restart_trial.fact_timing(first_warm),
            }
            if first_warm
            else {}
        ),
        "worker_end_to_end_s": row["broker_elapsed_s"],
        "peak_rss_kib": row["peak_rss_kib"],
        "max_observed_gpu_bytes": max(row["device_used_bytes"]),
        "startup_sampling": sampling_evidence.sampling_status(row),
    }


def complete_deliveries_per_second(source):
    return (
        sum(
            source["work"][k]
            for k in (
                "cold_mixed_deliveries",
                "warm_mixed_deliveries",
                "factual_deliveries",
                "initial_factual_deliveries",
            )
        )
        / source["worker_end_to_end_s"]
    )


def grouped(group, seeds, rows, accepted):
    paired = []
    invalid = {}
    for seed in seeds:
        left = rows.get(f"{group}-reference-{seed}.json")
        right = rows.get(f"{group}-candidate-{seed}.json")
        if left is None or right is None:
            continue
        try:
            trial.scores(left, right)
            require(work(left) == work(right), "different count of delivered work")
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            invalid[str(seed)] = str(exc)
            continue
        paired.append((arm(left), arm(right)))
    result = {
        "planned_pairs": len(seeds),
        "comparable_pairs": len(paired),
        "failed_pair_checks": invalid,
        "full_cohort": len(paired) == len(seeds),
        "qualifies_for_claim": accepted and len(paired) == len(seeds),
    }
    if not paired:
        return {**result, "medians": None, "work_per_source": None}
    sums = {k: sum(a["work"][k] for a, _ in paired) for k in paired[0][0]["work"]}
    keys = [
        k for k, v in paired[0][0].items() if type(v) in (int, float) and k != "seed"
    ]
    return {
        **result,
        "seeds": [a["seed"] for a, _ in paired],
        "medians": {
            k: {
                "reference": median(a[k] for a, _ in paired),
                "candidate": median(b[k] for _, b in paired),
            }
            for k in keys
        },
        "work_per_source": sums,
        "median_paired_cold_speedup": median(
            a["first_complete_wave_s"] / b["first_complete_wave_s"] for a, b in paired
        ),
        "median_paired_cold_wall_saved_s": median(
            a["first_complete_wave_s"] - b["first_complete_wave_s"] for a, b in paired
        ),
        "median_paired_worker_wall_saved_s": median(
            a["worker_end_to_end_s"] - b["worker_end_to_end_s"] for a, b in paired
        ),
        "equal_work_throughput": {
            "unit": "complete mixed requests plus independently graded factual requests per whole broker worker second",
            "median_reference": median(
                complete_deliveries_per_second(a) for a, _ in paired
            ),
            "median_candidate": median(
                complete_deliveries_per_second(b) for _, b in paired
            ),
        },
    }


def account(folder, data):
    verified = sampling_evidence.replay(folder, data)
    require(
        verified["qualified"] in (True, False) and (folder / "result.json").is_file(),
        "a completed physical verdict is required",
    )
    verdict = read(folder / "result.json")
    order = read(folder / "order.json")
    parent = read(data / "a100-restart" / order["parent_order"] / "control.json")
    prior = read(data / "a100-calibration-stability" / "holdout.json")
    hourly = read(data / "a100-calibration-sampling" / "prediction.json")["hourly_usd"]
    diagnosis = [
        read(data / name / "decision.json")
        for name in ("a100-post-restart-60", "a100-post-restart-native-60")
    ]
    costs = {
        "exact_return_and_first_sampling_holds_s": prior["measured_s"],
        "second_sampling_holds_s": order["simulation"]["new_discovery_s"],
        "target_trial_elapsed_s": verdict["elapsed_s"],
    }
    charged = sum(costs.values())
    result = {
        "schema": "ironmule.sampling-work-accounting.v1",
        "replay": verified,
        "scientific_verdict": verdict["passed"],
        "reason": verdict.get("reason", "complete prospective serving gates"),
        "quantities": {},
        "gates": {
            "mixed_and_factual_minimum_cold_speedup": 1.02,
            "warm_p95_max_ratio": 1.02,
            "resource_peak_max_ratio": 1.05,
            "independent_price_max_relative_error": 0.02,
        },
        "cost": {
            "seconds": costs,
            "attributable_target_lease_usd": charged * hourly / 3600,
            "attributable_information_only_usd": (charged - verdict["elapsed_s"])
            * hourly
            / 3600,
            "attributable_trial_only_usd": verdict["elapsed_s"] * hourly / 3600,
            "earlier_separate_attribution_s": [
                v["measured_discovery_s"] for v in diagnosis
            ],
            "earlier_separate_attribution_usd": sum(
                v["discovery_usd"] for v in diagnosis
            ),
            "hourly_usd": hourly,
            "whole_lease_snapshots": {
                f: read(folder / f)
                for f in sampling_evidence.COST_FILES
                if (folder / f).is_file()
            },
            "basis": "Wall-time attribution on the existing lease at its reserved hourly rate. Whole-lease snapshots already include that time. API-specific invoice unavailable.",
        },
        "payback_starts": None,
    }
    ctx = read(folder / "context.json")
    rows, checks = {}, {}
    for name in verdict["artifacts"]:
        entry = read(folder / name)
        if "source_label" not in entry:
            continue
        label, seed = entry["source_label"], entry["seed"]
        is_factual = name.startswith("facts-")
        try:
            restart_trial.validate(
                entry,
                seed,
                expected_source=ctx["source" if label == "reference" else "candidate"],
            )
            restart_trial.verify_binding(entry, parent["order"])
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            checks[name] = {"source_work_verified": False, "error": str(exc)}
            continue
        summary = {"source_work_verified": True, "arm": arm(entry)}
        try:
            trial.validate(
                entry, label, seed, ctx, parent, is_factual, name.startswith("screen-")
            )
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            summary.update(full_arm_valid=False, error=str(exc))
        else:
            summary["full_arm_valid"] = True
            rows[name] = entry
        checks[name] = summary
    result["individual_workers"] = checks
    for group, seeds in (("mixed", protocol.MIXED), ("facts", protocol.FACTS)):
        result["quantities"][group] = grouped(group, seeds, rows, verdict["passed"])
    result["instrumented_screens"] = [
        name for name in checks if name.startswith("screen-")
    ]
    result["workload_completeness"] = (
        "Mixed responses are checked for identical tokens, stop reasons, amounts and geometry across paired arms. Factual prompts are independently regraded. Instrumented screens and A/A are excluded from performance comparisons."
    )
    if verdict["passed"]:
        saving = result["quantities"]["mixed"]["median_paired_worker_wall_saved_s"]
        if saving > 0:
            result["payback_starts"] = math.ceil(charged / saving)
            result["payback_scope"] = (
                "Conditional on continued recurrence of the measured full mixed worker lifetime and its observed saving. Excludes other deployment/maintenance costs."
            )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("folder", "data", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    args = p.parse_args()
    write_once(args.out, account(args.folder, args.data))
