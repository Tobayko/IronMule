"""Activate, run, or withdraw a single qualified source over the pinned PR60 deployment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lab.pk1 import followup_contract as rules, followup_release, followup_trial as trial, restart_release
from training import scoped_release as release
from training.code_author import digest, write_once
from training.code_trial import read
from training.trial_release import require


def local(args, **fields):
    return SimpleNamespace(**{**vars(args), **fields})


def checked(args, receipt, label, seed, path, *, factual):
    """Use the full reviewed source tree for each worker; file changes invalidate admission."""
    expected = receipt["qualification"]
    order = read(args.order)
    require(receipt["order"] == order and receipt["adapter"] == followup_release.register_receipt(order, args.checkout, args.receipt, args.receipt_sha256)
            and expected["evaluator_sha256"] == trial.evaluator_sha()
            and expected["reference_sha256"] == release.checksum(args.checkout / release.SOURCE) == followup_release.PR60_SOURCE
            and expected["candidate_sha256"] == release.checksum(args.candidate_source),
            "source or evaluator differs from qualified program")
    require(release.selected(args.parent_state, rules.CONTROL) == "candidate", "parent release unavailable")
    parent_receipt = release.pinned(args.parent_state, rules.CONTROL)
    parent_binding = parent_receipt["qualification"]["binding"]
    require(expected["binding"]["profile"] == parent_binding and release.checksum(args.control) == rules.CONTROL
            and release.checksum(args.weights) == rules.HEAD
            and release.checksum(args.reference_lib) == parent_binding["runtime"]["w4_sha256"]
            and release.checksum(args.table) == parent_binding["runtime"]["table_sha256"]
            and release.checksum(args.cache) == parent_binding["cache_receipt_sha256"], "different target components")
    if order["schema"] == followup_release.SAMPLING:
        require(expected["binding"]["sampling_profile_sha256"] == release.checksum(args.trial / "sampling-profile.json")
                and expected["binding"]["sampling_parent_receipt_sha256"] == release.checksum(args.trial / "sampling-parent-receipt.json"),
                "different measured sampling profile or parent receipt")
    baseline = release.projected_tree(receipt, args.checkout, args.worker_root)
    candidate = release.projected_tree(receipt, args.checkout, args.worker_root, args.candidate_source)
    inputs = local(args, checkout=baseline, candidate_source=candidate / release.SOURCE)
    if label == "candidate":
        # live() caches a single-file projection. Validate all its importable contents before use.
        staged = trial.prior._tree(baseline, inputs.candidate_source, args.worker_root)
        release.check_tree(receipt, staged, True)
    row = trial.live(inputs, label, seed, path, factual_first=factual)
    trial.validate(row, label, seed, trial.context(inputs), parent_receipt, factual)
    return row


def charged(args, receipt, label, seed, path, *, factual):
    """Price source projection, hashing, selection and result validation without editing the raw worker evidence."""
    start = time.monotonic()
    row = checked(args, receipt, label, seed, path, factual=factual)
    elapsed = time.monotonic() - start
    outside = max(0.0, elapsed - row["broker_elapsed_s"])
    write_once(path.with_name(path.stem + "-dispatch.json"),
               {"elapsed_s": elapsed, "outside_arm_s": outside, "arm_sha256": release.checksum(path)})
    return {**row, "broker_to_process_s": row["broker_to_process_s"] + outside,
            "broker_elapsed_s": row["broker_elapsed_s"] + outside, "release_dispatch_and_validation_s": outside}


def serving_metrics(row):
    return {"cold_s": row["broker_to_process_s"] + row["cold_s"], "warm_s": row["warm"]["wall_s"],
            "warm_p95_s": sorted(row["warm"]["ttfts_s"])[30], "facts_s": row["factual_timing"]["wall_s"],
            "facts_p95_s": trial.prior.fact_timing(row["factual_timing"]), "rss": row["peak_rss_kib"],
            "device": max(row["device_used_bytes"])}


def fresh(args, receipt, directory):
    scores, reference, evidence = [], None, {}
    for i, (seed, factual) in enumerate(((97051, False), (97052, True))):
        rows = {}
        for label in (("reference", "candidate"), ("candidate", "reference"))[i]:
            file = directory / f"fresh-{label}-{seed}.json"
            rows[label] = charged(args, receipt, label, seed, file, factual=factual)
            evidence[file.name] = release.checksum(file)
            outer = file.with_name(file.stem + "-dispatch.json")
            evidence[outer.name] = release.checksum(outer)
        if not factual:
            reference = serving_metrics(rows["reference"])
        scores.append(trial.scores(rows["reference"], rows["candidate"]))
    qualification = receipt["qualification"]
    return {**{k: qualification[k] for k in release.BOUND},
            "passed": trial.gate([scores[0]], [scores[1]], qualification["metrics"]["aa_spread"]),
            "metrics": {"mixed": scores[0], "factual": scores[1], "reference": reference, "raw_sha256": evidence}}


def activate(args):
    followup_release.register_receipt(read(args.order), args.checkout, args.receipt, args.receipt_sha256)
    def verify(receipt):
        with TemporaryDirectory(dir=args.state) as temp:
            result = fresh(args, receipt, Path(temp))
            evidence = args.state / "deployment" / ("fresh-" + args.receipt_sha256)
            evidence.mkdir(exist_ok=True)
            for path in Path(temp).glob("*.json"):
                dest = evidence / path.name
                if dest.exists():
                    require(release.checksum(dest) == release.checksum(path), "conflicting fresh activation measurements")
                else:
                    os.link(path, dest)
            release.sync_dir(evidence)
            return result
    return release.activate(args.state, args.receipt, args.receipt_sha256, verify)


def preceding(args, seed, expected):
    """Run the actual preceding release, including its own fallback when its cached profile was withdrawn."""
    prior = local(args, state=args.parent_state, order=args.parent_order, trial=args.parent_trial, checkout=args.parent_checkout,
                  candidate_source=args.parent_candidate_source, receipt_sha256=rules.CONTROL)
    served = restart_release.run(prior, args.parent_state, seed)
    return {**served, "preceding_arm": served["arm"],
            "arm": "reference" if served["source_sha256"] == expected else "preceding-deployment"}


def run(args, state, seed):
    start = time.monotonic()
    order = read(args.order)
    try:
        followup_release.register_receipt(order, args.checkout, args.receipt, args.receipt_sha256)
        receipt = release.pinned(state, args.receipt_sha256)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        release.rollback(state, f"invalid follow-up release: {str(exc)[:320]}")
        return preceding(args, seed, followup_release.PR60_SOURCE)
    baseline = receipt["qualification"]["reference_sha256"]
    if release.selected(state, args.receipt_sha256) != "candidate":
        return preceding(args, seed, baseline)
    try:
        with TemporaryDirectory(dir=state) as scratch:
            row = charged(args, receipt, "candidate", seed, Path(scratch) / "work.json", factual=False)
            limits = read(state / "deployment/target-check.json")["metrics"]["reference"]
            metrics = serving_metrics(row)
            for name, value in metrics.items():
                margin = 1.05 if name in ("rss", "device") else (1.0 if name == "cold_s" else 1.02)
                require(value <= margin * limits[name], f"fresh {name} regressed against the qualified reference")
            require(release.selected(state, args.receipt_sha256) == "candidate", "withdrawn during serving")
            return {"arm": "candidate", "source_sha256": row["source_sha256"], "seed": seed,
                    "work_sha256": digest([row["cold"]["output_sha256"], row["warm"]["output_sha256"], row["factual"]]),
                    "graded_correct": sum(row["facts"]), "cold_performance_completions": row["cold"]["completed"],
                    "warm_performance_completions": row["warm"]["completed"], "selection": row["startup_selection"],
                    "full_worker_wall_s": row["broker_elapsed_s"], "release_dispatch_and_validation_s": row["release_dispatch_and_validation_s"],
                    "elapsed_through_validated_work_s": time.monotonic() - start, **metrics}
    except (OSError, ValueError, TimeoutError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        release.rollback(state, f"follow-up target changed or regressed: {str(exc)[:320]}")
        return preceding(args, seed, baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", choices=("activate", "run", "exercise", "rollback"))
    for name in ("state", "order", "trial", "checkout", "candidate-source", "receipt", "parent-state", "parent-trial", "parent-order",
                 "parent-checkout", "parent-candidate-source", "cache", "control", "weights", "gguf", "reference-lib", "table", "worker-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--worker-uid", type=int, default=60000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=97054)
    args = parser.parse_args()
    if args.op == "rollback":
        result = release.rollback(args.state)
    elif args.op == "activate":
        result = activate(args)
    elif args.op == "exercise":
        followup_release.register_receipt(read(args.order), args.checkout, args.receipt, args.receipt_sha256)
        result = release.exercise(args.state, args.receipt_sha256, lambda state: run(args, state, 97053))
    else:
        result = run(args, args.state, args.seed)
    print(json.dumps(result))
    if args.op == "activate" and result["arm"] != "candidate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
