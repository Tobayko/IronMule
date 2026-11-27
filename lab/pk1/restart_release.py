"""Publish, activate or withdraw measured repeat-startup code on its bound Qwen3-32B worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from lab.pk1 import restart_order
from lab.pk1 import restart_trial as trial
from training import scoped_release as release
from training.code_author import digest
from training.code_trial import read, save
from training.trial_release import require

ADAPTER = "qwen-restart"
ACTIVATION_SEED = 93741


def evaluator():
    return digest([release.checksum(path) for path in (__file__, trial.__file__, trial.HARNESS, trial.native.__file__)])


def approved(args):
    result = trial.qualification(args)
    ref = read(args.trial / "aa-0.json")
    return {**result, "evaluator_sha256": evaluator(),
            "binding": {"runtime": result["binding"], "cache_receipt_sha256": release.checksum(args.trial / "receipt.json"),
                        "pilot_sha256": ref["pilot_sha256"], "mask_bits": ref["mask_bits"], "vendor": ref["cuda_vendor"]}}


def control(args):
    order = read(args.order)
    release.validate_order(ADAPTER, order)
    require((args.candidate_source.parent / "author/response.json").is_file(), "missing previously attributed author reply")
    restart_order.author(args.order, args.checkout, args.candidate_source)
    authorship = read(args.candidate_source.parent / "authorship.json")
    require(authorship["order"] == order["order_id"] and authorship["source_sha256"] == release.checksum(args.candidate_source),
            "authorship receipt differs from evaluated candidate")
    return release.publish(args.state, order, args.candidate_source, adapter=ADAPTER,
                           verify=lambda: approved(args), rationale=authorship["rationale"], submit=args.publish)


def measured(args, receipt, arm, seed, directory):
    expected = receipt["qualification"]
    order = read(args.order)
    require(order == receipt["order"] and receipt["adapter"] == ADAPTER and evaluator() == expected["evaluator_sha256"],
            "unrelated startup order or evaluator")
    candidate = arm == "candidate"
    cached = None
    if candidate:
        cached = args.trial / "receipt.json"
        require(release.checksum(cached) == expected["binding"]["cache_receipt_sha256"]
                and 0 <= time.time() - read(cached)["created_s"] <= 86400, "qualified startup receipt changed or expired")
    require(release.checksum(args.reference_lib) == expected["binding"]["runtime"]["w4_sha256"]
            and release.checksum(args.table) == expected["binding"]["runtime"]["table_sha256"], "native inputs changed")
    started = time.monotonic()
    tree = release.projected_tree(receipt, args.checkout, args.worker_root, args.candidate_source if candidate else None)
    row = trial.arm(args, tree, seed, directory / (arm + ".json"), receipt=cached)
    trial.validate(row, seed, expected_source=expected[arm + "_sha256"])
    trial.verify_binding(row, order)
    require(row["binding"] == expected["binding"]["runtime"] and row["reused"] is candidate
            and row["pilot_sha256"] == expected["binding"]["pilot_sha256"], "runtime binding or profile reuse differs")
    return row, time.monotonic() - started


def fresh_pair(args, receipt, seed, directory, candidate=None):
    cand, cand_worker_s = candidate or measured(args, receipt, "candidate", seed, directory)
    ref, ref_worker_s = measured(args, receipt, "reference", seed, directory)
    gate = trial.fresh_pass(ref, cand, seed)
    return {**{k: receipt["qualification"][k] for k in release.BOUND}, "passed": gate["passed"],
            "metrics": {**gate, "seed": seed, "reference_cold_s": ref["cold_s"], "reference_warm_s": ref["warm"]["wall_s"],
                        "reference_ttft_p95_s": sorted(ref["warm"]["ttfts_s"])[30], "reference_rss": ref["peak_rss_kib"],
                        "reference_device_bytes": max(ref["device_used_bytes"]), "cold_worker_s": [ref_worker_s, cand_worker_s]}}


def run(args, state, seed):
    receipt = release.pinned(state, args.receipt_sha256)
    arm = release.selected(state, args.receipt_sha256)
    with tempfile.TemporaryDirectory(dir=state) as tmp:
        tmp = Path(tmp)
        if arm == "candidate":
            try:
                row, elapsed = measured(args, receipt, arm, seed, tmp)
                bounds = read(state / "deployment/target-check.json")["metrics"]
                if (row["cold_s"] >= bounds["reference_cold_s"] or row["warm"]["wall_s"] > 1.02 * bounds["reference_warm_s"]
                        or sorted(row["warm"]["ttfts_s"])[30] > 1.02 * bounds["reference_ttft_p95_s"]
                        or row["peak_rss_kib"] > 1.05 * bounds["reference_rss"]
                        or max(row["device_used_bytes"]) > 1.05 * bounds["reference_device_bytes"]):
                    require(fresh_pair(args, receipt, seed, tmp, (row, elapsed))["passed"], "cold or warm target regression")
            except Exception as exc:
                release.rollback(state, f"startup workload failed: {exc}"[:400])
                arm = "reference"
        if arm == "reference":
            with tempfile.TemporaryDirectory(dir=tmp) as clean:
                row, elapsed = measured(args, receipt, arm, seed, Path(clean))
        return {"arm": arm, "source_sha256": row["source_sha256"], "seed": seed, "cache_used": row["reused"],
                "work_sha256": digest([row["cold"]["output_sha256"], row["warm"]["output_sha256"], row["factual"]]),
                "cold_performance_completions": row["cold"]["completed"], "warm_performance_completions": row["warm"]["completed"],
                "graded_correct": sum(row["facts"]), "pilot_decisions": row["pilot_decisions"],
                "cold_to_first_batch_s": row["cold_s"], "warm_wall_s": row["warm"]["wall_s"], "full_worker_wall_s": elapsed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    for op in ("control", "activate", "run", "exercise", "rollback"):
        cmd = sub.add_parser(op)
        cmd.add_argument("--state", type=Path, required=True)
        if op == "rollback":
            continue
        for field in ("trial", "order", "checkout", "candidate-source"):
            cmd.add_argument("--" + field, type=Path, required=True)
        if op == "control":
            cmd.add_argument("--publish", action="store_true")
        else:
            for field in ("worker-root", "gguf", "reference-lib", "table"):
                cmd.add_argument("--" + field, type=Path, required=True)
            cmd.add_argument("--receipt-sha256", required=True)
            cmd.add_argument("--worker-uid", type=int, default=60000)
            cmd.add_argument("--timeout", type=int, choices=range(1, 901), metavar="1..900", default=600)
            if op == "activate":
                cmd.add_argument("--receipt", type=Path, required=True)
            else:
                cmd.add_argument("--seed", type=int, default=93742)
    a = p.parse_args()
    if a.operation == "control":
        result = control(a)
    elif a.operation == "rollback":
        result = release.rollback(a.state)
    elif a.operation == "activate":
        def verify(receipt):
            with tempfile.TemporaryDirectory(dir=a.state) as tmp:
                return fresh_pair(a, receipt, ACTIVATION_SEED, Path(tmp))
        result = release.activate(a.state, a.receipt, a.receipt_sha256, verify)
    elif a.operation == "exercise":
        result = release.exercise(a.state, a.receipt_sha256, lambda state: run(a, state, a.seed))
    else:
        result = run(a, a.state, a.seed)
        save(a.state / "deployment" / f"work-{a.seed}.json", result)
    print(json.dumps(result, indent=2))
    if a.operation == "activate" and result["arm"] != "candidate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
