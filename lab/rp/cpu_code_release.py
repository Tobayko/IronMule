"""Publish or operate a merged EPYC prefill candidate on its measured four-client copy workload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from lab.rp import cpu_code_trial as trial
from training import scoped_release as release
from training.code_author import read_request, read_response
from training.code_trial import develop, prepare, read, save
from training.trial_release import require

ADAPTER = "epyc-prefill"
ACTIVATION_SEED = 9351


def control(args):
    order = read(args.order)
    release.validate_order(ADAPTER, order)
    request_id, request = read_request(args.authorship / "author")
    require(request["order"] == order, "unrelated author request")
    proposal = read_response(args.authorship / "author", request_id)
    args.state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.state) as tmp:
        source = prepare(Path(tmp), order["base_commit"], "proposal")
        # The response above is already present. No new author is called.
        relative = develop(order, source, args.authorship)
        require(relative == release.SOURCE and release.checksum(source / relative) == release.checksum(args.candidate / relative),
                "evaluated source differs from the attributed response")
        def verify():
            options = argparse.Namespace(**{**vars(args), "state": args.trial, "seed": 8351})
            observed = trial.verify_offline(options)
            require(observed == read(args.trial / "verdict.json") and observed["metrics"]["fixed_four_threads_vs_candidate"]["passed"] is False,
                    "missing independently verified six-pair or static-rule verdict")
            return revision(observed)
        return release.publish(args.state, order, source / relative, adapter=ADAPTER, verify=verify,
                               rationale=proposal["rationale"], submit=args.publish)


def revision(observation):
    from training.code_author import digest
    return {**observation, "evaluator_sha256": digest([observation["evaluator_sha256"], release.checksum(__file__)])}


def measured(args, receipt, arm, seed, directory):
    candidate = arm == "candidate"
    expect = receipt["qualification"]
    started = time.monotonic()
    require(receipt["adapter"] == ADAPTER and receipt["order"] == read(args.order), "foreign CPU target order")
    base = release.projected_tree(receipt, args.reference, args.worker_root)
    tree = release.projected_tree(receipt, args.reference, args.worker_root, args.candidate / release.SOURCE) if candidate else base
    worker_args = argparse.Namespace(**{**vars(args), "reference": base})
    _, evaluator, helper = trial.inputs(worker_args)
    key = revision({"evaluator_sha256": trial.scoring_hash(evaluator, helper)})["evaluator_sha256"]
    require(key == expect["evaluator_sha256"], "CPU evaluator differs from publication")
    # Both arms use the evaluator and helper in the independent reference tree.
    require(release.checksum(evaluator) == receipt["runtime_manifest"]["training/serving_trial.py"]
            and release.checksum(helper) == receipt["runtime_manifest"]["training/pilot_threads.py"],
            "unbound serving evaluator or scratch qualification helper")
    dest = directory / (arm + ".json")
    trial.worker(worker_args, tree, "arm", dest, seed)
    row = read(dest)
    trial.check_arm(worker_args, tree, row, seed)
    workload = {k: v for k, v in row["workload"].items() if k != "seed"}
    require({**row["binding"], "workload": workload} == expect["binding"]
            and row["runtime_source_sha256"]["ggml_engine.py"] == expect[arm + "_sha256"], "unqualified runtime or work")
    return row, time.monotonic() - started


def fresh_pair(args, receipt, seed, directory, candidate=None):
    cand, cand_worker_s = candidate or measured(args, receipt, "candidate", seed, directory)
    ref, ref_worker_s = measured(args, receipt, "reference", seed, directory)
    require(all((r["requests"], r["reused_prompt_tokens"]) == (c["requests"], c["reused_prompt_tokens"])
                for r, c in zip(ref["waves"], cand["waves"], strict=True))
            and all(ref[k] == cand[k] for k in ("tokens", "completed", "correct", "output_sha256")),
            "target delivered different work")
    return {"passed": cand["wall_s"] < ref["wall_s"] and cand["ttft_s"] <= ref["ttft_s"]
            and cand["peak_rss_kib"] <= 1.05 * ref["peak_rss_kib"],
            **{k: receipt["qualification"][k] for k in release.BOUND},
            "metrics": {"seed": seed, "reference_wall_s": ref["wall_s"], "candidate_wall_s": cand["wall_s"],
                        "reference_ttft_s": ref["ttft_s"], "candidate_ttft_s": cand["ttft_s"],
                        "cold_worker_s": [ref_worker_s, cand_worker_s], "correct": ref["correct"],
                        "output_sha256": ref["output_sha256"]}}


def run(args, state, seed):
    receipt = release.pinned(state, args.receipt_sha256)
    arm = release.selected(state, args.receipt_sha256)
    with tempfile.TemporaryDirectory(dir=state) as tmp:
        tmp = Path(tmp)
        if arm == "candidate":
            try:
                row, elapsed = measured(args, receipt, arm, seed, tmp)
                bounds = read(state / "deployment/target-check.json")["metrics"]
                if row["wall_s"] > bounds["reference_wall_s"] or row["ttft_s"] > bounds["reference_ttft_s"]:
                    require(fresh_pair(args, receipt, seed, tmp, (row, elapsed))["passed"], "matched target timing regression")
            except Exception as exc:
                release.rollback(state, f"target workload failed: {exc}"[:400])
                arm = "reference"
        if arm == "reference":
            # A matched reference may have completed before the candidate regression.
            with tempfile.TemporaryDirectory(dir=tmp) as other:
                row, elapsed = measured(args, receipt, arm, seed, Path(other))
        return {"arm": arm, "source_sha256": row["runtime_source_sha256"]["ggml_engine.py"], "work_sha256": row["output_sha256"],
                "seed": seed, "correct": row["correct"], "completed": row["completed"], "tokens": row["tokens"],
                "scratch_choices": row["scratch_measured_decisions"], "serving_wall_s": row["wall_s"],
                "ttft_s": row["ttft_s"], "full_worker_wall_s": elapsed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    for op in ("control", "activate", "run", "exercise", "rollback"):
        cmd = sub.add_parser(op)
        cmd.add_argument("--state", type=Path, required=True)
        if op == "rollback":
            continue
        for field in ("reference", "candidate", "order"):
            cmd.add_argument("--" + field, type=Path, required=True)
        if op == "control":
            cmd.add_argument("--trial", type=Path, required=True)
            cmd.add_argument("--authorship", type=Path, required=True)
            cmd.add_argument("--publish", action="store_true")
        else:
            for field in ("worker-root", "gguf", "head", "python"):
                cmd.add_argument("--" + field, type=Path, required=True)
            cmd.add_argument("--receipt-sha256", required=True)
            cmd.add_argument("--timeout", type=int, choices=range(1, 601), metavar="1..600", default=300)
            if op == "activate":
                cmd.add_argument("--receipt", type=Path, required=True)
            else:
                cmd.add_argument("--seed", type=int, default=9352)
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
