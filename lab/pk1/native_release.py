"""Promote a measured Qwen3 width-32 CUDA variant through its qualification PR and scoped target runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from lab.pk1 import native_trial as trial
from lab.pk1.native_order import SOURCE
from training.code_author import write_once
from training.code_trial import PAIRS, git, prepare, read, save, submit_candidate
from training.trial_release import deployment, merged_source, require

QWEN32 = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"


def blob(tree, commit, path):
    return hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=tree)).hexdigest()


def score_screen(screen, order, binary, table):
    from ironmule.w4_kernels import ERROR_BOUND
    trial.validate_screen(screen, order["binding"], binary, table)
    for row in screen["measurements"]:
        require(len(row["arms"]) == 2, "screen needs both numerical results")
        for arm in row["arms"]:
            require(0 <= arm["relative_error"] <= ERROR_BOUND and math.isfinite(arm["microseconds"])
                    and arm["microseconds"] > 0, "failed numerical screen")


def check_row(row, order, binary, seed):
    binding = order["binding"]
    trial.validate_arm(row, binding, {"binary": binary, "table": binding["table_sha256"]}, seed)
    trial.verify_outputs(row, seed)
    require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] > 0
                for k in ("wall_s", "ttft_s", "peak_rss_kib", "tokens")), "invalid timing or completed work")
    require(row.get("profiled") is False and row["binding"]["flags"] == trial.FLAGS
            and row["binding"]["engine"] == binding["engine"], "wrong or profiled execution settings")
    require(len(row["device_used_bytes_after_phases"]) == 4
            and all(type(n) is int and n > 0 for n in row["device_used_bytes_after_phases"]),
            "incomplete GPU memory observations")


def qualification(args):
    directory = args.trial
    order = read(args.order)
    oid = hashlib.sha256(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode())
    binding = order["binding"]
    require(order["schema"] == "ironmule.native-code-experiment.v1" and order["order_id"] == oid.hexdigest()
            and order["evaluator"] == trial.SCHEMA and order["allowed_paths"] == [SOURCE]
            and order["target"]["path"] == SOURCE and order["workload"]["decode_rows"] == 32
            and binding["gguf"] == QWEN32 and len(binding["fp"]["devices"]) == 1
            and re.fullmatch("[a-f0-9]{40}", order["base_commit"]), "unbound native order")
    profile = args.order.with_suffix(".profile.json")
    require(trial.checksum(profile) == order["profile_sha256"] and read(profile)["binding"] == binding
            and read(profile)["base_commit"] == order["base_commit"], "profile or source changed")
    candidate = trial.checksum(directory / "candidate.so")
    source = trial.checksum(args.candidate_source)
    context = {"order": order["order_id"], "base_commit": order["base_commit"],
               "order_sha256": trial.checksum(args.order), "profile_sha256": order["profile_sha256"],
               "candidate_source": source, "harness": trial.checksum(trial.__file__),
               "reference": binding["w4_sha256"], "table": binding["table_sha256"]}
    require(source != binding["source_sha256"] and candidate != binding["w4_sha256"]
            and trial.checksum(args.table) == binding["table_sha256"]
            and context == read(directory / "context.json"), "mixed or unchanged candidate")
    build = read(directory / "build.json")
    require(build == {"source": source, "binary": candidate}, "binary does not match build")
    result = read(directory / "result.json")
    require(result["passed"] is True and result["schema"] == trial.SCHEMA and result["context"] == context
            and result["candidate"] == candidate and result["incumbent"] == binding["w4_sha256"],
            "missing positive bound decision")
    names = ["build.json", "screen.json", "calibration.json", *(f"{arm}-{i}.json" for i in range(PAIRS) for arm in ("ref", "cand"))]
    require(result["measurements_sha256"] == {name: trial.checksum(directory / name) for name in names},
            "measurement receipt mismatch")
    score_screen(read(directory / "screen.json"), order, candidate, args.table)
    frozen = trial.validate_calibration(read(directory / "calibration.json"), binding)
    pairs = []
    for i in range(PAIRS):
        ref, cand = (read(directory / f"{arm}-{i}.json") for arm in ("ref", "cand"))
        for row, binary in ((ref, binding["w4_sha256"]), (cand, candidate)):
            check_row(row, order, binary, 92741 + i)
        pairs.append((ref, cand))
    bound = pairs[0][0]["binding"]
    require(bound["profile"] == frozen, "measured profile differs from frozen calibration")
    require(all(row["binding"] == bound for pair in pairs for row in pair), "target changed between observations")
    decision = trial.judge_native(pairs)
    require(decision["passed"] and all(result[k] == v for k, v in decision.items()), "independent paired gate failed")
    return order, {"order": order, "build": build, "binding": bound, "qualification": decision,
                   "trial_result_sha256": trial.checksum(directory / "result.json"), "harness": context["harness"],
                   "wall_limit": max(r["wall_s"] for r, _ in pairs),
                   "ttft_limit": max(r["ttft_s"] for r, _ in pairs),
                   "rss_limit": 1.05 * max(r["peak_rss_kib"] for r, _ in pairs),
                   "device_limit": 1.05 * max(max(r["device_used_bytes_after_phases"]) for r, _ in pairs)}


def control(args):
    order, receipt = qualification(args)
    with deployment(args.state) as d:
        tree = prepare(d, order["base_commit"], "proposal")
        base = order["base_commit"]
        require(blob(tree, base, SOURCE) == order["binding"]["source_sha256"], "different original source")
        original = prepare(d, base, "reference")
        require(not git("status", "--porcelain", cwd=original), "reference checkout has changed")
        # Reapply the source-bound worker response to the pristine base, checking every candidate byte.
        require((args.candidate_source.parent / "author/response.json").is_file(), "missing hosted worker reply")
        trial.author(SimpleNamespace(order=args.order, checkout=original, output=args.candidate_source))
        (tree / SOURCE).write_bytes(args.candidate_source.read_bytes())
        git("add", "--", SOURCE, cwd=tree)
        runtime_paths = ("ironmule/ggml_engine.py", "ironmule/gguf_tokenizer.py", "ironmule/pilot.py")
        runtime_sources = receipt["binding"]["runtime_sources"]
        require(all(runtime_sources.get(name) == blob(original, base, name) for name in runtime_paths),
                "serving source in the target arm differs from the base revision")
        # All measured modules, including a separately identified tuning helper, must be present on the target.
        receipt["runtime_sources"] = {name: runtime_sources[name] for name in runtime_paths
                                      + ("ironmule/w4_kernels.py", "ironmule/kernel_gen.py")}
        require(git("diff", "--name-only", base, cwd=tree) == SOURCE
                and blob(tree, "", SOURCE) == receipt["build"]["source"], "unexpected source tree")
        measured_tree = git("write-tree", cwd=tree)
        if not args.publish:
            return {"status": "qualified, awaiting publication", "tree": measured_tree}
        authorship = read(args.candidate_source.parent / "authorship.json")
        require(authorship["order"] == order["order_id"] and authorship["candidate_sha256"] == receipt["build"]["source"],
                "author receipt mismatch")
        body = ("## Problem\n\n" + order["hypothesis"] + "\n\n## Change\n\n" + authorship["rationale"]
                + "\n\n## Validation\n\n" + order["binding"]["fp"]["gpu"] + ", Qwen3-32B Q4_K_M, 32 clients. Verified the numerical screen, 18 factual "
                "answers per arm and six counterbalanced serving pairs against the on-device tuned incumbent.\n\n"
                + json.dumps(receipt["qualification"], indent=2) + "\n\n## Rollback\n\n"
                "Use lab.pk1.native_release rollback on the bound target to select the prior W4 library.")
        pr = submit_candidate(d, order, tree, body)
        merged = merged_source(d, pr, order, allowed_targets={SOURCE})
        require(merged["candidate_tree"] == measured_tree
                and blob(d / "repo.git", merged["candidate"], SOURCE) == receipt["build"]["source"],
                "merged tree differs from the actual candidate bytes")
        write_once(d / "control.json", {**receipt, "publication": merged})
        return {"pr": pr, "receipt_sha256": trial.checksum(d / "control.json")}


def target_files(args, receipt, candidate):
    from ironmule.ggml_binary import identity
    from ironmule.ggml_engine import file_sha256
    from ironmule.kernel_gen import fingerprint

    binding = receipt["order"]["binding"]
    actual = fingerprint()
    require(all(actual[k] == binding["fp"][k] for k in ("devices", "cc", "driver", "generator", "kg"))
            and identity() == binding["native"], "target device or GGML runtime differs from qualification")
    require(all(trial.checksum(args.checkout / name) == value for name, value in receipt["runtime_sources"].items()),
            "target runtime source differs from measured baseline")
    require(trial.checksum(trial.__file__) == receipt["harness"]
            and trial.checksum(args.checkout / SOURCE) == binding["source_sha256"]
            and file_sha256(str(args.gguf)) == binding["gguf"] == QWEN32
            and trial.checksum(args.table) == binding["table_sha256"], "target artifacts changed")
    if candidate:
        require(trial.checksum(args.candidate_source) == receipt["build"]["source"], "candidate source changed")
    binary = args.trial / "candidate.so" if candidate else args.reference_lib
    expected = receipt["build"]["binary"] if candidate else binding["w4_sha256"]
    require(trial.checksum(binary) == expected, "selected binary changed")
    return binary, expected


def measured(args, receipt, candidate, seed, dest):
    binary, expected = target_files(args, receipt, candidate)
    row = trial.launch_arm(args, binary, seed, dest, receipt["binding"]["profile"])
    check_row(row, receipt["order"], expected, seed)
    require(row["binding"] == receipt["binding"], "execution target differs from qualification")
    return row


def target(args):
    """Trust the digest transferred by the control host. Worker output cannot write deployment state."""
    with deployment(args.state) as d:
        pinned = d / "target.json"
        if args.command == "activate":
            content = args.receipt.read_bytes()
            require(hashlib.sha256(content).hexdigest() == args.receipt_sha256, "untrusted control receipt")
            receipt = json.loads(content)
            pub = receipt["publication"]
            require(receipt["qualification"]["passed"] is True and pub["pr"] > 0
                    and pub["order_id"] == receipt["order"]["order_id"]
                    and pub["base"] == receipt["order"]["base_commit"], "incomplete publication")
            write_once(pinned, {"sha256": args.receipt_sha256, "path": str(args.receipt.resolve())})
        pin = read(pinned)
        if args.command == "rollback":
            save(d / "selection.json", {"arm": "ref", "withdrawn": True, "reason": "explicit rollback"})
            return read(d / "selection.json")
        content = Path(pin["path"]).read_bytes()
        require(hashlib.sha256(content).hexdigest() == pin["sha256"], "control receipt changed")
        receipt = json.loads(content)
        selection = d / "selection.json"
        current = read(selection) if selection.exists() else {"arm": "ref", "withdrawn": False}
        require(current["arm"] in ("ref", "cand") and type(current["withdrawn"]) is bool
                and (not current["withdrawn"] or current["arm"] == "ref"), "invalid target selection")
        if args.command == "activate" and current["withdrawn"]:
            return current
        # Each load uses the isolated engine benchmark with the complete 32-client profile.
        # The incumbent returns the requested result if candidate verification fails.
        with tempfile.TemporaryDirectory(dir=d) as tmp:
            tmp = Path(tmp)
            seed = args.seed
            if args.command == "run" and current["arm"] == "ref":
                return measured(args, receipt, False, seed, tmp / "ref.json")
            ref = None
            try:
                cand = measured(args, receipt, True, seed, tmp / "cand.json")
                if (args.command == "run" and cand["wall_s"] <= receipt["wall_limit"]
                        and cand["ttft_s"] <= receipt["ttft_limit"] and cand["peak_rss_kib"] <= receipt["rss_limit"]
                        and max(cand["device_used_bytes_after_phases"]) <= receipt["device_limit"]):
                    return cand
                # Fresh matched reference on activation or when ordinary measurements miss the envelope.
                ref = measured(args, receipt, False, seed, tmp / "ref.json")
                require(all(ref[k] == cand[k] for k in ("tokens", "output_sha256", "fact_digest", "completed"))
                        and cand["wall_s"] < ref["wall_s"] and cand["ttft_s"] <= ref["ttft_s"]
                        and cand["peak_rss_kib"] <= 1.05 * ref["peak_rss_kib"]
                        and max(cand["device_used_bytes_after_phases"]) <= 1.05 * max(ref["device_used_bytes_after_phases"]),
                        "target regression or changed output")
            except Exception as exc:
                current = {"arm": "ref", "withdrawn": True, "reason": str(exc)[:400]}
                save(selection, current)
                if args.command == "run":
                    return ref if ref is not None else measured(args, receipt, False, seed, tmp / "ref.json")
                return current
            current = {"arm": "cand", "withdrawn": False, "pr": receipt["publication"]["pr"]}
            save(selection, current)
            return cand if args.command == "run" else current


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    control_cmd = s.add_parser("control")
    control_cmd.add_argument("--order", type=Path, required=True)
    control_cmd.add_argument("--publish", action="store_true")
    cmds = [control_cmd]
    for op in ("activate", "run", "rollback"):
        cmd = s.add_parser(op)
        if op == "rollback":
            cmd.add_argument("--state", type=Path, required=True)
            continue
        cmds.append(cmd)
        cmd.add_argument("--checkout", type=Path, required=True)
        cmd.add_argument("--gguf", type=Path, required=True)
        cmd.add_argument("--reference-lib", type=Path, required=True)
        cmd.add_argument("--timeout", type=int, default=900)
        cmd.add_argument("--worker-root", type=Path)
        cmd.add_argument("--worker-uid", type=int, default=60000)
        cmd.add_argument("--seed", type=int, default=93741)
        if op == "activate":
            cmd.add_argument("--receipt", type=Path, required=True)
            cmd.add_argument("--receipt-sha256", required=True)
    for cmd in cmds:
        for flag in ("state", "trial", "candidate-source", "table"):
            cmd.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    args.state.mkdir(parents=True, mode=0o700, exist_ok=True)
    out = control(args) if args.command == "control" else target(args)
    print(json.dumps(out))
    if args.command == "activate" and out["withdrawn"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
