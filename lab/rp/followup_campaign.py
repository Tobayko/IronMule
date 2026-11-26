"""Run one prospective source experiment against the deployed PR60 worker on its reserved device."""
from __future__ import annotations

import argparse
import ast
import json
import os
from tempfile import NamedTemporaryFile
from pathlib import Path, PurePosixPath
import shlex
import textwrap
import time
from types import SimpleNamespace

from lab.pk1 import followup_contract as rules, followup_release, followup_trial as trial
from lab.pk1.native_trial import checksum
from lab.rp.native_campaign import CONFIG_KEYS, command, job, lease, shell, snapshot, upload
from training import calibration_sampling, experience_archive
from training.code_author import PendingAuthor, consume_cli, hosted_proposal, write_once
from training.code_trial import read
from training.trial_release import require


def author(order, checkout, state, mode, archive):
    directory = state / "authoring"
    directory.mkdir(mode=0o700, exist_ok=True)
    source = checkout / trial.PATH
    require(checksum(source) == order["source_sha256"], "incumbent source differs from the order")
    lines = source.read_text().splitlines(keepends=True)
    klass, = [n for n in ast.parse("".join(lines)).body if isinstance(n, ast.ClassDef) and n.name == order["target"]["class"]]
    methods = {n.name: n for n in klass.body if isinstance(n, ast.FunctionDef)}
    method = methods[order["target"]["function"]]
    require(method.lineno == order["target"]["line"] and not method.decorator_list, "different measured site")

    def fragment(node):
        return textwrap.dedent("".join(lines[node.lineno - 1:node.end_lineno]))

    inbox = directory / "author"
    context = None
    if not (inbox / "request.json").exists():
        context = {"constructor": fragment(methods["__init__"]), "source_sha256": checksum(source),
                   "prior_work": list(experience_archive.search(archive, method.name, limit=6))}
    try:
        reply = hosted_proposal(directory, order, fragment(method), knowledge=context)
    except PendingAuthor:
        if mode == "hosted":
            raise
        reply = consume_cli(directory)
    replacement = textwrap.indent(textwrap.dedent(reply["definition"]).strip(), " " * method.col_offset) + "\n"
    program = "".join(lines[:method.lineno - 1]) + replacement + "".join(lines[method.end_lineno:])
    dest = directory / "candidate.py"
    with NamedTemporaryFile("w", dir=directory) as stream:
        stream.write(program)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(stream.name, dest)
        except FileExistsError:
            require(dest.read_text() == program, "different candidate already exists for this order")
    followup_release.authorship(SimpleNamespace(checkout=checkout, candidate_source=dest), order)
    return dest


def trial_args(cfg, args, remote, original):
    return ["--order", remote / "order.json", "--checkout", args.remote_checkout, "--candidate-source", remote / "candidate.py",
            "--trial", remote / "trial", "--control", original / "merged.json", "--weights", original / "startup-head.npz",
            "--parent-state", original / "active", "--cache", original / "trial/receipt.json", "--worker-root", cfg["worker_root"],
            "--reference-lib", cfg["reference_lib"], "--table", cfg["table"], "--gguf", cfg["gguf"]]


def run_stage(pod, cfg, key, op, arguments, remote, local):
    name = op + ".json"
    dest = local / name
    snapshot(pod, remote, [name], local)
    if not dest.is_file():
        out = shlex.quote(str(remote / name))
        proc = command(cfg, "lab.pk1.followup_target", [op, *arguments])
        # A refused activation emits its own JSON and returns nonzero. Keep that terminal result.
        wrapped = (f"if {proc} > {out}.tmp; then mv {out}.tmp {out}; "
                   f"elif test -s {out}.tmp; then mv {out}.tmp {out}; else exit 1; fi")
        status = job(pod, "fup-" + key[:16] + "-" + op, wrapped)
        snapshot(pod, remote, [name], local)
        require(status != "finished" or dest.exists(), "target operation finished without a receipt")
    return read(dest) if dest.is_file() else None


FRAMEWORK = ("lab/pk1/followup_trial.py", "lab/pk1/followup_arm.py", "lab/pk1/followup_contract.py",
             "lab/pk1/followup_release.py", "lab/pk1/followup_target.py", "lab/pk1/sampling_eval.py",
             "lab/pk1/sampling_arm.py", "lab/pk1/restart_trial.py", "lab/pk1/restart_arm.py",
             "lab/pk1/restart_release.py", "lab/pk1/native_trial.py", "training/scoped_release.py",
             "training/calibration_sampling_contract.json", "training/pilot_restart_profiles.py",
             "lab/rp/restricted_worker.py")


def qualify_source(args, decision, source, cfg, pod, remote, original):
    state = args.state
    root = Path(__file__).resolve().parents[2]
    identity = {"order": decision["order_id"], "source": checksum(source), "config": checksum(args.config),
                "remote_checkout": str(args.remote_checkout), "framework": {n: checksum(root / n) for n in FRAMEWORK}}
    write_once(state / "campaign.json", identity)
    if not (state / "staged.json").exists():
        paths = [PurePosixPath(cfg["framework"]) / n for n in FRAMEWORK]
        hashes = shell(pod, shlex.join(["sha256sum", "--", *map(str, paths)])).splitlines()
        require(len(hashes) == len(paths) and all(row.split(maxsplit=1) == [identity["framework"][name], str(path)]
                for row, name, path in zip(hashes, FRAMEWORK, paths, strict=True)), "different target evaluator or adapter")
        check = ("from pathlib import Path;import json,sys;from training.scoped_release import check_tree;"
                 "check_tree(json.loads(Path(sys.argv[1]).read_bytes()),Path(sys.argv[2]),True)")
        shell(pod, "cd " + shlex.quote(cfg["framework"]) + " && " + shlex.join(
            [cfg["python"], "-c", check, str(original / "merged.json"), str(args.remote_checkout)]))
        shell(pod, shlex.join(["mkdir", "-p", str(remote)]))
        upload(pod, remote / "order.json", (state / "order.json").read_bytes())
        upload(pod, remote / "candidate.py", source.read_bytes())
        write_once(state / "staged.json", identity)
    destination = state / "trial"
    files = [*trial.file_names({"sampling_plan": decision["selected_plan"]}), "result.json"]
    snapshot(pod, remote / "trial", files, destination)
    if not (destination / "result.json").is_file():
        run = command(cfg, "lab.pk1.followup_trial", trial_args(cfg, args, remote, original))
        status = job(pod, "sampling-" + decision["order_id"][:16], run)
        snapshot(pod, remote / "trial", files, destination)
        require(status != "finished" or (destination / "result.json").exists(), "target trial finished without a verdict")
    path = destination / "result.json"
    return read(path) if path.is_file() else None


def cycle(args, decision, source):
    cfg = read(args.config)
    require(set(cfg) == CONFIG_KEYS and all(PurePosixPath(cfg[k]).is_absolute() for k in CONFIG_KEYS - {"pod_id"}),
            "invalid target config")
    key = decision["order_id"]
    pod = lease(cfg["pod_id"])
    root = PurePosixPath(cfg["state_root"])
    remote = root / ("sampling-" + key)
    original = root / ("restart-" + rules.PARENT)
    state = args.state
    physical = qualify_source(args, decision, source, cfg, pod, remote, original)
    if physical is None:
        return {"stage": "measuring", "order": key, "saved_receipts": len(list((state / "trial").glob("*.json")))}
    if physical.get("passed") is not True:
        return {"stage": "rejected", "order": key, "result": str(state / "trial/result.json"),
                "reason": physical.get("reason", "paired qualification thresholds")}
    local = state / "release"
    local.mkdir(mode=0o700, exist_ok=True)
    check = SimpleNamespace(order=state / "order.json", checkout=args.checkout, candidate_source=source,
                            trial=state / "trial", state=local, control=args.control, weights=args.weights,
                            sampling_evidence=args.prior_dir.parent, publish=args.publish)
    verified = followup_release.approved(check)
    if not args.publish:
        return {"stage": "qualified", "order": key, "metrics": verified["metrics"]}
    issued = local / "published.json"
    if not issued.exists():
        write_once(issued, followup_release.control(check))
    publication = read(issued)
    pin = publication["receipt_sha256"]
    receipt_path = Path(publication["receipt"])
    require(checksum(receipt_path) == pin and read(receipt_path)["qualification"] == verified, "different published source")
    upload(pod, remote / "merged.json", receipt_path.read_bytes())
    common = [*trial_args(cfg, args, remote, original), "--state", remote / "active", "--receipt", remote / "merged.json",
              "--receipt-sha256", pin, "--parent-trial", original / "trial", "--parent-order", original / "order.json",
              "--parent-checkout", cfg["checkout"], "--parent-candidate-source", original / "proposal.py"]
    for op in ("activate", "run", "exercise"):
        row = run_stage(pod, cfg, key, op, common, remote, local)
        if op == "activate":
            deployment = remote / "active/deployment"
            files = [f"fresh-{label}-{seed}{suffix}.json" for seed in (97051, 97052)
                     for label in ("reference", "candidate") for suffix in ("", "-dispatch")]
            snapshot(pod, deployment / ("fresh-" + pin), files, local / "fresh")
            snapshot(pod, deployment, ["target-check.json"], local)
        if row is None:
            return {"stage": op, "order": key, "pr": publication["pr"]}
        if op != "exercise" and row.get("arm") != "candidate":
            return {"stage": "withdrawn", "order": key, "reason": row.get("reason"), "result": str(local / (op + ".json"))}
        if op == "exercise":
            require(row["passed"] is True and row["receipt_sha256"] == pin, "physical rollback differs from release")
    return {"stage": "active", "order": key, "pr": publication["pr"], "receipt_sha256": pin,
            "work": read(local / "run.json"), "rollback": read(local / "exercise.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkout", "state", "prior-dir", "forecast", "control", "weights", "archive"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--remote-checkout", type=PurePosixPath, required=True)
    parser.add_argument("--observations", type=Path, nargs=2, required=True)
    parser.add_argument("--author", choices=("hosted", "cli"), default="hosted")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if not (args.remote_checkout.is_absolute() and 0 <= args.wait_seconds <= 7200):
        parser.error("need an absolute target checkout and polling window between 0 and 7200 seconds")
    state = args.state
    state.mkdir(mode=0o700, exist_ok=True, parents=True)
    decision = calibration_sampling.reconstruct(args.prior_dir, args.forecast, args.observations,
                                                args.control, args.weights, args.checkout)
    write_once(state / "decision.json", decision)
    if not decision.get("order_id"):
        outcome = {"stage": "deferred", "decision": str(state / "decision.json")}
        write_once(state / "outcome.json", outcome)
        print(json.dumps(outcome))
        return
    write_once(state / "order.json", decision)
    try:
        candidate = author(decision, args.checkout, state, args.author, args.archive)
    except PendingAuthor:
        print(json.dumps({"stage": "awaiting author", "request": str(state / "authoring/author/request.json")}))
        return
    until = time.monotonic() + args.wait_seconds
    while True:
        result = cycle(args, decision, candidate)
        print(json.dumps(result), flush=True)
        if result["stage"] not in ("measuring", "activate", "run", "exercise"):
            write_once(state / ("qualification.json" if result["stage"] == "qualified" else "outcome.json"), result)
            return
        if time.monotonic() >= until:
            return
        time.sleep(min(10, max(0, until - time.monotonic())))


if __name__ == "__main__":
    main()
