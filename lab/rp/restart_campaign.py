"""Run one source-bound repeat-start experiment on the already reserved A100."""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from lab.pk1 import restart_order, restart_trial
from lab.pk1.native_trial import checksum
from lab.rp.native_campaign import (
    CONFIG_KEYS,
    command,
    job,
    lease,
    shell,
    snapshot,
    upload,
)
from training.code_author import PendingAuthor, write_once
from training.code_trial import PAIRS, read, save

FILES = ["context.json", "aa-0.json", "aa-1.json", "receipt.json", "stale-receipt.json", "stale.json", "missing.json",
         *(f"{tag}-{i}.json" for i in range(PAIRS) for tag in ("ref", "cand")), "result.json"]
FRAMEWORK = ["lab/pk1/restart_trial.py", "lab/pk1/restart_arm.py", "lab/pk1/restart_order.py", "lab/pk1/native_trial.py"]


def cycle(args):
    cfg = read(args.config)
    if set(cfg) != CONFIG_KEYS or any(not PurePosixPath(cfg[k]).is_absolute() for k in CONFIG_KEYS - {"pod_id"}):
        raise ValueError("invalid reserved target")
    order = read(args.order)
    restart_trial.verify_order(order, args)
    key = order["order_id"]
    pod = lease(cfg["pod_id"])
    state = args.state
    state.mkdir(exist_ok=True, parents=True)
    source = state / "authoring/candidate.py"
    try:
        restart_order.author(args.order, args.checkout, source)
    except PendingAuthor:
        return {"stage": "awaiting author", "request": str(source.parent / "author/request.json")}
    root = Path(__file__).resolve().parents[2]
    framework = {p: checksum(root / p) for p in FRAMEWORK}
    write_once(state / "campaign.json", {"order": key, "candidate": checksum(source), "framework": framework,
                                         "config_sha256": checksum(args.config)})
    remote = PurePosixPath(cfg["state_root"]) / ("restart-" + key)
    data = remote / "trial"
    local = state / "trial"
    if not (state / "staged.json").exists():
        remote_root = PurePosixPath(cfg["framework"])
        for p, d in framework.items():
            if shell(pod, shlex.join(["sha256sum", str(remote_root / p)])).split()[0] != d:
                raise ValueError("different remote evaluator: " + p)
        upload(pod, remote / "order.json", args.order.read_bytes())
        upload(pod, remote / "proposal.py", source.read_bytes())
        write_once(state / "staged.json", {"order": key, "candidate": checksum(source)})
    if args.stage_only:
        return {"stage": "staged", "order": key}
    snapshot(pod, data, FILES, local)
    outcome = local / "result.json"
    if not outcome.exists():
        worker = command(cfg, "lab.pk1.restart_trial", ["--order", remote / "order.json", "--candidate-source", remote / "proposal.py",
                         "--trial", data, "--checkout", cfg["checkout"], "--reference-lib", cfg["reference_lib"],
                         "--table", cfg["table"], "--gguf", cfg["gguf"], "--worker-root", cfg["worker_root"]])
        try:
            job(pod, "restart-" + key[:18], worker)
        except RuntimeError:
            snapshot(pod, data, FILES, local)
            if not outcome.exists():
                raise
        if not outcome.exists():
            return {"stage": "measuring", "order": key, "saved_receipts": len(list(local.glob("*.json")))}
    result = read(outcome)
    check = SimpleNamespace(order=args.order, checkout=args.checkout, trial=local, candidate_source=source)
    if result["schema"] != restart_order.SCHEMA or result["context"] != restart_trial.context(check, order):
        raise ValueError("unbound target result")
    if not result["passed"]:
        return {"stage": "rejected", "order": key, "reason": result.get("reason", "cold/warm hardware gate"),
                "elapsed_s": result["elapsed_s"]}
    verified = restart_trial.qualification(check)
    save(state / "qualified.json", verified)
    return {"stage": "qualified", "order": key, "metrics": verified["metrics"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "order", "checkout", "state"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--stage-only", action="store_true")
    p.add_argument("--wait-seconds", type=int, default=0)
    a = p.parse_args()
    if not 0 <= a.wait_seconds <= 7200:
        p.error("invalid polling window")
    until = time.monotonic() + a.wait_seconds
    while True:
        outcome = cycle(a)
        print(json.dumps(outcome), flush=True)
        if outcome["stage"] != "measuring" or time.monotonic() >= until:
            return
        time.sleep(min(10, max(0, until - time.monotonic())))


if __name__ == "__main__":
    main()
