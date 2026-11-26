"""Commission and evaluate one measured GPU order on an existing budgeted RunPod lease."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import tempfile
import time
import urllib.parse
from pathlib import Path, PurePosixPath
from statistics import median
from types import SimpleNamespace

from lab.pk1 import native_release, native_trial
from training.code_author import PendingAuthor, digest, write_once
from training.code_trial import PAIRS, read, save

TRIAL_FILES = ["context.json", "build.json", "candidate.so", "calibration.json", "screen.json",
               *(f"{arm}-{i}.json" for i in range(PAIRS) for arm in ("ref", "cand")), "result.json"]
CONFIG_KEYS = {"pod_id", "framework", "checkout", "python", "env", "gguf",
               "reference_lib", "table", "worker_root", "state_root"}


def lease(expected_id):
    from lab.rp import rp
    from lab.rp.accounting import held, limit, locked

    with locked(rp.STATE):
        state = rp.load()
        pod = state.get("pod")
        if not pod or pod.get("id") != expected_id or pod["deadline"] <= time.time():
            raise ValueError("need the same live, reserved pod")
        if state["spent"] + held(state) > limit(state):
            raise ValueError("campaign reservation exhausted")
        return pod


def call(pod, method, endpoint, body=None, raw=False):
    from lab.rp import rp
    return rp.agent(pod, method, endpoint, body, timeout=95, raw=raw)


def endpoint(op, path):
    return "/" + op + "?" + urllib.parse.urlencode({"path": str(path)})


def upload(pod, remote, content):
    if call(pod, "POST", endpoint("put", remote), content).get("bytes") != len(content):
        raise OSError("incomplete transfer to target")


def shell(pod, command):
    result = call(pod, "POST", "/run", json.dumps({"cmd": command, "timeout": 50}).encode())
    if result.get("rc") != 0:
        raise RuntimeError("target preparation failed: " + result.get("out", "")[-600:])
    return result["out"].strip()


def snapshot(pod, directory, names, local):
    """Capture complete, atomically published files. Existing bytes are immutable for this order."""
    local.mkdir(exist_ok=True, parents=True)
    missing = [name for name in names if not (local / name).is_file()]
    if not missing:
        return
    paths = shlex.join([str(directory / name) for name in missing])
    manifest = shell(pod, 'for p in ' + paths + '; do if test -f "$p"; then sha256sum -- "$p"; fi; done')
    for line in manifest.splitlines():
        digest, full = line.split(None, 1)
        name = PurePosixPath(full).name
        if name not in missing or full != str(directory / name):
            raise ValueError("unexpected target artifact")
        data = call(pod, "GET", endpoint("get", full), raw=True)
        if hashlib.sha256(data).hexdigest() != digest:
            raise OSError("target bytes changed during transfer")
        if name.endswith(".json"):
            json.loads(data)
        with tempfile.NamedTemporaryFile(dir=local, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, local / name)


def command(cfg, module, argv):
    return ("set -e; . " + shlex.quote(cfg["env"]) + "; cd " + shlex.quote(cfg["framework"]) +
            "; export PYTHONPATH=" + shlex.quote(cfg["framework"]) + "; " +
            shlex.join([cfg["python"], "-m", module, *map(str, argv)]))


def job(pod, name, cmd):
    """A bridge restart may forget a process. Target-side flock protects the native qualifier on resume."""
    query = "/job?" + urllib.parse.urlencode({"name": name, "tail": 4})
    current = call(pod, "GET", query)
    if current.get("running"):
        return "running"
    if current.get("rc") is not None:
        if current["rc"] != 0:
            raise RuntimeError(f"{name} failed: {current['tail'][-500:]}")
        return "finished"
    started = call(pod, "POST", "/job", json.dumps({"name": name, "cmd": cmd}).encode())
    if started.get("started") != name and started.get("error") != "running":
        raise OSError("target did not acknowledge " + name)
    return "running"


def cost_window(pod, trial, passed):
    """Charge the complete lease-to-date, including calibration, contention, idle periods and author wait."""
    elapsed = max(0.0, time.time() - pod["started"])
    summary = {"lease_elapsed_s": elapsed, "target_hourly_usd": pod["cost"],
               "estimated_lease_usd_before_tail": elapsed * pod["cost"] / 3600,
               "reserved_usd_including_tail": pod["reserved"], "requests_to_amortize_lease_time": None}
    if passed:
        gain = median(read(trial / f"ref-{i}.json")["wall_s"] - read(trial / f"cand-{i}.json")["wall_s"]
                      for i in range(PAIRS))
        if gain > 0:
            summary["requests_to_amortize_lease_time"] = 32 * math.ceil(elapsed / gain)
    return summary


def cycle(args):
    cfg = read(args.config)
    if set(cfg) != CONFIG_KEYS or any(not PurePosixPath(cfg[k]).is_absolute() for k in CONFIG_KEYS - {"pod_id"}):
        raise ValueError("invalid target configuration")
    order = read(args.order)
    key = order["order_id"]
    if len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key):
        raise ValueError("invalid order id")
    if (hashlib.sha256(json.dumps({k: v for k, v in order.items() if k != "order_id"}, sort_keys=True).encode()).hexdigest() != key
            or native_trial.checksum(args.order.with_suffix(".profile.json")) != order["profile_sha256"]):
        raise ValueError("order or adjacent profile changed")
    state = args.state
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_once(state / "campaign.json", {"order": key, "config": digest(cfg),
               "evaluator": native_trial.checksum(native_trial.__file__), "base": order["base_commit"]})
    done = state / "outcome.json"
    if done.is_file():
        return read(done)
    with (state / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidate = state / "authoring/candidate.cu"
        author_args = SimpleNamespace(order=args.order, checkout=args.checkout, output=candidate)
        try:
            native_trial.author(author_args)
        except PendingAuthor:
            if not args.author_cli:
                return {"stage": "awaiting_author", "request": str(candidate.parent / "author/request.json")}
            from training.code_author import consume_cli
            consume_cli(candidate.parent)
            native_trial.author(author_args)
        pod = lease(cfg["pod_id"])
        remote = PurePosixPath(cfg["state_root"]) / key
        trial = state / "trial"
        remote_trial = remote / "trial"
        native_order = remote / "order.json"
        remote_candidate = remote / "candidate.cu"
        if not (state / "staged.json").exists():
            if shell(pod, shlex.join(["sha256sum", str(PurePosixPath(cfg["framework"]) / "lab/pk1/native_trial.py")])).split()[0] != native_trial.checksum(native_trial.__file__):
                raise ValueError("target uses a different evaluator")
            for target, local in [(native_order, args.order), (remote / "order.profile.json", args.order.with_suffix(".profile.json")),
                                  (remote_candidate, candidate)]:
                upload(pod, target, local.read_bytes())
            snapshot(pod, PurePosixPath(cfg["table"]).parent, [PurePosixPath(cfg["table"]).name], state)
            if native_trial.checksum(state / PurePosixPath(cfg["table"]).name) != order["binding"]["table_sha256"]:
                raise ValueError("staged dispatch table differs from profile")
            write_once(state / "staged.json", {"candidate": native_trial.checksum(candidate), "order": key})
        if getattr(args, "stage_only", False):
            return {"stage": "staged", "order": key, "source": native_trial.checksum(candidate)}
        common = ["--checkout", cfg["checkout"], "--candidate-source", remote_candidate,
                  "--reference-lib", cfg["reference_lib"], "--table", cfg["table"], "--gguf", cfg["gguf"],
                  "--worker-root", cfg["worker_root"]]
        snapshot(pod, remote_trial, TRIAL_FILES, trial)
        if not (trial / "result.json").exists():
            status = job(pod, "qualify-" + key[:16], command(cfg, "lab.pk1.native_trial",
                         [*common, "--order", native_order, "--output", remote_trial / "result.json"]))
            if status == "finished":
                snapshot(pod, remote_trial, TRIAL_FILES, trial)
                if not (trial / "result.json").exists():
                    raise ValueError("completed target job missing its qualification")
            else:
                return {"stage": "measuring", "captured": len(list(trial.glob("*.json")))}
        outcome = read(trial / "result.json")
        if outcome.get("context", {}).get("order") != key:
            raise ValueError("unbound trial result")
        if outcome["passed"] is False:
            final = {"stage": "rejected", "order": key, "result": outcome, "cost_window": cost_window(pod, trial, False)}
            save(done, final)
            return final
        control = native_release.control(SimpleNamespace(order=args.order, trial=trial, state=state / "release",
                    table=state / PurePosixPath(cfg["table"]).name, candidate_source=candidate, publish=args.publish))
        if not args.publish:
            return {"stage": "qualified", **control}
        pod = lease(cfg["pod_id"])
        receipt = state / "release/deployment/control.json"
        upload(pod, remote / "control.json", receipt.read_bytes())
        deploy = [*common, "--trial", remote_trial, "--state", remote / "active"]
        for op, extra in [("activate", ["--receipt", remote / "control.json", "--receipt-sha256", control["receipt_sha256"]]),
                          ("run", ["--seed", "93742"])]:
            filename = op + ".json"
            snapshot(pod, remote, [filename], state)
            if not (state / filename).exists():
                cmd = command(cfg, "lab.pk1.native_release", [op, *deploy, *extra])
                dest = shlex.quote(str(remote / filename))
                status = job(pod, op + "-" + key[:16], cmd + " > " + dest + ".tmp && mv " + dest + ".tmp " + dest)
                if status != "finished":
                    return {"stage": op, "pr": control["pr"]}
                snapshot(pod, remote, [filename], state)
            if op == "activate" and read(state / filename).get("arm") != "cand":
                raise ValueError("activation fell back to its reference")
        row = read(state / "run.json")
        active = row["treatment"]["binary"] == read(trial / "build.json")["binary"]
        final = {"stage": "active" if active else "withdrawn", "order": key, **control,
                 "completed": row["completed"], "wall_s": row["wall_s"], "ttft_s": row["ttft_s"],
                 "cost_window": cost_window(pod, trial, active)}
        save(done, final)
        return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "order", "state", "checkout"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stage-only", action="store_true", help="checkpoint immutable inputs while target admission is pending")
    p.add_argument("--publish", action="store_true", help="submit a qualifying tree and activate after passing CI")
    p.add_argument("--author-cli", action="store_true", help="consume pending orders via subscription Codex")
    p.add_argument("--wait-seconds", type=int, default=0, help="bounded polling window; rerun to resume")
    args = p.parse_args()
    if not 0 <= args.wait_seconds <= 7200:
        p.error("wait-seconds must be in 0..7200")
    until = time.monotonic() + args.wait_seconds
    while True:
        outcome = cycle(args)
        print(json.dumps(outcome), flush=True)
        if outcome["stage"] in {"active", "withdrawn", "qualified", "rejected", "staged"} or time.monotonic() >= until:
            return
        time.sleep(max(0, min(10, until - time.monotonic())))


if __name__ == "__main__":
    main()
