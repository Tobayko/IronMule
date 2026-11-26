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


def collect_http(pod, remote, local, native):
    records = local / "http-evidence"
    target = remote / "active/deployment"
    snapshot(pod, target, ["http-trial.json"], records)
    marker = records / "http-trial.json"
    if not marker.is_file():
        return None
    pin = read(marker)
    binding = pin["binding"]
    if (pin["control_receipt"] != native["receipt_sha256"] or not isinstance(binding, str)
            or len(binding) != 64 or any(ch not in "0123456789abcdef" for ch in binding)):
        raise ValueError("foreign HTTP snapshot manifest")
    snapshots = ["decision.json", "gate-ref.json", "gate-cand.json",
                 *(f"{arm}-{half}.json" for arm in ("ref", "cand") for half in (0, 1))]
    snapshot(pod, target / ("http-" + binding), snapshots, records)
    return binding


def published_http(args, cfg, order, native):
    """Only a positive, merged, target-activated native release may enter the separate product gate."""
    if not getattr(args, "serve_qualified", False) or native["stage"] != "active":
        return native
    state = args.state
    done = state / "http-outcome.json"
    key = order["order_id"]
    route = ["--reference-url", args.http_reference_url] if args.http_reference_url else ["--scoped-only"]
    write_once(state / "http-request.json", {"native": key, "reference_url": args.http_reference_url,
               "lifetime": args.http_serve_seconds, "port": args.http_port, "timeout": args.http_timeout})
    if done.exists():
        return read(done)
    pod = lease(cfg["pod_id"])
    remote = PurePosixPath(cfg["state_root"]) / key
    deploy = ["--state", remote / "active", "--checkout", cfg["checkout"], "--candidate-source", remote / "candidate.cu",
              "--reference-lib", cfg["reference_lib"], "--table", cfg["table"], "--gguf", cfg["gguf"],
              "--worker-root", cfg["worker_root"], "--trial", remote / "trial"]
    addon = state / "http-framework.json"
    if not addon.exists():
        files = ["lab/pk1/native_http.py", "lab/pk1/native_http_worker.py", "lab/rp/restricted_worker.py",
                 "ironmule_product/backend.py", "ironmule_product/http_server.py"]
        root = Path(__file__).resolve().parents[2]
        expected = [native_trial.checksum(root / f) for f in files]
        paths = [str(PurePosixPath(cfg["framework"]) / f) for f in files]
        actual = [line.split()[0] for line in shell(pod, shlex.join(["sha256sum", "--", *paths])).splitlines()]
        if actual != expected:
            raise ValueError("target HTTP qualification framework differs from control")
        write_once(addon, dict(zip(files, actual, strict=True)))
    for op in ("qualify", "exercise", "serve"):
        binding = collect_http(pod, remote, state, native)
        file = "http-" + op + ".json"
        dest = remote / file
        snapshot(pod, remote, [file], state)
        if not (state / file).exists():
            left = int(pod["deadline"] - time.time()) - 15
            timeout = min(left, args.http_timeout)
            if timeout < 1 or op == "serve" and timeout < args.http_serve_seconds:
                return {"stage": "http_lease_limit", "order": key, "pr": native["pr"], "pending": op}
            extra = [] if op == "qualify" else ["--port", args.http_port]
            if op == "serve":
                extra += [*route, "--lifetime", args.http_serve_seconds, "--canary", "--report", dest]
            cmd = command(cfg, "lab.pk1.native_http", [op, *deploy, "--timeout", timeout, *extra])
            if op != "serve":
                output = shlex.quote(str(dest))
                cmd += " > " + output + ".tmp && mv " + output + ".tmp " + output
            status = job(pod, "http-" + op + "-" + key[:16], cmd)
            if status != "finished":
                return {"stage": "http_" + op, "order": key, "pr": native["pr"], "port": args.http_port}
            snapshot(pod, remote, [file], state)
        row = read(state / file)
        if row["control_receipt"] != native["receipt_sha256"]:
            raise ValueError("foreign HTTP control receipt")
        if op == "qualify":
            binding = collect_http(pod, remote, state, native)
            if binding is None or row["binding"] != binding:
                raise ValueError("HTTP decision differs from immutable target manifest")
            if row["decision"]["passed"] is not True:
                final = {**native, "stage": "http_declined", "http": row,
                         "cost_window": cost_window(pod, state / "trial", False)}
                save(done, final)
                return final
        elif row["binding"] != binding:
            raise ValueError("mixed HTTP campaign bindings")
        elif op == "exercise":
            if not row["withdrawn"] or [r["arm"] for r in row["responses"]] != ["cand", "ref"]:
                raise ValueError("missing HTTP rollback witness")
        elif (row["selection"].get("arm") != "cand" or row["completed"] < 1
              or not any(r.get("arm") == "cand" and r.get("external_completion") for r in row["starts"])):
            final = {**native, "stage": "http_withdrawn", "http": row,
                     "cost_window": cost_window(pod, state / "trial", False)}
            save(done, final)
            return final
    cost = cost_window(pod, state / "trial", False)
    savings = []
    for half in (0, 1):
        ref, cand = (read(state / "http-evidence" / f"{arm}-{half}.json")["rows"] for arm in ("ref", "cand"))
        savings.extend(r["wall_s"] - c["wall_s"] for r, c in zip(ref, cand, strict=True))
    if len(savings) != PAIRS or min(savings) <= 0:
        raise ValueError("missing positive paired HTTP throughput evidence")
    cost["requests_to_amortize_lease_time"] = 32 * math.ceil(cost["lease_elapsed_s"] / median(savings))
    cost["amortization_basis"] = "measured 32-request HTTP waves"
    final = {**native, "stage": "http_verified", "http": row, "cost_window": cost}
    save(done, final)
    return final


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
    with (state / ".campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if done.is_file():
            return published_http(args, cfg, order, read(done))
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
            try:
                status = job(pod, "qualify-" + key[:16], command(cfg, "lab.pk1.native_trial",
                             [*common, "--order", native_order, "--output", remote_trial / "result.json"]))
            except RuntimeError:
                # The evaluator can atomically publish a negative decision just before its nonzero exit.
                snapshot(pod, remote_trial, TRIAL_FILES, trial)
                result = trial / "result.json"
                if not result.exists():
                    raise
                refused = read(result)
                if (refused.get("passed") is not False or refused.get("schema") != native_trial.SCHEMA
                        or not (trial / "context.json").is_file() or refused.get("context") != read(trial / "context.json")
                        or refused["context"].get("order") != key or refused["context"].get("base_commit") != order["base_commit"]
                        or refused["context"].get("candidate_source") != native_trial.checksum(candidate)
                        or not isinstance(refused.get("reason"), str)):
                    raise
                status = "finished"
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
        return published_http(args, cfg, order, final)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "order", "state", "checkout"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stage-only", action="store_true", help="checkpoint immutable inputs while target admission is pending")
    p.add_argument("--publish", action="store_true", help="submit a qualifying tree and activate after passing CI")
    p.add_argument("--author-cli", action="store_true", help="consume pending orders via subscription Codex")
    p.add_argument("--wait-seconds", type=int, default=0, help="bounded polling window; rerun to resume")
    p.add_argument("--serve-qualified", action="store_true", help="qualify published native winners over real product HTTP")
    routes = p.add_mutually_exclusive_group()
    routes.add_argument("--http-reference-url", help="ordinary checked model server for requests outside the validated profile")
    routes.add_argument("--dedicated-http", action="store_true", help="provide an explicitly scoped in-pod endpoint")
    p.add_argument("--http-port", type=int, default=8230)
    p.add_argument("--http-serve-seconds", type=int, default=120)
    p.add_argument("--http-timeout", type=int, default=1800)
    args = p.parse_args()
    if args.serve_qualified and (not args.publish or not (args.http_reference_url or args.dedicated_http)):
        p.error("--serve-qualified requires --publish and --http-reference-url or --dedicated-http")
    if not 1 <= args.http_port <= 65535 or not 1 <= args.http_serve_seconds <= args.http_timeout <= 3600:
        p.error("invalid HTTP port or lifetime")
    if not 0 <= args.wait_seconds <= 7200:
        p.error("wait-seconds must be in 0..7200")
    until = time.monotonic() + args.wait_seconds
    while True:
        outcome = cycle(args)
        print(json.dumps(outcome), flush=True)
        if outcome["stage"] in {"active", "withdrawn", "qualified", "rejected", "staged", "http_verified", "http_declined", "http_withdrawn", "http_lease_limit"} or time.monotonic() >= until:
            return
        time.sleep(max(0, min(10, until - time.monotonic())))


if __name__ == "__main__":
    main()
