"""One bounded EPYC prefill order, restricted target trial, scoped publication and fresh-serving release."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

from lab.rp import cpu_code_release, cpu_code_trial
from lab.rp import native_campaign as transport
from training import cpu_prefill_order, experience_archive, scoped_release
from training.code_author import PendingAuthor, consume_cli, write_once
from training.code_trial import PAIRS, ROOT, develop, git, prepare, read, save

SOURCE = cpu_code_trial.SOURCE
@dataclass(frozen=True)
class Suite:
    matched: bool = False

    @property
    def seed(self):
        return 18351 if self.matched else 8351

    @property
    def suffix(self):
        return "-matched-v2" if self.matched else ""

    @property
    def broker(self):
        if self.matched:
            from lab.rp import cpu_matched_trial
            return cpu_matched_trial
        return cpu_code_trial

    def arms(self):
        return [name for i in range(PAIRS) for name in
                [f"reference-{self.seed+i}", f"candidate-{self.seed+i}", f"rule-{self.seed+i if self.matched else i}"]]


BROKER = ("lab/rp/cpu_code_trial.py", "lab/rp/cpu_four_thread_rule.py", "lab/rp/restricted_worker.py",
          "training/code_trial.py", "lab/pk1/native_trial.py")
DEPLOY = ("training/scoped_release.py", "training/trial_release.py", "training/code_author.py", "lab/rp/cpu_code_release.py")
MATCHED_FILES = ("lab/rp/cpu_matched_profile.py", "lab/rp/cpu_matched_trial.py")
MATCHED_RELEASE_FILES = (*MATCHED_FILES, "lab/rp/cpu_matched_release.py")
PRODUCT = ("lab/rp/cpu_product_release.py", "lab/rp/cpu_product_worker.py")


def sha(path):
    return cpu_code_trial.sha(path)


def target(config):
    cfg = read(config)
    keys = {"pod_id", "framework", "env", "python", "lane", "worker_root", "gguf"}
    if set(cfg) != keys or not re.fullmatch(r"[a-zA-Z0-9-]+", cfg["pod_id"]):
        raise ValueError("invalid target configuration")
    for key in keys - {"pod_id"}:
        path = PurePosixPath(cfg[key])
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"expected absolute target path for {key}")
    return cfg


def load_order(args):
    if args.order:
        order = read(args.order)
    else:
        if not (args.evidence and args.prior):
            raise ValueError("planning requires a serving receipt and its attribution directory")
        archive = args.archive or args.state / "archive.jsonl"
        if not archive.exists():
            if args.archive:
                raise ValueError("supplied experience archive is missing")
            experience_archive.write_archive(archive, experience_archive.build())
        order = cpu_prefill_order.plan(args.prior, args.evidence / "waves", args.evidence / "prefill-attribution.json",
                                       args.evidence / "planner-decision.json", archive, ROOT)
    write_once(args.state / "order.json", order)
    if order.get("schema") == cpu_prefill_order.DEFERRED:
        return order
    scoped_release.validate_order(cpu_code_release.ADAPTER, order)
    return order


def make_source(order, authorship, cli):
    authorship.mkdir(mode=0o700, parents=True, exist_ok=True)
    trees = [prepare(authorship, order["base_commit"], name) for name in ("reference", "candidate")]
    ref, candidate = trees
    if git("status", "--porcelain", cwd=ref):
        raise ValueError("reference checkout changed")
    with patch.dict(os.environ, IRONMULE_CODE_AUTHOR="hosted"):
        try:
            edited = develop(order, candidate, authorship)
        except PendingAuthor:
            if not cli:
                raise
            consume_cli(authorship)
            edited = develop(order, candidate, authorship)
    if edited != SOURCE or git("diff", "HEAD", "--name-only", cwd=candidate) not in ("", SOURCE):
        raise ValueError("unexpected authored source changes")
    if git("ls-files", "--others", "--exclude-standard", cwd=candidate):
        raise ValueError("untracked source in generated trial")
    scoped_release.check_method(cpu_code_release.ADAPTER, ref / SOURCE, candidate / SOURCE)
    return trees


def put_once(pod, path, data):
    """Publish an immutable artifact from a content-addressed upload; never overwrite a different target."""
    q = shlex.quote
    expected = hashlib.sha256(data).hexdigest()
    dest = q(str(path))
    found = transport.shell(pod, f"if test -e {dest}; then sha256sum -- {dest}; fi")
    if found:
        if found.split()[0] != expected:
            raise ValueError(f"different target contents at {path}")
        return
    tmp = str(path) + "." + secrets.token_hex(12) + ".incoming"
    transport.upload(pod, tmp, data)
    quoted = q(tmp)
    transport.shell(pod, f"set -e; test \"$(sha256sum -- {quoted} | cut -d' ' -f1)\" = {expected}; "
                    f"chmod 0444 {quoted}; ln -- {quoted} {dest} || cmp -s -- {dest} {quoted}; "
                    f"test \"$(sha256sum -- {dest} | cut -d' ' -f1)\" = {expected}; rm -- {quoted}")


def verify_remote(pod, framework, files):
    paths = [PurePosixPath(framework) / f for f in files]
    rows = transport.shell(pod, "sha256sum -- " + shlex.join([str(p) for p in paths])).splitlines()
    if len(rows) != len(paths) or any(line.split(maxsplit=1) != [sha(ROOT / name), str(path)]
                                       for line, name, path in zip(rows, files, paths, strict=True)):
        raise ValueError("trusted broker on target differs from the control checkout")


def release_framework(pod, cfg, lane, files):
    # An immutable overlay avoids replacing modules used by other lanes or by a currently running broker.
    pin = hashlib.sha256(json.dumps({name: sha(ROOT / name) for name in files}, sort_keys=True).encode()).hexdigest()
    prefix = lane / "trusted" / pin
    for name in files:
        put_once(pod, prefix / name, (ROOT / name).read_bytes())
    verify_remote(pod, prefix, files)
    return {**cfg, "framework": str(prefix), "dependency_path": cfg["framework"]}


def stage(pod, cfg, order, args, reference, candidate):
    lane = PurePosixPath(cfg["lane"])
    remote = lane / "inputs" / order["order_id"]
    config = lane / "config"
    transport.shell(pod, "umask 022; mkdir -p -- " + shlex.join([str(remote), str(config), str(lane / "runs")]))
    source = candidate / SOURCE
    head = args.head.read_bytes()
    if hashlib.sha256(head).hexdigest() != order["binding"]["scratch_thread_head"]["weights_sha256"]:
        raise ValueError("CPU head differs from discovery")
    archive = subprocess.check_output(["git", "-C", str(reference), "archive", "--format=tar.gz", "HEAD"])
    stage_file = Path(__file__).with_name("cpu_campaign_stage.py")
    for path, data in [(remote / "archive.tar.gz", archive), (remote / "engine.py", source.read_bytes()),
                       (remote / "stage.py", stage_file.read_bytes()), (config / "head.npz", head),
                       (config / "order.json", ((args.order if args.order else args.state / "order.json").read_bytes()))]:
        put_once(pod, path, data)
    return [str(remote / "stage.py"), "--archive", str(remote / "archive.tar.gz"),
            "--archive-sha", hashlib.sha256(archive).hexdigest(), "--reference-sha", sha(reference / SOURCE),
            "--candidate-sha", sha(source), "--candidate", str(remote / "engine.py"), "--target", str(lane)]


def state_of_job(pod, name):
    return transport.call(pod, "GET", "/job?" + urllib.parse.urlencode({"name": name, "tail": 3}))


def active_qualifier(pod, cfg, remote_state):
    """A bridge restart forgets process IDs. Inspect actual module argv before issuing a duplicate."""
    code = """from pathlib import Path
import sys
for f in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv = f.read_bytes().split(b'\\0')
    except OSError:
        continue
    if any(m in argv for m in (b'lab.rp.cpu_code_trial', b'lab.rp.cpu_matched_trial')) and b'qualify' in argv and sys.argv[1].encode() in argv:
        print('running')
        break
"""
    return transport.shell(pod, shlex.join([cfg["python"], "-c", code, str(remote_state)])) == "running"


def job_command(cfg, pod, argv, affinity, lock, seconds):
    remaining = min(seconds, int(pod["deadline"] - time.time() - 10))
    if remaining < 1:
        raise ValueError("lease deadline reached")
    # Wait for custody of the whole host as well as protecting the individual order.
    # Keep the lease deadline across both the wait and the subsequent CPU measurement.
    measured = transport.exclusive_measurement(cfg, ["flock", "-n", str(lock),
        "taskset", "-c", ",".join(map(str, affinity)), *argv], max(1, remaining - 1))
    cmd = ["timeout", "-k", "5", str(remaining), "/bin/sh", "-ec", measured]
    return ("set -e; . " + shlex.quote(cfg["env"]) + "; cd " + shlex.quote(cfg["framework"]) + "; export PYTHONPATH=" +
            shlex.quote(os.pathsep.join(filter(None, (cfg["framework"], cfg.get("dependency_path"))))) + "; export PYTHONDONTWRITEBYTECODE=1; exec " + shlex.join(cmd))


def snapshot_trial(pod, remote, raw, suite):
    receipts = ["context.json", "build.json", "verdict.json", *(arm + ".json" for arm in suite.arms())]
    if suite.matched:
        receipts.append("profile.json")
    transport.snapshot(pod, remote, receipts, raw)
    if (raw / "build.json").is_file():
        transport.snapshot(pod, remote, ["build-0.log", "build-1.log"], raw)


def profile_conflict(raw, order):
    """Describe an invalid v1 pair even when its target worker stopped before writing a verdict."""
    from training.code_trial import verify_deliveries
    for i in range(PAIRS):
        seed = 8351 + i
        paths = {a: raw / f"{a}-{seed}.json" for a in ("reference", "candidate")}
        if not all(p.is_file() for p in paths.values()):
            break
        rows = {a: read(p) for a, p in paths.items()}
        profiles = {a: r["binding"]["scratch_thread_head"]["qualified_offsets"] for a, r in rows.items()}
        if profiles["reference"] == profiles["candidate"]:
            continue
        context = read(raw / "context.json")
        if context["order_id"] != order["order_id"]:
            raise ValueError("target conflict belongs to another order")
        for a, row in rows.items():
            if (row["runtime_source_sha256"]["ggml_engine.py"] != context[a + "_sha256"]
                    or row["workload"]["seed"] != seed or row["completed"] != row["correct"] or row["correct"] != 16):
                raise ValueError("incomplete or misleading first-pair conflict")
            verify_deliveries(row, seed)
        ref, cand = rows.values()
        same = ref["output_sha256"] == cand["output_sha256"]
        quality = (f"both completed 16 correct and {ref['tokens']} matching output tokens, "
                   "not a matched speed comparison") if same else "output differs across these arms"
        rejection = {"schema": "ironmule.confounded-cpu-pair.v1", "order": order["order_id"], "seed": seed,
                     "passed": False, "training_label": False, "observed_profiles": profiles, "quality_scope": quality,
                     "correct": {a: r["correct"] for a, r in rows.items()}, "wall_s": {a: r["wall_s"] for a, r in rows.items()},
                     "reason": "independent startup qualification produced different complete decode profiles"}
        write_once(raw / "comparison-rejected.json", rejection)
        return {"stage": "confounded", "result": rejection}
    if (raw / "comparison-rejected.json").is_file():
        raise ValueError("unsubstantiated v1 profile mismatch")


def summary(pod, trial, outcome, seed):
    cost = transport.cost_window(pod, trial, False)
    if outcome and outcome["passed"]:
        gains = [read(trial / f"reference-{seed+i}.json")["wall_s"] - read(trial / f"candidate-{seed+i}.json")["wall_s"]
                 for i in range(PAIRS)]
        # Each arm completes 16 requests.
        if median(gains) > 0:
            import math
            cost["requests_to_amortize_lease_time"] = 16 * math.ceil(cost["lease_elapsed_s"] / median(gains))
    return cost


def advance(args):
    state = args.state
    cfg = target(args.target)
    suite = Suite(args.matched)
    order = load_order(args)
    if order.get("schema") == cpu_prefill_order.DEFERRED:
        return {"stage": "deferred", "reason": order["reason"]}
    key = order["order_id"]
    lane = PurePosixPath(cfg["lane"])
    remote = lane / "runs" / (key[:16] + suite.suffix)
    raw = state / "qualification"
    old_context = raw / "context.json"
    if old_context.exists():
        context = read(old_context)
        if context["order_id"] != key or suite.matched != (context.get("schema") == "ironmule.epyc-matched-code-trial.v2"):
            raise ValueError("reuse the original suite and order or a separate local state directory")
    if not suite.matched and (stopped := profile_conflict(raw, order)):
        return stopped
    pod = transport.lease(cfg["pod_id"])
    auth = args.authorship or state / "authoring"
    try:
        reference, candidate = make_source(order, auth, args.author_cli)
    except PendingAuthor:
        return {"stage": "awaiting Codex", "request": str(auth / "author/request.json")}
    if sha(reference / SOURCE) == sha(candidate / SOURCE):
        return {"stage": "declined", "reason": "no change to the measured source"}
    name = args.job_name or "epyc-qualify-" + key[:16] + suite.suffix
    if not re.fullmatch(r"epyc-qualify-" + key[:16] + r"(?:-[a-z0-9-]+)?", name):
        raise ValueError("expected a qualification job for this order")
    write_once(state / "target-binding.json", {"order": key, "target": cfg, "reference": sha(reference / SOURCE),
               "candidate": sha(candidate / SOURCE), "seed": suite.seed,
               "broker": {f: sha(ROOT / f) for f in (*BROKER, *(MATCHED_FILES if suite.matched else ())) }})
    write_once(state / (name + ".json"), {"job": name, "order": key, "remote_state": str(remote)})
    options = {"state": raw, "trial": raw, "authorship": auth, "reference": reference,
               "candidate": candidate, "order": state / "order.json", "seed": suite.seed, "matched": suite.matched}
    common = ["--worker-root", cfg["worker_root"], "--python", cfg["python"], "--reference", str(lane / "base"),
              "--candidate", str(lane / "candidate"), "--order", str(lane / "config/order.json"),
              "--head", str(lane / "config/head.npz"), "--gguf", cfg["gguf"]]
    affinity = order["binding"]["affinity"]
    if affinity != [0, 1, 2, 3]:
        raise ValueError("the scoped EPYC lane requires its qualified four-core affinity")
    snapshot_trial(pod, remote, raw, suite)
    if not suite.matched and (stopped := profile_conflict(raw, order)):
        return {**stopped, "cost": summary(pod, raw, None, suite.seed)}
    if not (raw / "verdict.json").exists():
        found = state_of_job(pod, name)
        if found["running"] or active_qualifier(pod, cfg, remote):
            return {"stage": "measuring", "job": name, "completed_arms": sum((raw / (a + ".json")).is_file() for a in suite.arms())}
        if found["rc"] not in (None, 75):
            snapshot_trial(pod, remote, raw, suite)
            if not (raw / "verdict.json").exists():
                if not suite.matched and (stopped := profile_conflict(raw, order)):
                    return {**stopped, "cost": summary(pod, raw, None, suite.seed)}
                ext = ".log" if suite.matched else "-0.log"
                logs = [arm + ext for arm in suite.arms()]
                transport.snapshot(pod, remote, ["build-0.log", "build-1.log", *logs, "profile.log"], raw)
                return {"stage": "incomplete", "job": name, "exit": found["rc"], "diagnostic": found["tail"][-600:],
                        "cost": summary(pod, raw, None, suite.seed)}
        else:
            verify_remote(pod, cfg["framework"], BROKER)
            stage_argv = stage(pod, cfg, order, args, reference, candidate)
            # One target-side lock covers staging, the complete archive and the actual evaluator.
            prepare = shlex.join([cfg["python"], *stage_argv])
            if suite.matched:
                cfg = release_framework(pod, cfg, lane, MATCHED_FILES)
            run = shlex.join([cfg["python"], "-u", "-m", suite.broker.__name__, "qualify", "--state", str(remote), *common])
            argv = ["/bin/bash", "-ec", prepare + " && exec " + run]
            cmd = job_command(cfg, pod, argv, affinity, lane / "runs" / (key[:16] + ".lock"), 2200)
            if args.prepare_only:
                write_once(state / "prepared.json", {"order": key, "suite": suite.broker.__name__, "job": name,
                           "remote_state": str(remote), "framework": cfg["framework"], "stage": stage_argv})
                return {"stage": "waiting for target", "job": name, "prepared": str(state / "prepared.json")}
            transport.job(pod, name, cmd)
            return {"stage": "measuring", "job": name}
    if args.prepare_only:
        return {"stage": "waiting for target", "job": name}
    observation = suite.broker.verify_offline(SimpleNamespace(**options))
    if observation != read(raw / "verdict.json"):
        raise ValueError("target result differs from independent raw replay")
    cost = summary(pod, raw, observation, suite.seed)
    if not observation["passed"]:
        return {"stage": "rejected", "metrics": observation["metrics"], "cost": cost}
    control = state / "control"
    pin = control / "deployment/control.json"
    if not pin.exists():
        result = cpu_code_release.control(SimpleNamespace(**{**options, "state": control, "publish": args.publish}))
        if "receipt_sha256" not in result:
            return {"stage": "qualified", **result, "cost": cost}
    if args.stop_after_publication:
        receipt = read(pin)
        return {"stage": "published", "pr": receipt["publication"]["pr"], "receipt_sha256": sha(pin), "cost": cost}
    # CI can outlast the original control pass. Re-read the live budget and deadline before activating.
    pod = transport.lease(cfg["pod_id"])
    result = deploy(pod, cfg, state, lane, key, pin, common, affinity, cost, suite, remote)
    if result["stage"] in ("active", "withdrawn") and args.product_requests:
        return product(pod, cfg, state, lane, key, common, affinity, result, args.product_requests, suite, remote / "profile.json")
    return result


def deploy(pod, cfg, state, lane, key, pin, common, affinity, cost, suite, trial):
    remote = lane / "deployments" / (key + suite.suffix)
    deployment = remote / "deployment"
    local = state / "target-release"
    local.mkdir(parents=True, exist_ok=True)
    token = sha(pin)
    receipt = read(pin)
    adapter = cpu_code_release.MATCHED if suite.matched else cpu_code_release.ADAPTER
    if cpu_code_release.author_order(receipt)["order_id"] != key or receipt["adapter"] != adapter:
        raise ValueError("unrelated control receipt")
    verify_remote(pod, cfg["framework"], BROKER)
    cfg = release_framework(pod, cfg, lane, (*DEPLOY, *(MATCHED_RELEASE_FILES if suite.matched else ())))
    dest = lane / "inputs" / key / ("control-" + token + ".json")
    put_once(pod, dest, pin.read_bytes())
    transport.shell(pod, "mkdir -p -- " + shlex.quote(str(remote)))

    def current():
        path = shlex.quote(str(deployment / "selection.json"))
        value = transport.shell(pod, f"if test -f {path}; then cat -- {path}; fi")
        return json.loads(value) if value else {}

    if suite.matched:
        common = [*common, "--profile", str(trial / "profile.json")]
    base_seed = 19352 if suite.matched else 9352
    stages = (("activate", "target-check.json", ["--receipt", str(dest)]),
              ("run", f"work-{base_seed}.json", ["--seed", str(base_seed)]),
              ("exercise", "rollback-exercise.json", ["--seed", str(base_seed + 1)]),
              # A new target process reads the real persisted selection after the separate rollback exercise.
              ("run", f"work-{base_seed+2}.json", ["--seed", str(base_seed + 2)]))
    for mode, marker, extras in stages:
        selection = current()
        if selection.get("withdrawn") is True:
            save(local / "current.json", selection)
            return {"stage": "withdrawn", "selection": selection, "pr": receipt["publication"]["pr"], "cost": cost}
        transport.snapshot(pod, deployment, [marker], local)
        if (local / marker).exists():
            continue
        name = "epyc-" + marker.removesuffix(".json") + "-" + key[:16] + suite.suffix
        command = [cfg["python"], "-m", "lab.rp.cpu_code_release", mode, "--state", str(remote),
                   "--receipt-sha256", token, *common, *extras]
        cmd = job_command(cfg, pod, command, affinity, lane / "runs" / (key[:16] + ".lock"), 720)
        job = state_of_job(pod, name)
        if job["rc"] is not None:
            transport.snapshot(pod, deployment, [marker], local)
            if not (local / marker).exists():
                selection = current()
                return {"stage": "withdrawn" if selection.get("withdrawn") else "incomplete",
                        "job": name, "exit": job["rc"], "selection": selection, "diagnostic": job["tail"][-600:], "cost": cost}
        elif not job["running"]:
            transport.job(pod, name, cmd)
        return {"stage": "target " + mode, "job": name, "pr": receipt["publication"]["pr"], "cost": cost}

    checked, first, rollback, restarted = [read(local / m) for _, m, _ in stages]
    expected = receipt["qualification"]
    scoped_release.validate(checked)
    if (any(checked[k] != expected[k] for k in scoped_release.BOUND)
            or any(row["arm"] != "candidate" or row["source_sha256"] != expected["candidate_sha256"] or row["correct"] != 16
                   or row["completed"] != 16 or not row["scratch_choices"] for row in (first, restarted))
            or rollback.get("passed") is not True or rollback["receipt_sha256"] != token
            or rollback["before"]["arm"] != "candidate" or rollback["after"]["arm"] != "reference"
            or rollback["before"]["work_sha256"] != rollback["after"]["work_sha256"]):
        raise ValueError("incomplete target qualification, use, rollback or restart evidence")
    selection = current()
    save(local / "current.json", selection)
    if (selection.get("arm") != "candidate" or selection.get("withdrawn") is not False
            or selection.get("receipt_sha256") != token or selection.get("target_check_sha256") != sha(local / stages[0][1])):
        return {"stage": "withdrawn", "selection": selection, "pr": receipt["publication"]["pr"], "cost": cost}
    return {"stage": "active", "selection": selection, "pr": receipt["publication"]["pr"], "receipt_sha256": token,
            "use": first, "restart": restarted, "rollback": rollback, "cost": cost}


def product(pod, cfg, state, lane, key, common, affinity, deployed, requests, suite, profile):
    """Independently qualify the product bridge and dispatch the supplied batch through a fresh process."""
    remote = lane / "deployments" / (key + suite.suffix)
    deployment = remote / "deployment"
    local = state / "target-release"
    content = requests.read_bytes()
    if len(content) > 128 * 1024:
        raise ValueError("oversized product request file")
    request = json.loads(content)
    if not isinstance(request.get("prompts"), list) or not 1 <= len(request["prompts"]) <= 4:
        raise ValueError("expected a batch of one to four prompts")
    req_hash = hashlib.sha256(content).hexdigest()
    name = "epyc-product-run-" + key[:16] + suite.suffix + "-" + req_hash[:12]
    output = name + ".json"
    transport.snapshot(pod, deployment, [output], local)
    if (local / output).is_file():
        answer = read(local / output)
        if answer["arm"] not in ("reference", "candidate") or len(answer["work"]["answers"]) != len(request["prompts"]):
            raise ValueError("incomplete delivered product batch")
        return {**deployed, "stage": "product served", "product_arm": answer["arm"], "response": str(local / output)}
    if deployed["stage"] == "withdrawn":
        return deployed
    cfg = release_framework(pod, cfg, lane, (*DEPLOY, *PRODUCT, *(MATCHED_RELEASE_FILES if suite.matched else ())))
    if suite.matched:
        common = [*common, "--profile", str(profile)]
    name = "epyc-product-qualify-" + key[:16] + suite.suffix
    command = [cfg["python"], "-m", "lab.rp.cpu_product_release"]
    opts = ["--state", str(remote), "--receipt-sha256", deployed["receipt_sha256"], "--timeout", "360", *common]
    lock = lane / "runs" / (key[:16] + ".lock")
    info = state_of_job(pod, name)
    if info["running"]:
        return {**deployed, "stage": "product qualification", "job": name}
    # qualify initially writes a negative "requalifying" checkpoint. Only collect terminal marker bytes.
    path = shlex.quote(str(deployment / "product-gate.json"))
    raw = transport.shell(pod, f"if test -f {path}; then cat -- {path}; fi")
    gate = json.loads(raw) if raw else None
    if not gate or gate.get("reason") == "requalifying":
        if info["rc"] is not None:
            return {**deployed, "stage": "product incomplete", "job": name, "diagnostic": info["tail"][-600:]}
        transport.job(pod, name, job_command(cfg, pod, [*command, "qualify", *opts], affinity, lock, 1000))
        return {**deployed, "stage": "product qualification", "job": name}
    transport.snapshot(pod, deployment, ["product-gate.json", *(f"product-pair-{i}.json" for i in range(PAIRS))], local)
    if gate.get("binding", {}).get("receipt_sha256") != deployed["receipt_sha256"]:
        raise ValueError("product qualification belongs to another release")
    if not gate["passed"]:
        return {**deployed, "stage": "product declined", "product_gate": gate}
    dest = lane / "inputs" / key / ("request-" + req_hash + ".json")
    put_once(pod, dest, content)
    name = "epyc-product-run-" + key[:16] + suite.suffix + "-" + req_hash[:12]
    output = name + ".json"
    transport.snapshot(pod, deployment, [output], local)
    if not (local / output).is_file():
        final = shlex.quote(str(deployment / output))
        cmd = shlex.join([*command, "run", *opts, "--requests", str(dest)]) + f" > {final}.tmp && mv -- {final}.tmp {final}"
        info = state_of_job(pod, name)
        if info["rc"] is not None:
            transport.snapshot(pod, deployment, [output], local)
            if not (local / output).is_file():
                return {**deployed, "stage": "product incomplete", "job": name, "diagnostic": info["tail"][-600:]}
        else:
            transport.job(pod, name, job_command(cfg, pod, ["/bin/bash", "-ec", cmd], affinity, lock, 720))
            return {**deployed, "stage": "product dispatch", "job": name}
    answer = read(local / output)
    if answer["arm"] not in ("reference", "candidate") or len(answer["work"]["answers"]) != len(request["prompts"]):
        raise ValueError("incomplete delivered product batch")
    return {**deployed, "stage": "product served", "product_arm": answer["arm"], "response": str(local / output)}


def replay(args):
    """Recompute the historical decision from local receipts, with no provider or author calls."""
    suite = Suite(args.matched)
    raw = args.state / "qualification"
    order = read(args.order or args.state / "order.json")
    scoped_release.validate_order(cpu_code_release.ADAPTER, order)
    if not suite.matched and (stopped := profile_conflict(raw, order)):
        return stopped
    authorship = args.authorship or args.state / "authoring"
    if not (authorship / "author/response.json").is_file():
        raise ValueError("replay needs the original correlated author response")
    reference, candidate = make_source(order, authorship, False)
    options = SimpleNamespace(state=raw, trial=raw, reference=reference, candidate=candidate,
                              order=args.order or args.state / "order.json", seed=suite.seed)
    observed = suite.broker.verify_offline(options)
    if observed != read(raw / "verdict.json"):
        raise ValueError("historical replay differs from its raw target verdict")
    return {"stage": "historically qualified" if observed["passed"] else "rejected", "result": observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    for key in ("target", "head", "order", "authorship", "evidence", "prior", "archive", "product-requests"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--job-name", help="adopt an existing qualification job for this order, including its incident suffix")
    parser.add_argument("--author-cli", action="store_true", help="consume the pending order through the included subscription")
    parser.add_argument("--replay", action="store_true", help="recompute the historical verdict from local raw receipts")
    parser.add_argument("--matched", action="store_true", help="use a separate v2 target path and a freshly captured complete profile")
    parser.add_argument("--prepare-only", action="store_true", help="stage immutable inputs and wait before any qualification or release")
    parser.add_argument("--publish", action="store_true", help="publish a qualified PR and activate only its independently verified merge")
    parser.add_argument("--stop-after-publication", action="store_true", help="reserve a separate quiet window for activation after CI")
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.wait_seconds < 0:
        parser.error("--wait-seconds must be non-negative")
    if args.stop_after_publication and not args.publish:
        parser.error("--stop-after-publication requires --publish")
    if args.replay:
        if args.publish or args.author_cli or args.prepare_only:
            parser.error("replay reads existing receipts only")
        print(json.dumps(replay(args), sort_keys=True))
        return
    if not (args.target and args.head):
        parser.error("a live campaign needs --target and --head")
    args.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    begin = time.monotonic()
    with (args.state / ".cpu-campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            result = advance(args)
            save(args.state / "campaign-outcome.json", result)
            print(json.dumps(result, sort_keys=True), flush=True)
            left = begin + args.wait_seconds - time.monotonic()
            if (left <= 0 or result["stage"] in {"deferred", "awaiting Codex", "declined", "incomplete", "confounded", "waiting for target",
                                                 "rejected", "qualified", "published", "active", "withdrawn", "product declined", "product served", "product incomplete"}):
                return
            time.sleep(min(15, left))


if __name__ == "__main__":
    main()
