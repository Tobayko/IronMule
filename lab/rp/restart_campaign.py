"""Run one source-bound repeat-start experiment on the already reserved A100."""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from lab.pk1 import restart_order, restart_release, restart_trial
from lab.pk1.native_trial import checksum
from lab.rp.native_campaign import (
    CONFIG_KEYS,
    call,
    command,
    endpoint,
    job,
    lease,
    shell,
    snapshot,
    upload,
)
from training import scoped_release
from training.code_author import PendingAuthor, write_once
from training.code_trial import PAIRS, read, save

FILES = ["context.json", "aa-0.json", "aa-1.json", "receipt.json", "stale-receipt.json", "stale.json", "missing.json",
         *(f"{tag}-{i}.json" for i in range(PAIRS) for tag in ("ref", "cand")),
         *(f"facts-{tag}-{i}.json" for i in range(2) for tag in ("ref", "cand")), "result.json"]
FRAMEWORK = ["lab/pk1/restart_trial.py", "lab/pk1/restart_arm.py", "lab/pk1/restart_order.py", "lab/pk1/native_trial.py"]


def release(args, cfg, remote, check):
    local = args.state / "release"
    publication = local / "published.json"
    if not publication.exists():
        result = restart_release.control(SimpleNamespace(**vars(check), state=local, publish=True))
        write_once(publication, result)
    result = read(publication)
    order = read(args.order)
    receipt = scoped_release.pinned(local, result["receipt_sha256"])
    if (receipt["order"] != order or receipt["publication"]["pr"] != result["pr"]
            or receipt["qualification"] != restart_release.approved(check)):
        raise ValueError("cached publication differs from verified target evidence")
    pod = lease(cfg["pod_id"])
    key = order["order_id"]
    release_files = ["lab/pk1/restart_release.py", "training/scoped_release.py", "training/trial_release.py",
                     "training/code_trial.py", "training/code_author.py", "training/experiment_planner.py",
                     "lab/rp/restricted_worker.py", *FRAMEWORK]
    root = Path(__file__).resolve().parents[2]
    paths = [PurePosixPath(cfg["framework"]) / name for name in release_files]
    rows = shell(pod, shlex.join(["sha256sum", "--", *map(str, paths)])).splitlines()
    if (len(rows) != len(paths) or any(row.split(maxsplit=1) != [checksum(root / name), str(path)]
            for row, path, name in zip(rows, paths, release_files, strict=True))):
        raise ValueError("different target release adapter")
    pin = result["receipt_sha256"]
    upload(pod, remote / "merged.json", Path(result["receipt"]).read_bytes())
    common = ["--state", remote / "active", "--trial", remote / "trial", "--order", remote / "order.json",
              "--candidate-source", remote / "proposal.py", "--checkout", cfg["checkout"], "--worker-root", cfg["worker_root"],
              "--gguf", cfg["gguf"], "--reference-lib", cfg["reference_lib"], "--table", cfg["table"],
              "--receipt-sha256", pin]
    for op, extra in (("activate", ["--receipt", remote / "merged.json"]), ("run", ["--seed", "93742"]),
                      ("exercise", ["--seed", "93743"])):
        name = "release-" + op + ".json"
        path = local / name
        snapshot(pod, remote, [name], local)
        if not path.exists():
            cmd = command(cfg, "lab.pk1.restart_release", [op, *common, *extra])
            out = shlex.quote(str(remote / name))
            # The target can complete with an ordinary, bound withdrawal even at nonzero exit.
            cmd = "(" + cmd + ") > " + out + ".tmp; rc=$?; if test -s " + out + ".tmp; then mv " + out + ".tmp " + out + '; fi; exit "$rc"'
            try:
                status = job(pod, "restart-release-" + op + "-" + key[:16], cmd)
            except RuntimeError:
                snapshot(pod, remote, [name], local)
                if not path.exists() or op != "activate" or read(path).get("withdrawn") is not True:
                    raise
                status = "finished"
            if status != "finished":
                return {"stage": op, "order": key, "pr": result["pr"]}
            snapshot(pod, remote, [name], local)
        observed = read(path)
        if op in ("activate", "run") and observed.get("arm") != "candidate":
            return {"stage": "withdrawn", "order": key, "pr": result["pr"], "result": observed}
        if op == "exercise" and not (observed.get("passed") is True and observed["receipt_sha256"] == pin):
            raise ValueError("rollback evidence missing")
    deployed = remote / "active/deployment"
    snapshot(pod, deployed, ["target-check.json", "rollback-exercise.json"], local)
    activation, work, proof = (read(local / f"release-{op}.json") for op in ("activate", "run", "exercise"))
    check_sha = checksum(local / "target-check.json")
    if (activation.get("target_check_sha256") != check_sha or activation.get("receipt_sha256") != pin
            or activation.get("pr") != result["pr"] or proof != read(local / "rollback-exercise.json")
            or proof["before"]["work_sha256"] != proof["after"]["work_sha256"]):
        raise ValueError("release stages differ from their target receipts")
    for row, seed, arm in ((work, 93742, "candidate"), (proof["before"], 93743, "candidate"), (proof["after"], 93743, "reference")):
        if (row["arm"] != arm or row["seed"] != seed or row["cache_used"] is not (arm == "candidate")
                or row["source_sha256"] != receipt["qualification"][arm + "_sha256"] or row["graded_correct"] != 18
                or row["cold_performance_completions"] != 32 or row["warm_performance_completions"] != 32
                or row["pilot_decisions"] <= 0):
            raise ValueError("unbound actual startup workload")
    selected = json.loads(call(pod, "GET", endpoint("get", deployed / "selection.json"), raw=True))
    if selected != activation or selected["withdrawn"]:
        return {"stage": "withdrawn", "order": key, "pr": result["pr"], "selection": selected}
    return {"stage": "active", "order": key, "pr": result["pr"], "receipt_sha256": pin,
            "work": work, "rollback_test": proof["passed"]}


def tokenizer_retry(state, check, receipt, path):
    """Bind a separate product attempt to the failed reference and an actual successful unmodified GATE_IDS."""
    import ast
    import hashlib
    import re

    prior = read(state / "product/gate-reference.json")
    diagnosis = prior.get("diagnostic", {})
    source = check.checkout / "ironmule/ggml_engine.py"
    tree = ast.parse(source.read_text())
    gate = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "GATE_IDS" for t in node.targets))
    record = read(path)
    if (prior.get("passes") is not False or diagnosis.get("stage") != "tokenize" or diagnosis.get("code") != "missing_dependency"
            or prior.get("source_sha256") != checksum(source)
            or record.get("schema") != "ironmule.product-tokenizer-preflight.v1" or record.get("order_id") != receipt["order"]["order_id"]
            or record.get("source_sha256") != checksum(source) or record.get("gguf_sha256") != receipt["qualification"]["binding"]["runtime"]["model_sha256"]
            or record.get("gate_ids_sha256") != hashlib.sha256(gate.encode()).hexdigest() or record.get("returncode") != 0
            or record.get("ids_count") != 8193
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("facts_sha256"))) or not record.get("diagnosed_dependency")
            or not record.get("correction")):
        raise ValueError("need the recorded failed reference and an actual full corrected tokenizer preflight")
    return record


def product(args, cfg, remote, check, active):
    """Use the existing lease and ordinary HTTP broker only after the independent native release is active."""
    from lab.pk1 import restart_http
    from training import pilot_restart_profiles

    state = args.state
    recovery = getattr(args, "product_preflight", None)
    local = state / ("product-v2" if recovery else "product")
    output = remote / "product-v2" if recovery else remote
    local.mkdir(exist_ok=True)
    control = state / "release"
    pod = lease(cfg["pod_id"])
    pin = active["receipt_sha256"]
    receipt = scoped_release.pinned(control, pin)
    fit = pilot_restart_profiles.fit(SimpleNamespace(**vars(check), weights=state / "startup-head.npz",
                                                      report=state / "startup-fit.json", seed=18471))
    head_sha = fit["weights_sha256"]
    require_hash = ["lab/pk1/restart_http.py", "lab/pk1/restart_http_worker.py", "lab/pk1/native_http.py",
                    "lab/pk1/native_http_worker.py", "training/pilot_restart_profiles.py", "ironmule/pilot.py",
                    "ironmule/ggml_profiles.py", "ironmule_product/backend.py", "ironmule_product/http_server.py"]
    root = Path(__file__).resolve().parents[2]
    paths = [PurePosixPath(cfg["framework"]) / k for k in require_hash]
    hashes = shell(pod, shlex.join(["sha256sum", "--", *map(str, paths)])).splitlines()
    if hashes != [f"{checksum(root / n)}  {p}" for n, p in zip(require_hash, paths, strict=True)]:
        raise ValueError("different CPU head, product broker or release helper on target")
    geometry_record = restart_http.measured_geometry(check.trial)
    if recovery:
        proof = tokenizer_retry(state, check, receipt, recovery)
        write_once(local / "tokenizer-preflight.json", proof)
        geometry_record["runtime_preflight_sha256"] = checksum(local / "tokenizer-preflight.json")
        shell(pod, shlex.join(["mkdir", "-p", str(output)]))
    upload(pod, output / "startup-head.npz", (state / "startup-head.npz").read_bytes())
    geometry = local / "geometry.json"
    write_once(geometry, geometry_record)
    geometry_sha = checksum(geometry)
    upload(pod, output / "geometry.json", geometry.read_bytes())
    bound = restart_http.key_for(SimpleNamespace(receipt_sha256=pin, head_sha256=head_sha, geometry_sha256=geometry_sha), receipt)
    evidence = remote / "active/deployment" / ("restart-http-" + bound)
    files = ["gate-candidate.json", "gate-reference.json", "decision.json",
             *(f"{arm}-{seed}.json" for seed in restart_http.SEEDS for arm in ("reference", "candidate"))]
    snapshot(pod, evidence, files, local)
    for op in ("qualify", "exercise", "serve"):
        name = ("product-listener" if op == "serve" else "product-" + op) + ".json"
        snapshot(pod, output, [name], local)
        if op == "serve":
            from lab.pk1 import restart_live
            helper = PurePosixPath(cfg["framework"]) / "lab/pk1/restart_live.py"
            if shell(pod, shlex.join(["sha256sum", str(helper)])).split()[0] != checksum(restart_live.__file__):
                raise ValueError("unbound passive listener or external client")
            if args.verify_product_client:
                snapshot(pod, evidence, ["live-ready.json"], local)
                snapshot(pod, output, ["product-client.json"], local)
                if (local / "live-ready.json").exists() and not (local / "product-client.json").exists():
                    # The passive server owns the GPU measurement lock. This separate client only submits HTTP requests.
                    commandline = shlex.join([cfg["python"], "-m", "lab.pk1.restart_live", "--ready", str(evidence / "live-ready.json"),
                                              "--out", str(output / "product-client.json"), "--receipt-sha256", pin,
                                              "--timeout", str(min(args.product_timeout, 240))])
                    destdir = shlex.quote(cfg["framework"])
                    job(pod, "external-product-" + bound[:16], f"cd {destdir} && PYTHONPATH={destdir} {commandline}")
        if not (local / name).exists():
            remaining = int(pod["deadline"] - time.time()) - 10
            timeout = min(args.product_timeout, remaining)
            if timeout < 60 or (op == "serve" and timeout < args.product_lifetime + 60):
                return {"stage": "product lease limit", "order": active["order"], "pr": active["pr"], "pending": op}
            common = ["--state", remote / "active", "--trial", remote / "trial", "--candidate-source", remote / "proposal.py",
                      "--checkout", cfg["checkout"], "--worker-root", cfg["worker_root"], "--gguf", cfg["gguf"],
                      "--reference-lib", cfg["reference_lib"], "--table", cfg["table"], "--receipt-sha256", pin,
                      "--head", output / "startup-head.npz", "--head-sha256", head_sha, "--port", args.product_port,
                      "--geometry", output / "geometry.json", "--geometry-sha256", geometry_sha,
                      "--timeout", timeout, "--lifetime", args.product_lifetime]
            route = ["--reference-url", args.product_reference_url] if args.product_reference_url else ["--dedicated"]
            cmd = command(cfg, "lab.pk1.restart_http", [op, *common, *route])
            dest = shlex.quote(str(output / name))
            job(pod, "product-restart-" + op + "-" + (bound if recovery else active["order"])[:16], cmd + " > " + dest + ".tmp && mv " + dest + ".tmp " + dest)
            return {"stage": "product " + op, "order": active["order"], "pr": active["pr"]}
        value = read(local / name)
        if op == "qualify":
            if value.get("passed") is not True:
                from training.restart_replan import declined
                note = declined(state, checkout=args.checkout, product_dir=local)
                return {**active, "stage": "product declined", "product_result": value, "next_experiment": note["next"]}
            if restart_http.assess(local, receipt) != value:
                raise ValueError("product verdict differs from exported HTTP holdouts")
        elif op == "exercise":
            if (value.get("passed") is not True or value["before"]["arm"] != "candidate" or value["after"]["arm"] != "reference"
                    or value["before"]["work_sha256"] != value["after"]["work_sha256"]):
                return {**active, "stage": "product withdrawn", "product_result": value}
        else:
            if not args.verify_product_client:
                return {**active, "stage": "product service completed", "product_result": value}
            if not (local / "product-client.json").exists():
                return {**active, "stage": "product client missing", "product_result": value}
            value = restart_live.verify(value, read(local / "product-client.json"))
            write_once(local / "product-serve.json", value)
            if value["arm"] != "candidate" or value["completions"][-1]["graded_correct"] != 18:
                return {**active, "stage": "product retained reference", "product_result": value}
    from training.restart_replan import replan

    next_run = replan(state, checkout=args.checkout, product_dir=local)
    return {**active, "stage": "product verified", "product_result": value, "head": head_sha, "http_binding": bound,
            "next_experiment": next_run["next"], "replan_sha256": checksum((local if recovery else state) / "replan-after-product.json")}


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
    if getattr(args, "publish", False):
        active = release(args, cfg, remote, check)
        if getattr(args, "exercise_pilot", False) and active["stage"] == "active":
            from lab.rp.restart_pilot_campaign import exercise
            from training.pilot_restart_profiles import fit
            weights, report = state / "startup-head.npz", state / "startup-fit.json"
            if not (weights.is_file() and report.is_file()):
                fit(SimpleNamespace(**vars(check), weights=weights, report=report, seed=18471))
            active = exercise(args, cfg, remote, active)
        return product(args, cfg, remote, check, active) if getattr(args, "serve_product", False) and active["stage"] == "active" else active
    return {"stage": "qualified", "order": key, "metrics": verified["metrics"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "order", "checkout", "state"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--stage-only", action="store_true")
    p.add_argument("--exercise-pilot", action="store_true", help="serve the same held-out native work under random and trained CPU weights")
    p.add_argument("--publish", action="store_true", help="publish a qualifying source and verify activation and rollback on its target")
    p.add_argument("--wait-seconds", type=int, default=0)
    p.add_argument("--serve-product", action="store_true", help="gate the ordinary HTTP startup, then serve it with the trained CPU head")
    p.add_argument("--verify-product-client", action="store_true", help="explicitly submit and score a separate client batch on the live port")
    p.add_argument("--product-preflight", type=Path, help="separate v2 attempt after an actual corrected tokenizer preflight")
    p.add_argument("--product-reference-url", help="existing ordinary local product URL for other traffic")
    p.add_argument("--product-dedicated", action="store_true", help="run a separate scoped product lane")
    p.add_argument("--product-port", type=int, default=8232)
    p.add_argument("--product-timeout", type=int, default=1800)
    p.add_argument("--product-lifetime", type=int, default=120)
    a = p.parse_args()
    if a.serve_product and (not a.publish or bool(a.product_reference_url) == a.product_dedicated
            or not 1 <= a.product_port <= 65535 or not 1 <= a.product_timeout <= 3600 or not 1 <= a.product_lifetime <= 1800):
        p.error("product use needs publication, one route, valid port and bounded lifetime")
    if a.verify_product_client and not a.serve_product:
        p.error("the external product client needs --serve-product")
    if a.exercise_pilot and not a.publish:
        p.error("the startup policy operates on a merged and activated release; pass --publish")
    if not 0 <= a.wait_seconds <= 7200:
        p.error("invalid polling window")
    until = time.monotonic() + a.wait_seconds
    while True:
        outcome = cycle(a)
        print(json.dumps(outcome), flush=True)
        if outcome["stage"] not in ("measuring", "activate", "run", "exercise", "pilot serving", "product qualify", "product exercise", "product serve") or time.monotonic() >= until:
            return
        time.sleep(min(10, max(0, until - time.monotonic())))


if __name__ == "__main__":
    main()
