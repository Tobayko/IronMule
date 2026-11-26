"""Run one bounded development experiment on an already available Linux machine."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from training import code_trial, experience_archive, experiment_planner, trial_release
from training.code_author import PendingAuthor
from training.code_trial import ROOT, git, read, save


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance():
    return {name: sha(ROOT / "training" / name) for name in (
        "serving_trial.py", "optimization_run.py", "experiment_planner.py", "experience_archive.py",
        "code_trial.py", "code_author.py", "trial_release.py", "pilot_threads.py")}


def discover(tree, run_dir, model, provider):
    """Promote the four observations and matching trial together after a complete discovery."""
    target = run_dir / "discovery"
    if target.is_dir() or (run_dir / "discovery-rejected/trial.json").is_file():
        return
    env = {**code_trial.runtime_env(provider), "PYTHONPATH": str(tree)}
    with tempfile.TemporaryDirectory(dir=run_dir) as temp:
        out = Path(temp) / "result"
        out.mkdir()
        argv = [sys.executable, str(ROOT / "training" / "serving_trial.py"), "--gguf", str(model),
                "--out", str(out / "trial.json"), "--profile-dir", str(out / "waves"),
                "--require-provider", provider, "--waves", "4", "--threads", "2", "--clients", "4", "--seed", "1241"]
        checkpoint = run_dir / "trial/controller.npz"
        if checkpoint.exists():
            argv.extend(["--thread-candidate", str(checkpoint)])
        with (run_dir / "discovery.log").open("ab") as log:
            try:
                subprocess.run(argv, cwd=tree, env=env, stdout=log, stderr=log, timeout=900, check=True)
            except subprocess.SubprocessError:
                if (out / "trial.json").is_file():
                    measured = read(out / "trial.json")
                    if (checkpoint.is_file() and measured.get("scratch_measured_decisions") == 0
                            and measured.get("correct") == measured.get("completed") == 16
                            and measured["binding"]["scratch_thread_head"]["weights_sha256"] == sha(checkpoint)):
                        os.replace(out, run_dir / "discovery-rejected")
                        return
                os.replace(out, run_dir / f"rejected-discovery-{time.time_ns()}")
                raise
        trial = read(out / "trial.json")
        if (len(list((out / "waves").glob("wave-*.json"))) != 4 or trial["completed"] != 16
                or trial["correct"] != 16 or trial["pilot_decisions"] <= 0):
            raise ValueError("incomplete Pilot discovery")
        os.replace(out, target)


def record(run_dir, context, order, outcome, elapsed):
    name = order["target"]["function"]
    source = f"{run_dir.name}/trial/result.json"
    notes = f"{name}: {outcome.get('reason')}. " + ("Hardware gate passed." if outcome["passed"] else "Rejected.")
    return experience_archive.record(
        "code_experiment", source, sha(run_dir / "trial" / "result.json"), order["order_id"], run_dir.name,
        title=name, note=notes, base_commit=context["base"], hardware=order["binding"],
        profiles=[order["target"]], model=context["gguf_sha256"], workload=read(run_dir / "discovery/trial.json")["workload"],
        decision=outcome["passed"], reason=outcome.get("reason"), metrics=[outcome], cost_usd=None,
        overhead_s=elapsed, source_order=order["order_id"], new_resource_bookings_usd=0,
        authorship="IronMule selected the experiment. Codex authored the proposed code.",
        uncertainty="Wall time covers the optimization job on existing hardware. See the provider ledger for charges.")


def paid_back_after(trial, elapsed_s):
    """Estimate the completed requests needed to recover measured experimentation wall time."""
    import math
    import statistics

    savings = []
    for path in sorted(trial.glob("pair-*.json")):
        pair = read(path)
        ref, cand = pair["reference"], pair["candidate"]
        if ref["completed"] != cand["completed"] or ref["completed"] <= 0:
            raise ValueError("unequal work in the completed trial")
        savings.append((ref["wall_s"] - cand["wall_s"]) / ref["completed"])
    if len(savings) != code_trial.PAIRS or min(savings) <= 0:
        return None
    return math.ceil(elapsed_s / statistics.median(savings))


def current_deployment(trial_dir):
    if not (trial_dir / "deployment/release.json").exists():
        return {"active": None, "withdrawn": False}
    with trial_release.deployment(trial_dir) as d:
        release = read(d / "release.json")
        if (d / "withdrawn.json").exists():
            return trial_release.restore(d, release, read(d / "withdrawn.json")["reason"])
        try:
            if (d / "active").readlink() != Path(release["candidate"]):
                raise ValueError("active checkout differs from the expected commit")
            trial_release.checkout(d, release["candidate"])
        except (OSError, ValueError, subprocess.SubprocessError):
            return trial_release.restore(d, release, "active selection or source changed")
        return {"active": release["candidate"], "withdrawn": False, "pr": release["pr"]}


def run(args):
    started = time.monotonic()
    state = args.state.resolve()
    model = args.gguf.resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("invalid run ID")
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (state / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir = state / "runs" / args.run_id
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = run_dir / "context.json"
        prior = read(path) if path.exists() else {}
        base = git("rev-parse", "--verify", (args.base or prior.get("base") or "HEAD") + "^{commit}")
        source = getattr(args, "thread_candidate", None)
        checkpoint = run_dir / "trial/controller.npz"
        expected = sha(source) if source else prior.get("thread_candidate_sha256")
        if expected and args.provider != "cpu":
            raise ValueError("scratch thread qualification is CPU-only")
        context = {"schema": 1, "base": base, "provider": args.provider, "thread_candidate_sha256": expected,
                   "gguf_sha256": sha(model), "implementation_sha256": provenance()}
        demand_path = getattr(args, "demand_summary", None)
        demand = None
        if demand_path:
            from ironmule_product.experiment_queue import validate
            demand = validate(read(demand_path))
            if demand["model_sha256"] != context["gguf_sha256"]:
                raise ValueError("queued demand belongs to another model")
            context["demand"] = demand
        if prior and context != prior:
            raise ValueError("different model, provider, source revision or experiment implementation on resume")
        if expected and not checkpoint.exists():
            if source is None:
                raise ValueError("missing snapshotted thread candidate; supply the original checkpoint")
            checkpoint.parent.mkdir(mode=0o700, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=checkpoint.parent) as tmp:
                copy = Path(tmp) / checkpoint.name
                shutil.copyfile(source, copy)
                if sha(copy) != expected:
                    raise ValueError("source checkpoint changed during snapshot")
                copy.chmod(0o400)
                os.replace(copy, checkpoint)
        if (sha(checkpoint) if checkpoint.exists() else None) != expected:
            raise ValueError("stored thread checkpoint differs from the expected model")
        save(path, context)
        done = run_dir / "complete.json"
        if done.exists():
            saved = read(done)
            if not args.publish or saved["publish_requested"] or not saved["historical_qualification"]["passed"]:
                return {**saved, "deployment": current_deployment(run_dir / "trial")}
        attempt = run_dir / f"attempt-{time.time_ns()}.json"
        log = {"started_unix_s": time.time(), "status": "running", "publish_requested": args.publish,
               "new_resource_bookings_usd": 0, "api_charges_usd": 0,
               "hardware_billing": "existing allocation; provider ledger tracks the lease"}
        save(attempt, log)
        stage = "discovery"
        try:
            tree = code_trial.prepare(run_dir, base, "baseline")
            if git("status", "--porcelain", cwd=tree):
                raise ValueError("frozen discovery checkout has changes")
            discover(tree, run_dir, model, args.provider)
            rejected = run_dir / "discovery-rejected/trial.json"
            if rejected.exists():
                measured = read(rejected)
                if (not expected or measured["binding"]["model_sha256"] != context["gguf_sha256"]
                        or measured["binding"]["scratch_thread_head"]["weights_sha256"] != expected
                        or measured["evaluator_sha256"] != context["implementation_sha256"]["serving_trial.py"]
                        or measured["commit"] != base or measured["scratch_measured_decisions"] != 0
                        or measured["completed"] != 16 or measured["correct"] != 16):
                    raise ValueError("rejected discovery differs from the pinned run")
                reason = "no qualified scratch thread action used in the measured waves"
                result = {"historical_qualification": {"passed": False, "stage": "controller discovery", "reason": reason},
                          "order_id": None, "base": base, "run_id": args.run_id, "publish_requested": args.publish,
                          "recorded_optimization_s": time.monotonic() - started, "new_resource_bookings_usd": 0}
                row = experience_archive.record(
                    "controller_discovery", f"{args.run_id}/discovery-rejected/trial.json", sha(rejected), "waves",
                    args.run_id, title="CPU thread qualification", reason=reason, decision=False,
                    hardware=measured["binding"], workload=measured["workload"], model=context["gguf_sha256"],
                    base_commit=base, metrics=[{k: measured[k] for k in (
                        "wall_s", "ttft_s", "correct", "completed", "scratch_measured_decisions", "total_elapsed_s")}],
                    authorship="IronMule measured and declined the scratch thread head.")
                save(run_dir / "experience.json", row)
                save(done, result)
                log["status"] = "finished"
                return {**result, "deployment": current_deployment(run_dir / "trial")}
            stage = "archive"
            archive = run_dir / "experience.jsonl"
            others = sorted(d for d in run_dir.parent.iterdir() if d != run_dir and d.is_dir())
            if not archive.exists():
                past = (read(p) for d in others for p in (
                    [d / "experience.json"] if (d / "experience.json").is_file() else sorted(d.glob("failure-*.json"))))
                experience_archive.write_archive(archive, itertools.chain(experience_archive.build(tree), past))
            stage = "plan"
            order_file = run_dir / "order.json"
            if not order_file.exists():
                profiles = [read(p) for p in (run_dir / "discovery/waves").glob("wave-*.json")]
                past_orders = [read(d / "order.json") for d in others if (d / "order.json").is_file()]
                order = experiment_planner.plan(profiles, past_orders, repetitions=code_trial.PAIRS, archive=archive,
                                                demand=demand)
                if order["base_commit"] != base or order["binding"]["model_sha256"] != context["gguf_sha256"]:
                    raise ValueError("proposal source or model differs from frozen discovery inputs")
                save(order_file, order)
            order = read(order_file)
            if provenance() != context["implementation_sha256"]:
                raise ValueError("experiment implementation changed after discovery")
            result_dir = run_dir / "trial"
            stage = "development and qualification"
            code_trial.cycle(SimpleNamespace(order=order_file, state=result_dir, gguf=model, provider=args.provider,
                                            author_only=False, publish=args.publish))
            outcome = read(result_dir / "result.json")
            elapsed = time.monotonic() - started
            prior_logs = [read(p) for p in run_dir.glob("attempt-*.json") if p != attempt]
            recorded_s = elapsed + sum(p.get("elapsed_s", 0) for p in prior_logs)
            save(run_dir / "experience.json", record(run_dir, context, order, outcome, recorded_s))
            result = {"historical_qualification": outcome, "order_id": order["order_id"], "base": base, "run_id": args.run_id,
                      "publish_requested": args.publish, "recorded_optimization_s": recorded_s,
                      "unfinished_prior_attempts": sum(p["status"] == "running" for p in prior_logs),
                      "new_resource_bookings_usd": 0}
            result["estimated_same_workload_requests_to_recover_measured_time"] = (
                paid_back_after(result_dir, recorded_s)
                if outcome["passed"] and not result["unfinished_prior_attempts"] else None)
            save(done, result)
            log["status"] = "finished"
            return {**result, "deployment": current_deployment(result_dir)}
        except PendingAuthor:
            log.update(status="awaiting_author", stage=stage)
            raise
        except BaseException as exc:
            log.update(status="interrupted", stage=stage, error_type=type(exc).__name__)
            error = f"{stage}: {type(exc).__name__}"
            if isinstance(exc, subprocess.CalledProcessError):
                error += f" (exit {exc.returncode})"
            log["error"] = error
            raise
        finally:
            log.update(finished_unix_s=time.time(), elapsed_s=time.monotonic() - started)
            save(attempt, log)
            if log["status"] == "interrupted":
                target = read(run_dir / "order.json")["target"] if (run_dir / "order.json").is_file() else {}
                row = experience_archive.record(
                    "code_experiment_failure", f"{run_dir.name}/{attempt.name}", sha(attempt), stage, run_dir.name,
                    base_commit=base, model=context["gguf_sha256"], reason=log["error"], title=target.get("function", stage),
                    note=f"Interrupted during {stage}. {log['error']}", decision=False)
                save(run_dir / attempt.name.replace("attempt-", "failure-"), row)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--run-id", default="first")
    parser.add_argument("--demand-summary", type=Path, help="aggregate serving workload ticket")
    parser.add_argument("--base", help="baseline Git revision, pinned in state on first invocation")
    parser.add_argument("--thread-candidate", type=Path, help="snapshot a four-slot CPU thread checkpoint")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--publish", action="store_true", help="publish a qualifying change through the PR gate")
    args = parser.parse_args()
    # The experiment and its child workers yield CPU time to higher-priority serving processes.
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    print(json.dumps(run(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
