"""Activate or withdraw a merged, measured code trial for its bound GGUF copy workload."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from training.code_trial import PAIRS, ROOT, TARGETS, git, judge, measure, read, run, save
from training.experiment_planner import EVALUATOR, ORDER, digest

REPOSITORY = "Tobayko/IronMule"
URL = f"https://github.com/{REPOSITORY}.git"
SOURCES = ("ggml_engine.py", "gguf_tokenizer.py", "pilot.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def deployment(state):
    d = state.resolve() / "deployment"
    d.mkdir(mode=0o700, exist_ok=True)
    with (d / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield d


@contextmanager
def switch_lock(d):
    # This lock protects only the small state update, never a measurement or a GitHub request.
    with (d / ".switch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def checkout(d, commit):
    """A checkout comes from an immutable tree, including when the caller resumes after a crash."""
    require(re.fullmatch(r"[a-f0-9]{40}", commit), "invalid Git commit")
    path = d / commit
    if not path.exists():
        with TemporaryDirectory(dir=d) as tmp:
            stage = Path(tmp) / "checkout"
            git("clone", "--quiet", "--no-local", "--no-checkout", str(d / "repo.git"), str(stage))
            git("checkout", "--quiet", "--detach", commit, cwd=stage)
            stage.rename(path)
    require(git("rev-parse", "HEAD", cwd=path) == commit
            and not git("status", "--porcelain", "--ignored", "--untracked-files=all", cwd=path),
            "release checkout has changed")
    return path


def select(d, commit):
    target = checkout(d, commit)
    with TemporaryDirectory(dir=d) as tmp:
        link = Path(tmp) / "active"
        link.symlink_to(os.path.relpath(target, d), target_is_directory=True)
        os.replace(link, d / "active")
        sync_dir(d)


def merged_source(d, pr, order):
    require(type(pr) is int and pr > 0, "missing pull request number")
    meta = json.loads(run(["gh", "api", f"repos/{REPOSITORY}/pulls/{pr}"]))
    branch = "qualification/" + order["order_id"][:16]
    require(meta["merged"] and meta["merged_at"] and meta["base"]["repo"]["full_name"] == REPOSITORY
            and meta["base"]["ref"] == branch, "expected a merged pull request on this order's qualification branch")
    repo = d / "repo.git"
    if not repo.exists():
        with TemporaryDirectory(dir=d) as tmp:
            stage = Path(tmp) / "repo.git"
            git("init", "--quiet", "--bare", str(stage))
            stage.rename(repo)
    # Fetch named refs, then verify immutable object IDs from GitHub. No branch or protection changes.
    git("fetch", "--quiet", "--no-tags", URL, f"+refs/heads/{branch}:refs/heads/qualified",
        f"+refs/pull/{pr}/head:refs/heads/proposal", cwd=repo)
    merge = git("rev-parse", "refs/heads/qualified", cwd=repo)
    head = git("rev-parse", "refs/heads/proposal", cwd=repo)
    base = order["base_commit"]
    require(merge == meta["merge_commit_sha"] and head == meta["head"]["sha"]
            and git("rev-parse", f"{merge}^1", cwd=repo) == base, "PR moved or qualification base changed")
    candidate_tree = git("rev-parse", f"{merge}^{{tree}}", cwd=repo)
    require(candidate_tree == git("rev-parse", f"{head}^{{tree}}", cwd=repo),
            "merged source differs from the reviewed proposal")
    target = order["target"]["path"]
    require(target in TARGETS and git("diff", "--name-only", base, merge, cwd=repo) == target,
            "changes outside the measured source file")
    return {"base": base, "candidate": merge, "pr": pr, "branch": branch, "order_id": order["order_id"],
            "reference_tree": git("rev-parse", f"{base}^{{tree}}", cwd=repo), "candidate_tree": candidate_tree}


def source_hashes(repo, commit):
    return {name: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{commit}:ironmule/{name}"], cwd=repo)).hexdigest() for name in SOURCES}


def wave_shape(i, wave):
    # Cache residency, wave position and each lane's actual work must match before timing is comparable.
    return digest([i, wave["workload"], wave["requests"], wave["reused_prompt_tokens"]])


def validate_arm(row, release, seed, arm):
    w = row["workload"]
    require(row["binding"] == release["binding"] and row["schema"] == EVALUATOR and row["profiled"] is False
            and row["evaluator_sha256"] == release["evaluator_sha256"]
            and row["runtime_source_sha256"] == release[arm + "_hashes"]
            and row["pilot_decisions"] > 0, "measurement source, evaluator, Pilot or hardware differs")
    require(w == {**release["workload"], "seed": seed}, "workload or paired seed differs")
    require(row["completed"] == w["clients"] * w["waves"] and len(row["waves"]) == w["waves"]
            and row["correct"] == row["completed"], "incomplete measured work")


def evidence(state, d, order, publication):
    context = read(state / "context.json")
    require(context["base"] == publication["base"] and read(state / "result.json").get("passed") is True,
            "no successful qualification for the recorded reference")
    release = {**publication, "reference_hashes": source_hashes(d / "repo.git", publication["base"]),
               "candidate_hashes": source_hashes(d / "repo.git", publication["candidate"]),
               "evaluator_sha256": hashlib.sha256((ROOT / "training/serving_trial.py").read_bytes()).hexdigest()}
    signature = {k: release[k] for k in ("reference_tree", "candidate_tree", "order_id", "evaluator_sha256")}
    signature["gate_sha256"] = hashlib.sha256((ROOT / "training/code_trial.py").read_bytes()).hexdigest()
    require(signature == read(state / "evaluation.json"), "qualification source, gate, evaluator or order changed")
    pairs = []
    for i in range(PAIRS):
        pair = read(state / f"pair-{i}.json")
        require(all(pair.get(k) == v for k, v in signature.items()),
                "paired data is from another source, gate, evaluator or order")
        ref, cand = pair["reference"], pair["candidate"]
        if i == 0:
            release["binding"] = ref["binding"]
            release["workload"] = {k: v for k, v in ref["workload"].items() if k != "seed"}
            require(release["binding"]["provider"] == context["provider"]
                    and release["binding"]["provider"] in ("cpu", "cuda")
                    and release["workload"] == {"clients": 4, "waves": 4, "max_tokens": 3, "warmup_waves": 1,
                                                "context": 512, "phases": ["short", "document"]},
                    "unexpected deployment workload")
            for key in ("provider", "model_sha256", "machine", "native", "cores", "pilot_weights_sha256"):
                require(order["binding"][key] == release["binding"][key], "hardware differs from development order")
        validate_arm(ref, release, 8351 + i, "reference")
        validate_arm(cand, release, 8351 + i, "candidate")
        pairs.append((ref, cand))
    require(judge(pairs)["passed"], "stored pairs fail the current qualification gate")
    cells = {}
    for ref, _ in pairs:
        for i, wave in enumerate(ref["waves"]):
            group = cells.setdefault(wave_shape(i, wave), [])
            group.append((wave["wall_s"], max(wave["ttfts_s"])))
    # Each shape contains just one observation per independent pair seed.
    release["reference_envelopes"] = {k: [max(t for t, _ in v), max(t for _, t in v)]
                                      for k, v in cells.items() if len(v) >= 2}
    return release


def restore_locked(d, release, reason):
    # Persist withdrawal before pointing back. A restart completes either interrupted operation.
    save(d / "withdrawn.json", {"reason": reason, "order_id": release["order_id"]})
    sync_dir(d)
    select(d, release["base"])
    return {"active": release["base"], "reason": reason, "withdrawn": True}


def restore(d, release, reason):
    with switch_lock(d):
        return restore_locked(d, release, reason)


def observe_live(d, release, row):
    """Three consecutive matched regressions withdraw locally. The completed answer is already available."""
    phases = {}
    for i, wave in enumerate(row["waves"]):
        limits = release["reference_envelopes"].get(wave_shape(i, wave))
        if limits is not None:
            slower = wave["wall_s"] > limits[0] or max(wave["ttfts_s"]) > limits[1]
            phases.setdefault(wave["workload"], []).append(slower)
    # Require both shapes to miss throughout their comparable cells; unknown work cancels the run of misses.
    missed = set(phases) == set(release["workload"]["phases"]) and all(all(values) for values in phases.values())
    with switch_lock(d):
        if (d / "withdrawn.json").exists():
            return
        path = d / "live-watch.json"
        prior = read(path) if path.exists() else {}
        count = prior.get("consecutive_misses", 0) if prior.get("order_id") == release["order_id"] else 0
        count = count + 1 if missed else 0
        save(path, {"order_id": release["order_id"], "consecutive_misses": count,
                    "compared_phases": phases, "last_seed": row["workload"]["seed"]})
        if count >= 3:
            restore_locked(d, release, "three consecutive latency regressions on matched short and document waves")


def activate(state: Path, order: dict, pr: int, gguf: Path):
    require(order["schema"] == ORDER and order["evaluator"] == EVALUATOR and
            digest({k: v for k, v in order.items() if k != "order_id"}) == order["order_id"], "invalid order")
    with deployment(state) as d:
        publication = merged_source(d, pr, order)
        if (d / "release.json").exists():
            previous = read(d / "release.json")
            require(all(previous[k] == v for k, v in publication.items()), "campaign already has another deployment")
            if (d / "withdrawn.json").exists():
                return restore(d, previous, read(d / "withdrawn.json")["reason"])
        release = evidence(state, d, order, publication)
        checkout(d, release["candidate"])
        reference = checkout(d, release["base"])
        save(d / "release.json", release)
        sync_dir(d)
        # Verify the deployment target using actual native work before switching this machine.
        if not (d / "active").exists():
            select(d, release["base"])
        try:
            args = SimpleNamespace(gguf=gguf, provider=release["binding"]["provider"])
            validate_arm(measure(reference, args, 9351, state), release, 9351, "reference")
        except Exception as exc:
            return restore(d, release, f"target verification failed: {exc}")
        with switch_lock(d):
            if (d / "withdrawn.json").exists():
                return restore_locked(d, release, read(d / "withdrawn.json")["reason"])
            select(d, release["candidate"])
        return {"active": release["candidate"], "withdrawn": False, "pr": pr}


def check(state: Path, gguf: Path):
    """Repeat the paired gate on the local deployment with different seeds. Withdraw on any failure."""
    with deployment(state) as d:
        release = read(d / "release.json")
        if (d / "withdrawn.json").exists():
            return restore(d, release, read(d / "withdrawn.json")["reason"])
        try:
            require((d / "active").readlink() == Path(release["candidate"]), "activation did not complete")
            args = SimpleNamespace(gguf=gguf, provider=release["binding"]["provider"])
            ref = checkout(d, release["base"])
            cand = checkout(d, release["candidate"])
            pairs = []
            for i in range(PAIRS):
                seed = 9352 + i
                order = [("reference", ref), ("candidate", cand)]
                pair = {}
                for arm, tree in reversed(order) if i % 2 else order:
                    pair[arm] = measure(tree, args, seed, state)
                    validate_arm(pair[arm], release, seed, arm)
                pairs.append((pair["reference"], pair["candidate"]))
            result = judge(pairs)
            save(d / "recheck.json", {"pairs": pairs, "result": result})
            require(result["passed"], result["reason"])
        except Exception as exc:
            return restore(d, release, f"recheck failed: {exc}")
        with switch_lock(d):
            if (d / "withdrawn.json").exists():
                return restore_locked(d, release, read(d / "withdrawn.json")["reason"])
            save(d / "live-watch.json", {"order_id": release["order_id"], "consecutive_misses": 0})
        return {"active": release["candidate"], "withdrawn": False, "result": result}


def run_active(state: Path, gguf: Path, seed: int):
    """Exercise the deployed checkout on its qualified fixture family without waiting for a recheck."""
    d = state.resolve() / "deployment"
    release = read(d / "release.json")
    args = SimpleNamespace(gguf=gguf, provider=release["binding"]["provider"])
    try:
        active = (d / "active").readlink()
    except FileNotFoundError:
        active = Path(release["base"])
    candidate = active == Path(release["candidate"]) and not (d / "withdrawn.json").exists()
    if candidate:
        try:
            selected = checkout(d, release["candidate"])
            result = measure(selected, args, seed, state)
            validate_arm(result, release, seed, "candidate")
            observe_live(d, release, result)
            return result
        except Exception as exc:
            restore(d, release, f"active workload failed: {exc}")
    reference = checkout(d, release["base"])
    result = measure(reference, args, seed, state)
    validate_arm(result, release, seed, "reference")
    return result


def status(state: Path):
    """Read the recorded trial verdict and verify the selected checkout without running a model."""
    from training.optimization_run import current_deployment

    selected = current_deployment(state)
    manifest = state / "deployment/release.json"
    release = read(manifest) if manifest.is_file() else {}
    result = state / "result.json"
    history = read(result) if result.is_file() else {}
    pair = state / "pair-0.json"
    observation = read(pair)["reference"] if history and not release and pair.is_file() else {}
    target = release or observation
    return {"historical_gate": history.get("passed"), "gate_reason": history.get("reason"), "checkout": selected,
            "recorded_target": target.get("binding"), "recorded_workload": target.get("workload"),
            "hardware_tested_by_status": False}


def rollback(state: Path):
    with deployment(state) as d:
        return restore(d, read(d / "release.json"), "explicit withdrawal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("activate", "check", "run", "rollback", "status"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--state", type=Path, required=True)
        if name not in ("rollback", "status"):
            cmd.add_argument("--gguf", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--seed", type=int, default=9351)
            cmd.add_argument("--output", type=Path, required=True)
        if name == "activate":
            cmd.add_argument("--order", type=Path, required=True)
            cmd.add_argument("--pr", type=int, required=True)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(args.state), sort_keys=True))
        return
    if args.command == "activate":
        result = activate(args.state, read(args.order), args.pr, args.gguf)
    elif args.command == "check":
        result = check(args.state, args.gguf)
    elif args.command == "run":
        result = run_active(args.state, args.gguf, args.seed)
        save(args.output, result)
        print(json.dumps({k: result[k] for k in ("completed", "correct", "tokens", "wall_s")}))
        return
    else:
        result = rollback(args.state)
    print(json.dumps(result, sort_keys=True))
    if args.command != "rollback" and result["withdrawn"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
