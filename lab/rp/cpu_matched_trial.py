"""Compare a CPU source proposal on one measured local startup profile and six counterbalanced workloads."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from lab.pk1.native_trial import copy_result
from lab.rp import cpu_code_trial as first
from lab.rp import restricted_worker
from training.code_author import digest, write_once
from training.code_trial import PAIRS, judge, read, validate_arm

HARNESS = Path(__file__).with_name("cpu_matched_profile.py")
SEED = 18351
SCHEMA = "ironmule.epyc-matched-code-trial.v2"


def identity(args):
    order, evaluator, helper = first.inputs(args, on_target=False)
    return {"schema": SCHEMA, "order_id": order["order_id"], "reference_sha256": first.sha(args.reference / first.SOURCE),
            "candidate_sha256": first.sha(args.candidate / first.SOURCE),
            "evaluator_sha256": digest([first.scoring_hash(evaluator, helper), first.sha(HARNESS), first.sha(__file__)])}


def worker(args, tree, mode, dest, seed, profile):
    order, evaluator, helper = first.inputs(args)
    tree = tree.resolve(strict=True)
    if first.sha(helper) != order["binding"]["scratch_thread_head"]["helper_sha256"]:
        raise ValueError("changed qualifier")
    env = {k: os.environ[k] for k in ("IRONMULE_GGML_BINDINGS", "LD_LIBRARY_PATH", "LLAMA_CPP_LIB_PATH")}
    env.update(IRONMULE_GGML_GPU_LAYERS="0", IRONMULE_PILOT="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(tree))
    readonly = [tree, args.gguf, args.head, evaluator.parent, args.python.parent.parent, Path("/etc/machine-id"),
                Path(env["IRONMULE_GGML_BINDINGS"]), Path(env["LLAMA_CPP_LIB_PATH"]),
                *(Path(p) for p in env["LD_LIBRARY_PATH"].split(":") if p and not Path(p).resolve().is_relative_to("/usr"))]
    # Stage just the public protocol inputs. The worker can't read the broker's campaign state.
    with TemporaryDirectory(dir=args.worker_root) as tmp:
        stage = Path(tmp)
        stage.chmod(0o711)
        trusted = stage / "trusted"
        trusted.mkdir(mode=0o755)
        scripts = [HARNESS, first.RULE, args.order]
        if profile is not None:
            scripts.append(profile)
        for path in scripts:
            staged = trusted / path.name
            staged.write_bytes(path.read_bytes())
            staged.chmod(0o444)
        wanted = "profile.json" if mode == "calibrate" else "arm.json"
        out = stage / "out"
        out.mkdir(mode=0o700)
        os.chown(out, 60000, 60000)
        cmd = ["/usr/bin/env", "-u", "CUDA_CACHE_PATH", str(args.python), str(trusted / HARNESS.name),
               "calibrate" if mode == "calibrate" else "arm", "--order", str(trusted / args.order.name),
               "--gguf", str(args.gguf), "--head", str(args.head), "--out", str(out / wanted)]
        if profile is not None:
            cmd += ["--profile", str(trusted / profile.name), "--seed", str(seed), "--mode", mode]
        proc = restricted_worker.run(cmd, env=env, cwd=tree, output=out, readonly=[*readonly, trusted],
                                     timeout=args.timeout)
        dest.with_suffix(".log").write_text(proc.stdout)
        if proc.returncode:
            raise RuntimeError(f"isolated {mode} exited {proc.returncode}; see {dest.with_suffix('.log')}")
        copy_result(out / wanted, dest, 60000)


def check_arm(args, tree, row, profile, mode, seed):
    from lab.rp import cpu_matched_profile as bound

    order, evaluator, _ = first.inputs(args, on_target=False)
    raw = read(profile)
    bound.verify_arm(row, raw, first.sha(profile), mode, order)
    expected = {**order, "binding": raw["binding"]}
    validate_arm(row, tree, expected, seed, {"evaluator_sha256": first.sha(evaluator)})
    if mode == "rule":
        fixed = row.get("simple_thread_rule", {})
        if (fixed.get("source_sha256") != first.sha(first.RULE) or fixed.get("actual_threads") != [4]
                or fixed.get("overrides", 0) <= 0):
            raise ValueError("missing or foreign constant four-thread control")
    if row.get("commit") != order["base_commit"]:
        raise ValueError("worker reference revision differs")


def profile_source(args):
    from lab.rp import cpu_matched_profile as bound

    path = args.state / "profile.json"
    record = read(path)
    order, _, _ = first.inputs(args, on_target=False)
    bound.verify_profile(record, order)
    if record["runtime_source_sha256"]["ggml_engine.py"] != first.sha(args.reference / first.SOURCE):
        raise ValueError("profile was measured on a different reference source")
    return path


def verify_offline(args):
    sig = identity(args)
    if read(args.state / "context.json") != sig or sig["reference_sha256"] == sig["candidate_sha256"]:
        raise ValueError("mismatched source, author order or evaluator")
    built = read(args.state / "build.json")
    if (built.get("passed") is not True or built["source_sha256"] != sig["candidate_sha256"]
            or built["log_sha256"] != first.sha(args.state / "build-1.log")):
        raise ValueError("missing candidate build or validation")
    profile = profile_source(args)
    pairs, rule_pairs = [], []
    for i in range(PAIRS):
        seed = SEED + i
        arms = {}
        for mode in ("reference", "candidate", "rule"):
            tree = args.candidate if mode == "candidate" else args.reference
            row = read(args.state / f"{mode}-{seed}.json")
            check_arm(args, tree, row, profile, mode, seed)
            arms[mode] = row
        ref, cand, rule = (arms[m] for m in ("reference", "candidate", "rule"))
        for iref, others in zip(ref["waves"], zip(cand["waves"], rule["waves"], strict=True), strict=True):
            if any((iref["requests"], iref["reused_prompt_tokens"]) != (r["requests"], r["reused_prompt_tokens"])
                   for r in others):
                raise ValueError("different token work or KV cache state")
        pairs.append((ref, cand))
        rule_pairs.append((cand, rule))
    incumbent, challenger = judge(pairs), judge(rule_pairs)
    if incumbent["reason"] != "paired hardware gate" or challenger["reason"] != "paired hardware gate":
        raise ValueError("incomplete, incorrect or unpaired work")
    passed = incumbent["passed"] and not challenger["passed"]
    metrics = {**incumbent, "passed": passed, "fixed_four_threads_vs_candidate": challenger,
               "correct_per_arm": [sum(row[k]["correct"] for row in pairs) for k in (0, 1)],
               "measured_wall_s_per_arm": [sum(row[k]["wall_s"] for row in pairs) for k in (0, 1)],
               "qualification_and_worker_s": sum(r["total_elapsed_s"] for pair in pairs for r in pair)
                   + sum(pair[1]["total_elapsed_s"] for pair in rule_pairs)}
    if challenger["passed"]:
        metrics["reason"] = "fixed four-thread rule dominates the candidate"
    binding = {**pairs[0][0]["binding"], "matched_profile_sha256": first.sha(profile),
               "workload": {k: v for k, v in pairs[0][0]["workload"].items() if k != "seed"}}
    return {**sig, "binding": binding, "metrics": metrics, "passed": passed}


def qualify(args):
    first.inputs(args)
    ctx = identity(args)
    write_once(args.state / "context.json", ctx)
    if not (args.state / "build.json").exists():
        first.worker(args, args.candidate, "build", args.state / "build.json", SEED)
    path = args.state / "profile.json"
    if not path.exists():
        worker(args, args.reference, "calibrate", path, SEED, None)
    profile = profile_source(args)
    for i, sequence in enumerate(first.SEQUENCES):
        seed = SEED + i
        for mode in sequence:
            tree = args.candidate if mode == "candidate" else args.reference
            dest = args.state / f"{mode}-{seed}.json"
            if not dest.exists():
                worker(args, tree, mode, dest, seed, profile)
            check_arm(args, tree, read(dest), profile, mode, seed)
    verdict = verify_offline(args)
    write_once(args.state / "verdict.json", verdict)
    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("qualify", "verify"))
    for name in ("worker-root", "state", "reference", "candidate", "gguf", "head", "order", "python"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--timeout", type=int, default=300, choices=range(1, 601), metavar="1..600")
    args = p.parse_args()
    if any(not path.is_absolute() for path in vars(args).values() if isinstance(path, Path)):
        p.error("use absolute paths")
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = verify_offline(args) if args.operation == "verify" else qualify(args)
        print(json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
