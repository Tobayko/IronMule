"""Run the frozen four-client copy evaluator in unprivileged processes on a dedicated Linux pod."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from lab.pk1.native_trial import copy_result
from lab.rp import restricted_worker
from training.code_author import digest, write_once
from training.code_trial import PAIRS, judge, read, validate_arm, verify_deliveries

SOURCE = "ironmule/ggml_engine.py"
RULE = Path(__file__).with_name("cpu_four_thread_rule.py")
SEQUENCES = (("reference", "candidate", "rule"), ("candidate", "reference", "rule"),
             ("reference", "rule", "candidate"), ("candidate", "rule", "reference"),
             ("rule", "reference", "candidate"), ("rule", "candidate", "reference"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scoring_hash(evaluator, helper):
    from training import code_trial
    return digest({k: sha(path) for k, path in (
        ("broker", restricted_worker.__file__), ("qualification", __file__), ("score", code_trial.__file__),
        ("serving", evaluator), ("scratch_qualification", helper), ("four_thread_rule", RULE))})


def order_digest(order):
    from training.experiment_planner import digest as planner_digest
    return planner_digest({k: v for k, v in order.items() if k != "order_id"})


def inputs(args, *, on_target=True):
    order = read(args.order)
    if (order["binding"]["provider"] != "cpu" or order["target"]["path"] != SOURCE
            or not re.fullmatch("[0-9a-f]{40}", order["base_commit"])
            or order["order_id"] != order_digest(order)):
        raise ValueError("invalid CPU order, digest or source target")
    if on_target and (sha(args.gguf) != order["binding"]["model_sha256"]
                      or sha(args.head) != order["binding"]["scratch_thread_head"]["weights_sha256"]
                      or sorted(os.sched_getaffinity(0)) != order["binding"]["affinity"]):
        raise ValueError("model, CPU affinity, controller or order target differs")
    evaluator = args.reference / "training/serving_trial.py"
    helper = evaluator.with_name("pilot_threads.py")
    if sha(helper) != order["binding"]["scratch_thread_head"]["helper_sha256"]:
        raise ValueError("scratch qualifier changed")
    return order, evaluator, helper


def worker(args, tree, mode, destination, seed):
    order, evaluator, helper = inputs(args)
    env = {k: os.environ[k] for k in ("IRONMULE_GGML_BINDINGS", "LD_LIBRARY_PATH", "LLAMA_CPP_LIB_PATH")}
    env.update(IRONMULE_GGML_GPU_LAYERS="0", IRONMULE_PILOT="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(tree))
    readonly = [tree, args.gguf, args.head, evaluator, helper, RULE, args.python.parent.parent, Path("/etc/machine-id"),
                Path(env["IRONMULE_GGML_BINDINGS"]), Path(env["LLAMA_CPP_LIB_PATH"]),
                *(Path(p) for p in env["LD_LIBRARY_PATH"].split(":") if p and not Path(p).resolve().is_relative_to("/usr"))]
    # Execute the venv entry point by its original path so Python picks up its pyvenv.cfg.
    # The private CUDA cache is irrelevant to CPU work and its per-worker path cannot identify an arm.
    python = ["/usr/bin/env", "-u", "CUDA_CACHE_PATH", str(args.python)]
    with tempfile.TemporaryDirectory(dir=args.worker_root) as temp:
        stage = Path(temp)
        stage.chmod(0o711)
        out = stage / "out"
        out.mkdir(mode=0o700)
        os.chown(out, 60000, 60000)
        if mode == "build":
            commands = [[*python, "-X", f"pycache_prefix={out}/pycache", "-m", "compileall", "-q", str(tree / "ironmule")],
                        [*python, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q",
                         "tests/engine/test_gguf_tokenizer.py", "tests/engine/test_ggml_engine.py"]]
        else:
            script = RULE if mode == "rule" else evaluator
            commands = [[*python, str(script), "--gguf", str(args.gguf), "--out", str(out / "arm.json"),
                         "--thread-candidate", str(args.head), "--require-provider", "cpu", "--threads", "2",
                         "--clients", "4", "--waves", "4", "--seed", str(seed), "--revision", order["base_commit"]]]
        for index, argv in enumerate(commands):
            proc = restricted_worker.run(argv, env=env, cwd=tree, output=out, readonly=readonly, timeout=args.timeout)
            log = destination.with_name(f"{destination.stem}-{index}.log")
            log.write_text(proc.stdout)
            if proc.returncode:
                raise RuntimeError(f"restricted {mode} exited {proc.returncode}; see {log}")
        if mode == "build":
            write_once(destination, {"passed": True, "log_sha256": sha(log), "source_sha256": sha(tree / SOURCE)})
        else:
            copy_result(out / "arm.json", destination, 60000)


def check_arm(args, tree, row, seed, *, on_target=True, rule=False):
    order, evaluator, _ = inputs(args, on_target=on_target)
    validate_arm(row, tree, order, seed, {"evaluator_sha256": sha(evaluator)})
    if row["commit"] != order["base_commit"]:
        raise ValueError("different reference revision")
    if rule:
        witness = row["simple_thread_rule"]
        if (witness["source_sha256"] != sha(RULE) or witness["actual_threads"] != [4]
                or witness["overrides"] <= 0 or row["scratch_measured_decisions"] != 0):
            raise ValueError("unverified four-thread rule")
    elif row.get("scratch_measured_decisions", 0) <= 0:
        raise ValueError("no bound scratch decision in the measured workload")


def verify_offline(args):
    """Recompute quality, source, binding and all six comparisons from immutable raw receipts."""
    order, evaluator, helper = inputs(args, on_target=False)
    signature = dict(reference_sha256=sha(args.reference / SOURCE), candidate_sha256=sha(args.candidate / SOURCE),
                     evaluator_sha256=scoring_hash(evaluator, helper), order_id=order["order_id"])
    if read(args.state / "context.json") != signature or signature["reference_sha256"] == signature["candidate_sha256"]:
        raise ValueError("source, order or evaluator changed")
    expected_build = read(args.state / "build.json")
    if (expected_build.get("passed") is not True or expected_build["source_sha256"] != signature["candidate_sha256"]
            or expected_build["log_sha256"] != sha(args.state / "build-1.log")):
        raise ValueError("missing verified build and test run")
    pairs, rule_pairs = [], []
    for i in range(PAIRS):
        seed = args.seed + i
        both = []
        for arm, tree in (("reference", args.reference), ("candidate", args.candidate)):
            row = read(args.state / f"{arm}-{seed}.json")
            check_arm(args, tree, row, seed, on_target=False)
            both.append(row)
        ref, cand = both
        if any((r["requests"], r["reused_prompt_tokens"]) != (c["requests"], c["reused_prompt_tokens"])
               for r, c in zip(ref["waves"], cand["waves"], strict=True)):
            raise ValueError("unequal prompt cache state or delivered work")
        fixed = read(args.state / f"rule-{i}.json")
        check_arm(args, args.reference, fixed, seed, on_target=False, rule=True)
        if any((r["requests"], r["reused_prompt_tokens"]) != (f["requests"], f["reused_prompt_tokens"])
               for r, f in zip(ref["waves"], fixed["waves"], strict=True)):
            raise ValueError("rule measured different work or cache conditions")
        pairs.append(both)
        rule_pairs.append((cand, fixed))
    verdict = judge(pairs)
    challenger = judge(rule_pairs)
    if challenger["reason"] != "paired hardware gate":
        raise ValueError("unmatched simple rule comparison")
    if challenger["passed"]:
        verdict = {**verdict, "passed": False, "reason": "fixed four-thread rule dominates the candidate"}
    return {**signature, "passed": verdict["passed"], "binding": {**pairs[0][0]["binding"],
            "workload": {k: v for k, v in pairs[0][0]["workload"].items() if k != "seed"}},
            "metrics": {**verdict, "fixed_four_threads_vs_candidate": challenger, "correct_per_arm": [sum(pair[arm]["correct"] for pair in pairs) for arm in (0, 1)],
                        "measured_wall_s_per_arm": [sum(pair[arm]["wall_s"] for pair in pairs) for arm in (0, 1)],
                        "worker_s_including_qualification_per_arm": [
                            sum(pair[arm]["total_elapsed_s"] for pair in pairs) for arm in (0, 1)]}}


def verify(args):
    inputs(args)
    return verify_offline(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "arm", "build", "verify"))
    for key in ("worker-root", "state", "reference", "gguf", "head", "order", "python"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--tree", type=Path, help="implementation for an individual arm or build")
    parser.add_argument("--seed", type=int, default=8351)
    parser.add_argument("--timeout", type=int, default=300, choices=range(1, 601), metavar="1..600")
    args = parser.parse_args()
    if not all(p.is_absolute() for p in vars(args).values() if isinstance(p, Path)):
        parser.error("use absolute target paths")
    args.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    order, evaluator, helper = inputs(args)
    if args.mode in ("arm", "build"):
        if not args.tree:
            parser.error("an individual run needs --tree")
        dest = args.state / ("build.json" if args.mode == "build" else f"arm-{args.seed}.json")
        if not dest.exists():
            worker(args, args.tree, args.mode, dest, args.seed)
        if args.mode == "arm":
            row = read(dest)
            check_arm(args, args.tree, row, args.seed)
            verify_deliveries(row, args.seed)
        print(dest)
        return
    if not args.candidate:
        parser.error("paired evaluation needs --candidate")
    if args.mode == "qualify":
        start = time.monotonic()
        write_once(args.state / "context.json", dict(reference_sha256=sha(args.reference / SOURCE),
                   candidate_sha256=sha(args.candidate / SOURCE), evaluator_sha256=scoring_hash(evaluator, helper),
                   order_id=order["order_id"]))
        if not (args.state / "build.json").exists():
            worker(args, args.candidate, "build", args.state / "build.json", args.seed)
        for i in range(PAIRS):
            seed = args.seed + i
            for arm in SEQUENCES[i]:
                tree = args.candidate if arm == "candidate" else args.reference
                dest = args.state / (f"rule-{i}.json" if arm == "rule" else f"{arm}-{seed}.json")
                if not dest.exists():
                    worker(args, tree, "rule" if arm == "rule" else "arm", dest, seed)
                check_arm(args, tree, read(dest), seed, rule=arm == "rule")
        print(f"qualify process wall seconds {time.monotonic() - start:.3f}", flush=True)
    verdict = verify(args)
    write_once(args.state / "verdict.json", verdict)
    print(json.dumps(verdict["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
