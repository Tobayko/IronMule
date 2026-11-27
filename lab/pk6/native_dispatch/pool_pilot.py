"""Fit the existing CPU Pilot on paired pool profiles and price a restart from observed traffic."""

import argparse
import gzip
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

from ironmule.pilot import Pilot
from lab.pk6.mixed_progress import grade, trial
from lab.pk6.native_dispatch import document_quality, score
from lab.pk6.native_dispatch.run import SEEDS
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save
from training.pilot_w4_profiles import fit_head
from training.scoped_release import rollback
from training.trial_release import switch_lock
from lab.rp.accounting import save as persist

NAMES = ("stock", "compact")
WIDTHS = (1, 8, 32)
FIT_SEEDS = (30, 31, 32)
MARGIN = math.log(1.10)


def traffic(rows, width):
    group = [row for row in rows if row["width"] == width and row["repeat"] < 3]
    if len(group) != 3 or [w["repeat"] for w in group] != [0, 1, 2]:
        raise ValueError("need three complete observations at this concurrency")
    counts = []
    for w in group:
        grade.measure(w)
        counts.extend(r["prompt_tokens"] for r in w["requests"])
    return np.array(
        [
            math.log2(width) / 5,
            median(counts) / 2048,
            sum(v > 1024 for v in counts) / len(counts),
        ],
        dtype=np.float32,
    ), counts


def choose(heads, inputs, qualified):
    return Pilot.startup_profile(
        heads, inputs, NAMES, qualified, "stock", in_support=True, margin_error=MARGIN
    )


def fit(root, out):
    started = time.perf_counter()
    inputs, times, files, bounds, qualifications = {}, {}, {}, {}, {}
    native = score.read(root / "operator-calibration/completed.json")
    identity = native["hardware"], native["gguf"], native["ggml_cuda"]
    if identity[1] != GGUF:
        raise ValueError("foreign controller target")
    engines = set()
    quality_root = root / "controls/document-retrieval"
    retrieval = document_quality.assess(quality_root) if quality_root.exists() else {}
    retrieval_pass = False
    if set(retrieval) == {"R2", "H2"}:
        stock, candidate = (retrieval[a] for a in ("R2", "H2"))
        # Every extraction the reference answered correctly must survive. The comparison
        # needs correct reference examples for all three insertion positions in both seeds.
        covers_positions = all(
            {int(case.split("/")[1]) % 3 for case in wave} == {0, 1, 2}
            for wave in stock["correct_case_ids"][:2]
        )
        retrieval_pass = (
            covers_positions
            and all(
                row["correct_by_wave"][-1] == 32
                and row["hardware"] == identity[0]
                and row["model"] == GGUF
                for row in retrieval.values()
            )
            and all(
                set(a) <= set(b)
                for a, b in zip(
                    stock["correct_case_ids"],
                    candidate["correct_case_ids"],
                    strict=True,
                )
            )
        )
    for family, folder in (
        ("mixed", root / "http"),
        ("documents", root / "controls/capacity"),
    ):
        result = score.run(folder)
        for idx in (2, 1):
            arms = [f"{a}{idx}" for a in ("R", "H")]
            groups = []
            for arm in arms:
                (wave_file,) = folder.rglob(f"waves-{arm}.jsonl")
                target = score.read(wave_file.parent / "target.json")
                if family == "documents":
                    engines.add((target["source"], target["engine_sha256"]))
                measured = result["arms"][arm]
                if (
                    measured["excluded_from_promotion"]
                    or not measured["proof"]["shared_pool"]
                    or set(measured["quality"].values()) != {32}
                    or target["hardware"] != identity[0]
                    or target["gguf_sha256"] != identity[1]
                ):
                    raise ValueError("unqualified source process")
                if (
                    arm.startswith("H")
                    and not measured["proof"]["verified_compact_pool"]
                ):
                    raise ValueError("missing physical allocation")
                files[str(wave_file.relative_to(root))] = checksum(wave_file)
                groups.append(score.waves(wave_file.parent, arm))
            for width in WIDTHS:
                key = (family, idx, width)
                inputs[key], counts = traffic(groups[0], width)
                pair = result["pairs"][f"{arms[1]}/{arms[0]}/{width}"]
                if not pair["same_fixed_work"]:
                    raise ValueError("foreign workload")
                times[key] = [
                    median(
                        w["elapsed_s"]
                        for w in g
                        if w["width"] == width and w["repeat"] < 3
                    )
                    for g in groups
                ]
                if family == "documents":
                    qualifications.setdefault(str(width), []).append(pair["clear_win"])
                    if idx == 2:
                        bounds[str(width)] = [
                            64 * (min(counts) // 64),
                            64 * math.ceil(max(counts) / 64),
                        ]
    if len(engines) != 1:
        raise ValueError("inconsistent source in physical capacity qualification")
    source, engine = engines.pop()
    if retrieval and any(
        v["source"] != source or v["engine_sha256"] != engine
        for v in retrieval.values()
    ):
        raise ValueError("document quality uses a different serving engine")
    fit_keys = [k for k in inputs if k[1] == 2]
    heads = [fit_head(times, fit_keys, seed, inputs) for seed in FIT_SEEDS]
    random = [fit_head(times, fit_keys, seed, inputs, epochs=0) for seed in FIT_SEEDS]
    scores = []
    for key, features in inputs.items():
        family, idx, width = key
        d = choose(heads, features, NAMES)
        scores.append(
            {
                "workload": family,
                "server_seed_index": idx,
                "width": width,
                "split": "fit" if idx == 2 else "heldout",
                "random": choose(random, features, NAMES)["profile"],
                "trained": d["profile"],
                "stock_rule": "stock",
                "document_rule": "compact" if family == "documents" else "stock",
                "measured_compact_over_stock_speed": times[key][0] / times[key][1],
                "inference_us": d["cpu_us"],
                "min_log_margin": d["margin"],
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "pilot.npz",
        **{
            f"h{i}_{j}": weights
            for i, head in enumerate(heads)
            for j, weights in enumerate(head)
        },
    )
    report = {
        "source": source,
        "engine_sha256": engine,
        "hardware": identity[0],
        "model": identity[1],
        "native": identity[2],
        "pool_plan_sha256": checksum(root / "compact-pool.json"),
        "weights_sha256": checksum(out / "pilot.npz"),
        "files": files,
        "scores": scores,
        "initialization_seeds": list(FIT_SEEDS),
        "bounds": bounds,
        "qualified_widths": [
            int(w)
            for w, items in qualifications.items()
            if all(items) and retrieval_pass
        ],
        "document_quality": retrieval,
        "elapsed_cpu_host_s": time.perf_counter() - started,
        "scope": "Restart planning after measured sustained document traffic. Fit uses seed 943819. Seed 180341 runs in separate processes. A simple document-length rule is the comparison. Throughput labels alone cannot authorize use.",
    }
    save(out / "fit.json", report)
    return report


def select(policy, rows, width, hardware, native):
    started = time.perf_counter()
    proof = score.read(policy / "fit.json")
    features, counts = traffic(rows, width)
    bounds = proof["bounds"].get(str(width))
    supported = (
        proof["hardware"] == hardware
        and proof["native"] == native
        and proof["model"] == GGUF
        and bounds
        and bounds[0] <= min(counts) <= max(counts) <= bounds[1]
        and width in proof["qualified_widths"]
    )
    if not supported:
        decision = {
            "profile": "stock",
            "reason": "unqualified hardware, workload or document accuracy",
        }
    else:
        try:
            if checksum(policy / "pilot.npz") != proof["weights_sha256"]:
                raise ValueError("changed checkpoint")
            with np.load(policy / "pilot.npz", allow_pickle=False) as bundle:
                heads = [
                    [bundle[f"h{i}_{j}"] for j in range(6)]
                    for i in range(len(FIT_SEEDS))
                ]
            decision = choose(heads, features, NAMES)
        except (OSError, ValueError, KeyError):
            decision = {"profile": "stock", "reason": "unusable CPU checkpoint"}
    return {
        **decision,
        "width": width,
        "observed_prompt_range": [min(counts), max(counts)],
        "total_decision_us": (time.perf_counter() - started) * 1e6,
    }


def verify_trial(folder, arm, reference, width, *, controller_s=0.0):
    started = time.perf_counter()
    start, end = (
        score.read(folder / f"{kind}-{arm}.json") for kind in ("start", "end")
    )
    target = score.read(folder / "target.json")
    rows = score.waves(folder, arm)
    if (
        start["hardware"] != end["hardware"]
        or start["hardware"] != target["hardware"]
        or start["arm"] != arm
        or end["arm"] != arm
        or target["gguf_sha256"] != GGUF
        or not all(
            start["unix"] <= row["unix_start"] < row["unix_end"] <= end["unix"]
            for row in rows
        )
    ):
        raise ValueError("invalid serving process or timing")
    quality = [w for w in rows if w["repeat"] == 3]
    checks = [w for w in reference if w["repeat"] == 3]
    if len(rows) != 10 or len(quality) != 1 or len(checks) != 1:
        raise ValueError("incomplete profile qualification")
    verified = grade.bindings(folder, arm, rows)
    pair = grade.paired(
        [w for w in reference if w["width"] == width and w["repeat"] < 3],
        [w for w in rows if w["width"] == width and w["repeat"] < 3],
    )
    original, actual = checks[0], quality[0]
    if original["input_digest"] != actual["input_digest"] or [
        r["case"] for r in original["requests"]
    ] != [r["case"] for r in actual["requests"]]:
        raise ValueError("different quality inputs")
    same_quality = len(actual["requests"]) == 32 and all(
        r["correct"] is True for r in actual["requests"]
    )
    path = folder / f"trace-{arm}.txt"
    trace = (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(path.with_suffix(".txt.gz").read_bytes())
    )
    active = b"physical_pool cells=16384 allocated=16384 cap=32" in trace
    healthy = (
        score.read(folder / f"ready-{arm}.json")["health"]["gate"].get("passes") is True
    )
    ref_s, live_s = (
        sum(w["elapsed_s"] for w in block if w["width"] == width and w["repeat"] < 3)
        for block in (reference, rows)
    )
    overhead = controller_s + time.perf_counter() - started
    net = ref_s / (live_s + overhead)
    return {
        "retained": bool(
            score.clear_win(pair)
            and same_quality
            and active
            and healthy
            and net >= 1.10
        ),
        "controller_fit_launch_verify_s": overhead,
        "net_fixed_work_speed_ratio": net,
        "comparison": pair,
        "factual_correct": sum(r["correct"] for r in actual["requests"]),
        "verified_allocation": active,
        "quality_gate": healthy,
        "trace_sha256": verified["trace_sha256"],
    }


def serve(args):
    started = time.perf_counter()
    target = score.read(args.history.parent / "target.json")
    hardware = trial.board()
    proof = score.read(args.policy / "fit.json")
    # A prior run must have finished on this board. It supplies only input measurements.
    arm = args.history.stem.removeprefix("waves-")
    prior_end = score.read(args.history.parent / f"end-{arm}.json")
    if (
        not args.history.name.startswith("waves-")
        or prior_end["arm"] != arm
        or prior_end["hardware"] != hardware
        or target["hardware"] != hardware
        or target["gguf_sha256"] != GGUF
    ):
        raise ValueError("foreign or unfinished traffic history")
    if checksum(args.pool_plan) != proof["pool_plan_sha256"]:
        raise ValueError("physical pool plan changed")
    if (
        checksum(Path("ironmule/ggml_engine.py")) != proof["engine_sha256"]
        or target["source"] != proof["source"]
    ):
        raise ValueError("serving implementation differs from qualification")
    rows = score.waves(args.history.parent, arm)
    if any(r["arm"] != arm or r["unix_end"] > prior_end["unix"] for r in rows):
        raise ValueError("invalid traffic receipt")
    native = checksum(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so")
    decision = select(args.policy, rows, args.width, hardware, native)
    args.out.mkdir(parents=True, exist_ok=False)
    if decision["profile"] == "compact" and (
        arm != f"R{args.seed_index}"
        or checksum(args.history) not in proof["files"].values()
        or args.workload != "stress"
    ):
        raise ValueError(
            "compact qualification needs its matched stock history and workload"
        )
    deployment = args.state / "deployment"
    deployment.mkdir(parents=True, exist_ok=True)
    pending = deployment / "trial.json"
    with switch_lock(deployment):
        unfinished = (
            pending.exists() and score.read(pending).get("status") != "finished"
        )
        if (deployment / "selection.json").exists() or unfinished:
            decision = {
                **decision,
                "profile": "stock",
                "reason": "pending trial" if unfinished else "persistently withdrawn",
            }
        elif decision["profile"] == "compact":
            persist(pending, {"status": "pending", "output": args.out.name})
    compact = decision["profile"] == "compact"
    chosen = ("H" if compact else "R") + str(args.seed_index)
    prelaunch = time.perf_counter() - started
    save(
        args.out / "decision.json",
        {
            "decision": decision,
            "arm": chosen,
            "history_sha256": checksum(args.history),
            "policy_sha256": checksum(args.policy / "fit.json"),
            "hardware": hardware,
            "native": native,
            "gguf": GGUF,
            "seed": SEEDS[args.seed_index - 1],
            "prelaunch_s": prelaunch,
        },
    )
    command = [
        sys.executable,
        "-m",
        "lab.pk6.native_dispatch." + args.workload,
        "--out",
        str(args.out / "http"),
        "--arms",
        chosen,
        "--reference-archive",
        str(args.reference_archive),
    ]
    if compact:
        command += ["--pool-plan", str(args.pool_plan)]
    try:
        subprocess.run(command, check=True)
        if compact:
            result = verify_trial(
                args.out / "http",
                chosen,
                rows,
                args.width,
                controller_s=proof["elapsed_cpu_host_s"] + prelaunch,
            )
            save(args.out / "verification.json", result)
            if not result["retained"]:
                rollback(args.state, "new fixed-work service comparison failed")
            with switch_lock(deployment):
                persist(
                    pending,
                    {
                        "status": "finished",
                        "retained": result["retained"],
                        "output": args.out.name,
                    },
                )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        if compact:
            rollback(args.state, "compact service or measurement failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    train = sub.add_parser("fit")
    train.add_argument("evidence", type=Path)
    train.add_argument("out", type=Path)
    withdraw = sub.add_parser("withdraw")
    withdraw.add_argument("state", type=Path)
    withdraw.add_argument("--reason", required=True)
    launch = sub.add_parser("serve")
    for field in (
        "policy",
        "pool-plan",
        "history",
        "reference-archive",
        "out",
        "state",
    ):
        launch.add_argument("--" + field, required=True, type=Path)
    launch.add_argument("--width", required=True, type=int, choices=WIDTHS)
    launch.add_argument("--seed-index", type=int, choices=(1, 2), required=True)
    launch.add_argument(
        "--workload",
        choices=("run", "stress"),
        required=True,
        help="Choose the evaluator for new traffic after the startup decision.",
    )
    args = parser.parse_args()
    if args.mode == "fit":
        fit(args.evidence, args.out)
    elif args.mode == "withdraw":
        rollback(args.state, args.reason)
    else:
        serve(args)
