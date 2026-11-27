"""Persist a same-admission random-vs-trained policy comparison on a merged target release."""
import shlex
from pathlib import Path, PurePosixPath

from lab.rp.native_campaign import command, job, lease, shell, snapshot, upload
from training import scoped_release
from training.code_author import write_once
from training.code_trial import read


def assess(rows, receipt, pin, weights):
    ref, cand = rows
    for row, ablated in ((ref, True), (cand, False)):
        work, choice = row["work"], row["selection"]
        if (row["schema"] != "ironmule.restart-pilot-serving.v1" or row["order"] != receipt["order"]["order_id"]
                or row["pr"] != receipt["publication"]["pr"] or row["control_receipt_sha256"] != pin
                or row["weights_sha256"] != weights or choice["weights_sha256"] != weights or row["seed"] != 94752
                or choice["ablate_weights"] is not ablated or choice["previous"] or choice["family"] != "mixed-first-32"
                or work["source_sha256"] != receipt["qualification"][work["arm"] + "_sha256"]
                or work["seed"] != 94752 or work["cache_used"] != (work["arm"] == "candidate")
                or work["graded_correct"] != 18 or work["pilot_decisions"] <= 0
                or (work["cold_performance_completions"], work["warm_performance_completions"]) != (32, 32)):
            raise ValueError("policy comparison does not match the measured release, weights or workload")
    applied = (all(row["eligible_release"] == row["durable_selection_after"] == "candidate"
                   and row["qualified_cache_available"] for row in rows)
               and ref["selection"]["reason"] == "unresolved model margin" and cand["selection"]["reason"] == "qualified model choice"
               and ref["selection"]["profile"] == ref["work"]["arm"] == "reference"
               and cand["selection"]["profile"] == cand["work"]["arm"] == "candidate"
               and ref["work"]["work_sha256"] == cand["work"]["work_sha256"])
    return {"order": ref["order"], "pr": ref["pr"], "weights_sha256": weights, "same_work_policy_changed_source": applied,
            "reference_and_trained_broker_wall_s": [r["broker_elapsed_through_completed_work_s"] for r in rows],
            "reference_and_trained_decision_ms": [r["selection"]["cpu_ms"] for r in rows],
            "work_sha256": cand["work"]["work_sha256"]}


def exercise(args, cfg, remote, active):
    folder = args.state / "native-pilot"
    folder.mkdir(exist_ok=True)
    pod = lease(cfg["pod_id"])
    pin = active["receipt_sha256"]
    receipt = scoped_release.pinned(args.state / "release", pin)
    head = args.state / "startup-head.npz"
    weights = scoped_release.checksum(head)
    if read(args.state / "startup-fit.json")["weights_sha256"] != weights:
        raise ValueError("weights differ from qualified fit")
    root = Path(__file__).resolve().parents[2]
    files = ("lab/pk1/restart_pilot_run.py", "training/pilot_restart_profiles.py", "ironmule/pilot.py")
    sources = [str(PurePosixPath(cfg["framework"]) / name) for name in files]
    remote_hashes = shell(pod, shlex.join(["sha256sum", "--", *sources])).splitlines()
    if remote_hashes != [f"{scoped_release.checksum(root / name)}  {path}" for name, path in zip(files, sources, strict=True)]:
        raise ValueError("target Pilot implementation differs")
    upload(pod, remote / "startup-head.npz", head.read_bytes())
    common = ["--state", remote / "active", "--trial", remote / "trial", "--order", remote / "order.json",
              "--candidate-source", remote / "proposal.py", "--checkout", cfg["checkout"], "--worker-root", cfg["worker_root"],
              "--gguf", cfg["gguf"], "--reference-lib", cfg["reference_lib"], "--table", cfg["table"],
              "--receipt-sha256", pin, "--weights", remote / "startup-head.npz", "--weights-sha256", weights, "--seed", "94752"]
    names = ["pilot-initial.json", "pilot-trained.json"]
    for name in names:
        snapshot(pod, remote, [name], folder)
        if not (folder / name).exists():
            flags = ["--ablate-weights"] if name == names[0] else []
            cmd = command(cfg, "lab.pk1.restart_pilot_run", [*common, *flags, "--out", remote / name])
            job(pod, name[:-5] + "-" + active["order"][:16], cmd)
            return {**active, "stage": "pilot serving", "pending": name}
    result = assess([read(folder / n) for n in names], receipt, pin, weights)
    write_once(folder / "comparison.json", result)
    return {**active, "startup_policy": result}
