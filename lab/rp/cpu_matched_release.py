"""Run a fresh arm of the separately qualified and merged, reference-calibrated CPU experiment."""
from __future__ import annotations

import argparse
import base64
import hashlib
import time

from lab.rp import cpu_code_release, cpu_matched_trial
from training import scoped_release as release
from training.code_trial import read
from training.trial_release import require


def profile_path(receipt, supplied, directory, arm):
    expected = receipt["order"]["paired_profile_sha256"]
    record = receipt["qualification"]
    raw = base64.b64decode(record["matched_profile_base64"], validate=True)
    require(hashlib.sha256(raw).hexdigest() == record["binding"]["matched_profile_sha256"] == expected,
            "published profile differs from the measured target")
    if supplied and arm == "candidate":
        require(release.checksum(supplied) == expected, "supplied target profile changed")
    path = directory / "profile.json"
    if not path.exists():
        path.write_bytes(raw)
    require(release.checksum(path) == expected, "temporary profile changed")
    return path


def measured(args, receipt, arm, seed, directory):
    started = time.monotonic()
    expected = receipt["qualification"]
    require(cpu_code_release.author_order(receipt) == read(args.order) and receipt["adapter"] == cpu_code_release.MATCHED,
            "foreign matched-profile development order")
    profile = profile_path(receipt, getattr(args, "profile", None), directory, arm)
    base = release.projected_tree(receipt, args.reference, args.worker_root)
    candidate = arm == "candidate"
    tree = release.projected_tree(receipt, args.reference, args.worker_root, args.candidate / release.SOURCE) if candidate else base
    options = argparse.Namespace(**{**vars(args), "reference": base, "candidate": tree})
    identity = cpu_matched_trial.identity(options)
    require(cpu_code_release.revision(identity, matched=True)["evaluator_sha256"] == expected["evaluator_sha256"]
            and release.checksum(tree / release.SOURCE) == expected[arm + "_sha256"], "changed matched evaluator or source")
    dest = directory / (arm + ".json")
    cpu_matched_trial.worker(options, tree, arm, dest, seed, profile)
    row = read(dest)
    cpu_matched_trial.check_arm(options, tree, row, profile, arm, seed)
    workload = {k: v for k, v in row["workload"].items() if k != "seed"}
    require({**row["binding"], "matched_profile_sha256": release.checksum(profile), "workload": workload} == expected["binding"]
            and row["scratch_measured_decisions"] > 0, "fresh arm failed to apply the measured CPU head and profile")
    return row, time.monotonic() - started
