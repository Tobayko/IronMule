"""Run one bounded, source-attributed observation through the live, trained startup Pilot."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lab.pk1 import native_trial as native
from lab.pk1 import restart_trial
from training import scoped_release as release
from training.code_author import write_once
from training.code_trial import read
from training.pilot_restart_profiles import select

WORKER = Path(__file__).with_name("post_restart_profile.py")
CONTRACT = Path(__file__).resolve().parents[2] / "training/post_restart_contract.json"


def observe(args):
    t0 = time.monotonic()
    contract = read(CONTRACT)
    if args.seed not in contract["seeds"]:
        raise ValueError("unregistered discovery seed")
    pin = contract["control_receipt_sha256"]
    receipt = release.pinned(args.state, pin)
    cache = args.trial / "receipt.json"
    q = receipt["qualification"]
    if (receipt["publication"]["candidate"] != contract["base_commit"] or receipt["order"]["order_id"] != contract["parent_order"]
            or q["candidate_sha256"] != contract["source_sha256"]):
        raise ValueError("foreign or unpublished incumbent")
    before = release.selected(args.state, pin)
    ready = native.checksum(cache) == q["binding"]["cache_receipt_sha256"] and 0 <= time.time() - read(cache)["created_s"] <= 86400
    choice = select(args.weights, receipt, family=contract["worker_family"], cache_ready=ready, active=before == "candidate",
                    expected_sha256=contract["weights_sha256"])
    if choice["profile"] != "candidate" or choice["reason"] != "qualified model choice":
        raise ValueError("current startup head has not selected this exact incumbent")
    tree = release.projected_tree(receipt, args.checkout, args.worker_root, args.candidate_source)
    env = {"IRONMULE_" + k: v for k, v in native.FLAGS.items()}
    env.update(IRONMULE_KG_LIB="/kernel/active.so", IRONMULE_W4_TABLE="/kernel/dispatch.table",
               IRONMULE_KG_RESTART_RECEIPT="/kernel/restart-receipt.json", IRONMULE_KG_RESTART_SHA256=native.checksum(cache))
    mounts = [(args.reference_lib, env["IRONMULE_KG_LIB"]), (args.table, env["IRONMULE_W4_TABLE"]),
              (cache, env["IRONMULE_KG_RESTART_RECEIPT"]), (restart_trial.HARNESS, "/kernel/restart_arm.py"),
              (WORKER, "/kernel/post_restart_profile.py")]
    run = SimpleNamespace(**{**vars(args), "checkout": tree})
    measured = time.monotonic()
    with TemporaryDirectory(dir=args.out.parent) as temp:
        output = Path(temp) / "observation.json"
        native.launch(run, output, ["/kernel/post_restart_profile.py", "--gguf", "/models/" + args.gguf.name,
                                    "--output", "/output/" + output.name, "--seed", str(args.seed)], mounts, env)
        observation = read(output)
    arm = observation["arm"]
    arm["broker_elapsed_s"] = time.monotonic() - measured
    restart_trial.validate(arm, args.seed, expected_source=q["candidate_sha256"])
    after = release.selected(args.state, pin)
    if (before != "candidate" or after != before or not arm["reused"] or arm["binding"] != q["binding"]["runtime"]
            or arm["mask_bits"] != q["binding"]["mask_bits"] or arm["cuda_vendor"] != q["binding"]["vendor"]
            or arm["pilot_sha256"] != q["binding"]["pilot_sha256"] or observation["profiler_sha256"] != native.checksum(WORKER)):
        raise ValueError("measured worker differs from the live qualified incumbent")
    return {**observation, "schema": "ironmule.post-restart-observation.v1", "seed": args.seed,
            "base_commit": contract["base_commit"], "parent_order": contract["parent_order"], "control_receipt_sha256": pin,
            "contract_sha256": native.checksum(CONTRACT), "weights_sha256": native.checksum(args.weights),
            "selection": choice, "selection_before": before, "selection_after": after,
            "broker_elapsed_s": time.monotonic() - t0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("state", "trial", "checkout", "candidate-source", "gguf", "reference-lib", "table", "worker-root", "weights", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timeout", type=int, choices=range(1, 301), default=150)
    p.add_argument("--worker-uid", type=int, default=60000)
    args = p.parse_args()
    if args.out.exists():
        p.error("this seed already has an observation")
    write_once(args.out, observe(args))


if __name__ == "__main__":
    main()
