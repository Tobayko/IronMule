"""Reproduce the Qwen order, scratch weights, live decision and target release offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from lab.pk1 import restart_order, restart_release
from lab.pk1.restart_payback import measured_sources
from lab.rp.restart_pilot_campaign import assess
from training import pilot_restart_profiles, scoped_release
from training.code_author import read_request, read_response
from training.code_trial import read
from training.native_evidence import ROOT


def check(folder):
    with measured_sources(folder) as args:
        expected = read(args.order)
        historical = args.checkout / restart_order.ARCHIVE
        historical.parent.mkdir(parents=True)
        historical.symlink_to(ROOT / restart_order.ARCHIVE)
        regenerated = args.checkout / "regenerated.json"
        restart_order.build(SimpleNamespace(checkout=args.checkout, previous=folder / "observation-0.json",
                            canonical=folder / "observation-1.json", output=regenerated))
        if read(regenerated) != expected:
            raise ValueError("startup order differs from source traces")
        request_id, request = read_request(folder)
        read_response(folder, request_id)
        if request["order"] != expected:
            raise ValueError("development request differs from reconstructed order")
        saved_fit = read(folder / "startup-fit.json")
        fit = pilot_restart_profiles.fit(SimpleNamespace(**vars(args), weights=args.checkout / "replayed.npz",
                           report=args.checkout / "fit.json", seed=saved_fit["seed"]))
        if not ((args.checkout / "replayed.npz").read_bytes() == (folder / "startup-head.npz").read_bytes()
                and fit["holdout_seconds"] == saved_fit["holdout_seconds"]):
            raise ValueError("trained weights or held-out decisions differ")
        approved = restart_release.approved(args)
        pin = scoped_release.checksum(folder / "control.json")
        deployment = args.checkout / "deployment"
        deployment.mkdir()
        (deployment / "control.json").symlink_to(folder / "control.json")
        (deployment / "target-check.json").symlink_to(folder / "target-check.json")
        (deployment / "selection.json").symlink_to(folder / "release-activate.json")
        control = scoped_release.pinned(args.checkout, pin)
        if (control["qualification"] != approved or control["order"] != expected
                or scoped_release.selected(args.checkout, pin) != "candidate"):
            raise ValueError("published or activated source differs from the measured program")
        heads = [read(folder / f"pilot-{arm}.json") for arm in ("initial", "trained")]
        measured = assess(heads, control, pin, fit["weights_sha256"])
        if measured != read(folder / "pilot-comparison.json"):
            raise ValueError("recorded live decision differs from actual arms")
        rolled = read(folder / "release-exercise.json")
        before, after = rolled["before"], rolled["after"]
        if (rolled["receipt_sha256"] != pin or not rolled["passed"] or before["arm"] != "candidate"
                or after["arm"] != "reference" or before["source_sha256"] != approved["candidate_sha256"]
                or after["source_sha256"] != approved["reference_sha256"] or before["work_sha256"] != after["work_sha256"]
                or min(before["graded_correct"], after["graded_correct"]) != 18):
            raise ValueError("the measured rollback differs from the control receipt")
        return {"order": expected["order_id"], "measured_startup_phase": expected["evidence"]["resolved_source_phase"],
                "hypothesis_origin": "measured phase selection from an engineer-defined cache-replay protocol",
                "generated_request": request_id, "code_author": read(folder / "response.json")["model"],
                "qualified_source_sha256": approved["candidate_sha256"], "github_pr": control["publication"]["pr"],
                "merged_tree": control["publication"]["candidate_tree"], "reproduced_weights_sha256": fit["weights_sha256"],
                "training_family": fit["train_family"], "unseen_family": fit["holdout_family"],
                "median_cold_speedup": approved["metrics"]["median_cold_speedup"],
                "measured_policy_effect": measured, "explicit_rollback_preserved_correct_work": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="the exported PR 60 order folder")
    print(json.dumps(check(parser.parse_args().archive.resolve(strict=True)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
