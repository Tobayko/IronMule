"""Recheck two measured, profiled A100 restarts against the actually merged source and frozen release."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from lab.pk1 import post_restart_profile, restart_trial
from training import pilot_restart_profiles, scoped_release
from training.trial_release import require

FROZEN = Path(__file__).resolve().parents[2] / "training/post_restart_contract.json"
SOURCE = "ironmule/ggml_engine.py"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value, *, allow_zero=False):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if allow_zero else value > 0)


def check_phases(phases, names, sites, arm):
    require([p["phase"] for p in phases] == names, "missing, repeated or reordered phase")
    total = 0
    for phase in phases:
        span, unused = phase["wall_s"], phase["unassigned_s"]
        require(finite(span) and finite(unused, allow_zero=True), "invalid phase timing")
        seen = set()
        times = []
        for row in phase["sites"]:
            method = row["function"]
            require(method in sites and method not in seen and row["path"] == SOURCE, "unbound or duplicate source site")
            line, eligible, reason = sites[method]
            require(row["line"] == line and row["eligible"] is eligible and row["reason"] == reason
                    and type(row["calls"]) is int and row["calls"] >= 0 and finite(row["self_s"], allow_zero=True),
                    "altered attribution, source location or trust exclusion")
            times.append(row["self_s"])
            seen.add(method)
        require(abs(math.fsum([unused, *times]) - span) <= 1.01e-6, "unassigned and exclusive times do not sum to phase wall")
        if phase["phase"] == "startup":
            require(span <= arm["initialized_s"] + 1e-6, "instrumented constructor exceeds initialization")
        else:
            wave = arm["factual_timing"] if phase["phase"] == "factual" else arm[phase["phase"]]
            require(wave["wall_s"] <= span + 1e-6, "profile excludes part of the measured request wave")
        total += span
    require(total <= arm["broker_elapsed_s"] + 1e-6, "phase accounting exceeds the complete worker lifetime")


def check_selection(choice, weights, receipt, contract):
    expected = pilot_restart_profiles.select(weights, receipt, family=contract["worker_family"], cache_ready=True,
                                            active=True, expected_sha256=contract["weights_sha256"])
    variable = {"cpu_ms", "cpu_us", "scores", "margin"}
    require(expected["profile"] == "candidate" and expected["reason"] == "qualified model choice"
            and set(choice) == set(expected) and all(choice[k] == expected[k] for k in expected.keys() - variable),
            "observed startup decision differs from the qualified scratch controller")
    logits = np.asarray(choice["scores"])
    reference = np.asarray(expected["scores"])
    require(logits.shape == reference.shape and np.allclose(logits, reference, rtol=1e-5, atol=1e-6)
            and finite(choice["margin"]) and math.isclose(choice["margin"], expected["margin"], rel_tol=1e-5, abs_tol=1e-6)
            and finite(choice["cpu_ms"]) and finite(choice["cpu_us"]) and choice["cpu_us"] <= choice["cpu_ms"] * 1000,
            "startup head scores, confidence margin or measured inference time differ")


def verify_pair(observations, control, weights, contract, checkout):
    limits = json.loads(contract.read_bytes())
    require(sha(contract) == sha(FROZEN), "unrecognized discovery protocol")
    receipt = json.loads(control.read_bytes())
    require(sha(control) == limits["control_receipt_sha256"] and sha(weights) == limits["weights_sha256"],
            "different scoped control receipt or scratch controller")
    qualification = receipt["qualification"]
    scoped_release.validate_order("qwen-restart", receipt["order"])
    scoped_release.validate(qualification)
    require(receipt["adapter"] == "qwen-restart" and receipt["order"]["order_id"] == limits["parent_order"]
            and receipt["publication"]["candidate"] == limits["base_commit"] and receipt["publication"]["pr"] == 60,
            "unrelated publication")
    source = checkout / SOURCE
    actual = subprocess.check_output(["git", "rev-parse", "HEAD^{commit}"], cwd=checkout, text=True).strip()
    require(actual == limits["base_commit"] and sha(source) == limits["source_sha256"] == qualification["candidate_sha256"],
            "checkout differs from the merged, qualified source")
    site_map = post_restart_profile.sites(source)
    required = {"base_commit": actual, "parent_order": limits["parent_order"], "source_sha256": sha(source),
                "control_receipt_sha256": sha(control), "weights_sha256": sha(weights), "contract_sha256": sha(contract)}
    rows = [(p, json.loads(p.read_bytes())) for p in observations]
    rows.sort(key=lambda item: item[1]["seed"])
    require([row["seed"] for _, row in rows] == limits["seeds"], "expected exactly the two preregistered seeds")
    for _, row in rows:
        require(row["schema"] == "ironmule.post-restart-observation.v1" and all(row[k] == v for k, v in required.items())
                and row["profiler_sha256"] == sha(post_restart_profile.__file__)
                and row["selection_before"] == row["selection_after"] == "candidate",
                "changed publication, live selection, model or observer")
        check_selection(row["selection"], weights, receipt, limits)
        arm = row["arm"]
        restart_trial.validate(arm, row["seed"], expected_source=required["source_sha256"])
        restart_trial.verify_binding(arm, receipt["order"])
        bound = qualification["binding"]
        require(arm["binding"] == bound["runtime"] and arm["reused"] is True and arm["factual_first"] == []
                and arm["cuda_vendor"] == bound["vendor"] and arm["mask_bits"] == bound["mask_bits"]
                and arm["pilot_sha256"] == bound["pilot_sha256"] and finite(row["broker_elapsed_s"])
                and arm["broker_elapsed_s"] <= row["broker_elapsed_s"] + 1e-6, "unmatched native serving or timing")
        check_phases(row["phases"], limits["phases"], site_map, arm)
    return {**required, "seeds": limits["seeds"], "observation_sha256": [sha(p) for p, _ in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, action="append", required=True)
    for name in ("control", "weights", "contract", "checkout"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_pair(args.observation, args.control, args.weights, args.contract, args.checkout), sort_keys=True))


if __name__ == "__main__":
    main()
