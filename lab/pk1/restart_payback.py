"""Replay an A100 repeat-start comparison and estimate its compute-time break-even."""
from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lab.pk1 import restart_order, restart_trial
from training import restart_evidence
from training.code_author import read_request, read_response
from training.code_trial import read


@contextmanager
def measured_sources(folder):
    """Rebuild the authored method and independently replay the complete immutable hardware gate."""
    archive = restart_evidence.manifest(folder)
    if archive.get("qualified") is not True:
        raise ValueError("A completed, passing hardware qualification is required")
    order = read(folder / "order.json")
    request_id, request = read_request(folder)
    original = (folder / "reference-source.txt").read_text()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        base = root / restart_order.PATH
        base.parent.mkdir()
        base.write_text(original)
        proposal = root / "candidate.py"
        proposal.write_bytes(restart_evidence.candidate_bytes(original, request, order, read_response(folder, request_id)))
        check = SimpleNamespace(order=folder / "order.json", checkout=root, candidate_source=proposal, trial=folder)
        restart_trial.qualification(check)
        yield check


def replay(folder, journal=None):
    with measured_sources(folder) as check:
        order = read(check.order)
    seconds = read(folder / "result.json")["elapsed_s"]
    report = {"order": order["order_id"], "scope": "qualified repeat-start engine on the measured A100",
              "qualification_elapsed_s": seconds, "workloads": {}}
    lease_hours = booked = price = None
    if journal is not None:
        costs = read(journal)
        leases = costs["leases"]
        snapshot = read(folder / "cost-during-qualification.json")
        if (snapshot["order_id"] != order["order_id"] or costs["confirmed_owned_pods_remaining"] != 0
                or not leases or any(not lease["verified_absent"] for lease in leases)
                or not any("A100" in lease["gpu"] for lease in leases)):
            raise ValueError("requires a closed campaign journal covering the original A100")
        lease_hours = sum(lease["hours"] for lease in leases)
        booked = costs["booked_with_tail_usd"]
        price = snapshot["quoted_compute_usd_per_h"] + snapshot["disk_rate_estimate_usd_per_h"]
        if not all(math.isfinite(x) and x > 0 for x in (lease_hours, booked, price)):
            raise ValueError("incomplete final billing figures")
        report.update(whole_campaign_leases=len(leases), whole_campaign_lease_hours=lease_hours, whole_campaign_booked_usd_with_tail=booked,
                      estimated_marginal_usd_per_billable_hour=price)
    for label, prefix, count in (("32_mixed_completions", "", restart_trial.PAIRS), ("18_correct_facts", "facts-", 2)):
        first = []
        lifetime = []
        for index in range(count):
            ref, cand = [read(folder / f"{prefix}{arm}-{index}.json") for arm in ("ref", "cand")]
            if prefix:
                first.append(ref["factual_first"][0]["completed_s"] - cand["factual_first"][0]["completed_s"])
            else:
                first.append(ref["cold_s"] - cand["cold_s"])
            lifetime.append(ref["broker_elapsed_s"] - cand["broker_elapsed_s"])
        # This scenario prices only time common to the paired worker-lifetime and delivery savings.
        saving = min(median(first), median(lifetime))
        row = {"seed_pairs": count, "median_first_work_s_saved": median(first),
               "median_entire_worker_s_saved": median(lifetime), "projected_s_per_repeat": saving,
               "repeats_to_recover_qualification_time": math.ceil(seconds / saving) if saving > 0 else None}
        if journal is not None and saving > 0:
            row.update(repeats_to_recover_entire_lease_time=math.ceil(lease_hours * 3600 / saving),
                       repeats_to_recover_booked_cost=math.ceil(booked * 3600 / (price * saving)))
        report["workloads"][label] = row
    report["interpretation"] = ("Counts are projections at the observed medians for repeated, qualified cache hits. "
                                "Dollar payback requires each saved second to shorten a billable lease. "
                                "A continuously billed worker can free capacity without reducing its invoice.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="portable, content-addressed order directory")
    parser.add_argument("--journal", type=Path, help="final, provider-reconciled campaign costs")
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.journal), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
