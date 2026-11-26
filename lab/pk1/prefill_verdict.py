"""Decide whether one isolated Lt screen provides enough headroom for a much dearer code experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from training.code_author import digest, write_once


def decide(plan, receipt):
    if (receipt["request_id"] != digest(plan) or receipt.get("status") != "completed" or
            plan["targets"] != [s["shape"] for s in receipt["targets"]]):
        raise ValueError("unfinished or foreign information experiment")
    chosen = []
    for target in receipt["targets"]:
        if target["return_code"] != 0 or target["events"][0].get("kind") != "info":
            raise ValueError("incomplete per-shape receipt")
        info, *events = target["events"]
        if {e["index"] for e in events} != set(range(info["heuristics"])):
            raise ValueError("missing or unexpected heuristic")
        candidates = []
        for index in range(info["heuristics"]):
            rows = [e for e in events if e["index"] == index]
            if len(rows) == 1 and rows[0]["status"] in {"ineligible", "declined", "changed_output"}:
                continue
            if (any(r["status"] != "paired" for r in rows) or [r["pair"] for r in rows] != list(range(4))
                    or len({r["algo_id"] for r in rows}) != 1):
                raise ValueError("incomplete micro-pair series")
            if any(not math.isfinite(v) or v <= 0 for r in rows for v in (r["reference_ms"], r["candidate_ms"])):
                raise ValueError("invalid micro-timing")
            deltas = [(r["reference_ms"] - r["candidate_ms"]) * target["shape"]["calls"] for r in rows]
            candidates.append({"index": index, "algo_id": rows[0]["algo_id"], "pair_projections_ms": deltas,
                               "projected_wave_saving_ms": median(deltas)})
        best = max(candidates, key=lambda r: r["projected_wave_saving_ms"], default=None)
        chosen.append({"family": target["shape"]["family"], "bitwise_candidate_count": len(candidates),
                       "heuristics": info["heuristics"], "best": best})
    projected = sum(max(0, s["best"]["projected_wave_saving_ms"]) for s in chosen if s["best"])
    # A wave with the observed prefill must also pay decode. This lower time bound deliberately favors a new trial.
    minimum = 1000 * plan["profiled_prefill_s"] * (1 - 1 / 1.02)
    return {"schema": "ironmule.prefill-headroom.v1", "kind": "planning_extrapolation", "request_id": receipt["request_id"],
            "receipt_digest": digest(receipt), "throughput_requirement": 1.02, "baseline_prefill_s": plan["profiled_prefill_s"],
            "minimum_ms_for_2pct_if_decode_were_free": minimum, "optimistic_best_sum_ms": projected,
            "per_shape": chosen, "outcome": "defer source order" if projected < minimum else "eligible for independent source trial",
            "uncertainty": "Isolated synthetic inputs and selected four-pair medians. The mixed serving workload, "
            "library integration costs, quality and a new source variant have no measured benefit here."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--plan", "--receipt", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
    write_once(args.out, decide(json.loads(args.plan.read_text()), json.loads(args.receipt.read_text())))


if __name__ == "__main__":
    main()
