"""Serve the registered mixed 32-client workload through the learned CPU startup policy."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from lab.pk1 import restart_release
from training import scoped_release
from training.code_author import write_once
from training.code_trial import read


def serve(args):
    started = time.monotonic()
    from training.pilot_restart_profiles import select

    receipt = scoped_release.pinned(args.state, args.receipt_sha256)
    eligible = scoped_release.selected(args.state, args.receipt_sha256)
    model_cache = args.trial / "receipt.json"
    try:
        cache_ready = (scoped_release.checksum(model_cache) == receipt["qualification"]["binding"]["cache_receipt_sha256"]
                       and 0 <= time.time() - read(model_cache)["created_s"] <= 86400)
    except (OSError, KeyError, ValueError, TypeError):
        cache_ready = False
    # The registered native runner always receives one mixed 32-client wave, one warm wave and the factual suite.
    # The HTTP adapter derives its admission from actual request arrivals.
    policy = select(args.weights, receipt, family="mixed-first-32", cache_ready=cache_ready,
                    active=eligible == "candidate", ablate_weights=args.ablate_weights,
                    expected_sha256=args.weights_sha256)
    dispatch_s = time.monotonic() - started
    if policy["profile"] == "candidate":
        work = restart_release.run(args, args.state, args.seed)
    else:
        # A single reference decision must leave a previously qualified candidate available for future work.
        # The ordinary release runner uses the same trusted receipt and defaults to reference without a selection.
        with TemporaryDirectory(dir=args.state) as scratch:
            one_request = Path(scratch)
            (one_request / "deployment").mkdir()
            (one_request / "deployment/control.json").hardlink_to(args.state / "deployment/control.json")
            work = restart_release.run(args, one_request, args.seed)
    elapsed = time.monotonic() - started
    if (work["source_sha256"] != receipt["qualification"][work["arm"] + "_sha256"]
            or work["cache_used"] != (work["arm"] == "candidate") or work["graded_correct"] != 18
            or work["cold_performance_completions"] != 32 or work["warm_performance_completions"] != 32
            or work["pilot_decisions"] <= 0 or (policy["profile"] == "reference" and work["arm"] != "reference")):
        raise ValueError("completed work differs from the registered source or controller decision")
    return {"schema": "ironmule.restart-pilot-serving.v1", "order": receipt["order"]["order_id"],
            "pr": receipt["publication"]["pr"], "control_receipt_sha256": args.receipt_sha256,
            "weights_sha256": args.weights_sha256, "seed": args.seed, "eligible_release": eligible,
            "qualified_cache_available": cache_ready, "selection": policy, "work": work,
            "broker_decision_s": dispatch_s, "broker_elapsed_through_completed_work_s": elapsed,
            "durable_selection_after": scoped_release.selected(args.state, args.receipt_sha256)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("state", "trial", "order", "checkout", "candidate-source", "gguf", "reference-lib", "table",
                  "worker-root", "weights", "out"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--seed", type=int, default=94752)
    parser.add_argument("--worker-uid", type=int, default=60000)
    parser.add_argument("--timeout", type=int, default=300, choices=range(1, 601), metavar="1..600")
    parser.add_argument("--ablate-weights", action="store_true", help="use the frozen random initialization for comparison")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("this invocation requires a fresh output path")
    write_once(args.out, serve(args))


if __name__ == "__main__":
    main()
