#!/usr/bin/env python3
"""What a latency request that arrives during a throughput group can actually be given.

`B56` attempt 2 found the one case its criteria cannot pass: a `latency` request declared
to arrive after a `throughput` cohort has started waits for that cohort, and its
completion latency is `2.10x` the same request with no foreign work around it. The
hand-written arm produced the same number, so it is not a routing defect.

This measures the only alternatives an engine with one device and no preemption has, on
the same arrival pattern (`0, 60, 0, 60 ms`, two throughput then two latency):

    A_cohorts       what the router does: the throughput cohort runs, then the latency
                    cohort, each on its own qualified path
    B_all_grouped   the late requests join the group instead of waiting -- one grouped
                    dispatch, every request marked throughput
    C_separate      the late requests are their own dispatch, which is what a server
                    that receives them 60 ms later actually does

`C_separate` is the reference: it is the same two requests with nothing foreign around
them. The question is which of `A` and `B` is closer to it.

Not a preregistered gate. It decides nothing and no threshold depends on it; it exists so
the `B56` report recommends a schedule from a measurement instead of from an argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

from ironmule.hw import fingerprint, swap_used_bytes  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.router import AppleRuntime, ExecutionRouter  # noqa: E402
from ironmule.service import InteractiveMode, Request, Runtime, ThroughputMode  # noqa: E402
from ironmule.tune import DEFAULT_MODEL, gpu_busy, load_profile, resolve_local_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASURED_SOURCES = ("ironmule/router.py", "ironmule/service.py", "ironmule/executor.py",
                    "tools/b56_staggered_alternatives.py")
MAX_TOKENS = 32
PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
    "Summarise what a tokenizer does before a model sees any text.",
    "State one reason a quantised model reads fewer bytes per token.",
)
# (objective, arrival_ms, prompt_index) -- the B56 W6 pattern, unchanged.
PATTERN = (("throughput", 0.0, 0), ("latency", 60.0, 1),
           ("throughput", 0.0, 2), ("latency", 60.0, 3))
LATE = (1, 3)


def source_binding() -> dict[str, object]:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def build(prompt_ids, override=None):
    return [Request(prompt_ids=list(prompt_ids[index]), max_tokens=MAX_TOKENS,
                    plan=StrictOneShotPlan(), objective=override or objective,
                    arrival_ms=arrival)
            for objective, arrival, index in PATTERN]


def late_latency_ms(results, telemetry, requests) -> list[float]:
    rows = {row["rid"]: row for row in telemetry["per_request"]}
    return [rows[requests[position].rid]["latency_ms"] - PATTERN[position][1]
            for position in LATE]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    resolved = resolve_local_model(args.model)
    profile = load_profile(args.model, model_identity=resolved.identity)
    if profile is None:
        raise SystemExit("no tuned profile for this machine and model")

    swap_start = swap_used_bytes()
    runtime = Runtime.load(args.model, mode=InteractiveMode())
    router = ExecutionRouter(profile, identity_sha256=resolved.identity.identity_sha256,
                             fingerprint=fingerprint(), mlx=mx.__version__,
                             mlx_lm=mlx_lm.__version__)
    routed = AppleRuntime(runtime, router)
    prompt_ids = [runtime.encode(prompt) for prompt in PROMPTS]

    def arm_cohorts():
        requests = build(prompt_ids)
        results = routed.serve(requests)
        return late_latency_ms(results, routed.last_decision["telemetry"], requests)

    def arm_all_grouped():
        requests = build(prompt_ids, override="throughput")
        results = routed.serve(requests)
        return late_latency_ms(results, routed.last_decision["telemetry"], requests)

    def arm_separate():
        """The late pair as their own dispatch, with nothing foreign around them."""
        requests = build(prompt_ids)
        late = [requests[position] for position in LATE]
        results = routed.serve(late)
        rows = {row["rid"]: row for row in routed.last_decision["telemetry"]["per_request"]}
        return [rows[request.rid]["latency_ms"] - request.arrival_ms for request in late]

    arms = {"A_cohorts": arm_cohorts, "B_all_grouped": arm_all_grouped,
            "C_separate": arm_separate}
    order = list(arms)
    for run in arms.values():                          # warm-up, discarded
        run()

    blocks = []
    try:
        for block in range(args.blocks):
            shift = block % len(order)
            sequence = order[shift:] + order[:shift]
            blocks.append({"block": block, "arm_order": list(sequence),
                           "late_latency_ms": {arm: arms[arm]() for arm in sequence}})
            print(f"block {block} done")
    finally:
        runtime.close()

    medians = {arm: median([median(record["late_latency_ms"][arm]) for record in blocks])
               for arm in order}
    record = {
        "experiment": "B56_staggered_alternatives",
        "question": ("what a latency request arriving during a throughput group can be "
                     "given, on one device without preemption"),
        "gate": "none; this measurement decides nothing and no threshold depends on it",
        "arrival_pattern_ms": [arrival for _o, arrival, _i in PATTERN],
        "source_binding": source_binding(),
        "environment": {
            "platform": platform.platform(), "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__, "hardware_fingerprint": fingerprint(),
            "model_id": resolved.identity.model_id,
            "model_revision": resolved.identity.revision,
            "model_identity_sha256": resolved.identity.identity_sha256,
        },
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "blocks": len(blocks),
        "late_latency_p50_ms": medians,
        "over_separate": {arm: medians[arm] / medians["C_separate"] for arm in order},
        "resources": {"swap_used_bytes_start": swap_start,
                      "swap_used_bytes_end": swap_used_bytes()},
        "raw_blocks": blocks,
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"late_latency_p50_ms": medians,
                      "over_separate": record["over_separate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
