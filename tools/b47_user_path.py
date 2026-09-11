#!/usr/bin/env python3
"""The ordinary user path, then a non-regression check on the opt-in surface.

Part one walks what a local user actually does: load with the default, read status, run
one request, run a pair, switch the option off again, and reload to confirm the default
is still off. Correctness is checked against independent library runs, not against the
paired path's own output.

Part two asks whether the new surface costs anything. The research build measured in
`B46` and the opt-in mode do the same work; if wrapping it in a service mode added work
to the request path, this is where it shows. Preregistered: 8 rotated blocks, A/A
control, the same gate, one run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b52_automatic_selection import write_once  # noqa: E402

from ironmule import paired_research as pr  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.runtime import BASELINE  # noqa: E402
from ironmule.service import (  # noqa: E402
    InteractiveMode,
    PairedThroughputMode,
    Request,
    Runtime,
    ThroughputMode,
    paired_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATE = 0.90
PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Describe how a key value cache grows during autoregressive decoding.",
)


class ResearchMode:
    """The `B46` build, named so the non-regression check can compare against it."""

    name = "paired_research_build"

    def __init__(self) -> None:
        self.share = True

    def executor(self, backend, telemetry):
        return pr.PairedGroupedExecutor(backend, telemetry, share=True)


def _vm() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return {k: values[k] for k in ("Pageins", "Pageouts", "Swapins", "Swapouts") if k in values}


def _fingerprint(results) -> list[dict]:
    return [{"tokens": list(r.tokens), "stop_reason": r.stop_reason,
             "text_sha256": hashlib.sha256(r.text.encode()).hexdigest()} for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    walk: dict = {"default_knob_off_at_import": BASELINE.k3840_matvec is False}

    # -- the ordinary user path ------------------------------------------------
    runtime = Runtime.load(args.model, use_tuned_profile=False)
    try:
        walk["load_default_mode"] = runtime.mode.name
        walk["status_when_disabled"] = paired_status(runtime.mode)

        single = [Request(prompt_ids=runtime.encode(PROMPTS[0]), max_tokens=8,
                          plan=StrictOneShotPlan())]
        pair = [Request(prompt_ids=runtime.encode(p), max_tokens=args.max_tokens,
                        plan=StrictOneShotPlan()) for p in PROMPTS]

        runtime.mode = InteractiveMode()
        truth_single = _fingerprint(runtime.serve(single))
        truth_pair = _fingerprint(runtime.serve(pair))

        opt_in = PairedThroughputMode()
        runtime.mode = opt_in
        got_single = _fingerprint(runtime.serve(single))
        walk["status_after_single"] = paired_status(opt_in)
        got_pair = _fingerprint(runtime.serve(pair))
        walk["status_after_pair"] = paired_status(opt_in)

        runtime.mode = ThroughputMode()
        walk["status_after_switching_off"] = paired_status(runtime.mode)
        after_off = _fingerprint(runtime.serve(pair))

        walk["single_identical"] = truth_single[0] == got_single[0]
        walk["pair_identical"] = [a == b for a, b in zip(truth_pair, got_pair)]
        walk["after_off_identical"] = [a == b for a, b in zip(truth_pair, after_off)]
    finally:
        runtime.close()

    reloaded = Runtime.load(args.model, use_tuned_profile=False)
    try:
        walk["reload_default_mode"] = reloaded.mode.name
        walk["reload_status_disabled"] = paired_status(reloaded.mode)
        walk["reload_default_is_off"] = paired_status(reloaded.mode)["enabled"] is False
    finally:
        reloaded.close()

    walk_clean = (
        walk["single_identical"]
        and all(walk["pair_identical"])
        and all(walk["after_off_identical"])
        and walk["reload_default_is_off"]
        and walk["status_when_disabled"]["enabled"] is False
        and walk["status_after_pair"]["paired_steps"] > 0
        and walk["status_after_single"]["paired_steps"] == 0
    )
    print(json.dumps({"user_path_clean": walk_clean}, sort_keys=True))

    # -- non-regression of the surface itself ---------------------------------
    runtime = Runtime.load(args.model, mode=ThroughputMode(), use_tuned_profile=False)
    arms_ns: dict[str, list[float]] = {}
    paging: dict = {}
    try:
        modes = {
            "A_throughput": ThroughputMode(),
            "A_throughput_aa": ThroughputMode(),
            "R_research_build": ResearchMode(),
            "O_opt_in_mode": PairedThroughputMode(),
        }

        def one_round(mode):
            runtime.mode = mode
            requests = [Request(prompt_ids=runtime.encode(p), max_tokens=args.max_tokens,
                                plan=StrictOneShotPlan()) for p in PROMPTS]
            began = time.perf_counter_ns()
            runtime.serve(requests)
            return time.perf_counter_ns() - began

        for mode in modes.values():
            one_round(mode)

        before = _vm()
        arms_ns = {name: [] for name in modes}
        order = list(modes)
        for block in range(args.blocks):
            rotation = order[block % len(order):] + order[: block % len(order)]
            for name in rotation:
                arms_ns[name].append(one_round(modes[name]))
        after = _vm()
        paging = {key: after.get(key, 0) - value for key, value in before.items()}
    finally:
        runtime.close()

    def interval(numerator, denominator):
        ratios = [n / d for n, d in zip(arms_ns[numerator], arms_ns[denominator])]
        rng = random.Random(20260909)
        draws = sorted(median([ratios[rng.randrange(len(ratios))] for _ in ratios])
                       for _ in range(10000))
        return {"median": median(ratios), "min": min(ratios), "max": max(ratios),
                "ci95_low": draws[int(0.025 * len(draws))],
                "ci95_high": draws[int(0.975 * len(draws)) - 1],
                "blocks_at_or_under_gate": sum(1 for v in ratios if v <= GATE)}

    opt_over_a = interval("O_opt_in_mode", "A_throughput")
    research_over_a = interval("R_research_build", "A_throughput")
    opt_over_research = interval("O_opt_in_mode", "R_research_build")
    aa = interval("A_throughput_aa", "A_throughput")
    aa_passes = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
    keeps_gate = opt_over_a["ci95_high"] <= GATE
    no_surface_cost = opt_over_research["ci95_low"] <= 1.0 <= opt_over_research["ci95_high"]
    decision = ("OPT-IN READY" if walk_clean and keeps_gate and aa_passes
                and paging.get("Swapouts", 0) == 0
                else ("BLOCKED" if paging.get("Swapouts", 0) else "NOT RELEASED"))

    record = {
        "schema": "ironmule.paired_opt_in_release.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "release_check",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the opt-in surface work for a user and cost nothing in the request path",
        "decision": decision,
        "model": args.model,
        "gate": GATE,
        "preregistered": {
            "blocks": args.blocks, "max_tokens": args.max_tokens,
            "arms": ["A_throughput", "A_throughput_aa", "R_research_build", "O_opt_in_mode"],
            "order": "rotated so every arm meets every position",
            "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
            "success_rule": (
                "the user walk is clean, the opt-in arm still clears the 0.90 gate against "
                "A, and the A/A control contains 1.0"
            ),
            "budget": "one run, no extension",
        },
        "user_path": walk,
        "user_path_clean": walk_clean,
        "arm_wall_ns_median": {name: median(values) for name, values in arms_ns.items()},
        "arm_wall_ns": arms_ns,
        "opt_in_over_A": opt_over_a,
        "research_build_over_A": research_over_a,
        "opt_in_over_research_build": opt_over_research,
        "surface_costs_nothing_measurable": no_surface_cost,
        "aa_null_control": aa,
        "aa_null_control_passes": aa_passes,
        "keeps_gate": keeps_gate,
        "paging_during_run": paging,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "limits": [
            "two fixed prompts, greedy, one machine, one MLX build",
            "this checks the surface, it does not re-prove the B46 result",
            "cancellation remains untested; the shipped Request has no cancel handle",
        ],
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({key: value for key, value in record.items()
                      if key in ("decision", "user_path_clean", "keeps_gate",
                                 "opt_in_over_A", "opt_in_over_research_build",
                                 "surface_costs_nothing_measurable",
                                 "aa_null_control_passes", "paging_during_run")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
