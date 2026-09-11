#!/usr/bin/env python3
"""What shadow matching costs the router, held to the standard `B55` set.

`B55` is why this exists. A wrapper that added nothing to the work still made a routed
single request `3.7%` slower, and direct profiling found the cause: `decide()` cost
`0.006 ms` while two subprocess probes for a telemetry field cost `17.4 ms` of a `21 ms`
overhead. A diagnostic that changes the number it reports on is not a diagnostic.

So shadow matching is measured the same way, before it is believed to be free: paired arms,
rotated order, an A/A control, and an equivalence gate fixed here rather than after the fact.
No model is loaded; this measures the decision, which is the only thing that changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ironmule import silicon_profile as sp  # noqa: E402
from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan  # noqa: E402
from ironmule.router import ExecutionRouter  # noqa: E402
from ironmule.service import Request  # noqa: E402

PROFILE_PATH = PROJECT_ROOT / "research" / "raw" / "silicon_profile_v1_20260910_corrected.json"
BLOCKS = 12
CALLS = 2000
WARMUPS = 3
EQUIVALENCE_MARGIN = 0.02
MEASURED_SOURCES = ("tools/b70_router_overhead.py", "ironmule/router.py",
                    "ironmule/silicon_profile.py")

PREREGISTRATION = {
    "experiment": "B70_router_shadow_overhead",
    "question": "what does loading and matching a silicon profile cost the router's decision",
    "standard": ("B55: decide() cost 0.006 ms and a telemetry field that shelled out cost "
                 "17.4 ms of a 21 ms wrapper overhead. Shadow matching is measured before "
                 "it is believed to be free"),
    "arms": {"A": "the router as it was, no profile",
             "B": "the same router with a loaded silicon profile and shadow matching",
             "A_aa": "arm A again under another name, as the noise floor"},
    "what_changed_after_two_failures": (
        "the first two implementations computed the shadow inside decide() and cost 78 and "
        "63 per cent of it, against a 2 per cent gate. Both failures are recorded in "
        "B70_router_overhead_20260910.json and ..._attempt2.json. The diagnostic now runs "
        "once per dispatch in the record builder, after the route exists and after every "
        "cohort has been served, so decide() is byte for byte what it was. The B over A "
        "ratio below is therefore 1.0 BY CONSTRUCTION and is not an achievement; the "
        "number that means something is annotate_ns, the absolute cost of one annotation "
        "against the dispatch it annotates"),
    "scenarios": ["1 request", "2 requests", "4 requests", "3 session requests",
                  "match", "mismatch", "unknown hardware", "invalid profile"],
    "design": (f"{BLOCKS} blocks, arm order rotated per block, {CALLS} decide() calls timed "
               f"per arm per block, {WARMUPS} warmup passes, per-block ratios, "
               "10000-resample bootstrap"),
    "equivalence_gate": (f"the 95 per cent interval of B over A must lie entirely below "
                         f"{1 + EQUIVALENCE_MARGIN}. An interval reaching above it fails, "
                         "whatever the median says"),
    "no_claim_from_a_median": "a median alone decides nothing here",
    "nothing_is_activated": "this measures a diagnostic field and changes no route",
}


def _requests(count: int, session: bool = False, max_tokens: int = 32):
    prefix = [1, 2, 3]
    out = []
    for index in range(count):
        plan = ReusableSessionPlan(prefix, name="s") if session else StrictOneShotPlan()
        ids = prefix + [10 + index] if session else [10 + index] * 8
        out.append(Request(prompt_ids=ids, max_tokens=max_tokens, plan=plan,
                           objective="latency"))
    return out


def _context(profile_body, **overrides):
    row = profile_body["parameters"][0]
    base = sp.RuntimeContext(
        hardware_fingerprint=row["hardware_fingerprint"],
        gpu_architecture=row["gpu_architecture"], mlx=row["mlx"], mlx_lm=row["mlx_lm"],
        model_id=row["model_id"], model_identity_sha256=row["model_identity_sha256"],
        model_revision=row["model_revision"],
        quantization_bits=row["quantization"]["bits"],
        quantization_group_size=row["quantization"]["group_size"],
        hidden_size=row["shape"]["k"], projection_widths=tuple(row["shape"]["admitted_n"]),
        workload_class="", objective="latency", decode_width=1)
    return replace(base, **overrides)


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(statistics.median([rng.choice(ratios) for _ in ratios])
                   for _ in range(resamples))
    return {"median": statistics.median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios), "ratios": ratios}


def source_binding() -> dict:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    body = json.loads(PROFILE_PATH.read_text())
    profile = sp.load(body)
    empty = sp.load({"schema": sp.SCHEMA, "generated_from": "empty", "parameters": [],
                     "digest": sp.canonical_digest([])})

    def router(silicon=None, context=None):
        return ExecutionRouter({"knobs": {}}, identity_sha256="id", fingerprint="fp",
                               mlx="0.32.0", mlx_lm="0.31.3",
                               silicon_profile=silicon, silicon_context=context)

    matched = _context(body)
    foreign = _context(body, hardware_fingerprint="somewhere else")
    scenarios = {
        "one_request_match": (_requests(1), "latency", profile, matched),
        "two_requests_match": (_requests(2), "throughput", profile, matched),
        "four_requests_match": (_requests(4), "throughput", profile, matched),
        "three_session_requests_match": (_requests(3, session=True), "latency", profile, matched),
        "one_request_mismatch": (_requests(1), "latency", profile, foreign),
        "one_request_unknown_hardware": (_requests(1), "latency", profile,
                                         sp.RuntimeContext()),
        "one_request_empty_profile": (_requests(1), "latency", empty, matched),
        "one_request_invalid_profile": (_requests(1), "latency",
                                        sp.load_or_none({"schema": "wrong"}), matched),
    }

    results, shadow_seen = {}, {}
    for name, (requests, objective, silicon, context) in scenarios.items():
        arms = {"A": router(), "B": router(silicon, context), "A_aa": router()}
        shadow = arms["B"].annotate(arms["B"].decide(requests, objective))
        shadow_seen[name] = {
            "route": shadow.route, "workload_class": shadow.workload_class,
            "silicon_match": (shadow.silicon or {}).get("silicon_match"),
            "parameter": (shadow.silicon or {}).get("silicon_parameter_id"),
            "route_identical_to_arm_A": shadow.route == arms["A"].decide(requests, objective).route,
        }
        for arm in arms.values():
            for _ in range(WARMUPS):
                for _ in range(CALLS // 10):
                    arm.decide(requests, objective)
        per_block = []
        names = list(arms)
        for block in range(BLOCKS):
            shift = block % len(names)
            order = names[shift:] + names[:shift]
            timings = {}
            for arm in order:
                decide = arms[arm].decide
                start = time.perf_counter_ns()
                for _ in range(CALLS):
                    decide(requests, objective)
                timings[arm] = (time.perf_counter_ns() - start) / CALLS
            # The annotation itself, measured on its own: it runs once per dispatch, not
            # once per decide, so it is reported in absolute nanoseconds and not as a ratio.
            annotate = arms["B"].annotate
            settled = arms["B"].decide(requests, objective)
            start = time.perf_counter_ns()
            for _ in range(CALLS):
                annotate(settled)
            timings["annotate"] = (time.perf_counter_ns() - start) / CALLS
            per_block.append(timings)
        results[name] = {
            "ns_per_decide_A": statistics.median(b["A"] for b in per_block),
            "ns_per_decide_B": statistics.median(b["B"] for b in per_block),
            "B_over_A": bootstrap([b["B"] / b["A"] for b in per_block]),
            "A_aa_over_A": bootstrap([b["A_aa"] / b["A"] for b in per_block]),
            "annotate_ns": statistics.median(b["annotate"] for b in per_block),
            "per_block": per_block,
        }
        row = results[name]["B_over_A"]
        print(f'  {name:32} A={results[name]["ns_per_decide_A"]/1000:6.3f} us  '
              f'B={results[name]["ns_per_decide_B"]/1000:6.3f} us  '
              f'ratio={row["median"]:.4f} ci=[{row["ci_low"]:.4f}; {row["ci_high"]:.4f}]',
              flush=True)

    failures = [name for name, row in results.items()
                if row["B_over_A"]["ci_high"] is None
                or row["B_over_A"]["ci_high"] > 1 + EQUIVALENCE_MARGIN]
    route_changes = [name for name, row in shadow_seen.items()
                     if not row["route_identical_to_arm_A"]]

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "profile_used": str(PROFILE_PATH.relative_to(PROJECT_ROOT)),
        "profile_digest": profile.digest,
        "results": results,
        "shadow_observations": shadow_seen,
        "scenarios_failing_the_gate": failures,
        "scenarios_where_the_route_changed": route_changes,
        "verdict": "PASS" if not failures and not route_changes else "FAIL",
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": record["verdict"], "failing": failures,
                      "route_changes": route_changes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
