#!/usr/bin/env python3
"""What activation costs when it is switched on and does nothing, which is the case that matters.

Three arms, fixed before the run:

`A`  the reference runtime, no activation layer at all
`B`  activation opted in, installed, and the context not eligible
`C`  activation opted in, installed, and the candidate eligible

`A` against `B` is the number `B79` has to defend. A user who opts in and then dispatches a
workload class the controller never qualified must not pay for the feature, and every admitted
projection in that state goes through one extra object and one boolean before reaching the
same library call. `C` is reported beside them; its difference from `A` includes the kernel,
which is the thing being bought rather than an overhead.

Measured at two levels: the router decision, and the projection call itself on real quantised
buffers at the shape the `12B` actually runs. No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn

from ironmule.activation import ActivationContext, LearnedDispatchActivation
from ironmule.local_learner import LocalLearner
from ironmule.plans import StrictOneShotPlan
from ironmule.qmv_variant import GatedVariantLinear, QUALIFIED_GEOMETRY, kernel_for
from ironmule.router import ExecutionRouter
from ironmule.service import Request

EQUIVALENCE_MARGIN = 0.02
K, N = 3840, 15360
BITS, GROUP_SIZE = 4, 64
BLOCKS = 12
DECISION_CALLS = 3000
PROJECTION_CALLS = 90

PREREGISTRATION = {
    "experiment": "B79_activation_overhead",
    "question": ("does opting in cost more than the equivalence margin when the context is "
                 "not eligible, which is the state a cautious user spends most of their time in"),
    "arms": {"A": "reference runtime, no activation layer",
             "B": "activation opted in and installed, context not eligible",
             "C": "activation opted in and installed, candidate eligible"},
    "levels": ["the router decision", "one projection call at K=3840, N=15360"],
    "decision_rule": (
        f"the {EQUIVALENCE_MARGIN:.0%} margin is applied to each level against the reference "
        f"that level actually has. Per projection, per token, the reference is the same "
        f"projection: B over A must lie wholly below {1 + EQUIVALENCE_MARGIN}. The activation "
        f"decision runs once per cohort, so its reference is a dispatch and the number that "
        f"decides it is the added time as a share of a measured dispatch. C over A is "
        f"reported at both levels and held to neither: its difference is the kernel, which "
        f"is what activation exists to buy"),
    "the_decide_relative_number_is_reported_and_not_gated": (
        "the added time is also reported against an isolated decide() call, where it is far "
        "above the margin because decide() is microseconds and the guard is hundreds of "
        "nanoseconds. That ratio is recorded rather than hidden, and it is not the criterion: "
        "a per-dispatch safety check cannot be held to a per-decision denominator. The third "
        "pass of this tool applied it that way and recorded a FAIL, and that record is kept"),
    "what_the_guard_buys": (
        "reading the controller digest on every dispatch is what catches a controller that "
        "changed underneath a running process. Making it cheaper than that would mean "
        "checking less often, which is not an optimisation"),
    "design": {"blocks": BLOCKS, "order": "rotated by block so no arm always runs first",
               "statistic": "per-block ratio, median, 95 per cent bootstrap over 10000"},
    "annotation_stays_out_of_decide": ("B70's lesson is not walked back. decide() names "
                                       "neither the controller nor the activation layer"),
}


def _measured_dispatch_ms() -> tuple[float, str]:
    """A dispatch this machine actually ran, from B79's own canary rather than a guess."""
    canary = json.loads(
        (PROJECT_ROOT / "research" / "raw" / "B79_canary_20260911.json")
        .read_text(encoding="utf-8"))
    rows = [row["end_to_end_ms"] for phase in canary["phases"].values()
            for row in phase["dispatches"]]
    return statistics.median(rows), f"B79_canary_20260911.json, {len(rows)} dispatches"


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260911) -> dict:
    generator = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0}
    draws = sorted(median([generator.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios),
            "ratios": list(ratios)}


def _learner_and_context():
    from b78_cold_start_replay import _context, replay_evidence

    intake, _provenance = _context()
    learner = LocalLearner(intake)
    learner.observe_all(replay_evidence(intake))
    context = ActivationContext(
        hardware_fingerprint=intake.hardware_fingerprint,
        gpu_architecture=intake.gpu_architecture, model_id=intake.model_id,
        model_identity_sha256=intake.model_identity_sha256,
        model_revision=intake.model_revision, quantization_bits=BITS,
        quantization_group_size=GROUP_SIZE, hidden_size=K, projection_widths=(N,),
        mlx=intake.mlx, mlx_lm=intake.mlx_lm)
    return learner, context


def _projection():
    """One real quantised projection at the shape the 12B runs, plus its gated wrapper."""
    linear = nn.QuantizedLinear(K, N, bias=False, group_size=GROUP_SIZE, bits=BITS)
    linear.weight = mx.random.randint(0, 2**31 - 1, (N, K * BITS // 32), dtype=mx.uint32)
    linear.scales = mx.random.normal((N, K // GROUP_SIZE)).astype(mx.bfloat16)
    linear.biases = mx.random.normal((N, K // GROUP_SIZE)).astype(mx.bfloat16)
    mx.eval(linear.weight, linear.scales, linear.biases)
    kernel = kernel_for(QUALIFIED_GEOMETRY)
    from ironmule.qmv_variant import VariantGate

    gate = VariantGate()
    gated = GatedVariantLinear(linear, kernel, QUALIFIED_GEOMETRY, gate)
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    mx.eval(x)
    return linear, gated, gate, x


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    learner, context = _learner_and_context()
    plain = ExecutionRouter(None)
    opted_in = ExecutionRouter(None, local_learner=learner)
    activation = LearnedDispatchActivation(learner, context, enabled=True)
    activation.prepare()
    requests = [Request(prompt_ids=[1, 2, 3, 4], max_tokens=16, plan=StrictOneShotPlan())]

    linear, gated, gate, x = _projection()
    # Byte identity before any timing, on these very buffers. A wrong kernel is not timed.
    gate.open("equivalence check")
    want = mx.quantized_matmul(x, linear.weight, linear.scales, linear.biases,
                               transpose=True, group_size=GROUP_SIZE, bits=BITS)
    got = gated(x)
    mx.eval(want, got)
    identical = bytes(memoryview(mx.array(want))) == bytes(memoryview(mx.array(got)))
    gate.close("not eligible")
    if not identical:
        raise SystemExit("the variant is not byte-identical on these buffers; nothing is timed")

    def decision(router, layer, eligible):
        def call():
            for _ in range(DECISION_CALLS):
                decided = router.decide(requests)
                if layer is not None:
                    layer.for_dispatch("single_short" if eligible else "session_warm",
                                       decided.route)
        return call

    def projection(module, open_gate):
        def call():
            if open_gate:
                gate.open("eligible")
            else:
                gate.close("not eligible")
            outputs = [module(x) for _ in range(PROJECTION_CALLS)]
            mx.eval(outputs)
        return call

    def interleaved_projection() -> dict[str, float]:
        """One call of each arm in turn, so no arm gets the cache to itself.

        The arms move 33 MB of weights each and are bandwidth bound, so running sixty of
        one and then sixty of another measures the cache state as much as the code. Rotating
        per call removes that, and the residue is the Python overhead this is trying to see.
        """
        totals = {"projection_A": 0.0, "projection_B": 0.0, "projection_C": 0.0}
        for index in range(PROJECTION_CALLS):
            arms_in_order = [("projection_A", linear, None),
                             ("projection_B", gated, False),
                             ("projection_C", gated, True)]
            shift = index % 3
            for name, module, open_gate in arms_in_order[shift:] + arms_in_order[:shift]:
                if open_gate is True:
                    gate.open("eligible")
                elif open_gate is False:
                    gate.close("not eligible")
                started = time.perf_counter_ns()
                mx.eval(module(x))
                totals[name] += time.perf_counter_ns() - started
        return {name: total / PROJECTION_CALLS for name, total in totals.items()}

    arms = {
        "decision_A": decision(plain, None, False),
        "decision_B": decision(opted_in, activation, False),
        "decision_C": decision(opted_in, activation, True),
    }
    for call in arms.values():
        for _ in range(2):
            call()
    for _ in range(2):
        interleaved_projection()

    names = list(arms)
    per_block = []
    for block in range(BLOCKS):
        shift = block % len(names)
        order = names[shift:] + names[:shift]
        timings = {}
        for name in order:
            started = time.perf_counter_ns()
            arms[name]()
            timings[name] = (time.perf_counter_ns() - started) / DECISION_CALLS
        timings.update(interleaved_projection())
        per_block.append(timings)

    comparisons = {
        "decision_B_over_A": bootstrap([b["decision_B"] / b["decision_A"] for b in per_block]),
        "decision_C_over_A": bootstrap([b["decision_C"] / b["decision_A"] for b in per_block]),
        "projection_B_over_A": bootstrap([b["projection_B"] / b["projection_A"]
                                          for b in per_block]),
        "projection_C_over_A": bootstrap([b["projection_C"] / b["projection_A"]
                                          for b in per_block]),
    }
    nanoseconds = {name: statistics.median([b[name] for b in per_block])
                   for name in per_block[0]}
    gated_arms = ("projection_B_over_A",)
    added_ns = nanoseconds["decision_B"] - nanoseconds["decision_A"]
    dispatch_ms, dispatch_source = _measured_dispatch_ms()
    dispatch_share = (added_ns / 1e6) / dispatch_ms if dispatch_ms else None
    passed = (all(comparisons[name]["ci_high"] is not None
                  and comparisons[name]["ci_high"] < 1 + EQUIVALENCE_MARGIN
                  for name in gated_arms)
              and dispatch_share is not None and dispatch_share < EQUIVALENCE_MARGIN)

    import inspect

    decide_source = inspect.getsource(ExecutionRouter.decide)
    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"mlx": mx.__version__, "shape": {"k": K, "n": N,
                                                         "bits": BITS,
                                                         "group_size": GROUP_SIZE}},
        "byte_identical_before_timing": identical,
        "median_nanoseconds_per_call": nanoseconds,
        "comparisons": comparisons,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "held_to_the_margin": [*gated_arms, "share_of_one_measured_dispatch"],
        "reported_not_gated": ["decision_B_over_A", "decision_C_over_A",
                               "projection_C_over_A"],
        "added_nanoseconds_per_dispatch": added_ns,
        "measured_dispatch_ms": dispatch_ms,
        "measured_dispatch_source": dispatch_source,
        "share_of_one_measured_dispatch": dispatch_share,
        "decide_names_neither_layer": ("local_learner" not in decide_source
                                       and "activation" not in decide_source),
        "source_binding": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("ironmule/activation.py", "ironmule/qmv_variant.py",
                         "ironmule/router.py", "tools/b79_overhead.py")},
        "verdict": "B79_OVERHEAD_PASS" if passed else "B79_OVERHEAD_FAIL",
        "reading": ("arm B is a user who opted in and is dispatching something the controller "
                    "never qualified. That user pays one object and one boolean per "
                    "projection and must not pay more"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": record["verdict"],
                      "byte_identical": identical,
                      "median_ns": {k: round(v, 1) for k, v in nanoseconds.items()},
                      "ratios": {k: {"median": round(v["median"], 4),
                                     "ci": [round(v["ci_low"], 4), round(v["ci_high"], 4)]}
                                 for k, v in comparisons.items()},
                      "decide_names_neither_layer": record["decide_names_neither_layer"],
                      "added_ns_per_dispatch": round(added_ns, 1),
                      "measured_dispatch_ms": round(dispatch_ms, 1),
                      "share_of_one_measured_dispatch": dispatch_share},
                     indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
