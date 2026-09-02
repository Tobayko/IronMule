"""H1.0 — where speculation stops costing and starts paying.

The (K,N) dispatcher in the master plan rests on a number nobody has measured.
D5 measured `897`/`32` on the 4B and found speculation *losing* at every width
against the combined path; S1 measured `96` tokens and found it winning at every
width. The sign changes somewhere in between, and this study measures where.

Two things are deliberately different from `experiments/serve_gain/measure.py`:

* the baseline arm is the **serving default** (`head_skip` + `fixed_compiled` +
  `bundled_readback`), not the bare engine. The dispatcher would switch from
  that path, so that is the path speculation has to beat;
* A/A noise is measured per (model, answer length) and the pair count is
  derived from it. `0.612 %` (F1), `3.69 %` (D5 4B/32) and `14.25 %` (D5 1B/32)
  are too far apart for one regime's noise to stand in for another's.

Why a switch point must exist at all: `ironmule/runtime.py` runs two different
loops. `_decode` batches the readback (`readback_every = 8`);
`_decode_speculative` calls `mx.eval` and `mx.synchronize()` every iteration and
reads back with `.tolist()`, so `bundled_readback` is inert there. Speculation
pays a fixed per-iteration synchronisation cost and earns it back only through
accepted draft tokens.

Preregistration: `docs/H10_VORREGISTRIERUNG.md`. Nothing here is activated and
no formal claim is made.

Run (A/A first, then one width per process):

    python experiments/switch_point/measure.py --execute --model 4b --tokens 64 --aa-only
    python experiments/switch_point/measure.py --execute --model 4b --tokens 64 \
        --widths 1 --mde 0.031 --pairs 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "switch-point-20260902-01"
OUTPUT = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / "docs" / "H10_VORREGISTRIERUNG.md"

#: The serving default: the arm the dispatcher would switch away from.
COMBINED = {
    "head_skip_prefill": True,
    "compiled_fixed_cache": True,
    "readback_every": 8,
}
#: Draft widths measured against it. Width 0 *is* the baseline.
WIDTHS = (1, 2, 3)
NGRAM = 3
#: Answer lengths. 128 is the ceiling: `continuous_gpu_limit_s = 6.0` caps one
#: uninterrupted GPU block and D5 put the measurable ceiling at 287 tokens.
LENGTHS = (32, 48, 64, 96, 128)

AA_PAIRS = 6
#: Frozen resolution target: a draft-width change worth less than 3 % per
#: request does not justify a dispatcher in the serving path.
RESOLUTION = 0.03
PAIR_FLOOR, PAIR_CAP = 6, 24


def pairs_for(aa_spread: float) -> int:
    """Pair count for a regime with this A/A spread — the frozen rule.

    ``1/sqrt(n)`` scaling against the resolution target, floored, capped, and
    rounded up to even because `_pair_ratios_checked` rejects an unbalanced
    AB/BA set outright.
    """

    raw = math.ceil(AA_PAIRS * (aa_spread / RESOLUTION) ** 2)
    bounded = max(PAIR_FLOOR, min(PAIR_CAP, raw))
    return bounded + (bounded % 2)


def verdict_for(status: str) -> str:
    """Map the evaluator's status onto the frozen three-way decision."""

    return {"qualified": "wins", "rejected": "loses"}.get(status, "tie")


def self_check() -> int:
    assert pairs_for(0.0) == PAIR_FLOOR
    assert pairs_for(0.03) == PAIR_FLOOR
    assert pairs_for(0.0369) == 10, pairs_for(0.0369)
    assert pairs_for(0.1425) == PAIR_CAP
    assert verdict_for("qualified") == "wins"
    assert verdict_for("rejected") == "loses"
    assert verdict_for("below_threshold") == verdict_for("inconclusive") == "tie"
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID,
        "baseline_arm": COMBINED, "widths": list(WIDTHS), "lengths": list(LENGTHS),
        "aa_pairs": AA_PAIRS, "resolution": RESOLUTION,
        "pair_rule": "clamp(ceil(6*(s/0.03)^2), 6, 24) rounded up to even",
        "terminal_gate": "exact token identity per pair",
        "formal_claim": False, "no_activation": True,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--model", choices=("4b", "1b"), default="4b")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--widths", default="", help="comma-separated draft widths")
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--mde", type=float, default=None,
                        help="this regime's measured A/A spread, from the --aa-only run")
    parser.add_argument("--aa-only", action="store_true", dest="aa_only")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import release_gate, study_provenance

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate
    if args.tokens not in LENGTHS:
        raise SystemExit(f"tokens must be one of {LENGTHS}")

    from friday_optimizer.integration import evaluate_integration

    from friday_calibrate.runner import MODELS, build_runner, noise_mde, paired_arms

    model_id = MODELS[args.model]
    runner, identity, guard = build_runner(
        AA_PAIRS, model_id=model_id, output_tokens=args.tokens,
        prompt_tokens=897 if args.model == "4b" else 0,
    )
    started = time.time()
    workload = f"sealed-{identity['prompt_tokens']}-{args.tokens}"

    def warm(knobs: dict) -> None:
        """One unevaluated call per arm so `mx.compile` does not land in pair 0."""
        runner(knobs)

    if args.aa_only:
        warm(COMBINED)
        aa_spread, _ = noise_mde(runner, pairs=AA_PAIRS, knobs=COMBINED)
        report = {
            "study_id": STUDY_ID, "state": "aa_only", "model": args.model, **identity,
            "baseline_arm": COMBINED, "aa_pairs": AA_PAIRS, "aa_spread": aa_spread,
            "derived_pairs": pairs_for(aa_spread), "resolution": RESOLUTION,
            "budget": guard.summary(), "wall_seconds": round(time.time() - started, 1),
            "formal_claim": False, "no_activation": True,
        }
        destination = args.out or OUTPUT / f"aa_{args.model}_{args.tokens}.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.mde is None:
        raise SystemExit("pass --mde from this regime's --aa-only run, or run --aa-only first")
    if not 0.0 < args.mde < 1.0:
        raise SystemExit("--mde must be a fraction within (0, 1)")
    aa_spread = args.mde
    pairs = args.pairs or pairs_for(aa_spread)
    if not 2 <= pairs <= 32 or pairs % 2:
        raise SystemExit("pairs must be an even number between 2 and 32")
    widths = tuple(int(w) for w in args.widths.split(",") if w) or WIDTHS

    from friday_evidence.budget import BudgetError

    rows = []
    for width in widths:
        knobs = dict(COMBINED, speculate_k=width, speculate_ngram=NGRAM)
        spent = guard.gpu_work_seconds
        try:
            warm(COMBINED)
            warm(knobs)
        except BudgetError as error:
            # One call — compile plus generate — exceeded a frozen limit. That is
            # a second measurement ceiling on top of D5's 287 tokens, and it is
            # reported as one. The limit is not raised to make the cell fit.
            rows.append({"width": width, "verdict": "budget_exceeded",
                         "reason": str(error), "budget": guard.summary()})
            break
        # The warmups price one call on this device. A run that cannot finish
        # inside the guard's 120 s stops *before* spending them, rather than
        # dying half way through and leaving unusable GPU seconds behind. The
        # budget is not raised; the regime is reported as not measurable.
        per_call = (guard.gpu_work_seconds - spent) / 2
        headroom = guard.policy.gpu_work_limit_s - guard.gpu_work_seconds
        needed = per_call * 2 * pairs * 1.15
        if needed > headroom:
            rows.append({"width": width, "verdict": "budget_insufficient",
                         "pairs": pairs, "seconds_per_call": round(per_call, 3),
                         "seconds_needed": round(needed, 1),
                         "seconds_available": round(headroom, 1)})
            break
        try:
            baseline, candidate, breaks = paired_arms(
                runner, knobs, pairs=pairs, workload=workload, baseline_knobs=COMBINED,
            )
        except BudgetError as error:
            rows.append({"width": width, "verdict": "budget_exceeded",
                         "reason": str(error), "budget": guard.summary()})
            break
        if breaks:
            rows.append({"width": width, "verdict": "identity_break",
                         "reason": breaks[0], "pairs": len(baseline)})
            break
        # min_gain == mde == this regime's A/A spread: a win has to clear the
        # noise of measuring nothing, in the direction it claims.
        result = evaluate_integration(
            baseline, candidate, arm="warm", min_gain=aa_spread, mde=aa_spread,
            min_pairs=max(2, pairs // 2),
        )
        low, high = result.ci or (None, None)
        rows.append({
            "width": width, "verdict": verdict_for(result.status), "status": result.status,
            "pairs": result.pairs, "ratio_median": result.ratio_median,
            "gain_percent": result.gain_percent,
            "ci": None if result.ci is None else [low, high],
            "token_identical": True, "reasons": list(result.reasons),
        })

    terminal = ("identity_break", "budget_insufficient", "budget_exceeded")
    broken = [row for row in rows if row["verdict"] in terminal]
    winners = [row["width"] for row in rows if row["verdict"] == "wins"]
    report = {
        "study_id": STUDY_ID, "state": "measured", "model": args.model, **identity,
        "baseline_arm": COMBINED, "workload": workload,
        "aa_spread": aa_spread, "mde": aa_spread, "pairs": pairs,
        "rows": rows,
        "winning_widths": winners,
        "token_identity_held": not broken,
        "verdict": broken[0]["verdict"] if broken else "measured",
        "wall_seconds": round(time.time() - started, 1),
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "friday_calibrate" / "runner.py"],
            preregistration=PREREGISTRATION,
            extra={"model_id": model_id, "model_revision": identity["model_revision"]},
        ),
        "formal_claim": False, "no_activation": True,
    }
    suffix = "_".join(str(w) for w in widths)
    destination = args.out or OUTPUT / f"switch_{args.model}_{args.tokens}_w{suffix}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"},
                     indent=2, sort_keys=True))
    return 2 if broken and broken[0]["verdict"] == "identity_break" else 0


if __name__ == "__main__":
    raise SystemExit(main())
