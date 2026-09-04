"""D5 — what `friday_serve` actually delivers with its knobs on.

D2 proved the precondition: with every knob off, `friday_serve` reproduces
`mlx_lm.stream_generate` token for token. That is equivalence, not a result.
This study measures the thing the project exists for — the same request path
against itself, knobs off against knobs on, paired, with exact token identity as
a terminal gate.

Design follows F1, because F1 is the measurement this has to be comparable to:
alternating AB/BA pairs inside one process, one loaded model, the sealed prompt.
Unpaired comparison is worthless here (`20.5 %` coefficient of variation against
`1.32 %` paired), and a fixed arm order would let warm-up masquerade as a gain.

Both model sizes are measured, and the small one is not an afterthought:
`experiments/prompt_lookup/code_edit_1b.json` reports `1.6946` against the 4B's
`1.2148`, so speculation carries far more where the model is cheaper to verify
against. A different picture on 1B is itself a result.

Speculation runs at a **fixed** draft width, swept over `1..3`. S1 closed the
adaptive variant: the widths differ by `0.016`, which is inside the noise, so
there is nothing for a bandit to learn.

Run: ``python experiments/serve_gain/measure.py --execute --model 4b --tokens 32``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "serve-gain-20260902-01"
OUTPUT = Path(__file__).resolve().parent
DEFAULT_PAIRS = 6

#: Each arm on top of the baseline, in the order they are measured. "combined"
#: is the one that answers the project's question; the single knobs are there so
#: a surprise in the combination can be attributed.
ARMS = {
    "head_skip": {"head_skip_prefill": True},
    "fixed_compiled": {"compiled_fixed_cache": True},
    "bundled_readback": {"readback_every": 8},
    "combined": {
        "head_skip_prefill": True,
        "compiled_fixed_cache": True,
        "readback_every": 8,
    },
}
#: Fixed draft widths, on top of the combined knobs.
SPECULATION_WIDTHS = (1, 2, 3)


def self_check() -> int:
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID,
        "arms": sorted(ARMS), "speculation_widths": list(SPECULATION_WIDTHS),
        "pairs": DEFAULT_PAIRS, "terminal_gate": "exact token identity per pair",
        "formal_claim": False,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--model", choices=("4b", "1b"), default="4b")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--arms", default="", help="comma-separated subset of the arms")
    parser.add_argument("--speculation", default="", help="comma-separated widths, or 'none'")
    parser.add_argument("--aa-only", action="store_true", dest="aa_only",
                        help="measure only this regime's A/A noise and stop")
    parser.add_argument("--mde", type=float, default=None,
                        help="use a previously measured A/A noise instead of measuring it again")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import release_gate, study_provenance

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate
    if not 2 <= args.pairs <= 32:
        raise SystemExit("pairs must be between 2 and 32")

    from friday_optimizer.integration import evaluate_integration

    from friday_calibrate.runner import MODELS, build_runner, noise_mde, paired_arms

    model_id = MODELS[args.model]
    # The sealed 897-token count holds for the 4B tokenizer only; on 1B the same
    # text tokenises differently and the count is recorded, not enforced.
    runner, identity, guard = build_runner(
        args.pairs, model_id=model_id, output_tokens=args.tokens,
        prompt_tokens=897 if args.model == "4b" else 0,
    )

    started = time.time()
    # The guard's 120 s GPU budget is deliberate and is not raised: a study that
    # does not fit runs in pieces, each piece inside its own budget. Splitting
    # costs an extra A/A unless a measured one is passed back in.
    if args.mde is not None:
        if not 0.0 < args.mde < 1.0:
            raise SystemExit("--mde must be a fraction within (0, 1)")
        aa_spread, mde = args.mde, args.mde
    else:
        aa_spread, mde = noise_mde(runner, pairs=args.pairs)
    if args.aa_only:
        report = {
            "study_id": STUDY_ID, "state": "aa_only", "model": args.model, **identity,
            "pairs": args.pairs, "aa_spread": aa_spread, "mde": mde,
            "budget": guard.summary(), "formal_claim": False,
        }
        destination = args.out or OUTPUT / f"aa_{args.model}_{args.tokens}.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    selected = [name for name in ARMS if not args.arms or name in args.arms.split(",")]
    widths = (
        ()
        if args.speculation == "none"
        else tuple(int(w) for w in args.speculation.split(",") if w)
        or SPECULATION_WIDTHS
    )

    def evaluate(name: str, knobs: dict) -> dict:
        baseline, candidate, breaks = paired_arms(runner, knobs, pairs=args.pairs)
        if breaks:
            return {"arm": name, "knobs": knobs, "status": "token_identity_broken",
                    "reason": breaks[0], "pairs": len(baseline)}
        result = evaluate_integration(
            baseline, candidate, arm="warm", min_gain=0.0, mde=mde,
            min_pairs=max(2, args.pairs // 2),
        )
        low, high = result.ci or (None, None)
        return {
            "arm": name, "knobs": knobs, "status": result.status, "pairs": result.pairs,
            "ratio_median": result.ratio_median,
            "gain_percent": result.gain_percent,
            "ci": None if result.ci is None else [low, high],
            # A serving knob has to be real and identical, not big: the interval
            # wholly below 1.0 is the bar, and it is weaker than a promotion bar
            # on purpose (see BACKLOG D4).
            "real_gain": bool(high is not None and high < 1.0),
            "token_identical": True,
            "reasons": list(result.reasons),
        }

    rows = [evaluate(name, ARMS[name]) for name in selected]
    for width in widths:
        knobs = dict(ARMS["combined"])
        knobs.update({"speculate_k": width, "speculate_ngram": 3})
        rows.append(evaluate(f"combined+speculate_{width}", knobs))

    broken = [row for row in rows if row["status"] == "token_identity_broken"]
    report = {
        "study_id": STUDY_ID,
        "state": "measured",
        "model": args.model,
        **identity,
        "pairs": args.pairs,
        "aa_spread": aa_spread,
        "mde": mde,
        "rows": rows,
        "token_identity_held": not broken,
        "verdict": "identity_break" if broken else "measured",
        "wall_seconds": round(time.time() - started, 1),
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "friday_calibrate" / "runner.py",
             ROOT / "friday_serve" / "server.py"],
            extra={"model_id": model_id, "model_revision": identity["model_revision"]},
        ),
        "formal_claim": False,
        "no_activation": True,
    }
    destination = args.out or OUTPUT / f"gain_{args.model}_{args.tokens}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"},
                     indent=2, sort_keys=True))
    return 2 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
