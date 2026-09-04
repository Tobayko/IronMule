"""H1.3 — does this machine actually throttle, and does the rate drift?

The master plan wanted `BudgetGuard` pulled into `Server.generate`. That is a
category error: its `>= 4 s` break and `<= 25 %` duty cycle exist so a *number*
means something, and in a product they would mean "wait four seconds before
your second question". Nothing goes into the serving path without evidence that
something needs protecting. This study is that evidence, or its absence.

**What this measures, honestly:** decode rate over a multi-minute session *at
the project's own duty cycle* — 6 s blocks with the mandated breaks, repeated.
It answers "does the rate drift over a long session as this project runs the
GPU?".

**What it cannot measure:** thermal throttling under continuous 100 % load.
That needs `continuous_gpu_limit_s` to be exceeded, which is a user rule, not a
knob for a study to pick. It is filed as an open user decision in `BACKLOG.md`;
until it is answered the serving path gets nothing, which is the default
anyway.

Run: ``python experiments/thermal_drift/measure.py --execute --minutes 6``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "thermal-drift-20260902-01"
OUTPUT = Path(__file__).resolve().parent
#: The serving default, so the rate measured is the rate the product would see.
COMBINED = {"head_skip_prefill": True, "compiled_fixed_cache": True, "readback_every": 8}
TOKENS = 96
#: A drift larger than this over the session is what would justify *observing*
#: temperature in the serving path. Frozen before the run.
DRIFT_THRESHOLD = 0.10


def self_check() -> int:
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID, "arm": COMBINED,
        "tokens_per_block": TOKENS, "drift_threshold": DRIFT_THRESHOLD,
        "measures": "decode rate over time at the project's own duty cycle",
        "cannot_measure": "throttling under continuous load; that needs a user decision",
        "formal_claim": False, "no_activation": True,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--model", choices=("4b", "1b"), default="4b")
    parser.add_argument("--minutes", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import release_gate, study_provenance

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate
    if not 1.0 <= args.minutes <= 18.0:
        raise SystemExit("minutes must be between 1 and 18 (the guard's wall limit is 20)")

    from friday_calibrate.runner import MODELS, build_runner

    runner, identity, guard = build_runner(
        6, model_id=MODELS[args.model], output_tokens=TOKENS,
        prompt_tokens=897 if args.model == "4b" else 0,
    )
    runner(COMBINED)  # warmup: the first call pays for `mx.compile`
    started = time.time()
    deadline = started + args.minutes * 60
    blocks = []
    while time.time() < deadline:
        try:
            sample = runner(COMBINED)
        except Exception as error:  # BudgetError ends the session, it does not fail it
            blocks.append({"stopped": type(error).__name__, "reason": str(error)})
            break
        blocks.append({
            "at_seconds": round(time.time() - started, 1),
            "decode_tps": sample.decode_tps,
            "ttft_seconds": sample.ttft_seconds,
        })

    rates = [block["decode_tps"] for block in blocks if "decode_tps" in block]
    third = max(1, len(rates) // 3)
    first_third, last_third = rates[:third], rates[-third:]
    drift = (
        None if not rates
        else (statistics.median(last_third) - statistics.median(first_third))
        / statistics.median(first_third)
    )
    report = {
        "study_id": STUDY_ID, "state": "measured", "model": args.model, **identity,
        "arm": COMBINED, "blocks": blocks, "block_count": len(rates),
        "median_tps": statistics.median(rates) if rates else None,
        "first_third_median_tps": statistics.median(first_third) if rates else None,
        "last_third_median_tps": statistics.median(last_third) if rates else None,
        "drift_fraction": drift,
        "drift_threshold": DRIFT_THRESHOLD,
        "verdict": (
            "no_evidence" if rates and drift is not None and abs(drift) < DRIFT_THRESHOLD
            else "drift_observed" if rates else "no_data"
        ),
        "wall_seconds": round(time.time() - started, 1),
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "friday_calibrate" / "runner.py"],
            extra={"model_id": MODELS[args.model], "model_revision": identity["model_revision"]},
        ),
        "formal_claim": False, "no_activation": True,
    }
    destination = args.out or OUTPUT / f"drift_{args.model}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k not in ("provenance", "blocks")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
