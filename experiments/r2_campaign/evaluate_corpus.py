"""Off-policy evaluation of the measured R2 corpus.

Reads the campaign's own records out of Optimization Memory and prices one
deterministic target policy per measurable action, on the full corpus and on the
holdout that ``docs/R2_VORREGISTRIERUNG.md`` froze before the first point.

Nothing here measures. It reports what the 352 measured points support, and --
just as importantly -- what they do not.

Run: ``python experiments/r2_campaign/evaluate_corpus.py``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

STUDY_ID = "r2-corpus-eval-20260905-01"
CAMPAIGN_PREFIX = "r2-corpus-20260904-01."
#: Frozen in the preregistration: the last 20 % of the sealed draw order.
HOLDOUT_FROM = 320
#: The measurable action space after Amendment A2.
ACTIONS = ("baseline", "head_skip_prefill", "fixed_compiled_cache", "readback_every_2")


def _index(step) -> int:
    return int(step.decision.decision_id.rsplit(".", 1)[1])


def _price(env, action: str, min_samples: int, resamples: int) -> dict:
    from friday_optimizer.decisions import SelectionPolicy
    from friday_optimizer.replay import ips, snips

    policy = SelectionPolicy(f"target-{action}", rule="deterministic_order", epsilon=0.0)
    hints = () if action == "baseline" else (action,)
    estimate = ips(env, policy, min_samples=min_samples, target_hints=hints, resamples=resamples)
    stable = snips(env, policy, min_samples=min_samples, target_hints=hints, resamples=resamples)
    return {
        "action": action,
        # The gate rests on ips: it is the estimator the CLI can reach that
        # carries a bootstrap interval (Amendment A4).
        "ips": estimate.value,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "effective_samples": estimate.effective_samples,
        "samples": estimate.samples,
        "status": estimate.status,
        "conclusive": estimate.conclusive,
        # snips has no interval by construction (replay.py:387); reported as a
        # stability comparison only, with no say in the verdict.
        "snips": stable.value,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--memory", default=str(PROJECT_ROOT / ".friday-data" / "optimizer-v2.sqlite3"))
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    from friday_optimizer.memory import OptimizationMemoryV2
    from friday_optimizer.replay import ReplayEnv, load_steps

    with OptimizationMemoryV2(args.memory) as memory:
        chain_ok = memory.verify_chain()
        steps = [s for s in load_steps(memory) if s.decision.decision_id.startswith(CAMPAIGN_PREFIX)]

    splits = {
        "full": steps,
        "train": [s for s in steps if _index(s) < HOLDOUT_FROM],
        "holdout": [s for s in steps if _index(s) >= HOLDOUT_FROM],
    }
    report = {
        "study_id": STUDY_ID,
        "chain_verified": chain_ok,
        "steps": len(steps),
        "holdout_from_index": HOLDOUT_FROM,
        "min_samples": args.min_samples,
        "splits": {},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_claim": False,
        "learning_claim": False,
    }
    for name, subset in splits.items():
        env = ReplayEnv(subset)
        rows = [_price(env, action, args.min_samples, args.resamples) for action in ACTIONS]
        report["splits"][name] = {"n": len(subset), "estimates": rows}
        print(f"--- {name}  n={len(subset)}")
        print(f"{'target policy':24}{'ips':>9}{'ci_low':>9}{'ci_high':>9}{'ess':>8}  status")
        for row in rows:
            lo = f"{row['ci_low']:9.4f}" if row["ci_low"] is not None else f"{'-':>9}"
            hi = f"{row['ci_high']:9.4f}" if row["ci_high"] is not None else f"{'-':>9}"
            value = row["ips"] if row["ips"] is not None else float("nan")
            print(f"{row['action']:24}{value:9.4f}{lo}{hi}{row['effective_samples']:8.1f}  {row['status']}")
        print()

    holdout = report["splits"]["holdout"]["estimates"]
    inconclusive = [row["action"] for row in holdout if not row["conclusive"]]
    report["holdout_inconclusive"] = inconclusive
    report["gate_assessable"] = not inconclusive
    if inconclusive:
        print(
            "The preregistered gate cannot be assessed on the frozen holdout: "
            f"{len(inconclusive)} of {len(ACTIONS)} target policies stay below the "
            f"effective-sample floor of {args.min_samples} -- {', '.join(inconclusive)}."
        )

    out = Path(__file__).resolve().parent / "corpus_evaluation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
