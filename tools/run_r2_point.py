#!/usr/bin/env python3
"""One R2 campaign point, end to end: drawn decision -> measurement -> label.

Before four hundred points are worth starting, one has to work. This runs
exactly one, and it exists because the path the plan assumed does not:
``friday_optimizer session`` measures a single hardcoded candidate
(``real_session.ALLOWED_CANDIDATE = "combined_core_profile"``, three sites) and
wants a machine-readable preregistration artefact with sixteen hash fields. It
has never been run on this machine, and its readiness gate has never passed.

So the measurement goes through ``friday_calibrate.runner`` instead -- the
harness that produced the only real device profile this repository has. It is
paired AB/BA, it checks token identity on every pair, and it carries the budget
guard and the mains-power gate. The ratio comes from
``integration.paired_request_ratios``, the project's own pairing rules.

What this does NOT do: start a campaign, write more than one point, or ground
any learning claim. It answers "does one point work end to end".

Run:  python tools/run_r2_point.py --execute
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

#: Frozen in docs/R2_VORREGISTRIERUNG.md.
CAMPAIGN_ID = "r2-corpus-20260904-01"
EPSILON = 0.6
SEED_BASE = 20260904
POINTS = 400
HINT = "head_skip_prefill"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

#: How a drawn candidate id becomes engine knobs. ``candidates.py`` carries no
#: parameters for these ids -- the mapping lives in the tune adapter, which this
#: path does not use, so it is stated here and in the preregistration.
#:
#: ``persistent_process`` is deliberately absent. It is a process strategy, not
#: an engine knob (``ironmule.runtime.Knobs`` has no such field), and this
#: harness already keeps one process across every pair -- so measuring it
#: against baseline would compare two identical configurations and produce a
#: ratio near 1.0 by construction. A drawn point for it is skipped and gets no
#: outcome record at all: ``ReplayEnv.reward_of`` returns None only for a step
#: with no outcome, and ``_weights`` skips exactly those, whereas a censored
#: record would enter as reward 0.0 and dilute every IPS estimate. See
#: docs/R2_VORREGISTRIERUNG.md, Amendment A2.
CANDIDATE_KNOBS = {
    "baseline": {},
    "head_skip_prefill": {"head_skip_prefill": True},
    "fixed_compiled_cache": {"compiled_fixed_cache": True},
    "readback_every_2": {"readback_every": 2},
}


def _fingerprint():
    """The real (environment, model, workload) identity of this run."""

    from _bench import resolve_local_model_snapshot

    from friday_optimizer.collector import collect_environment
    from friday_optimizer.fingerprint import (
        EnvironmentFingerprint,
        ExactFingerprint,
        ModelFingerprint,
        WorkloadFingerprint,
    )

    import subprocess

    head = subprocess.run(
        ["/usr/bin/git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    environment = EnvironmentFingerprint(
        **dict(collect_environment(runtime_commit=head).as_dict()["environment"])
    )
    # The snapshot resolver reports identity but no digests, so both are computed
    # here from what is actually on disk: the manifest over the resolver's own
    # identity record (revision, weight files, byte counts), the tokenizer over
    # the tokenizer files themselves.
    import hashlib

    from friday_optimizer.canonical import canonical_bytes

    manifest = hashlib.sha256(canonical_bytes(snapshot.report_identity())).hexdigest()
    tokenizer_digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        path = Path(snapshot.path) / name
        if path.exists():
            tokenizer_digest.update(name.encode())
            tokenizer_digest.update(path.read_bytes())
    tokenizer_sha256 = tokenizer_digest.hexdigest()
    model = ModelFingerprint(
        model_id=MODEL_ID,
        revision=snapshot.revision,
        manifest=manifest,
        architecture="gemma3",
        quant_bits=4,
        quant_group_size=64,
        tokenizer=tokenizer_sha256,
    )
    workload = WorkloadFingerprint(
        prompt_family="sealed-897-32",
        tokenizer=tokenizer_sha256,
        generator="ironmule",
        context_bucket="897",
        batch=1,
        concurrency=1,
        max_tokens=32,
        greedy=True,
        prompt_logprobs=False,
        power_mode="AC",
        mode="interactive",
    )
    return ExactFingerprint(environment=environment, model=model, workload=workload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--index", type=int, default=0, help="Which point of the sealed sequence")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument(
        "--memory",
        default=str(PROJECT_ROOT / ".friday-data" / "optimizer-v2.sqlite3"),
    )
    parser.add_argument("--write", action="store_true", help="Persist the decision and its label")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78

    from friday_calibrate.runner import build_runner, paired_arms
    from friday_optimizer.campaign import CampaignPlan
    from friday_optimizer.decisions import SelectionPolicy
    from friday_optimizer.integration import paired_request_ratios
    from friday_optimizer.reward import outcome_from_ratios

    fingerprint = _fingerprint()
    if not fingerprint.recommendation_allowed:
        print(json.dumps({"state": "fingerprint_incomplete", "reason": fingerprint.ood_reason}))
        return 1

    plan = CampaignPlan(
        campaign_id=CAMPAIGN_ID,
        policy=SelectionPolicy("r2-logging-v1", rule="epsilon_greedy", epsilon=EPSILON),
        seed_base=SEED_BASE,
        points=POINTS,
        hints=(HINT,),
    )
    decisions = plan.decisions(fingerprint)
    decision = decisions[args.index]
    knobs = CANDIDATE_KNOBS.get(decision.chosen)
    if knobs is None:
        # Not an error: a preregistered skip. Nothing is written, so the step
        # stays unlabelled and every estimator passes over it.
        print(
            json.dumps(
                {
                    "state": "skipped_unmeasurable_action",
                    "index": args.index,
                    "decision_id": decision.decision_id,
                    "chosen": decision.chosen,
                    "written": False,
                    "reason": "no engine knob; see R2_VORREGISTRIERUNG.md A2",
                }
            )
        )
        return 0

    print(
        json.dumps(
            {
                "campaign_hash": plan.campaign_hash,
                "index": args.index,
                "decision_id": decision.decision_id,
                "chosen": decision.chosen,
                "propensity": decision.propensity,
                "knobs": knobs,
                "fingerprint_hash": fingerprint.fingerprint_hash,
            }
        )
    )

    started = time.perf_counter()
    run, identity, guard = build_runner(args.pairs, model_id=MODEL_ID)
    baseline, candidate, breaks = paired_arms(run, knobs, pairs=args.pairs)
    ratios, reasons = paired_request_ratios(baseline, candidate)
    elapsed = time.perf_counter() - started

    if breaks:
        outcome = outcome_from_ratios(
            [], decision_id=decision.decision_id, notes=";".join(breaks)
        )
    else:
        outcome = outcome_from_ratios(
            ratios,
            decision_id=decision.decision_id,
            notes=";".join(reasons) if reasons else "measured",
        )

    written = False
    if args.write:
        from friday_optimizer.memory import OptimizationMemoryV2

        with OptimizationMemoryV2(args.memory) as memory:
            memory.append(decision.as_record())
            memory.append(outcome.as_record())
        written = True

    print(
        json.dumps(
            {
                "state": "measured",
                "chosen": decision.chosen,
                "pairs": len(ratios),
                "token_identity_breaks": list(breaks),
                "pairing_reasons": list(reasons),
                "censoring": outcome.censoring,
                "reward_ratio_median": outcome.reward,
                "gain_percent": None if outcome.reward is None else round((1 - outcome.reward) * 100, 3),
                "ratio_spread": [round(min(ratios), 4), round(max(ratios), 4)] if ratios else None,
                "wall_seconds": round(elapsed, 1),
                "budget": guard.summary(),
                "written": written,
                "formal_claim": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
