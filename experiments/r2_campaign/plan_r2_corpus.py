"""How many gated blocks until the R2 corpus can actually answer a question?

Offline planning. It draws a sealed campaign, fills it with synthetic rewards,
runs the real replay estimators over it and compares the measured effective
sample size against the analytic prediction. Synthetic rewards are used only
to exercise the estimator plumbing; no performance claim follows from them.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from friday_optimizer.campaign import (  # noqa: E402
    BLOCK_SECONDS,
    MEASURED_POINT_SECONDS,
    CampaignPlan,
    expected_effective_samples,
    points_for_effective_samples,
)
from friday_optimizer.candidates import CandidateRegistry  # noqa: E402
from friday_optimizer.decisions import OutcomeEvent, SelectionPolicy  # noqa: E402
from friday_optimizer.replay import DEFAULT_MIN_SAMPLES, ReplayEnv, ReplayStep, ips, snips  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from test_optimizer_decisions import make_fingerprint  # noqa: E402

HINT = "head_skip_prefill"


def main() -> int:
    fingerprint = make_fingerprint()
    registry = CandidateRegistry()
    candidates = registry.ordered_ids(fingerprint, historical_hints=(HINT,))
    print(f"eligible actions ({len(candidates)}): {', '.join(candidates)}")
    print(f"block budget {BLOCK_SECONDS:.0f} s, measured {MEASURED_POINT_SECONDS:.0f} s per point"
          f" -> {int(BLOCK_SECONDS // MEASURED_POINT_SECONDS)} points per approved block")
    print()

    greedy = SelectionPolicy("target-greedy-v1")
    print(f"{'epsilon':>8} {'p(hint)':>8} {'ESS/point':>10} {'points':>7} {'blocks':>7}  target")
    for epsilon in (0.2, 0.3, 0.5, 0.8):
        logging_policy = SelectionPolicy("log-v1", rule="epsilon_greedy", epsilon=epsilon)
        share = logging_policy.distribution(candidates, (HINT,))[HINT]
        for label, target in (("greedy = hinted action", greedy), ("a rarely logged action", None)):
            if target is None:
                # Evaluating a target that always picks the least likely action.
                rare = min(candidates, key=lambda action: logging_policy.distribution(candidates, (HINT,))[action])
                probability = logging_policy.distribution(candidates, (HINT,))[rare]
                per_point = probability
                needed = None if per_point <= 0 else -(-DEFAULT_MIN_SAMPLES // per_point)
            else:
                per_point = expected_effective_samples(
                    logging_policy=logging_policy, target_policy=target,
                    candidates=candidates, points=1, hints=(HINT,),
                )
                needed = points_for_effective_samples(
                    logging_policy=logging_policy, target_policy=target,
                    candidates=candidates, required=DEFAULT_MIN_SAMPLES, hints=(HINT,),
                )
            if needed is None:
                print(f"{epsilon:8.2f} {share:8.3f} {per_point:10.3f} {'-':>7} {'-':>7}  {label} (no overlap)")
                continue
            needed = int(needed)
            plan = CampaignPlan(campaign_id="r2-sizing", policy=logging_policy, seed_base=1, points=min(needed, 512), hints=(HINT,))
            blocks = -(-needed // plan.points_per_block)
            print(f"{epsilon:8.2f} {share:8.3f} {per_point:10.3f} {needed:7d} {blocks:7d}  {label}")
    print()

    # Cross-check the analytic prediction against the real estimator plumbing.
    logging_policy = SelectionPolicy("log-v1", rule="epsilon_greedy", epsilon=0.5)
    points = points_for_effective_samples(
        logging_policy=logging_policy, target_policy=greedy,
        candidates=candidates, required=DEFAULT_MIN_SAMPLES, hints=(HINT,),
    )
    plan = CampaignPlan(campaign_id="r2-check", policy=logging_policy, seed_base=7, points=points, hints=(HINT,))
    events = plan.decisions(fingerprint, registry=registry)
    steps = [ReplayStep(event, OutcomeEvent(event.decision_id, "observed", reward=0.9)) for event in events]
    environment = ReplayEnv(steps)
    predicted = expected_effective_samples(
        logging_policy=logging_policy, target_policy=greedy,
        candidates=candidates, points=points, hints=(HINT,),
    )
    measured = ips(environment, greedy, resamples=200)
    print(f"campaign {plan.campaign_id}: {points} points, {plan.blocks} block(s), hash {plan.campaign_hash[:16]}")
    print(f"  predicted effective samples {predicted:.2f}")
    print(f"  measured  effective samples {measured.effective_samples:.2f}  status {measured.status}")
    print(f"  snips status {snips(environment, greedy, resamples=200).status}")
    drawn = {}
    for event in events:
        drawn[event.chosen] = drawn.get(event.chosen, 0) + 1
    print(f"  drawn actions {dict(sorted(drawn.items(), key=lambda item: -item[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
