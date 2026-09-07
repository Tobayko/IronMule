"""Population-corrected, descriptive evaluation of the frozen R2 corpus.

The original evaluator prices only labelled rows.  With unconditioned
propensities that is not a valid conditional policy estimate, and it is not the
preregistered population average after Amendment A2:
the campaign drew 400 points and deliberately left the 48
``persistent_process`` points without records.  This module reconstructs the
sealed draw schedule from its immutable plan and the observed decision
metadata, then prices every target over the original population denominator.

This is deliberately read-only and descriptive.  It does not change the
historical report, gate, policy, or any runtime setting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from friday_optimizer.campaign import CampaignPlan
from friday_optimizer.candidates import CandidateRegistry
from friday_optimizer.decisions import SelectionPolicy
from friday_optimizer.memory import OptimizationMemoryV2
from friday_optimizer.replay import ReplayStep, _interval, effective_sample_size, load_steps


STUDY_ID = "r2-corpus-eval-20260905-01-corrected"
CAMPAIGN_ID = "r2-corpus-20260904-01"
CAMPAIGN_PREFIX = CAMPAIGN_ID + "."
ORIGINAL_POPULATION = 400
HOLDOUT_FROM = 320
HINT = "head_skip_prefill"
MIN_SAMPLES = 30
ACTIONS = ("baseline", "head_skip_prefill", "fixed_compiled_cache", "readback_every_2")
ALL_DRAW_ACTIONS = ACTIONS + ("persistent_process",)
EXPECTED_DRAW_COUNTS = {
    "head_skip_prefill": 214,
    "readback_every_2": 49,
    "persistent_process": 48,
    "fixed_compiled_cache": 46,
    "baseline": 43,
}


class ReconstructionError(ValueError):
    """The immutable campaign evidence cannot support a safe reconstruction."""


def _index(step: ReplayStep) -> int:
    prefix, _, suffix = step.decision.decision_id.rpartition(".")
    if prefix != CAMPAIGN_ID or not suffix.isdigit() or len(suffix) != 4:
        raise ReconstructionError(f"malformed campaign decision id: {step.decision.decision_id!r}")
    index = int(suffix)
    if not 0 <= index < ORIGINAL_POPULATION:
        raise ReconstructionError(f"campaign index outside 0..399: {index}")
    return index


def _frozen_plan() -> tuple[CampaignPlan, SelectionPolicy, CandidateRegistry]:
    policy = SelectionPolicy("r2-logging-v1", rule="epsilon_greedy", epsilon=0.6)
    plan = CampaignPlan(
        campaign_id=CAMPAIGN_ID,
        policy=policy,
        seed_base=20260904,
        points=ORIGINAL_POPULATION,
        hints=(HINT,),
    )
    return plan, policy, CandidateRegistry()


def reconstruct_schedule(
    steps: Sequence[ReplayStep],
) -> tuple[dict[int, tuple[str, float, int]], dict[str, int], tuple[int, ...]]:
    """Verify observed metadata and recover the frozen action schedule.

    Only schedule metadata is reconstructed for absent rows.  No absent row is
    turned into a labelled observation or assigned a reward.
    """

    if not steps:
        raise ReconstructionError("campaign has no recorded decisions")
    plan, policy, registry = _frozen_plan()
    first = steps[0].decision
    candidates = first.candidate_set
    if set(candidates) != set(ALL_DRAW_ACTIONS) or len(candidates) != len(ALL_DRAW_ACTIONS):
        raise ReconstructionError("observed candidate set differs from frozen registry")
    if first.policy_hash != policy.policy_hash:
        raise ReconstructionError("observed policy hash differs from frozen logging policy")
    if first.registry_hash != registry.registry_hash:
        raise ReconstructionError("observed registry hash differs from frozen registry")
    stable_context = {key: value for key, value in first.context.items() if key != "environment.runtime_commit"}

    schedule: dict[int, tuple[str, float, int]] = {}
    counts: dict[str, int] = {action: 0 for action in ALL_DRAW_ACTIONS}
    for index in range(ORIGINAL_POPULATION):
        seed = plan.seed_for(index)
        chosen, propensity = policy.select(candidates, hints=(HINT,), seed=seed)
        schedule[index] = (chosen, propensity, seed)
        counts[chosen] += 1
    if counts != EXPECTED_DRAW_COUNTS:
        raise ReconstructionError(f"frozen schedule count mismatch: {counts!r}")

    observed: dict[int, ReplayStep] = {}
    for step in steps:
        index = _index(step)
        if index in observed:
            raise ReconstructionError(f"duplicate campaign index: {index}")
        decision = step.decision
        expected_action, expected_propensity, expected_seed = schedule[index]
        if decision.candidate_set != candidates:
            raise ReconstructionError(f"candidate set drift at index {index}")
        if decision.policy_id != policy.policy_id or decision.policy_hash != policy.policy_hash:
            raise ReconstructionError(f"policy metadata drift at index {index}")
        if decision.registry_hash != registry.registry_hash:
            raise ReconstructionError(f"registry metadata drift at index {index}")
        if {key: value for key, value in decision.context.items() if key != "environment.runtime_commit"} != stable_context:
            raise ReconstructionError(f"non-commit fingerprint context drift at index {index}")
        if decision.selection_rule != "epsilon_greedy" or decision.hints != (HINT,):
            raise ReconstructionError(f"selection metadata drift at index {index}")
        if decision.seed != expected_seed or decision.chosen != expected_action:
            raise ReconstructionError(f"frozen action/seed mismatch at index {index}")
        if not math.isclose(decision.propensity, expected_propensity, rel_tol=0.0, abs_tol=1e-15):
            raise ReconstructionError(f"propensity mismatch at index {index}")
        if step.outcome is None or not step.outcome.observed:
            raise ReconstructionError(f"target-supported index {index} is missing an observed outcome")
        if step.outcome.reward_metric != "ratio_median" or step.outcome.reward is None:
            raise ReconstructionError(f"index {index} has an unsupported reward metric")
        if not math.isfinite(step.outcome.reward) or step.outcome.reward <= 0.0:
            raise ReconstructionError(f"index {index} has an invalid ratio reward")
        observed[index] = step

    missing = tuple(index for index in range(ORIGINAL_POPULATION) if index not in observed)
    invalid_missing = tuple(index for index in missing if schedule[index][0] != "persistent_process")
    if invalid_missing:
        raise ReconstructionError(
            "missing records include target-supported actions: "
            + ", ".join(map(str, invalid_missing[:8]))
        )
    if len(observed) != ORIGINAL_POPULATION - EXPECTED_DRAW_COUNTS["persistent_process"]:
        raise ReconstructionError("observed population is not 352 records")
    return schedule, counts, missing


def population_estimate(
    steps_by_index: Mapping[int, ReplayStep],
    schedule: Mapping[int, tuple[str, float, int]],
    indices: Iterable[int],
    target_action: str,
    *,
    min_samples: int = MIN_SAMPLES,
    resamples: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Price one deterministic target over the supplied original population."""

    if target_action not in ACTIONS:
        raise ReconstructionError(f"target action is not measurable: {target_action}")
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples < 1:
        raise ReconstructionError("min_samples must be a positive integer")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ReconstructionError("resamples must be a positive integer")
    population = tuple(indices)
    if not population:
        raise ReconstructionError("population split must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in population):
        raise ReconstructionError("population indices must be integers")
    if len(set(population)) != len(population):
        raise ReconstructionError("population indices must be unique")
    for index, (action, propensity, seed_value) in schedule.items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < ORIGINAL_POPULATION:
            raise ReconstructionError("schedule indices must be unique integers in 0..399")
        if action not in ALL_DRAW_ACTIONS:
            raise ReconstructionError(f"schedule action is not registered: {action}")
        if isinstance(propensity, bool) or not isinstance(propensity, (int, float)) or not math.isfinite(propensity) or not 0.0 < propensity <= 1.0:
            raise ReconstructionError(f"invalid schedule propensity at index {index}")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int) or seed_value < 0:
            raise ReconstructionError(f"invalid schedule seed at index {index}")
    for index in population:
        if index not in schedule:
            raise ReconstructionError(f"population index {index} has no frozen schedule entry")
    for index, step in steps_by_index.items():
        if isinstance(index, bool) or not isinstance(index, int) or index not in schedule:
            raise ReconstructionError(f"observed record index {index!r} has no frozen schedule entry")
        expected_action, expected_propensity, _ = schedule[index]
        if step.decision.chosen != expected_action or not math.isclose(
            step.decision.propensity, expected_propensity, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ReconstructionError(f"observed action/propensity mismatch at index {index}")
        if step.outcome is None or not step.outcome.observed:
            raise ReconstructionError(f"observed record index {index} has no observed outcome")
    contributions: list[float] = []
    weights: list[float] = []
    observed_count = 0
    support_count = 0
    for index in population:
        step = steps_by_index.get(index)
        if step is None:
            if schedule[index][0] != "persistent_process":
                raise ReconstructionError(f"missing target-supported record at index {index}")
            contributions.append(0.0)
            continue
        observed_count += 1
        chosen, propensity, _ = schedule[index]
        if chosen != target_action:
            contributions.append(0.0)
            continue
        support_count += 1
        weight = 1.0 / propensity
        weights.append(weight)
        # ratio_median is a candidate/baseline ratio; replay's reward convention
        # is the measured relative gain, so higher remains better.
        reward = 1.0 - float(step.outcome.reward)  # type: ignore[union-attr]
        contributions.append(weight * reward)

    numerator = math.fsum(contributions)
    denominator = len(population)
    estimate = numerator / denominator
    ess = effective_sample_size(weights)
    status = "ok" if ess >= min_samples else "insufficient_data"
    weight_sum = math.fsum(weights)
    snips = numerator / weight_sum if weight_sum > 0.0 else None
    ci_low, ci_high = _interval(contributions, seed=seed, resamples=resamples)
    observed_fraction = observed_count / denominator
    return {
        "action": target_action,
        "ips": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "effective_samples": ess,
        "samples": observed_count,
        "censored_samples": 0,
        "support_samples": support_count,
        "status": status,
        "conclusive": status == "ok",
        "snips": snips,
        "numerator": numerator,
        "denominator": denominator,
        "weight_sum": weight_sum,
        "normalised_diagnostics": {
            "observed_fraction": observed_fraction,
            "support_fraction": support_count / denominator,
            "zero_contribution_fraction": 1.0 - support_count / denominator,
            "weight_mass_per_population": weight_sum / denominator,
            "weight_mass_per_observed": weight_sum / observed_count if observed_count else None,
        },
        "ci_method": "nonparametric iid bootstrap",
        "iid_assumption": True,
        "independence_verified": False,
        "interval_validity": "diagnostic_only_temporal_dependence_unchecked",
    }


def build_report(memory_path: Path, *, resamples: int = 2_000) -> dict[str, Any]:
    """Read, verify, reconstruct, and return the corrected descriptive report."""

    with OptimizationMemoryV2.open_read_only(memory_path) as memory:
        integrity = memory.integrity()
        chain_verified = memory.verify_chain()
        if not integrity.ok or not chain_verified:
            raise ReconstructionError("read-only source integrity or hash-chain verification failed")
        all_steps = load_steps(memory)
        steps = tuple(step for step in all_steps if step.decision.decision_id.startswith(CAMPAIGN_PREFIX))
        snapshot_hash = memory.snapshot_hash()

    schedule, draw_counts, missing = reconstruct_schedule(steps)
    by_index = {_index(step): step for step in steps}
    splits = {
        "full": tuple(range(ORIGINAL_POPULATION)),
        "train": tuple(range(HOLDOUT_FROM)),
        "holdout": tuple(range(HOLDOUT_FROM, ORIGINAL_POPULATION)),
    }
    report_splits: dict[str, Any] = {}
    for split_name, indices in splits.items():
        rows = [
            population_estimate(
                by_index,
                schedule,
                indices,
                action,
                min_samples=MIN_SAMPLES,
                resamples=resamples,
                seed=17_000 + len(indices) + action_index,
            )
            for action_index, action in enumerate(ACTIONS)
        ]
        observed_count = sum(index in by_index for index in indices)
        report_splits[split_name] = {
            "original_population_count": len(indices),
            "observed_population_count": observed_count,
            "omitted_population_count": len(indices) - observed_count,
            "estimates": rows,
        }

    holdout_rows = report_splits["holdout"]["estimates"]
    inconclusive = [row["action"] for row in holdout_rows if not row["conclusive"]]
    plan, policy, registry = _frozen_plan()
    return {
        "study_id": STUDY_ID,
        "campaign_id": CAMPAIGN_ID,
        "campaign_hash": plan.campaign_hash,
        "policy_hash": policy.policy_hash,
        "registry_hash": registry.registry_hash,
        "database_name": memory_path.name,
        "database_snapshot_sha256": snapshot_hash,
        "chain_verified": chain_verified,
        "original_population_count": ORIGINAL_POPULATION,
        "observed_population_count": len(steps),
        "omitted_population_count": len(missing),
        "omitted_action": "persistent_process",
        "draw_counts": draw_counts,
        "omitted_indices": list(missing),
        "reconstruction": {
            "schedule_verified": True,
            "missing_records_are_only_persistent_process": True,
            "fingerprint_runtime_commit_excluded_from_stable_check": True,
        },
        "data_quality": {
            "runtime_commit_surface_equivalence_verified": False,
            "timing_comparability_note": "schedule metadata agrees across commits; executed dependency equivalence and temporal independence were not revalidated",
            "observed_outcome_records": len(steps),
            "censored_outcome_records": 0,
            "token_identity_breaks_recorded": 0,
            "token_identity_breaks_source": "docs/ARBEITSJOURNAL.md:11694-11698",
            "token_identity_revalidated": False,
            "unrecorded_failed_attempts": [
                {
                    "index": 378,
                    "resolution": "retried_and_recorded",
                    "note": "the transient failed attempt is not a censored label in the append-only corpus",
                }
            ],
            "unrecorded_failed_attempts_source": "docs/ARBEITSJOURNAL.md:11770-11774",
            "censoring_note": "zero censored labels does not imply zero failed attempts; the known index-378 attempt was retried successfully",
        },
        "holdout_from_index": HOLDOUT_FROM,
        "min_samples": MIN_SAMPLES,
        "splits": report_splits,
        "historical_gate": {
            "status": "INCONCLUSIVE",
            "unchanged": True,
            "assessable": not inconclusive,
            "inconclusive_targets": inconclusive,
            "activation_allowed": False,
            "reason": "the preregistered 80-draw holdout remains below the ESS floor for three targets",
        },
        "gate_assessable": not inconclusive,
        "formal_claim": False,
        "learning_claim": False,
        "activation_allowed": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--memory", default=str(PROJECT_ROOT / ".friday-data" / "optimizer-v2.sqlite3"))
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--resamples", type=int, default=2_000)
    args = parser.parse_args(argv)
    if args.min_samples < 1 or args.resamples < 1:
        parser.error("--min-samples and --resamples must be positive")
    if args.min_samples != MIN_SAMPLES:
        parser.error(f"the historical ESS floor is frozen at {MIN_SAMPLES}")
    report = build_report(Path(args.memory), resamples=args.resamples)
    output = Path(__file__).resolve().parent / "corpus_evaluation_corrected.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, split in report["splits"].items():
        print(f"--- {name} original={split['original_population_count']} observed={split['observed_population_count']}")
        for row in split["estimates"]:
            print(f"{row['action']:24}{row['ips']: .6f} ESS={row['effective_samples']:6.1f} {row['status']}")
    print(f"\nreport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
