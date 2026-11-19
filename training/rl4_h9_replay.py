"""Train two RL4 dispatch policies from zero and replay frozen policies on unseen H2 traffic.

Run with `python -m training.rl4_h9_replay --out report.json` from the repository root.
The report embeds both checkpoints and identifies the validation winner. The simulator uses
the documented tuned Tesla T4 H9 costs. Add `--width-followup` to train the RL6 width head
and use fresh test traffic against the first study's frozen checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

from training import ag1_online, rl4_agent as rl4

EPISODES = 16_000
ACTION_SEEDS = (0, 1)
VALIDATION_TRACES = 60
TEST_START = 720_000
TEST_TRACES = 128
BASE = Path(__file__).resolve().parents[1]
FIRST_STUDY = "evidence/runtime/rl4-h9-holdout.json"
PUBLISHED = "ironmule_product/dispatch/rl4s-t4x2-qwen3-1.7b/policy.json"


def evaluate_phase(model, traces, policy):
    place, order, capacity, *width = policy
    first, end = [], []
    choices = Counter()
    width_choices = Counter()
    choose_width = width[0] if width else None

    def observe_order(queue, now):
        action = order(queue, now)
        if len(queue) >= 2:
            choices[action] += 1
        return action

    def observe_width(queue, now):
        chosen = choose_width(queue, now)
        if len(queue) >= 2:
            width_choices[chosen] += 1
        return chosen

    for items in traces:
        requests = rl4.simulate(model, items, place, observe_order, capacity, observe_width if choose_width else None)
        first.extend(r.first - r.arrive for r in requests)
        end.extend(r.end - r.arrive for r in requests)
    first_s, end_s = statistics.fmean(first), statistics.fmean(end)
    metrics = {"mean_first_s": first_s, "mean_end_s": end_s, "mean_cost_s": first_s + end_s,
            "order_when_two_or_more_waiting": {"sjf": choices["sjf"], "fifo": choices["fifo"]}}
    if choose_width:
        metrics["width_cap_when_two_or_more_waiting"] = {n: width_choices[n] for n in rl4.rl2.WIDTH_CHOICES}
    return metrics


def study(width_followup=False):
    started = time.monotonic()
    test_start = 820_000 if width_followup else TEST_START
    h9 = ag1_online.MACHINES["t4"]
    model = rl4.PoolModel(prefill_s=h9["prefill_s"], prefill_s_per_token=h9["prefill_s_per_ktoken"] / 1000,
                          step_s=h9["step_s"], step_s_per_session=h9["step_s_per_session"],
                          release=rl4.release_of({"plan": "h9"}), joins=rl4.joins_of({"plan": "h9"}, "tuned"))
    rate = ag1_online.CAPACITY_RPS
    fixed = rl4.FIXED_ARMS + rl4.WIDTH_ARMS
    rules = rl4.SIMULATOR_ONLY_ARMS
    policies = {name: rl4.arm_policy(name) for name in fixed + rules}
    checkpoints = {}
    phases = rl4.rl2.H2_STATES
    for seed in ACTION_SEEDS:
        name = f"seed_{seed}"
        policy, _ = rl4.train(model, rate, episodes=EPISODES, seed=seed, widths=width_followup)
        policies[name] = rl4._learned(policy)
        validation = rl4.evaluate(model, rate, {name: policies[name]}, VALIDATION_TRACES,
                                  seed0=rl4.VALIDATION_SEED0)[name]
        checkpoints[name] = {"policy": policy.to_json(), "validation_cost_s": dict(zip(phases, validation)),
                             "validation_weighted_cost_s": sum(w * c for w, c in zip(rl4.rl2.WEIGHTS, validation))}
    selected = min(checkpoints, key=lambda name: (checkpoints[name]["validation_weighted_cost_s"], name))
    published = rl4.Policy.from_json(json.loads((BASE / PUBLISHED).read_text()))
    policies["published_spike_checkpoint"] = rl4._learned(published)
    if width_followup:
        first = json.loads((BASE / FIRST_STUDY).read_text())
        frozen = first["checkpoints"][first["selected_by_validation"]]["policy"]
        policies["frozen_without_width_head"] = rl4._learned(rl4.Policy.from_json(frozen))
    test = {}
    for state, phase in enumerate(phases):
        traces = [rl4.traffic(state, test_start + i, rate) for i in range(TEST_TRACES)]
        results = {name: evaluate_phase(model, traces, policy) for name, policy in policies.items()}
        def cost(name):
            return results[name]["mean_cost_s"]

        best_fixed, best_rule = min(fixed, key=cost), min(rules, key=cost)
        test[phase] = {"by_policy": results, "best_fixed": best_fixed, "best_leastwork_rule": best_rule,
                       "selected_cost_change_vs_best_fixed": cost(selected) / cost(best_fixed) - 1,
                       "selected_cost_change_vs_best_leastwork_rule": cost(selected) / cost(best_rule) - 1}
        if width_followup:
            test[phase]["selected_cost_change_vs_frozen_without_width_head"] = (
                cost(selected) / cost("frozen_without_width_head") - 1)
    sources = ("training/ag1_online.py", "training/rl2_headroom.py", "training/rl4_agent.py",
               "training/rl4_h9_replay.py", PUBLISHED)
    if width_followup:
        sources += (FIRST_STUDY,)
    return {
        "schema": "ironmule.lab.rl4_h9_replay.v1",
        "kind": "offline simulation with rounded H9 measured timings; no hardware qualification",
        "protocol": {
            "hardware_model": "H9 tuned Qwen3-1.7B bf16 two-card Tesla T4 pool",
            "world_model": asdict(model), "capacity_rps_per_card": rate, "training": "RL4 REINFORCE",
            "initial_weights": "zero",
            "trainable_weights": {"placement": 10, "order": 4, **({"width": 16} if width_followup else {})},
            "learning_rate": 0.05, "action_seeds": list(ACTION_SEEDS), "episodes_per_seed": EPISODES,
            "training_traffic_indices": [rl4.TRAIN_SEED0, rl4.TRAIN_SEED0 + EPISODES - 1],
            "validation_indices_per_phase": [rl4.VALIDATION_SEED0, rl4.VALIDATION_SEED0 + VALIDATION_TRACES - 1],
            "test_indices_per_phase": [test_start, test_start + TEST_TRACES - 1],
            "test_traces_per_phase": TEST_TRACES, "requests_per_trace": rl4.rl2.REQUESTS,
            "selection": "lowest equal-phase validation mean, seed breaks tie",
            "cost": "mean simulated time to first token plus time to completion from arrival, in seconds",
            "heldout": "same synthetic traces replayed by every frozen policy; group width capped at eight",
            "source_sha256": {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in sources},
            "python_version": platform.python_version(),
        },
        "selected_by_validation": selected, "checkpoints": checkpoints, "test": test,
        "wall_s": round(time.monotonic() - started, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width-followup", action="store_true",
                        help="train group width as an additional action and test on fresh traffic")
    args = parser.parse_args()
    if args.width_followup and args.out.resolve() == (BASE / FIRST_STUDY).resolve():
        parser.error("the follow-up needs a separate output from its first-study control")
    args.out.write_text(json.dumps(study(args.width_followup), indent=2, allow_nan=False) + "\n")
