"""Stdlib-only contract tests for the Q4 method layer."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from dataclasses import replace
import hashlib


ROOT = Path(__file__).parents[1]
pkg = types.ModuleType("ironmule")
pkg.__path__ = [str(ROOT / "ironmule")]
sys.modules.setdefault("ironmule", pkg)

q4 = importlib.import_module("ironmule.q4_methods")


def row(action: str, group: str, reward: float = 1.0, *, stage: str = "KNOB_DELTA", propensity: float = 0.5):
    return q4.ReplayTransition(
        state={"features": (1.0, 0.2), "in_domain": True},
        action_id=action,
        reward=reward,
        next_state={"features": (1.0, 0.2), "in_domain": True},
        terminal=False,
        stage=stage,
        context_id=group,
        group_id=group,
        propensity=propensity,
    )


class Q4MethodsTests(unittest.TestCase):
    def test_frozen_bounds_are_exact(self):
        self.assertAlmostEqual(q4.q_lcb((1.0, 3.0), 3), 2.0 - 1.0 - 0.5)
        expected = min(1.0, (1 + 1) / (4 + 2) + (2.995732273553991 / 8) ** 0.5)
        self.assertAlmostEqual(q4.failure_ucb(1, 4), expected)
        self.assertAlmostEqual(q4.behaviour_score(2.0, 0.5), 2.0 + 0.1 * __import__("math").log(0.5))

    def test_mask_fails_closed_for_domain_support_and_risk_probe(self):
        action = "a" * 64
        rows = tuple(row(action, f"group-{n}") for n in range(3))
        scores = q4.mask_actions({"features": (1.0,), "in_domain": True}, (action,), rows)
        self.assertTrue(scores[0].allowed)
        outside = q4.mask_actions({"features": (1.0,), "in_domain": False}, (action,), rows)
        self.assertFalse(outside[0].allowed)
        self.assertEqual(outside[0].reason, "OUT_OF_DOMAIN")

    def test_fqi_and_strategy_heads_are_separate(self):
        knob = "b" * 64
        strategy = "c" * 64
        rows = tuple(row(knob, f"g-{n}", stage="KNOB_DELTA") for n in range(5))
        rows += tuple(row(strategy, f"s-{n}", stage="STRATEGY_SELECT") for n in range(5))
        policy = q4.EBHCORL().fit(rows)
        self.assertEqual(len(policy.knob_head.rows), 5)
        self.assertEqual(len(policy.strategy_head.rows), 5)
        self.assertEqual(policy.knob_head.iterations, 20)
        self.assertEqual(policy.strategy_head.alpha, 1.0)

    def test_method_values_are_exactly_the_frozen_external_names(self):
        self.assertEqual({item.value for item in q4.Method}, {
            "BASELINE", "CURRENT_COORDINATE", "SEEDED_RANDOM", "BO",
            "SURROGATE", "CONTEXTUAL_BANDIT", "OFFLINE_RL",
        })
        self.assertIs(q4.normalize_method("DETERMINISTIC_BO"), q4.Method.BO)
        self.assertIs(q4.normalize_method("EB_HCORL"), q4.Method.OFFLINE_RL)

    def test_failed_rows_stay_in_risk_critic_but_not_value_fit(self):
        action = q4.KNOB_CANDIDATE_IDS[0]
        safe = tuple(q4.ReplayTransition(
            state={"features": (1.0, 0.1), "in_domain": True}, action_id="a" * 64,
            candidate_id=action, reward=1.0, next_state={"features": (1.0,), "in_domain": True},
            context_id=f"safe-{n}", group_id=f"safe-{n}", propensity=.5,
        ) for n in range(3))
        failed = q4.ReplayTransition(
            state={"features": (1.0, 0.1), "in_domain": True}, action_id="a" * 64,
            candidate_id=action, reward=1.0, next_state={"features": (1.0,), "in_domain": True},
            context_id="failed", group_id="failed", propensity=.5, failed=True,
        )
        head = q4.KnobFQI().fit(safe + (failed,))
        self.assertEqual(len(head.rows), 3)
        self.assertEqual(len(head.risk_rows), 1)
        score = head.score({"features": (1.0,), "in_domain": True}, action)
        self.assertEqual(score.failure_ucb, q4.failure_ucb(1, 4))

    def test_full_q4_state_uses_sparse_ridge_path(self):
        contracts = importlib.import_module("ironmule.q4_contracts")
        context = contracts.Q4Context(
            study_digest="a" * 64, model_digest="b" * 64,
            model_manifest_digest="c" * 64, workload_digest="d" * 64,
            hardware_digest="e" * 64, runtime_digest="f" * 64,
            time_digest="1" * 64, objective_class="LATENCY",
            workload_stratum="homogeneous", arrival_pattern="homogeneous",
        )
        action = q4.KNOB_CANDIDATE_IDS[0]
        state = contracts.Q4State(
            context_id=context.context_id, stage="KNOB_DELTA", step_index=0,
            model_size="4B", memory_bucket="medium", gpu_core_bucket="large",
            prompt_bucket="small", output_bucket="medium", concurrency_bucket="small",
            objective_class="LATENCY", plan_kind="StrictOneShotPlan",
            workload_stratum="homogeneous", arrival_pattern="homogeneous",
            knob_action_id=contracts.KNOB_ACTIONS[0].action_id, strategy_candidate_index=None,
        )
        self.assertGreater(len(state.feature_vector()), 1000)
        rows = tuple(q4.ReplayTransition(
            state=state, action_id="a" * 64, candidate_id=action, reward=1.0,
            next_state=state, context_id=f"ctx-{n}", group_id=f"ctx-{n}", propensity=.5,
        ) for n in range(100))
        fitted = q4.KnobFQI().fit(rows)
        self.assertEqual(len(fitted.ensemble.weights[0]), len(state.feature_vector()) + 1)

    def test_seeded_order_is_byte_stable(self):
        items = ("a" * 64, "b" * 64, "c" * 64)
        self.assertEqual(q4._seeded_order(items, "fixed"), q4._seeded_order(items, "fixed"))
        self.assertNotEqual(q4._seeded_order(items, "fixed"), q4._seeded_order(items, "other"))

    def test_ope_invalid_support_is_explicit(self):
        action = "d" * 64
        trajectories = [(row(action, "one"),)]
        result = q4.weighted_importance_sampling(trajectories, {action: action})
        self.assertEqual(result.status, q4.OPE_UNSUPPORTED)
        self.assertIsNone(result.estimate)

    def test_h17_holdout_and_coordinate_ope_fail_closed(self):
        groups = []
        names = []
        folds = set()
        number = 0
        while len(names) < 5:
            group = hashlib.sha256(f"ope-{number}".encode()).hexdigest()
            fold = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 5
            number += 1
            if fold in folds:
                continue
            folds.add(fold)
            names.append(group)
        for group in names:
            trajectory_id = hashlib.sha256((group + "-trajectory").encode()).hexdigest()
            groups.append(tuple(q4.ReplayTransition(
                state={"features": (1.0,), "in_domain": True}, action_id="a" * 64,
                reward=1.0, next_state={"features": (1.0,), "in_domain": True},
                terminal=step == 16, stage="KNOB_DELTA" if step <= 10 else "STRATEGY_SELECT" if step <= 15 else "REVALIDATE",
                split="Q4_TRAIN", context_id=group, group_id=group, propensity=.5, behaviour_policy_digest="f" * 64, trajectory_id=trajectory_id, step_index=step,
            ) for step in range(17)))
        supported = q4.weighted_importance_sampling(groups, lambda _: "a" * 64)
        self.assertEqual(supported.status, "SUPPORTED")
        dr = q4.grouped_doubly_robust(groups, lambda _: "a" * 64)
        self.assertEqual(dr.status, "SUPPORTED")
        self.assertIsNone(dr.estimate)  # knob/strategy/revalidate stay separate
        self.assertEqual(set(dr.estimates_by_stage), {"KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"})
        holdout = [tuple(replace(item, split="Q4_SEALED_HOLDOUT") for item in group) for group in groups]
        self.assertEqual(q4.weighted_importance_sampling(holdout, lambda _: "a" * 64).status, q4.OPE_UNSUPPORTED)
        coordinate = [tuple(replace(item, propensity=1.0) for item in group) for group in groups]
        self.assertEqual(q4.weighted_importance_sampling(coordinate, lambda _: "a" * 64).status, q4.OPE_UNSUPPORTED)

    def test_contract_join_requires_explicit_state_and_reward(self):
        contract_row = {"action_id": "a" * 64, "state_digest": "b" * 64, "next_state_digest": "c" * 64, "terminal": True, "stage": "KNOB_DELTA", "split": "Q4_TRAIN", "behaviour_propensity": .5}
        with self.assertRaises(q4.DataInsufficientError):
            q4.ReplayTransition.from_contract(contract_row)
        with self.assertRaises(q4.DataInsufficientError):
            q4.join_contract_rows((contract_row,), reward_lookup={})

    def test_strategy_reward_requires_objective_metric_gate(self):
        record = q4.DerivedRewardRecord(
            transition_id="a" * 64, reference_outcome_id="b" * 64,
            candidate_outcome_id="c" * 64, objective_class="LATENCY",
            current_cost=2.0, candidate_cost=1.0, reward=__import__("math").log(2.0),
        )
        self.assertFalse(record.valid_for({"transition_id": "a" * 64, "reference_outcome_id": "b" * 64, "outcome_id": "c" * 64, "stage": "STRATEGY_SELECT"}))
        throughput = q4.DerivedRewardRecord(
            transition_id="a" * 64, reference_outcome_id="b" * 64,
            candidate_outcome_id="c" * 64, objective_class="THROUGHPUT",
            current_cost=2.0, candidate_cost=1.0, reward=__import__("math").log(2.0),
            candidate_physical_tokens_per_second=1.0, p95_inflation=0.11,
        )
        self.assertFalse(throughput.valid_for({"transition_id": "a" * 64, "reference_outcome_id": "b" * 64, "outcome_id": "c" * 64, "stage": "STRATEGY_SELECT"}))

    def test_strict_transition_join_maps_dynamic_delta_to_candidate_slot(self):
        contracts = importlib.import_module("ironmule.q4_contracts")
        context = contracts.Q4Context(
            study_digest="1" * 64, model_digest="2" * 64,
            model_manifest_digest="3" * 64, workload_digest="4" * 64,
            hardware_digest="5" * 64, runtime_digest="6" * 64,
            time_digest="7" * 64, objective_class="LATENCY",
            workload_stratum="homogeneous", arrival_pattern="homogeneous",
        )
        source = contracts.KNOB_ACTIONS[0]
        candidate = contracts.KNOB_CANDIDATES[0]
        target_values = source.as_dict()
        target_values[candidate.changed_field] = candidate.target_value
        target = contracts.KnobAction(**target_values)
        delta = contracts.KnobDelta(
            stage="KNOB_DELTA", source_action_id=source.action_id,
            target_action_id=target.action_id, changed_field=candidate.changed_field,
            target_value=candidate.target_value,
        )
        state = contracts.Q4State(
            context_id=context.context_id, stage="KNOB_DELTA", step_index=0,
            model_size="4B", memory_bucket="medium", gpu_core_bucket="large",
            prompt_bucket="small", output_bucket="medium", concurrency_bucket="small",
            objective_class="LATENCY", plan_kind="StrictOneShotPlan",
            workload_stratum="homogeneous", arrival_pattern="homogeneous",
            knob_action_id=source.action_id, strategy_candidate_index=None,
        )
        next_state = contracts.Q4State(
            context_id=context.context_id, stage="KNOB_DELTA", step_index=1,
            model_size="4B", memory_bucket="medium", gpu_core_bucket="large",
            prompt_bucket="small", output_bucket="medium", concurrency_bucket="small",
            objective_class="LATENCY", plan_kind="StrictOneShotPlan",
            workload_stratum="homogeneous", arrival_pattern="homogeneous",
            knob_action_id=target.action_id, strategy_candidate_index=None,
        )
        transition = contracts.Transition(
            trajectory_id="8" * 64, context=context, stage="KNOB_DELTA", step_index=0,
            horizon=17, state_digest=state.state_digest, action_space="KNOB_DELTA",
            action_id=delta.action_id, previous_action_id=source.action_id,
            outcome_id="9" * 64, next_state_digest=next_state.state_digest, terminal=False,
            split="Q4_TRAIN", evidence_ids=("evidence",), behaviour_propensity=.5,
            behaviour_policy_digest="a" * 64, seed="seed", decision_budget_index=0,
            strategy_candidate_index=None, candidate_id=candidate.candidate_id,
            reference_outcome_id="a" * 64,
        )
        reward = contracts.RewardRecord(
            transition_id=transition.transition_id, candidate_outcome_id=transition.outcome_id,
            reference_outcome_id=transition.reference_outcome_id, objective_class="LATENCY",
            candidate_cost=1.0, reference_cost=2.0, reward=__import__("math").log(2.0),
        )
        joined = q4.join_contract_rows(
            (transition,), state_lookup={state.state_digest: state, next_state.state_digest: next_state},
            derived_rewards={transition.transition_id: reward.to_dict()},
        )
        self.assertEqual(joined[0].candidate_id, candidate.candidate_id)
        self.assertEqual(q4._ope_action_id(joined[0]), candidate.candidate_id)
        head = q4.KnobFQI().fit(joined)
        self.assertEqual(head.action_ids, (candidate.candidate_id,))

    def test_transition_rejects_invalid_propensity(self):
        with self.assertRaises(ValueError):
            row("e" * 64, "g", propensity=0.0)


if __name__ == "__main__":
    unittest.main()
