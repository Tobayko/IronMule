"""Stdlib-only tests for equal-budget replay and the shadow hybrid."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
pkg = types.ModuleType("ironmule")
pkg.__path__ = [str(ROOT / "ironmule")]
sys.modules.setdefault("ironmule", pkg)

contracts = importlib.import_module("ironmule.q4_contracts")
methods = importlib.import_module("ironmule.q4_methods")
optimizer = importlib.import_module("ironmule.q4_optimizer")


class Q4OptimizerTests(unittest.TestCase):
    def test_equal_budget_is_eleven_plus_five(self):
        panel = optimizer.ActionPanel()
        ledger = optimizer.equal_budget_plan(methods.Method.SEEDED_RANDOM, panel, plan_kind="StrictOneShotPlan")
        self.assertEqual(len(ledger.knob_actions), 11)
        self.assertEqual(len(ledger.strategy_actions), 5)
        self.assertEqual(ledger.total, 16)
        self.assertEqual(len(set(ledger.knob_actions)), 11)
        self.assertEqual(optimizer.equal_budget_plan(methods.Method.BASELINE, panel, plan_kind="StrictOneShotPlan").total, 0)

    def test_all_comparator_rankings_use_supplied_rows(self):
        panel = optimizer.ActionPanel()
        first, second = panel.knob_candidates[:2]
        rows = ({"action_id": first.action_id, "candidate_id": first.action_id, "reward": 0.1, "group_id": "g1"},
                {"action_id": second.action_id, "candidate_id": second.action_id, "reward": 2.0, "group_id": "g1"})
        surrogate = optimizer.equal_budget_plan(methods.Method.SURROGATE, panel, rows, plan_kind="StrictOneShotPlan")
        bo = optimizer.equal_budget_plan(methods.Method.BO, panel, rows, plan_kind="StrictOneShotPlan")
        self.assertEqual(surrogate.knob_actions[0], second.action_id)
        self.assertEqual(bo.knob_actions[0], second.action_id)

    def test_dynamic_neighbors_are_one_field_only(self):
        panel = optimizer.ActionPanel()
        neighbors = optimizer.legal_knob_neighbors(contracts.KNOB_ACTIONS[0], panel)
        self.assertTrue(neighbors)
        self.assertTrue(all(optimizer._one_field_diff(contracts.KNOB_ACTIONS[0], item.action) for item in neighbors))

    def test_hybrid_fails_closed_without_exact_cross_product(self):
        hybrid = optimizer.HybridOptimizer()
        result = hybrid.recommend(objective_class="LATENCY", plan_kind="StrictOneShotPlan")
        self.assertEqual(result.status, optimizer.SHADOW_RECOMMENDATION)
        self.assertEqual(result.safety_decision, "BASE_FALLBACK")
        self.assertFalse(result.eligible_for_future_revalidation)
        self.assertEqual(result.knob_action_id, contracts.KNOB_ACTIONS[0].action_id)
        self.assertEqual(result.strategy_action_id, contracts.ScheduleAction.from_label("S01").action_id)

    def test_hybrid_requires_dataset_backed_sixty_pairs_and_never_executes(self):
        panel = optimizer.ActionPanel()
        strategies = panel.strategies("StrictOneShotPlan")
        pairs = [(knob.action_id, strategy.action_id) for knob in panel.interaction_anchors for strategy in strategies]
        result = optimizer.HybridOptimizer().recommend(
            objective_class="THROUGHPUT",
            plan_kind="StrictOneShotPlan",
            knob_action_id=panel.interaction_anchors[1].action_id,
            strategy_action_id=strategies[2].action_id,
            measured_pairs=pairs,
            evidence_ids=("e" * 64,),
        )
        self.assertEqual(result.status, optimizer.SHADOW_RECOMMENDATION)
        self.assertEqual(result.safety_decision, "BASE_FALLBACK")
        self.assertFalse(result.eligible_for_future_revalidation)
        self.assertFalse(hasattr(optimizer.HybridOptimizer, "execute"))
        self.assertEqual(result.objective_class, "THROUGHPUT")

    def test_hybrid_validates_dynamic_knob_and_plan_binding(self):
        panel = optimizer.ActionPanel()
        pairs = [(knob.action_id, strategy.action_id) for knob in panel.interaction_anchors for strategy in panel.strategies("ReusableSessionPlan")]
        reusable = panel.strategies("ReusableSessionPlan")
        result = optimizer.HybridOptimizer().recommend(
            objective_class="LATENCY", plan_kind="ReusableSessionPlan",
            knob_action_id=contracts.Q2_CURRENT_ACTION.action_id,
            strategy_action_id=reusable[0].action_id,
            measured_pairs=pairs, evidence_ids=("a" * 64,),
        )
        self.assertEqual(result.knob_action_id, contracts.KNOB_ACTIONS[0].action_id)
        self.assertFalse(result.eligible_for_future_revalidation)
        with self.assertRaises((ValueError, TypeError)):
            optimizer.HybridRecommendation(
                schema=optimizer.HybridRecommendation.SCHEMA, status=optimizer.SHADOW_RECOMMENDATION,
                knob_action_id="f" * 64, strategy_action_id=contracts.ScheduleAction.from_label("S01").action_id,
                stage_order=("KNOB_DELTA", "STRATEGY_SELECT", "REVALIDATE"), objective_class="LATENCY",
                plan_kind="StrictOneShotPlan", knob_score=None, strategy_score=None,
                knob_uncertainty=None, strategy_uncertainty=None, safety_decision="BASE_FALLBACK",
                eligible_for_future_revalidation=False,
            )

    def test_shadow_envelope_requires_external_signer_and_verifier(self):
        recommendation = optimizer.HybridOptimizer().recommend(objective_class="LATENCY", plan_kind="StrictOneShotPlan")
        unsigned = optimizer.HybridOptimizer().sign_recommendation(recommendation)
        self.assertFalse(unsigned.verified)
        class Signer:
            key_id = "local-test-key"
            algorithm = "Ed25519"
            def sign(self, payload):
                self.payload = payload
                return "signature"
        class Verifier:
            def verify(self, payload, signature, key_id):
                return payload and signature == "signature" and key_id == "local-test-key"
        envelope = optimizer.HybridOptimizer().sign_recommendation(recommendation, signer=Signer(), verifier=Verifier())
        self.assertTrue(envelope.verified)
        self.assertFalse(envelope.eligible)
        loaded = optimizer.ShadowRecommendationEnvelope.from_dict(envelope.to_dict())
        self.assertFalse(loaded.verified)
        self.assertFalse(loaded.eligible)

    def test_mapping_cannot_claim_evaluator_safety_and_non_supported_ope_blocks(self):
        fake = {"outcome": {"complete_safe": True, "raw_sample_refs": ["a" * 64], "raw_sample_count": 1}}
        self.assertFalse(optimizer._safe_qualified_evidence(fake))
        decision = optimizer.decide_rl(
            advantage=.5, lower_bound=.5, rl_time_to_best=1, simpler_time_to_best=2,
            rl_experiments_to_best=1, simpler_experiments_to_best=2,
            rl_regression_rate=0, simpler_regression_rate=1, coverage=.9,
            support_pass=True, model_strata_pass=True, data_complete=True, safety_ok=True,
            ope_status="UNKNOWN",
        )
        self.assertEqual(decision, optimizer.Decision.OPE_UNSUPPORTED)

    def test_metrics_require_sealed_complete_panels_and_frozen_decision_gates(self):
        oracle_rows = []
        method_rows = []
        baseline = {}
        ref = contracts.ArtifactRef("metric", "a" * 64, contracts.EvidenceQuality.RAW_SAMPLES)
        outcome_kwargs = dict(
            raw_sample_refs=(ref,), raw_sample_count=1, total_ns=1.0, prefill_ns=1.0, decode_ns=1.0,
            objective_class="LATENCY", request_sample_count=1, p95_full_response_ms=None, p95_full_response_sample_count=0,
            physical_tokens_per_second=None, p95_physical_tokens_per_second=None, p95_physical_tokens_per_second_sample_count=0,
            knob_action_id=contracts.KNOB_ACTIONS[0].action_id, strategy_action_id=None, plan_kind="StrictOneShotPlan",
            samples={"total_ns": (1.0,), "prefill_ns": (1.0,), "decode_ns": (1.0,)}, uncertainty={"total_ns": 1.0},
            logical_token_identity=True, physical_token_identity=True, visible_token_identity=True,
            token_count_identity=True, stop_reason_identity=True, state_identity=True, capacity_identity=True, deterministic=True,
            mlx_active_memory_bytes=1, mlx_peak_memory_bytes=1, rss_peak_bytes=1, swap_before_bytes=0, swap_after_bytes=0, swap_delta_bytes=0,
            timeout=False, crash=False, fallbacks=0, worker_status="REAPED", worker_reaped=True, hard_gates_passed=True,
            rollback="NOT_REQUIRED", status="MEASURED", preregistration_sha256="b" * 64, code_digest="c" * 64,
            model_digest="d" * 64, model_manifest_digest="e" * 64, environment_digest="f" * 64, workload_digest="1" * 64,
            researcher_id="r", reviewer_id="v", evaluator_id="e",
        )
        for index, context in enumerate(("1" * 64, "2" * 64)):
            baseline[context] = 10.0
            for action, cost in (("a" * 64, 10.0), ("b" * 64, 8.0)):
                outcome = contracts.Outcome(**{**outcome_kwargs, "knob_action_id": action, "total_ns": cost, "samples": {"total_ns": (cost,), "prefill_ns": (1.0,), "decode_ns": (1.0,)}})
                oracle_rows.append(optimizer.EvaluatorCell(context, action, outcome, cost))
            method_rows.append(optimizer.EvaluatorCell(context, "b" * 64, contracts.Outcome(**{**outcome_kwargs, "knob_action_id": "b" * 64, "total_ns": 8.0, "samples": {"total_ns": (8.0,), "prefill_ns": (1.0,), "decode_ns": (1.0,)}}), 8.0))
            method_rows.append(optimizer.EvaluatorCell(context, "a" * 64, contracts.Outcome(**{**outcome_kwargs, "knob_action_id": "a" * 64, "total_ns": 1.0, "samples": {"total_ns": (1.0,), "prefill_ns": (1.0,), "decode_ns": (1.0,)}}), 1.0))
        plan = optimizer.BudgetLedger(("b" * 64,), ())
        report = optimizer.evaluate_method(methods.Method.BO, method_rows, oracle_rows=oracle_rows, baseline_cost_by_context=baseline, plan=plan)
        self.assertEqual(report.status, "COMPUTABLE")
        self.assertEqual(report.regression["context_count"], 2)
        blocked = optimizer.evaluate_method(methods.Method.BO, method_rows, oracle_rows=oracle_rows, plan=plan, split="Q4_VALIDATION")
        self.assertEqual(blocked.decision, optimizer.Decision.DATA_INSUFFICIENT.value)
        self.assertEqual(optimizer.decide_rl(
            advantage=.04, lower_bound=.03, rl_time_to_best=1, simpler_time_to_best=2,
            rl_experiments_to_best=1, simpler_experiments_to_best=2,
            rl_regression_rate=.01, simpler_regression_rate=.03, coverage=.9,
            support_pass=True, model_strata_pass=True, data_complete=True, safety_ok=True,
        ), optimizer.Decision.RL_WINS)

    def test_replay_is_deterministic(self):
        engine = optimizer.ReplayEngine()
        first = engine.replay(methods.Method.SEEDED_RANDOM, plan_kind="ReusableSessionPlan").to_dict()
        second = engine.replay(methods.Method.SEEDED_RANDOM, plan_kind="ReusableSessionPlan").to_dict()
        self.assertEqual(first, second)

    def test_offline_rl_replay_requires_full_strict_dataset(self):
        report = optimizer.ReplayEngine().replay(methods.Method.OFFLINE_RL, rows=(), plan_kind="StrictOneShotPlan")
        self.assertEqual(report.status, "NOT_APPLICABLE")
        self.assertEqual(report.ledger.total, 0)


if __name__ == "__main__":
    unittest.main()
