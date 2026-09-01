"""Unittest coverage for the offline Q4 contracts (no MLX/runtime import)."""

from __future__ import annotations

import unittest

from q4_offline_loader import load_offline_modules


_OFFLINE = load_offline_modules("evidence", "q4_contracts", namespace="q4_contracts_test_modules")
q = _OFFLINE["q4_contracts"]
evidence = _OFFLINE["evidence"]


class ContractTests(unittest.TestCase):
    def test_frozen_action_catalogues_and_ids(self) -> None:
        self.assertEqual(12, len(q.KNOB_ACTIONS))
        self.assertEqual(1024, len(q.ALL_DECLARED_KNOB_ACTIONS))
        self.assertEqual(12, len(q.SCHEDULE_ACTIONS))
        self.assertEqual(5, len(q.ScheduleAction.safe_pool(q.PlanKind.STRICT)))
        self.assertEqual(5, len(q.ScheduleAction.safe_pool(q.PlanKind.REUSABLE)))
        self.assertEqual(["S11", "S12"], [item.label for item in q.ScheduleAction.risk_pool()])
        self.assertIn(q.Q2_CURRENT_ACTION.action_id, {item.action_id for item in q.INTERACTION_KNOB_ANCHORS})
        self.assertEqual("712a6d6ea2cf1bb588fcd74a509f52dac5015b08f3b4bc5ae067232592c3a56a", q.ScheduleAction.from_label("S01").action_id)
        self.assertEqual("3c61c473394070a7bc77bf41518ad463a9a3cf7b4d688f7613368a802d44a123", q.ScheduleAction.from_label("S10").action_id)

    def test_dynamic_single_field_delta_and_q2_anchor(self) -> None:
        base = q.KnobAction.baseline()
        one = q.KnobAction(compiled_fixed_cache=True)
        two = q.KnobAction(compiled_fixed_cache=True, fused_argmax=True)
        first = q.KnobDelta("KNOB_DELTA", base.action_id, one.action_id, "compiled_fixed_cache", True)
        second = q.KnobDelta("KNOB_DELTA", one.action_id, two.action_id, "fused_argmax", True)
        self.assertIn(first.action_id, q.KNOB_DELTA_IDS)
        self.assertIn(second.action_id, q.KNOB_DELTA_IDS)
        self.assertEqual(q.KnobAction.q2_current(), q.Q2_CURRENT_ACTION)
        hybrid = q.HybridAction(two.action_id, q.ScheduleAction.from_label("S01").action_id)
        self.assertEqual(two.action_id, hybrid.knob_action_id)

        with self.assertRaises(q.Q4ValidationError):
            q.Transition("f" * 64, q.Q4Context(*("0" * 64 for _ in range(7)), "LATENCY", "homogeneous", "homogeneous"), "KNOB_DELTA", 0, 17, "1" * 64, "KNOB_DELTA", first.action_id, base.action_id, "2" * 64, "3" * 64, False, "Q4_TRAIN", ("evidence",), 1.0, "4" * 64, "seed", 0, None, candidate_id=q.KNOB_CANDIDATES[1].candidate_id, reference_outcome_id="5" * 64)

    def test_strict_unknown_path_and_executable_rejection(self) -> None:
        with self.assertRaises(q.Q4ValidationError):
            q.Q4Context("/tmp/not-a-digest", *("0" * 64 for _ in range(6)), "LATENCY", "homogeneous", "homogeneous")
        with self.assertRaises(q.Q4ValidationError):
            q.canonical_json({"command": "python dangerous.py"})
        with self.assertRaises(q.Q4ValidationError):
            q.KnobAction(readback_every=3)
        with self.assertRaises(q.Q4ValidationError):
            q.KnobAction(wired_fraction=float("nan"))

    def test_state_feature_vector_is_hash_bound_and_ood_masked(self) -> None:
        ctx = q.Q4Context(*("0" * 64 for _ in range(7)), "LATENCY", "homogeneous", "homogeneous")
        state = q.Q4State(ctx.context_id, "KNOB_DELTA", 0, "4B", "medium", "large", "small", "small", "medium", "LATENCY", "StrictOneShotPlan", "homogeneous", "homogeneous", q.Q2_CURRENT_ACTION.action_id, None)
        self.assertTrue(state.in_domain)
        self.assertEqual(len(q.FEATURE_VECTOR_ORDER), len(state.feature_vector()))
        self.assertEqual(1.0, sum(state.feature_vector()[32:-1]))
        self.assertEqual(state, q.Q4State.from_dict(state.to_dict()))
        ood = q.Q4State(ctx.context_id, "KNOB_DELTA", 0, "27B", "unknown-bucket", "large", "small", "small", "medium", "LATENCY", "StrictOneShotPlan", "homogeneous", "homogeneous", q.Q2_CURRENT_ACTION.action_id, None)
        self.assertFalse(ood.in_domain)
        self.assertEqual(1.0, sum(ood.feature_vector()[32:-1]))

    def test_manifest_group_split_leakage_and_horizon_contract(self) -> None:
        context_id = "a" * 64
        with self.assertRaises(q.Q4ValidationError):
            q.SplitManifest({"Q4_TRAIN": (context_id,), "Q4_VALIDATION": (context_id,), "Q4_SEALED_HOLDOUT": ()}, {"Q4_TRAIN": (), "Q4_VALIDATION": (), "Q4_SEALED_HOLDOUT": ()}, {}, "seed")
        with self.assertRaises(q.Q4ValidationError):
            q.Trajectory(context_id, "Q4_TRAIN", 16, "COMPLETE", tuple("b" * 64 for _ in range(16)), 15, None)
        aborted = q.PartialAbort("PARTIAL_ABORT", "KNOB_DELTA", True, 2, "child failed", "BASE", False, ())
        trajectory = q.Trajectory(context_id, "Q4_TRAIN", 17, "ABORTED", tuple("b" * 64 for _ in range(3)), 2, aborted)
        self.assertEqual(2, trajectory.partial_abort.terminal_step_index)

    def test_outcome_round_trip_and_foreign_signature_contract(self) -> None:
        ref = evidence.ArtifactRef("sample", "a" * 64, evidence.EvidenceQuality.RAW_SAMPLES)
        kwargs = dict(
            raw_sample_refs=(ref,), raw_sample_count=1, total_ns=1.0, prefill_ns=1.0, decode_ns=1.0,
            objective_class="LATENCY", request_sample_count=1, p95_full_response_ms=None, p95_full_response_sample_count=0,
            physical_tokens_per_second=None, p95_physical_tokens_per_second=None, p95_physical_tokens_per_second_sample_count=0,
            full_response_ms_samples=(), physical_tokens_per_second_samples=(), context_id="0" * 64,
            knob_action_id=q.KNOB_ACTIONS[0].action_id, strategy_action_id=None, plan_kind="StrictOneShotPlan",
            samples={"total_ns": (1.0,), "prefill_ns": (1.0,), "decode_ns": (1.0,)}, uncertainty={"total_ns": 0.0},
            logical_token_identity=True, physical_token_identity=True, visible_token_identity=True,
            token_count_identity=True, stop_reason_identity=True, state_identity=True, capacity_identity=True,
            deterministic=True, mlx_active_memory_bytes=1, mlx_peak_memory_bytes=1, rss_peak_bytes=1,
            swap_before_bytes=0, swap_after_bytes=0, swap_delta_bytes=0, timeout=False, crash=False,
            fallbacks=0, worker_status="REAPED", worker_reaped=True, hard_gates_passed=True,
            rollback="NOT_REQUIRED", status="MEASURED", preregistration_sha256="0" * 64,
            code_digest="1" * 64, model_digest="2" * 64, model_manifest_digest="3" * 64,
            environment_digest="4" * 64, workload_digest="5" * 64, researcher_id="r",
            reviewer_id="v", evaluator_id="e",
        )
        outcome = q.Outcome(**kwargs)
        self.assertTrue(outcome.complete_safe)
        self.assertEqual(outcome.outcome_id, q.Outcome.from_dict(outcome.to_dict()).outcome_id)
        foreign = dict(bundle_id="6" * 64, exporter_id="x", host_class="M1", hardware_digest="0" * 64,
                       model_digest="1" * 64, model_manifest_digest="2" * 64, runtime_digest="3" * 64,
                       code_digest="4" * 64, workload_digest="5" * 64, preregistration_sha256="6" * 64,
                       raw_artifacts=(ref,), reviewer_record_sha256="7" * 64, signature_algorithm="Ed25519",
                       signer_key_fingerprint="fingerprint", signature="c2ln", exported_at_utc="2026-09-01T00:00:00Z", public_key_id="key")
        self.assertEqual(foreign["bundle_id"], q.ForeignBundleMetadata(**foreign).bundle_id)

    def test_dataset_binds_states_outcomes_and_transition_groups(self) -> None:
        ctx = q.Q4Context("0" * 64, "2" * 64, "3" * 64, "5" * 64, "0" * 64, "4" * 64, "0" * 64, "LATENCY", "homogeneous", "homogeneous")
        s0 = q.Q4State(ctx.context_id, "KNOB_DELTA", 0, "1B", "small", "small", "small", "small", "small", "LATENCY", "StrictOneShotPlan", "homogeneous", "homogeneous", q.KNOB_ACTIONS[0].action_id, None)
        s1 = q.Q4State(ctx.context_id, "KNOB_DELTA", 1, "1B", "small", "small", "small", "small", "small", "LATENCY", "StrictOneShotPlan", "homogeneous", "homogeneous", q.KNOB_ACTIONS[1].action_id, None)
        delta = q.KnobDelta("KNOB_DELTA", q.KNOB_ACTIONS[0].action_id, q.KNOB_ACTIONS[1].action_id, "compiled_fixed_cache", True)
        ref = evidence.ArtifactRef("sample", "a" * 64, evidence.EvidenceQuality.RAW_SAMPLES)
        outcome = q.Outcome(
            raw_sample_refs=(ref,), raw_sample_count=1, total_ns=1., prefill_ns=1., decode_ns=1.,
            objective_class="LATENCY", request_sample_count=1, p95_full_response_ms=None, p95_full_response_sample_count=0,
            physical_tokens_per_second=None, p95_physical_tokens_per_second=None, p95_physical_tokens_per_second_sample_count=0,
            full_response_ms_samples=(), physical_tokens_per_second_samples=(), context_id=ctx.context_id,
            knob_action_id=q.KNOB_ACTIONS[1].action_id, strategy_action_id=None, plan_kind="StrictOneShotPlan",
            samples={"total_ns": (1.,), "prefill_ns": (1.,), "decode_ns": (1.,)}, uncertainty={"total_ns": 0.},
            logical_token_identity=True, physical_token_identity=True, visible_token_identity=True,
            token_count_identity=True, stop_reason_identity=True, state_identity=True, capacity_identity=True, deterministic=True,
            mlx_active_memory_bytes=1, mlx_peak_memory_bytes=1, rss_peak_bytes=1, swap_before_bytes=0, swap_after_bytes=0, swap_delta_bytes=0,
            timeout=False, crash=False, fallbacks=0, worker_status="REAPED", worker_reaped=True, hard_gates_passed=True,
            rollback="NOT_REQUIRED", status="MEASURED", preregistration_sha256="0" * 64, code_digest="1" * 64,
            model_digest="2" * 64, model_manifest_digest="3" * 64, environment_digest="4" * 64, workload_digest="5" * 64,
            researcher_id="r", reviewer_id="v", evaluator_id="e",
        )
        reference_payload = outcome.to_dict()
        reference_payload["knob_action_id"] = q.KNOB_ACTIONS[0].action_id
        reference_payload["total_ns"] = 2.0
        reference_payload["samples"]["total_ns"] = [2.0]
        reference_payload["outcome_id"] = ""
        reference = q.Outcome.from_dict(reference_payload)
        transition_template = q.Transition("f" * 64, ctx, "KNOB_DELTA", 0, 17, s0.state_digest, "KNOB_DELTA", delta.action_id, q.KNOB_ACTIONS[0].action_id, outcome.outcome_id, s1.state_digest, False, "Q4_TRAIN", ("sample",), 1., "6" * 64, "seed", 0, None, candidate_id=q.KNOB_CANDIDATES[0].candidate_id, reference_outcome_id=reference.outcome_id)
        trajectory = q.Trajectory(ctx.context_id, "Q4_TRAIN", 17, "RUNNING", (transition_template.transition_id,), None, None)
        transition = q.Transition(trajectory.trajectory_id, ctx, "KNOB_DELTA", 0, 17, s0.state_digest, "KNOB_DELTA", delta.action_id, q.KNOB_ACTIONS[0].action_id, outcome.outcome_id, s1.state_digest, False, "Q4_TRAIN", ("sample",), 1., "6" * 64, "seed", 0, None, candidate_id=q.KNOB_CANDIDATES[0].candidate_id, reference_outcome_id=reference.outcome_id, transition_id=transition_template.transition_id)
        manifest = q.SplitManifest({"Q4_TRAIN": (ctx.context_id,), "Q4_VALIDATION": (), "Q4_SEALED_HOLDOUT": ()}, {"Q4_TRAIN": (), "Q4_VALIDATION": (), "Q4_SEALED_HOLDOUT": ()}, {}, "seed")
        artifact = q.ArtifactRecord("sample", "a" * 64, "RAW_SAMPLES", "B36", "Q3_SEALED_HOLDOUT", "complete", "raw")
        pools = {"knob": q.KNOB_ACTIONS, "strict_safe": q.ScheduleAction.safe_pool("StrictOneShotPlan"), "reusable_safe": q.ScheduleAction.safe_pool("ReusableSessionPlan"), "risk": q.ScheduleAction.risk_pool()}
        dataset = q.Dataset("0" * 64, (artifact,), pools, contexts=(ctx,), transitions=(transition,), outcomes=(outcome, reference), states=(s0, s1), trajectories=(trajectory,), split_manifest=manifest, seed_manifest={"seed": "seed"}, no_invented_performance=True)
        self.assertEqual(dataset.dataset_id, q.Dataset.from_dict(dataset.to_dict()).dataset_id)
        rewards = dataset.derive_rewards()
        self.assertEqual(1, len(rewards))
        self.assertGreater(next(iter(rewards.values())).reward, 0.0)
        p95_payload = outcome.to_dict()
        p95_payload.update({"objective_class": "THROUGHPUT", "p95_physical_tokens_per_second": 2.0, "p95_physical_tokens_per_second_sample_count": 19})
        with self.assertRaises(q.Q4ValidationError):
            q.Outcome.from_dict(p95_payload).metric_value("STRATEGY_SELECT")
        with self.assertRaises(q.Q4ValidationError):
            q.Dataset("0" * 64, (artifact,), pools, contexts=(ctx,), transitions=(), outcomes=(outcome,), states=(transition,), trajectories=(), split_manifest=manifest, seed_manifest={"seed": "seed"}, no_invented_performance=True)

    def test_strategy_rewards_use_same_knob_base_reference_and_p95_guard(self) -> None:
        ctx = q.Q4Context("0" * 64, "2" * 64, "3" * 64, "5" * 64, "0" * 64, "4" * 64, "0" * 64, "THROUGHPUT", "homogeneous", "homogeneous")
        base = q.KNOB_ACTIONS[0]
        s02 = q.ScheduleAction.from_label("S02")
        s01 = q.ScheduleAction.from_label("S01")
        state11 = q.Q4State(ctx.context_id, "STRATEGY_SELECT", 11, "1B", "small", "small", "small", "small", "small", "THROUGHPUT", "StrictOneShotPlan", "homogeneous", "homogeneous", base.action_id, 0)
        state12 = q.Q4State(ctx.context_id, "STRATEGY_SELECT", 12, "1B", "small", "small", "small", "small", "small", "THROUGHPUT", "StrictOneShotPlan", "homogeneous", "homogeneous", base.action_id, 1)
        ref = evidence.ArtifactRef("sample", "a" * 64, evidence.EvidenceQuality.RAW_SAMPLES)
        common = dict(raw_sample_refs=(ref,), raw_sample_count=1, total_ns=1., prefill_ns=1., decode_ns=1., objective_class="THROUGHPUT", request_sample_count=20, p95_full_response_ms=100., p95_full_response_sample_count=20, physical_tokens_per_second=10., p95_physical_tokens_per_second=None, p95_physical_tokens_per_second_sample_count=0, full_response_ms_samples=(100.,) * 20, physical_tokens_per_second_samples=(10.,), context_id=ctx.context_id, knob_action_id=base.action_id, plan_kind="StrictOneShotPlan", samples={"total_ns": (1.,), "prefill_ns": (1.,), "decode_ns": (1.,)}, uncertainty={"physical_tokens_per_second": 0.}, logical_token_identity=True, physical_token_identity=True, visible_token_identity=True, token_count_identity=True, stop_reason_identity=True, state_identity=True, capacity_identity=True, deterministic=True, mlx_active_memory_bytes=1, mlx_peak_memory_bytes=1, rss_peak_bytes=1, swap_before_bytes=0, swap_after_bytes=0, swap_delta_bytes=0, timeout=False, crash=False, fallbacks=0, worker_status="REAPED", worker_reaped=True, hard_gates_passed=True, rollback="NOT_REQUIRED", status="MEASURED", preregistration_sha256="0" * 64, code_digest="1" * 64, model_digest="2" * 64, model_manifest_digest="3" * 64, environment_digest="4" * 64, workload_digest="5" * 64, researcher_id="r", reviewer_id="v", evaluator_id="e")
        candidate_payload = dict(common, physical_tokens_per_second=20., physical_tokens_per_second_samples=(20.,), strategy_action_id=s02.action_id)
        reference_payload = dict(common, strategy_action_id=s01.action_id)
        candidate = q.Outcome(**candidate_payload)
        reference = q.Outcome(**reference_payload)
        template = q.Transition("f" * 64, ctx, "STRATEGY_SELECT", 11, 17, state11.state_digest, "STRATEGY_SELECT", s02.action_id, base.action_id, candidate.outcome_id, state12.state_digest, False, "Q4_TRAIN", ("sample",), 1., "6" * 64, "seed", 11, 0, reference_outcome_id=reference.outcome_id)
        trajectory = q.Trajectory(ctx.context_id, "Q4_TRAIN", 17, "RUNNING", (template.transition_id,), None, None)
        transition = q.Transition(trajectory.trajectory_id, ctx, "STRATEGY_SELECT", 11, 17, state11.state_digest, "STRATEGY_SELECT", s02.action_id, base.action_id, candidate.outcome_id, state12.state_digest, False, "Q4_TRAIN", ("sample",), 1., "6" * 64, "seed", 11, 0, reference_outcome_id=reference.outcome_id, transition_id=template.transition_id)
        artifact = q.ArtifactRecord("sample", "a" * 64, "RAW_SAMPLES", "B36", "Q3_SEALED_HOLDOUT", "complete", "raw")
        manifest = q.SplitManifest({"Q4_TRAIN": (ctx.context_id,), "Q4_VALIDATION": (), "Q4_SEALED_HOLDOUT": ()}, {"Q4_TRAIN": (), "Q4_VALIDATION": (), "Q4_SEALED_HOLDOUT": ()}, {}, "seed")
        pools = {"knob": q.KNOB_ACTIONS, "strict_safe": q.ScheduleAction.safe_pool("StrictOneShotPlan"), "reusable_safe": q.ScheduleAction.safe_pool("ReusableSessionPlan"), "risk": q.ScheduleAction.risk_pool()}
        panel = q.PanelCell(ctx.context_id, base.action_id, s02.action_id, candidate.outcome_id, reference.outcome_id)
        dataset = q.Dataset("0" * 64, (artifact,), pools, contexts=(ctx,), transitions=(transition,), outcomes=(candidate, reference), states=(state11, state12), trajectories=(trajectory,), panel_cells=(panel,), split_manifest=manifest, seed_manifest={"seed": "seed"}, no_invented_performance=True)
        reward = next(iter(dataset.derive_rewards().values()))
        self.assertGreater(reward.reward, 0.0)
        wrong_base = q.Outcome.from_dict(dict(reference.to_dict(), strategy_action_id=s02.action_id, outcome_id=""))
        with self.assertRaises(q.Q4ValidationError):
            q.Dataset("0" * 64, (artifact,), pools, contexts=(ctx,), transitions=(transition,), outcomes=(candidate, wrong_base), states=(state11, state12), trajectories=(trajectory,), panel_cells=(q.PanelCell(ctx.context_id, base.action_id, s02.action_id, candidate.outcome_id, wrong_base.outcome_id),), split_manifest=manifest, seed_manifest={"seed": "seed"}, no_invented_performance=True)
        low_p95 = dict(candidate.to_dict(), p95_full_response_sample_count=19, outcome_id="")
        with self.assertRaises(q.Q4ValidationError):
            q.Outcome.from_dict(low_p95)
        bad_reference = q.Outcome.from_dict(dict(reference.to_dict(), knob_action_id=q.KNOB_ACTIONS[1].action_id, outcome_id=""))
        bad_transition = q.Transition(trajectory.trajectory_id, ctx, "STRATEGY_SELECT", 11, 17, state11.state_digest, "STRATEGY_SELECT", s02.action_id, base.action_id, candidate.outcome_id, state12.state_digest, False, "Q4_TRAIN", ("sample",), 1., "6" * 64, "seed", 11, 0, reference_outcome_id=bad_reference.outcome_id)
        bad_dataset = object.__new__(q.Dataset)
        for name, value in {"contexts": (ctx,), "states": (state11, state12), "trajectories": (trajectory,), "transitions": (bad_transition,), "outcomes": (candidate, bad_reference), "risk_observations": (), "source_artifacts": (artifact,)}.items():
            object.__setattr__(bad_dataset, name, value)
        self.assertIn(bad_transition.transition_id, bad_dataset.derive_rewards().excluded)
        final_state = q.Q4State(ctx.context_id, "REVALIDATE", 16, "1B", "small", "small", "small", "small", "small", "THROUGHPUT", "StrictOneShotPlan", "homogeneous", "homogeneous", q.KNOB_ACTIONS[1].action_id, None)
        final_candidate = q.Outcome.from_dict(dict(candidate.to_dict(), knob_action_id=q.KNOB_ACTIONS[1].action_id, outcome_id=""))
        final_template = q.Transition("b" * 64, ctx, "REVALIDATE", 16, 17, final_state.state_digest, "REVALIDATE", s02.action_id, s02.action_id, final_candidate.outcome_id, final_state.state_digest, True, "Q4_TRAIN", ("sample",), 1., "7" * 64, "seed", 16, None, reference_outcome_id=reference.outcome_id)
        final_dataset = object.__new__(q.Dataset)
        for name, value in {"contexts": (ctx,), "states": (final_state,), "transitions": (final_template,), "outcomes": (final_candidate, reference), "trajectories": (), "risk_observations": (), "source_artifacts": ()}.items():
            object.__setattr__(final_dataset, name, value)
        self.assertGreater(next(iter(final_dataset.derive_rewards().values())).reward, 0.0)

    def test_readiness_keeps_abort_support_outside_safe_h17_denominator(self) -> None:
        """72 safe H17 trajectories plus retained failure support is admissible."""
        from types import SimpleNamespace

        contexts = tuple(SimpleNamespace(context_id=f"{index + 1:064x}", workload_stratum="homogeneous") for index in range(24))
        split_contexts = {"Q4_TRAIN": tuple(item.context_id for item in contexts[:12]), "Q4_VALIDATION": tuple(item.context_id for item in contexts[12:18]), "Q4_SEALED_HOLDOUT": tuple(item.context_id for item in contexts[18:])}
        model_map = {}
        for split, ids in split_contexts.items():
            for index, context_id in enumerate(ids):
                model_map[context_id] = ("1B", "4B", "12B")[index % 3]
        manifest = SimpleNamespace(split_contexts=tuple(split_contexts.items()), model_size_by_context=tuple(model_map.items()), stratum_by_context=tuple((context.context_id, context.workload_stratum) for context in contexts))
        safe_outcome = SimpleNamespace(complete_safe=True)
        outcomes = {}
        trajectories = []
        transitions = []
        for trajectory_index in range(72):
            context = contexts[trajectory_index // 3]
            trajectory_id = f"{trajectory_index + 100:064x}"
            transition_ids = []
            for step in range(17):
                transition_id = f"{trajectory_index * 17 + step + 1000:064x}"
                transition_ids.append(transition_id)
                stage = "KNOB_DELTA" if step < 11 else "STRATEGY_SELECT" if step < 16 else "REVALIDATE"
                if stage == "STRATEGY_SELECT":
                    action_id = q.ScheduleAction.safe_pool("StrictOneShotPlan")[step - 11].action_id
                else:
                    action_id = q.KNOB_DELTA_IDS.__iter__().__next__() if step < 11 else q.ScheduleAction.from_label("S01").action_id
                outcome_id = f"{trajectory_index * 17 + step + 5000:064x}"
                outcomes[outcome_id] = SimpleNamespace(outcome_id=outcome_id, complete_safe=True)
                transitions.append(SimpleNamespace(trajectory_id=trajectory_id, context=context, stage=stage, action_id=action_id, outcome_id=outcome_id, partial_abort=None))
            trajectories.append(SimpleNamespace(trajectory_id=trajectory_id, context_id=context.context_id, trajectory_status="COMPLETE", transition_ids=tuple(transition_ids)))
        # Retained aborted evidence adds failure/risk support without changing
        # the 72 complete safe trajectories or their 1224-transition count.
        abort_tid = "f" * 64
        abort_outcome_id = "e" * 64
        outcomes[abort_outcome_id] = SimpleNamespace(outcome_id=abort_outcome_id, complete_safe=False)
        transitions.append(SimpleNamespace(trajectory_id=abort_tid, context=contexts[0], stage="KNOB_DELTA", action_id=next(iter(q.KNOB_DELTA_IDS)), outcome_id=abort_outcome_id, partial_abort=object()))
        trajectories.append(SimpleNamespace(trajectory_id=abort_tid, context_id=contexts[0].context_id, trajectory_status="ABORTED", transition_ids=("d" * 64,)))
        panels = [SimpleNamespace(context_id=context.context_id, knob_action_id=knob.action_id, strategy_action_id=strategy.action_id) for context in contexts for knob in q.KNOB_ACTIONS for strategy in q.ScheduleAction.safe_pool("StrictOneShotPlan")]
        dataset = object.__new__(q.Dataset)
        for name, value in {"contexts": contexts, "states": (), "trajectories": tuple(trajectories), "transitions": tuple(transitions), "outcomes": tuple(outcomes.values()), "risk_observations": (), "panel_cells": tuple(panels), "split_manifest": manifest}.items():
            object.__setattr__(dataset, name, value)
        ready, reasons = dataset._rl_readiness()
        self.assertTrue(ready, reasons)
        object.__setattr__(dataset, "trajectories", tuple(item for item in trajectories if item.trajectory_status == "COMPLETE"))
        object.__setattr__(dataset, "transitions", tuple(item for item in transitions if item.trajectory_id != abort_tid))
        object.__setattr__(dataset, "outcomes", tuple(item for item in outcomes.values() if item.outcome_id != abort_outcome_id))
        ready, reasons = dataset._rl_readiness()
        self.assertFalse(ready)
        self.assertTrue(any("risk/failure support" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
