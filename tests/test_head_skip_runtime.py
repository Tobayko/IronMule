"""Policy, execution, and bounded qualification tests for head-skip runtime."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from experiments.head_skip_formal import study
from friday_head_skip_runtime import benchmark, policy, provenance
from friday_head_skip_runtime.constants import (
    BASELINE_PLAN,
    DEFAULT_RUNTIME_DATABASE_PATH,
    FORMAL_CANDIDATE_ID,
    FORMAL_CHAIN_HEAD,
    FORMAL_CONFIRMATION_SEAL_SHA256,
    FORMAL_DATABASE_SHA256,
    FORMAL_DECISION_SHA256,
    FORMAL_PREREGISTRATION_SHA256,
    FORMAL_PROVENANCE_SHA256,
    FORMAL_STUDY_ID,
    GPU_MAX_EXTRA_PEAK_BYTES,
    GPU_MAX_RATIO,
    GPU_MEASUREMENT_ORDERS,
    GPU_RUN_ID,
    HEAD_SKIP_PLAN,
    MODEL_ID,
    MODEL_REVISION,
    OUTPUT_TOKENS,
    POLICY_MAX_INCREMENTAL_NS,
    POLICY_MAX_LOAD_NS,
    POLICY_MAX_MEDIAN_NS,
    POLICY_MAX_P95_NS,
    POLICY_RUN_ID,
    PREFILL_CHUNK,
    PROMPT_CONTENT_SHA256,
    PROMPT_TOKENS,
    QUALIFICATION_ID,
)
from friday_head_skip_runtime.executor import (
    GenerationOutput,
    GenerationRequest,
    RuntimeController,
    RuntimeExecutionError,
)
from friday_evidence.canonical import canonical_sha256
from friday_head_skip_runtime.policy import (
    FormalSnapshot,
    PolicyEvidence,
    REGISTERED_SCOPE,
    decision_for,
    load_gpu_qualification_policy,
    load_policy,
    load_runtime_policy,
)


def _sealed_identity() -> dict[str, object]:
    return {
        "git_dirty": False,
        "code_sha256": "1" * 64,
        "spec_sha256": "2" * 64,
        "environment_sha256": "3" * 64,
        "hardware_sha256": "4" * 64,
        "model_sha256": "5" * 64,
    }


def _formal_rows() -> list[dict[str, object]]:
    provenance = _sealed_identity()
    preregistration: dict[str, object] = {
        "kind": "preregistration",
        "study_id": FORMAL_STUDY_ID,
        "candidate_id": FORMAL_CANDIDATE_ID,
        "preregistration_sha256": FORMAL_PREREGISTRATION_SHA256,
        "provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "provenance": provenance,
        "study_specification": {
            "scope": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sampling": "greedy_fixed_horizon",
                "prompt_logprobs": False,
                "claim_scope": "one-device-one-model-one-prompt-one-prefill-plan",
            },
            "workload": {
                "prompt_content_sha256": PROMPT_CONTENT_SHA256,
                "prompt_tokens": PROMPT_TOKENS,
                "prefill_chunk": PREFILL_CHUNK,
                "batch": 1,
                "correctness_tokens": OUTPUT_TOKENS,
                "arm_a": BASELINE_PLAN,
                "arm_b": HEAD_SKIP_PLAN,
                "primary_endpoint": "paired_prefill_duration_ratio_B_over_A",
            },
        },
    }
    decision: dict[str, object] = {
        "kind": "study_decision",
        "study_id": FORMAL_STUDY_ID,
        "candidate_id": FORMAL_CANDIDATE_ID,
        "decision_sha256": FORMAL_DECISION_SHA256,
        "preregistration_sha256": FORMAL_PREREGISTRATION_SHA256,
        "provenance_sha256": FORMAL_PROVENANCE_SHA256,
        "confirmation_seal_sha256": FORMAL_CONFIRMATION_SEAL_SHA256,
        "status": "head_skip_gain_confirmed",
        "action": "permit_bounded_architecture_review",
        "claim": "prefill_head_skip_is_faster_beyond_mde",
        "claim_scope": "one-device-one-model-one-prompt-one-prefill-plan",
        "formal_claim": True,
        "gates": {
            "all_sessions_token_identical": True,
            "equivalence_all_splits": False,
            "gain_all_splits": True,
            "regression_all_splits": False,
        },
    }
    rows = [preregistration]
    rows.extend({"kind": "calibration_session"} for _ in range(6))
    rows.append({"kind": "calibration_summary"})
    rows.append({"kind": "confirmation_seal"})
    rows.extend({"kind": "confirmation_session"} for _ in range(6))
    rows.append(decision)
    return rows


def _load(rows=None, identity=None) -> PolicyEvidence:
    selected = _formal_rows() if rows is None else rows
    current = _sealed_identity() if identity is None else identity
    return load_policy(
        "ignored.sqlite3",
        evidence_reader=lambda _path: FormalSnapshot(tuple(selected), FORMAL_CHAIN_HEAD),
        identity_provider=lambda: current,
    )


def _authorized() -> PolicyEvidence:
    return PolicyEvidence(
        True,
        "formal_gain_and_user_approval_exact_scope",
        FORMAL_STUDY_ID,
        FORMAL_DECISION_SHA256,
        FORMAL_PREREGISTRATION_SHA256,
        "d" * 64,
        FORMAL_CHAIN_HEAD,
        16,
    )


def _runtime_identity() -> dict[str, object]:
    return {
        "git_dirty": False,
        "code_sha256": "6" * 64,
        "spec_sha256": "7" * 64,
        "environment_sha256": "8" * 64,
        "hardware_sha256": "9" * 64,
    }


def _formal_projection() -> dict[str, object]:
    return {
        "authorized": True,
        "reason": "formal_gain_and_user_approval_exact_scope",
        "study_id": FORMAL_STUDY_ID,
        "decision_sha256": FORMAL_DECISION_SHA256,
        "preregistration_sha256": FORMAL_PREREGISTRATION_SHA256,
        "formal_database_sha256": FORMAL_DATABASE_SHA256,
        "formal_chain_head": FORMAL_CHAIN_HEAD,
        "evidence_records": 16,
        "qualification_id": None,
        "runtime_validation_record_id": None,
    }


def _runtime_rows() -> list[dict[str, object]]:
    identity = _runtime_identity()
    cpu_report = {
        "kind": "policy_overhead",
        "run_id": POLICY_RUN_ID,
        "status": "policy_overhead_passed",
        "qualification_id": QUALIFICATION_ID,
        "policy": _formal_projection(),
        "policy_load_ns": 1_000_000,
        "thresholds": {
            "policy_max_median_ns": POLICY_MAX_MEDIAN_NS,
            "policy_max_p95_ns": POLICY_MAX_P95_NS,
            "policy_max_incremental_ns": POLICY_MAX_INCREMENTAL_NS,
            "policy_max_load_ns": POLICY_MAX_LOAD_NS,
        },
        "metrics": {
            "gate_passed": True,
            "policy_load_gate_passed": True,
            "policy_median_ns": 1_000.0,
            "policy_p95_ns": 1_500.0,
            "incremental_median_ns": 500.0,
        },
    }
    attempt_report = {
        "kind": "runtime_validation_attempt",
        "run_id": GPU_RUN_ID,
        "status": "runtime_validation_started",
        "qualification_id": QUALIFICATION_ID,
        "policy": _formal_projection(),
    }
    gpu_report = {
        "kind": "runtime_validation",
        "run_id": GPU_RUN_ID,
        "status": "runtime_validation_passed",
        "qualification_id": QUALIFICATION_ID,
        "policy": _formal_projection(),
        "policy_load_ns": 1_000_000,
        "workload": {
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "prefill_chunk": PREFILL_CHUNK,
            "power_source": "ac_power",
        },
        "correctness": {
            "token_identical": True,
            "candidate_path_exercised": True,
        },
        "thresholds": {
            "max_ratio": GPU_MAX_RATIO,
            "max_extra_peak_bytes": GPU_MAX_EXTRA_PEAK_BYTES,
        },
        "metrics": {
            "gate_passed": True,
            "policy_load_gate_passed": True,
            "byte_identical": True,
            "ratio": 0.8,
            "peak_memory_delta_bytes": 0,
            "swap_delta_bytes": 0,
        },
        "blocks": [
            {"order": order, "candidate_plan": HEAD_SKIP_PLAN}
            for order in GPU_MEASUREMENT_ORDERS
        ],
    }
    reports = (cpu_report, attempt_report, gpu_report)
    kinds = ("policy_overhead", "runtime_validation_attempt", "runtime_validation")
    return [
        {
            "record_kind": kind,
            "record_id": str(index) * 64,
            "report": report,
            "provenance": identity,
        }
        for index, (kind, report) in enumerate(zip(kinds, reports), start=1)
    ]


def _load_runtime(
    rows: list[dict[str, object]], *, gpu_only: bool = False, runtime_path=None
) -> PolicyEvidence:
    loader = load_gpu_qualification_policy if gpu_only else load_runtime_policy
    return loader(
        evidence_path="ignored.sqlite3",
        runtime_path=(
            DEFAULT_RUNTIME_DATABASE_PATH if runtime_path is None else runtime_path
        ),
        evidence_reader=lambda _path: FormalSnapshot(
            tuple(_formal_rows()), FORMAL_CHAIN_HEAD
        ),
        identity_provider=_sealed_identity,
        runtime_reader=lambda _path: tuple(rows),
        runtime_identity_provider=_runtime_identity,
    )


class FakeBackend:
    model_id = MODEL_ID
    model_revision = MODEL_REVISION
    prefill_chunk = PREFILL_CHUNK

    def __init__(self, *, fail_candidate: bool = False) -> None:
        self.fail_candidate = fail_candidate
        self.baseline_calls = 0
        self.candidate_calls = 0

    def encode_prompt(self, prompt_content: str):
        return tuple(range(PROMPT_TOKENS)) if prompt_content == study.PROMPT_CONTENT else (1, 2)

    def _output(self, *, candidate: bool, output_tokens: int) -> GenerationOutput:
        tokens = tuple(range(output_tokens))
        blocks = 4 if candidate else 4
        return GenerationOutput(
            token_ids=tokens,
            token_sha256=canonical_sha256(list(tokens)),
            text="ok",
            prefill_ns=80 if candidate else 100,
            total_ns=120 if candidate else 140,
            prefill_blocks=blocks,
            head_calls=1 if candidate else blocks,
            memory={"mlx_peak_memory_bytes": 1_000},
        )

    def generate_baseline(self, _token_ids, request):
        self.baseline_calls += 1
        return self._output(candidate=False, output_tokens=request.output_tokens)

    def generate_head_skip(self, _token_ids, request):
        self.candidate_calls += 1
        if self.fail_candidate:
            raise MemoryError("simulated")
        return self._output(candidate=True, output_tokens=request.output_tokens)

    def reset_peak_memory(self) -> None:
        return None


class FakeGuard:
    def __init__(self, _policy) -> None:
        self.gpu = 0.0
        self.breaks = 0

    def before_candidate(self) -> None:
        return None

    def record_gpu(self, seconds: float) -> None:
        self.gpu += seconds

    def required_break(self) -> None:
        self.breaks += 1

    def finish_candidate(self) -> None:
        return None

    def summary(self):
        return {"gpu_work_seconds": self.gpu, "breaks": self.breaks}


class RuntimePolicyTest(unittest.TestCase):
    def test_runtime_contract_and_preregistration_are_byte_frozen(self) -> None:
        provenance._verify_frozen_inputs()
        original = provenance._regular_bytes

        def changed(relative: str) -> bytes:
            if relative == "experiments/head_skip_runtime/PREREGISTRATION.md":
                return b"changed"
            return original(relative)

        with patch.object(provenance, "_regular_bytes", side_effect=changed):
            with self.assertRaises(provenance.ProvenanceError):
                provenance._verify_frozen_inputs()

    def test_only_exact_scope_is_authorized(self) -> None:
        evidence = _load()
        self.assertTrue(evidence.authorized)
        self.assertEqual(evidence.evidence_records, 16)
        self.assertEqual(decision_for(evidence, REGISTERED_SCOPE).strategy, "head_skip")
        changed = replace(REGISTERED_SCOPE, prompt_tokens=PROMPT_TOKENS - 1)
        fallback = decision_for(evidence, changed)
        self.assertEqual(fallback.strategy, "baseline")
        self.assertEqual(fallback.reason, "request_out_of_scope")

    def test_mutated_evidence_and_identity_fail_closed(self) -> None:
        changed = _formal_rows()
        changed[-1]["action"] = "automatic_product_activation"
        self.assertFalse(_load(changed).authorized)
        for field, reason in (
            ("code_sha256", "formal_code_mismatch"),
            ("spec_sha256", "formal_spec_mismatch"),
            ("environment_sha256", "environment_mismatch"),
            ("hardware_sha256", "hardware_mismatch"),
            ("model_sha256", "model_snapshot_mismatch"),
        ):
            with self.subTest(field=field):
                identity = _sealed_identity()
                identity[field] = "f" * 64
                self.assertEqual(_load(identity=identity).reason, reason)
        dirty = _sealed_identity()
        dirty["git_dirty"] = True
        self.assertEqual(_load(identity=dirty).reason, "worktree_dirty")

    def test_unexpected_reader_failure_never_authorizes(self) -> None:
        evidence = load_policy(
            "missing.sqlite3",
            evidence_reader=lambda _path: (_ for _ in ()).throw(RuntimeError("boom")),
            identity_provider=_sealed_identity,
        )
        self.assertFalse(evidence.authorized)
        self.assertEqual(evidence.reason, "evidence_unavailable_or_invalid")

    def test_real_formal_store_replays_with_its_sealed_identity(self) -> None:
        snapshot = policy._read_verified(policy.DEFAULT_FORMAL_DATABASE_PATH)
        identity = dict(snapshot.rows[0]["provenance"])
        identity["git_dirty"] = False
        evidence = load_policy(identity_provider=lambda: identity)
        self.assertTrue(evidence.authorized)
        self.assertEqual(evidence.formal_chain_head, FORMAL_CHAIN_HEAD)

    def test_runtime_requires_cpu_attempt_and_gpu_terminal_records(self) -> None:
        rows = _runtime_rows()
        cpu_only = _load_runtime(rows[:1], gpu_only=True)
        self.assertTrue(cpu_only.authorized)
        self.assertFalse(_load_runtime(rows[:1]).authorized)

        qualified = _load_runtime(rows)
        self.assertTrue(qualified.authorized)
        self.assertEqual(qualified.qualification_id, QUALIFICATION_ID)
        self.assertEqual(qualified.runtime_validation_record_id, "3" * 64)

    def test_runtime_gate_rejects_retries_mutations_and_other_database(self) -> None:
        rows = _runtime_rows()
        self.assertFalse(_load_runtime(rows[:2], gpu_only=True).authorized)

        rows = _runtime_rows()
        rows[-1]["report"]["correctness"]["token_identical"] = False
        rejected = _load_runtime(rows)
        self.assertFalse(rejected.authorized)
        self.assertEqual(rejected.reason, "runtime_gpu_gate_invalid")

        rows = _runtime_rows()
        rows[-1]["report"]["metrics"]["swap_delta_bytes"] = None
        self.assertFalse(_load_runtime(rows).authorized)

        rejected_path = _load_runtime(
            _runtime_rows(), runtime_path=DEFAULT_RUNTIME_DATABASE_PATH.with_name("other.sqlite3")
        )
        self.assertFalse(rejected_path.authorized)
        self.assertEqual(rejected_path.reason, "runtime_database_path_mismatch")


class RuntimeExecutorTest(unittest.TestCase):
    def request(self, **changes) -> GenerationRequest:
        values = {
            "prompt_content": study.PROMPT_CONTENT,
            "output_tokens": OUTPUT_TOKENS,
        }
        values.update(changes)
        return GenerationRequest(**values)

    def test_exact_request_uses_candidate_and_other_request_uses_baseline(self) -> None:
        backend = FakeBackend()
        controller = RuntimeController(_authorized())
        result = controller.execute(backend, self.request())
        self.assertEqual(result.decision.strategy, "head_skip")
        self.assertEqual((backend.candidate_calls, backend.baseline_calls), (1, 0))
        fallback = controller.execute(backend, self.request(prompt_content="different"))
        self.assertEqual(fallback.decision.strategy, "baseline")
        self.assertEqual(fallback.decision.reason, "request_out_of_scope")
        self.assertEqual((backend.candidate_calls, backend.baseline_calls), (1, 1))

    def test_candidate_failure_is_not_retried_and_latches_fallback(self) -> None:
        backend = FakeBackend(fail_candidate=True)
        controller = RuntimeController(_authorized())
        with self.assertRaisesRegex(RuntimeExecutionError, "was not retried"):
            controller.execute(backend, self.request())
        self.assertEqual((backend.candidate_calls, backend.baseline_calls), (1, 0))
        backend.fail_candidate = False
        result = controller.execute(backend, self.request())
        self.assertEqual(result.decision.reason, "circuit_breaker_latched")
        self.assertEqual((backend.candidate_calls, backend.baseline_calls), (1, 1))

    def test_each_scope_field_falls_back(self) -> None:
        controller = RuntimeController(_authorized())
        cases = (
            {"temperature": 0.5},
            {"prompt_logprobs": True},
            {"prompt_logprobs": 0},
            {"fixed_horizon": False},
            {"fixed_horizon": 1},
            {"batch": 2},
            {"output_tokens": OUTPUT_TOKENS - 1},
        )
        for change in cases:
            with self.subTest(change=change):
                backend = FakeBackend()
                result = controller.execute(backend, self.request(**change))
                self.assertEqual(result.decision.strategy, "baseline")

    def test_malformed_backend_metadata_and_tokens_fail_closed(self) -> None:
        controller = RuntimeController(_authorized())
        for field, value in (
            ("model_id", object()),
            ("model_revision", object()),
            ("prefill_chunk", True),
            ("prefill_chunk", str(PREFILL_CHUNK)),
        ):
            with self.subTest(field=field, value=value):
                backend = FakeBackend()
                setattr(backend, field, value)
                result = controller.execute(backend, self.request())
                self.assertEqual(result.decision.strategy, "baseline")
                self.assertEqual((backend.candidate_calls, backend.baseline_calls), (0, 1))

    def test_malformed_candidate_output_is_not_returned_or_retried(self) -> None:
        backend = FakeBackend()
        valid = backend._output(candidate=True, output_tokens=OUTPUT_TOKENS)
        backend.generate_head_skip = lambda _tokens, _request: replace(
            valid, token_sha256="f" * 64
        )
        controller = RuntimeController(_authorized())
        with self.assertRaisesRegex(RuntimeExecutionError, "was not retried"):
            controller.execute(backend, self.request())
        self.assertEqual((backend.candidate_calls, backend.baseline_calls), (0, 0))
        self.assertEqual(controller.circuit_reason, "RuntimeExecutionError")

        for token_ids in ((0.0,) * PROMPT_TOKENS, (False,) * PROMPT_TOKENS):
            with self.subTest(token_type=type(token_ids[0]).__name__):
                backend = FakeBackend()
                backend.encode_prompt = lambda _prompt, values=token_ids: values
                result = controller.execute(backend, self.request())
                self.assertEqual(result.decision.strategy, "baseline")
                self.assertEqual((backend.candidate_calls, backend.baseline_calls), (0, 1))


class RuntimeBenchmarkTest(unittest.TestCase):
    def test_policy_benchmark_is_balanced_and_bounded(self) -> None:
        controller = RuntimeController(_authorized())

        def measured(function, iterations):
            function()
            return iterations * (10 if function.__name__ == "direct" else 15)

        with patch.object(benchmark, "_measure_loop", side_effect=measured):
            result = benchmark.benchmark_policy_overhead(
                controller, warmup_blocks=1, measurement_blocks=3, iterations=100
            )
        self.assertTrue(result["metrics"]["gate_passed"])
        self.assertEqual(result["metrics"]["policy_median_ns"], 15.0)
        self.assertEqual([block["order"] for block in result["blocks"]], ["ab", "ba", "ab"])

    def test_gpu_qualification_checks_tokens_ratio_paths_and_budget(self) -> None:
        controller = RuntimeController(_authorized())
        fake = FakeBackend()
        with (
            patch.object(benchmark, "BudgetGuard", FakeGuard),
            patch.object(benchmark, "require_ac_power", return_value="ac_power"),
            patch.object(
                benchmark.MlxGenerationBackend, "load_local", return_value=fake
            ),
            patch.object(benchmark, "_swap_used", side_effect=[0, 0]),
        ):
            result = benchmark.run_mlx_validation(controller)
        self.assertTrue(result["correctness"]["token_identical"])
        self.assertTrue(result["correctness"]["candidate_path_exercised"])
        self.assertEqual(result["metrics"]["ratio"], 0.8)
        self.assertTrue(result["metrics"]["gate_passed"])
        self.assertEqual([block["order"] for block in result["blocks"]], ["ab", "ba", "ab", "ba"])

    def test_unknown_swap_usage_fails_the_resource_gate(self) -> None:
        controller = RuntimeController(_authorized())
        fake = FakeBackend()
        with (
            patch.object(benchmark, "BudgetGuard", FakeGuard),
            patch.object(benchmark, "require_ac_power", return_value="ac_power"),
            patch.object(
                benchmark.MlxGenerationBackend, "load_local", return_value=fake
            ),
            patch.object(benchmark, "_swap_used", side_effect=[None, None]),
        ):
            result = benchmark.run_mlx_validation(controller)
        self.assertIsNone(result["metrics"]["swap_delta_bytes"])
        self.assertFalse(result["metrics"]["gate_passed"])


if __name__ == "__main__":
    unittest.main()
