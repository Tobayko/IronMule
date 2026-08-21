import math
import sys
import types
import unittest
from unittest import mock

from friday_h0 import benchmark, worker
from friday_h0.protocol import close_manifest
from tests.test_protocol import valid_manifest


def _domain(closed, *, classification, status="completed", action="aggregation_required", error=None, evidence=None):
    domain_evidence = {
        "fixture": {},
        "correctness": {},
        "memory": [],
        "memory_limit": None,
        "memory_gate": "aggregation_required",
        "cache_state": "unknown",
        "fresh_process_required": True,
        "aggregation_required": action == "aggregation_required",
        "compile_wrapper_setup_ns": None,
        "first_eval_compile_inclusive_ns": None,
        "total_elapsed_ns": 1,
        "arms": {},
        "comparison": {},
        "raw_samples": [],
    }
    domain_evidence.update(evidence or {})
    if status == "invalid" and "failure_diagnostic" not in domain_evidence:
        code = (error or {}).get("code", "")
        details = {}
        if code == "correctness_failed":
            details = {"correctness": {}}
        domain_evidence = {"failure_diagnostic": {"schema_version": 1, "code": code, "details": details}}
    elif status == "invalid":
        domain_evidence = {"failure_diagnostic": domain_evidence["failure_diagnostic"]}
    return {
        "schema_version": 1,
        "run_id": closed.run_id,
        "mode": closed.mode,
        "manifest_sha256": closed.sha256,
        "status": status,
        "classification": classification,
        "benchmark_classification": classification,
        "action": action,
        "error": error,
        "evidence": domain_evidence,
        "adapter_contract": {
            "common_result_ready": False,
            "reason": "single-process measurements require aggregation before any global decision",
            "mapping": {
                "runtime_unavailable": "invalid/baseline_fallback",
                "invalid*": "invalid/baseline_fallback",
                "measurement_complete": "aggregation_required",
                "baseline_reference": "not_run",
            },
        },
    }


class WorkerBenchmarkTests(unittest.TestCase):
    def _run_fake(self, mode, domain):
        closed = close_manifest(valid_manifest(mode))
        captured = []
        fake = mock.Mock(return_value=domain)
        fake_module = types.SimpleNamespace(run_mlx_benchmark=fake)
        with mock.patch.object(worker, "_rss_sample", return_value=(1234, None)), mock.patch.object(
            worker, "_write_result", side_effect=lambda _manifest, result: captured.append(result)
        ), mock.patch.dict(sys.modules, {"friday_h0.benchmark": fake_module}):
            self.assertEqual(worker._run(closed), 0)
        fake.assert_called_once_with(closed.value)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["run_id"], closed.run_id)
        self.assertEqual(captured[0]["mode"], closed.mode)
        self.assertEqual(captured[0]["manifest_sha256"], closed.sha256)
        return captured[0]

    def test_measurement_complete_is_neutral_and_bound(self):
        closed = close_manifest(valid_manifest("compile_comparison"))
        result = self._run_fake(
            "compile_comparison",
            _domain(
                closed,
                classification="measurement_complete",
                evidence={"aggregation_required": True, "cache_state": "unknown"},
            ),
        )
        self.assertEqual(
            (result["status"], result["classification"], result["action"], result["error"]),
            ("completed", "measurement_complete", "baseline_fallback", None),
        )
        self.assertEqual(result["evidence"]["benchmark_classification"], "measurement_complete")
        self.assertEqual(result["evidence"]["benchmark_action"], "aggregation_required")
        self.assertTrue(result["evidence"]["aggregation_required"])
        self.assertEqual(result["evidence"]["adapter_contract"]["common_result_ready"], False)

    def test_baseline_reference_is_not_a_promotion(self):
        closed = close_manifest(valid_manifest("eager_baseline"))
        result = self._run_fake(
            "eager_baseline",
            _domain(closed, classification="baseline_reference", action="not_run"),
        )
        self.assertEqual(result["classification"], "measurement_complete")
        self.assertEqual(result["action"], "baseline_fallback")
        self.assertNotIn(result["classification"], {"promoted", "regression"})

    def test_completed_domain_mode_binding_is_fail_closed(self):
        cases = (
            ("eager_baseline", "baseline_reference", "not_run", False, "measurement_complete", "aggregation_required", True),
            ("compile_comparison", "measurement_complete", "aggregation_required", True, "baseline_reference", "not_run", False),
            ("aa_gpu", "measurement_complete", "aggregation_required", True, "baseline_reference", "not_run", False),
        )
        for mode, classification, action, aggregation, bad_classification, bad_action, bad_aggregation in cases:
            closed = close_manifest(valid_manifest(mode))
            domain = _domain(
                closed,
                classification=classification,
                action=action,
                evidence={"aggregation_required": aggregation},
            )
            valid = self._run_fake(mode, domain)
            self.assertIsNone(valid["error"], mode)

            mode_crossed = _domain(
                closed,
                classification=bad_classification,
                action=bad_action,
                evidence={"aggregation_required": bad_aggregation},
            )
            invalid = self._run_fake(mode, mode_crossed)
            self.assertEqual(invalid["error"]["code"], "benchmark_domain_invalid", mode)

    def test_runtime_unavailable_maps_to_bounded_fallback(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        result = self._run_fake(
            "aa_gpu",
            _domain(
                closed,
                classification="runtime_unavailable",
                status="invalid",
                action="baseline_fallback",
                error={"code": "runtime_unavailable", "message": "runtime unavailable"},
            ),
        )
        self.assertEqual(
            (result["status"], result["classification"], result["action"]),
            ("invalid", "runtime_unavailable", "baseline_fallback"),
        )

    def test_rss_fallback_reasons_are_preserved_in_the_exact_envelope(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        for reason in ("unavailable", "ps_exit", "ps_parse", "ps_negative", "parent_setup_failure"):
            with mock.patch.object(worker, "_rss_sample", return_value=(None, reason)):
                result = worker._error_result(closed, code="runtime_unavailable", message="runtime unavailable")
            self.assertEqual(result["evidence"], {"rss_peak_bytes": None, "rss_missing_reason": reason})

    def test_worker_registered_error_codes_match_correctness_contract_boundary(self):
        self.assertIn("correctness_contract", worker._DIAGNOSTIC_CODES)
        self.assertIs(worker._DIAGNOSTIC_CODES, benchmark.REGISTERED_BENCHMARK_ERROR_CODES)

    def test_diagnostic_positive_int_matches_signed_int64_boundary(self):
        maximum = (1 << 63) - 1
        self.assertTrue(worker._diagnostic_positive_int(maximum))
        for value in (1 << 63, 0, -1, True):
            self.assertFalse(worker._diagnostic_positive_int(value))

    def test_correctness_and_other_invalid_errors_are_separated(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        correctness = self._run_fake(
            "aa_gpu",
            _domain(
                closed,
                classification="invalid: correctness",
                status="invalid",
                action="baseline_fallback",
                error={"code": "correctness_failed", "message": "hard cap"},
            ),
        )
        self.assertEqual(correctness["classification"], "invalid: correctness")
        other = self._run_fake(
            "aa_gpu",
            _domain(
                closed,
                classification="invalid",
                status="invalid",
                action="baseline_fallback",
                error={"code": "warmup_unstable", "message": "calibration failed"},
            ),
        )
        self.assertEqual(other["classification"], "invalid")

    def test_completed_domain_rejects_failure_diagnostic(self):
        closed = close_manifest(valid_manifest("compile_comparison"))
        domain = _domain(closed, classification="measurement_complete", evidence={"failure_diagnostic": {}})
        result = self._run_fake("compile_comparison", domain)
        self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

    def test_completed_domain_requires_exact_success_keys_and_boolean_action_binding(self):
        closed = close_manifest(valid_manifest("compile_comparison"))
        for mutation in ("missing", "extra", "zero", "one", "wrong_bool"):
            domain = _domain(closed, classification="measurement_complete")
            if mutation == "missing":
                del domain["evidence"]["comparison"]
            elif mutation == "extra":
                domain["evidence"]["failure_diagnostic"] = {}
            elif mutation == "zero":
                domain["evidence"]["aggregation_required"] = 0
            elif mutation == "one":
                domain["evidence"]["aggregation_required"] = 1
            else:
                domain["evidence"]["aggregation_required"] = False
            result = self._run_fake("compile_comparison", domain)
            self.assertEqual(result["error"]["code"], "benchmark_domain_invalid", mutation)

    def test_unregistered_domain_error_code_is_rejected(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        domain = _domain(
            closed,
            classification="invalid",
            status="invalid",
            action="baseline_fallback",
            error={"code": "warmup_unstalbe", "message": "typo"},
            evidence={"failure_diagnostic": {"schema_version": 1, "code": "warmup_unstalbe", "details": {}}},
        )
        result = self._run_fake("aa_gpu", domain)
        self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

    def test_domain_schema_version_requires_integer_one(self):
        closed = close_manifest(valid_manifest("compile_comparison"))
        domain = _domain(closed, classification="measurement_complete")
        domain["schema_version"] = 1.0
        result = self._run_fake("compile_comparison", domain)
        self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

    def test_invalid_diagnostic_envelope_is_closed_and_bounded(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        valid = {
            "schema_version": 1,
            "code": "evaluation_timeout_observed",
            "details": {"evaluation_ns": 10},
        }
        malformed = []
        missing = dict(valid)
        del missing["details"]
        malformed.append(missing)
        extra = dict(valid)
        extra["extra"] = True
        malformed.append(extra)
        schema = dict(valid)
        schema["schema_version"] = True
        malformed.append(schema)
        mismatch = dict(valid)
        mismatch["code"] = "backend_error"
        malformed.append(mismatch)
        for details in ({"evaluation_ns": 0}, {"evaluation_ns": -1}, {"evaluation_ns": 1 << 63}, {"evaluation_ns": 1.5}, {"evaluation_ns": float("inf")}, {"evaluation_ns": 1, "extra": 2}):
            malformed.append({"schema_version": 1, "code": "evaluation_timeout_observed", "details": details})
        malformed.extend(
            [
                {"schema_version": 1, "code": "warmup_unstable", "details": {"warmups_ns": [1] * 15}},
                {"schema_version": 1, "code": "warmup_unstable", "details": {"warmups_ns": [1] * 16 + [2]}},
                {"schema_version": 1, "code": "repetition_window_unreachable", "details": {"repetitions": 3, "batch_ns": 1}},
                {"schema_version": 1, "code": "repetition_window_unreachable", "details": {"repetitions": 2, "batch_ns": 0}},
                {"schema_version": 1, "code": "backend_error", "details": {"unexpected": True}},
                {"schema_version": 1, "code": "correctness_failed", "details": {"correctness": []}},
                {"schema_version": 1, "code": "result_too_large", "details": {"truncated": True, "missing_reason": "other"}},
            ]
        )
        deep = {}
        cursor = deep
        for _ in range(20):
            cursor["nested"] = {}
            cursor = cursor["nested"]
        malformed.append({"schema_version": 1, "code": "backend_error", "details": deep})
        oversized = {"schema_version": 1, "code": "backend_error", "details": {"blob": "x" * 70_000}}
        malformed.append(oversized)
        for diagnostic in malformed:
            domain = _domain(
                closed,
                classification="invalid",
                status="invalid",
                action="baseline_fallback",
                error={"code": "evaluation_timeout_observed", "message": "timeout"},
                evidence={"failure_diagnostic": diagnostic},
            )
            result = self._run_fake("aa_gpu", domain)
            self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

    def test_correctness_diagnostic_tree_is_recursively_bounded(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        malformed = []
        oversized_int = {"value": 1 << 63}
        malformed.append(oversized_int)
        malformed.append({"value": float("inf")})
        deep = {}
        cursor = deep
        for _ in range(20):
            cursor["nested"] = {}
            cursor = cursor["nested"]
        malformed.append(deep)
        cyclic = {}
        cyclic["self"] = cyclic
        malformed.append(cyclic)
        malformed.append({"values": [None] * 10_001})
        malformed.append({"blob": "x" * 70_000})
        for details in malformed:
            domain = _domain(
                closed,
                classification="invalid",
                status="invalid",
                action="baseline_fallback",
                error={"code": "correctness_failed", "message": "failure"},
                evidence={
                    "failure_diagnostic": {
                        "schema_version": 1,
                        "code": "correctness_failed",
                        "details": {"correctness": details},
                    }
                },
            )
            result = self._run_fake("aa_gpu", domain)
            self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

    def test_invalid_diagnostic_variants_are_preserved(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        variants = [
            ("evaluation_timeout_observed", {"evaluation_ns": 11}),
            ("synchronization_timeout_observed", {"synchronize_ns": 12}),
            ("warmup_unstable", {"warmups_ns": [1] * 16}),
            ("repetition_window_unreachable", {}),
            ("repetition_window_unreachable", {"repetitions": 2, "batch_ns": 13}),
            ("correctness_failed", {"correctness": {"passed": False}}),
            ("total_timeout_observed", {"total_ns": 14}),
            ("result_too_large", {"truncated": True, "missing_reason": "result_limit"}),
            ("runtime_unavailable", {}),
            ("manifest_invalid", {}),
            ("backend_error", {}),
        ]
        for code, details in variants:
            domain = _domain(
                closed,
                classification="invalid",
                status="invalid",
                action="baseline_fallback",
                error={"code": code, "message": "failure"},
                evidence={"failure_diagnostic": {"schema_version": 1, "code": code, "details": details}},
            )
            result = self._run_fake("aa_gpu", domain)
            self.assertEqual(result["error"]["code"], code)
            self.assertEqual(result["evidence"]["benchmark_evidence"]["failure_diagnostic"]["code"], code)

    def test_warmup_v2_diagnostic_roundtrip_and_cross_consistency(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        blocks = [
            {
                "block_index": index,
                "evaluations": 500,
                "block_ns": benchmark.H0_BATCH_MIN_NS,
                "per_eval_ns": 100_000,
                "median_eval_ns": 100,
                "min_eval_ns": 90,
                "max_eval_ns": 110,
            }
            for index in range(16)
        ]
        diagnostic = {
            "schema_version": 2,
            "code": "warmup_unstable",
            "details": {
                "warmup_block_per_eval_ns": [100_000] * 16,
                "warmup_blocks": blocks,
            },
        }
        domain = _domain(
            closed,
            classification="invalid",
            status="invalid",
            action="baseline_fallback",
            error={"code": "warmup_unstable", "message": "calibration failed"},
            evidence={"failure_diagnostic": diagnostic},
        )
        result = self._run_fake("aa_gpu", domain)
        self.assertEqual(result["error"]["code"], "warmup_unstable")
        self.assertEqual(result["evidence"]["benchmark_evidence"]["failure_diagnostic"], diagnostic)

        malformed = dict(diagnostic)
        malformed["details"] = dict(diagnostic["details"])
        malformed["details"]["warmup_blocks"] = [dict(blocks[0]) for _ in range(16)]
        malformed["details"]["warmup_blocks"][1]["block_index"] = 0
        bad_domain = _domain(
            closed,
            classification="invalid",
            status="invalid",
            action="baseline_fallback",
            error={"code": "warmup_unstable", "message": "calibration failed"},
            evidence={"failure_diagnostic": malformed},
        )
        bad_result = self._run_fake("aa_gpu", bad_domain)
        self.assertEqual(bad_result["error"]["code"], "benchmark_domain_invalid")

        short_blocks = [dict(block) for block in blocks]
        short_blocks[0]["block_ns"] = benchmark.H0_BATCH_MIN_NS - 1
        short_blocks[0]["per_eval_ns"] = benchmark._per_eval_ns(
            short_blocks[0]["block_ns"], short_blocks[0]["evaluations"]
        )
        short_values = [block["per_eval_ns"] for block in short_blocks]
        short_diagnostic = {
            "schema_version": 2,
            "code": "warmup_unstable",
            "details": {
                "warmup_block_per_eval_ns": short_values,
                "warmup_blocks": short_blocks,
            },
        }
        short_domain = _domain(
            closed,
            classification="invalid",
            status="invalid",
            action="baseline_fallback",
            error={"code": "warmup_unstable", "message": "calibration failed"},
            evidence={"failure_diagnostic": short_diagnostic},
        )
        short_result = self._run_fake("aa_gpu", short_domain)
        self.assertEqual(short_result["error"]["code"], "benchmark_domain_invalid")

    def test_registered_codes_without_special_details_require_empty_details(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        special = {
            "evaluation_timeout_observed", "synchronization_timeout_observed", "warmup_unstable",
            "repetition_window_unreachable", "correctness_failed", "total_timeout_observed", "result_too_large",
        }
        for code in sorted(worker._DIAGNOSTIC_CODES - special):
            domain = _domain(
                closed,
                classification="invalid",
                status="invalid",
                action="baseline_fallback",
                error={"code": code, "message": "failure"},
                evidence={"failure_diagnostic": {"schema_version": 1, "code": code, "details": {}}},
            )
            result = self._run_fake("aa_gpu", domain)
            self.assertEqual(result["error"]["code"], code, code)

    def test_promoted_regression_and_unknown_domain_are_fail_closed(self):
        closed = close_manifest(valid_manifest("compile_comparison"))
        for classification in ("promoted", "regression", "unknown"):
            domain = _domain(closed, classification=classification)
            result = self._run_fake("compile_comparison", domain)
            self.assertEqual(result["classification"], "invalid")
            self.assertEqual(result["action"], "baseline_fallback")
            self.assertNotIn(result["classification"], {"promoted", "regression"})

    def test_unknown_key_nonfinite_and_oversize_evidence_fail_closed(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        top_level_unknown = _domain(closed, classification="measurement_complete")
        top_level_unknown["unexpected"] = True
        result = self._run_fake("aa_gpu", top_level_unknown)
        self.assertEqual(result["classification"], "invalid")

        unknown = _domain(closed, classification="measurement_complete", evidence={"unexpected": True})
        result = self._run_fake("aa_gpu", unknown)
        self.assertEqual(result["classification"], "invalid")

        nonfinite = _domain(closed, classification="measurement_complete", evidence={"total_ns": math.inf})
        result = self._run_fake("aa_gpu", nonfinite)
        self.assertEqual(result["classification"], "invalid")

        oversized = _domain(
            closed,
            classification="measurement_complete",
            evidence={"raw_samples": [{"sample": "x" * 1_100_000}]},
        )
        result = self._run_fake("aa_gpu", oversized)
        self.assertEqual(result["classification"], "invalid")
        self.assertLessEqual(len(result["error"]["message"]), 256)
        self.assertLessEqual(len(str(result).encode()), 1 << 20)

    def test_contract_schema_bool_depth_cycle_and_false_ready_fail_closed(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        invalid_contracts = []
        missing_mapping = _domain(closed, classification="measurement_complete")
        del missing_mapping["adapter_contract"]["mapping"]
        invalid_contracts.append(missing_mapping)
        ready = _domain(closed, classification="measurement_complete")
        ready["adapter_contract"]["common_result_ready"] = True
        invalid_contracts.append(ready)
        wrong_mapping = _domain(closed, classification="measurement_complete")
        wrong_mapping["adapter_contract"]["mapping"]["measurement_complete"] = "promoted"
        invalid_contracts.append(wrong_mapping)
        bool_schema = _domain(closed, classification="measurement_complete")
        bool_schema["schema_version"] = True
        invalid_contracts.append(bool_schema)
        for domain in invalid_contracts:
            result = self._run_fake("aa_gpu", domain)
            self.assertEqual(result["classification"], "invalid")
            self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

        cyclic = _domain(closed, classification="measurement_complete")
        cyclic["evidence"]["cycle"] = cyclic["evidence"]
        result = self._run_fake("aa_gpu", cyclic)
        self.assertEqual(result["classification"], "invalid")
        self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

        deep = _domain(closed, classification="measurement_complete")
        cursor = deep["evidence"]
        for _ in range(17):
            cursor["nested"] = {}
            cursor = cursor["nested"]
        result = self._run_fake("aa_gpu", deep)
        self.assertEqual(result["classification"], "invalid")
        self.assertEqual(result["error"]["code"], "benchmark_domain_invalid")

    def test_long_domain_error_is_bounded(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        result = self._run_fake(
            "aa_gpu",
            _domain(
                closed,
                classification="invalid",
                status="invalid",
                action="baseline_fallback",
                error={"code": "backend_error", "message": "E" * 10_000},
            ),
        )
        self.assertEqual(result["classification"], "invalid")
        self.assertLessEqual(len(result["error"]["code"]), 256)
        self.assertLessEqual(len(result["error"]["message"]), 256)

    def test_system_exit_and_empty_protocol_messages_are_bounded(self):
        closed = close_manifest(valid_manifest("aa_gpu"))
        fake = mock.Mock(side_effect=SystemExit("backend stopped"))
        fake_module = types.SimpleNamespace(run_mlx_benchmark=fake)
        captured = []
        with mock.patch.object(worker, "_rss_sample", return_value=(1234, None)), mock.patch.object(
            worker, "_write_result", side_effect=lambda _manifest, result: captured.append(result)
        ), mock.patch.dict(sys.modules, {"friday_h0.benchmark": fake_module}):
            self.assertEqual(worker._run(closed), 0)
        self.assertEqual(captured[0]["classification"], "invalid")
        self.assertEqual(captured[0]["error"]["code"], "benchmark_exception")
        with mock.patch.object(worker, "_rss_sample", return_value=(None, "test")):
            result = worker._error_result(closed, code="worker_protocol", message="")
        self.assertTrue(result["error"]["message"])

    def test_analysis_path_does_not_call_benchmark(self):
        closed = close_manifest(valid_manifest("analysis_known_win"))
        captured = []
        with mock.patch.object(worker, "_rss_sample", return_value=(None, "test")), mock.patch.object(
            worker, "_write_result", side_effect=lambda _manifest, result: captured.append(result)
        ), mock.patch.object(worker.importlib, "import_module") as loader:
            self.assertEqual(worker._run(closed), 0)
        loader.assert_not_called()
        self.assertFalse(hasattr(worker, "benchmark"))
        self.assertEqual(captured[0]["classification"], "promoted")


if __name__ == "__main__":
    unittest.main()
