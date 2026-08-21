import ast
import inspect
import json
import unittest
from unittest import mock

import numpy as np

from friday_h0 import benchmark
from friday_h0 import worker
from friday_h0.protocol import close_manifest
from tests.test_manifest import valid_manifest


class FakeClock:
    def __init__(self, step_ns=10_000_000):
        self.now = 0
        self.step_ns = step_ns

    def __call__(self):
        self.now += self.step_ns
        return self.now


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeBackend:
    def __init__(self, *, memory=None, wrong=False):
        self.memory = memory or [
            {"name": "mlx_peak_memory", "api": "fake.peak", "unit": "bytes", "value": 1024},
            {"name": "rss", "api": "fake.rss", "unit": "bytes", "value": 2048},
        ]
        self.wrong = wrong
        self.compile_calls = 0
        self.compile_shapeless = []
        self.eval_calls = 0
        self.sync_calls = 0
        self.memory_limit_calls = []

    def from_host(self, value):
        return np.asarray(value)

    def matmul(self, a, b):
        if self.wrong:
            return np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
        return np.matmul(a, b).astype(np.float16)

    def eval(self, value):
        self.eval_calls += 1
        return value

    def synchronize(self):
        self.sync_calls += 1

    def to_host(self, value):
        return np.asarray(value)

    def compile(self, function, *, shapeless=False):
        self.compile_calls += 1
        self.compile_shapeless.append(shapeless)
        return lambda a, b: function(a, b)

    def set_memory_limit(self, limit_bytes):
        self.memory_limit_calls.append(limit_bytes)

    def memory_metrics(self):
        return self.memory


class CallLogBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.call_log = []

    def tagged(self, tag):
        def call():
            self.call_log.append(tag)
            return np.zeros((2, 2), dtype=np.float16)
        return call


_ORIGINAL_GENERATE_FIXTURE = benchmark._generate_fixture


def small_fixture(np_module, seed):
    return _ORIGINAL_GENERATE_FIXTURE(np_module, seed, shape=4)


def passing_correctness(_backend, _np_module, fixture):
    case = {"name": "fake_correctness", "passed": True, "metrics": {"abs_max": 0.0}}
    return {"cases": [case], "passed": True, "performance": case, "sign_invariant": case}


class BenchmarkTests(unittest.TestCase):
    def test_registered_error_code_set_is_explicit_and_complete(self):
        self.assertEqual(
            benchmark.REGISTERED_BENCHMARK_ERROR_CODES,
            frozenset(
                {
                    "non_finite_measurement", "non_finite_result", "non_json_result",
                    "fixture_metadata_invalid", "invalid_fixture_seed", "invalid_fixture_shape",
                    "backend_contract", "compile_contract", "runtime_unavailable", "clock_contract",
                    "evaluation_timeout_observed", "synchronization_timeout_observed", "empty_measurements",
                    "warmup_unstable", "repetition_window_unreachable", "order_contract", "correctness_shape",
                    "correctness_nonfinite", "invalid_ratio", "correctness_failed", "mode_not_benchmarkable",
                    "total_timeout_observed", "manifest_invalid", "backend_error", "result_too_large",
                    "correctness_contract",
                }
            ),
        )

    def test_registered_error_code_allowlist_matches_static_calls_and_worker(self):
        tree = ast.parse(inspect.getsource(benchmark))
        static_codes = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"BenchmarkError", "RuntimeUnavailable"} or not node.args:
                continue
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                static_codes.add(node.args[0].value)
        expected = static_codes | {"manifest_invalid", "backend_error", "result_too_large"}
        self.assertEqual(benchmark.REGISTERED_BENCHMARK_ERROR_CODES, frozenset(expected))
        self.assertIs(worker._DIAGNOSTIC_CODES, benchmark.REGISTERED_BENCHMARK_ERROR_CODES)

    def test_unregistered_benchmark_error_codes_are_closed_at_domain_boundary(self):
        manifest = valid_manifest("aa_gpu")
        closed = close_manifest(manifest)
        cases = ["bogus", 17, ""]
        for code in cases:
            with self.subTest(code=code):
                error = benchmark.BenchmarkError(code, "x" * 100_000, evidence={"unexpected": "x" * 100_000})
                with mock.patch.object(benchmark, "_load_numpy", return_value=np), mock.patch.object(
                    benchmark, "_generate_fixture", side_effect=error
                ):
                    domain = benchmark.run_mlx_benchmark(manifest, backend_factory=FakeBackend(), clock_ns=FakeClock())
                self.assertEqual(domain["error"], {"code": "backend_error", "message": "benchmark failure code is not registered"})
                self.assertEqual(domain["evidence"]["failure_diagnostic"], {"schema_version": 1, "code": "backend_error", "details": {}})
                common = worker._benchmark_result(closed, domain, {"rss_peak_bytes": 1, "rss_missing_reason": None})
                self.assertEqual(common["error"]["code"], "backend_error")
                self.assertNotEqual(common["error"]["code"], "benchmark_domain_invalid")

    def test_invalid_benchmark_errors_roundtrip_through_worker(self):
        cases = [
            ("evaluation_timeout_observed", {"evaluation_ns": 11}),
            ("synchronization_timeout_observed", {"synchronize_ns": 12}),
            ("warmup_unstable", {"warmups_ns": [1] * 16}),
            ("repetition_window_unreachable", {}),
            ("repetition_window_unreachable", {"repetitions": 2, "batch_ns": 13}),
            ("correctness_failed", {"correctness": {"passed": False}}),
            ("total_timeout_observed", {"total_ns": 14}),
            ("backend_error", {}),
        ]
        manifest = valid_manifest("aa_gpu")
        closed = close_manifest(manifest)
        for code, evidence in cases:
            with self.subTest(code=code):
                error = benchmark.BenchmarkError(code, "failure", evidence=evidence)
                with mock.patch.object(benchmark, "_load_numpy", return_value=np), mock.patch.object(
                    benchmark, "_generate_fixture", side_effect=error
                ):
                    domain = benchmark.run_mlx_benchmark(manifest, backend_factory=FakeBackend(), clock_ns=FakeClock())
                diagnostic = domain["evidence"]["failure_diagnostic"]
                self.assertEqual(domain["status"], "invalid")
                self.assertEqual(domain["error"]["code"], code)
                self.assertEqual(diagnostic["code"], code)
                common = worker._benchmark_result(closed, domain, {"rss_peak_bytes": 1, "rss_missing_reason": None})
                self.assertEqual(common["error"]["code"], code)
                self.assertEqual(common["evidence"]["benchmark_evidence"], domain["evidence"])
                self.assertEqual(common["evidence"]["benchmark_evidence"]["failure_diagnostic"], diagnostic)

    def test_warmup_unstable_diagnostic_is_deterministic_with_fake_clock(self):
        class WarmupClock:
            def __init__(self):
                self.now = 0
                self.index = 0

            def __call__(self):
                phase = self.index % 3
                sample = self.index // 3
                self.index += 1
                if phase == 0:
                    return self.now
                self.now += 1 if phase == 1 else (100 if sample % 2 == 0 else 10_000_000)
                return self.now

        adapter = benchmark._BackendAdapter(FakeBackend(), np)
        function = lambda: np.zeros((2, 2), dtype=np.float16)
        with self.assertRaises(benchmark.BenchmarkError) as raised:
            benchmark._warmups(adapter, function, WarmupClock())
        self.assertEqual(raised.exception.code, "warmup_unstable")
        evidence = raised.exception.evidence
        self.assertEqual(len(evidence["warmup_block_per_eval_ns"]), 16)
        self.assertEqual(len(evidence["warmup_blocks"]), 16)
        self.assertEqual(
            benchmark._failure_diagnostic(raised.exception.code, raised.exception.evidence),
            {
                "schema_version": 2,
                "code": "warmup_unstable",
                "details": {
                    "warmup_block_per_eval_ns": evidence["warmup_block_per_eval_ns"],
                    "warmup_blocks": evidence["warmup_blocks"],
                },
            },
        )

    def test_warmup_gate_uses_outer_batch_time_and_dilutes_one_outlier(self):
        evaluations = 100
        block_ns = 99 * 500_000 + 620_000
        self.assertEqual(benchmark._per_eval_ns(block_ns, evaluations), 501_200)
        self.assertTrue(
            benchmark._stable_last_five(
                [benchmark._per_eval_ns(50_000_000, 100)] * 4
                + [benchmark._per_eval_ns(block_ns, evaluations)]
            )
        )

    def test_warmup_gate_detects_batch_regression_hidden_by_single_eval_median(self):
        individual_medians = [100, 100, 100, 100, 100]
        batch_per_eval = [100, 100, 100, 100, 106]
        self.assertEqual(sorted(individual_medians), [100] * 5)
        self.assertFalse(benchmark._stable_last_five(batch_per_eval))

    def test_warmup_block_cap_fails_closed_before_minimum_window(self):
        adapter = benchmark._BackendAdapter(FakeBackend(), np)
        function = lambda: np.zeros((2, 2), dtype=np.float16)
        clock = FakeClock(step_ns=1)
        with self.assertRaises(benchmark.BenchmarkError) as raised:
            benchmark._warmup_block(adapter, function, clock, 0)
        self.assertEqual(raised.exception.code, "repetition_window_unreachable")
        self.assertEqual(
            raised.exception.evidence,
            {"repetitions": benchmark.H0_MAX_REPETITIONS, "batch_ns": 4 * benchmark.H0_MAX_REPETITIONS},
        )

    def test_successful_warmup_keeps_only_bounded_block_summaries(self):
        adapter = benchmark._BackendAdapter(FakeBackend(), np)
        function = lambda: np.zeros((2, 2), dtype=np.float16)
        block = benchmark._warmup_block(adapter, function, FakeClock(), 0)
        self.assertIsInstance(block, dict)
        self.assertEqual(
            set(block),
            {"block_index", "evaluations", "block_ns", "per_eval_ns", "median_eval_ns", "min_eval_ns", "max_eval_ns"},
        )
        self.assertFalse(any(isinstance(value, list) for value in block.values()))

        warmup = benchmark._warmups(adapter, function, FakeClock())
        self.assertEqual(set(warmup), {"count", "durations_ns", "stable", "median_ns", "samples", "blocks"})
        self.assertEqual(len(warmup["blocks"]), warmup["count"])
        self.assertEqual(len(warmup["samples"]), warmup["count"])
        self.assertEqual([item["per_eval_ns"] for item in warmup["blocks"]], warmup["durations_ns"])

    def test_failure_diagnostic_does_not_emit_v2_for_short_warmup_block(self):
        blocks = [
            {
                "block_index": index,
                "evaluations": 500,
                "block_ns": benchmark.H0_BATCH_MIN_NS,
                "per_eval_ns": benchmark.H0_BATCH_MIN_NS // 500,
                "median_eval_ns": 100_000,
                "min_eval_ns": 99_999,
                "max_eval_ns": 100_001,
            }
            for index in range(benchmark.H0_MAX_WARMUPS)
        ]
        values = [block["per_eval_ns"] for block in blocks]
        blocks[0]["block_ns"] = benchmark.H0_BATCH_MIN_NS - 1
        blocks[0]["per_eval_ns"] = benchmark._per_eval_ns(blocks[0]["block_ns"], blocks[0]["evaluations"])
        values[0] = blocks[0]["per_eval_ns"]
        diagnostic = benchmark._failure_diagnostic(
            "warmup_unstable",
            {"warmup_block_per_eval_ns": values, "warmup_blocks": blocks},
        )
        self.assertEqual(diagnostic, {"schema_version": 1, "code": "warmup_unstable", "details": {}})

    def test_timed_and_batch_do_not_retain_outputs(self):
        adapter = benchmark._BackendAdapter(FakeBackend(), np)
        function = lambda: np.zeros((2, 2), dtype=np.float16)
        elapsed, timings = benchmark._batch(adapter, function, 2, FakeClock())
        self.assertGreater(elapsed, 0)
        self.assertEqual(len(timings), 2)
        self.assertEqual(set(benchmark._Timed.__dataclass_fields__), {"duration_ns", "evaluation_ns", "synchronize_ns"})
        self.assertFalse(any(hasattr(timing, "output") for timing in timings))

    def test_fixture_uses_replayable_little_endian_fp16(self):
        first = benchmark._generate_fixture(np, 0xF17A2026, shape=8)
        second = benchmark._generate_fixture(np, 0xF17A2026, shape=8)
        self.assertEqual(first.fixture_sha256, second.fixture_sha256)
        self.assertEqual(first.a.tobytes(), second.a.tobytes())
        self.assertEqual(first.a.dtype, np.dtype("<f2"))
        self.assertEqual(len(first.fixture_sha256), 64)

    def test_balanced_order_and_warmup_boundary(self):
        order = benchmark._balanced_order(0xB10C2026)
        self.assertEqual(len(order), 30)
        self.assertEqual(order.count("baseline"), 15)
        self.assertEqual(order.count("candidate"), 15)
        self.assertTrue(benchmark._stable_last_five([100, 100, 100, 105, 95]))
        self.assertFalse(benchmark._stable_last_five([100, 100, 100, 106, 100]))

    def test_repetition_window_and_unreachable_window(self):
        adapter = benchmark._BackendAdapter(FakeBackend(), np)
        function = lambda: np.zeros((2, 2), dtype=np.float16)
        selected = benchmark.choose_repetitions(adapter, function, FakeClock())
        self.assertEqual(selected["repetitions"], 2)
        with self.assertRaises(benchmark.BenchmarkError) as context:
            benchmark.choose_repetitions(adapter, function, FakeClock(step_ns=300_000_000))
        self.assertEqual(context.exception.code, "repetition_window_unreachable")

    def test_correctness_metrics_and_fail_closed_wrong_output(self):
        a = np.asarray([[1, 2], [3, 4]], dtype="<f2")
        b = np.asarray([[2, 0], [0, 2]], dtype="<f2")
        good = benchmark._BackendAdapter(FakeBackend(), np)
        passed = benchmark._correctness_case(good, np, name="small", a=a, b=b, seed=1)
        self.assertTrue(passed["passed"])
        bad = benchmark._BackendAdapter(FakeBackend(wrong=True), np)
        failed = benchmark._correctness_case(bad, np, name="small", a=a, b=b, seed=1)
        self.assertFalse(failed["passed"])
        self.assertIn("abs_q99", failed["metrics"])

    def test_eager_baseline_has_no_compile_and_is_json_safe(self):
        backend = FakeBackend()
        manifest = valid_manifest("eager_baseline")
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", side_effect=passing_correctness
        ):
            result = benchmark.run_mlx_benchmark(manifest, backend_factory=backend, clock_ns=FakeClock())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["benchmark_classification"], "baseline_reference")
        self.assertEqual(backend.compile_calls, 0)
        self.assertGreater(len(result["evidence"]["raw_samples"]), 30)
        self.assertEqual(result["evidence"]["cache_state"], "unknown")
        json.dumps(result, allow_nan=False)

    def test_aa_has_thirty_paired_blocks_and_separate_callables(self):
        backend = FakeBackend()
        manifest = valid_manifest("aa_gpu")
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", side_effect=passing_correctness
        ):
            result = benchmark.run_mlx_benchmark(manifest, backend_factory=backend, clock_ns=FakeClock())
        comparison = result["evidence"]["comparison"]
        self.assertEqual(result["benchmark_classification"], "measurement_complete")
        self.assertEqual(result["action"], "aggregation_required")
        self.assertEqual(len(comparison["blocks"]), 30)
        self.assertGreater(len(result["evidence"]["raw_samples"]), 60)
        self.assertEqual(comparison["order"].count("baseline"), 15)
        self.assertEqual(comparison["order"].count("candidate"), 15)
        self.assertEqual(backend.compile_calls, 0)

    def test_single_process_ratios_are_neutral_observations(self):
        order = ["baseline", "candidate"] * 15
        def arm(values):
            return {"batches": [{"per_eval_ns": value, "batch_ns": value, "repetitions": 1} for value in values], "raw_samples": []}
        for ratio in (.90, .99, 1.10):
            result = benchmark._comparison_result({"baseline": arm([100] * 30), "candidate": arm([100 * ratio] * 30)}, order, aa=False)
            self.assertEqual(result["benchmark_classification"], "session_observation")
            self.assertEqual(result["action"], "aggregation_required")
            self.assertIsNone(result["global_decision"])

    def test_paired_blocks_execute_in_actual_balanced_order(self):
        backend = CallLogBackend()
        adapter = benchmark._BackendAdapter(backend, np)
        order = ["baseline", "candidate"] * 15
        benchmark._run_paired_arms(adapter, backend.tagged("baseline"), backend.tagged("candidate"), FakeClock(), order)
        expected = [arm for first in order for arm in (first, "candidate" if first == "baseline" else "baseline") for _ in range(2)]
        self.assertEqual(backend.call_log[-len(expected):], expected)

    def test_memory_never_reports_a_pass_gate(self):
        metrics = [
            {"name": "mlx_peak_memory", "value": 2_000_000, "unit": "bytes"},
            {"name": "rss", "value": 2_000_000, "unit": "bytes"},
        ]
        self.assertEqual(benchmark._memory_gate(metrics), "aggregation_required")
        backend = FakeBackend(memory=metrics)
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", side_effect=passing_correctness
        ):
            result = benchmark.run_mlx_benchmark(valid_manifest("compile_comparison"), backend_factory=backend, clock_ns=FakeClock())
        self.assertEqual(result["evidence"]["memory_gate"], "aggregation_required")
        self.assertEqual(result["action"], "aggregation_required")

    def test_cache_memory_uses_the_registered_public_api_name(self):
        class CacheMemoryOnlyBackend:
            def get_cache_memory(self):
                return 1234

            def get_memory_size(self):
                raise AssertionError("obsolete MLX API must not be called")

        adapter = benchmark._BackendAdapter(CacheMemoryOnlyBackend(), np)
        metrics = adapter.memory_metrics()
        cache = next(item for item in metrics if item["name"] == "mlx_cache_memory")
        self.assertEqual(cache["api"], "get_cache_memory")
        self.assertEqual(cache["value"], 1234)
        self.assertIsNone(cache["missing_reason"])

    def test_correctness_case_digest_and_empty_relative_quantile_are_explicit(self):
        a = np.ones((2, 2), dtype="<f2") * np.float16(0.001)
        b = np.ones((2, 2), dtype="<f2") * np.float16(0.001)
        result = benchmark._correctness_case(benchmark._BackendAdapter(FakeBackend(), np), np, name="small", a=a, b=b, seed=9)
        metric = result["metrics"]["rel_q99_abs_oracle_ge_1"]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["missing_reason"], "no_oracle_elements_abs_ge_1")
        self.assertEqual(len(result["fixture_digest"]), 64)

    def test_result_size_is_bounded(self):
        result = benchmark._bounded_result({"schema_version": 1, "run_id": "x", "mode": "eager_baseline", "evidence": {"blob": "x" * (benchmark.H0_MAX_RESULT_BYTES + 1)}})
        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.assertLessEqual(len(encoded), benchmark.H0_MAX_RESULT_BYTES)
        self.assertEqual(result["error"]["code"], "result_too_large")
        self.assertEqual(
            result["evidence"],
            {"failure_diagnostic": {"schema_version": 1, "code": "result_too_large", "details": {"truncated": True, "missing_reason": "result_limit"}}},
        )

    def test_json_safe_rejects_cycles_and_unbounded_scalars_but_allows_aliases(self):
        cyclic = []
        cyclic.append(cyclic)
        with self.assertRaises(benchmark.BenchmarkError) as cycle_error:
            benchmark._json_safe(cyclic)
        self.assertEqual(cycle_error.exception.code, "non_json_result")

        deep = value = {}
        for _ in range(benchmark.PRODUCTION_JSON_DEPTH + 1):
            value["nested"] = {}
            value = value["nested"]
        with self.assertRaises(benchmark.BenchmarkError) as depth_error:
            benchmark._json_safe(deep)
        self.assertEqual(depth_error.exception.code, "result_too_large")

        with self.assertRaises(benchmark.BenchmarkError) as int_error:
            benchmark._json_safe(1 << 63)
        self.assertEqual(int_error.exception.code, "result_too_large")

        shared = {"value": 1}
        safe = benchmark._json_safe({"left": shared, "right": shared})
        self.assertEqual(safe, {"left": {"value": 1}, "right": {"value": 1}})

    def test_oversize_domain_fallback_remains_worker_compatible(self):
        manifest = close_manifest(valid_manifest("eager_baseline"))
        result = benchmark._bounded_result(
            {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "mode": manifest.mode,
                "manifest_sha256": manifest.sha256,
                "evidence": {"blob": "x" * (benchmark.H0_MAX_RESULT_BYTES + 1)},
            }
        )
        common = worker._benchmark_result(
            manifest,
            result,
            {"rss_peak_bytes": 1, "rss_missing_reason": None},
        )
        self.assertEqual(common["error"]["code"], "result_too_large")
        self.assertEqual(
            common["evidence"]["benchmark_evidence"]["failure_diagnostic"],
            result["evidence"]["failure_diagnostic"],
        )

    def test_compile_uses_shapeless_false_and_separates_setup(self):
        backend = FakeBackend()
        manifest = valid_manifest("compile_comparison")
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", side_effect=passing_correctness
        ):
            result = benchmark.run_mlx_benchmark(manifest, backend_factory=backend, clock_ns=FakeClock())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(backend.compile_calls, 1)
        self.assertEqual(backend.compile_shapeless, [False])
        self.assertIsNotNone(result["evidence"]["compile_wrapper_setup_ns"])
        self.assertIsNotNone(result["evidence"]["first_eval_compile_inclusive_ns"])

    def test_memory_missing_reason_is_explicit(self):
        backend = FakeBackend(memory=[{"name": "mlx_peak_memory", "api": "fake.peak", "unit": "bytes", "value": None, "missing_reason": "api_unavailable"}])
        manifest = valid_manifest("eager_baseline")
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", side_effect=passing_correctness
        ):
            result = benchmark.run_mlx_benchmark(manifest, backend_factory=backend, clock_ns=FakeClock())
        self.assertEqual(result["evidence"]["memory_gate"], "not_evaluable_missing_required_metric")
        self.assertEqual(result["evidence"]["memory"][0]["missing_reason"], "api_unavailable")

    def test_memory_limit_invalid_source_values_use_exact_closed_envelope(self):
        for returned in (True, None, -1, 1 << 63):
            class InvalidLimitBackend(FakeBackend):
                def set_memory_limit(self, _limit):
                    return returned

            backend = InvalidLimitBackend()
            manifest = valid_manifest("eager_baseline")
            with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
                benchmark, "_correctness_suite", side_effect=passing_correctness
            ):
                result = benchmark.run_mlx_benchmark(manifest, backend_factory=backend, clock_ns=FakeClock())
            self.assertEqual(
                result["evidence"]["memory_limit"],
                {"attempted": True, "hard_limit": False, "applied": False, "missing_reason": "invalid_source_value"},
            )

    def test_missing_mlx_is_structured_and_never_raises(self):
        manifest = valid_manifest("eager_baseline")
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_load_numpy", return_value=np
        ), mock.patch.object(
            benchmark.importlib, "import_module", side_effect=ImportError("no mlx")
        ):
            result = benchmark.run_mlx_benchmark(manifest, backend_factory=None, clock_ns=FakeClock())
        self.assertEqual(result["classification"], "runtime_unavailable")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["action"], "baseline_fallback")
        json.dumps(result, allow_nan=False)

    def test_eval_and_sync_thresholds_are_observed_after_return(self):
        adapter = benchmark._BackendAdapter(FakeBackend(), np)
        function = lambda: np.zeros((1, 1), dtype=np.float16)
        with self.assertRaises(benchmark.BenchmarkError) as evaluation:
            benchmark._measure_once(adapter, function, SequenceClock([0, benchmark.H0_FIRST_EVAL_LIMIT_NS + 1, benchmark.H0_FIRST_EVAL_LIMIT_NS + 2]))
        self.assertEqual(evaluation.exception.code, "evaluation_timeout_observed")
        with self.assertRaises(benchmark.BenchmarkError) as synchronization:
            benchmark._measure_once(adapter, function, SequenceClock([0, 1, benchmark.H0_SYNC_LIMIT_NS + 2]))
        self.assertEqual(synchronization.exception.code, "synchronization_timeout_observed")


if __name__ == "__main__":
    unittest.main()
