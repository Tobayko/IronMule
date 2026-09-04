import copy
import math
import resource
import time
import unittest

from friday_h0.aggregation import _finite, aggregate_h0_aa
from friday_h0.benchmark import H0_BATCH_MIN_NS, _balanced_order
from friday_h0.canonical import canonical_json_bytes
from friday_h0.correctness_contract import (
    CORRECTNESS_CASE_SPECS,
    CORRECTNESS_HARD_CAPS,
    fixture_digest,
    registered_digests,
    trusted_performance_fixture_identity,
)
from friday_h0.protocol import close_manifest
from tests.test_manifest import valid_manifest


def _arm(name, values, order):
    repetitions = 1
    warmup_evaluations = 500
    warmup_per_eval_ns = H0_BATCH_MIN_NS // warmup_evaluations
    batches = []
    raw_samples = [
        {"phase": "warmup", "sample_index": index, "value": warmup_per_eval_ns, "unit": "ns", "arm": name, "position": "calibration"}
        for index in range(8)
    ]
    raw_samples.append({"phase": "repetition_probe", "repetitions": repetitions, "sample_index": 0, "value": 100.0, "unit": "ns", "arm": name, "position": "calibration"})
    for block, value in enumerate(values):
        position = "first" if order[block] == name else "second"
        batch_ns = value * repetitions
        batches.append(
            {
                "block_index": block,
                "batch_ns": batch_ns,
                "per_eval_ns": value,
                "repetitions": repetitions,
                "position": position,
                "evaluation_ns": [1.0],
                "synchronize_ns": [1.0],
            }
        )
        raw_samples.append(
            {
                "phase": "measurement",
                "sample_kind": "timing_batch",
                "sample_index": block,
                "block_index": block,
                "arm": name,
                "position": position,
                "repetitions": repetitions,
                "value": batch_ns,
                "unit": "ns",
            }
        )
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    deviations = sorted(abs(value - median) for value in ordered)
    mad = deviations[len(deviations) // 2] if len(deviations) % 2 else (deviations[len(deviations) // 2 - 1] + deviations[len(deviations) // 2]) / 2.0
    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "warmup": {
            "count": 8,
            "durations_ns": [warmup_per_eval_ns] * 8,
            "stable": True,
            "median_ns": float(warmup_per_eval_ns),
            "samples": [
                {"phase": "warmup", "sample_index": index, "value": warmup_per_eval_ns, "unit": "ns"}
                for index in range(8)
            ],
            "blocks": [
                {
                    "block_index": index,
                    "evaluations": warmup_evaluations,
                    "block_ns": H0_BATCH_MIN_NS,
                    "per_eval_ns": warmup_per_eval_ns,
                    "median_eval_ns": warmup_per_eval_ns,
                    "min_eval_ns": warmup_per_eval_ns - 1,
                    "max_eval_ns": warmup_per_eval_ns + 1,
                }
                for index in range(8)
            ],
        },
        "repetitions": {
            "repetitions": repetitions,
            "batch_ns": 100_000_000.0,
            "probe_timings": [100.0],
            "calibration_samples": [
                {"phase": "repetition_probe", "repetitions": repetitions, "sample_index": 0, "value": 100.0, "unit": "ns"}
            ],
        },
        "batches": batches,
        "statistics": {
            "count": len(values),
            "median_ns": median,
            "mad_ns": mad,
            "iqr_ns": quantile(0.75) - quantile(0.25),
            "min_ns": min(values),
            "max_ns": max(values),
        },
        "calibration_samples": list(raw_samples[:9]),
        "raw_samples": raw_samples,
    }


def _metrics():
    return {
        "abs_q50": 0.0, "abs_q95": 0.0, "abs_q99": 0.0, "abs_max": 0.0,
        "rel_q50": 0.0, "rel_q95": 0.0, "rel_q99": 0.0, "rel_max": 0.0,
        "rel_q99_abs_oracle_ge_1": {"value": 0.0, "missing_reason": None},
        "normalized_l2": 0.0, "scaled_normalized_inf": 0.0,
    }


def _correctness_cases(*, performance_seed=0):
    cases = []
    for name, spec in CORRECTNESS_CASE_SPECS.items():
        shape, seed, zero_rhs = spec["shape"], spec["seed"], spec["zero_rhs"]
        a_digest, b_digest, digest = registered_digests(name)
        cases.append({
            "name": name, "shape": shape, "dtype": "float16", "layout": "C-contiguous",
            "seed": seed, "a_sha256": a_digest, "b_sha256": b_digest,
            "fixture_digest": digest, "zero_rhs": zero_rhs, "metrics": _metrics(),
            "passed": True, "hard_caps": dict(CORRECTNESS_HARD_CAPS),
        })
    identity = trusted_performance_fixture_identity(
        a_shape=[2048, 2048], b_shape=[2048, 2048], dtype="float16",
        layout="C-contiguous", fixture_seed=performance_seed,
    )
    digest = fixture_digest(identity["a_sha256"], identity["b_sha256"], performance_seed)
    cases.append({
        "name": "performance_fixture", "shape": [2048, 2048, 2048, 2048],
        "dtype": "float16", "layout": "C-contiguous", "seed": performance_seed,
        "a_sha256": identity["a_sha256"], "b_sha256": identity["b_sha256"],
        "fixture_digest": digest, "zero_rhs": False, "metrics": _metrics(),
        "passed": True, "hard_caps": dict(CORRECTNESS_HARD_CAPS),
    })
    cases.append({
        "name": "sign_invariant", "seed": performance_seed, "fixture_digest": digest,
        "reference": "performance_fixture", "relation": "negate_left_operand", "passed": True,
    })
    return cases


def _session(process_set, index, scale=1.0, *, run_id=None, values=None):
    manifest = valid_manifest("aa_gpu", set_name=process_set, index=index)
    manifest["run_id"] = run_id or f"run-{process_set}-{index}"
    identity = trusted_performance_fixture_identity(
        a_shape=manifest["workload"]["a_shape"], b_shape=manifest["workload"]["b_shape"],
        dtype=manifest["workload"]["dtype"], layout=manifest["workload"]["layout"],
        fixture_seed=manifest["seeds"]["fixture"],
    )
    order = _balanced_order(manifest["seeds"]["order"])
    baseline = [100_000_000.0] * 30
    candidate = list(values) if values is not None else [100_000_000.0 * scale] * 30
    arms = {"baseline": _arm("baseline", baseline, order), "candidate": _arm("candidate", candidate, order)}
    blocks = []
    for block, first in enumerate(order):
        baseline_value = arms["baseline"]["batches"][block]["per_eval_ns"]
        candidate_value = arms["candidate"]["batches"][block]["per_eval_ns"]
        blocks.append(
            {
                "block_index": block,
                "first": first,
                "second": "candidate" if first == "baseline" else "baseline",
                "baseline_per_eval_ns": baseline_value,
                "candidate_per_eval_ns": candidate_value,
                "ratio": candidate_value / baseline_value,
            }
        )
    ratio_values = [candidate[index] / baseline[index] for index in range(len(baseline))]
    ratio_ordered = sorted(ratio_values)
    ratio_middle = len(ratio_ordered) // 2
    ratio_median = ratio_ordered[ratio_middle] if len(ratio_ordered) % 2 else (ratio_ordered[ratio_middle - 1] + ratio_ordered[ratio_middle]) / 2.0
    ratio_deviations = sorted(abs(value - ratio_median) for value in ratio_ordered)
    ratio_mad = ratio_deviations[len(ratio_deviations) // 2] if len(ratio_deviations) % 2 else (ratio_deviations[len(ratio_deviations) // 2 - 1] + ratio_deviations[len(ratio_deviations) // 2]) / 2.0
    def ratio_quantile(fraction):
        position = (len(ratio_ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ratio_ordered[low] if low == high else ratio_ordered[low] + (ratio_ordered[high] - ratio_ordered[low]) * (position - low)
    domain = {
        "fixture": {
            "fixture_seed": identity["fixture_seed"],
            "a_sha256": identity["a_sha256"],
            "b_sha256": identity["b_sha256"],
            "metadata_sha256": identity["metadata_sha256"],
            "fixture_sha256": identity["fixture_sha256"],
        },
        "correctness": {
            "passed": True,
            "cases": _correctness_cases(performance_seed=manifest["seeds"]["fixture"]),
            "performance": next(case for case in _correctness_cases(performance_seed=manifest["seeds"]["fixture"]) if case["name"] == "performance_fixture"),
            "sign_invariant": next(case for case in _correctness_cases(performance_seed=manifest["seeds"]["fixture"]) if case["name"] == "sign_invariant"),
        },
        "memory": [
            {"name": "mlx_peak_memory", "api": "fake", "unit": "bytes", "value": 1000, "missing_reason": None, "measurement_phase": "after_calibration", "arm": "baseline", "measured_at_ns": 1, "reset_state": "not_reset_or_api_unavailable"},
            {"name": "rss", "api": "fake", "unit": "bytes", "value": 2000, "missing_reason": None, "measurement_phase": "after_calibration", "arm": "baseline", "measured_at_ns": 1, "reset_state": "not_reset_or_api_unavailable"},
        ],
        "memory_limit": {"attempted": True, "hard_limit": False, "applied": True, "missing_reason": None},
        "memory_gate": "aggregation_required",
        "cache_state": "unknown",
        "fresh_process_required": True,
        "aggregation_required": True,
        "compile_wrapper_setup_ns": None,
        "first_eval_compile_inclusive_ns": None,
        "total_elapsed_ns": 1_000_000_000,
        "arms": arms,
        "comparison": {
            "order": order,
            "blocks": blocks,
            "raw_samples": arms["baseline"]["raw_samples"] + arms["candidate"]["raw_samples"],
            "ratio_statistics": {"count": 30, "median_ratio": ratio_median, "mad_ratio": ratio_mad, "iqr_ratio": ratio_quantile(0.75) - ratio_quantile(0.25), "min_ratio": min(ratio_values), "max_ratio": max(ratio_values)},
            "benchmark_classification": "session_observation",
            "action": "aggregation_required",
            "aggregation_required": True,
            "aggregation_gate": "aa_gate",
            "comparison_kind": "aa_gpu_null_control",
            "global_decision": None,
        },
        "raw_samples": arms["baseline"]["raw_samples"] + arms["candidate"]["raw_samples"],
    }
    closed = close_manifest(manifest)
    result = {
        "schema_version": 1,
        "run_id": closed.run_id,
        "mode": closed.mode,
        "manifest_sha256": closed.sha256,
        "status": "completed",
        "classification": "measurement_complete",
        "action": "baseline_fallback",
        "error": None,
        "evidence": {
            "rss_peak_bytes": 2000,
            "rss_missing_reason": None,
            "benchmark_classification": "measurement_complete",
            "benchmark_action": "aggregation_required",
            "aggregation_required": True,
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
            "benchmark_evidence": domain,
        },
    }
    return {"manifest": manifest, "result": result}


def _fixture(scale=1.0, *, char_scale=None, conf_scale=None):
    char = [_session("characterization", index, char_scale if char_scale is not None else scale) for index in range(3)]
    conf = [_session("confirmation", index, conf_scale if conf_scale is not None else scale) for index in range(3)]
    return char, conf


class AggregationTests(unittest.TestCase):
    def test_pass_fixture_is_reproducible_and_complete(self):
        char, conf = _fixture()
        first = aggregate_h0_aa(char, conf)
        second = aggregate_h0_aa(char, conf)
        self.assertEqual(first, second)
        self.assertEqual(first["classification"], "h0_valid")
        self.assertTrue(first["h0_valid"])
        self.assertEqual(first["action"], "baseline_reference")
        self.assertFalse(first["scientific_equivalence_claim"])
        self.assertEqual(first["bootstrap_replicates"], 10_000)
        self.assertTrue(first["bootstrap_seed_manifest_bound"])
        self.assertTrue(first["aggregation_contract_ready"])
        self.assertFalse(first["live_execution_authorized"])
        self.assertEqual(first["gates"]["characterization"]["bootstrap_seed"], 0xAA052026)
        self.assertEqual(first["gates"]["confirmation"]["bootstrap_seed"], 0xAA052126)
        canonical_json_bytes(first)

    def test_median_band_and_session_band_failures(self):
        char, conf = _fixture(char_scale=1.10)
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertTrue(any("r_aa_outside_engineering_band" in reason for reason in result["failures"]))
        char, conf = _fixture(char_scale=1.0)
        char[0] = _session("characterization", 0, 1.10)
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("session_0_outside_session_band" in reason for reason in result["failures"]))

    def test_confidence_interval_and_characterization_only_pass_are_not_promotions(self):
        values = [90_000_000.0] * 15 + [110_000_000.0] * 15
        char, conf = _fixture()
        char = [_session("characterization", index, values=values) for index in range(3)]
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertTrue(any("ci_outside_engineering_band" in reason for reason in result["failures"]))
        char, conf = _fixture(char_scale=1.0, conf_scale=1.10)
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertNotIn(result["classification"], {"promoted", "regression"})

    def test_missing_duplicate_process_and_provenance_fail_closed(self):
        char, conf = _fixture()
        result = aggregate_h0_aa(char[:2], conf)
        self.assertIn("sets:must_contain_exactly_three_sessions", result["failures"])
        char, conf = _fixture()
        char[2] = copy.deepcopy(char[1])
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("duplicate_run_id" in reason for reason in result["failures"]))
        char, conf = _fixture()
        char[1]["manifest"]["provenance"]["code_sha256"] = "e" * 64
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("manifest hash mismatch" in reason or "provenance_mismatch" in reason for reason in result["failures"]))

    def test_ratio_direction_order_common_status_and_nonfinite_fail_closed(self):
        char, conf = _fixture(char_scale=1.10)
        wrong = copy.deepcopy(char[0])
        wrong["result"]["evidence"]["benchmark_evidence"]["comparison"]["blocks"][0]["ratio"] = 1.0
        char[0] = wrong
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("comparison_ratio_or_arm_mismatch" in reason for reason in result["failures"]))
        char, conf = _fixture()
        wrong = copy.deepcopy(char[0])
        wrong["result"]["evidence"]["benchmark_evidence"]["comparison"]["order"][0] = "candidate"
        char[0] = wrong
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("order_evidence_mismatch" in reason for reason in result["failures"]))
        char, conf = _fixture()
        wrong = copy.deepcopy(char[0])
        wrong["result"]["status"] = "invalid"
        wrong["result"]["error"] = {"code": "x", "message": "x"}
        char[0] = wrong
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")

    def test_timing_warmup_and_derived_statistics_are_reconstructed(self):
        for mutation in ("zero", "negative", "bool"):
            char, conf = _fixture()
            value = {"zero": 0.0, "negative": -1.0, "bool": True}[mutation]
            char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["batches"][0]["per_eval_ns"] = value
            result = aggregate_h0_aa(char, conf)
            self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["warmup"]["samples"][0]["value"] = 101.0
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("warmup_declared_sample_mismatch" in reason for reason in result["failures"]))
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["comparison"]["ratio_statistics"]["median_ratio"] = 2.0
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("ratio_statistics_not_reconstructed_from_blocks" in reason for reason in result["failures"]))
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["warmup"]["median_ns"] = 101.0
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("warmup_median_not_reconstructed" in reason for reason in result["failures"]))
        char, conf = _fixture()
        warmup = char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["warmup"]
        self.assertEqual(len(warmup["blocks"]), warmup["count"])
        warmup["blocks"][0]["block_ns"] = H0_BATCH_MIN_NS - 1
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("warmup_block_cross_bind_invalid" in reason for reason in result["failures"]))
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["warmup"]["blocks"][0]["per_eval_ns"] += 1
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("warmup_block_cross_bind_invalid" in reason for reason in result["failures"]))
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["repetitions"]["probe_timings"][0] = 101.0
        result = aggregate_h0_aa(char, conf)
        self.assertTrue(any("probe_timing_calibration_mismatch" in reason for reason in result["failures"]))
        char, conf = _fixture()
        wrong = copy.deepcopy(char[0])
        wrong["result"]["evidence"]["benchmark_evidence"]["arms"]["candidate"]["batches"][0]["per_eval_ns"] = 0.0
        char[0] = wrong
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")

    def test_memory_not_evaluable_is_visible_and_fail_closed(self):
        char, conf = _fixture()
        evidence = char[0]["result"]["evidence"]["benchmark_evidence"]
        evidence["memory_gate"] = "not_evaluable_missing_required_metric"
        for row in evidence["memory"]:
            if row["name"] in {"mlx_peak_memory", "rss"}:
                row["value"] = None
                row["missing_reason"] = "api_unavailable"
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_valid")
        self.assertEqual(result["session_contracts"]["characterization"][0]["memory"]["status"], "not_evaluable")
        self.assertFalse(result["session_contracts"]["characterization"][0]["memory"]["promotion_gate_applicable"])
        self.assertEqual(
            result["session_contracts"]["characterization"][0]["memory_limit"],
            {"attempted": True, "hard_limit": False, "applied": True, "missing_reason": None},
        )

    def test_memory_name_and_reason_are_closed_and_registered_fallbacks_are_accepted(self):
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]["name"] = "arbitrary_memory"
        self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_invalid")

        char, conf = _fixture()
        row = char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]
        row["value"] = None
        row["missing_reason"] = "arbitrary_reason"
        self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_invalid")

        reasons = (
            "not_recorded", "not_applicable", "api_unavailable", "unavailable", "no_sample",
            "source_missing", "entry_limit", "ps_exit", "ps_parse", "ps_negative",
            "parent_setup_failure", "invalid_source_value",
        )
        for reason in reasons:
            char, conf = _fixture()
            evidence = char[0]["result"]["evidence"]["benchmark_evidence"]
            evidence["memory_gate"] = "not_evaluable_missing_required_metric"
            for memory in evidence["memory"]:
                if memory["name"] in {"mlx_peak_memory", "rss"}:
                    memory["value"] = None
                    memory["missing_reason"] = reason
            self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_valid", reason)

        for value in (True, 1 << 63):
            char, conf = _fixture()
            char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]["value"] = value
            self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_invalid")

    def test_size_limit_and_output_are_bounded(self):
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["raw_samples"] = ["x" * 1_100_000]
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertLess(len(canonical_json_bytes(result)), 1 << 20)

    def test_adapter_contract_and_supervisor_rss_contract(self):
        char, conf = _fixture()
        char[0]["result"]["evidence"]["adapter_contract"]["common_result_ready"] = True
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertTrue(any("adapter_contract" in reason for reason in result["failures"]))
        char, conf = _fixture()
        del char[0]["result"]["evidence"]["adapter_contract"]
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        char[0]["result"]["evidence"]["rss_peak_bytes"] = None
        char[0]["result"]["evidence"]["rss_missing_reason"] = "unavailable"
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_valid")
        self.assertEqual(result["session_contracts"]["characterization"][0]["supervisor_rss"]["value"], None)
        char, conf = _fixture()
        char[0]["result"]["evidence"]["rss_peak_bytes"] = None
        del char[0]["result"]["evidence"]["rss_missing_reason"]
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")

    def test_manifest_bootstrap_seed_is_bound_and_replayable(self):
        char, conf = _fixture()
        first = aggregate_h0_aa(char, conf)
        second = aggregate_h0_aa(copy.deepcopy(char), copy.deepcopy(conf))
        self.assertEqual(first, second)
        wrong = copy.deepcopy(char)
        wrong[0]["manifest"]["seeds"]["bootstrap_seed"] += 1
        result = aggregate_h0_aa(wrong, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertTrue(any("bootstrap" in failure for failure in result["failures"]))

    def test_memory_xor_and_duplicate_are_rejected(self):
        char, conf = _fixture()
        memory = char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]
        memory["missing_reason"] = "telemetry_missing"
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        duplicate = copy.deepcopy(char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0])
        char[0]["result"]["evidence"]["benchmark_evidence"]["memory"].append(duplicate)
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]["measurement_phase"] = "unknown"
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")

    def test_correctness_and_raw_sample_closure(self):
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["correctness"]["cases"] = []
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["correctness"]["cases"][0]["name"] = "unknown_case"
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["raw_samples"][0]["phase"] = "unknown"
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        raw = char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["candidate"]["raw_samples"]
        raw[-1]["block_index"] = 0
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        char, conf = _fixture()
        raw = char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["candidate"]["raw_samples"]
        raw[-1]["sample_index"] = 0
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")

    def test_huge_integer_is_bounded_without_exception(self):
        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["arms"]["baseline"]["batches"][0]["batch_ns"] = 1 << 5000
        result = aggregate_h0_aa(char, conf)
        self.assertEqual(result["classification"], "h0_invalid")
        self.assertLess(len(canonical_json_bytes(result)), 1 << 20)

    def test_integer_boundaries_match_runner_worker_without_capping_float_metrics(self):
        maximum = (1 << 63) - 1
        self.assertEqual(_finite(maximum, "timing", positive=True), float(maximum))
        self.assertEqual(_finite(1.0e300, "ratio", positive=True), 1.0e300)
        for value in (1 << 63, 0, -1, True):
            with self.assertRaises(ValueError):
                _finite(value, "timing", positive=True)

        char, conf = _fixture()
        char[0]["result"]["evidence"]["benchmark_evidence"]["total_elapsed_ns"] = maximum
        char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]["measured_at_ns"] = maximum
        self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_valid")

        for value in (1 << 63, -1, True):
            char, conf = _fixture()
            char[0]["result"]["evidence"]["benchmark_evidence"]["total_elapsed_ns"] = value
            self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_invalid")

            char, conf = _fixture()
            char[0]["result"]["evidence"]["benchmark_evidence"]["memory"][0]["measured_at_ns"] = value
            self.assertEqual(aggregate_h0_aa(char, conf)["classification"], "h0_invalid")


if __name__ == "__main__":
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AggregationTests)
    result = unittest.TextTestRunner().run(suite)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(f"elapsed_s={time.monotonic()-started:.6f} user_s={usage.ru_utime:.6f} sys_s={usage.ru_stime:.6f} maxrss={usage.ru_maxrss}")
    raise SystemExit(not result.wasSuccessful())
