from __future__ import annotations

import copy
import hashlib
import math
import unittest

from friday_h01.analysis import analyze_trace
from friday_h01.canonical import canonical_sha256
from friday_h01.constants import (
    BURN_IN_SAMPLES,
    MAIN_SAMPLES,
    SCHEMA_VERSION,
    SESSION_COMPLETE_STATUS,
    SESSION_INVALID_STATUS,
    TOTAL_SAMPLES,
)
from friday_h01.protocol import ProtocolError, build_trace, validate_result
from tests.test_h01_protocol import integer_leaf_paths, make_manifest, path_text, replace_leaf


BASE_NS = 1_000_000


def hash_noise_main(label: str) -> list[int]:
    values = []
    for index in range(MAIN_SAMPLES):
        word = int.from_bytes(hashlib.sha256(f"{label}:{index}".encode("ascii")).digest()[:8], "big")
        centered = word / float(1 << 64) - 0.5
        values.append(round(BASE_NS * math.exp(centered * 0.008)))
    return values


def stationary_ar1_main(label: str) -> list[int]:
    state = 0.0
    values = []
    for index in range(2_000):
        word = int.from_bytes(hashlib.sha256(f"{label}:{index}".encode("ascii")).digest()[:8], "big")
        innovation = (word / float(1 << 64) - 0.5) * 0.001
        state = 0.97 * state + innovation
        if index >= 2_000 - MAIN_SAMPLES:
            values.append(round(BASE_NS * math.exp(state)))
    return values


def _durations(main_values: list[int]) -> list[int]:
    if len(main_values) != MAIN_SAMPLES:
        raise AssertionError("test fixture must have exactly 80 main values")
    return [BASE_NS] * BURN_IN_SAMPLES + main_values


def _overshoots(label: str) -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"gap:{label}:{index}".encode("ascii")).digest()[:4], "big")
        % 20_000_001
        for index in range(TOTAL_SAMPLES)
    ]


def analyze_fixture(main_values: list[int], session_id: str = "C0") -> tuple[dict, dict, dict]:
    manifest = make_manifest(session_id)
    trace = build_trace(
        manifest,
        _durations(main_values),
        gap_overshoots_ns=_overshoots(session_id),
    )
    return manifest, trace, analyze_trace(manifest, trace)


def _rehash_result(result: dict) -> None:
    result["decision_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "decision_sha256"}
    )


class H01AnalysisTests(unittest.TestCase):
    def test_realistic_nonconstant_hash_noise_session_is_complete_with_all_gates_pass(self) -> None:
        manifest, trace, first = analyze_fixture(hash_noise_main("stationary"))
        second = analyze_trace(manifest, trace)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], SESSION_COMPLETE_STATUS)
        self.assertEqual(first["conclusion"], "session_characterized")
        self.assertTrue(all(gate["status"] == "pass" for gate in first["gates"].values()))
        self.assertGreater(first["metrics"]["trajectory"]["observed_span_seconds"], 0.0)
        self.assertEqual(
            first["metrics"]["actual_gap_sha256"],
            canonical_sha256(
                [
                    sample["gap_end_ns"] - sample["gap_start_ns"]
                    for sample in trace["samples"][BURN_IN_SAMPLES:]
                ]
            ),
        )
        self.assertEqual(validate_result(first, manifest, trace), first)

    def test_six_percent_drift_fails_trend_but_session_remains_complete(self) -> None:
        main = [round(BASE_NS * math.exp(math.log(1.06) * index / 79.0)) for index in range(MAIN_SAMPLES)]
        _manifest, _trace, result = analyze_fixture(main, "V0")
        self.assertEqual(result["status"], SESSION_COMPLETE_STATUS)
        self.assertEqual(result["gates"]["trend"]["status"], "fail")
        self.assertGreater(result["gates"]["trend"]["observed"], 0.05)

    def test_seven_percent_step_fails_changepoint_but_session_remains_complete(self) -> None:
        main = [BASE_NS] * 40 + [round(BASE_NS * 1.07)] * 40
        _manifest, _trace, result = analyze_fixture(main, "C1")
        self.assertEqual(result["status"], SESSION_COMPLETE_STATUS)
        self.assertEqual(result["gates"]["changepoint"]["status"], "fail")
        self.assertGreater(result["gates"]["changepoint"]["observed"], 0.05)

    def test_stationary_ar1_with_deterministic_innovations_fails_acf_and_ess(self) -> None:
        _manifest, _trace, result = analyze_fixture(stationary_ar1_main("ar1"), "V1")
        self.assertEqual(result["status"], SESSION_COMPLETE_STATUS)
        self.assertEqual(result["gates"]["acf"]["status"], "fail")
        self.assertEqual(result["gates"]["ess"]["status"], "fail")

    def test_thirty_percent_spike_fails_tail_gate(self) -> None:
        main = [BASE_NS] * MAIN_SAMPLES
        main[40] = round(BASE_NS * 1.30)
        _manifest, _trace, result = analyze_fixture(main, "C2")
        self.assertEqual(result["status"], SESSION_COMPLETE_STATUS)
        self.assertEqual(result["gates"]["tail"]["status"], "fail")
        self.assertGreater(result["gates"]["tail"]["observed"], 1.20)

    def test_four_percent_pacing_effect_fails_pacing_gate(self) -> None:
        manifest = make_manifest("V2")
        labels = [entry["gap_label"] for entry in manifest["schedule"]["entries"][BURN_IN_SAMPLES:]]
        main = [BASE_NS if label.startswith("short") else round(BASE_NS * 1.04) for label in labels]
        trace = build_trace(manifest, _durations(main), gap_overshoots_ns=_overshoots("V2"))
        result = analyze_trace(manifest, trace)
        self.assertEqual(result["status"], SESSION_COMPLETE_STATUS)
        self.assertEqual(result["gates"]["pacing"]["status"], "fail")
        self.assertGreater(result["gates"]["pacing"]["observed"], 0.03)

    def test_trace_fault_is_invalid_and_replays_as_invalid(self) -> None:
        manifest = make_manifest()
        trace = build_trace(manifest, [BASE_NS] * TOTAL_SAMPLES)
        del trace["samples"][3]
        result = analyze_trace(manifest, trace)
        self.assertEqual(result["status"], SESSION_INVALID_STATUS)
        self.assertEqual(result["conclusion"], "invalid_input")
        self.assertIsNone(result["metrics"])
        self.assertEqual(validate_result(result, manifest, trace), result)

    def test_result_metric_gate_status_and_full_co_mutations_are_rejected(self) -> None:
        manifest, trace, result = analyze_fixture(hash_noise_main("replay"))

        metric = copy.deepcopy(result)
        metric["metrics"]["trajectory"]["effect_ratio"] += 0.001
        _rehash_result(metric)
        with self.assertRaises(ProtocolError):
            validate_result(metric, manifest, trace)

        gate = copy.deepcopy(result)
        gate["gates"]["trend"]["observed"] += 0.001
        _rehash_result(gate)
        with self.assertRaises(ProtocolError):
            validate_result(gate, manifest, trace)

        status = copy.deepcopy(result)
        status.update(
            {
                "trace_sha256": None,
                "status": SESSION_INVALID_STATUS,
                "conclusion": "invalid_input",
                "error": {"code": "forged", "message": "co-mutated complete envelope"},
                "sample_accounting": None,
                "metrics": None,
                "gates": None,
            }
        )
        _rehash_result(status)
        with self.assertRaises(ProtocolError):
            validate_result(status, manifest, trace)

        full = copy.deepcopy(result)
        full["metrics"]["pace_effect_ratio"] = 0.02
        full["metrics"]["tail_ratio"] = 1.10
        full["gates"]["pacing"].update({"observed": 0.02, "status": "pass"})
        full["gates"]["tail"].update({"observed": 1.10, "status": "pass"})
        _rehash_result(full)
        with self.assertRaises(ProtocolError):
            validate_result(full, manifest, trace)

    def test_bootstrap_and_p_cp_fields_are_not_permitted(self) -> None:
        manifest, trace, result = analyze_fixture(hash_noise_main("no-inference"))
        self.assertNotIn("bootstrap", result["metrics"])
        self.assertNotIn("bootstrap", result["gates"])
        self.assertNotIn("p_cp", result["metrics"])
        forged = copy.deepcopy(result)
        forged["metrics"]["bootstrap"] = {}
        _rehash_result(forged)
        with self.assertRaises(ProtocolError):
            validate_result(forged, manifest, trace)

    def test_every_result_integer_leaf_rejects_bool_before_replay(self) -> None:
        manifest, trace, result = analyze_fixture(hash_noise_main("integer"))
        paths = integer_leaf_paths(result)
        self.assertEqual(len(paths), 8)
        for path in paths:
            mutation = replace_leaf(result, path, True)
            _rehash_result(mutation)
            expected_path = path_text("result", path)
            with self.subTest(path=expected_path), self.assertRaises(ProtocolError) as raised:
                validate_result(mutation, manifest, trace)
            self.assertIn(expected_path, str(raised.exception))
        false_result = replace_leaf(result, ("sample_accounting", "dropped_samples"), False)
        _rehash_result(false_result)
        with self.assertRaises(ProtocolError):
            validate_result(false_result, manifest, trace)
        float_result = replace_leaf(result, ("schema_version",), float(SCHEMA_VERSION))
        _rehash_result(float_result)
        with self.assertRaises(ProtocolError):
            validate_result(float_result, manifest, trace)


if __name__ == "__main__":
    unittest.main()
