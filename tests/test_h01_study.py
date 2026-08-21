from __future__ import annotations

import copy
import unittest

from friday_h01.analysis import analyze_trace
from friday_h01.canonical import canonical_sha256
from friday_h01.constants import (
    BURN_IN_SAMPLES,
    MAIN_SAMPLES,
    SCHEMA_VERSION,
    SESSION_ORDER,
    STUDY_STATUSES,
)
from friday_h01.protocol import build_trace
from friday_h01.study import StudyError, analyze_study, validate_study_result
from tests.test_h01_analysis import BASE_NS, hash_noise_main
from tests.test_h01_protocol import integer_leaf_paths, make_manifest, path_text, replace_leaf


def _durations(main: list[int]) -> list[int]:
    return [BASE_NS] * BURN_IN_SAMPLES + main


def make_records(*, drift_session: str | None = None, mixed_environment: str | None = None) -> list[dict]:
    records = []
    for session_id in SESSION_ORDER:
        environment_label = (
            "different-environment"
            if mixed_environment == session_id
            else "environment"
        )
        manifest = make_manifest(session_id, environment_label=environment_label)
        if drift_session == session_id:
            main = [
                round(BASE_NS * (1.0 + 0.07 * index / (MAIN_SAMPLES - 1)))
                for index in range(MAIN_SAMPLES)
            ]
        else:
            main = hash_noise_main(f"study-{session_id}")
        trace = build_trace(manifest, _durations(main))
        result = analyze_trace(manifest, trace)
        records.append({"manifest": manifest, "trace": trace, "result": result})
    self_check = {record["manifest"]["session"]["id"] for record in records}
    if self_check != set(SESSION_ORDER) or len(records) != len(SESSION_ORDER):
        raise AssertionError("test study fixture is not six exact sessions")
    return records


def _rehash_study(result: dict) -> None:
    if result["status"] != "h01_invalid":
        identity = {
            "schema_version": result["schema_version"],
            "phase": result["phase"],
            "study": result["study"],
            "session_order": result["session_order"],
            "shared_provenance": result["shared_provenance"],
            "session_bindings": result["session_bindings"],
        }
        result["study_id"] = f"h01-study-{canonical_sha256(identity)}"
    result["decision_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "decision_sha256"}
    )


class H01StudyTests(unittest.TestCase):
    def test_all_six_stationary_sessions_are_required_for_supported_status(self) -> None:
        records = make_records()
        result = analyze_study(records)
        self.assertEqual(result["status"], "h01_stationarity_supported")
        self.assertEqual(result["session_order"], list(SESSION_ORDER))
        self.assertEqual(result["session_count"], 6)
        self.assertEqual(result["failed_gate_count"], 0)
        self.assertTrue(all(binding["all_gates_pass"] for binding in result["session_bindings"]))
        self.assertEqual(validate_study_result(result, records), result)

    def test_any_session_gate_failure_makes_complete_study_unresolved(self) -> None:
        records = make_records(drift_session="V2")
        self.assertEqual(records[-1]["result"]["status"], "h01_session_complete")
        result = analyze_study(records)
        self.assertEqual(result["status"], "h01_complete_unresolved")
        self.assertGreater(result["failed_gate_count"], 0)
        self.assertEqual(validate_study_result(result, records), result)

    def test_missing_duplicate_reordered_and_mixed_provenance_are_invalid(self) -> None:
        records = make_records()
        cases = {
            "missing": records[:-1],
            "duplicate": [records[0], records[0], *records[2:]],
            "reordered": [records[1], records[0], *records[2:]],
            "mixed_provenance": make_records(mixed_environment="V2"),
        }
        for name, candidate in cases.items():
            with self.subTest(name=name):
                result = analyze_study(candidate)
                self.assertEqual(result["status"], "h01_invalid")
                self.assertEqual(result["session_count"], 0)
                self.assertEqual(validate_study_result(result, candidate), result)

    def test_selectively_invalid_session_and_co_mutated_session_result_invalidate_study(self) -> None:
        invalid_trace_records = make_records()
        broken = invalid_trace_records[2]["trace"]
        broken["samples"][40]["gap_end_ns"] += 300_000_000
        broken["samples"][40]["start_ns"] = broken["samples"][40]["gap_end_ns"]
        self.assertEqual(analyze_study(invalid_trace_records)["status"], "h01_invalid")

        forged_records = make_records()
        forged = forged_records[3]["result"]
        forged["metrics"]["tail_ratio"] = 1.10
        forged["gates"]["tail"].update({"observed": 1.10, "status": "pass"})
        forged["decision_sha256"] = canonical_sha256(
            {key: value for key, value in forged.items() if key != "decision_sha256"}
        )
        self.assertEqual(analyze_study(forged_records)["status"], "h01_invalid")

    def test_study_result_full_co_mutation_is_rejected_by_six_session_replay(self) -> None:
        records = make_records()
        result = analyze_study(records)
        forged = copy.deepcopy(result)
        forged_binding = forged["session_bindings"][0]
        forged_binding["failed_gates"] = ["trend"]
        forged_binding["all_gates_pass"] = False
        forged["failed_gate_count"] = 1
        forged["status"] = "h01_complete_unresolved"
        forged["conclusion"] = "replicated_stationarity_not_supported"
        _rehash_study(forged)
        with self.assertRaises(StudyError):
            validate_study_result(forged, records)

    def test_study_status_set_has_only_three_symmetric_terminal_values(self) -> None:
        self.assertEqual(
            STUDY_STATUSES,
            frozenset(
                {"h01_stationarity_supported", "h01_complete_unresolved", "h01_invalid"}
            ),
        )

    def test_every_study_integer_leaf_and_boolean_impostor_is_rejected(self) -> None:
        records = make_records()
        result = analyze_study(records)
        paths = integer_leaf_paths(result)
        self.assertEqual(len(paths), 3)
        for path in paths:
            mutation = replace_leaf(result, path, True)
            _rehash_study(mutation)
            expected_path = path_text("study_result", path)
            with self.subTest(path=expected_path), self.assertRaises(StudyError) as raised:
                validate_study_result(mutation, records)
            self.assertIn(expected_path, str(raised.exception))

        bool_impostors = []
        flag = copy.deepcopy(result)
        flag["h0_reclassification"] = 0
        _rehash_study(flag)
        bool_impostors.append(flag)
        binding = copy.deepcopy(result)
        binding["session_bindings"][0]["all_gates_pass"] = 1
        _rehash_study(binding)
        bool_impostors.append(binding)
        for index, mutation in enumerate(bool_impostors):
            with self.subTest(bool_impostor=index), self.assertRaises(StudyError):
                validate_study_result(mutation, records)


if __name__ == "__main__":
    unittest.main()
