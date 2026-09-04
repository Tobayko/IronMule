"""Offline contract tests for the one-shot Gemma-4B planner study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT_ROOT / "experiments" / "planner_4b"


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


measure = load("planner_4b_measure", EXPERIMENT / "measure_planner_4b.py")
worker = load("planner_4b_worker_test", EXPERIMENT / "worker.py")
dashboard = load("planner_4b_dashboard", EXPERIMENT / "dashboard.py")


class FrozenContractTest(unittest.TestCase):
    def test_preregistration_is_the_frozen_input(self) -> None:
        digest = hashlib.sha256((EXPERIMENT / "PREREGISTRATION.md").read_bytes()).hexdigest()
        self.assertEqual(digest, measure.FROZEN_PREREGISTRATION_SHA256)

    def test_hardware_settings_are_closed(self) -> None:
        self.assertEqual(measure.MODEL_ID, worker.MODEL_ID)
        self.assertEqual(measure.MODEL_REVISION, worker.MODEL_REVISION)
        self.assertEqual(measure.RUN_ID, worker.RUN_ID)
        self.assertEqual(worker.MAX_OUTPUT_TOKENS, 32)
        self.assertEqual(worker.PREFILL_STEP_SIZE, 256)
        self.assertEqual(worker.EXPECTED_CANDIDATE, measure.EXPECTED_CANDIDATE)
        self.assertEqual(measure.EXPECTED_CPU_BRAND, "Apple M1 Max")
        self.assertEqual(measure.EXPECTED_MACHINE, "arm64")
        self.assertEqual(measure.EXPECTED_MEMORY_BYTES, 32 * 1024**3)
        self.assertEqual(measure.REQUIRED_PACKAGES, {"mlx": "0.32.0", "mlx-lm": "0.31.3"})
        self.assertEqual(measure.POLICY.duty_cycle_limit, 0.15)
        self.assertLess(measure.PACING_TARGET, measure.POLICY.duty_cycle_limit)

    def test_default_invocation_cannot_touch_hardware(self) -> None:
        self.assertEqual(measure.main([]), 78)
        self.assertEqual(worker.main(["--self-check"]), 0)


class PlannerAnswerTest(unittest.TestCase):
    def test_accepts_only_one_fixed_candidate_field(self) -> None:
        self.assertEqual(
            worker.parse_choice('{"candidate_id":"persistent_service_qualification"}'),
            worker.EXPECTED_CANDIDATE,
        )
        for invalid in (
            "persistent_service_qualification",
            '```json\n{"candidate_id":"persistent_service_qualification"}\n```',
            '{"candidate_id":"unknown"}',
            '{"candidate_id":"persistent_service_qualification","command":"run"}',
            '{"candidate_id":"persistent_service_qualification","candidate_id":"batched_readback"}',
            "NaN",
            "",
        ):
            with self.assertRaises(worker.WorkerError):
                worker.parse_choice(invalid)

    def test_prompt_and_choice_list_are_fixed(self) -> None:
        self.assertEqual(len(worker.ALLOWED_CANDIDATES), 4)
        self.assertEqual(
            hashlib.sha256(worker.PLANNER_PROMPT.encode("utf-8")).hexdigest(),
            worker.PROMPT_SHA256,
        )
        self.assertIn("65.3032%", worker.PLANNER_PROMPT)


class DecisionTableTest(unittest.TestCase):
    def decide(self, **changes):
        values = {
            "identity": True,
            "contract": True,
            "priority": True,
            "resources": True,
            "budget": True,
        }
        values.update(changes)
        return measure.decision_for(**values)

    def test_each_failure_has_the_frozen_terminal_meaning(self) -> None:
        self.assertEqual(self.decide(identity=False), "correctness_failed")
        self.assertEqual(self.decide(contract=False), "planner_contract_failed")
        self.assertEqual(self.decide(priority=False), "planner_priority_failed")
        self.assertEqual(self.decide(resources=False), "resource_or_budget_failed")
        self.assertEqual(self.decide(budget=False), "resource_or_budget_failed")
        self.assertEqual(self.decide(), "planner_4b_qualified_exact_case")


class MeasurementContractTest(unittest.TestCase):
    @staticmethod
    def sample_run(pid: int, token: int = 7, candidate: str | None = None) -> dict:
        return {
            "candidate_id": candidate or measure.EXPECTED_CANDIDATE,
            "finish_reason": "stop",
            "load_count": 1,
            "mlx_peak_bytes": 2_000_000_000,
            "pid": pid,
            "prompt_tokens": 300,
            "rss_peak_bytes": 3_000_000_000,
            "text": '{"candidate_id":"persistent_service_qualification"}',
            "tokens": [token, 1],
        }

    def test_identity_requires_all_three_fresh_exact_runs(self) -> None:
        runs = [self.sample_run(10), self.sample_run(11), self.sample_run(12)]
        self.assertTrue(measure._identity_gate(runs))
        runs[1]["tokens"] = [8, 1]
        self.assertFalse(measure._identity_gate(runs))
        runs[1]["tokens"] = [7, 1]
        runs[2]["pid"] = 11
        self.assertFalse(measure._identity_gate(runs))

    def test_length_finish_is_not_a_success(self) -> None:
        runs = [self.sample_run(10), self.sample_run(11), self.sample_run(12)]
        runs[2]["finish_reason"] = "length"
        self.assertFalse(measure._identity_gate(runs))

    def test_contract_and_priority_are_separate(self) -> None:
        runs = [self.sample_run(10), self.sample_run(11), self.sample_run(12)]
        self.assertTrue(measure._contract_gate(runs))
        self.assertTrue(measure._priority_gate(runs))
        for run in runs:
            run["candidate_id"] = "batched_readback"
        self.assertTrue(measure._contract_gate(runs))
        self.assertFalse(measure._priority_gate(runs))

    def test_partial_resources_fail_safely(self) -> None:
        resources = measure._resource_summary([self.sample_run(10)], 0, 0)
        self.assertFalse(resources["gate_passed"])
        resources = measure._resource_summary(
            [self.sample_run(10), self.sample_run(11), self.sample_run(12)], 0, 1
        )
        self.assertFalse(resources["gate_passed"])

    def test_child_environment_is_offline_and_ignores_python_injection(self) -> None:
        poisoned = {name: "unsafe" for name in measure.UNSAFE_PYTHON_ENVIRONMENT}
        with mock.patch.dict(os.environ, {**poisoned, "HF_HUB_OFFLINE": "0"}):
            environment = measure._worker_environment()
        for name in poisoned:
            self.assertNotIn(name, environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["FRIDAY_PLANNER_4B_RUN_ID"], measure.RUN_ID)

    def test_target_environment_rejects_wrong_hardware(self) -> None:
        with mock.patch.object(measure.platform, "machine", return_value="x86_64"):
            with self.assertRaises(measure.StudyError):
                measure._require_target_environment()

    def test_attempt_directory_must_be_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "attempt"
            path.mkdir(mode=0o700)
            os.chmod(path, 0o700)
            measure._require_private_directory(path)
            os.chmod(path, 0o755)
            with self.assertRaises(measure.StudyError):
                measure._require_private_directory(path)

    def test_worker_output_must_be_exactly_one_json_line(self) -> None:
        self.assertEqual(measure._decode_event(b'{"event":"complete"}\n')["event"], "complete")
        for payload in (b"", b"{}\n{}\n", b"NaN\n"):
            with self.assertRaises(measure.WorkerError):
                measure._decode_event(payload)


class DashboardTest(unittest.TestCase):
    def result(self) -> dict:
        run = {
            "candidate_id": measure.EXPECTED_CANDIDATE,
            "compute_ns": 2_000_000_000,
            "finish_reason": "stop",
            "process_wall_ns": 5_000_000_000,
            "text": "secret raw answer",
            "token_sha256": "a" * 64,
            "tokens": [1, 2, 3],
        }
        return {
            "decision": "planner_4b_qualified_exact_case",
            "formal_claim": False,
            "gates": {"H1_greedy_identity": True},
            "metrics": {"runs_completed": 3},
            "resources": {"swap_delta_bytes": 0},
            "run_id": measure.RUN_ID,
            "runs": [dict(run), dict(run), dict(run)],
            "study_id": measure.STUDY_ID,
        }

    def test_snapshot_is_read_only_and_hides_raw_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(self.result()), encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            value = dashboard.snapshot(path)
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(value["read_only"])
            self.assertEqual(len(value["rows"]), 3)
            self.assertNotIn("tokens", value["rows"][0])
            self.assertNotIn("text", value["rows"][0])

    def test_symlinked_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text(json.dumps(self.result()), encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(dashboard.DashboardError):
                dashboard.snapshot(link)

    def test_page_uses_text_nodes_and_rejects_rebound_hosts(self) -> None:
        self.assertNotIn(b"innerHTML", dashboard.HTML)
        self.assertIn(b"textContent", dashboard.HTML)
        self.assertTrue(dashboard._host_header_allowed("127.0.0.1:8783", "127.0.0.1", 8783))
        self.assertTrue(dashboard._host_header_allowed("[::1]:8783", "::1", 8783))
        for value in (None, "evil.example:8783", "localhost:8783"):
            self.assertFalse(dashboard._host_header_allowed(value, "127.0.0.1", 8783))


if __name__ == "__main__":
    unittest.main()
