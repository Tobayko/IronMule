"""Offline contract tests for the one-shot persistent-process study."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT_ROOT / "experiments" / "persistent_process"


def load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


measure = load("persistent_process_measure", EXPERIMENT / "measure_persistent_process.py")
worker = load("persistent_process_worker", EXPERIMENT / "worker.py")
dashboard = load("persistent_process_dashboard", EXPERIMENT / "dashboard.py")


class FrozenContractTest(unittest.TestCase):
    def test_preregistration_is_the_frozen_input(self) -> None:
        digest = hashlib.sha256((EXPERIMENT / "PREREGISTRATION.md").read_bytes()).hexdigest()
        self.assertEqual(digest, measure.FROZEN_PREREGISTRATION_SHA256)

    def test_hardware_settings_are_closed(self) -> None:
        self.assertEqual(measure.MODEL_REVISION, worker.MODEL_REVISION)
        self.assertEqual(measure.MODEL_ID, worker.MODEL_ID)
        self.assertEqual(worker.OUTPUT_TOKENS, 32)
        self.assertEqual(worker.PREFILL_CHUNK, 256)
        self.assertEqual(set(worker.QUESTIONS), {"P", "Q", "R", "S"})
        self.assertEqual(worker.EXPECTED_PROMPT_TOKENS, dict.fromkeys("PQRS", 897))
        self.assertEqual(measure.POLICY.duty_cycle_limit, 0.15)

    def test_default_invocation_cannot_touch_hardware(self) -> None:
        self.assertEqual(measure.main([]), 78)
        self.assertEqual(worker.main(["--self-check"]), 0)


class WorkerProtocolTest(unittest.TestCase):
    def test_accepts_only_fixed_prompt_requests(self) -> None:
        value = worker.parse_request(
            '{"command":"request","prompt_key":"Q","request_id":"round-1"}'
        )
        self.assertEqual(value["prompt_key"], "Q")
        for invalid in (
            '{"command":"request","prompt_key":"Q","request_id":"ROUND"}',
            '{"command":"request","prompt_key":"X","request_id":"round-1"}',
            '{"command":"request","prompt_key":"Q","request_id":"round-1","text":"x"}',
            '{"command":"shutdown","extra":true}',
            "NaN",
        ):
            with self.assertRaises(worker.WorkerProtocolError):
                worker.parse_request(invalid)

    def test_shutdown_has_no_payload(self) -> None:
        self.assertEqual(worker.parse_request('{"command":"shutdown"}'), {"command": "shutdown"})


class DecisionTableTest(unittest.TestCase):
    def decide(self, **changes):
        values = {
            "calibration": True,
            "correctness": True,
            "characterization": True,
            "validation": True,
            "resources": True,
            "budget": True,
        }
        values.update(changes)
        return measure.decision_for(**values)

    def test_each_failure_has_the_frozen_terminal_meaning(self) -> None:
        self.assertEqual(self.decide(calibration=False), "calibration_failed")
        self.assertEqual(self.decide(correctness=False), "correctness_failed")
        self.assertEqual(
            self.decide(characterization=False, validation=None),
            "candidate_characterized_no_gain",
        )
        self.assertEqual(self.decide(validation=False), "candidate_not_confirmed")
        self.assertEqual(self.decide(resources=False), "resource_or_budget_failed")
        self.assertEqual(self.decide(budget=False), "resource_or_budget_failed")
        self.assertEqual(self.decide(), "engineering_gain_confirmed_exact_scope")

    def test_phase_gate_uses_every_pair_without_outlier_removal(self) -> None:
        pairs = [
            {"ratio": ratio, "token_identical": True}
            for ratio in (0.30, 0.32, 0.66)
        ]
        gate, summary = measure._phase_gate(pairs)
        self.assertFalse(gate)
        self.assertEqual(summary["ratios"], [0.30, 0.32, 0.66])

    def test_token_mismatch_is_not_replaced_by_a_quality_score(self) -> None:
        pairs = [
            {"ratio": 0.30, "token_identical": True},
            {"ratio": 0.31, "token_identical": False},
            {"ratio": 0.32, "token_identical": True},
        ]
        gate, summary = measure._phase_gate(pairs)
        self.assertFalse(gate)
        self.assertFalse(summary["token_identical"])

    def test_median_and_spread_are_deterministic(self) -> None:
        self.assertEqual(measure._median([0.31, 0.33, 0.35]), 0.33)
        self.assertTrue(math.isclose(measure._mad([0.31, 0.33, 0.35]), 0.02))


class MeasurementSafetyTest(unittest.TestCase):
    @staticmethod
    def phase(warm_pid: int, cold_base: int) -> dict:
        pairs = []
        for index in range(3):
            pairs.append(
                {
                    "cold": {
                        "load_count": 1,
                        "pid": cold_base + index,
                        "request_count": 1,
                    },
                    "token_identical": True,
                    "warm": {
                        "load_count": 1,
                        "pid": warm_pid,
                        "request_count": index + 2,
                    },
                }
            )
        return {
            "pairs": pairs,
            "ready": {"load_count": 1, "pid": warm_pid},
            "warmup": {"load_count": 1, "pid": warm_pid, "request_count": 1},
        }

    def test_characterization_no_gain_keeps_correctness_true(self) -> None:
        phase = self.phase(20, 100)
        state = {
            "characterization": phase,
            "validation": None,
            "cold_pids": [1, 2, 3, 4, 100, 101, 102],
            "warm_pids": [20],
        }
        correctness = measure._path_and_correctness(state)
        self.assertTrue(correctness)
        self.assertEqual(
            measure.decision_for(
                calibration=True,
                correctness=correctness,
                characterization=False,
                validation=None,
                resources=False,
                budget=True,
            ),
            "candidate_characterized_no_gain",
        )

    def test_complete_validation_path_is_exact(self) -> None:
        state = {
            "characterization": self.phase(20, 100),
            "validation": self.phase(30, 200),
            "cold_pids": [1, 2, 3, 4, 100, 101, 102, 200, 201, 202],
            "warm_pids": [20, 30],
        }
        self.assertTrue(measure._path_and_correctness(state))
        state["validation"]["pairs"][1]["token_identical"] = False
        self.assertFalse(measure._path_and_correctness(state))

    def test_partial_phase_and_resources_fail_safely(self) -> None:
        phase = self.phase(20, 100)
        phase["pairs"].pop()
        state = {
            "characterization": phase,
            "validation": None,
            "cold_pids": [1, 2, 3, 4, 100, 101],
            "warm_pids": [20],
        }
        self.assertFalse(measure._path_and_correctness(state))
        resources = measure._resource_summary(
            {"characterization": {"pairs": [], "warmup": None}}, 10, 10
        )
        self.assertFalse(resources["gate_passed"])

    def test_partial_calibration_samples_survive_an_error(self) -> None:
        class Guard:
            def before_candidate(self):
                return None

            def finish_candidate(self):
                return None

        values = [
            {"pid": 1, "tokens": [7], "ttft_ns": 100},
            {"pid": 2, "tokens": [7], "ttft_ns": 101},
            measure.WorkerError("stopped"),
        ]
        state = {"cold_pids": []}
        with (
            mock.patch.object(measure, "_run_cold", side_effect=values),
            mock.patch.object(measure, "_record_and_pace", side_effect=lambda _g, value: value),
            self.assertRaises(measure.WorkerError),
        ):
            measure._calibration(Guard(), state)
        self.assertEqual(len(state["calibration"]["pairs"]), 1)

    def test_child_environment_is_offline_and_ignores_python_injection(self) -> None:
        poisoned = {name: "unsafe" for name in measure.UNSAFE_PYTHON_ENVIRONMENT}
        with mock.patch.dict(os.environ, {**poisoned, "HF_HUB_OFFLINE": "0"}):
            environment = measure._worker_environment()
        for name in poisoned:
            self.assertNotIn(name, environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")

    def test_early_server_exit_is_not_silently_accepted(self) -> None:
        managed = measure.ManagedWorker.__new__(measure.ManagedWorker)
        managed.server = True
        managed.process = types.SimpleNamespace(poll=lambda: 9)
        managed._stderr = io.BytesIO(b"early exit")
        with self.assertRaises(measure.WorkerError):
            managed.shutdown()

    def test_completion_timing_and_request_count_are_checked(self) -> None:
        complete = {
            "cache_instances": 1,
            "compute_ns": 10,
            "event": "complete",
            "load_count": 1,
            "mlx_peak_bytes": 1,
            "pid": 42,
            "prompt_key": "P",
            "prompt_tokens": measure.EXPECTED_PROMPT_TOKENS,
            "request_count": 1,
            "request_id": "request-1",
            "rss_peak_bytes": 1,
            "tokens": [3] * measure.OUTPUT_TOKENS,
        }
        first = {
            "event": "first_token",
            "first_compute_ns": 11,
            "request_id": "request-1",
            "token_id": 3,
        }
        fake = types.SimpleNamespace(
            process=types.SimpleNamespace(pid=42), read=mock.Mock(side_effect=[first, complete])
        )
        with self.assertRaises(measure.WorkerError):
            measure._validate_answer_events(
                fake, request_id="request-1", prompt_key="P", started_ns=0
            )

    def test_attempt_directory_must_be_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "attempt"
            path.mkdir(mode=0o700)
            os.chmod(path, 0o700)
            measure._require_private_directory(path)
            os.chmod(path, 0o755)
            with self.assertRaises(measure.StudyError):
                measure._require_private_directory(path)


class DashboardTest(unittest.TestCase):
    def result(self) -> dict:
        pair = {
            "order": "AB",
            "prompt_key": "P",
            "ratio": 0.33,
            "token_identical": True,
        }
        return {
            "calibration": {
                "pairs": [
                    {
                        "prompt_key": "P",
                        "ratio": 1.0,
                        "token_identical": True,
                    }
                ]
            },
            "characterization": {"pairs": [pair]},
            "decision": "engineering_gain_confirmed_exact_scope",
            "formal_claim": False,
            "gates": {"H1_correctness_and_path": True},
            "metrics": {"effect_percent": -67.0},
            "resources": {"swap_delta_bytes": 0},
            "run_id": measure.RUN_ID,
            "study_id": measure.STUDY_ID,
            "validation": {"pairs": [{**pair, "order": "BA"}]},
        }

    def test_snapshot_is_read_only_and_projects_pair_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(self.result()), encoding="utf-8")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            value = dashboard.snapshot(path)
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(value["read_only"])
            self.assertEqual(len(value["rows"]), 3)
            self.assertFalse(value["formal_claim"])

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
        self.assertTrue(dashboard._host_header_allowed("127.0.0.1:8781", "127.0.0.1", 8781))
        self.assertTrue(dashboard._host_header_allowed("[::1]:8781", "::1", 8781))
        for value in (None, "evil.example:8781", "localhost:8781"):
            self.assertFalse(dashboard._host_header_allowed(value, "127.0.0.1", 8781))


if __name__ == "__main__":
    unittest.main()
