"""Offline contracts for the sealed Cycle-12 LM-head prefill study."""

from __future__ import annotations

import copy
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.head_skip_formal import study


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "head_skip_formal" / "study.py"


def provenance() -> dict[str, object]:
    code_files = {study.SCRIPT_PATH: "1" * 64}
    spec_files = {study.PREREGISTRATION_PATH: "2" * 64}
    environment = {"python": "test", "packages": {}}
    hardware = {"model": "test"}
    model = {
        "model_id": study.MODEL_ID,
        "model_revision": study.MODEL_REVISION,
        "model_snapshot_weight_files": ["model.safetensors"],
        "model_snapshot_weight_bytes": 1,
        "model_source": "validated_project_local_snapshot",
    }
    value: dict[str, object] = {
        "schema_version": study.SCHEMA_VERSION,
        "study_id": study.STUDY_ID,
        "git_revision": "a" * 40,
        "git_dirty": False,
        "git_diff_sha256": study.EMPTY_SHA256,
        "code_files": code_files,
        "code_sha256": study.canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": study.canonical_sha256(spec_files),
        "environment": environment,
        "environment_sha256": study.canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": study.canonical_sha256(hardware),
        "model": model,
        "model_sha256": study.canonical_sha256(model),
    }
    value["provenance_sha256"] = study.canonical_sha256(value)
    return value


def correctness() -> dict[str, object]:
    tokens = list(range(study.CORRECTNESS_TOKENS))
    return {
        "status": "token_identical",
        "token_ids": tokens,
        "token_sha256": hashlib.sha256(study.canonical_json(tokens).encode()).hexdigest(),
        "token_count": len(tokens),
        "finish_reason": "fixed_horizon",
        "prompt_truncated": False,
        "silent_fallback": False,
        "candidate_path_exercised": True,
    }


def budget() -> dict[str, object]:
    return {
        "passed": True,
        "gpu_work_seconds": 1.0,
        "max_continuous_gpu_seconds": 1.0,
        "cooldown_seconds": 0.0,
        "required_break_seconds": 12.0,
        "wall_seconds": 14.0,
        "gpu_work_limit_seconds": study.GPU_WORK_LIMIT_SECONDS,
        "continuous_gpu_limit_seconds": study.CONTINUOUS_GPU_LIMIT_SECONDS,
        "duty_cycle_limit": study.DUTY_CYCLE_LIMIT,
        "wall_limit_seconds": study.WALL_LIMIT_SECONDS,
        "candidate_cooldown_seconds": study.CANDIDATE_COOLDOWN_SECONDS,
        "required_break_limit_seconds": study.REQUIRED_BREAK_SECONDS,
    }


def resources() -> dict[str, object]:
    return {
        "cpu_process_ns": 1,
        "rss_peak_bytes": 1,
        "mlx_active_memory_bytes": None,
        "mlx_peak_memory_bytes": None,
        "mlx_cache_memory_bytes": None,
    }


def session(
    *,
    stage: str,
    session_id: str,
    preregistration_sha256: str,
    seal_sha256: str | None,
    provenance_sha256: str,
    ratio: float,
) -> dict[str, object]:
    def blocks(warmup: bool) -> list[dict[str, object]]:
        return [
            {
                "block_index": index,
                "order": order,
                "a_ns": 1_000_000 + index * 100,
                "b_ns": round((1_000_000 + index * 100) * ratio),
            }
            for index, order in enumerate(study.orders_for(stage, session_id, warmup=warmup))
        ]

    return study.build_session(
        stage=stage,
        session_id=session_id,
        preregistration_sha256=preregistration_sha256,
        confirmation_seal_sha256=seal_sha256,
        provenance_sha256=provenance_sha256,
        warmups=blocks(True),
        measurements=blocks(False),
        correctness=correctness(),
        budget=budget(),
        resources=resources(),
    )


def complete_history(ratio: float = 0.8) -> list[dict[str, object]]:
    prereg = study.build_preregistration(provenance())
    calibration = [
        session(
            stage=study.CALIBRATION,
            session_id=session_id,
            preregistration_sha256=prereg["preregistration_sha256"],
            seal_sha256=None,
            provenance_sha256=prereg["provenance_sha256"],
            ratio=1.0,
        )
        for session_id in study.SESSION_ORDER
    ]
    summary = study.build_calibration_summary(calibration)
    seal = study.build_confirmation_seal(summary, calibration)
    confirmation = [
        session(
            stage=study.CONFIRMATION,
            session_id=session_id,
            preregistration_sha256=prereg["preregistration_sha256"],
            seal_sha256=seal["confirmation_seal_sha256"],
            provenance_sha256=prereg["provenance_sha256"],
            ratio=ratio,
        )
        for session_id in study.SESSION_ORDER
    ]
    decision = study.build_study_decision(confirmation, seal)
    return [prereg, *calibration, summary, seal, *confirmation, decision]


class HeadSkipFormalTests(unittest.TestCase):
    def test_self_check_and_release_gate_are_offline(self) -> None:
        checked = subprocess.run(
            [sys.executable, str(SCRIPT), "self-check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "study.sqlite3"
            gated = subprocess.run(
                [sys.executable, str(SCRIPT), "--database", str(database), "seal"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(gated.returncode, 78)
            self.assertFalse(database.exists())

    def test_orders_are_balanced_and_frozen(self) -> None:
        for stage_name in study.STAGES:
            for session_id in study.SESSION_ORDER:
                self.assertEqual(
                    sorted(study.orders_for(stage_name, session_id, warmup=False)),
                    ["ab", "ab", "ba", "ba"],
                )
                self.assertEqual(
                    sorted(study.orders_for(stage_name, session_id, warmup=True)),
                    ["ab", "ba"],
                )

    def test_correctness_gate_rejects_any_token_mutation(self) -> None:
        value = correctness()
        value["token_ids"][0] += 1  # type: ignore[index]
        with self.assertRaises(study.ProtocolError):
            study._validated_correctness(value)

    def test_calibration_and_gain_decision_replay(self) -> None:
        values = complete_history(0.8)
        checked = study.validate_history(values)
        self.assertEqual(checked[-1]["status"], "head_skip_gain_confirmed")
        self.assertEqual(sum(value["formal_claim"] is True for value in checked), 1)
        self.assertLess(checked[-1]["intervals"]["all"]["ci_high"], 0.95)

    def test_equivalence_and_regression_are_symmetric_terminal_regions(self) -> None:
        equivalent = complete_history(1.0)[-1]
        regression = complete_history(1.2)[-1]
        self.assertEqual(equivalent["status"], "head_skip_equivalent_within_mde")
        self.assertEqual(regression["status"], "head_skip_regression_confirmed")

    def test_history_rejects_later_record_after_formal_decision(self) -> None:
        values = complete_history()
        with self.assertRaises(study.ProtocolError):
            study.validate_history([*values, copy.deepcopy(values[-1])])

    def test_storage_is_hash_chained_append_only_and_read_only_for_snapshot(self) -> None:
        prereg = study.build_preregistration(provenance())
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "study.sqlite3"
            with study.Storage.open(database, initialize=True) as storage:
                outcome = storage.append(prereg)
                self.assertEqual(outcome["seq"], 1)
                with self.assertRaises(sqlite3.DatabaseError):
                    storage.connection.execute("UPDATE records SET kind='x' WHERE seq=1")
                storage.connection.rollback()
                with self.assertRaises(sqlite3.DatabaseError):
                    storage.connection.execute("DELETE FROM records WHERE seq=1")
                storage.connection.rollback()
            value = study.snapshot(database)
            self.assertTrue(value["read_only"])
            self.assertEqual(value["records"], 1)
            self.assertEqual(value["formal_claim_records"], 0)
            with study.Storage.open(database, read_only=True) as storage:
                with self.assertRaises(study.StorageError):
                    storage.append(prereg)

    def test_preregistration_rejects_mutated_provenance_projection(self) -> None:
        prereg = study.build_preregistration(provenance())
        prereg["provenance"]["model"]["model_revision"] = "b" * 40  # type: ignore[index]
        prereg["preregistration_sha256"] = study.canonical_sha256(
            {key: value for key, value in prereg.items() if key != "preregistration_sha256"}
        )
        with self.assertRaises(study.ProtocolError):
            study.validate_history([prereg])


if __name__ == "__main__":
    unittest.main()
