"""Fail-closed authorization tests for the N10-derived runtime policy."""

from __future__ import annotations

import unittest
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from friday_runtime_n10.constants import (
    N10_CODE_SHA256,
    DEFAULT_N10_DATABASE_PATH,
    N10_DATABASE_SHA256,
    N10_DECISION_RECORD_ID,
    N10_DECISION_SHA256,
    N10_ENVIRONMENT_SHA256,
    N10_HARDWARE_SHA256,
    N10_PREREGISTRATION_SHA256,
    N10_PROVENANCE_SHA256,
    N10_SPEC_SHA256,
    N10_SNAPSHOT_REVISION,
    N10_STUDY_ID,
)
from friday_runtime_n10.policy import REGISTERED_WORKLOAD, decision_for, load_policy


def identity() -> dict[str, object]:
    return {
        "git_dirty": False,
        "code_sha256": N10_CODE_SHA256,
        "spec_sha256": N10_SPEC_SHA256,
        "environment_sha256": N10_ENVIRONMENT_SHA256,
        "hardware_sha256": N10_HARDWARE_SHA256,
    }


def rows() -> list[dict[str, object]]:
    provenance = identity()
    provenance["provenance_sha256"] = N10_PROVENANCE_SHA256
    preregistration = {
        "record_kind": "preregistration",
        "provenance_sha256": N10_PROVENANCE_SHA256,
        "payload": {
            "preregistration_sha256": N10_PREREGISTRATION_SHA256,
            "provenance_sha256": N10_PROVENANCE_SHA256,
            "study_specification": {
                "workload": {
                    "operation": "matmul",
                    "dtype": "float16",
                    "lhs_shape": [2048, 2048],
                    "rhs_shape": [2048, 2048],
                    "output_shape": [2048, 2048],
                    "rhs_count": 10,
                    "baseline_plan": "serial_per_op_eval_and_sync",
                    "candidate_plan": "enqueue_all_then_single_eval_and_sync",
                }
            },
        },
        "provenance": provenance,
    }
    decision = {
        "record_kind": "study_decision",
        "record_id": N10_DECISION_RECORD_ID,
        "formal_claim": True,
        "provenance": provenance,
        "payload": {
            "study_id": N10_STUDY_ID,
            "decision_sha256": N10_DECISION_SHA256,
            "preregistration_sha256": N10_PREREGISTRATION_SHA256,
            "provenance_sha256": N10_PROVENANCE_SHA256,
            "status": "n10_gain_confirmed",
            "action": "permit_bounded_n10_runtime_prototype",
            "claim": "n10_batched_dispatch_is_faster_beyond_mde",
            "claim_scope": "one-device-one-workload-one-execution-plan",
            "gates": {
                "all_sessions_byte_identical": True,
                "gain_all_splits": True,
                "equivalence_all_splits": False,
                "regression_all_splits": False,
            },
        },
    }
    result = [preregistration]
    result.extend({"record_kind": "calibration_session"} for _ in range(6))
    result.append({"record_kind": "calibration_summary"})
    result.append({"record_kind": "confirmation_seal"})
    result.extend({"record_kind": "confirmation_session"} for _ in range(6))
    result.append(decision)
    return result


class RuntimePolicyTest(unittest.TestCase):
    def load(self, evidence_rows=None, current=None):
        selected = rows() if evidence_rows is None else evidence_rows
        current_identity = identity() if current is None else current
        return load_policy(
            "ignored.sqlite3",
            evidence_reader=lambda _path: selected,
            identity_provider=lambda: current_identity,
        )

    def test_exact_terminal_history_authorizes_only_registered_workload(self) -> None:
        evidence = self.load()
        self.assertTrue(evidence.authorized)
        self.assertEqual(evidence.evidence_records, 16)
        self.assertEqual(decision_for(evidence, REGISTERED_WORKLOAD).strategy, "batched")
        changed = replace(REGISTERED_WORKLOAD, rhs_count=9)
        fallback = decision_for(evidence, changed)
        self.assertEqual(fallback.strategy, "serial")
        self.assertEqual(fallback.reason, "workload_out_of_scope")

    def test_identity_mismatches_and_dirty_tree_fail_closed(self) -> None:
        cases = {
            "code_sha256": "n10_code_mismatch",
            "spec_sha256": "n10_spec_mismatch",
            "environment_sha256": "environment_mismatch",
            "hardware_sha256": "hardware_mismatch",
        }
        for field, reason in cases.items():
            with self.subTest(field=field):
                current = identity()
                current[field] = "f" * 64
                evidence = self.load(current=current)
                self.assertFalse(evidence.authorized)
                self.assertEqual(evidence.reason, reason)
        current = identity()
        current["git_dirty"] = True
        self.assertEqual(self.load(current=current).reason, "worktree_dirty")

    def test_altered_decision_or_history_shape_fails_closed(self) -> None:
        changed = rows()
        changed[-1]["payload"]["action"] = "something_else"
        self.assertEqual(self.load(changed).reason, "evidence_scope_mismatch")
        self.assertFalse(self.load(rows()[:-1]).authorized)

    def test_reader_failure_never_authorizes_batching(self) -> None:
        for failure in (OSError("not available"), RuntimeError("unexpected verifier failure")):
            with self.subTest(failure=type(failure).__name__):

                def fail(_path, selected=failure):
                    raise selected

                evidence = load_policy(
                    "missing.sqlite3", evidence_reader=fail, identity_provider=identity
                )
                self.assertFalse(evidence.authorized)
                self.assertEqual(evidence.reason, "evidence_unavailable_or_invalid")

    def test_real_terminal_store_replays_with_injected_clean_identity(self) -> None:
        evidence = load_policy(identity_provider=identity)
        self.assertTrue(evidence.authorized)
        self.assertEqual(evidence.decision_record_id, N10_DECISION_RECORD_ID)
        self.assertEqual(evidence.formal_database_sha256, N10_DATABASE_SHA256)
        self.assertEqual(evidence.formal_snapshot_revision, N10_SNAPSHOT_REVISION)

    def test_byte_different_copy_cannot_substitute_for_terminal_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "n10.sqlite3"
            shutil.copyfile(DEFAULT_N10_DATABASE_PATH, path)
            os.chmod(path, 0o600)
            with path.open("ab") as stream:
                stream.write(b"foreign")
            evidence = load_policy(path, identity_provider=identity)
        self.assertFalse(evidence.authorized)
        self.assertEqual(evidence.reason, "evidence_scope_mismatch")


if __name__ == "__main__":
    unittest.main()
