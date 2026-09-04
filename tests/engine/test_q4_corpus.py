"""Read-only migration tests using tiny synthetic artifacts."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from q4_offline_loader import load_offline_modules


ROOT = Path(__file__).resolve().parents[2]


_OFFLINE = load_offline_modules(
    "evidence", "q4_contracts", "q4_corpus", namespace="q4_corpus_test_modules"
)
c = _OFFLINE["q4_corpus"]

_BUILD_SPEC = importlib.util.spec_from_file_location("research.q4_build_corpus", ROOT / "research" / "q4_build_corpus.py")
assert _BUILD_SPEC.loader is not None
build_module = importlib.util.module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(build_module)


def _build_offline_modules():
    """Keep the imported corpus builder on this test's coherent graph."""
    return _OFFLINE["q4_contracts"], _OFFLINE["q4_corpus"]


# The builder's production-only convenience loader historically installs a
# fake ``ironmule`` package.  Patch that test seam so invoking it cannot poison
# the process-wide public module namespace during the full pytest run.
build_module._load_offline_modules = _build_offline_modules


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _b36_payload() -> dict:
    child = {"schema": "ironmule.b36.child.v1", "returncode": 0, "crashed": False,
             "identity_gate": True, "canonical_correctness_gate": True,
             "post_evidence_complete": True, "no_crash": True,
             "token_identity": True, "warmups": [0, 0], "measured": [1, 1, 1, 1, 1],
             "mlx_peak_memory_bytes": 1}
    pair = {
        "hard_gates": {"complete": True, "identity": True, "peak_memory": True, "swap": True, "no_crash": True, "timings": True, "token_identity": True},
        "pair_result": {"status": "ok", "hard_gates": {"complete": True, "identity": True, "peak_memory": True, "swap": True, "no_crash": True, "timings": True, "token_identity": True}},
        "children": [child, dict(child)],
    }
    return {
        "schema": "ironmule.b36.v1", "experiment": "B36", "status": "complete", "model_size": "12B",
        "constants": {"max_tokens": 32, "no_retry": True, "pairs": 16, "repeats": 5, "warmups": 2},
        "pairs": [pair for _ in range(16)],
    }


class CorpusTests(unittest.TestCase):
    def test_roles_quality_gates_and_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            _write(left / "B36_result.json", _b36_payload())
            # Same bytes at a second source is one content artifact with two
            # path references, not two performance observations.
            (right / "B36_copy.json").write_bytes((left / "B36_result.json").read_bytes())
            _write(left / "B35_invalid.json", {"experiment": "B35", "status": "failed", "valid_for_metrics": False, "children": [{"measured": [1]}]})
            _write(left / "Q3d_failure.json", {"experiment": "Q3d", "status": "FAILED", "samples": [1]})
            _write(left / "Q3e_failure.json", {"experiment": "Q3e", "status": "FAILED", "samples": [1]})
            _write(left / "E14b_summary.json", {"experiment": "E14b", "correctness": {"generated_tokens_equal": False, "divergent": [{"batch": 8}]}})
            _write(left / "X1_summary.json", {"experiment": "X1", "cells": {"4b_strict": {"mean": 19.2}}})
            corpus = c.inspect_sources((("left", left), ("right", right)))
            report = corpus.report()
            self.assertEqual(6, report["unique_artifact_count"])
            self.assertGreaterEqual(len(report["duplicate_groups"]), 1)
            by_source = {(item.source_name, item.logical_name): item for item in corpus.artifacts}
            b36 = next(item for item in corpus.artifacts if item.source_name == "B36" and item.quality == "RAW_SAMPLES")
            self.assertTrue(b36.eligible_for_performance)
            b35 = next(item for item in corpus.artifacts if item.source_name == "B35")
            self.assertFalse(b35.eligible_for_performance)
            self.assertIn("excluded", b35.excluded_reason)
            self.assertTrue(any(item.source_name == "Q3d" and item.role == "CENSORED_FAILURE" for item in corpus.artifacts))
            self.assertTrue(any(item.source_name == "Q3e" and item.role == "CENSORED_FAILURE" for item in corpus.artifacts))
            e14 = next(item for item in corpus.artifacts if item.source_name == "E14b")
            self.assertEqual(("batch=8",), e14.invalid_cells)
            self.assertFalse(next(item for item in corpus.artifacts if item.source_name == "X1").eligible_for_performance)
            self.assertEqual({"1B": 0, "4B": 0, "12B": 0}, report["q4_coverage"]["Q4_TRAIN"])
            self.assertEqual(["1B", "4B", "12B"], report["missing_required_model_cells"]["Q4_TRAIN"])

    def test_summary_null_raw_and_symlink_are_not_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write(root / "B36_summary.json", {"experiment": "B36", "samples": None})
            _write(root / "B36_null.json", {"experiment": "B36", "samples": None, "model_size": "4B"})
            target = root / "target.json"
            _write(target, _b36_payload())
            try:
                (root / "B36_symlink.json").symlink_to(target)
            except OSError:
                pass
            previous = c.MAX_ARTIFACT_BYTES
            c.MAX_ARTIFACT_BYTES = 8
            try:
                corpus = c.inspect_sources((("synthetic", root),))
            finally:
                c.MAX_ARTIFACT_BYTES = previous
            self.assertFalse(any(item.logical_name == "B36_symlink.json" for item in corpus.artifacts))
            self.assertTrue(all(not item.eligible_for_performance for item in corpus.artifacts))

    def test_historical_import_has_no_q4_horizon_or_train(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write(root / "B36_result.json", _b36_payload())
            corpus = c.inspect_sources((("synthetic", root),))
            report = corpus.report()
            self.assertEqual(0, report["sequential_horizon"]["transitions"])
            self.assertFalse(report["sequential_horizon"]["eligible"])
            self.assertTrue(all(sum(report["q4_coverage"][split].values()) == 0 for split in report["q4_coverage"]))
            dataset = corpus.to_dataset("0" * 64)
            self.assertEqual(0, len(dataset.transitions))
            self.assertEqual(0, len(dataset.trajectories))
            self.assertTrue(dataset.no_invented_performance)

    def test_import_output_is_canonical_new_private_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write(root / "B36_result.json", _b36_payload())
            output = root / "import.json"
            bundle = build_module.build_import_bundle((root,), None, output)
            mode = stat.S_IMODE(output.stat().st_mode)
            self.assertEqual(0o600, mode)
            self.assertEqual(bundle["dataset"]["dataset_id"], json.loads(output.read_text())["dataset"]["dataset_id"])
            self.assertTrue(build_module.validate_import_bundle(bundle))
            tampered = json.loads(output.read_text())
            tampered["import_report"]["no_27b_q4_cell"] = False
            self.assertFalse(build_module.validate_import_bundle(tampered))
            with self.assertRaises(FileExistsError):
                build_module.build_import_bundle((root,), None, output)

    def test_derived_implementation_report_is_skipped_from_dataset_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write(root / "B36_result.json", _b36_payload())
            report = root / "Q4_implementation_report_20260901.md"
            report.write_text("derived report v1", encoding="utf-8")
            first = c.inspect_sources((("synthetic", root),))
            first_dataset = first.to_dataset("0" * 64).dataset_id
            first_report = first.report()
            report.write_text("derived report v2 with changed prose", encoding="utf-8")
            second = c.inspect_sources((("synthetic", root),))
            self.assertEqual(first_dataset, second.to_dataset("0" * 64).dataset_id)
            self.assertEqual(first_report["artifact_count"], second.report()["artifact_count"])
            self.assertEqual(1, second.report()["skipped_derived_reports"])


if __name__ == "__main__":
    unittest.main()
