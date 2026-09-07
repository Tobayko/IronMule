"""Pure contract checks for the installed 4B reference regression harness."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import product_reference_regression as regression  # noqa: E402
from product_load_screen import LoadScreenFailure  # noqa: E402


def test_frozen_reference_has_four_complete_exact_4b_samples():
    samples, digest = regression._reference_samples()  # noqa: SLF001
    assert len(samples) == regression.REPEATS + 1
    assert len(digest) == 64
    assert all(len(item) == 64 for item in samples)


def test_frozen_12b_reference_is_exact_and_rejects_other_revision():
    model = "mlx-community/gemma-3-12b-it-4bit"
    samples, digest = regression._reference_samples(model, regression.FROZEN_MODELS[model])
    assert len(samples) == 4 and all(len(item) == 64 for item in samples)
    assert len(digest) == 64
    with pytest.raises(regression.RegressionFailure, match="reference_model_not_exact"):
        regression._reference_samples(model, "other-revision")


def test_reference_rejects_changed_source_or_nonterminal_model(tmp_path: Path, monkeypatch):
    value = json.loads(regression.REFERENCE.read_text(encoding="utf-8"))
    value["source_sha256_after"] = "0" * 64
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(regression, "REFERENCE", path)
    with pytest.raises(regression.RegressionFailure, match="reference_invalid"):
        regression._reference_samples()  # noqa: SLF001


def test_reference_rejects_nonterminal_exact_model_row(tmp_path: Path, monkeypatch):
    value = json.loads(regression.REFERENCE.read_text(encoding="utf-8"))
    row = next(item for item in value["models"] if item["model_id"] == regression.EXPECTED_MODEL)
    row["status"] = "failed"
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(regression, "REFERENCE", path)
    with pytest.raises(regression.RegressionFailure, match="reference_samples_invalid"):
        regression._reference_samples()  # noqa: SLF001


def test_installed_import_proof_rejects_source_tree_without_model_work():
    with pytest.raises(LoadScreenFailure, match="source_tree_package_in_use"):
        regression._installed_import_proof()  # noqa: SLF001
