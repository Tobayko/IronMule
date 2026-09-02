"""Offline contract tests for P2.

The tests never import MLX, load a model or touch the device. They exercise
the decision rule directly and inspect the worker source for the safety
properties that must hold before a gated run is permitted.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORENSICS = ROOT / "experiments" / "identity_forensics"
WORKER = FORENSICS / "measure_logit_gap.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def load_gap_analysis():
    spec = importlib.util.spec_from_file_location("gap_analysis", FORENSICS / "gap_analysis.py")
    module = importlib.util.module_from_spec(spec)
    # @dataclass(slots=True) rebuilds the class and looks its module up in
    # sys.modules, so the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gap_analysis = load_gap_analysis()


def gaps(values):
    return [{"position": index, "gap": value} for index, value in enumerate(values)]


NORMAL = [3.1, 2.8, 4.0, 1.9, 3.3, 2.2, 5.1, 3.0, 2.7, 3.4, 2.9, 3.1, 2.5, 3.8, 2.0, 3.6]


def test_a_collapsed_gap_at_the_divergence_is_a_tie():
    values = list(NORMAL)
    values[10] = 4e-4
    verdict = gap_analysis.classify(gaps(values), 10)
    assert verdict.verdict == "tie" and verdict.supports_tie_hypothesis
    assert verdict.is_minimum is True
    assert verdict.ratio > gap_analysis.TIE_RATIO
    assert verdict.as_dict()["gate_unchanged"] is True


def test_a_wide_gap_at_the_divergence_is_structural():
    verdict = gap_analysis.classify(gaps(NORMAL), 10)
    assert verdict.verdict == "structural"
    assert "divergence_gap_above_absolute_threshold" in verdict.reasons


def test_an_early_divergence_is_structural_whatever_the_gap():
    values = list(NORMAL)
    values[1] = 1e-9
    assert gap_analysis.classify(gaps(values), 1).verdict == "structural"
    assert gap_analysis.classify(gaps(values), 0).verdict == "structural"


def test_a_small_but_not_dominant_gap_stays_inconclusive():
    # Absolute threshold met, ratio not: every position is tiny, so the
    # divergence position is not special and nothing may be concluded.
    values = [5e-3] * 16
    verdict = gap_analysis.classify(gaps(values), 10)
    assert verdict.verdict == "inconclusive"
    assert "divergence_gap_not_small_against_the_median" in verdict.reasons


def test_identical_runs_report_no_divergence():
    verdict = gap_analysis.classify(gaps(NORMAL), None)
    assert verdict.verdict == "no_divergence" and verdict.first_diff is None


def test_malformed_evidence_is_rejected():
    for bad in ([], gaps([1.0, float("nan")]), [{"gap": -1.0}]):
        with pytest.raises(gap_analysis.GapError):
            gap_analysis.classify(bad, 0 if bad else None)
    with pytest.raises(gap_analysis.GapError):
        gap_analysis.classify(gaps(NORMAL), 99)


def test_summary_needs_every_variant_to_agree():
    tie = {"verdict": "tie"}
    structural = {"verdict": "structural"}
    clean = {"verdict": "no_divergence"}
    assert gap_analysis.summarise([tie, tie])["answer"] == "tie_hypothesis_supported"
    assert gap_analysis.summarise([tie, structural])["answer"] == "tie_hypothesis_rejected"
    assert gap_analysis.summarise([clean, clean])["answer"] == "no_divergence_reproduced"
    assert gap_analysis.summarise([tie, {"verdict": "inconclusive"}])["answer"] == "inconclusive"
    assert gap_analysis.summarise([tie])["formal_claim"] is False
    with pytest.raises(gap_analysis.GapError):
        gap_analysis.summarise([{"verdict": "maybe"}])


def test_worker_keeps_every_model_import_behind_the_gate():
    tree = ast.parse(WORKER.read_text())
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module)
    assert not any(name.startswith(("mlx", "_bench")) for name in top_level), top_level


def test_worker_carries_the_standard_run_guards():
    source = WORKER.read_text()
    for guard in ("require_ac_power", "BudgetGuard", "release_gate",
                  "resolve_local_model_snapshot", "guard.record_gpu"):
        assert guard in source, guard
    for forbidden in ("import requests", "import urllib", "import socket", "hf_hub_download"):
        assert forbidden not in source, forbidden


def test_worker_pins_the_prompt_it_is_allowed_to_answer_about():
    source = WORKER.read_text()
    # A different prompt has a different sensitive position, so the run must
    # abort rather than silently answer a question nobody asked.
    assert "EXPECTED_PROMPT_TOKENS = 677" in source
    assert "raise SystemExit(" in source


def test_worker_refuses_to_run_without_explicit_release():
    result = subprocess.run([str(PYTHON), str(WORKER)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 78
    assert json.loads(result.stdout)["state"] == "not_released"


def test_worker_self_check_runs_offline():
    result = subprocess.run([str(PYTHON), str(WORKER), "--self-check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["state"] == "self_check"
    assert report["expected_prompt_tokens"] == 677
    assert report["formal_claim"] is False and report["gate_unchanged"] is True


def test_worker_binds_its_result_to_the_code_that_produced_it():
    source = WORKER.read_text()
    # Without provenance a later edit to the thresholds would make an old
    # result look as if it had used the new ones.
    assert "study_provenance" in source
    assert "gap_analysis.py" in source and "PREREGISTRATION.md" in source


def test_study_provenance_hashes_every_named_file_and_the_tree():
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tools"))
    from _bench import study_provenance

    block = study_provenance(
        [FORENSICS / "gap_analysis.py"],
        preregistration=FORENSICS / "PREREGISTRATION.md",
        extra={"model_snapshot": "rev"},
    )
    assert len(block["git_revision"]) == 40
    assert set(block["code_files_sha256"]) == {
        "experiments/identity_forensics/gap_analysis.py",
        "experiments/identity_forensics/PREREGISTRATION.md",
    }
    assert len(block["code_sha256"]) == 64
    assert block["model_snapshot"] == "rev"
    assert isinstance(block["git_dirty"], bool)


def test_study_provenance_refuses_a_missing_file():
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tools"))
    from _bench import study_provenance
    from friday_evidence.provenance import ProvenanceError

    with pytest.raises(ProvenanceError):
        study_provenance([FORENSICS / "does_not_exist.py"])


def test_offline_is_enforced_before_the_model_library_is_imported():
    """Order matters: the Hugging Face client reads these once, at import.

    Setting them afterwards is a comforting no-op, which is exactly the kind
    of guard that looks present in a source grep and does nothing at runtime.
    """

    source = WORKER.read_text()
    assert "enforce_offline()" in source
    assert source.index("enforce_offline()") < source.index("import mlx.core")
    assert "offline_environment" in source, "the applied environment belongs in provenance"


def test_enforce_offline_actually_sets_what_it_promises(monkeypatch):
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tools"))
    from _bench import OFFLINE_ENVIRONMENT, enforce_offline

    import os

    for name in OFFLINE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:3128")
    applied = enforce_offline()
    assert applied == OFFLINE_ENVIRONMENT
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["NO_PROXY"] == "*"
    # A proxy already in the environment must be cleared, not preserved.
    assert os.environ["HTTPS_PROXY"] == ""
    assert enforce_offline() == applied, "must be idempotent"
