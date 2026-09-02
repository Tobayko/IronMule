"""Offline contract tests for W1.

No MLX import, no model, no device. The decision rule is exercised directly
and the worker is inspected for its guards.
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
W1 = ROOT / "experiments" / "w1_regime"
WORKER = W1 / "measure_long_answer.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, W1 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regime = load("regime_analysis")
BASE = {"ttft": 1.7851, "control_tps": 70.99, "tokens": 256}


def test_a_constant_rate_keeps_the_short_answer_ordering():
    verdict = regime.classify(**BASE, long_tps=70.99)
    assert verdict.verdict == "rate_stable" and verdict.model_holds
    assert verdict.leader == "head_skip_prefill"
    assert verdict.head_skip_gain > verdict.fixed_compiled_gain


def test_a_degrading_rate_flips_the_leading_candidate():
    # A ten percent drop is enough: at 256 tokens the two candidates sit
    # within half a point of each other, which is why this must be measured.
    verdict = regime.classify(**BASE, long_tps=63.0)
    assert verdict.verdict == "rate_degrades" and not verdict.model_holds
    assert verdict.leader == "fixed_compiled_cache"
    assert verdict.fixed_compiled_gain > verdict.head_skip_gain


def test_the_tolerance_band_is_symmetric_and_preregistered():
    assert regime.RATE_TOLERANCE == 0.10
    assert regime.classify(**BASE, long_tps=70.99 * 0.91).verdict == "rate_stable"
    assert regime.classify(**BASE, long_tps=70.99 * 0.89).verdict == "rate_degrades"
    assert regime.classify(**BASE, long_tps=70.99 * 1.11).verdict == "rate_improves"


def test_the_combined_gain_falls_below_the_f1_threshold_at_this_length():
    # F1's warm arm is preregistered at 10 %. At 256 tokens the same two
    # confirmed gains compose to less than that, which bounds F1's scope.
    verdict = regime.classify(**BASE, long_tps=70.99)
    assert verdict.combined_gain < 0.10
    assert verdict.combined_gain == pytest.approx(0.098, abs=5e-3)


def test_short_answers_stay_far_above_that_threshold():
    verdict = regime.classify(ttft=1.7851, control_tps=70.99, long_tps=70.99, tokens=32)
    assert verdict.combined_gain > 0.13


def test_request_ratio_is_identity_without_a_change():
    assert regime.request_ratio(ttft=1.8, tokens=32, decode_tps=70.0) == pytest.approx(1.0)


def test_malformed_input_is_rejected():
    for kwargs in ({"long_tps": 0.0}, {"long_tps": float("inf")}, {"long_tps": -1.0}):
        with pytest.raises(regime.RegimeError):
            regime.classify(**BASE, **kwargs)
    with pytest.raises(regime.RegimeError):
        regime.request_ratio(ttft=1.8, tokens=0, decode_tps=70.0)


def test_worker_keeps_every_model_import_behind_the_gate():
    tree = ast.parse(WORKER.read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    assert not any(name.startswith(("mlx", "_bench")) for name in names), names


def test_worker_carries_the_standard_run_guards():
    source = WORKER.read_text()
    for guard in ("require_ac_power", "BudgetGuard", "release_gate",
                  "resolve_local_model_snapshot", "guard.record_gpu"):
        assert guard in source, guard
    for forbidden in ("import requests", "import urllib", "import socket", "hf_hub_download"):
        assert forbidden not in source, forbidden


def test_worker_refuses_to_run_without_explicit_release():
    result = subprocess.run([str(PYTHON), str(WORKER)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 78
    assert json.loads(result.stdout)["state"] == "not_released"


def test_worker_self_check_runs_offline():
    result = subprocess.run([str(PYTHON), str(WORKER), "--self-check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["control_tokens"] == 32 and report["long_tokens"] == 256
    assert report["formal_claim"] is False


def test_worker_binds_its_result_to_the_code_that_produced_it():
    source = WORKER.read_text()
    assert "study_provenance" in source
    assert "regime_analysis.py" in source and "PREREGISTRATION.md" in source


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
