"""Offline contract tests for the F1 warm-arm worker.

No MLX import, no IronMule import, no model, no device. The worker is
inspected for its guards and exercised through its gate.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "experiments" / "f1_integration" / "measure_f1.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(PYTHON), str(WORKER), *arguments],
                          capture_output=True, text=True, cwd=ROOT)


def test_it_refuses_to_run_without_explicit_release():
    result = run()
    assert result.returncode == 78
    assert json.loads(result.stdout)["state"] == "not_released"


def test_the_self_check_runs_offline_and_names_the_candidate_knobs():
    result = run("--self-check")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["prompt_tokens"] == 897 and report["output_tokens"] == 32
    assert report["candidate_knobs"] == ["compiled_fixed_cache", "head_skip_prefill"]
    assert report["modes"] == ["aa", "ab"]
    assert report["formal_claim"] is False


def test_every_heavy_import_stays_behind_the_gate():
    tree = ast.parse(WORKER.read_text())
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    assert not any(name.startswith(("mlx", "ironmule", "_bench")) for name in names), names


def test_it_carries_the_standard_run_guards():
    source = WORKER.read_text()
    for guard in ("require_ac_power", "BudgetGuard", "release_gate",
                  "resolve_local_model_snapshot", "guard.record_gpu",
                  "study_provenance", "enforce_offline", "check_prompt_length"):
        assert guard in source, guard
    for forbidden in ("import requests", "import urllib", "hf_hub_download"):
        assert forbidden not in source, forbidden
    assert source.index("enforce_offline()") < source.index("import mlx.core")


def test_it_binds_the_ironmule_checkout_it_was_written_against():
    source = WORKER.read_text()
    assert "03e884cb28a05d090d20844460fc3afc8e738a91" in source
    assert "expected" in source and "_ironmule_head" in source


def test_it_reuses_the_sealed_workload_verbatim():
    """A different prompt is not the workload the gains were confirmed on."""

    worker = WORKER.read_text()
    sealed = (ROOT / "experiments" / "persistent_process" / "worker.py").read_text()
    for fragment in ("You are a careful engineering assistant working in a Python repository. ",
                     "Follow the existing style and explain your reasoning briefly. ",
                     '"P": "Why is false sharing slow?"',
                     '"Q": "What are TLB misses?"',
                     '"R": "When does store forwarding fail?"',
                     '"S": "Why can branch prediction fail?"'):
        assert fragment in worker, fragment
        assert fragment in sealed, f"{fragment} is no longer the sealed workload"
    assert ") * 40" in worker and ") * 40" in sealed
    # The separator is worth its own assertion: leaving it out yields 895
    # tokens instead of 897, which the first run of this worker did.
    joiner = 'FILLER + "\\n\\n" + QUESTIONS[key]'
    assert joiner in worker, "the sealed prompt joins filler and question with a blank line"
    assert joiner in sealed


def test_token_identity_is_terminal_not_a_warning():
    source = WORKER.read_text()
    assert 'token_sha256"] != right["token_sha256"]' in source
    assert "raise SystemExit(" in source
    assert "identity_break" in source, "the breaking pair must be written out"


def test_an_aa_run_puts_the_baseline_on_both_arms():
    source = WORKER.read_text()
    assert 'other_knobs = BASELINE if args.mode == "aa" else candidate_knobs(Knobs)' in source


def test_the_result_carries_the_wire_shape_integrate_reads():
    source = WORKER.read_text()
    for field in ('"baseline_samples"', '"candidate_samples"', '"payload"',
                  '"friday.ironmule.result.v1"'):
        assert field in source, field


def test_pair_order_alternates_so_neither_arm_runs_only_on_a_warm_cache():
    source = WORKER.read_text()
    assert 'order = "AB" if index % 2 == 0 else "BA"' in source
    assert 'if order == "AB":' in source


def test_the_pair_count_is_bounded():
    assert run("--execute", "--pairs", "0").returncode != 0
    assert run("--execute", "--pairs", "999").returncode != 0
