"""Real CLI/process checks; these do not assert model or GPU performance."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_inventory_empty_explicit_root(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "ironmule_cli", "models", "list", "--json",
         "--family", "gemma", "--cache-root", str(tmp_path / "empty")],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["models"] == []
    assert report["models_loaded"] is False
    assert report["hardware_qualified"] is False


def test_inventory_never_imports_inference_dependencies(tmp_path):
    # An import prohibition checks dependency isolation. No inference backend is
    # substituted, and no output here is evidence about GPU/model behaviour.
    script = """
import sys
class DenyInferenceImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'numpy', 'huggingface_hub', 'ironmule'}:
            raise AssertionError('inventory imported an inference dependency: ' + fullname)
sys.meta_path.insert(0, DenyInferenceImports())
import ironmule_cli
raise SystemExit(ironmule_cli.main(['models', 'list', '--json', '--cache-root', sys.argv[1]]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["models"] == []


def test_inventory_help_explains_no_model_load():
    result = subprocess.run(
        [sys.executable, "-m", "ironmule_cli", "models", "list", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "without loading models" in result.stdout
    assert "--cache-root" in result.stdout


def test_doctor_help_exposes_json_without_probing():
    result = subprocess.run(
        [sys.executable, "-m", "ironmule_cli", "doctor", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--json" in result.stdout
