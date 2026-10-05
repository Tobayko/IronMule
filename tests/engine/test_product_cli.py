"""Real CLI/process checks; these do not assert model or GPU performance."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


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


def test_start_downloads_nothing_when_the_user_declines(tmp_path, monkeypatch, capsys):
    import ironmule_product.cli as product_cli

    monkeypatch.setattr(product_cli, "inventory_rows", lambda roots=None: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr(product_cli, "serve", lambda _argv: pytest.fail("must not serve"))
    monkeypatch.setattr(product_cli, "models", lambda *_args: pytest.fail("must not download or register"))
    assert product_cli.start(["--state-dir", str(tmp_path / "state")]) == 1
    assert "nothing downloaded" in capsys.readouterr().err


def test_serve_refuses_a_numeric_plan_on_the_ironmule_engine(tmp_path, capsys):
    import ironmule_product.cli as product_cli

    with pytest.raises(SystemExit):
        product_cli.serve(["--model", "m", "--engine", "ironmule", "--compute-dtype", "native",
                           "--state-dir", str(tmp_path)])
    assert "--compute-dtype runs on the stock engine only" in capsys.readouterr().err
@pytest.mark.parametrize("cpu_only, answer, expected", [
    (False, None, ["--model", "mlx-community/gemma-3-4b-it-4bit"]),
    (True, "y", ["--model", "mlx-community/Qwen3-0.6B-4bit", "--compute-dtype", "dequantize"]),
    (True, "n", ["--model", "mlx-community/Qwen3-0.6B-4bit"]),
])
def test_start_without_a_gpu_picks_a_small_model_and_asks_before_dequantize(
        tmp_path, monkeypatch, cpu_only, answer, expected):
    import ironmule_product.cli as product_cli

    served, asked = [], []
    monkeypatch.setattr(product_cli, "_cpu_only", lambda: cpu_only)
    monkeypatch.setattr(product_cli, "inventory_rows", lambda roots=None: [
        {"model_id": model, "status": "available"} for model in (product_cli.DEFAULT_MODEL, product_cli.CPU_MODEL)])
    monkeypatch.setattr(product_cli, "models", lambda *_args: None)
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or answer)
    monkeypatch.setattr(product_cli, "serve", lambda argv: served.append(argv) or 0)
    assert product_cli.start(["--no-browser", "--state-dir", str(tmp_path / "state")]) == 0
    argv = served[0][:served[0].index("--state-dir")]
    assert [a for a in argv if a not in ("--port", "8080")] == expected
    assert len(asked) == (1 if cpu_only else 0)  # a GPU machine's start is unchanged
    from ironmule_product.state import ProductStore
    timeout = ProductStore(tmp_path / "state").settings()["request_timeout_s"]
    assert timeout == (product_cli.CPU_REQUEST_TIMEOUT_S if cpu_only else 120)
