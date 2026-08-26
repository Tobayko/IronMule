"""Focused tests for the dependency-light CLI and package metadata."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import ironmule_cli as cli


def test_doctor_reports_missing_prerequisites_without_runtime_import(monkeypatch, capsys):
    monkeypatch.setattr(cli.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    assert cli.main(["doctor"]) == 1
    output = capsys.readouterr().out
    assert "Apple Silicon architecture" in output
    assert "MLX" in output and "MLX-LM" in output


def test_benchmark_dispatch_preserves_arguments(monkeypatch):
    captured = {}

    def fake(args):
        captured["args"] = args
        return 7

    monkeypatch.setattr(cli, "_run_benchmark", fake)
    assert cli.main(["benchmark", "--requests", "2", "--max-tokens", "3"]) == 7
    assert captured["args"] == ["--requests", "2", "--max-tokens", "3"]


@pytest.mark.parametrize("failure_during", [False, True])
def test_benchmark_reports_missing_mlx_lm(monkeypatch, capsys, failure_during):
    error = ModuleNotFoundError("No module named 'mlx_lm'")
    error.name = "mlx_lm"
    if failure_during:
        def fake_benchmark(_args):
            raise error

        monkeypatch.setattr(cli, "_load_benchmark", lambda: fake_benchmark)
    else:
        monkeypatch.setattr(cli, "_load_benchmark", lambda: (_ for _ in ()).throw(error))

    assert cli._run_benchmark([]) == 1
    assert "ironmule doctor" in capsys.readouterr().err


def test_benchmark_does_not_swallow_unrelated_import_error(monkeypatch):
    error = ImportError("broken benchmark implementation")
    monkeypatch.setattr(cli, "_load_benchmark", lambda: (_ for _ in ()).throw(error))
    with pytest.raises(ImportError, match="broken benchmark"):
        cli._run_benchmark([])


def test_info_is_available_from_source_checkout(capsys):
    assert cli.main(["info"]) == 0
    output = capsys.readouterr().out
    assert "IronMule" in output
    assert "Measured, not assumed." in output


def test_cli_can_be_imported_without_mlx():
    result = subprocess.run(
        [sys.executable, "-c", "import ironmule_cli; print('cli ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pyproject_metadata_and_entry_point():
    tomllib = pytest.importorskip("tomllib")
    metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert metadata["requires-python"] == ">=3.10"
    assert metadata["urls"]["Documentation"].endswith("/docs/RUNTIME.md")
    assert {"mlx", "local-llm", "ttft"}.issubset(metadata["keywords"])
    assert metadata["scripts"]["ironmule"] == "ironmule_cli:main"
    assert not any("License ::" in classifier for classifier in metadata["classifiers"])
