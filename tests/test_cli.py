"""Focused tests for the dependency-light CLI and package metadata."""

from __future__ import annotations

import os
import subprocess
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ironmule_cli as cli
from ironmule import model_identity


def test_failed_optional_probe_isolated_from_parent_import_state(monkeypatch):
    calls = []

    def failed_probe(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="transformers import failed")

    monkeypatch.setattr(cli.subprocess, "run", failed_probe)
    parent_calls = []
    sentinel = object()
    monkeypatch.setattr(
        cli.importlib, "import_module",
        lambda name: (parent_calls.append(name), sentinel)[1],
    )

    ok, _version, detail = cli._load_optional("mlx_lm", "mlx-lm")
    assert not ok and "isolated probe failed" in detail
    assert calls and calls[0][0][0] == sys.executable
    # A subsequent parent import remains usable; the failed child cannot poison it.
    assert cli.importlib.import_module("safe_parent_module") is sentinel
    assert parent_calls == ["safe_parent_module"]


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


def test_tune_dispatch_preserves_existing_options_including_show(monkeypatch):
    captured = {}

    def fake(args):
        captured["args"] = args
        return 4

    monkeypatch.setattr(cli, "_load_tune", lambda: SimpleNamespace(main=fake))
    assert cli.main(["tune", "--show", "--model", "local-model"]) == 4
    assert captured["args"] == ["--show", "--model", "local-model"]


def test_load_tune_requests_the_submodule_not_reexported_function(monkeypatch):
    module = SimpleNamespace(main=lambda _args: 0, DEFAULT_MODEL="model")
    requested = []
    monkeypatch.setattr(
        cli.importlib, "import_module",
        lambda name: (requested.append(name), module)[1],
    )
    assert cli._load_tune() is module
    assert requested == ["ironmule.tune"]


def test_revalidate_is_a_lazy_alias_with_explicit_arguments(monkeypatch, capsys):
    captured = {}

    def fake(*, model_id, max_tokens):
        captured.update(model_id=model_id, max_tokens=max_tokens)
        return {"verdict": "no_profile"}

    monkeypatch.setattr(
        cli, "_load_tune",
        lambda: SimpleNamespace(DEFAULT_MODEL="default-model", revalidate=fake),
    )
    assert cli.main(["revalidate", "--model", "local-model", "--max-tokens", "17"]) == 0
    assert captured == {"model_id": "local-model", "max_tokens": 17}
    assert '"verdict": "no_profile"' in capsys.readouterr().out


def test_status_reports_existing_profile_state_without_loading_a_model(monkeypatch, capsys):
    calls = []

    def fake_profile(model, *, require_compatible):
        calls.append((model, require_compatible))
        return {"model_id": model} if not require_compatible else None

    monkeypatch.setattr(
        cli, "_load_tune",
        lambda: SimpleNamespace(
            DEFAULT_MODEL="default-model", fingerprint=lambda: "hw-test",
            load_profile=fake_profile, PROFILES=Path("/tmp/profiles.json"),
        ),
    )
    assert cli.main(["status", "--model", "local-model"]) == 0
    output = capsys.readouterr().out
    assert '"profile_status": "stale"' in output
    assert '"hardware_fingerprint": "hw-test"' in output
    assert calls == [("local-model", False), ("local-model", True)]


def test_help_lists_only_implemented_cli_commands(capsys):
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "{doctor|benchmark|serve|models|tune|revalidate|status|info}" in output
    assert all(command in output for command in
               ("doctor", "benchmark", "serve", "models", "tune", "revalidate", "status", "info"))
    assert "\n  cache " not in output


def test_cache_scan_imports_huggingface_hub_only_when_called(monkeypatch):
    """`ironmule --help` must not pay for the cache inspector."""
    import sys as _sys

    monkeypatch.delitem(_sys.modules, "huggingface_hub", raising=False)
    assert "huggingface_hub" not in _sys.modules
    model_identity.scan_local_cache()
    assert "huggingface_hub" in _sys.modules


def test_cache_scan_reads_a_missing_cache_directory_as_empty(monkeypatch):
    """A machine that never downloaded a model has no cache dir; that is not an error."""
    import huggingface_hub
    from huggingface_hub.utils import CacheNotFound

    def absent(*_args, **_kwargs):
        raise CacheNotFound("Cache directory not found", cache_dir=Path("/nope"))

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", absent)
    cache = model_identity.scan_local_cache()
    assert list(cache.repos) == []
    assert list(cache.warnings) == []


def test_models_on_a_machine_without_any_cache(tmp_path):
    """End to end: the first command a new user runs must not raise a traceback."""
    env = {
        **os.environ,
        "HF_HOME": str(tmp_path / "nothing-here"),
        "HF_HUB_CACHE": str(tmp_path / "nothing-here" / "hub"),
    }
    done = subprocess.run(
        [sys.executable, "-m", "ironmule_cli", "models"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "Traceback" not in done.stderr
    assert json.loads(done.stdout) == {"models": [], "warnings": []}


def test_models_filters_models_and_sorts_revisions_without_download(monkeypatch, capsys):
    scanned = []
    revisions = [
        SimpleNamespace(commit_hash="b", snapshot_path="/cache/b", size_on_disk=2, last_modified=2.0),
        SimpleNamespace(commit_hash="a", snapshot_path="/cache/a", size_on_disk=1, last_modified=1.0),
    ]
    repos = [
        SimpleNamespace(repo_id="z/model", repo_type="model", revisions=revisions,
                         size_on_disk=3, last_modified=3.0),
        SimpleNamespace(repo_id="ignored/data", repo_type="dataset", revisions=[],
                         size_on_disk=9, last_modified=9.0),
        SimpleNamespace(repo_id="a/model", repo_type="model", revisions=[],
                         size_on_disk=0, last_modified=None),
    ]

    class Warning:
        def __str__(self):
            return "cache warning"

    def scan_cache_dir():
        scanned.append(True)
        return SimpleNamespace(repos=repos, warnings=[Warning()])

    monkeypatch.setattr(model_identity, "scan_local_cache", scan_cache_dir)
    assert cli.main(["models", "--model", "z/model"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert scanned == [True]
    assert [entry["repo_id"] for entry in payload["models"]] == ["z/model"]
    assert [entry["commit_hash"] for entry in payload["models"][0]["revisions"]] == ["a", "b"]
    assert payload["warnings"] == ["cache warning"]


def test_models_empty_cache_is_deterministic(monkeypatch, capsys):
    monkeypatch.setattr(
        model_identity, "scan_local_cache",
        lambda: SimpleNamespace(repos=[], warnings=[]),
    )
    assert cli.main(["models"]) == 0
    assert json.loads(capsys.readouterr().out) == {"models": [], "warnings": []}


def test_models_reports_missing_huggingface_dependency(monkeypatch, capsys):
    error = ModuleNotFoundError("No module named 'huggingface_hub'")
    error.name = "huggingface_hub"
    monkeypatch.setattr(model_identity, "scan_local_cache", lambda: (_ for _ in ()).throw(error))
    assert cli._run_models([]) == 1
    assert "huggingface_hub" in capsys.readouterr().err


def test_models_does_not_swallow_unrelated_import_error(monkeypatch):
    error = ImportError("broken cache inspector")
    monkeypatch.setattr(model_identity, "scan_local_cache", lambda: (_ for _ in ()).throw(error))
    with pytest.raises(ImportError, match="broken cache inspector"):
        cli._run_models([])


def test_tune_reports_missing_runtime_dependencies(monkeypatch, capsys):
    error = ModuleNotFoundError("No module named 'mlx_lm'")
    error.name = "mlx_lm"
    monkeypatch.setattr(cli, "_load_tune", lambda: (_ for _ in ()).throw(error))
    assert cli._run_tune(["--show"]) == 1
    assert "ironmule doctor" in capsys.readouterr().err


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


def test_help_and_doctor_survive_a_broken_mlx_install(tmp_path):
    """`doctor` diagnoses a broken MLX install, so it must not import MLX to start."""
    script = (
        "import sys\n"
        "for name in ('mlx', 'mlx.core', 'mlx_lm', 'ironmule'):\n"
        "    sys.modules[name] = None\n"
        "import ironmule_cli\n"
        "assert ironmule_cli.main(['--help']) == 0\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_benchmark_reports_an_uncached_model_without_a_traceback(tmp_path):
    env = {
        **os.environ,
        "HF_HOME": str(tmp_path / "nothing-here"),
        "HF_HUB_CACHE": str(tmp_path / "nothing-here" / "hub"),
    }
    done = subprocess.run(
        [sys.executable, "-m", "ironmule_cli", "benchmark",
         "--model", "mlx-community/gemma-3-4b-it-4bit"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env, capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert "hf download mlx-community/gemma-3-4b-it-4bit" in done.stderr


def test_models_reports_a_broken_runtime_install_without_a_traceback(tmp_path):
    """`models` reads the cache through `ironmule`, so a broken MLX must not traceback."""
    script = (
        "import sys\n"
        "sys.modules['mlx'] = None\n"
        "sys.modules['mlx.core'] = None\n"
        "import ironmule_cli\n"
        "sys.exit(ironmule_cli.main(['models']))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert "ironmule doctor" in done.stderr


def _cache(*revisions, repo_id="org/model"):
    from types import SimpleNamespace as NS

    revs = [NS(commit_hash=r, snapshot_path=f"/cache/{r}") for r in revisions]
    return NS(repos=[NS(repo_id=repo_id, repo_type="model", revisions=revs)], warnings=())


def test_nothing_cached_says_so_and_repeats_the_revision():
    """Without the revision the user fetches `main`, which still will not resolve."""
    from types import SimpleNamespace as NS

    from ironmule.model_identity import ModelIdentityError, select_cached_snapshot

    empty = NS(repos=(), warnings=())
    with pytest.raises(ModelIdentityError) as caught:
        select_cached_snapshot(empty, "org/model", revision="abc123")
    message = str(caught.value)
    assert "model is not cached" in message
    assert "hf download org/model --revision abc123" in message


def test_wrong_revision_does_not_claim_the_model_is_missing():
    """It is cached, just not at that pin. Saying otherwise blames the wrong thing."""
    from ironmule.model_identity import ModelIdentityError, select_cached_snapshot

    with pytest.raises(ModelIdentityError) as caught:
        select_cached_snapshot(_cache("aaa111"), "org/model", revision="bbb222")
    message = str(caught.value)
    assert "model is not cached" not in message
    assert "is cached, but not at revision 'bbb222'" in message
    assert "ironmule models --model org/model" in message


def test_ambiguous_revision_does_not_advise_a_flag_the_cli_lacks():
    """`ironmule_cli` has no --revision; pinning is a Python API argument."""
    from ironmule.model_identity import ModelIdentityError, select_cached_snapshot

    with pytest.raises(ModelIdentityError) as caught:
        select_cached_snapshot(_cache("aaa111", "bbb222"), "org/model")
    message = str(caught.value)
    assert "Runtime.load" in message
    assert "--revision" not in message.split("Runtime.load")[0]
