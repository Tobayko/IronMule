"""Focused release-blocker regressions for profile/tuning policy (R6/R7)."""

from __future__ import annotations

import json
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ironmule.runtime import BASELINE, Knobs
from ironmule._version import __version__ as CURRENT_VERSION

tune = importlib.import_module("ironmule.tune")


def test_import_does_not_mutate_huggingface_offline_environment():
    """A library import must preserve values chosen by the embedding process."""
    root = Path(__file__).resolve().parents[1]
    code = "import os; before=(os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE')); import ironmule; after=(os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE')); print(before == after)"
    env = {**os.environ, "HF_HUB_OFFLINE": "caller-value", "TRANSFORMERS_OFFLINE": "caller-value-2"}
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"


def _fake_load_engine(monkeypatch):
    seen = {}

    def fake_load(source):
        seen["source"] = source
        seen["environment"] = (os.environ.get("HF_HUB_OFFLINE"),
                                os.environ.get("TRANSFORMERS_OFFLINE"))
        return object(), object()

    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=fake_load))
    monkeypatch.setattr(tune, "Engine", lambda model, tokenizer, knobs: (model, tokenizer, knobs))
    monkeypatch.setenv("HF_HUB_OFFLINE", "caller-value")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "caller-value-2")
    return seen


def test_load_engine_offline_local_path_is_direct_and_preserves_environment(monkeypatch, tmp_path):
    seen = _fake_load_engine(monkeypatch)
    local_model = tmp_path / "model"
    local_model.mkdir()
    before = (os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"])

    tune.load_engine(str(local_model), BASELINE, offline=True)

    assert seen["source"] == str(local_model)
    assert seen["environment"] == before
    assert (os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]) == before


def test_load_engine_offline_hub_id_resolves_cached_snapshot(monkeypatch):
    seen = _fake_load_engine(monkeypatch)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=lambda *args, **kwargs:
                                              calls.append((args, kwargs)) or "/cached/model"))

    tune.load_engine("org/model", BASELINE, offline=True)

    assert len(calls) == 1
    assert calls[0][0] == ("org/model",)
    assert calls[0][1]["local_files_only"] is True
    assert set(calls[0][1]["allow_patterns"]) == set(tune.OFFLINE_ALLOW_PATTERNS)
    assert {"*.safetensors", "*.json", "*.py", "tokenizer.model", "*.tiktoken",
            "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"} <= set(calls[0][1]["allow_patterns"])
    assert seen["source"] == "/cached/model"


@pytest.mark.parametrize("offline", [False, None])
def test_load_engine_online_or_caller_mode_passes_hub_id_unchanged(monkeypatch, offline):
    seen = _fake_load_engine(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=lambda *_args, **_kwargs:
                                              pytest.fail("snapshot resolution is offline-only")))

    tune.load_engine("org/model", BASELINE, offline=offline)

    assert seen["source"] == "org/model"


def test_load_engine_never_mutates_preimported_huggingface_state(monkeypatch, tmp_path):
    seen = _fake_load_engine(monkeypatch)
    constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    local_model = tmp_path / "model"
    local_model.mkdir()

    tune.load_engine(str(local_model), BASELINE, offline=True)

    assert constants.HF_HUB_OFFLINE is False
    assert seen["environment"] == ("caller-value", "caller-value-2")


def test_revalidate_uses_tokenized_prompt_length_from_canary(monkeypatch):
    from ironmule import ab

    profile = {"conditions": {"prompt_tokens": 100, "max_tokens": 32},
               "knobs": BASELINE.as_dict(), "gain": 0.0}
    monkeypatch.setattr(tune, "load_profile", lambda _model, **_kwargs: profile)
    observed = {}

    def fake_stale(_profile, _model, prompt_tokens, max_tokens):
        observed["prompt_tokens"] = prompt_tokens
        observed["max_tokens"] = max_tokens
        return []

    monkeypatch.setattr(tune, "stale", fake_stale)
    monkeypatch.setattr(ab, "run", lambda *_args, **_kwargs: {
        "raw": [{"arms": {"baseline": {"prompt_tokens": 247}}}],
        "ratios": {"stored/baseline": {"total_ns": {"median_ratio": 0.5}}},
        "token_identity": True,
    })

    result = tune.revalidate(prompt="a materially different prompt", max_tokens=17)
    assert observed == {"prompt_tokens": 247, "max_tokens": 17}
    assert result["verdict"] == "still_valid"


def test_corrupt_or_incomplete_profile_is_not_reused(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda: "fp")
    monkeypatch.setattr(tune, "_all_profiles", lambda: {
        "fp/model": {"model_id": "model", "conditions": {}, "knobs": {}}
    })
    assert tune.load_profile("model") is None


def test_complete_freshly_shaped_profile_is_reusable(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    from ironmule import bench, hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "test-chip", "memory_bytes": 1, "gpu_cores": 1,
    })
    monkeypatch.setattr(bench, "environment", lambda: {
        "mlx": "0.32.0", "mlx_lm": "0.31.3", "power_source": "AC", "os": "test",
    })
    conditions = tune.conditions("model", prompt_tokens=2, max_tokens=2)
    assert conditions["runtime_version"] == CURRENT_VERSION
    profile = {
        "model_id": "model",
        "conditions": conditions,
        "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(tune, "_all_profiles", lambda: {"fp/model": profile})
    assert tune.load_profile("model") == profile


@pytest.mark.parametrize("field,value", [("mlx_lm", "0.31.4"),
                                          ("runtime_version", "9.9.9"),
                                          ("os", "different-os")])
def test_profile_identity_drift_is_rejected_but_raw_revalidate_access_remains(monkeypatch, field, value):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    base = {
        "fingerprint": "fp", "model_id": "model", "mlx": "0.32.0",
        "mlx_lm": "0.31.3", "runtime_version": CURRENT_VERSION, "os": "test",
        "prompt_tokens": 2, "max_tokens": 2,
    }
    profile = {"model_id": "model", "conditions": base, "knobs": BASELINE.as_dict()}
    monkeypatch.setattr(tune, "_all_profiles", lambda: {"fp/model": profile})
    monkeypatch.setattr(tune, "conditions", lambda *_args: dict(base, **{field: value}))

    assert tune.load_profile("model") is None
    assert tune.load_profile("model", require_compatible=False) == profile


def test_static_facts_queries_system_profiler_once_per_process(monkeypatch):
    from ironmule import hw

    calls = []

    def fake_run(command, **_kwargs):
        if command[0] == "system_profiler":
            calls.append(command)
            return types.SimpleNamespace(stdout=json.dumps({
                "SPDisplaysDataType": [{"sppci_cores": "10"}],
            }))
        return types.SimpleNamespace(stdout="")

    fake_mlx = types.SimpleNamespace(__version__="0.32.0",
                                     metal=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(hw.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx)
    hw._gpu_cores.cache_clear()
    try:
        first = hw.static_facts()
        second = hw.static_facts()
    finally:
        hw._gpu_cores.cache_clear()

    assert first["gpu_cores"] == second["gpu_cores"] == 10
    assert len(calls) == 1


def test_unsupported_candidate_is_typed_and_search_continues(monkeypatch):
    from ironmule.fast import FusionUnsupported

    class FakeEngine:
        def __init__(self, knobs):
            self.knobs = knobs
            self._compiled = None

        @staticmethod
        def needs_reload(old, new):
            return old.fuse_projections != new.fuse_projections

    monkeypatch.setattr(tune, "Engine", FakeEngine)
    monkeypatch.setattr(tune, "gpu_busy", lambda: None)
    monkeypatch.setattr(tune, "probe", lambda: {"fingerprint": "test"})
    monkeypatch.setattr(tune, "load_engine", lambda _model, knobs: (FakeEngine(knobs), object()))
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(tune, "conditions", lambda *_args: {"prompt_tokens": 2, "max_tokens": 2})
    monkeypatch.setattr(tune, "save_profile", lambda _profile: None)
    monkeypatch.setattr(tune, "SEARCH", [("fuse_projections", [True]), ("readback_every", [2])])

    def fake_measure(engine, _ids, _max_tokens, _eos, **_kwargs):
        if engine.knobs.fuse_projections:
            raise FusionUnsupported("fusion verified for mlx_lm 0.31.3")
        return {"total_ns": 10, "prefill_ns": 5, "decode_ns": 5,
                "logical_tokens": [7], "deterministic": True, "capacity": 2}

    monkeypatch.setattr(tune, "measure", fake_measure)
    profile = tune.tune(repeats=1, confirm_winner=False)
    unsupported = next(t for t in profile["trials"] if t["knob"] == "fuse_projections")
    continued = next(t for t in profile["trials"] if t["knob"] == "readback_every")
    assert unsupported["disposition"] == "unsupported"
    assert unsupported["verdict"] == "unsupported"
    assert continued["disposition"] in {"accepted", "rejected"}


def test_only_typed_or_explicitly_unsupported_candidate_errors_are_skippable():
    from ironmule.fast import FusionUnsupported

    assert tune._is_unsupported_candidate(FusionUnsupported("fusion verified for version"))
    assert tune._is_unsupported_candidate(TypeError("unsupported cache type"))
    assert tune._is_unsupported_candidate(ValueError("unsupported cache state"))
    assert not tune._is_unsupported_candidate(RuntimeError("unsupported wording is incidental"))
    assert not tune._is_unsupported_candidate(TypeError("invalid tensor shape"))


def test_metadata_has_qualified_dependency_intervals_and_dynamic_version():
    import tomllib

    metadata = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    project = metadata["project"]
    assert project["dynamic"] == ["version"]
    assert "mlx>=0.32,<0.33" in project["dependencies"]
    assert "mlx-lm>=0.31.3,<0.32" in project["dependencies"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "ironmule._version.__version__"
