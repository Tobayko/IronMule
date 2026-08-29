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
from ironmule.model_identity import (
    ModelIdentity, ModelIdentityError, canonical_json, canonical_sha256,
)

tune = importlib.import_module("ironmule.tune")


def _identity(model_id="model", revision="revision"):
    quantisation = {"bits": 4, "group_size": 64}
    return ModelIdentity(
        model_id=model_id,
        revision=revision,
        model_manifest_sha256="a" * 64,
        architecture="test-architecture",
        quantisation_json=canonical_json(quantisation),
        quantisation_sha256=canonical_sha256(quantisation),
        tokenizer_sha256="b" * 64,
        manifest_file_count=3,
        manifest_bytes=100,
        tokenizer_file_count=1,
    )


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

    class FakeLoadedEngine:
        def __init__(self, model, tokenizer, knobs):
            self.model = model
            self.tokenizer = tokenizer
            self.knobs = knobs
            self.model_identity = None

    def fake_load(source):
        seen["source"] = source
        seen["environment"] = (os.environ.get("HF_HUB_OFFLINE"),
                                os.environ.get("TRANSFORMERS_OFFLINE"))
        return object(), object()

    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=fake_load))
    monkeypatch.setattr(tune, "Engine", FakeLoadedEngine)
    identity = _identity("org/model")

    def fake_resolve(model_id, revision=None):
        seen.setdefault("resolve_calls", []).append((model_id, revision))
        source = Path(model_id) if Path(model_id).is_dir() else Path("/cached/model")
        selected = _identity("local:model" if Path(model_id).is_dir() else "org/model",
                             revision or "revision")
        return types.SimpleNamespace(path=source, identity=selected)

    monkeypatch.setattr(tune, "resolve_local_model", fake_resolve)
    monkeypatch.setattr(tune, "verify_resolved_model", lambda _model, _resolved: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "caller-value")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "caller-value-2")
    seen["identity"] = identity
    return seen


def test_load_engine_offline_local_path_is_direct_and_preserves_environment(monkeypatch, tmp_path):
    seen = _fake_load_engine(monkeypatch)
    local_model = tmp_path / "model"
    local_model.mkdir()
    before = (os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"])

    engine, _ = tune.load_engine(str(local_model), BASELINE, offline=True)

    assert seen["source"] == str(local_model)
    assert seen["environment"] == before
    assert engine.model_identity.model_id == "local:model"
    assert (os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]) == before


def test_load_engine_offline_hub_id_resolves_cached_snapshot(monkeypatch):
    seen = _fake_load_engine(monkeypatch)

    engine, _ = tune.load_engine("org/model", BASELINE, offline=True, revision="revision")

    assert seen["resolve_calls"] == [("org/model", "revision")]
    assert seen["source"] == "/cached/model"
    assert engine.model_identity.model_id == "org/model"


@pytest.mark.parametrize("offline", [False, None])
def test_load_engine_online_or_caller_mode_passes_hub_id_unchanged(monkeypatch, offline):
    seen = _fake_load_engine(monkeypatch)

    engine, _ = tune.load_engine("org/model", BASELINE, offline=offline)

    assert seen["source"] == "org/model"
    assert seen.get("resolve_calls") is None
    assert engine.model_identity is None


def test_load_engine_never_mutates_preimported_huggingface_state(monkeypatch, tmp_path):
    seen = _fake_load_engine(monkeypatch)
    constants = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    local_model = tmp_path / "model"
    local_model.mkdir()

    tune.load_engine(str(local_model), BASELINE, offline=True)

    assert constants.HF_HUB_OFFLINE is False
    assert seen["environment"] == ("caller-value", "caller-value-2")


def test_load_engine_rejects_injected_model_or_revision_identity(monkeypatch):
    _fake_load_engine(monkeypatch)
    wrong_model = types.SimpleNamespace(
        path=Path("/cached/model"), identity=_identity("other/model")
    )
    with pytest.raises(ModelIdentityError, match="different model_id"):
        tune.load_engine("org/model", BASELINE, resolved_source=wrong_model)
    wrong_revision = types.SimpleNamespace(
        path=Path("/cached/model"), identity=_identity("org/model", "other")
    )
    with pytest.raises(ModelIdentityError, match="different revision"):
        tune.load_engine(
            "org/model", BASELINE, revision="wanted", resolved_source=wrong_revision
        )


def test_post_load_identity_check_detects_source_change(monkeypatch):
    exact = _identity("org/model", "revision")
    resolved = types.SimpleNamespace(path=Path("/cached/model"), identity=exact)
    monkeypatch.setattr(
        tune, "build_model_identity", lambda *_args, **_kwargs: _identity("org/model", "changed")
    )
    with pytest.raises(ModelIdentityError, match="changed during load"):
        tune.verify_resolved_model("org/model", resolved)


def test_revalidate_uses_tokenized_prompt_length_from_canary(monkeypatch):
    from ironmule import ab

    profile = {"conditions": {"prompt_tokens": 100, "max_tokens": 32},
               "knobs": BASELINE.as_dict(), "gain": 0.0}
    identity = _identity()
    monkeypatch.setattr(
        tune, "resolve_local_model",
        lambda _model, _revision=None: types.SimpleNamespace(
            path=Path("/cached/model"), identity=identity
        ),
    )
    monkeypatch.setattr(tune, "load_profile", lambda _model, **_kwargs: profile)
    observed = {}

    def fake_stale(_profile, _model, prompt_tokens, max_tokens, **_kwargs):
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
    identity = _identity()
    monkeypatch.setattr(tune, "_all_profiles", lambda: {
        f"fp/{identity.identity_sha256}": {
            "model_id": "model", "conditions": {}, "knobs": {},
            "model_identity": identity.to_dict(),
        }
    })
    assert tune.load_profile("model", model_identity=identity) is None
    assert tune.load_profile(
        "model", require_compatible=False, model_identity=identity
    ) is None


def test_complete_freshly_shaped_profile_is_reusable(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    from ironmule import bench, hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "test-chip", "memory_bytes": 1, "gpu_cores": 1,
    })
    monkeypatch.setattr(bench, "environment", lambda: {
        "mlx": "0.32.0", "mlx_lm": "0.31.3", "power_source": "AC", "os": "test",
    })
    identity = _identity()
    conditions = tune.conditions(
        "model", prompt_tokens=2, max_tokens=2, model_identity=identity
    )
    assert set(conditions) == tune.PROFILE_CONDITION_FIELDS
    assert conditions["runtime_version"] == CURRENT_VERSION
    profile = {
        "model_id": "model",
        "model_identity": identity.to_dict(),
        "conditions": conditions,
        "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": profile}
    )
    assert tune.load_profile("model", model_identity=identity) == profile
    unknown = dict(profile, conditions=dict(conditions, future_field=True))
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": unknown}
    )
    assert tune.load_profile("model", model_identity=identity) is None
    mismatched = dict(profile, conditions=dict(conditions, model_revision="tampered"))
    with pytest.raises(ModelIdentityError, match="do not match"):
        tune.save_profile(mismatched)
    wrong_schema = dict(
        profile,
        conditions=dict(conditions, conditions_schema="ironmule.tuned_profile.conditions.v1"),
    )
    with pytest.raises(ModelIdentityError, match="do not match"):
        tune.save_profile(wrong_schema)


@pytest.mark.parametrize("field,value", [("mlx_lm", "0.31.4"),
                                          ("runtime_version", "9.9.9"),
                                          ("os", "different-os")])
def test_profile_identity_drift_is_rejected_but_raw_revalidate_access_remains(monkeypatch, field, value):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    identity = _identity()
    base = {
        "conditions_schema": tune.PROFILE_CONDITIONS_SCHEMA,
        "fingerprint": "fp", "model_id": "model",
        "model_revision": identity.revision,
        "model_manifest_sha256": identity.model_manifest_sha256,
        "model_architecture": identity.architecture,
        "quantisation": identity.quantisation,
        "quantisation_sha256": identity.quantisation_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "chip": "test-chip", "memory_bytes": 1, "gpu_cores": 1,
        "mlx": "0.32.0",
        "mlx_lm": "0.31.3", "runtime_version": CURRENT_VERSION, "os": "test",
        "power_source": "AC", "prompt_tokens": 2, "max_tokens": 2,
        "execution_plan": "single_shot",
    }
    profile = {
        "model_id": "model", "model_identity": identity.to_dict(),
        "conditions": base, "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{identity.identity_sha256}": profile}
    )
    monkeypatch.setattr(
        tune, "conditions", lambda *_args, **_kwargs: dict(base, **{field: value})
    )

    assert tune.load_profile("model", model_identity=identity) is None
    assert tune.load_profile(
        "model", require_compatible=False, model_identity=identity
    ) == profile


def test_profile_model_revision_change_is_not_reused_even_for_raw_revalidation(monkeypatch):
    monkeypatch.setattr(tune, "fingerprint", lambda *_args: "fp")
    stored = _identity(revision="old")
    current = _identity(revision="new")
    profile = {
        "model_id": stored.model_id,
        "model_identity": stored.to_dict(),
        "conditions": {
            "conditions_schema": tune.PROFILE_CONDITIONS_SCHEMA,
            "model_id": stored.model_id,
            "model_identity_sha256": stored.identity_sha256,
        },
        "knobs": BASELINE.as_dict(),
    }
    monkeypatch.setattr(
        tune, "_all_profiles", lambda: {f"fp/{stored.identity_sha256}": profile}
    )
    assert tune.load_profile("model", model_identity=current) is None
    assert tune.load_profile(
        "model", require_compatible=False, model_identity=current
    ) is None
    with pytest.raises(ModelIdentityError, match="model_id"):
        tune.load_profile("org/model", model_identity=current)


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
    identity = _identity(tune.DEFAULT_MODEL)
    resolved = types.SimpleNamespace(path=Path("/cached/model"), identity=identity)
    monkeypatch.setattr(tune, "resolve_local_model", lambda *_args, **_kwargs: resolved)
    monkeypatch.setattr(
        tune, "load_engine",
        lambda _model, knobs, **_kwargs: (FakeEngine(knobs), object()),
    )
    monkeypatch.setattr(tune, "prompt_ids", lambda _tokenizer, _prompt: [1, 2])
    monkeypatch.setattr(tune, "_eos_ids", lambda _tokenizer: (99,))
    monkeypatch.setattr(
        tune, "conditions",
        lambda *_args, **_kwargs: {"prompt_tokens": 2, "max_tokens": 2},
    )
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
    assert profile["model_identity"] == identity.to_dict()


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


def test_gpu_busy_does_not_report_a_foreign_command_line(monkeypatch):
    """Callers write this string into evidence records; other processes' args stay out."""
    # `ironmule/__init__.py` rebinds the name `tune` to the function, so the module
    # is only reachable through the import system, not as an attribute of the package.
    tune_module = importlib.import_module("ironmule.tune")

    # Must look like a loaded MLX job, or gpu_busy correctly ignores it.
    secret = "/usr/bin/python3 /Users/someone/private/mlx_train.py --token hunter2"
    line = f"4242 2000000 python {secret}"
    monkeypatch.setattr(
        tune_module.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(stdout=line + "\n", stderr="", returncode=0),
    )

    busy = tune_module.gpu_busy()
    assert busy is not None, "a 2 GB MLX python process must still be detected"
    assert "4242" in busy
    assert "hunter2" not in busy
    assert "private" not in busy
    assert "/Users/" not in busy
