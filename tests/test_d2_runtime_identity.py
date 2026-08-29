import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ironmule.model_identity import (
    ModelIdentity,
    ModelIdentityError,
    canonical_json,
    canonical_sha256,
)
from ironmule.runtime import BASELINE
from ironmule.service import Runtime


fingerprint_module = importlib.import_module("ironmule.fingerprint")
tune = importlib.import_module("ironmule.tune")


def identity(revision="revision"):
    quantisation = {"bits": 4, "group_size": 64}
    return ModelIdentity(
        model_id="org/model",
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


def fingerprint_environment(monkeypatch):
    from ironmule import bench, hw

    monkeypatch.setattr(hw, "static_facts", lambda: {
        "chip": "Apple Test", "memory_bytes": 32, "gpu_cores": 8, "machine": "arm64",
    })
    monkeypatch.setattr(hw, "fingerprint", lambda _facts=None: "hardware")
    monkeypatch.setattr(bench, "environment", lambda: {
        "os": "test-os", "mlx": "0.32.0", "mlx_lm": "0.31.3", "power_source": "AC",
    })


def test_runtime_fingerprint_requires_and_binds_exact_identity(monkeypatch):
    fingerprint_environment(monkeypatch)
    exact = identity()
    record = fingerprint_module.build(
        exact.model_id, exact.quantisation, "strict_one_shot", "interactive",
        {"prompt_tokens": 10}, model_identity=exact,
    )
    assert record["fingerprint_schema"] == "ironmule.runtime_fingerprint.v2"
    assert record["model_revision"] == "revision"
    assert record["model_manifest_sha256"] == "a" * 64
    assert record["model_identity_sha256"] == exact.identity_sha256
    assert record["tokenizer_sha256"] == "b" * 64

    changed = dict(record, model_revision="changed")
    ok, why = fingerprint_module.usable(record, changed)
    assert not ok and any("model_revision" in item for item in why["incompatible"])
    with pytest.raises(ModelIdentityError, match="requires exact"):
        fingerprint_module.build("org/model", exact.quantisation,
                                 "strict_one_shot", "interactive")
    with pytest.raises(ModelIdentityError, match="quantisation"):
        fingerprint_module.build(
            "org/model", {"bits": 8}, "strict_one_shot", "interactive",
            model_identity=exact,
        )


class FakeTokenizer:
    eos_token_ids = (1,)

    @staticmethod
    def decode(_tokens):
        return ""


def test_runtime_manual_execution_is_allowed_but_validity_fails_closed(monkeypatch):
    fingerprint_environment(monkeypatch)
    engine = SimpleNamespace(model_identity=None)
    runtime = Runtime(engine, FakeTokenizer(), model_id="org/model")
    with pytest.raises(ModelIdentityError, match="requires exact"):
        runtime.fingerprint()

    engine.model_identity = identity()
    runtime = Runtime(engine, FakeTokenizer(), model_id="org/model")
    assert runtime.model_id == "org/model"
    assert runtime.quantisation == {"bits": 4, "group_size": 64}
    assert runtime.fingerprint()["model_revision"] == "revision"


def test_runtime_rejects_conflicting_explicit_identity_or_model_id():
    loaded = identity("loaded")
    engine = SimpleNamespace(model_identity=loaded)
    with pytest.raises(ModelIdentityError, match="Engine identity"):
        Runtime(
            engine, FakeTokenizer(), model_id="org/model",
            model_identity=identity("other"),
        )
    with pytest.raises(ModelIdentityError, match="model_id conflicts"):
        Runtime(engine, FakeTokenizer(), model_id="other/model")


def test_runtime_load_resolves_once_and_passes_identity_to_profile_and_engine(monkeypatch):
    exact = identity()
    resolved = SimpleNamespace(path=Path("/cached/model"), identity=exact)
    observed = {}

    def fake_resolve(model_id, revision=None):
        observed["resolve"] = (model_id, revision)
        return resolved

    def fake_profile(model_id, **kwargs):
        observed["profile"] = (model_id, kwargs)
        return None

    engine = SimpleNamespace(model_identity=exact)

    def fake_load(model_id, knobs, **kwargs):
        observed["load"] = (model_id, knobs, kwargs)
        return engine, FakeTokenizer()

    monkeypatch.setattr(tune, "resolve_local_model", fake_resolve)
    monkeypatch.setattr(tune, "load_profile", fake_profile)
    monkeypatch.setattr(tune, "load_engine", fake_load)
    runtime = Runtime.load("org/model", revision="revision")
    assert observed["resolve"] == ("org/model", "revision")
    assert observed["profile"][1]["model_identity"] == exact
    assert observed["load"][2]["resolved_source"] is resolved
    assert runtime.model_identity == exact


def test_runtime_revalidation_invalidates_exact_model_drift(monkeypatch, tmp_path):
    fingerprint_environment(monkeypatch)
    first_engine = SimpleNamespace(model_identity=identity("one"))
    first = Runtime(first_engine, FakeTokenizer(), model_id="org/model")
    store = tmp_path / "fingerprint.json"
    assert first.revalidate(store=store)["verdict"] == "recorded_first_fingerprint"

    second_engine = SimpleNamespace(model_identity=identity("two"))
    second = Runtime(second_engine, FakeTokenizer(), model_id="org/model")
    result = second.revalidate(store=store)
    assert result["verdict"] == "revalidation_required"
    assert any("model_revision" in item for item in result["incompatible"])
