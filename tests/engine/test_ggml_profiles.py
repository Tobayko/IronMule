"""A worker's quality gate can admit only the same complete kernel profile on its bound target."""

import os
from types import SimpleNamespace

import pytest

from ironmule import ggml_binary
from ironmule import ggml_engine as engine
from ironmule import ggml_profiles as profiles
from ironmule_product import cli, pinned_ggml
from ironmule_product.errors import InvalidRequest

A100 = {"provider": "cuda", "description": "NVIDIA A100-SXM4-80GB"}
NATIVE = {"sha256": {"backend/CUDA": "pinned-cuda"}}


def test_profile_choices_are_bound_and_explicit_overrides_survive():
    def select(layers=999, devices=(A100,), model=profiles.QWEN32, native=NATIVE, overrides=None):
        return profiles.select(layers, devices, model, native, overrides or {})
    ref = profiles.profile("conservative")
    k = profiles.profile("ps1-ps1a-pk6")
    assert select() == ("ps1-ps1a-pk6", k)
    assert select(devices=[dict(A100, description="NVIDIA H100 80GB HBM3")]) == ("conservative", ref)
    assert select(devices=[dict(A100, provider="rocm")]) == ("conservative", ref)
    assert select(model="other-model") == ("conservative", ref)
    assert select(layers=1) == ("conservative", ref)
    assert select(devices=(A100, A100)) == ("conservative", ref)
    assert select(native={}) == ("conservative", ref)
    assert select(layers=0, devices=()) == ("cpu", k)
    assert select(model="other", overrides={"IRONMULE_PS1": "1"}) == ("explicit", {**ref, "IRONMULE_PS1": "1"})


def test_failed_a100_candidate_qualifies_reference_and_binds_worker(monkeypatch):
    # Qualification operates in a fresh process per worker. Avoid altering the runner's settings.
    for key in (*profiles.FLAGS, "IRONMULE_GPU_PROFILE_GATE", "IRONMULE_KG_LIB"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(engine, "ggml_devices", lambda: [A100])
    monkeypatch.setattr(engine, "gpu_layers", lambda: 999)
    monkeypatch.setattr(engine, "file_sha256", lambda gguf: profiles.QWEN32 if gguf == "qwen" else "other")
    monkeypatch.setattr(engine, "machine_key", lambda: "target")
    monkeypatch.setattr(ggml_binary, "identity", lambda: NATIVE)
    attempts = []
    def measure(*args, **kwargs):
        attempts.append({k: os.environ[k] for k in profiles.FLAGS})
        if attempts[-1]["IRONMULE_PK6"] == "1":
            raise InvalidRequest("--engine ggml: the GGUF did not pass the quality gate on this machine")
        return {"ggml_gate": {"passes": True}, "ggml_kernel_library": ""}
    monkeypatch.setattr(cli, "_local_ggml_options", measure)
    result = pinned_ggml.qualify(SimpleNamespace(gguf="qwen"), None, None, True, {})
    assert attempts == [profiles.profile("ps1-ps1a-pk6"), profiles.profile("conservative")]
    receipt = result["ggml_gate"]["kernel_profile"]
    assert receipt["fallback_reason"] and receipt["flags"] == result["ggml_worker_env"]
    pinned_ggml.verify(receipt, "qwen")
    with pytest.raises(ValueError, match="requalify"):
        pinned_ggml.verify(receipt, "other-model")
    with monkeypatch.context() as m:
        m.setattr(engine, "machine_key", lambda: "different-host")
        with pytest.raises(ValueError, match="requalify"):
            pinned_ggml.verify(receipt, "qwen")
    with monkeypatch.context() as m:
        m.setattr(ggml_binary, "identity", lambda: {"sha256": {"backend/CUDA": "rebuilt"}})
        with pytest.raises(ValueError, match="requalify"):
            pinned_ggml.verify(receipt, "qwen")
    monkeypatch.setenv("IRONMULE_PK6", "1")
    with pytest.raises(ValueError, match="requalify"):
        pinned_ggml.verify(receipt, "qwen")


def test_gate_cache_accounts_for_both_flags(monkeypatch, tmp_path):
    f = tmp_path / "fixture.gguf"
    f.write_bytes(b"fixture")
    monkeypatch.setattr(engine, "machine_key", lambda: "host")
    keys = set()
    for ps1 in ("0", "1"):
        for pk6 in ("0", "1"):
            monkeypatch.setenv("IRONMULE_PS1", ps1)
            monkeypatch.setenv("IRONMULE_PK6", pk6)
            keys.add(engine.gate_key(str(f), str(f)))
    assert len(keys) == 4


def test_launcher_omits_implicit_parent_flags_and_preserves_lab_override(monkeypatch, tmp_path):
    import json

    from ironmule_product.types import ModelSpec
    monkeypatch.setattr(engine, "_IMPLICIT_KERNEL_FLAGS", {"IRONMULE_PS1": "1"})
    for k in profiles.FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IRONMULE_PS1", "1")  # inserted by the standalone engine at import
    monkeypatch.setenv("IRONMULE_PK6", "0")  # provided by caller
    captures = []
    def child(argv, *, input, **kwargs):
        captures.append(json.loads(input))
        return SimpleNamespace(returncode=0, stdout='{"ggml_gate":{"passes":true}}')
    monkeypatch.setattr(cli.subprocess, "run", child)
    args = SimpleNamespace(gguf="file", draft_gguf=None, speculate="off", agent="on", model=None)
    for device in ("local", "cpu"):
        cli._pinned_ggml_options(device, args, ModelSpec("model", "sha256:model", "/test", 1),
                                 SimpleNamespace(root=tmp_path))
    assert [c["kernel_overrides"] for c in captures] == [{"IRONMULE_PK6": "0"}] * 2
    assert os.environ["IRONMULE_PS1"] == "1" and os.environ["IRONMULE_PK6"] == "0"
