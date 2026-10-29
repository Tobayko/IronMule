"""KP1: class-A knobs for the tuned engine, shipped only with the comparison that qualified them."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ironmule_product import knob_profile as kp

BINDING = {"model_id": "mlx-community/Qwen3-1.7B-bf16", "revision": "9cd6692855d3e06772228e9a962b2606359b2d24",
           "gpu": "Tesla T4", "engine": "tuned", "mlx": "0.32.4", "mlx_lm": "0.31.3"}


def _baseline():
    from ironmule.runtime import BASELINE

    return BASELINE


def test_only_knobs_that_leave_every_answer_unchanged_are_class_a():
    pytest.importorskip("mlx.core")
    candidate = kp.kp1_knobs()
    assert (candidate.compiled_fixed_cache, candidate.readback_every, candidate.fused_argmax) == (True, 4, False)
    assert kp.class_a_knobs(candidate.as_dict()) == candidate
    assert kp.class_a_knobs(replace(candidate, fuse_projections=True, fused_argmax=True).as_dict()) is not None
    for changed in ({"head_skip_prefill": True}, {"speculate_k": 3}, {"prefill_into_fixed": True},
                    {"capacity_slack": 64}, {"wired_fraction": 0.5}):
        assert kp.class_a_knobs({**candidate.as_dict(), **changed}) is None, changed
    assert kp.class_a_knobs({"compiled_fixed_cache": "yes"}) is None and kp.class_a_knobs(None) is None


def _ship(folder, binding=BINDING, **overrides):
    record = kp.knobs_record("KP1-run1-20261029", binding, kp.kp1_knobs(), ["quiet", "bursts", "spike"],
                             "tuned:shortest/w2/g0/sjf", 0.07, [0.04, 0.10], 0.01, 0.05, True)
    record.update(overrides)
    (folder / "kp1").mkdir(parents=True, exist_ok=True)
    (folder / "kp1" / kp.KNOBS_FILE).write_text(json.dumps(record))
    return record


def test_a_shipped_knob_set_serves_only_on_exactly_its_setup(tmp_path):
    pytest.importorskip("mlx.core")
    assert not kp.shipped_knobs(tmp_path) and kp.qualified_knobs(BINDING, tmp_path) is None
    _ship(tmp_path)
    assert kp.shipped_knobs(tmp_path)
    knobs, record = kp.qualified_knobs(BINDING, tmp_path)
    assert knobs == kp.kp1_knobs() and record["experiment"] == "KP1-run1-20261029"
    for key, other in (("gpu", "NVIDIA A100"), ("revision", "main"), ("engine", "reference"), ("mlx", "0.33.0"),
                       ("mlx_lm", "0.32.0"), ("model_id", "mlx-community/Qwen3-4B-bf16")):
        assert kp.qualified_knobs({**BINDING, key: other}, tmp_path) is None, key
    _ship(tmp_path, gain=0.5)  # edited after the digest
    assert kp.qualified_knobs(BINDING, tmp_path) is None


def test_a_failed_or_harmful_comparison_qualifies_nothing(tmp_path):
    pytest.importorskip("mlx.core")
    for bound, floor, harm in ((0.0, 0.0, True), (-0.01, 0.0, True), (0.04, 0.0, False), (0.02, 0.03, True)):
        record = kp.knobs_record("KP1", BINDING, kp.kp1_knobs(), ["quiet"], "tuned", 0.05, [bound, 0.1], floor,
                                 0.05, harm)
        assert record["verdict"] == "fail", (bound, floor, harm)
    with pytest.raises(ValueError, match="class-A"):
        kp.knobs_record("KP1", BINDING, replace(kp.kp1_knobs(), speculate_k=3), ["quiet"], "tuned", 0.05,
                        [0.04, 0.1], 0.0, 0.05, True)
    (tmp_path / "x").mkdir()
    body = {k: v for k, v in _ship(tmp_path).items() if k != "sha256"}
    body["knobs"] = {**body["knobs"], "speculate_k": 3}  # re-signed, but not class A
    (tmp_path / "kp1" / kp.KNOBS_FILE).write_text(json.dumps({**body, "sha256": kp._digest(body)}))
    assert kp.qualified_knobs(BINDING, tmp_path) is None


def test_the_cli_passes_qualified_knobs_to_the_tuned_workers_only_when_bound(monkeypatch):
    import ironmule_product.cli as cli
    from ironmule_product import device_pool

    spec = SimpleNamespace(model_id=BINDING["model_id"], revision=BINDING["revision"])
    monkeypatch.setattr(device_pool, "cuda_devices", lambda: ["0", "1"])
    monkeypatch.setattr(device_pool, "gpu_name", lambda devices: "Tesla T4")
    monkeypatch.setattr(kp, "framework_versions", lambda: {"mlx": "0.32.4", "mlx_lm": "0.31.3"})
    monkeypatch.setattr(kp, "shipped_knobs", lambda folder=None: False)
    assert cli._qualified_knob_options(spec, [None]) == {}
    seen = []

    def found(binding, folder=None):
        seen.append(binding)
        return SimpleNamespace(as_dict=lambda: {"compiled_fixed_cache": True}), {"experiment": "KP1-run1"}

    monkeypatch.setattr(kp, "shipped_knobs", lambda folder=None: True)
    monkeypatch.setattr(kp, "qualified_knobs", found)
    options = cli._qualified_knob_options(spec, ["1"])
    assert options == {"engine_knobs": {"knobs": {"compiled_fixed_cache": True}, "source": "qualified:KP1-run1"}}
    assert seen == [BINDING]
    monkeypatch.setattr(kp, "qualified_knobs", lambda binding, folder=None: None)
    assert cli._qualified_knob_options(spec, ["1"]) == {}


def test_the_worker_client_carries_only_class_a_knobs_to_the_tuned_engine():
    pytest.importorskip("mlx.core")
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.types import ModelSpec

    spec = ModelSpec("local/test", "revision", "/models/test", 1)
    knobs = {"knobs": kp.kp1_knobs().as_dict(), "source": "KP1-candidate"}
    client = MLXWorkerClient(spec, execution_variant="current_engine", engine_knobs=knobs)
    assert client.engine_knobs == knobs
    for bad in ({"execution_variant": "reference"}, {"execution_variant": "current_engine",
                                                     "engine_configuration": "dense_stacked"}):
        with pytest.raises(ValueError, match="class-A knobs"):
            MLXWorkerClient(spec, engine_knobs=knobs, **bad)
    for bad_knobs in ({**knobs, "knobs": {**knobs["knobs"], "head_skip_prefill": True}},
                      {**knobs, "source": "two words"}, {"knobs": knobs["knobs"]}):
        with pytest.raises(ValueError, match="class-A knobs"):
            MLXWorkerClient(spec, execution_variant="current_engine", engine_knobs=bad_knobs)


def test_the_bridge_serves_class_a_knobs_only_when_no_stored_profile_applies(monkeypatch, tmp_path):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import qwen3

    import importlib

    identity_module = importlib.import_module("ironmule.model_identity")
    tune_module = importlib.import_module("ironmule.tune")  # the package exports a function named `tune`
    from ironmule_product.engine_bridge import CurrentEngineBridge
    from ironmule_product.types import ModelSpec

    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=1, intermediate_size=128,
                           num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=1,
                           max_position_embeddings=256, rope_theta=10000.0, head_dim=32, tie_word_embeddings=True)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    tokenizer = SimpleNamespace(eos_token_id=127, eos_token_ids={127}, encode=lambda text: [1, 2],
                                decode=lambda ids: "")
    identity = SimpleNamespace(to_dict=lambda: {"identity_sha256": "a" * 64}, identity_sha256="a" * 64)
    monkeypatch.setattr(identity_module, "build_model_identity", lambda *a, **k: identity)
    service_module = importlib.import_module("ironmule.service")
    real_runtime = service_module.Runtime  # the stand-in identity is not a ModelIdentity; the rest is real
    monkeypatch.setattr(service_module, "Runtime",
                        lambda engine, tok, mode, model_id, model_identity: real_runtime(engine, tok, mode=mode))
    stored = {"profile": None}
    monkeypatch.setattr(tune_module, "load_profile", lambda *a, **k: stored["profile"])
    spec = ModelSpec("local/test", "revision", str(tmp_path), 1)
    requested = {"knobs": kp.kp1_knobs().as_dict(), "source": "KP1-candidate"}

    bridge = CurrentEngineBridge.from_loaded(model, tokenizer, spec, class_a_knobs=requested)
    assert bridge.metadata()["selected_knobs"] == kp.kp1_knobs().as_dict()
    assert bridge.metadata()["profile_source"] == "class_a:KP1-candidate"
    assert bridge.generate([1, 2, 3], 4).tokens == CurrentEngineBridge.from_loaded(
        model, tokenizer, spec).generate([1, 2, 3], 4).tokens  # class A: the same answer
    assert CurrentEngineBridge.from_loaded(model, tokenizer, spec).metadata()["selected_knobs"] == \
        _baseline().as_dict()
    stored["profile"] = {"knobs": replace(_baseline(), readback_every=2).as_dict()}  # this machine's own tune wins
    assert CurrentEngineBridge.from_loaded(model, tokenizer, spec, class_a_knobs=requested).metadata()[
        "profile_source"] == "compatible_tuned_profile"
    for bad in ({**requested, "knobs": {**requested["knobs"], "speculate_k": 3}}, {"knobs": requested["knobs"]}):
        with pytest.raises(ValueError, match="class-A knobs"):
            CurrentEngineBridge.from_loaded(model, tokenizer, spec, class_a_knobs=bad)


def _tiny_model():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models import qwen3

    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=1, intermediate_size=128,
                           num_attention_heads=2, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=1,
                           max_position_embeddings=256, rope_theta=10000.0, head_dim=32, tie_word_embeddings=True)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model, SimpleNamespace(eos_token_id=127, eos_token_ids={127})


def test_the_preflight_compares_logits_and_state_bit_for_bit_at_every_capacity_that_fits(monkeypatch):
    model, tokenizer = _tiny_model()
    prompts = {"short": [3, 5, 7], "long": list(range(1, 90))}
    result = kp.compare_class_a(model, tokenizer, replace(_baseline(), readback_every=4), prompts,
                                [64, 128, 192, 128], 6)
    assert result["ok"] and result["differ"] == []
    assert kp.compare_class_a(model, tokenizer, kp.kp1_knobs(), prompts, [128], 6)["ok"]  # a CPU smoke, not T4 evidence
    assert [(r["capacity"], r["prompt"]) for r in result["rows"]] == [
        (64, "short"), (128, "short"), (128, "long"), (192, "short"), (192, "long")]  # 89 + 6 + 8 > 64
    from ironmule import runtime

    original = runtime.Engine._body

    def drifting(self, capacity, width):  # a candidate whose steps move one logit by one ulp
        body = original(self, capacity, width)
        if self.knobs == _baseline():
            return body
        return lambda ids, state: (lambda out: (out[0] * (1 + 2 ** -20), out[1]))(body(ids, state))

    monkeypatch.setattr(runtime.Engine, "_body", drifting)
    result = kp.compare_class_a(model, tokenizer, replace(_baseline(), readback_every=4), prompts, [128], 6)
    assert not result["ok"] and result["differ"] == [[128, "short"], [128, "long"]]
