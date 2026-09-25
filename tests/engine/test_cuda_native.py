import mlx.core as mx
import mlx.nn as nn
import pytest

from ironmule import cuda_native

TURING = {"compute_capability_major": 7, "compute_capability_minor": 5}


class Tiny(nn.Module):
    def __init__(self, bits=4, group_size=64):
        super().__init__()
        self.embed = nn.Embedding(96, 128)
        self.proj = nn.Linear(128, 64, bias=False)
        nn.quantize(self, group_size=group_size, bits=bits)
        self.set_dtype(mx.bfloat16)


def test_refused_off_pre_ampere_cuda():
    for info in (None, {}, {"compute_capability_major": 8}, {"compute_capability_major": True}):
        with pytest.raises(ValueError, match="below compute capability 8"):
            cuda_native.install(Tiny(), info)


def test_refused_when_nothing_is_eligible():
    with pytest.raises(ValueError, match="no 4-bit"):
        cuda_native.install(Tiny(bits=8), TURING)
    model = Tiny()
    model.set_dtype(mx.float32)  # float32 scales are the float32 plan's, not this one's
    with pytest.raises(ValueError, match="no 4-bit"):
        cuda_native.install(model, TURING)


def test_refused_when_the_probe_disagrees(monkeypatch):
    monkeypatch.setattr(cuda_native, "_probe", lambda module: 0.5)
    model = Tiny()
    with pytest.raises(ValueError, match="disagrees"):
        cuda_native.install(model, TURING)
    assert type(model.proj) is nn.QuantizedLinear  # nothing half-installed


def test_install_swaps_only_this_models_modules(monkeypatch):
    monkeypatch.setattr(cuda_native, "_probe", lambda module: 0.0)
    model, other = Tiny(), Tiny()
    tokens = mx.array([[1, 2, 3]])
    before = model.embed(tokens)
    record = cuda_native.install(model, TURING)
    assert record == {"modules": 2, "probe_rel_error": 0.0}
    assert type(model.proj) is cuda_native.NativeQuantizedLinear
    assert type(model.embed) is cuda_native.NativeQuantizedEmbedding
    assert type(other.proj) is nn.QuantizedLinear
    assert mx.array_equal(model.embed(tokens), before)  # lookups unchanged


def test_prefill_path_and_float32_fallback_match_the_reference():
    model = Tiny()
    w, scales, biases = model.proj["weight"], model.proj["scales"], model.proj["biases"]
    reference_w = mx.dequantize(w, scales.astype(mx.float32), biases.astype(mx.float32), group_size=64, bits=4)
    x = mx.random.normal((1, 16, 128), key=mx.random.key(1))
    reference = x @ reference_w.T
    prefill = cuda_native.matmul(x.astype(mx.bfloat16), w, scales, biases).astype(mx.float32)
    assert prefill.shape == (1, 16, 64)
    assert float(mx.max(mx.abs(prefill - reference)) / mx.max(mx.abs(reference))) < 1e-2
    fallback = cuda_native.matmul(x, w, scales.astype(mx.float32), biases.astype(mx.float32))
    assert float(mx.max(mx.abs(fallback - reference)) / mx.max(mx.abs(reference))) < 1e-3


def test_load_engine_refuses_fusion_with_the_native_plan_before_loading(monkeypatch):
    import importlib
    import sys
    import types

    from ironmule.runtime import Knobs

    tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` is also a function

    loaded = []
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=lambda source: loaded.append(source)))
    with pytest.raises(ValueError, match="fuse_projections is unsupported"):
        tune.load_engine("org/model", Knobs(fuse_projections=True), compute_dtype="native")
    assert loaded == []
    assert tune._is_unsupported_candidate(ValueError(
        "fuse_projections is unsupported with compute_dtype='native'"))
