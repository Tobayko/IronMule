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


def _reference_gather_qmv(x, w, scales, biases, idx, xs):
    rows = x[mx.arange(idx.size) // xs].astype(mx.float32)
    dense = mx.dequantize(w[idx], scales[idx].astype(mx.float32), biases[idx].astype(mx.float32),
                          group_size=64, bits=4)
    return (dense @ rows[:, :, None]).squeeze(-1).astype(mx.bfloat16)


def _switch_glu():
    from mlx_lm.models.switch_layers import SwitchGLU

    glu = SwitchGLU(128, 64, 16)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(glu, name, getattr(glu, name).to_quantized(64, 4))
    glu.set_dtype(mx.bfloat16)
    return glu


def test_experts_route_by_rows_and_match_stock(monkeypatch):
    # PERF1-S: the CUDA kernel is replaced by its float32 reference; what is checked here is
    # which (token, expert) rows reach it, with which shared activation row, and the shapes
    # mlx-lm's SwitchGLU gets back, for one request, eight, and a prefill.
    calls = []
    monkeypatch.setattr(cuda_native, "_gather_qmv",
                        lambda *a: calls.append((a[4].size, a[5])) or _reference_gather_qmv(*a))
    monkeypatch.setattr(cuda_native, "_probe", lambda module: 0.0)
    glu = _switch_glu()
    stock = _switch_glu()
    stock.update(glu.parameters())
    assert cuda_native.install(glu, TURING)["modules"] == 3
    assert type(glu.down_proj) is cuda_native.NativeQuantizedSwitchLinear
    for batch, length, routed in ((1, 1, [(8, 8), (8, 8), (8, 1)]), (8, 1, [(64, 1)] * 3),
                                  (1, 40, [])):
        calls.clear()
        x = mx.random.normal((batch, length, 128), key=mx.random.key(2)).astype(mx.bfloat16)
        scores = mx.random.normal((batch, length, 16), key=mx.random.key(3))
        indices = mx.argpartition(scores, kth=-8, axis=-1)[..., -8:]
        want = stock(x, indices).astype(mx.float32)
        got = glu(x, indices).astype(mx.float32)
        assert got.shape == want.shape == (batch, length, 8, 128)
        assert float(mx.max(mx.abs(got - want)) / mx.max(mx.abs(want))) < 2e-2
        # Eight requests are 64 rows, which SwitchGLU sorts by expert first; a prefill's 320
        # rows go to float16 gather_qmm instead.
        assert calls == routed
