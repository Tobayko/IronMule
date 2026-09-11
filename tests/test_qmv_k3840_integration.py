from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ironmule import qmv_k3840  # noqa: E402
from ironmule.runtime import BASELINE, Engine, Knobs  # noqa: E402


def _projection(out_features: int, in_features: int = 3840, *, group_size: int = 64,
                bits: int = 4, dtype=mx.bfloat16) -> nn.QuantizedLinear:
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = layer.weight.astype(dtype)
    return nn.QuantizedLinear.from_linear(layer, group_size=group_size, bits=bits)


class _Holder(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.proj = inner


def test_the_knob_is_off_by_default() -> None:
    assert BASELINE.k3840_matvec is False
    assert Knobs().k3840_matvec is False


def test_the_kernel_source_is_the_studied_one() -> None:
    """Integration must not quietly alter the candidate B42 and B43 qualified."""

    studied = (ROOT / "tools" / "b42_qmv_kernel.py").read_text()
    shipped = (ROOT / "ironmule" / "qmv_k3840.py").read_text()
    body = studied[studied.index("SIMD_SIZE = 32"):]
    assert body in shipped, "the shipped kernel diverged from the qualified source"


def test_admission_refuses_shapes_it_was_not_qualified_for() -> None:
    assert qmv_k3840._admit_projection(_projection(4096)) is True
    # Not a multiple of the eight rows a threadgroup writes.
    assert qmv_k3840._admit_projection(_projection(4100)) is False
    assert qmv_k3840._admit_projection(_projection(4096, group_size=128)) is False
    assert qmv_k3840._admit_projection(_projection(4096, bits=8)) is False
    assert qmv_k3840._admit_projection(_projection(4096, dtype=mx.float32)) is False
    assert qmv_k3840._admit_projection(_projection(4096, in_features=2560)) is False


def test_admission_fails_closed_without_a_model_identity() -> None:
    with pytest.raises(qmv_k3840.K3840Unsupported):
        qmv_k3840.enable(_Holder(_projection(4096)), None)


def test_a_directly_built_engine_does_not_admit_without_identity() -> None:
    """load_engine attaches identity; a bare Engine must stay on the library path."""

    engine = Engine.__new__(Engine)
    engine.knobs = Knobs(k3840_matvec=True)
    engine._k3840_pending = True
    engine.k3840_admission = None
    engine.model = _Holder(_projection(4096))
    with pytest.raises(qmv_k3840.K3840Unsupported):
        engine.admit_k3840(None)
    assert engine.k3840_admission is None


def test_disable_restores_the_original_module_object() -> None:
    source = _projection(4096)
    holder = _Holder(source)
    admitted, skipped = [], []
    qmv_k3840._replace(holder, admitted, skipped)

    assert admitted == ["proj"] and skipped == []
    assert qmv_k3840.disable(holder) == 1
    assert holder.proj is source


def test_multi_token_input_takes_the_library_path() -> None:
    source = _projection(256)
    swapped = qmv_k3840.K3840QuantizedLinear(source)
    wide = mx.random.normal((1, 3, 3840)).astype(mx.bfloat16)
    mx.eval(wide)

    through = swapped(wide)
    reference = mx.quantized_matmul(
        wide, source.weight, source.scales, source.biases,
        transpose=True, group_size=64, bits=4,
    )
    mx.eval(through, reference)

    assert bytes(memoryview(through)) == bytes(memoryview(reference))
