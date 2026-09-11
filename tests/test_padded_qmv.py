from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from b41_padded_qmv_ab import BOUNDARY, PadCache, PaddedQuantizedLinear, _pad_model  # noqa: E402


def _linear(out_features: int, in_features: int) -> nn.QuantizedLinear:
    layer = nn.Linear(in_features, out_features, bias=False)
    return nn.QuantizedLinear.from_linear(layer, group_size=64, bits=4)


def test_padded_weights_are_exactly_zero_but_the_result_is_not_bit_identical() -> None:
    """Both halves of the entry's finding, held in one test.

    The added columns dequantise to exactly 0.0, so no term is added mathematically. The
    result still moves, because crossing the 512 boundary selects a different kernel and
    that kernel reduces in a different order. Exact arithmetic, different rounding.
    """

    source = _linear(128, 1152)
    padded = PaddedQuantizedLinear(source, 1536, PadCache())
    x = mx.random.normal((1, 1152)).astype(mx.float32)

    expected = source(x)
    actual = padded(x)
    mx.eval(expected, actual)

    assert padded.padded_columns % BOUNDARY == 0
    weights = mx.dequantize(
        padded.weight, padded.scales, padded.biases, group_size=64, bits=4
    )
    assert float(mx.max(mx.abs(weights[:, 1152:])).item()) == 0.0

    difference = float(mx.max(mx.abs(actual - expected)).item())
    scale = float(mx.max(mx.abs(expected)).item())
    assert difference <= 1e-5 * scale, "a reduction-order difference, not a different sum"


def test_the_pad_cache_reuses_one_padded_copy_per_input() -> None:
    cache = PadCache()
    x = mx.random.normal((1, 1152))

    first = cache.padded(x, 1536)
    again = cache.padded(x, 1536)
    other = cache.padded(mx.random.normal((1, 1152)), 1536)

    assert first is again
    assert other is not first


def test_only_off_boundary_linears_are_replaced() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.aligned = _linear(64, 2560)
            self.off = _linear(64, 3840)

    block = Block()
    changed: list[str] = []

    count = _pad_model(block, changed, PadCache())

    assert count == 1
    assert changed == ["off:3840->4096"]
    assert isinstance(block.aligned, nn.QuantizedLinear)
    assert isinstance(block.off, PaddedQuantizedLinear)
