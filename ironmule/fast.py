"""Graph surgery: fuse the projections that share an input into one matmul.

Gemma 3 runs three quantised matmuls on the same attention input (q/k/v) and two
on the same MLP input (gate/up). Concatenating them along the *output* axis
leaves every output row with its own group scales and biases, so the numbers are
bit identical while the kernel count per layer drops from five to two.

The rewrite is a copy of the upstream `__call__` bodies, so it is pinned to the
mlx_lm version it was verified against and fails closed on anything else.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

VERIFIED_MLX_LM = ("0.31.3",)


class FusionUnsupported(RuntimeError):
    """The model or library version is outside what this rewrite was verified on."""


def _concat_linear(parts: list[nn.Module]) -> nn.Module:
    """One linear whose output is the concatenation of the parts' outputs."""
    first = parts[0]
    if any(getattr(part, "bias", None) is not None for part in parts):
        raise FusionUnsupported("biased projections are not fused")
    out_dims = sum(part.scales.shape[0] if hasattr(part, "scales") else part.weight.shape[0] for part in parts)

    if isinstance(first, nn.QuantizedLinear):
        if any(p.group_size != first.group_size or p.bits != first.bits for p in parts):
            raise FusionUnsupported("mixed quantisation cannot be fused")
        in_dims = first.weight.shape[1] * 32 // first.bits
        # The freshly initialised arrays are never evaluated; they are replaced below.
        fused = nn.QuantizedLinear(in_dims, out_dims, bias=False,
                                   group_size=first.group_size, bits=first.bits)
        fused.weight = mx.concatenate([p.weight for p in parts], axis=0)
        fused.scales = mx.concatenate([p.scales for p in parts], axis=0)
        fused.biases = mx.concatenate([p.biases for p in parts], axis=0)
        return fused

    fused = nn.Linear(first.weight.shape[1], out_dims, bias=False)
    fused.weight = mx.concatenate([p.weight for p in parts], axis=0)
    return fused


def _out_features(part: nn.Module) -> int:
    return part.scales.shape[0] if hasattr(part, "scales") else part.weight.shape[0]


def _fused_attention_class(base: type) -> type:
    from mlx_lm.models.base import scaled_dot_product_attention

    class FusedAttention(base):  # type: ignore[valid-type,misc]
        def __call__(self, x, mask=None, cache=None):
            B, L, _ = x.shape
            queries, keys, values = mx.split(self.qkv_proj(x), self.qkv_splits, axis=-1)
            queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

    return FusedAttention


def _fused_mlp_class(base: type) -> type:
    class FusedMLP(base):  # type: ignore[valid-type,misc]
        def __call__(self, x):
            gate, up = mx.split(self.gate_up_proj(x), self.gate_up_split, axis=-1)
            return self.down_proj(nn.gelu_approx(gate) * up)

    return FusedMLP


def fuse_projections(model, *, check_version: bool = True) -> int:
    """Rewrite every transformer block in place. Returns the number of blocks fused."""
    if check_version:
        import mlx_lm
        if mlx_lm.__version__ not in VERIFIED_MLX_LM:
            raise FusionUnsupported(
                f"fusion verified for mlx_lm {VERIFIED_MLX_LM}, found {mlx_lm.__version__}"
            )
    layers = getattr(model, "layers", None)
    if not layers:
        raise FusionUnsupported("model exposes no layers")

    attention_class = mlp_class = None
    fused = 0
    for block in layers:
        attn, mlp = getattr(block, "self_attn", None), getattr(block, "mlp", None)
        if attn is None or mlp is None:
            raise FusionUnsupported("block has no self_attn/mlp pair")
        if all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            parts = [attn.q_proj, attn.k_proj, attn.v_proj]
            sizes = [_out_features(part) for part in parts]
            attn.qkv_proj = _concat_linear(parts)
            attn.qkv_splits = [sizes[0], sizes[0] + sizes[1]]
            for name in ("q_proj", "k_proj", "v_proj"):
                attn.pop(name)
            attention_class = attention_class or _fused_attention_class(type(attn))
            attn.__class__ = attention_class
        if all(hasattr(mlp, name) for name in ("gate_proj", "up_proj")):
            parts = [mlp.gate_proj, mlp.up_proj]
            mlp.gate_up_proj = _concat_linear(parts)
            mlp.gate_up_split = [_out_features(parts[0])]
            for name in ("gate_proj", "up_proj"):
                mlp.pop(name)
            mlp_class = mlp_class or _fused_mlp_class(type(mlp))
            mlp.__class__ = mlp_class
        fused += 1
    return fused


def _self_check() -> None:
    """Fusion must be bit identical. Runs on CPU so it never competes for the GPU."""
    mx.set_default_device(mx.cpu)
    from mlx_lm.models.gemma3_text import ModelArgs, Gemma3Model

    args = ModelArgs(model_type="gemma3_text", hidden_size=64, num_hidden_layers=4,
                     intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=16, vocab_size=128, sliding_window=8, sliding_window_pattern=2)
    model = Gemma3Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.eval(model.parameters())

    tokens = mx.array([[3, 9, 27, 81, 5, 6]])
    reference = model(tokens)
    mx.eval(reference)

    assert fuse_projections(model, check_version=False) == 4
    fused = model(tokens)
    mx.eval(fused)

    assert fused.shape == reference.shape, (fused.shape, reference.shape)
    assert mx.array_equal(fused, reference).item(), "fusion changed the numbers"

    for block in model.layers:
        assert not hasattr(block.self_attn, "q_proj"), "unfused projection left behind"
        assert not hasattr(block.mlp, "gate_proj"), "unfused projection left behind"

    print("fast self-check ok: 4 blocks fused, output bit identical")


if __name__ == "__main__":
    _self_check()
