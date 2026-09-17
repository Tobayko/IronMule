"""Graph surgery: fuse the projections that share an input into one matmul.

A transformer block runs three quantised matmuls on the same attention input (q/k/v)
and two on the same MLP input (gate/up). Concatenating them along the *output* axis
leaves every output row with its own group scales and biases, so the numbers are
bit identical while the kernel count per layer drops from five to two.

The rewrite is a transcription of the upstream `__call__` bodies, so it is pinned both
to the mlx_lm version and to the exact module it was transcribed from, and fails closed
on anything else. Pinning only the version is not enough: one body applied to every
architecture is how a fused MLP computes GELU where the model wants SwiGLU.

Every split half is copied out of the fused buffer with `mx.contiguous` before it is used.
Upstream hands each consumer a fresh matmul output; a split hands it a strided view into
one larger array, and that difference is not free. PORT2 run 3 measured it on a Kaggle T4:
fused llama returned `'_REF, , , The, was. The, The, …'` and disagreed with itself between
two calls in one process, with CUDA graph capture on it aborted the process outright, and
the same arm with this copy restored reproduced the reference exactly. Gemma 3 and Qwen 3
survived without it only because their QK norm materialises queries and keys before
`mx.fast.rope` ever sees them; llama has no QK norm, so nothing did. Do not remove these
copies to save an allocation without re-running that probe.
"""

from __future__ import annotations

from typing import Any

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


def _gemma3_attention(self, x, mask=None, cache=None):
    """`mlx_lm.models.gemma3_text.Attention.__call__`, q/k/v read from one matmul."""
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    queries, keys, values = (mx.contiguous(part) for part in
                             mx.split(self.qkv_proj(x), self.qkv_splits, axis=-1))
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


def _llama_attention(self, x, mask=None, cache=None):
    """`mlx_lm.models.llama.Attention.__call__`: no QK norm, otherwise identical."""
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    queries, keys, values = (mx.contiguous(part) for part in
                             mx.split(self.qkv_proj(x), self.qkv_splits, axis=-1))
    queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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


def _qwen3_attention(self, x, mask=None, cache=None):
    """`mlx_lm.models.qwen3.Attention.__call__`: QK norm before the transpose, not after."""
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    queries, keys, values = (mx.contiguous(part) for part in
                             mx.split(self.qkv_proj(x), self.qkv_splits, axis=-1))
    queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

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


def _gelu_mlp(self, x):
    """Gemma 3's `down_proj(gelu_approx(gate) * up)`."""
    gate, up = (mx.contiguous(part) for part in
                mx.split(self.gate_up_proj(x), self.gate_up_split, axis=-1))
    return self.down_proj(nn.gelu_approx(gate) * up)


def _swiglu_mlp(self, x):
    """`down_proj(swiglu(gate, up))` — llama, Qwen 3 and Ministral 3 share this body."""
    from mlx_lm.models.activations import swiglu

    gate, up = (mx.contiguous(part) for part in
                mx.split(self.gate_up_proj(x), self.gate_up_split, axis=-1))
    return self.down_proj(swiglu(gate, up))


# A fused body is a transcription of one upstream `__call__`, so it is only ever applied
# to the module it was transcribed from. Anything unlisted keeps its own projections.
# PORT2 run 1 is why this table exists: the single Gemma 3 body was applied to every
# architecture, which crashed on llama (no `q_norm`) and — had it got that far — would
# have silently computed GELU where llama and Qwen 3 want SwiGLU. Ministral 3 appears for
# its MLP only; its attention takes an extra `attn_scale` argument and is not fused.
_ATTENTION_BODIES = {
    "mlx_lm.models.gemma3_text": _gemma3_attention,
    "mlx_lm.models.llama": _llama_attention,
    "mlx_lm.models.qwen3": _qwen3_attention,
}
_MLP_BODIES = {
    "mlx_lm.models.gemma3_text": _gelu_mlp,
    "mlx_lm.models.llama": _swiglu_mlp,
    "mlx_lm.models.qwen3": _swiglu_mlp,
    "mlx_lm.models.ministral3": _swiglu_mlp,
}


def _fused_class(base: type, body) -> type:
    return type(f"Fused{base.__name__}", (base,), {"__call__": body})


def fuse_projections(model, *, check_version: bool = True) -> int:
    """Rewrite every transformer block in place.

    Returns the number of projection groups replaced — two per block when both the
    attention and the MLP have a transcribed body. Raises when none has, so a knob can
    never report itself on while having rewritten nothing.
    """
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
        attention_body = _ATTENTION_BODIES.get(type(attn).__module__)
        mlp_body = _MLP_BODIES.get(type(mlp).__module__)
        if attention_body and all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            parts = [attn.q_proj, attn.k_proj, attn.v_proj]
            sizes = [_out_features(part) for part in parts]
            attn.qkv_proj = _concat_linear(parts)
            attn.qkv_splits = [sizes[0], sizes[0] + sizes[1]]
            for name in ("q_proj", "k_proj", "v_proj"):
                attn.pop(name)
            attention_class = attention_class or _fused_class(type(attn), attention_body)
            attn.__class__ = attention_class
            fused += 1
        if mlp_body and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj")):
            parts = [mlp.gate_proj, mlp.up_proj]
            mlp.gate_up_proj = _concat_linear(parts)
            mlp.gate_up_split = [_out_features(parts[0])]
            for name in ("gate_proj", "up_proj"):
                mlp.pop(name)
            mlp_class = mlp_class or _fused_class(type(mlp), mlp_body)
            mlp.__class__ = mlp_class
            fused += 1
    if not fused:
        # Fail closed rather than leave a knob that reports "on" and rewrote nothing.
        raise FusionUnsupported(
            f"no verified fused body for attention {type(layers[0].self_attn).__module__} "
            f"or MLP {type(layers[0].mlp).__module__}")
    return fused


def _self_check() -> None:
    """Fusion must be bit identical. Runs on CPU so it never competes for the GPU.

    The default device is process global, so it is put back before returning: leaving it
    on the CPU changes what everything after this computes on.
    """
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        _self_check_body()
    finally:
        mx.set_default_device(previous)


def _toy_models() -> list[tuple[str, Any, int]]:
    """One tiny quantised model per verified family, plus one that must be refused."""
    from mlx_lm.models import gemma3_text, llama, qwen2, qwen3

    common = dict(hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128)
    return [
        ("gemma3_text", gemma3_text.Gemma3Model(gemma3_text.ModelArgs(
            model_type="gemma3_text", sliding_window=8, sliding_window_pattern=2, **common)), 8),
        ("llama", llama.LlamaModel(llama.ModelArgs(
            model_type="llama", rms_norm_eps=1e-5, **common)), 8),
        ("qwen3", qwen3.Qwen3Model(qwen3.ModelArgs(
            model_type="qwen3", rms_norm_eps=1e-5, max_position_embeddings=256,
            rope_theta=10000.0, tie_word_embeddings=True, **common)), 8),
        # Unlisted on purpose. Qwen 2's bodies are close enough to Qwen 3's to be tempting
        # and different enough to be wrong — its attention carries projection biases and
        # applies no QK norm. Fusion must refuse a body it was not transcribed from.
        ("qwen2", qwen2.Qwen2Model(qwen2.ModelArgs(
            model_type="qwen2", rms_norm_eps=1e-5, max_position_embeddings=256,
            **{k: v for k, v in common.items() if k != "head_dim"})), 0),
    ]


def _self_check_body() -> None:
    tokens = mx.array([[3, 9, 27, 81, 5, 6]])
    for name, model, expected in _toy_models():
        nn.quantize(model, group_size=32, bits=4)
        mx.eval(model.parameters())
        reference = model(tokens)
        mx.eval(reference)

        if expected == 0:
            try:
                fuse_projections(model, check_version=False)
            except FusionUnsupported:
                print(f"fast self-check ok: {name} refused, projections untouched")
                continue
            raise AssertionError(f"{name} must not be fused by a body it was not transcribed from")

        assert fuse_projections(model, check_version=False) == expected, name
        fused = model(tokens)
        mx.eval(fused)
        assert fused.shape == reference.shape, (name, fused.shape, reference.shape)
        assert mx.array_equal(fused, reference).item(), f"fusion changed the numbers on {name}"
        for block in model.layers:
            assert not hasattr(block.self_attn, "q_proj"), f"unfused projection left on {name}"
            assert not hasattr(block.mlp, "gate_proj"), f"unfused projection left on {name}"
        print(f"fast self-check ok: {name} 4 blocks fused, output bit identical")


if __name__ == "__main__":
    _self_check()
