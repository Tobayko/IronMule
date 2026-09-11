"""Research mode: two ready requests share one weight sweep. Off unless asked for.

This is not a second server. It subclasses the shipped grouped executor, so admission,
session handling, stop rules, telemetry and the sequential fallback are the existing
ones; only the group's inner step changes. With `share=False` it is the shipped
scheduling with separate projections, which is the control that tells scheduling apart
from weight reuse.

Sharing is confined to what `B42`, `B43` and `B45` qualified: the `K=3840` projections of
this Gemma 12B revision on this machine, at single-token decode. Inputs, partial sums,
attention and KV state stay per request. Prefill is untouched. A pair forms only from
requests that are already ready; nothing waits for a partner, and a lone request runs the
ordinary single path.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .executor import AsyncGroupedB1Executor, Session
from .qmv_k3840 import (
    RESULTS_PER_SIMDGROUP,
    NUM_SIMDGROUPS,
    K3840Unsupported,
    TARGET_K,
    VERIFIED_ARCHITECTURE,
    VERIFIED_BITS,
    VERIFIED_GROUP_SIZE,
    VERIFIED_IDENTITY_SHA256,
    VERIFIED_MLX,
    VERIFIED_MLX_LM,
    _admit_projection,
)
from .runtime import _caches_from_state, _project, _state_from_caches, _text, _trunk

PAIR_WIDTH = 2
ALIGNED_K = (4096, 15360)  # o_proj and down_proj: the qmv_fast shapes


def _shared_kernel():
    """Imported lazily: the research kernel must not load unless this mode runs."""

    from .qmv_shared import run_shared  # noqa: PLC0415

    return run_shared


def _fast_shared_kernel():
    from .qmv_fast_shared import run_fast  # noqa: PLC0415

    return run_fast


def _aligned(module) -> int | None:
    """The reduction length if this projection is one of the aligned families."""

    columns = int(module.weight.shape[1]) * 32 // int(module.bits)
    if columns not in ALIGNED_K:
        return None
    if int(module.bits) != VERIFIED_BITS or int(module.group_size) != VERIFIED_GROUP_SIZE:
        return None
    if int(module.weight.shape[0]) % (NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP):
        return None
    if str(module.scales.dtype).rsplit(".", 1)[-1] != "bfloat16":
        return None
    return columns


def _project_aligned(module, vectors, *, share: bool):
    """The same contract as `_project_pair`, on the aligned kernel."""

    run_fast = _fast_shared_kernel()
    columns = _aligned(module)
    out_features = int(module.weight.shape[0])
    flat = [v.reshape(1, columns) for v in vectors]
    if share and len(flat) == PAIR_WIDTH:
        results = run_fast(module.weight, module.scales, module.biases, flat,
                           out_features, columns)
    else:
        results = [
            run_fast(module.weight, module.scales, module.biases, [v],
                     out_features, columns)[0]
            for v in flat
        ]
    return [r.reshape(1, 1, out_features) for r in results]


def admit(model: Any, identity: Any) -> dict[str, Any]:
    """The same gate the opt-in kernel uses, plus the shapes this mode needs.

    Raises rather than degrading: a caller who asked for the research mode must not
    silently get something else.
    """

    import mlx_lm  # noqa: PLC0415

    from .hw import fingerprint  # noqa: PLC0415
    from .qmv_k3840 import VERIFIED_HARDWARE_FINGERPRINT  # noqa: PLC0415

    checks = {
        "hardware": fingerprint() == VERIFIED_HARDWARE_FINGERPRINT,
        "mlx": mx.__version__ == VERIFIED_MLX,
        "mlx_lm": mlx_lm.__version__ == VERIFIED_MLX_LM,
        "identity": getattr(identity, "identity_sha256", None) == VERIFIED_IDENTITY_SHA256,
        "architecture": getattr(identity, "architecture", None) == VERIFIED_ARCHITECTURE,
    }
    if not all(checks.values()):
        refused = sorted(name for name, ok in checks.items() if not ok)
        raise K3840Unsupported(f"paired research mode not admitted here: {refused}")

    text = _text(model)
    admitted, declined = [], []
    for index, layer in enumerate(text.model.layers):
        for owner, name in (
            (layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
            (layer.self_attn, "v_proj"), (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
        ):
            module = getattr(owner, name)
            columns = int(module.weight.shape[1]) * 32 // int(module.bits)
            target = admitted if (columns == TARGET_K and _admit_projection(module)) else declined
            target.append(f"layers.{index}.{name}")
    if not admitted:
        raise K3840Unsupported("no projection met the admission conditions")
    return {
        "admitted": len(admitted),
        "declined": len(declined),
        "target_k": TARGET_K,
        "bits": VERIFIED_BITS,
        "group_size": VERIFIED_GROUP_SIZE,
        "identity_sha256": VERIFIED_IDENTITY_SHA256,
    }


def _project_pair(module, vectors: list[mx.array], *, share: bool) -> list[mx.array]:
    run_shared = _shared_kernel()
    out_features = int(module.weight.shape[0])
    flat = [v.reshape(1, TARGET_K) for v in vectors]
    if share and len(flat) == PAIR_WIDTH:
        results = run_shared(module.weight, module.scales, module.biases, flat, out_features)
    else:
        results = [
            run_shared(module.weight, module.scales, module.biases, [v], out_features)[0]
            for v in flat
        ]
    return [r.reshape(1, 1, out_features) for r in results]


def _attention(layer, xs, masks, caches, *, share: bool, share_aligned: bool = False):
    from mlx_lm.models.base import scaled_dot_product_attention  # noqa: PLC0415

    attn = layer.self_attn
    queries = _project_pair(attn.q_proj, xs, share=share)
    keys = _project_pair(attn.k_proj, xs, share=share)
    values = _project_pair(attn.v_proj, xs, share=share)
    outputs = []
    for index in range(len(xs)):
        B, L, _ = xs[index].shape
        q = attn.q_norm(queries[index].reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3))
        k = attn.k_norm(keys[index].reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3))
        v = values[index].reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        cache = caches[index]
        q = attn.rope(q, offset=cache.offset)
        k = attn.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=attn.scale, mask=masks[index]
        )
        outputs.append(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
    if share_aligned and _aligned(attn.o_proj) is not None:
        return _project_aligned(attn.o_proj, outputs, share=share)
    return [attn.o_proj(value) for value in outputs]


def paired_forward(model, token_arrays: list[mx.array], caches_per_request: list[list],
                   *, share: bool, share_aligned: bool = False):
    """One decode step for several requests. Only admitted projections are shared."""

    from mlx_lm.models.base import create_attention_mask  # noqa: PLC0415
    from mlx_lm.models.gemma3_text import clip_residual  # noqa: PLC0415

    text = _text(model)
    inner = _trunk(model)
    hs = []
    for token in token_arrays:
        h = inner.embed_tokens(token)
        hs.append(h * mx.array(inner.args.hidden_size**0.5, mx.bfloat16).astype(h.dtype))
    masks = []
    for caches in caches_per_request:
        global_mask = create_attention_mask(hs[0], caches[inner.sliding_window_pattern - 1])
        sliding = (
            create_attention_mask(hs[0], caches[0], window_size=inner.window_size)
            if inner.sliding_window_pattern > 1 else None
        )
        masks.append((global_mask, sliding))
    for index, layer in enumerate(inner.layers):
        is_global = index % inner.sliding_window_pattern == inner.sliding_window_pattern - 1
        step_masks = [pair[0] if is_global else pair[1] for pair in masks]
        caches = [per_request[index] for per_request in caches_per_request]
        normed = [layer.input_layernorm(h) for h in hs]
        attended = _attention(layer, normed, step_masks, caches, share=share,
                              share_aligned=share_aligned)
        hs = [clip_residual(h, layer.post_attention_layernorm(r))
              for h, r in zip(hs, attended)]
        pre = [layer.pre_feedforward_layernorm(h) for h in hs]
        gate = _project_pair(layer.mlp.gate_proj, pre, share=share)
        up = _project_pair(layer.mlp.up_proj, pre, share=share)
        activated = [nn.gelu_approx(g) * u for g, u in zip(gate, up)]
        if share_aligned and _aligned(layer.mlp.down_proj) is not None:
            fed = _project_aligned(layer.mlp.down_proj, activated, share=share)
        else:
            fed = [layer.mlp.down_proj(value) for value in activated]
        hs = [clip_residual(h, layer.post_feedforward_layernorm(r)) for h, r in zip(hs, fed)]
    return [_project(model, inner.norm(h)) for h in hs]


class PairedBackend:
    """Wraps the shipped backend; only the grouped step is new."""

    def __init__(self, backend, *, share: bool, share_aligned: bool = False) -> None:
        self._backend = backend
        self.share = share
        self.share_aligned = share_aligned

    def __getattr__(self, name):  # everything else is the shipped behaviour
        return getattr(self._backend, name)

    def step_pair(self, states: list[dict], tokens: list[int], capacity: int):
        model = self._backend.engine.model
        caches_per_request = [_caches_from_state(state, capacity) for state in states]
        token_arrays = [mx.array([[token]]) for token in tokens]
        logits = paired_forward(model, token_arrays, caches_per_request,
                                share=self.share, share_aligned=self.share_aligned)
        handles = []
        for state, caches, out in zip(states, caches_per_request, logits):
            new_state = _state_from_caches(
                caches, {"offset": state["position"]["offset"] + 1}
            )
            pick = mx.argmax(out[:, -1, :].astype(mx.float32), axis=-1)
            handles.append(((out, new_state), pick))
        return handles


class PairedGroupedExecutor(AsyncGroupedB1Executor):
    """The shipped scheduler, with pairs of ready requests stepped together.

    `share=False` keeps every projection separate, so a comparison against it isolates
    what the shared weight load is worth from what the pairing itself is worth.
    """

    name = "paired_grouped_b1"

    def __init__(self, backend, telemetry, *, share: bool, share_aligned: bool = False,
                 max_width: int = PAIR_WIDTH):
        super().__init__(PairedBackend(backend, share=share, share_aligned=share_aligned),
                         telemetry, max_width=max_width)
        self.paired_steps = 0
        self.solo_steps = 0

    def _step_group(self, group: list[Session], capacity: int):
        # A pair forms only from requests already at a step boundary. A lone request is
        # served immediately on the single path; nothing waits for a partner.
        if len(group) != PAIR_WIDTH:
            self.solo_steps += len(group)
            return [self.backend.step(s.state, s.tokens[-1], capacity) for s in group]
        self.paired_steps += 1
        return self.backend.step_pair(
            [s.state for s in group], [s.tokens[-1] for s in group], capacity
        )
