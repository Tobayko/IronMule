"""CPU6 step 2: the stacked decode engine on a CPU, up to eight requests in one decode step (BACKLOG CPU6).

The dense plan's slot pool and executor (`ironmule/dense_plan.py`'s `DenseBackend` and `DenseStackedExecutor`)
with a decode step built for MLX's CPU device instead of CUDA kernels:

* every weight in float32 (a quantised checkpoint is dequantized at install, a bfloat16 one widened), so each
  linear layer is one float32 BLAS product, the fast path with `--cpu-blas openblas` (step 1);
* every linear layer at a fixed width of `rows` activation rows, the pool's slot count: fewer requests are padded
  with zero rows. A product's row then always comes from the same BLAS call shape, so a request's answer does not
  depend on how many others share the step (OpenBLAS is not row-invariant across widths: at K=1536 a row alone and
  the same row among two to eight differed in 35 of 36 placements; at a fixed width of eight none of 180 did);
* attention row by row over the request's own slot, read only up to its position, in float32;
* the model's own norms and rotary embedding, which act on each row alone.

Prefill runs the model's own layers on the float32 weights, one request at a time. Llama-shaped models (Llama,
Qwen 2, Qwen 3, SmolLM2) are served. The plan changes answers against the reference engine (float32 weights and
another summation), so it is opt-in like every numeric plan, and `install` probes the model's own weights first:
every stacked width bit-identical to one row, and the step within tolerance of mlx-lm's own forward.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from typing import Any, Sequence

MAX_ROWS = 8
PLAN_VERSION = 1
#: slot sizes tried from the largest; the pool of `slots` of them and the weights take at most `MEMORY_SHARE` of
#: the machine's memory
SLOT_TIERS = (8192, 4096, 2048, 1024, 512)
MEMORY_SHARE = 0.5
#: the step against mlx-lm's own forward on the same float32 weights (relative to the largest logit)
REFERENCE_TOLERANCE = 1e-3
FAMILIES = ("llama", "qwen2", "qwen3")


def _trunk(model: Any) -> Any:
    from .runtime import _trunk as trunk

    return trunk(model)


def refuse(model: Any) -> str | None:
    """Why the CPU plan cannot serve `model`, or None."""
    args = getattr(model, "args", None)
    family = getattr(args, "model_type", None)
    if family not in FAMILIES:
        return f"the CPU stacked plan serves {', '.join(FAMILIES)} checkpoints only"
    try:
        layer = _trunk(model).layers[0]
    except (AttributeError, IndexError, TypeError):
        return "the CPU stacked plan found no decoder layers"
    attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
    if attn is None or mlp is None or not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
        return "the CPU stacked plan serves gated (SwiGLU) decoder layers only"
    if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "rope", "n_heads",
                                                "n_kv_heads", "scale")):
        return "the CPU stacked plan does not know this attention layer"
    if getattr(args, "sliding_window", None) and getattr(args, "use_sliding_window", False):
        return "the CPU stacked plan covers full attention only"
    return None


def to_float32(model: Any) -> int:
    """Quantised layers dequantized to float32 (`numeric_plans.dequantize_model`), every other floating parameter
    widened to float32; returns how many layers were dequantized."""
    import mlx.core as mx

    from .numeric_plans import dequantize_model

    expanded = dequantize_model(model)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return expanded


def memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def slot_capacity(model: Any, slots: int, total: int | None = None, tiers: Sequence[int] = SLOT_TIERS) -> int | None:
    """The largest of `tiers` whose float32 pool fits beside the weights in `MEMORY_SHARE` of the memory."""
    from mlx.utils import tree_flatten

    total = memory_bytes() if total is None else total
    if total is None:
        return tiers[-1]
    layers = _trunk(model).layers
    attn = layers[0].self_attn
    head_dim = attn.k_proj.weight.shape[0] // attn.n_kv_heads
    weights = sum(value.size * 4 for _, value in tree_flatten(model.parameters()))
    per_token = 2 * len(layers) * attn.n_kv_heads * head_dim * 4
    return next((tier for tier in tiers if weights + slots * tier * per_token <= MEMORY_SHARE * total), None)


class CpuDensePlan:
    """The decode step over a pool of `slots` fixed-capacity float32 KV slots; rows (B <= slots) name their slot
    and position, and the same forward serves one row or all of them."""

    def __init__(self, model: Any, *, slots: int = MAX_ROWS, capacity: int = 2048):
        import mlx.core as mx

        refusal = refuse(model)
        if refusal:
            raise ValueError(refusal)
        if not 1 <= slots <= MAX_ROWS:
            raise ValueError(f"slots must be from 1 to {MAX_ROWS}")
        self.model, self.trunk, self.args = model, _trunk(model), model.args
        attn = self.trunk.layers[0].self_attn
        self.heads, self.kv_heads = attn.n_heads, attn.n_kv_heads
        self.dim = attn.k_proj.weight.shape[0] // self.kv_heads
        self.eps = getattr(self.args, "rms_norm_eps", 1e-6)
        self.slots, self.capacity, self.rows = slots, capacity, slots
        self.qk_norm = hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
        head = getattr(model, "lm_head", None)
        self.head = (head.weight if head is not None and not getattr(self.args, "tie_word_embeddings", False)
                     else self.trunk.embed_tokens.weight)
        rows = slots * self.kv_heads * capacity
        self.keys = [mx.zeros((rows, self.dim), mx.float32) for _ in self.trunk.layers]
        self.values = [mx.zeros((rows, self.dim), mx.float32) for _ in self.trunk.layers]
        mx.eval(self.keys, self.values)

    def linear(self, x: Any, w: Any, bias: Any = None) -> Any:
        """`x @ w.T` (+ bias) with x's rows padded to `self.rows`, so every product has one shape."""
        import mlx.core as mx

        m = x.shape[0]
        if m > self.rows:
            raise ValueError(f"the CPU plan stacks at most {self.rows} rows, got {m}")
        padded = x if m == self.rows else mx.concatenate([x, mx.zeros((self.rows - m, x.shape[1]), x.dtype)])
        out = (padded @ w.T)[:m]
        return out if bias is None else out + bias

    def _proj(self, x: Any, layer: Any) -> Any:
        return self.linear(x, layer.weight, getattr(layer, "bias", None))

    def _rows(self, slots: Any, positions: Any) -> Any:
        import mlx.core as mx

        heads = mx.arange(self.kv_heads, dtype=mx.int32)
        rows = (slots.astype(mx.int32)[:, None] * self.kv_heads + heads[None, :]) * self.capacity
        return (rows + positions.astype(mx.int32)[:, None]).reshape(-1)

    def write_prefill(self, slot: int, states: Sequence[tuple[Any, Any]]) -> None:
        """One request's prefilled keys and values (mlx-lm's cache state, (1, Hkv, L, D) per layer) into `slot`."""
        import mlx.core as mx

        length = states[0][0].shape[2]
        if length > self.capacity:
            raise ValueError(f"a {length}-token prefill does not fit a {self.capacity}-token slot")
        heads = mx.arange(self.kv_heads, dtype=mx.int32)[:, None]
        rows = ((slot * self.kv_heads + heads) * self.capacity + mx.arange(length, dtype=mx.int32)[None, :]).reshape(-1)
        for i, (k, v) in enumerate(states):
            self.keys[i][rows] = k[..., :length, :].reshape(-1, self.dim).astype(mx.float32)
            self.values[i][rows] = v[..., :length, :].reshape(-1, self.dim).astype(mx.float32)
        mx.eval(self.keys, self.values)

    def prefill(self, ids: Any, cache: list[Any]) -> Any:
        """The model's own prefill of one prompt (1, L) into an mlx-lm `cache`; its last position through the
        plan's head at the plan's width, as every decode step's. Float32 logits (1, V)."""
        import mlx.core as mx

        hidden = self.trunk(ids, cache=cache)
        mx.eval([c.state for c in cache])
        return self.linear(hidden[0, -1:, :], self.head)

    def _attend(self, q: Any, layer_index: int, slot: int, length: int, scale: float) -> Any:
        """One row's query (H, D) against its slot's first `length` positions."""
        import mlx.core as mx

        pool_k = self.keys[layer_index].reshape(self.slots, self.kv_heads, self.capacity, self.dim)
        pool_v = self.values[layer_index].reshape(self.slots, self.kv_heads, self.capacity, self.dim)
        k, v = pool_k[slot, :, :length], pool_v[slot, :, :length]  # (Hkv, L, D)
        grouped = q.reshape(self.kv_heads, self.heads // self.kv_heads, self.dim)  # (Hkv, G, D)
        scores = (grouped @ k.transpose(0, 2, 1)) * scale  # (Hkv, G, L)
        return (mx.softmax(scores, axis=-1) @ v).reshape(self.heads, self.dim)

    def forward(self, tokens: Any, slots: Any, positions: Any) -> Any:
        """Tokens (B,) at positions (B,) of slots (B,); writes their KV and returns float32 logits (B, V)."""
        import mlx.core as mx
        import mlx.nn as nn

        rows = tokens.shape[0]
        host_slots, host_positions = slots.tolist(), positions.tolist()
        if len(set(host_slots)) != rows:
            raise ValueError("each stacked row needs its own slot")
        if max(host_positions) >= self.capacity:
            raise ValueError("a row's position is past its slot")
        h = self.trunk.embed_tokens(tokens.astype(mx.uint32)).astype(mx.float32)
        written = self._rows(slots, positions)
        for i, layer in enumerate(self.trunk.layers):
            attn, mlp = layer.self_attn, layer.mlp
            x = mx.fast.rms_norm(h, layer.input_layernorm.weight, self.eps)
            q = self._proj(x, attn.q_proj).reshape(rows, self.heads, self.dim)
            k = self._proj(x, attn.k_proj).reshape(rows, self.kv_heads, self.dim)
            v = self._proj(x, attn.v_proj).reshape(rows, self.kv_heads, self.dim)
            if self.qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = self._rope(attn, q, positions), self._rope(attn, k, positions)
            self.keys[i][written] = k.reshape(-1, self.dim)
            self.values[i][written] = v.reshape(-1, self.dim)
            o = mx.stack([self._attend(q[r], i, host_slots[r], host_positions[r] + 1, attn.scale)
                          for r in range(rows)])
            h = h + self._proj(o.reshape(rows, self.heads * self.dim), attn.o_proj)
            x = mx.fast.rms_norm(h, layer.post_attention_layernorm.weight, self.eps)
            h = h + self._proj(nn.silu(self._proj(x, mlp.gate_proj)) * self._proj(x, mlp.up_proj), mlp.down_proj)
        h = mx.fast.rms_norm(h, self.trunk.norm.weight, self.eps)
        return self.linear(h, self.head)

    def _rope(self, attn: Any, x: Any, positions: Any) -> Any:
        """The layer's own rotary embedding, each row at its own position (one call, per-row offsets)."""
        import mlx.core as mx

        rows, heads, dim = x.shape
        return attn.rope(x.reshape(rows, heads, 1, dim), offset=positions.astype(mx.int32)).reshape(rows, heads, dim)


SOURCE_SHA256 = hashlib.sha256(inspect.getsource(CpuDensePlan).encode()).hexdigest()


def probe(plan: CpuDensePlan, steps: int = 3) -> dict[str, Any]:
    """Every stacked width bit-identical to one row (distinct prompts in distinct slots, `steps` decode steps
    each); and the plan's step against mlx-lm's own forward on the same weights."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    rows = plan.slots
    prompts = [[(r * 977 + j * 131 + 11) % plan.args.vocab_size for j in range(3 + r)] for r in range(rows)]

    def run(members: list[int]) -> dict[int, list[Any]]:
        caches, firsts = {}, {}
        for r in members:
            caches[r] = make_prompt_cache(plan.model)
            firsts[r] = mx.argmax(plan.prefill(mx.array(prompts[r])[None], caches[r]), axis=-1)
            plan.write_prefill(r, [c.state for c in caches[r]])
        tokens = mx.concatenate([firsts[r] for r in members]).astype(mx.uint32)
        out: dict[int, list[Any]] = {r: [] for r in members}
        for k in range(steps):
            logits = plan.forward(tokens, mx.array(members, mx.int32),
                                  mx.array([len(prompts[r]) + k for r in members], mx.int32))
            mx.eval(logits)
            for i, r in enumerate(members):
                out[r].append(logits[i])
            tokens = mx.argmax(logits, axis=-1).astype(mx.uint32)
        return out

    alone = {r: run([r])[r] for r in range(rows)}
    identical = True
    for width in range(2, rows + 1):
        stacked = run(list(range(width)))
        identical = identical and all(bool(mx.array_equal(a, b).item())
                                      for r in range(width) for a, b in zip(stacked[r], alone[r]))
    # mlx-lm's own forward: the same prompt, one token more, on the model's (float32) layers
    prompt = prompts[0]
    cache = make_prompt_cache(plan.model)
    first = mx.argmax(plan.prefill(mx.array(prompt)[None], cache), axis=-1)
    plan.write_prefill(0, [c.state for c in cache])
    mine = plan.forward(first.astype(mx.uint32), mx.array([0], mx.int32), mx.array([len(prompt)], mx.int32))[0]
    reference = plan.model(first.reshape(1, 1), cache=cache)[0, -1].astype(mx.float32)
    error = mx.max(mx.abs(mine - reference)) / mx.maximum(mx.max(mx.abs(reference)), 1e-6)
    return {"stacked_rows_identical": identical, "reference_rel_error": float(error.item()),
            "top1_equal": bool((mx.argmax(mine) == mx.argmax(reference)).item())}


def install(model: Any, *, slots: int = MAX_ROWS, capacity: int | None = None) -> tuple[CpuDensePlan, dict[str, Any]]:
    """Widen `model` to float32 in place, build the plan and probe it, or refuse."""
    refusal = refuse(model)
    if refusal:
        raise ValueError(refusal)
    dequantized = to_float32(model)
    capacity = slot_capacity(model, slots) if capacity is None else capacity
    if capacity is None:
        raise ValueError("the CPU stacked plan's smallest slot pool does not fit this machine's memory")
    plan = CpuDensePlan(model, slots=slots, capacity=capacity)
    record = probe(plan)
    if not record["stacked_rows_identical"]:
        raise ValueError("the CPU stacked plan's stacked rows differ from one row on this model")
    if not record["reference_rel_error"] <= REFERENCE_TOLERANCE:
        raise ValueError(f"the CPU stacked plan disagrees with mlx-lm's forward ({record['reference_rel_error']:.3g})")
    return plan, {**record, "kernels": "cpu_float32", "source_sha256": SOURCE_SHA256, "plan_version": PLAN_VERSION,
                  "slots": slots, "capacity": capacity, "dequantized_layers": dequantized,
                  "model_family": model.args.model_type}


__all__ = ["CpuDensePlan", "FAMILIES", "MAX_ROWS", "PLAN_VERSION", "REFERENCE_TOLERANCE", "SLOT_TIERS", "install",
           "probe", "refuse", "slot_capacity", "to_float32"]
