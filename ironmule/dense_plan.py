"""The `dense` numeric plan: up to eight requests stacked into one decode step on pre-Ampere CUDA (BACKLOG KB1).

Below compute capability 8 a bfloat16 checkpoint's matmuls run without bfloat16 arithmetic, and MLX's product at
2 to 32 rows is 5-7x slower than at one row and not bit-identical to it (TB1). This plan replaces the decode step
of an unquantized Qwen 3 model with IronMule's own kernels:

* every linear layer through a row kernel for up to eight activation rows (`ironmule/cuda_native.py`'s 4-bit row
  kernel without the dequantisation): each weight is read once for every row, and each row keeps its own float32
  chain with pinned roundings, so a row's answer does not depend on how many others share the step;
* attention through FA2, which reads a request's own slot of a fixed-capacity KV pool only up to its position,
  in 64-key chunks at absolute positions combined in index order, so neither the pool's capacity nor the other
  rows move a request's answer;
* MLX's own RoPE (one position per row), `rms_norm` and `silu`, which act on each row alone.

Prefill stays MLX's own, one request at a time. The plan changes answers against today's engine, so it is
opt-in like every numeric plan, and it is installed only on a model whose own weights pass a load-time probe:
one, three and eight stacked rows bit-identical, and the row kernel within tolerance of a float32 reference.
KB1-1 measured it on a Tesla T4 with Qwen 3 1.7B bf16: exact within the plan, eight requests 3.8x faster than
today's grouping, the numeric plans' quality gate passed on the decode path; one request alone 18 % slower.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import threading
from typing import Any, Callable, Sequence

MAX_ROWS = 8
FA2_CHUNK = 64
PROBE_TOLERANCE = 1e-2

DENSE_HEADER = r"""
__device__ __forceinline__ float lo_bf(unsigned int u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_bf(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ unsigned short to_bf(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
"""

# One warp per R output rows and M <= 8 activation rows of a bfloat16 weight; each lane walks K in 8-element
# chunks, every (row, activation row) pair keeps its own float32 chain with pinned roundings, then one shuffle
# tree. A row's arithmetic depends on neither M nor R.
DENSE_SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int row0 = (gid >> 5) * R;
  const unsigned int lane = threadIdx.x & 31;
  if (row0 >= N) return;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  float acc[R * M];
#pragma unroll
  for (int i = 0; i < R * M; ++i) acc[i] = 0.f;
  for (unsigned int c = lane; c < K / 8; c += 32) {
    float xf[M * 8];
#pragma unroll
    for (int mo = 0; mo < M; ++mo) {
      const uint4 xa = xv[(size_t)mo * (K / 8) + c];
      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        xf[mo * 8 + 2 * e] = lo_bf(xw[e]);
        xf[mo * 8 + 2 * e + 1] = hi_bf(xw[e]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const unsigned int row = row0 + r;
      if (row < N) {
        const uint4 q = w4[(size_t)row * (K / 8) + c];
        const unsigned int qw[4] = {q.x, q.y, q.z, q.w};
        float wf[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          wf[2 * e] = lo_bf(qw[e]);
          wf[2 * e + 1] = hi_bf(qw[e]);
        }
#pragma unroll
        for (int mo = 0; mo < M; ++mo) {
          float d = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) d = __fmaf_rn(wf[i], xf[mo * 8 + i], d);
          acc[r * M + mo] = __fadd_rn(acc[r * M + mo], d);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R * M; ++i) {
    float a = acc[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = __fadd_rn(a, __shfl_down_sync(0xffffffffu, a, o));
    const unsigned int row = row0 + i / M;
    if (lane == 0 && row < N) out[(i % M) * N + row] = to_bf(a);
  }
"""

FA2_HEADER = DENSE_HEADER + r"""
__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
#define NEG_INF __int_as_float(0xff800000)
"""

# KB1-1's FA2 with one change: a row reads its own slot of the pool (`slots[b]`) instead of pool row `b`. The
# loads and their order are the same, so a row's arithmetic is unchanged.
FA2_PARTIAL = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * HKV * NC) return;
  const unsigned int c = warp % NC;
  const unsigned int b = (warp / NC) / HKV;
  const unsigned int kh = (warp / NC) % HKV;
  const int len = lengths[b];
  const int start = c * 64;
  if (start >= len) return;
  const unsigned short* qs = reinterpret_cast<const unsigned short*>(q);
  const unsigned short* ks = reinterpret_cast<const unsigned short*>(k);
  const unsigned short* vs = reinterpret_cast<const unsigned short*>(v);
  const size_t base = ((size_t)slots[b] * HKV + kh) * C * D;
  const int count = min(64, len - start);
  for (int g = 0; g < G; ++g) {
    const size_t qo = ((size_t)b * H + kh * G + g) * D;
    float s[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int t = start + j * 32 + (int)lane;
      if (t < len) {
        const unsigned short* kr = ks + base + (size_t)t * D;
        float d = 0.f;
        for (int i = 0; i < D; ++i) d = __fmaf_rn(one_bf(qs[qo + i]), one_bf(kr[i]), d);
        s[j] = __fmul_rn(d, SCALE);
      } else {
        s[j] = NEG_INF;
      }
    }
    float m = fmaxf(s[0], s[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float pr[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) pr[j] = (s[j] == NEG_INF) ? 0.f : expf(s[j] - m);
    float sum = __fadd_rn(pr[0], pr[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, o));
    sum = __shfl_sync(0xffffffffu, sum, 0);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t = 0; t < count; ++t) {
      const float pt = __shfl_sync(0xffffffffu, pr[t >> 5], t & 31);
      const unsigned short* vr = vs + base + (size_t)(start + t) * D + lane * 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(pt, one_bf(vr[e]), acc[e]);
    }
    const size_t po = ((size_t)b * H + kh * G + g) * NC + c;
#pragma unroll
    for (int e = 0; e < 4; ++e) pacc[po * D + lane * 4 + e] = acc[e];
    if (lane == 0) {
      pml[po * 2] = m;
      pml[po * 2 + 1] = sum;
    }
  }
"""

FA2_COMBINE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * H) return;
  const unsigned int b = warp / H;
  const unsigned int h = warp % H;
  const int nc = (lengths[b] + 63) / 64;
  float M = NEG_INF, L = 0.f;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < nc; ++c) {
    const size_t po = ((size_t)b * H + h) * NC + c;
    const float mc = pml[po * 2];
    const float lc = pml[po * 2 + 1];
    const float mn = fmaxf(M, mc);
    const float a = expf(M - mn);
    const float bb = expf(mc - mn);
    L = __fmaf_rn(L, a, __fmul_rn(lc, bb));
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[e] = __fmaf_rn(acc[e], a, __fmul_rn(pacc[po * D + lane * 4 + e], bb));
    M = mn;
  }
#pragma unroll
  for (int e = 0; e < 4; ++e) out[((size_t)b * H + h) * D + lane * 4 + e] = to_bf(__fdiv_rn(acc[e], L));
"""

#: What the kernels' answers are bound to: a changed source is a changed plan.
KERNEL_SHA256 = hashlib.sha256((DENSE_SOURCE + FA2_PARTIAL + FA2_COMBINE).encode()).hexdigest()

_kernels: dict[str, Any] = {}
_kernel_lock = threading.Lock()


def _kernel(name: str) -> Any:
    import mlx.core as mx

    with _kernel_lock:
        if name not in _kernels:
            if name == "dense":
                _kernels[name] = mx.fast.cuda_kernel(name="ironmule_dense_rows", input_names=["x", "w"],
                                                     output_names=["out"], source=DENSE_SOURCE, header=DENSE_HEADER)
            elif name == "fa2_partial":
                _kernels[name] = mx.fast.cuda_kernel(
                    name="ironmule_fa2_slots", input_names=["q", "k", "v", "lengths", "slots"],
                    output_names=["pacc", "pml"], source=FA2_PARTIAL.replace("SCALE", f"{128 ** -0.5!r}f"),
                    header=FA2_HEADER)
            else:
                _kernels[name] = mx.fast.cuda_kernel(name="ironmule_fa2_combine", input_names=["pacc", "pml", "lengths"],
                                                     output_names=["out"], source=FA2_COMBINE, header=FA2_HEADER)
        return _kernels[name]


def dense_rows(x: Any, w: Any, rows: int | None = None) -> Any:
    """`x @ w.T` for a bfloat16 weight (N, K) and up to eight rows of bfloat16 activations."""
    import mlx.core as mx

    n, k = w.shape
    m = x.size // k
    if not 1 <= m <= MAX_ROWS:
        raise ValueError(f"the row kernel takes 1 to {MAX_ROWS} rows, got {m}")
    rows = rows or (4 if n >= 8192 else 2)
    threads = -(-(-(-n // rows) * 32) // 256) * 256
    out = _kernel("dense")(inputs=[mx.contiguous(x.reshape(m, k)), w],
                           template=[("N", n), ("K", k), ("R", rows), ("M", m)],
                           grid=(threads, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(m, n)],
                           output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16).reshape(*x.shape[:-1], n)


def fa2(q: Any, keys: Any, values: Any, lengths: Any, slots: Any, kv_heads: int, capacity: int) -> Any:
    """One decode query per row (B, H, 128) against its own slot of a pool laid out (slots, kv_heads, capacity,
    128), flat or not, reading only the first `lengths[b]` positions."""
    import mlx.core as mx

    b, h, d = q.shape
    nc = -(-capacity // FA2_CHUNK)
    lengths, slots = lengths.astype(mx.int32), slots.astype(mx.int32)
    template = [("B", b), ("H", h), ("HKV", kv_heads), ("G", h // kv_heads), ("D", d), ("C", capacity), ("NC", nc)]
    pacc, pml = _kernel("fa2_partial")(
        inputs=[mx.contiguous(q), keys, values, lengths, slots], template=template,
        grid=(-(-(b * kv_heads * nc * 32) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b, h, nc, d), (b, h, nc, 2)], output_dtypes=[mx.float32, mx.float32])
    out = _kernel("fa2_combine")(
        inputs=[pacc, pml, lengths], template=template, grid=(-(-(b * h * 32) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(b, h, d)], output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16)


class DensePlan:
    """The plan's decode step over a pool of `slots` fixed-capacity KV slots: rows (B <= 8) name their slot and
    position, and the same forward serves one row or eight."""

    def __init__(self, model: Any, *, slots: int = MAX_ROWS, capacity: int = 2048,
                 linear: Callable[[Any, Any], Any] | None = None, attention: Callable[..., Any] | None = None):
        import mlx.core as mx

        from .runtime import _trunk

        args = model.args
        refusal = refuse(model)
        if refusal:
            raise ValueError(refusal)
        self.model, self.trunk, self.args = model, _trunk(model), args
        self.heads, self.kv_heads = args.num_attention_heads, args.num_key_value_heads
        self.dim, self.eps = args.head_dim, args.rms_norm_eps
        self.slots, self.capacity = slots, capacity
        self.linear = linear or dense_rows
        self.attention = attention or fa2
        head = getattr(model, "lm_head", None)
        self.head = head.weight if head is not None and not args.tie_word_embeddings else self.trunk.embed_tokens.weight
        rows = slots * self.kv_heads * capacity
        self.keys = [mx.zeros((rows, self.dim), mx.bfloat16) for _ in self.trunk.layers]
        self.values = [mx.zeros((rows, self.dim), mx.bfloat16) for _ in self.trunk.layers]
        mx.eval(self.keys, self.values)

    def _rows(self, slots: Any, positions: Any) -> Any:
        """Flat pool rows of (slot, every KV head, position), shape (B * kv_heads,)."""
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
            self.keys[i][rows] = k.reshape(-1, self.dim).astype(mx.bfloat16)
            self.values[i][rows] = v.reshape(-1, self.dim).astype(mx.bfloat16)
        mx.eval(self.keys, self.values)

    def forward(self, tokens: Any, slots: Any, positions: Any) -> Any:
        """Tokens (B,) at positions (B,) of slots (B,); writes their KV and returns float32 logits (B, V)."""
        import mlx.core as mx
        from mlx_lm.models.activations import swiglu

        rows = tokens.shape[0]
        h = self.trunk.embed_tokens(tokens)
        written = self._rows(slots, positions)
        lengths = positions + 1
        for i, layer in enumerate(self.trunk.layers):
            attn, mlp = layer.self_attn, layer.mlp
            x = mx.fast.rms_norm(h, layer.input_layernorm.weight, self.eps)
            q = mx.fast.rms_norm(self.linear(x, attn.q_proj.weight).reshape(rows, self.heads, self.dim),
                                 attn.q_norm.weight, self.eps)
            k = mx.fast.rms_norm(self.linear(x, attn.k_proj.weight).reshape(rows, self.kv_heads, self.dim),
                                 attn.k_norm.weight, self.eps)
            v = self.linear(x, attn.v_proj.weight).reshape(rows, self.kv_heads, self.dim)
            q, k = self._rope(q, positions), self._rope(k, positions)
            self.keys[i][written] = k.reshape(-1, self.dim)
            self.values[i][written] = v.reshape(-1, self.dim)
            o = self.attention(q, self.keys[i], self.values[i], lengths, slots, self.kv_heads, self.capacity)
            h = h + self.linear(o.reshape(rows, self.heads * self.dim), attn.o_proj.weight)
            x = mx.fast.rms_norm(h, layer.post_attention_layernorm.weight, self.eps)
            h = h + self.linear(swiglu(self.linear(x, mlp.gate_proj.weight), self.linear(x, mlp.up_proj.weight)),
                                mlp.down_proj.weight)
        h = mx.fast.rms_norm(h, self.trunk.norm.weight, self.eps)
        return self.linear(h, self.head).astype(mx.float32)

    def _rope(self, x: Any, positions: Any) -> Any:
        import mlx.core as mx

        rows, heads, dim = x.shape
        return mx.fast.rope(x.reshape(rows, heads, 1, dim), dim, traditional=False, base=self.args.rope_theta,
                            scale=1.0, offset=positions.astype(mx.int32)).reshape(rows, heads, dim)


def refuse(model: Any) -> str | None:
    """Why the plan cannot serve `model`, or None: Qwen 3, unquantized bfloat16, head dim 128, plain RoPE."""
    import mlx.core as mx
    import mlx.nn as nn

    args = getattr(model, "args", None)
    if getattr(args, "model_type", None) != "qwen3":
        return "the dense plan serves Qwen 3 checkpoints only"
    if getattr(args, "head_dim", None) != 128:
        return "the dense plan's attention kernel serves head dim 128 only"
    if getattr(args, "rope_scaling", None):
        return "the dense plan covers unscaled rotary embeddings only"
    if any(isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)) for _, m in model.named_modules()):
        return "the dense plan serves unquantized checkpoints; 4-bit ones have the native plan"
    from .runtime import _trunk

    if _trunk(model).layers[0].self_attn.q_proj.weight.dtype != mx.bfloat16:
        return "the dense plan serves bfloat16 weights only"
    return None


def install(model: Any, device_info: dict[str, Any] | None, *, slots: int = MAX_ROWS,
            capacity: int = 2048) -> tuple[DensePlan, dict[str, Any]]:
    """Build the plan for `model`, or refuse. The probe on the model's own weights must find one, three and eight
    stacked rows bit-identical, and the row kernel within tolerance of a float32 reference."""
    from .numeric_plans import CUDA_PRE_AMPERE, device_class

    if device_class(device_info) != CUDA_PRE_AMPERE:
        raise ValueError("the dense plan exists for CUDA GPUs below compute capability 8 only")
    plan = DensePlan(model, slots=slots, capacity=capacity)
    record = probe(plan)
    if not record["stacked_rows_identical"]:
        raise ValueError("the dense plan's stacked rows differ from one row on this model")
    if not record["row_kernel_rel_error"] <= PROBE_TOLERANCE:
        raise ValueError(f"the row kernel disagrees with the float32 reference ({record['row_kernel_rel_error']:.3g})")
    return plan, {**record, "kernel_sha256": KERNEL_SHA256, "slots": slots, "capacity": capacity}


def probe(plan: DensePlan) -> dict[str, Any]:
    """One step at distinct positions in distinct slots, at one, three and eight rows, compared bit for bit; and
    the q projection against a float32 reference."""
    import mlx.core as mx

    rows = min(MAX_ROWS, plan.slots)
    tokens = (mx.arange(rows, dtype=mx.uint32) * 977 + 11) % plan.args.vocab_size
    positions = (mx.arange(rows, dtype=mx.int32) * 37 + 3) % (plan.capacity - 1)  # inside each slot
    slots = mx.arange(rows, dtype=mx.int32)[::-1]
    alone = [plan.forward(tokens[i:i + 1], slots[i:i + 1], positions[i:i + 1]) for i in range(rows)]
    mx.eval(*alone)
    identical = True
    for width in sorted({1, min(3, rows), rows}):
        stacked = plan.forward(tokens[:width], slots[:width], positions[:width])
        mx.eval(stacked)
        identical = identical and all(bool(mx.array_equal(stacked[i], alone[i][0]).item()) for i in range(width))
    w = plan.trunk.layers[0].self_attn.q_proj.weight
    x = mx.random.normal((1, w.shape[1]), key=mx.random.key(0)).astype(mx.bfloat16)
    reference = x.astype(mx.float32) @ w.astype(mx.float32).T
    error = mx.max(mx.abs(plan.linear(x, w).astype(mx.float32) - reference)) / mx.maximum(mx.max(mx.abs(reference)), 1e-6)
    return {"stacked_rows_identical": identical, "row_kernel_rel_error": float(error.item())}


@dataclass
class DenseState:
    """A session's place in the pool: its slot and the position of its next token."""

    slot: int
    offset: int


@dataclass
class _Handle:
    logits: Any
    row: int
    state: DenseState
    pick: Any = field(default=None)


class DenseBackend:
    """The executor's `DecodeBackend` over a `DensePlan`: prefill writes a request's MLX prefill into a free slot;
    `step` is the plan's one-row forward and `step_group` its stacked forward, so a fallback or a sequential finish
    stays inside the plan; `reset_state` only moves the position back."""

    def __init__(self, plan: DensePlan, eos_ids: Sequence[int]):
        self.plan = plan
        self.eos_ids = tuple(eos_ids)
        self._free = list(range(plan.slots))
        self._lock = threading.Lock()

    def free_slots(self) -> int:
        with self._lock:
            return len(self._free)

    def capacity_for(self, prompt_lens: Sequence[int], max_tokens: int) -> int:
        needed = max(prompt_lens) + max_tokens
        if needed > self.plan.capacity:
            raise ValueError(f"{needed} tokens do not fit the dense plan's {self.plan.capacity}-token slots")
        return self.plan.capacity

    def prefill(self, prompt_ids: Sequence[int], plan: Any, capacity: int) -> tuple[DenseState, int]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        from .plans import plan_kind

        if plan_kind(plan) != "strict_one_shot":
            raise ValueError("the dense plan prefills one-shot only")
        with self._lock:
            if not self._free:
                raise RuntimeError("no free slot in the dense plan's pool")
            slot = self._free.pop(0)
        try:
            cache = make_prompt_cache(self.plan.model)
            logits = self.plan.model(mx.array(list(prompt_ids))[None], cache=cache)
            first = mx.argmax(logits[:, -1, :].astype(mx.float32), axis=-1)
            self.plan.write_prefill(slot, [c.state for c in cache])
            mx.eval(first)
        except BaseException:
            self.release(DenseState(slot, 0))
            raise
        return DenseState(slot, len(prompt_ids)), int(first.item())

    def reset_state(self, base_state: DenseState, offset: int) -> DenseState:
        return DenseState(base_state.slot, offset)

    def step(self, state: DenseState, token: int, capacity: int) -> _Handle:
        return self.step_group([state], [token])[0]

    def step_group(self, states: Sequence[DenseState], tokens: Sequence[int]) -> list[_Handle]:
        import mlx.core as mx

        logits = self.plan.forward(mx.array(list(tokens), mx.uint32), mx.array([s.slot for s in states], mx.int32),
                                   mx.array([s.offset for s in states], mx.int32))
        picks = mx.argmax(logits, axis=-1)
        return [_Handle(logits, i, DenseState(s.slot, s.offset + 1), picks[i]) for i, s in enumerate(states)]

    def complete(self, handles: Sequence[_Handle]) -> None:
        import mlx.core as mx

        mx.eval(*[h.pick for h in handles])

    def read(self, handle: _Handle) -> tuple[int, DenseState]:
        return int(handle.pick.item()), handle.state

    def release(self, state: Any) -> None:
        slot = getattr(state, "slot", None)
        if slot is None:
            return
        with self._lock:
            if slot not in self._free and 0 <= slot < self.plan.slots:
                self._free.append(slot)
                self._free.sort()


def _executor_base() -> type:
    from .executor import AsyncGroupedB1Executor

    return AsyncGroupedB1Executor


class DenseStackedExecutor(_executor_base()):  # type: ignore[misc]
    """Grouped serving where a group is one stacked step of up to eight rows. Slots go back to the pool as soon
    as a session is done, cancelled or the run ends."""

    name = "dense_stacked"

    def __init__(self, backend: DenseBackend, telemetry: Any, max_width: int = MAX_ROWS):
        if not 1 <= max_width <= MAX_ROWS:
            raise ValueError(f"max_width must be between 1 and {MAX_ROWS}")
        self.backend, self.telemetry, self.max_width = backend, telemetry, max_width
        self._owned: list[Any] = []

    def _step_group(self, group: list[Any], capacity: int):
        return self.backend.step_group([s.state for s in group], [s.tokens[-1] for s in group])

    def _sweep(self) -> None:
        for session in [s for s in self._owned if s.done]:
            self.backend.release(session.state)
            self._owned.remove(session)

    def run(self, sessions: list[Any], capacity: int, admit=None) -> None:
        self._owned.extend(sessions)

        def take() -> list[Any]:
            self._sweep()  # every round starts here: done and cancelled sessions free their slots
            fresh = list(admit()) if admit is not None else []
            self._owned.extend(fresh)
            return fresh

        try:
            super().run(sessions, capacity, admit=take)
        finally:
            for session in self._owned:
                self.backend.release(session.state)
            self._owned.clear()
