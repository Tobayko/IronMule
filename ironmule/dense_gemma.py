"""The `dense` plan for Gemma 3 at 4 bit: up to eight requests stacked into one decode step on pre-Ampere CUDA
(BACKLOG KB2).

Today a 4-bit Gemma 3 under the `native` plan decodes one request at a time on the reference worker: Gemma 3 12B
makes about 21 tokens a second on a Tesla T4, and a step reads all 8 GB of weights for one token (AG3: about 53 %
of the card's bandwidth). This plan stacks up to eight requests into one step, so the weights are read once for
all of them:

* every linear layer through `ironmule/cuda_native.py`'s 4-bit row kernel, which already takes up to eight
  activation rows with a float32 chain per row and pinned roundings, so a row's answer does not depend on how many
  others share the step (the native plan's own decode path for one row);
* attention through FA2 as in `dense_plan.py`, widened to Gemma 3's head dimension (256) and its sliding window:
  a row reads its own slot of a fixed-capacity KV pool from `max(0, length - window)` up to its position, in
  64-key chunks at absolute positions combined in index order, so neither the pool's capacity nor the other rows
  move a request's answer;
* the model's own embedding, RMS norms (with Gemma's `1 + weight`), RoPE per layer kind (one position per row) and
  GELU, which act on each row alone.

Prefill runs MLX's own layers, one request at a time, with a plain per-layer cache and mlx-lm's own causal and
sliding-window masks, so its keys and values come out in position order and go straight into the request's slot.
Only the last prefill position goes through the head, the plan's own at one row.

The plan changes answers against the native plan's reference path (its attention is FA2's, not MLX's), so it is
opt-in like every numeric plan and needs its own quality gate (KB2-0). It is built only on a model whose own
weights pass a load-time probe: every stacked width bit-identical to one row.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Callable, Sequence

MAX_ROWS = 8
FA2_CHUNK = 64
PLAN_VERSION = 1
PREFILL_CHUNK = 2048
#: Slot sizes, largest first: the largest whose pool fits beside the weights and `PREFILL_RESERVE` in
#: `SLOT_HEADROOM` of the card. Gemma 3 12B keeps 384 KiB of keys and values per token (48 layers, 8 KV heads of
#: 256), so eight 1,024-token slots take 3.2 GB beside its 8 GB of weights on a 15 GB T4.
SLOT_TIERS = (2048, 1024, 512)
SLOT_HEADROOM = 0.9
#: The plan's prefill never builds the head's float16 copy (only the last position goes through the head, at one
#: row), so its working memory is a 2,048-position chunk's activations and one MLP weight's float16 copy.
PREFILL_RESERVE = 3 * 1024 ** 3 // 2

HEADER = r"""
__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
__device__ __forceinline__ unsigned short to_bf(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
#define NEG_INF __int_as_float(0xff800000)
"""

# `dense_plan.FA2_PARTIAL` with three changes, none of which makes a row's arithmetic depend on another row or on
# the pool's capacity: E = D / 32 values per lane (Gemma 3: D = 256), the score scale as a template constant, and a
# window: a row of length `len` reads positions [max(0, len - W), len), W = 0 meaning no window. A chunk wholly
# before the window is skipped; inside a chunk, positions before it score -inf, so they weigh exactly zero.
PARTIAL = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * HKV * NC) return;
  const unsigned int c = warp % NC;
  const unsigned int b = (warp / NC) / HKV;
  const unsigned int kh = (warp / NC) % HKV;
  const int len = lengths[b];
  const int lo = (W > 0 && len > W) ? len - W : 0;
  const int start = c * 64;
  if (start >= len || start + 64 <= lo) return;
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
      if (t < len && t >= lo) {
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
    float acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
    for (int t = 0; t < count; ++t) {
      const float pt = __shfl_sync(0xffffffffu, pr[t >> 5], t & 31);
      if (pt == 0.f) continue;
      const unsigned short* vr = vs + base + (size_t)(start + t) * D + lane * E;
#pragma unroll
      for (int e = 0; e < E; ++e) acc[e] = __fmaf_rn(pt, one_bf(vr[e]), acc[e]);
    }
    const size_t po = ((size_t)b * H + kh * G + g) * NC + c;
#pragma unroll
    for (int e = 0; e < E; ++e) pacc[po * D + lane * E + e] = acc[e];
    if (lane == 0) {
      pml[po * 2] = m;
      pml[po * 2 + 1] = sum;
    }
  }
"""

COMBINE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * H) return;
  const unsigned int b = warp / H;
  const unsigned int h = warp % H;
  const int len = lengths[b];
  const int lo = (W > 0 && len > W) ? len - W : 0;
  const int nc = (len + 63) / 64;
  float M = NEG_INF, L = 0.f;
  float acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.f;
  for (int c = lo / 64; c < nc; ++c) {
    const size_t po = ((size_t)b * H + h) * NC + c;
    const float mc = pml[po * 2];
    const float lc = pml[po * 2 + 1];
    const float mn = fmaxf(M, mc);
    const float a = expf(M - mn);
    const float bb = expf(mc - mn);
    L = __fmaf_rn(L, a, __fmul_rn(lc, bb));
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = __fmaf_rn(acc[e], a, __fmul_rn(pacc[po * D + lane * E + e], bb));
    M = mn;
  }
#pragma unroll
  for (int e = 0; e < E; ++e) out[((size_t)b * H + h) * D + lane * E + e] = to_bf(__fdiv_rn(acc[e], L));
"""

# KB3: FA2 version 2. Two changes to how the work is laid out, none to what is computed or in which order, so its
# outputs must be bit-identical to version 1's (KB3-0 tests that on the card):
# 1. one warp per (row, KV head, query head, chunk): the former loop over the G query heads of a KV head becomes G
#    warps, each running that loop's body for its head (G times the warps at one row);
# 2. each warp stages its chunk's keys in shared memory, 32 dimensions at a time, the 32 lanes reading 64
#    contiguous bytes of one key row per load; each lane then forms the same products in the same order (dimension
#    0 to D-1) from there, where version 1 read its own 512-byte key row 2 bytes at a time.
# Rows of the tile are padded to 34 values, so the lanes' reads of rows `lane` and `32 + lane` hit 32 banks.
PARTIAL_V2 = r"""
  __shared__ unsigned short tile[8][64 * 34];
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * HKV * G * NC) return;
  const unsigned int c = warp % NC;
  const unsigned int g = (warp / NC) % G;
  const unsigned int kh = (warp / (NC * G)) % HKV;
  const unsigned int b = warp / (NC * G * HKV);
  const int len = lengths[b];
  const int lo = (W > 0 && len > W) ? len - W : 0;
  const int start = c * 64;
  if (start >= len || start + 64 <= lo) return;
  const unsigned short* qs = reinterpret_cast<const unsigned short*>(q);
  const unsigned short* ks = reinterpret_cast<const unsigned short*>(k);
  const unsigned short* vs = reinterpret_cast<const unsigned short*>(v);
  const size_t base = ((size_t)slots[b] * HKV + kh) * C * D;
  const int count = min(64, len - start);
  unsigned short* kt = tile[threadIdx.x >> 5];
  const size_t qo = ((size_t)b * H + kh * G + g) * D;
  float d0 = 0.f, d1 = 0.f;
  for (int i0 = 0; i0 < D; i0 += 32) {
    for (int r = 0; r < 64; ++r) {
      const int t = start + r;
      kt[r * 34 + lane] = (t < len) ? ks[base + (size_t)t * D + i0 + lane] : (unsigned short)0;
    }
    __syncwarp();
    for (int i = 0; i < 32; ++i) {
      const float qi = one_bf(qs[qo + i0 + i]);
      d0 = __fmaf_rn(qi, one_bf(kt[lane * 34 + i]), d0);
      d1 = __fmaf_rn(qi, one_bf(kt[(32 + lane) * 34 + i]), d1);
    }
    __syncwarp();
  }
  float s[2];
  {
    const int t0 = start + (int)lane;
    const int t1 = start + 32 + (int)lane;
    s[0] = (t0 < len && t0 >= lo) ? __fmul_rn(d0, SCALE) : NEG_INF;
    s[1] = (t1 < len && t1 >= lo) ? __fmul_rn(d1, SCALE) : NEG_INF;
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
  float acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.f;
  for (int t = 0; t < count; ++t) {
    const float pt = __shfl_sync(0xffffffffu, pr[t >> 5], t & 31);
    if (pt == 0.f) continue;
    const unsigned short* vr = vs + base + (size_t)(start + t) * D + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = __fmaf_rn(pt, one_bf(vr[e]), acc[e]);
  }
  const size_t po = ((size_t)b * H + kh * G + g) * NC + c;
#pragma unroll
  for (int e = 0; e < E; ++e) pacc[po * D + lane * E + e] = acc[e];
  if (lane == 0) {
    pml[po * 2] = m;
    pml[po * 2 + 1] = sum;
  }
"""

# KB3-1: FA2 version 3, version 2 with three more changes that keep every operation and its order (KB3-0 found v2
# bit-identical but 3.2 ms slower than v1 at one row on a 12-token prompt, where it staged and multiplied all 64
# key rows of a chunk): the tile loads only the chunk's valid rows (`count`, the same for the whole warp), the
# second half of the keys is multiplied only where the chunk has more than 32, and the row loop is unrolled by 8.
# Rows past `count` keep stale values; their scores are -inf in every version, so nothing reads them.
PARTIAL_V3 = r"""
  __shared__ unsigned short tile[8][64 * 34];
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int lane = threadIdx.x & 31;
  if (warp >= B * HKV * G * NC) return;
  const unsigned int c = warp % NC;
  const unsigned int g = (warp / NC) % G;
  const unsigned int kh = (warp / (NC * G)) % HKV;
  const unsigned int b = warp / (NC * G * HKV);
  const int len = lengths[b];
  const int lo = (W > 0 && len > W) ? len - W : 0;
  const int start = c * 64;
  if (start >= len || start + 64 <= lo) return;
  const unsigned short* qs = reinterpret_cast<const unsigned short*>(q);
  const unsigned short* ks = reinterpret_cast<const unsigned short*>(k);
  const unsigned short* vs = reinterpret_cast<const unsigned short*>(v);
  const size_t base = ((size_t)slots[b] * HKV + kh) * C * D;
  const int count = min(64, len - start);
  unsigned short* kt = tile[threadIdx.x >> 5];
  const size_t qo = ((size_t)b * H + kh * G + g) * D;
  float d0 = 0.f, d1 = 0.f;
  const bool second = count > 32;  // the same for the whole warp
  for (int i0 = 0; i0 < D; i0 += 32) {
#pragma unroll 8
    for (int r = 0; r < count; ++r) kt[r * 34 + lane] = ks[base + (size_t)(start + r) * D + i0 + lane];
    __syncwarp();
    if (second) {
      for (int i = 0; i < 32; ++i) {
        const float qi = one_bf(qs[qo + i0 + i]);
        d0 = __fmaf_rn(qi, one_bf(kt[lane * 34 + i]), d0);
        d1 = __fmaf_rn(qi, one_bf(kt[(32 + lane) * 34 + i]), d1);
      }
    } else {
      for (int i = 0; i < 32; ++i) d0 = __fmaf_rn(one_bf(qs[qo + i0 + i]), one_bf(kt[lane * 34 + i]), d0);
    }
    __syncwarp();
  }
  float s[2];
  {
    const int t0 = start + (int)lane;
    const int t1 = start + 32 + (int)lane;
    s[0] = (t0 < len && t0 >= lo) ? __fmul_rn(d0, SCALE) : NEG_INF;
    s[1] = (t1 < len && t1 >= lo) ? __fmul_rn(d1, SCALE) : NEG_INF;
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
  float acc[E];
#pragma unroll
  for (int e = 0; e < E; ++e) acc[e] = 0.f;
  for (int t = 0; t < count; ++t) {
    const float pt = __shfl_sync(0xffffffffu, pr[t >> 5], t & 31);
    if (pt == 0.f) continue;
    const unsigned short* vr = vs + base + (size_t)(start + t) * D + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = __fmaf_rn(pt, one_bf(vr[e]), acc[e]);
  }
  const size_t po = ((size_t)b * H + kh * G + g) * NC + c;
#pragma unroll
  for (int e = 0; e < E; ++e) pacc[po * D + lane * E + e] = acc[e];
  if (lane == 0) {
    pml[po * 2] = m;
    pml[po * 2 + 1] = sum;
  }
"""

#: What the kernels' answers are bound to: a changed source is a changed plan.
KERNEL_SHA256 = hashlib.sha256((PARTIAL + COMBINE).encode()).hexdigest()
KERNEL_V2_SHA256 = hashlib.sha256((PARTIAL_V2 + COMBINE).encode()).hexdigest()
KERNEL_V3_SHA256 = hashlib.sha256((PARTIAL_V3 + COMBINE).encode()).hexdigest()
#: The FA2 version the plan serves; another only once its KB3 test found it bit-identical to 1 on the card and
#: fast enough (KB3-0: version 2 was bit-identical but not fast enough).
FA2_VERSION = 1

_kernels: dict[str, Any] = {}
_kernel_lock = threading.Lock()


def _kernel(name: str, scale: float) -> Any:
    import mlx.core as mx

    key = f"{name}:{scale!r}"
    with _kernel_lock:
        if key not in _kernels:
            if name in ("partial", "partial_v2", "partial_v3"):
                source = {"partial": PARTIAL, "partial_v2": PARTIAL_V2, "partial_v3": PARTIAL_V3}[name]
                _kernels[key] = mx.fast.cuda_kernel(
                    name=f"ironmule_fa2g_{name}", input_names=["q", "k", "v", "lengths", "slots"],
                    output_names=["pacc", "pml"], source=source.replace("SCALE", f"{scale!r}f"), header=HEADER)
            else:
                _kernels[key] = mx.fast.cuda_kernel(
                    name="ironmule_fa2g_combine", input_names=["pacc", "pml", "lengths"], output_names=["out"],
                    source=COMBINE, header=HEADER)
        return _kernels[key]


def fa2(q: Any, keys: Any, values: Any, lengths: Any, slots: Any, kv_heads: int, capacity: int,
        scale: float, window: int, version: int | None = None) -> Any:
    """One decode query per row (B, H, D) against its own slot of a pool laid out (slots, kv_heads, capacity, D),
    flat or not, reading positions [max(0, length - window), length) (`window` 0: from the start). `version` picks
    the partial kernel (None: `FA2_VERSION`); both compute the same values in the same order (KB3)."""
    import mlx.core as mx

    b, h, d = q.shape
    if d % 32:
        raise ValueError(f"FA2 takes a head dim divisible by 32, got {d}")
    version = FA2_VERSION if version is None else version
    if version not in (1, 2, 3):
        raise ValueError(f"unknown FA2 version {version}")
    nc = -(-capacity // FA2_CHUNK)
    lengths, slots = lengths.astype(mx.int32), slots.astype(mx.int32)
    template = [("B", b), ("H", h), ("HKV", kv_heads), ("G", h // kv_heads), ("D", d), ("E", d // 32),
                ("C", capacity), ("NC", nc), ("W", window)]
    warps = b * kv_heads * nc * (1 if version == 1 else h // kv_heads)
    pacc, pml = _kernel("partial" if version == 1 else f"partial_v{version}", scale)(
        inputs=[mx.contiguous(q), keys, values, lengths, slots], template=template,
        grid=(-(-(warps * 32) // 256) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(b, h, nc, d), (b, h, nc, 2)], output_dtypes=[mx.float32, mx.float32])
    out = _kernel("combine", scale)(
        inputs=[pacc, pml, lengths], template=template, grid=(-(-(b * h * 32) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(b, h, d)], output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16)


def reference_attention(q: Any, keys: Any, values: Any, lengths: Any, slots: Any, kv_heads: int, capacity: int,
                        scale: float, window: int) -> Any:
    """FA2's semantics in plain MLX, one row at a time (for tests on machines without CUDA; not batch-invariant)."""
    import mlx.core as mx

    b, h, d = q.shape
    pool_k = keys.reshape(-1, kv_heads, capacity, d)
    pool_v = values.reshape(-1, kv_heads, capacity, d)
    rows = []
    for r in range(b):
        length = int(lengths[r].item())
        lo = max(0, length - window) if window > 0 else 0
        slot = int(slots[r].item())
        k = mx.repeat(pool_k[slot, :, lo:length], h // kv_heads, axis=0).astype(mx.float32)
        v = mx.repeat(pool_v[slot, :, lo:length], h // kv_heads, axis=0).astype(mx.float32)
        scores = (k @ q[r].astype(mx.float32)[:, :, None])[..., 0] * scale
        rows.append((mx.softmax(scores, axis=-1)[:, None, :] @ v)[:, 0, :])
    return mx.stack(rows).astype(mx.bfloat16)


def _text(model: Any) -> Any:
    inner = getattr(model, "language_model", None)
    return model if inner is None else inner


def refuse(model: Any, *, native: bool = True) -> str | None:
    """Why the plan cannot serve `model`, or None: Gemma 3's text stack at 4 bit, plain or linearly scaled RoPE,
    and (with `native`, as on CUDA) every linear layer already on the native plan's row kernel."""
    import mlx.nn as nn

    text = _text(model)
    args = getattr(text, "args", None)
    if getattr(args, "model_type", None) not in ("gemma3_text", "gemma3"):
        return "the Gemma dense plan serves Gemma 3 checkpoints only"
    if getattr(args, "head_dim", 0) % 32:
        return "the plan's attention kernel needs a head dim divisible by 32"
    trunk = text.model
    attn = trunk.layers[0].self_attn
    if not isinstance(attn.q_proj, nn.QuantizedLinear):
        return "the Gemma dense plan serves 4-bit checkpoints; an unquantized one has no native row kernel"
    for layer in trunk.layers:
        if type(layer.self_attn.rope) is not nn.RoPE:
            return "the Gemma dense plan covers plain and linearly scaled rotary embeddings only"
    if native:
        from .cuda_native import NativeQuantizedEmbedding, NativeQuantizedLinear

        linears = [m for _, m in trunk.named_modules() if isinstance(m, nn.QuantizedLinear)]
        if not linears or not all(isinstance(m, NativeQuantizedLinear) for m in linears):
            return "install the native plan first: every linear layer must run the native row kernel"
        if getattr(text, "tie_word_embeddings", False) and not isinstance(trunk.embed_tokens, NativeQuantizedEmbedding):
            return "install the native plan first: the tied head must run the native row kernel"
    return None


class GemmaDensePlan:
    """The plan's decode step over a pool of `slots` fixed-capacity KV slots: rows (B <= 8) name their slot and
    position, and the same forward serves one row or eight. `attention` defaults to FA2 (CUDA); tests pass
    `reference_attention` together with `native=False`."""

    def __init__(self, model: Any, *, slots: int = MAX_ROWS, capacity: int = 1024,
                 attention: Callable[..., Any] | None = None, native: bool = True):
        import mlx.core as mx

        refusal = refuse(model, native=native)
        if refusal:
            raise ValueError(refusal)
        self.model, self.text = model, _text(model)
        self.trunk, self.args = self.text.model, self.text.args
        args = self.args
        self.heads, self.kv_heads, self.dim = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        self.scale = float(args.query_pre_attn_scalar) ** -0.5
        self.window = int(args.sliding_window)
        self.slots, self.capacity = slots, capacity
        self.attention = attention or fa2
        self.embed_scale = mx.array(args.hidden_size ** 0.5, mx.bfloat16)
        rows = slots * self.kv_heads * capacity
        self.keys = [mx.zeros((rows, self.dim), mx.bfloat16) for _ in self.trunk.layers]
        self.values = [mx.zeros((rows, self.dim), mx.bfloat16) for _ in self.trunk.layers]
        mx.eval(self.keys, self.values)

    def head(self, h: Any) -> Any:
        if getattr(self.text, "tie_word_embeddings", False):
            return self.trunk.embed_tokens.as_linear(h)
        return self.text.lm_head(h)

    def _rows(self, slots: Any, positions: Any) -> Any:
        """Flat pool rows of (slot, every KV head, position), shape (B * kv_heads,)."""
        import mlx.core as mx

        heads = mx.arange(self.kv_heads, dtype=mx.int32)
        rows = (slots.astype(mx.int32)[:, None] * self.kv_heads + heads[None, :]) * self.capacity
        return (rows + positions.astype(mx.int32)[:, None]).reshape(-1)

    def new_cache(self) -> list[Any]:
        """A prefill cache in position order: a plain cache for every layer (mlx-lm's own masks apply the window)."""
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.trunk.layers]

    def prefill(self, ids: Any, cache: list[Any]) -> Any:
        """MLX's own prefill of one prompt (1, L) into `cache` (from `new_cache`), in chunks of `PREFILL_CHUNK`;
        only the last position goes through the head, at one row. Float32 logits (1, V)."""
        import mlx.core as mx

        for start in range(0, ids.shape[1], PREFILL_CHUNK):
            if start:
                mx.eval([c.state for c in cache])
            hidden = self.trunk(ids[:, start:start + PREFILL_CHUNK], cache=cache)
        return self.head(hidden[0, -1:, :]).astype(mx.float32)

    def write_prefill(self, slot: int, states: Sequence[tuple[Any, Any]]) -> None:
        """One request's prefilled keys and values (each layer's `new_cache()` state after `prefill`, (1, Hkv, L, D))
        into `slot`; the same call as `DensePlan.write_prefill`, so `DenseBackend` serves either plan."""
        import mlx.core as mx

        length = states[0][0].shape[2]
        if length > self.capacity:
            raise ValueError(f"a {length}-token prefill does not fit a {self.capacity}-token slot")
        heads = mx.arange(self.kv_heads, dtype=mx.int32)[:, None]
        rows = ((slot * self.kv_heads + heads) * self.capacity + mx.arange(length, dtype=mx.int32)[None, :]).reshape(-1)
        for i, (k, v) in enumerate(states):
            self.keys[i][rows] = k[0].reshape(-1, self.dim).astype(mx.bfloat16)
            self.values[i][rows] = v[0].reshape(-1, self.dim).astype(mx.bfloat16)
        mx.eval(self.keys, self.values)

    def forward(self, tokens: Any, slots: Any, positions: Any) -> Any:
        """Tokens (B,) at positions (B,) of slots (B,); writes their KV and returns float32 logits (B, V)."""
        import mlx.core as mx
        import mlx.nn as nn

        rows = tokens.shape[0]
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"the plan stacks 1 to {MAX_ROWS} rows, got {rows}")
        h = self.trunk.embed_tokens(tokens)
        h = h * self.embed_scale.astype(h.dtype)
        written = self._rows(slots, positions)
        lengths = positions + 1
        for i, layer in enumerate(self.trunk.layers):
            attn, mlp = layer.self_attn, layer.mlp
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(rows, self.heads, self.dim))
            k = attn.k_norm(attn.k_proj(x).reshape(rows, self.kv_heads, self.dim))
            v = attn.v_proj(x).reshape(rows, self.kv_heads, self.dim)
            q, k = self._rope(q, positions, attn.rope), self._rope(k, positions, attn.rope)
            self.keys[i][written] = k.reshape(-1, self.dim).astype(mx.bfloat16)
            self.values[i][written] = v.reshape(-1, self.dim).astype(mx.bfloat16)
            o = self.attention(q, self.keys[i], self.values[i], lengths, slots, self.kv_heads, self.capacity,
                               self.scale, self.window if attn.is_sliding else 0)
            h = h + layer.post_attention_layernorm(attn.o_proj(o.reshape(rows, self.heads * self.dim).astype(h.dtype)))
            x = layer.pre_feedforward_layernorm(h)
            h = h + layer.post_feedforward_layernorm(mlp.down_proj(nn.gelu_approx(mlp.gate_proj(x)) * mlp.up_proj(x)))
        return self.head(self.trunk.norm(h)).astype(mx.float32)

    @staticmethod
    def _rope(x: Any, positions: Any, rope: Any) -> Any:
        import mlx.core as mx

        rows, heads, dim = x.shape
        return mx.fast.rope(x.reshape(rows, heads, 1, dim), rope.dims, traditional=rope.traditional, base=rope.base,
                            scale=rope.scale, offset=positions.astype(mx.int32)).reshape(rows, heads, dim)


def slot_capacity(model: Any, device_info: dict[str, Any] | None, slots: int = MAX_ROWS,
                  tiers: Sequence[int] = SLOT_TIERS) -> int | None:
    """The largest of `tiers` whose pool fits beside the weights and `PREFILL_RESERVE` in `SLOT_HEADROOM` of the
    card's memory; the smallest when the card reports none; None when not even that fits."""
    from mlx.utils import tree_flatten

    info = device_info or {}
    total = next((info[key] for key in ("total_memory", "memory_size", "total_memory_bytes", "memory_bytes")
                  if isinstance(info.get(key), int) and info[key] > 0), None)
    if total is None:
        return tiers[-1]
    text = _text(model)
    args = text.args
    weights = sum(value.nbytes for _, value in tree_flatten(text.parameters()))
    per_token = 2 * len(text.model.layers) * args.num_key_value_heads * args.head_dim * 2
    return next((tier for tier in tiers
                 if weights + slots * tier * per_token + PREFILL_RESERVE <= SLOT_HEADROOM * total), None)


def probe(plan: GemmaDensePlan) -> dict[str, Any]:
    """One step at distinct positions in distinct slots, at every width from one row to all of them, compared bit
    for bit against each row alone."""
    import mlx.core as mx

    rows = min(MAX_ROWS, plan.slots)
    tokens = (mx.arange(rows, dtype=mx.uint32) * 977 + 11) % plan.args.vocab_size
    positions = (mx.arange(rows, dtype=mx.int32) * 37 + 3) % (plan.capacity - 1)
    slots = mx.arange(rows, dtype=mx.int32)[::-1]
    alone = [plan.forward(tokens[i:i + 1], slots[i:i + 1], positions[i:i + 1]) for i in range(rows)]
    mx.eval(*alone)
    identical = True
    for width in range(1, rows + 1):
        stacked = plan.forward(tokens[:width], slots[:width], positions[:width])
        mx.eval(stacked)
        identical = identical and all(bool(mx.array_equal(stacked[i], alone[i][0]).item()) for i in range(width))
    return {"stacked_rows_identical": identical}


def install(model: Any, device_info: dict[str, Any] | None, *, slots: int = MAX_ROWS,
            capacity: int | None = None) -> tuple[GemmaDensePlan, dict[str, Any]]:
    """Build the plan for a Gemma 3 model already on the native plan, or refuse. The probe on the model's own
    weights must find every stacked width bit-identical to one row."""
    from .numeric_plans import CUDA_PRE_AMPERE, device_class

    if device_class(device_info) != CUDA_PRE_AMPERE:
        raise ValueError("the Gemma dense plan exists for CUDA GPUs below compute capability 8 only")
    capacity = slot_capacity(model, device_info, slots) if capacity is None else capacity
    if capacity is None:
        raise ValueError("the Gemma dense plan's smallest slot pool does not fit this card beside the model")
    plan = GemmaDensePlan(model, slots=slots, capacity=capacity)
    record = probe(plan)
    if not record["stacked_rows_identical"]:
        raise ValueError("the Gemma dense plan's stacked rows differ from one row on this model")
    kernel = {1: KERNEL_SHA256, 2: KERNEL_V2_SHA256, 3: KERNEL_V3_SHA256}[FA2_VERSION]
    return plan, {**record, "kernel_sha256": kernel,
                  "fa2_version": FA2_VERSION, "plan_version": PLAN_VERSION, "slots": slots,
                  "capacity": capacity}
