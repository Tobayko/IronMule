"""Native 4-bit matmuls for CUDA GPUs that emulate bfloat16 — the opt-in `native` plan.

Below compute capability 8 (Volta, Turing) CUDA has no bfloat16 arithmetic. MLX's CUDA
`quantized_matmul` accumulates in the activation type, so a bfloat16 checkpoint spends its
decode step in emulation: Qwen 3 8B decoded at ~6.5 tok/s on a T4 whose bandwidth allows
~70 (PORT1, PERF1). This plan keeps the checkpoint and its bfloat16 activations and changes
only how the 4-bit affine matmuls are computed:

* up to `MAX_ROWS` activation rows (decode, and the small batches of a server) go through one
  kernel that reads activations, scales and biases as raw 16-bit patterns — a bfloat16 is the
  top half of a float32, so widening is a shift — accumulates in float32, and rounds once;
* more rows (prefill) dequantise the weight to float16 and run one cuBLAS GEMM;
* MoE experts (mlx-lm's `QuantizedSwitchLinear`, i.e. `gather_qmm`) take the same row kernel
  with an index input for up to `GATHER_MAX` (token, expert) rows, each warp reading its
  expert's rows in place, and run larger calls as the same `gather_qmm` in float16.

It changes output, so it is opt-in like every numeric plan (`compute_dtype="native"`); its
measurements and quality gate are in `research/LEDGER.md`, PERF1. The arithmetic is pinned
(`__fmaf_rn`, `__fadd_rn`): each output row is computed with the same roundings whatever
else shares the launch, so a request's answer does not depend on how many others the server
batches with it. Installation is scoped to one model: eligible modules have their class
swapped, nothing global is patched, and nothing is installed unless a probe on the model's
own first eligible weight agrees with a float32 reference.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

GROUP_SIZE = 64
BITS = 4
MAX_ROWS = 8
PROBE_TOLERANCE = 1e-2
GATHER_MAX = 64  # (token, expert) rows: a server's eight requests times top-8

HEADER = r"""
__device__ __forceinline__ float lo_bf(unsigned int u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi_bf(unsigned int u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float one_bf(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
__device__ __forceinline__ unsigned short to_bf(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
"""

# One warp per R output rows and M <= 8 activation rows. A uint4 of packed weights is 32
# nibbles inside one group of 64 (MLX packing: low nibble first). Each chunk's nibbles are
# converted once and reused for all M rows.
SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int row0 = (gid >> 5) * R;
  const unsigned int lane = threadIdx.x & 31;
  if (row0 >= N) return;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);
  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);
  float acc[R * M];
#pragma unroll
  for (int i = 0; i < R * M; ++i) acc[i] = 0.f;
  for (unsigned int c = lane; c < K / 32; c += 32) {
    const unsigned int g = (c * 32) / GS;
    float xf[32];
    float sxm[M];
#pragma unroll
    for (int mo = 0; mo < M; ++mo) {
      LOAD_X
      float sx = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) sx = __fadd_rn(sx, xf[i]);
      sxm[mo] = sx;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const unsigned int row = row0 + r;
      if (row < N) {
        const uint4 q = w4[(size_t)row * (K / 32) + c];
        const unsigned int qw[4] = {q.x, q.y, q.z, q.w};
        float qf[32];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int e = 0; e < 8; ++e) qf[j * 8 + e] = (float)((qw[j] >> (4 * e)) & 0xFu);
        }
        const size_t sg = (size_t)row * (K / GS) + g;
        const float s = one_bf(ss[sg]);
        const float b = one_bf(bs[sg]);
#pragma unroll
        for (int mo = 0; mo < M; ++mo) {
          if (M > 1) {
            LOAD_X
          }
          float qx = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) qx = __fmaf_rn(qf[i], xf[i], qx);
          acc[r * M + mo] = __fadd_rn(acc[r * M + mo], __fmaf_rn(s, qx, __fmul_rn(b, sxm[mo])));
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
""".replace("LOAD_X", """
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 xa = xv[mo * (K / 8) + c * 4 + j];
      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        xf[j * 8 + 2 * e] = lo_bf(xw[e]);
        xf[j * 8 + 2 * e + 1] = hi_bf(xw[e]);
      }
    }
""")

# PERF1-S: warp w of P * W serves (token, expert) pair w / W, rows (w % W) * R.. of expert
# idx[pair]; XS pairs share one activation row (gate/up broadcast a token over its experts).
# From `acc` on it is the kernel above with M = 1, so each pair's dot is the qmv dot.
GATHER_SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int pair = (gid >> 5) / W;
  const unsigned int row0 = ((gid >> 5) % W) * R;
  const unsigned int lane = threadIdx.x & 31;
  if (pair >= P || row0 >= N) return;
  const unsigned int ex = idx[pair];
  const uint4* w4 = reinterpret_cast<const uint4*>(w) + (size_t)ex * N * (K / 32);
  const uint4* xv = reinterpret_cast<const uint4*>(x) + (size_t)(pair / XS) * (K / 8);
  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales) + (size_t)ex * N * (K / GS);
  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases) + (size_t)ex * N * (K / GS);
""" + SOURCE[SOURCE.index("  float acc[R * M];"):].replace("out[(i % M) * N + row]", "out[(size_t)pair * N + row]")

_kernel: Any = None
_gather_kernel: Any = None


def _rows_per_warp(n: int) -> int:
    # PERF1 run 1: two rows per warp won on the small Qwen 3 shapes, four from N >= 8192.
    return 4 if n >= 8192 else 2


def _qmv(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.cuda_kernel(name="ironmule_native_qmv4", input_names=["x", "w", "scales", "biases"],
                                      output_names=["out"], source=SOURCE, header=HEADER)
    n, k = w.shape[0], w.shape[1] * 8
    m = x.size // k
    rows = _rows_per_warp(n)
    threads = -(-(-(-n // rows) * 32) // 256) * 256  # whole blocks; surplus warps return at once
    out = _kernel(inputs=[mx.contiguous(x.reshape(m, k)), w, scales, biases],
                  template=[("N", n), ("K", k), ("GS", GROUP_SIZE), ("R", rows), ("M", m)],
                  grid=(threads, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(m, n)],
                  output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16).reshape(*x.shape[:-1], n)


def matmul(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """`x @ dequantize(w).T` for a 4-bit, group-64 affine weight with bfloat16 scales."""
    if x.dtype != mx.bfloat16:
        return mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=BITS)
    if x.size <= MAX_ROWS * x.shape[-1]:
        return _qmv(x, w, scales, biases)
    dense = mx.dequantize(w, scales.astype(mx.float16), biases.astype(mx.float16),
                          group_size=GROUP_SIZE, bits=BITS)
    return mx.matmul(x.astype(mx.float16), dense.T).astype(x.dtype)


def _gather_qmv(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, idx: mx.array,
                xs: int) -> mx.array:
    global _gather_kernel
    if _gather_kernel is None:
        _gather_kernel = mx.fast.cuda_kernel(
            name="ironmule_native_gather4", input_names=["x", "w", "scales", "biases", "idx"],
            output_names=["out"], source=GATHER_SOURCE, header=HEADER)
    n, k, p = w.shape[1], w.shape[2] * 8, idx.size
    rows = _rows_per_warp(n)
    warps = -(-n // rows)
    return _gather_kernel(
        inputs=[x, w, scales, biases, idx],
        template=[("N", n), ("K", k), ("GS", GROUP_SIZE), ("R", rows), ("M", 1), ("P", p), ("W", warps), ("XS", xs)],
        grid=(-(-(p * warps * 32) // 256) * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(p, n)],
        output_dtypes=[mx.uint16])[0].view(mx.bfloat16)


def gather_matmul(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, indices: mx.array,
                  sorted_indices: bool = False) -> mx.array:
    """`gather_qmm(x, w, scales, biases, rhs_indices=indices)` for the weights `matmul` takes."""
    if x.dtype != mx.bfloat16:
        return mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True,
                             group_size=GROUP_SIZE, bits=BITS, sorted_indices=sorted_indices)
    batch = tuple(mx.broadcast_shapes(x.shape[:-2], indices.shape))
    if x.shape[-2] == 1 and math.prod(batch) <= GATHER_MAX:
        if x.shape[:-2] == batch:  # down: one activation row per pair
            xs = 1
        elif x.shape[:-2] == batch[:-1] + (1,):  # gate/up: a token's row for all its experts
            xs = batch[-1]
        else:
            x, xs = mx.broadcast_to(x, batch + x.shape[-2:]), 1
        idx = mx.broadcast_to(indices, batch).flatten().astype(mx.uint32)
        out = _gather_qmv(mx.contiguous(x.reshape(-1, x.shape[-1])), w, scales, biases, idx, xs)
        return out.reshape(*batch, 1, w.shape[1])
    return mx.gather_qmm(x.astype(mx.float16), w, scales.astype(mx.float16), biases.astype(mx.float16),
                         rhs_indices=indices, transpose=True, group_size=GROUP_SIZE, bits=BITS,
                         sorted_indices=sorted_indices).astype(x.dtype)


class NativeQuantizedLinear(nn.QuantizedLinear):
    def __call__(self, x):
        y = matmul(x, self["weight"], self["scales"], self["biases"])
        return y + self["bias"] if "bias" in self else y


class NativeQuantizedEmbedding(nn.QuantizedEmbedding):
    def as_linear(self, x):
        return matmul(x, self["weight"], self["scales"], self["biases"])


class NativeQuantizedSwitchLinear(QuantizedSwitchLinear):
    def __call__(self, x, indices, sorted_indices=False):
        y = gather_matmul(x, self["weight"], self["scales"], self["biases"], indices, sorted_indices)
        return y + mx.expand_dims(self["bias"][indices], -2) if "bias" in self else y


def _eligible(module: nn.Module) -> bool:
    if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding, QuantizedSwitchLinear)):
        return False
    if (getattr(module, "mode", "affine") != "affine" or module.bits != BITS
            or module.group_size != GROUP_SIZE or module.get("biases") is None):
        return False
    return module["scales"].dtype == mx.bfloat16 and (module["weight"].shape[-1] * 8) % 32 == 0


def _probe(module: nn.Module) -> float:
    """Relative error of the kernel on the module's own weight against a float32 reference."""
    w, scales, biases = module["weight"], module["scales"], module["biases"]
    x = mx.random.normal((1, w.shape[-1] * 8), key=mx.random.key(0)).astype(mx.bfloat16)
    if isinstance(module, QuantizedSwitchLinear):  # one token against its first eight experts
        experts = mx.arange(min(8, w.shape[0]), dtype=mx.uint32)
        y = gather_matmul(x.reshape(1, 1, -1), w, scales, biases, experts).reshape(experts.size, -1)
        w, scales, biases = w[experts], scales[experts], biases[experts]
    else:
        y = _qmv(x, w, scales, biases)
    reference = (mx.dequantize(w, scales.astype(mx.float32), biases.astype(mx.float32),
                               group_size=GROUP_SIZE, bits=BITS) @ x.astype(mx.float32).T).squeeze(-1)
    error = mx.max(mx.abs(y.astype(mx.float32) - reference))
    return float(error / mx.maximum(mx.max(mx.abs(reference)), 1e-6))


def install(model: nn.Module, device_info: dict[str, Any] | None) -> dict[str, Any]:
    """Swap every eligible quantised module of `model` to the native kernels, or refuse."""
    from .numeric_plans import CUDA_PRE_AMPERE, device_class

    if device_class(device_info) != CUDA_PRE_AMPERE:
        raise ValueError("the native plan exists for CUDA GPUs below compute capability 8 only")
    modules = [m for _, m in model.named_modules() if _eligible(m)]
    if not modules:
        raise ValueError("the native plan found no 4-bit group-64 affine bfloat16 weights")
    switches = [m for m in modules if isinstance(m, QuantizedSwitchLinear)]
    error = max(_probe(m) for m in [modules[0]] + switches[:1])  # each kernel on its own weight
    if not error <= PROBE_TOLERANCE:
        raise ValueError(f"the native kernel disagrees with the float32 reference ({error:.3g})")
    for module in modules:
        module.__class__ = (NativeQuantizedEmbedding if isinstance(module, nn.QuantizedEmbedding)
                            else NativeQuantizedSwitchLinear if isinstance(module, QuantizedSwitchLinear)
                            else NativeQuantizedLinear)
    return {"modules": len(modules), "probe_rel_error": error}
