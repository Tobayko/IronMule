"""PERF1: native 4-bit matvec and tensor-core prefill against emulated bfloat16 on a T4.

Usage: python perf1.py kernel OUT.json
       python perf1.py chain OUT.json
       python perf1.py mma OUT.json
       python perf1.py e2e MODEL_PATH ARM OUT.json
       python perf1.py nll MODEL_PATH ARM PATH_MODE TEXT OUT.json    (PATH_MODE: prefill|decode)
       python perf1.py spec MODEL_PATH ARM DRAFT_PATH DRAFTS OUT.json  (DRAFTS: e.g. "2,3,4")
       python perf1.py server MODEL_PATH ARM WIDTHS OUT.json            (WIDTHS: e.g. "1,4,8")
       python perf1.py ironmule MODEL_ID REVISION ARM KNOBS_JSON interactive|throughput OUT.json

ARM is "+"-joined parts: "stock" (bf16 checkpoint as loaded), "fp32" (IronMule's float32
plan, `set_dtype`), "kernel" (single-row bf16 4-bit matmuls through the native kernel),
"k32" (the same kernel reading float32 activations and scales, for the float32 plan),
"p16" (multi-row 4-bit matmuls, i.e. prefill: dequantise to float16, one tensor-core GEMM,
cast back). E.g. "kernel+p16", "fp32+k32+p16".

Why: run 1 (`69dbc7af`) decoded Qwen 3 8B at 6.7 tok/s stock and 36.9 with the kernel; the
T4's bandwidth allows ~70. MLX's CUDA `qmv` accumulates in the activation type and bfloat16
is emulated below compute capability 8. A bfloat16 is the top half of a float32: widening
it is one shift. The kernel reads activations, scales and biases as raw bits, accumulates
in float32 and rounds once at the end. Prefill stayed at 26 s for 512 tokens, because the
multi-row path is the same emulated bfloat16; "p16" hands it to cuBLAS in float16 instead.
Both are numeric plans, not token-identical to stock; the "nll" mode is their quality check.
Weights and packing are MLX's own: 4-bit, low nibble first, group size 64.

Timing: every arm runs a full warmup generation first, then REPS measured generations of a
fixed ~512-token prompt; TTFT is prefill plus the first token, decode rate is the remaining
tokens over their wall time, mlx-lm's own async-eval pattern. No performance claim.
"""
import json
import math
import os
import statistics as st
import sys
import time

import mlx.core as mx

PROMPT_TOKENS = 512
NEW_TOKENS = 128
REPS = 3
GS = 64
NLL_CHUNKS = int(os.environ.get("PERF1_NLL_CHUNKS", "4"))
NLL_TOKENS = int(os.environ.get("PERF1_NLL_TOKENS", "256"))

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

# One warp per R output rows, for M <= 8 activation rows (M = 1 is decode; M > 1 is the
# speculative verify step). A uint4 of packed weights is 32 4-bit values inside one group of
# 64. Per chunk the nibbles are converted once and reused for all M rows. Each (row, m) dot
# is summed in the same order and with the same formula whatever M is, so a verify step
# computes bit for bit what M separate decode steps would.
SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int row0 = (gid >> 5) * R;
  const unsigned int lane = threadIdx.x & 31;
  if (row0 >= N) return;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  @PTRS@
  float acc[R * M];
#pragma unroll
  for (int i = 0; i < R * M; ++i) acc[i] = 0.f;
  for (unsigned int c = lane; c < K / 32; c += 32) {
    const unsigned int g = (c * 32) / GS;
    float xf[32];
    float sxm[M];
#pragma unroll
    for (int mo = 0; mo < M; ++mo) {
      @LOAD@
      float sx = 0.f;
#pragma unroll
      for (int i = 0; i < 32; ++i) sx = @SX@;
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
          for (int e = 0; e < 8; ++e) qf[j * 8 + e] = @QF@;
        }
        const size_t sg = (size_t)row * (K / GS) + g;
        const float s = @SCALE@;
        const float b = @BIAS@;
#pragma unroll
        for (int mo = 0; mo < M; ++mo) {
          if (M > 1) {
            @LOAD@
          }
          float qx = 0.f;
#pragma unroll
          for (int i = 0; i < 32; ++i) qx = @QX@;
          acc[r * M + mo] = @ACC@;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R * M; ++i) {
    float a = acc[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = @RED@;
    const unsigned int row = row0 + i / M;
    if (lane == 0 && row < N) out[(i % M) * N + row] = @STORE@;
  }
"""
VARIANTS = {
    "bf16": {"@PTRS@": "const uint4* xv = reinterpret_cast<const uint4*>(x);\n"
                       "  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);\n"
                       "  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);",
             "@LOAD@": "\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {\n"
                       "      const uint4 xa = xv[mo * (K / 8) + c * 4 + j];\n"
                       "      const unsigned int xw[4] = {xa.x, xa.y, xa.z, xa.w};\n"
                       "#pragma unroll\n      for (int e = 0; e < 4; ++e) {\n"
                       "        xf[j * 8 + 2 * e] = lo_bf(xw[e]);\n"
                       "        xf[j * 8 + 2 * e + 1] = hi_bf(xw[e]);\n      }\n    }\n",
             "@SCALE@": "one_bf(ss[sg])", "@BIAS@": "one_bf(bs[sg])", "@STORE@": "to_bf(a)"},
    "f32": {"@PTRS@": "const float4* xv = reinterpret_cast<const float4*>(x);\n"
                      "  const float* ss = reinterpret_cast<const float*>(scales);\n"
                      "  const float* bs = reinterpret_cast<const float*>(biases);",
            "@LOAD@": "\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n"
                      "      const float4 xa = xv[mo * (K / 4) + c * 8 + j];\n"
                      "      xf[4 * j] = xa.x; xf[4 * j + 1] = xa.y; xf[4 * j + 2] = xa.z; xf[4 * j + 3] = xa.w;\n"
                      "    }\n",
            "@SCALE@": "ss[sg]", "@BIAS@": "bs[sg]", "@STORE@": "a"},
}
MAX_M = 8
_kernels = {}


# Nibble to float. "cvt" is an integer conversion, a quarter-rate instruction on Turing;
# "magic" ORs the nibble into the mantissa of 2^23 and subtracts 2^23 — the same exact value
# from full-rate instructions, so both give bit-identical results.
QF = {"cvt": "(float)((qw[j] >> (4 * e)) & 0xFu)",
      "magic": "(__uint_as_float(0x4B000000u | ((qw[j] >> (4 * e)) & 0xFu)) - 8388608.0f)"}


# Arithmetic. "free" lets the compiler contract multiply-adds as it likes, which it does
# differently per M; "pinned" spells every rounding step, so a row computes the same bits
# whatever else shares the launch — a request's answer does not depend on its batch.
ARITH = {"free": {"@SX@": "sx + xf[i]", "@QX@": "qx + qf[i] * xf[i]",
                  "@ACC@": "acc[r * M + mo] + (s * qx + b * sxm[mo])",
                  "@RED@": "a + __shfl_down_sync(0xffffffffu, a, o)"},
         "pinned": {"@SX@": "__fadd_rn(sx, xf[i])", "@QX@": "__fmaf_rn(qf[i], xf[i], qx)",
                    "@ACC@": "__fadd_rn(acc[r * M + mo], __fmaf_rn(s, qx, __fmul_rn(b, sxm[mo])))",
                    "@RED@": "__fadd_rn(a, __shfl_down_sync(0xffffffffu, a, o))"}}


# Tensor cores for 2..16 activation rows (a server's batch). One warp per 8 weight rows; each
# lane dequantises its two weights of a k-step to float16 in registers, and one
# `mma.sync.m16n8k8` (float16 in, float32 accumulate, sm_75) multiplies them with up to 16
# activation rows at once, so a batch costs about what one row does. Fragment layout per the
# PTX ISA: A rows g and g+8, columns 2t and 2t+1; B column g, rows 2t and 2t+1; C likewise.
MMA_HEADER = HEADER + r"""
__device__ __forceinline__ unsigned int pack_h2(float lo, float hi) {
  unsigned short a, b;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(a) : "f"(lo));
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(b) : "f"(hi));
  return (unsigned int)a | ((unsigned int)b << 16);
}
"""
MMA_SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int n0 = (gid >> 5) * 8;
  const unsigned int lane = threadIdx.x & 31;
  if (n0 >= N) return;
  const unsigned int g = lane >> 2, t = lane & 3;
  const unsigned int row = n0 + g;
  const bool row_ok = row < N;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const unsigned int* xw = reinterpret_cast<const unsigned int*>(x);
  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);
  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  for (unsigned int grp = 0; grp < K / 64; ++grp) {
    uint4 qa = make_uint4(0u, 0u, 0u, 0u), qb = qa;
    float s = 0.f, b = 0.f;
    if (row_ok) {
      const size_t base = (size_t)row * (K / 32) + grp * 2;
      qa = w4[base];
      qb = w4[base + 1];
      s = one_bf(ss[(size_t)row * (K / 64) + grp]);
      b = one_bf(bs[(size_t)row * (K / 64) + grp]);
    }
    const unsigned int words[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned int k0 = grp * 64 + j * 8;
      const unsigned int v = words[j] >> (8 * t);
      const unsigned int bfrag = pack_h2(fmaf(s, (float)(v & 0xFu), b), fmaf(s, (float)((v >> 4) & 0xFu), b));
      unsigned int a0 = 0u, a1 = 0u;
      if (g < M) {
        const unsigned int p = xw[(g * K + k0) / 2 + t];
        a0 = pack_h2(lo_bf(p), hi_bf(p));
      }
      if (g + 8 < M) {
        const unsigned int p = xw[((g + 8) * K + k0) / 2 + t];
        a1 = pack_h2(lo_bf(p), hi_bf(p));
      }
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                   : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(bfrag));
    }
  }
  const unsigned int col = n0 + 2 * t;
  if (g < M) {
    if (col < N) out[g * N + col] = to_bf(c0);
    if (col + 1 < N) out[g * N + col + 1] = to_bf(c1);
  }
  if (g + 8 < M) {
    if (col < N) out[(g + 8) * N + col] = to_bf(c2);
    if (col + 1 < N) out[(g + 8) * N + col + 1] = to_bf(c3);
  }
"""
MMA_MAX = 16
MMA_MIN = int(os.environ.get("PERF1_MMA_MIN", "2"))
_mma = None


def mma_qmm(x, w, scales, biases):
    """x (..., K) bf16 with M <= 16 rows; w (N, K/8); scales/biases (N, K/64) bf16; K % 64 == 0."""
    global _mma
    if _mma is None:
        _mma = mx.fast.cuda_kernel(name="perf1_mma4", input_names=["x", "w", "scales", "biases"],
                                   output_names=["out"], source=MMA_SOURCE, header=MMA_HEADER)
    n, k = w.shape[0], w.shape[1] * 8
    m = x.size // k
    threads = -(-(-(-n // 8) * 32) // 256) * 256
    out = _mma(inputs=[mx.contiguous(x.reshape(m, k)), w, scales, biases],
               template=[("N", n), ("K", k), ("M", m)], grid=(threads, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(m, n)], output_dtypes=[mx.uint16])[0]
    return out.view(mx.bfloat16).reshape(*x.shape[:-1], n)


# v2: the same fragments, three latency fixes. The next group's weights are loaded into
# registers while the current group computes; even and odd k-steps accumulate into two
# independent mma chains; and K is split across S warps per 8-row tile (partials in float32,
# summed afterwards), so a small N still puts enough warps on the card.
MMA2_SOURCE = r"""
  const unsigned int gid = cooperative_groups::this_grid().thread_rank();
  const unsigned int warp = gid >> 5;
  const unsigned int n0 = (warp / S) * 8;
  const unsigned int split = warp % S;
  const unsigned int lane = threadIdx.x & 31;
  if (n0 >= N) return;
  const unsigned int g = lane >> 2, t = lane & 3;
  const unsigned int row = n0 + g;
  const bool row_ok = row < N;
  const unsigned int groups = K / 64;
  const unsigned int g_begin = split * groups / S, g_end = (split + 1) * groups / S;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const unsigned int* xw = reinterpret_cast<const unsigned int*>(x);
  const unsigned short* ss = reinterpret_cast<const unsigned short*>(scales);
  const unsigned short* bs = reinterpret_cast<const unsigned short*>(biases);
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  uint4 qa = make_uint4(0u, 0u, 0u, 0u), qb = qa;
  float s = 0.f, b = 0.f;
  if (row_ok && g_begin < g_end) {
    const size_t base = (size_t)row * (K / 32) + g_begin * 2;
    qa = w4[base];
    qb = w4[base + 1];
    s = one_bf(ss[(size_t)row * groups + g_begin]);
    b = one_bf(bs[(size_t)row * groups + g_begin]);
  }
  for (unsigned int grp = g_begin; grp < g_end; ++grp) {
    uint4 na = make_uint4(0u, 0u, 0u, 0u), nb = na;
    float ns = 0.f, nbias = 0.f;
    if (row_ok && grp + 1 < g_end) {
      const size_t base = (size_t)row * (K / 32) + (grp + 1) * 2;
      na = w4[base];
      nb = w4[base + 1];
      ns = one_bf(ss[(size_t)row * groups + grp + 1]);
      nbias = one_bf(bs[(size_t)row * groups + grp + 1]);
    }
    const unsigned int words[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const unsigned int k0 = grp * 64 + j * 8;
      const unsigned int v = words[j] >> (8 * t);
      const unsigned int bfrag = pack_h2(fmaf(s, (float)(v & 0xFu), b), fmaf(s, (float)((v >> 4) & 0xFu), b));
      unsigned int a0 = 0u, a1 = 0u;
      if (g < M) {
        const unsigned int p = xw[(g * K + k0) / 2 + t];
        a0 = pack_h2(lo_bf(p), hi_bf(p));
      }
      if (g + 8 < M) {
        const unsigned int p = xw[((g + 8) * K + k0) / 2 + t];
        a1 = pack_h2(lo_bf(p), hi_bf(p));
      }
      if (j & 1) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(bfrag));
      } else {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(bfrag));
      }
    }
    qa = na;
    qb = nb;
    s = ns;
    b = nbias;
  }
  c0 += d0; c1 += d1; c2 += d2; c3 += d3;
  const unsigned int col = n0 + 2 * t;
  float* part = out + (size_t)split * M * N;
  if (g < M) {
    if (col < N) part[g * N + col] = c0;
    if (col + 1 < N) part[g * N + col + 1] = c1;
  }
  if (g + 8 < M) {
    if (col < N) part[(g + 8) * N + col] = c2;
    if (col + 1 < N) part[(g + 8) * N + col + 1] = c3;
  }
"""
MMA2_TARGET_WARPS = 1280  # 40 SMs x 32 resident warps on a T4
_mma2 = None


def mma2_qmm(x, w, scales, biases):
    """As `mma_qmm`, with prefetch, two mma chains and split-K; M <= 16 rows, K % 64 == 0."""
    global _mma2
    if _mma2 is None:
        _mma2 = mx.fast.cuda_kernel(name="perf1_mma4_v2", input_names=["x", "w", "scales", "biases"],
                                    output_names=["out"], source=MMA2_SOURCE, header=MMA_HEADER)
    n, k = w.shape[0], w.shape[1] * 8
    m = x.size // k
    tiles = -(-n // 8)
    splits = max(1, min(k // 64 // 2, -(-MMA2_TARGET_WARPS // tiles)))
    threads = -(-(tiles * splits * 32) // 256) * 256
    part = _mma2(inputs=[mx.contiguous(x.reshape(m, k)), w, scales, biases],
                 template=[("N", n), ("K", k), ("M", m), ("S", splits)], grid=(threads, 1, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(splits, m, n)], output_dtypes=[mx.float32])[0]
    out = part[0] if splits == 1 else part.sum(axis=0)
    return out.astype(mx.bfloat16).reshape(*x.shape[:-1], n)


def _source(variant):
    source = SOURCE.replace("@QF@", QF[os.environ.get("PERF1_QF", "cvt")])
    for key, value in ARITH[os.environ.get("PERF1_ARITH", "free")].items():
        source = source.replace(key, value)
    for key, value in VARIANTS[variant].items():
        source = source.replace(key, value)
    return source


def rows_for(n):
    # Fixed from run 1's probe: two rows won on the small shapes, four on gate/up, down and
    # lm_head. PERF1_ROWS overrides it for the probe.
    forced = os.environ.get("PERF1_ROWS")
    return int(forced) if forced else (4 if n >= 8192 else 2)


def qmv(x, w, scales, biases, rows=None):
    """x (..., K) with M <= 8 rows; w (N, K/8) uint32; scales/biases (N, K/64), all bf16 or all f32."""
    variant = "f32" if x.dtype == mx.float32 else "bf16"
    if variant not in _kernels:
        _kernels[variant] = mx.fast.cuda_kernel(
            name=f"perf1_qmv4_{variant}_{os.environ.get('PERF1_QF', 'cvt')}_{os.environ.get('PERF1_ARITH', 'free')}", input_names=["x", "w", "scales", "biases"],
            output_names=["out"], source=_source(variant), header=HEADER)
    n, k = w.shape[0], w.shape[1] * 8
    m = x.size // k
    rows = rows or rows_for(n)
    threads = -(-n // rows) * 32
    threads = -(-threads // 256) * 256  # whole blocks; surplus warps return before any work
    out_dtype = mx.float32 if variant == "f32" else mx.uint16
    out = _kernels[variant](
        inputs=[mx.contiguous(x.reshape(m, k)), w, scales, biases],
        template=[("N", n), ("K", k), ("GS", GS), ("R", rows), ("M", m)], grid=(threads, 1, 1),
        threadgroup=(256, 1, 1), output_shapes=[(m, n)], output_dtypes=[out_dtype])[0]
    if variant == "bf16":
        out = out.view(mx.bfloat16)
    return out.reshape(*x.shape[:-1], n)


_original = mx.quantized_matmul
routes = set()
routed = {"kernel": 0, "k32": 0, "p16": 0, "mma": 0, "mma2": 0, "fallback": 0}


def _patched(x, w, scales, biases=None, transpose=True, group_size=None, bits=None, mode="affine", **kw):
    k = x.shape[-1]
    if (biases is not None and transpose and (group_size or 64) == GS and (bits or 4) == 4
            and mode == "affine" and not kw and k % 32 == 0 and w.ndim == 2 and w.shape[1] * 8 == k
            and scales.dtype == x.dtype):
        rows = x.size // k
        if "mma2" in routes and x.dtype == mx.bfloat16 and rows <= MMA_MAX and k % 64 == 0:
            routed["mma2"] += 1
            return mma2_qmm(x, w, scales, biases)
        if "mma" in routes and x.dtype == mx.bfloat16 and MMA_MIN <= rows <= MMA_MAX and k % 64 == 0:
            routed["mma"] += 1
            return mma_qmm(x, w, scales, biases)
        if x.size <= MAX_M * k:
            if "kernel" in routes and x.dtype == mx.bfloat16:
                routed["kernel"] += 1
                return qmv(x, w, scales, biases)
            if "k32" in routes and x.dtype == mx.float32:
                routed["k32"] += 1
                return qmv(x, w, scales, biases)
        if x.size > (MMA_MAX if routes & {"mma", "mma2"} else MAX_M) * k and "p16" in routes:
            routed["p16"] += 1
            wd = mx.dequantize(w, scales.astype(mx.float16), biases.astype(mx.float16),
                               group_size=GS, bits=4)
            return mx.matmul(x.astype(mx.float16), wd.T).astype(x.dtype)
    routed["fallback"] += 1
    return _original(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits,
                     mode=mode, **kw)


def apply_arm(model, arm):
    parts = set(arm.split("+"))
    if not parts <= {"stock", "fp32", "kernel", "k32", "p16", "mma", "mma2"}:
        raise ValueError(arm)
    if "fp32" in parts:
        model.set_dtype(mx.float32)
    routes.update(parts & {"kernel", "k32", "p16", "mma", "mma2"})
    if routes:
        mx.quantized_matmul = _patched  # nn.QuantizedLinear looks it up per call


def prompt_ids(tokenizer):
    text = ("The history of computing is a history of moving work closer to the hardware that does "
            "it: from interpreted loops to compiled kernels, from general processors to units built "
            "for one operation. Summarise the argument and give three examples. ")
    ids = list(tokenizer.encode(text * 40))
    return ids[:PROMPT_TOKENS]


def generate(model, ids):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    began = time.perf_counter()
    y = mx.argmax(model(mx.array(ids)[None, :], cache=cache)[:, -1, :], axis=-1)
    mx.eval(y)
    first = time.perf_counter()
    tokens = []
    for _ in range(NEW_TOKENS - 1):
        nxt = mx.argmax(model(y.reshape(1, 1), cache=cache)[:, -1, :], axis=-1)
        mx.async_eval(nxt)
        tokens.append(int(y.item()))
        y = nxt
    tokens.append(int(y.item()))
    done = time.perf_counter()
    return tokens, (first - began) * 1000, (NEW_TOKENS - 1) / (done - first)


def measure(model, tokenizer, arm):
    apply_arm(model, arm)
    ids = prompt_ids(tokenizer)
    generate(model, ids)  # warmup: JIT, graph capture, allocator
    ttft, tps, tokens = [], [], None
    for _ in range(REPS):
        out, t, r = generate(model, ids)
        if tokens is not None and out != tokens:
            raise RuntimeError("tokens differ between repetitions of one arm")
        tokens = out
        ttft.append(round(t, 2))
        tps.append(round(r, 3))
    return {"arm": arm, "prompt_tokens": len(ids), "new_tokens": NEW_TOKENS, "ttft_ms": ttft,
            "ttft_ms_median": st.median(ttft), "decode_tps": tps, "decode_tps_median": st.median(tps),
            "tokens": tokens, "routed": dict(routed), "peak_memory_bytes": int(mx.get_peak_memory()),
            "device": str(mx.default_device()), "graph_env": os.environ.get("MLX_MAX_OPS_PER_BUFFER"),
            "performance_claim": False}


def nll(model, tokenizer, arm, path_mode, text_path):
    """Per-chunk mean next-token NLL, through the prefill path (one forward over the chunk) or
    the decode path (teacher-forced, one token at a time through the cache) — the latter is the
    only way the single-row kernel is exercised."""
    from mlx_lm.models.cache import make_prompt_cache

    apply_arm(model, arm)
    with open(text_path) as stream:
        ids = tokenizer.encode(stream.read())
    stride = (len(ids) - NLL_TOKENS - 1) // NLL_CHUNKS
    chunks, nonfinite = [], 0
    started = time.time()
    for i in range(NLL_CHUNKS):
        seq = ids[i * stride:i * stride + NLL_TOKENS + 1]
        if path_mode == "prefill":
            logits = model(mx.array(seq[:-1])[None, :])[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            per = -mx.take_along_axis(lp, mx.array(seq[1:])[:, None], axis=-1)[:, 0]
            values = [float(v) for v in per.tolist()]
        elif path_mode == "decode":
            cache = make_prompt_cache(model)
            values = []
            for t in range(NLL_TOKENS):
                logits = model(mx.array([[seq[t]]]), cache=cache)[0, -1].astype(mx.float32)
                value = mx.logsumexp(logits) - logits[seq[t + 1]]
                mx.eval(value)
                values.append(float(value.item()))
        else:
            raise ValueError(path_mode)
        nonfinite += sum(not math.isfinite(v) for v in values)
        chunks.append(sum(values) / len(values))
        print(i, chunks[-1], flush=True)
    return {"arm": arm, "path_mode": path_mode, "chunks": NLL_CHUNKS, "chunk_tokens": NLL_TOKENS,
            "chunk_nll": chunks, "mean_nll": sum(chunks) / len(chunks), "nonfinite": nonfinite,
            "routed": dict(routed), "seconds": round(time.time() - started, 1),
            "performance_claim": False}


SPEC_PROMPTS = (
    "Explain how a refrigerator moves heat out of its interior, step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Summarise the causes of the French Revolution in one paragraph.",
)


def chat_ids(tokenizer, text):
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                             add_generation_prompt=True)
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def spec(model, draft, tokenizer, arm, drafts):
    """Greedy speculative decoding (mlx-lm's own loop) against plain greedy decode of the same
    arm, on three natural chat prompts. Verification is exact by construction only if the
    verify step computes what single steps compute; `identical_to_plain` checks it."""
    from mlx_lm.generate import speculative_generate_step

    apply_arm(model, arm)
    if "fp32" in arm:
        draft.set_dtype(mx.float32)

    def run(ids, nd):
        began, first, tokens, accepted = time.perf_counter(), None, [], 0
        for token, _, from_draft in speculative_generate_step(mx.array(ids), model, draft,
                                                              num_draft_tokens=nd, max_tokens=NEW_TOKENS):
            first = first or time.perf_counter()
            tokens.append(int(token))
            accepted += bool(from_draft)
        done = time.perf_counter()
        return tokens, (first - began) * 1000, (len(tokens) - 1) / (done - first), accepted

    warm = chat_ids(tokenizer, SPEC_PROMPTS[0])
    generate(model, warm)
    for nd in drafts:
        run(warm, nd)
    prompts = []
    for text in SPEC_PROMPTS:
        ids = chat_ids(tokenizer, text)
        plain, ttft, tps = generate(model, ids)
        row = {"prompt_tokens": len(ids), "plain": {"ttft_ms": round(ttft, 2), "decode_tps": round(tps, 3)},
               "spec": {}}
        for nd in drafts:
            tokens, t, r, accepted = run(ids, nd)
            row["spec"][str(nd)] = {
                "ttft_ms": round(t, 2), "decode_tps": round(r, 3), "accepted_from_draft": accepted,
                "identical_to_plain": tokens == plain,
                "first_diff": next((i for i, (a, b) in enumerate(zip(tokens, plain)) if a != b), None)}
        prompts.append(row)
        print(json.dumps(row), flush=True)
    summary = {"plain_tps_median": st.median(p["plain"]["decode_tps"] for p in prompts)}
    for nd in drafts:
        summary[f"spec{nd}_tps_median"] = st.median(p["spec"][str(nd)]["decode_tps"] for p in prompts)
        summary[f"spec{nd}_acceptance"] = round(sum(p["spec"][str(nd)]["accepted_from_draft"] for p in prompts)
                                                / (len(prompts) * NEW_TOKENS), 4)
        summary[f"spec{nd}_identical"] = sum(p["spec"][str(nd)]["identical_to_plain"] for p in prompts)
    return {"arm": arm, "drafts": list(drafts), "prompts": prompts, "summary": summary,
            "routed": dict(routed), "peak_memory_bytes": int(mx.get_peak_memory()), "performance_claim": False}


def ironmule_arm(model_id, revision, arm, knobs, mode):
    """IronMule's own runtime and knobs on `ironmule benchmark`'s workload (6 strict requests x
    48 tokens, one warm pass, one measured), with the kernel routes of ARM installed first —
    the combination the product would ship. Same child as cross.py."""
    import ironmule
    from ironmule import benchmark as bench
    from ironmule.tune import load_engine

    parts = set(arm.split("+"))
    routes.update(parts & {"kernel", "k32", "p16", "mma", "mma2"})
    if routes:
        mx.quantized_matmul = _patched
    engine, tokenizer = load_engine(model_id, ironmule.Knobs(**knobs), revision=revision,
                                    compute_dtype="float32" if "fp32" in parts else None)
    runtime_mode = ironmule.ThroughputMode if mode == "throughput" else ironmule.InteractiveMode
    with ironmule.Runtime(engine, tokenizer, model_id=model_id) as rt:
        walls, outputs = [], None
        for index in range(3):
            requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", 6, 48)
            results, snapshot = bench._run(rt, ironmule, runtime_mode(), requests)
            outputs = [[int(t) for t in r.tokens] for r in results]
            if index:
                walls.append(snapshot["outer_wall_ms"])
    return {"arm": arm, "knobs": knobs, "runtime_mode": mode, "walls_ms": walls, "tokens": outputs,
            "routed": dict(routed), "peak_memory_bytes": int(mx.get_peak_memory()), "performance_claim": False}


def bench(fn, n=100):
    for _ in range(10):
        mx.eval(fn())
    began = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - began) / n * 1000


def kernel_probe(out):
    # Qwen 3 8B decode shapes: q/o, k/v, gate/up, down, lm_head. (N, K)
    shapes = {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (12288, 4096),
              "down": (4096, 12288), "lm_head": (151936, 4096)}
    per_layer = {"q_o": 2, "k_v": 2, "gate_up": 2, "down": 1}  # 36 layers
    report = {"schema": "ironmule.perf1-kernel.v3", "device": str(mx.default_device()), "shapes": {},
              "performance_claim": False}
    mx.random.seed(0)
    for name, (n, k) in shapes.items():
        wf = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        w, s, b = mx.quantize(wf, group_size=GS, bits=4)
        s32, b32 = s.astype(mx.float32), b.astype(mx.float32)
        x = mx.random.normal((1, k)).astype(mx.bfloat16)
        x32 = x.astype(mx.float32)
        ref = x32 @ mx.dequantize(w, s32, b32, group_size=GS, bits=4).T
        scale = mx.max(mx.abs(ref))
        row = {"N": n, "K": k}
        for label, xin, sin, bin_ in (("bf16", x, s, b), ("f32", x32, s32, b32)):
            for qf in QF:
                os.environ["PERF1_QF"] = qf
                _kernels.clear()
                try:
                    y = qmv(xin, w, sin, bin_).astype(mx.float32)
                    row[f"kernel_{label}_{qf}_rel_err"] = float(mx.max(mx.abs(y - ref)) / scale)
                    row[f"kernel_{label}_{qf}_ms"] = bench(lambda: qmv(xin, w, sin, bin_))
                    if label == "bf16":
                        for m in (2, 3, 4, 5):
                            xm = mx.random.normal((m, k)).astype(mx.bfloat16)
                            ym = qmv(xm, w, s, b).astype(mx.float32)
                            rows_1 = mx.concatenate([qmv(xm[i:i + 1], w, s, b) for i in range(m)]).astype(mx.float32)
                            row[f"kernel_bf16_{qf}_m{m}_equal_to_m1"] = bool(mx.array_equal(ym, rows_1))
                            row[f"kernel_bf16_{qf}_m{m}_ms"] = bench(lambda: qmv(xm, w, s, b))
                except Exception as exc:  # noqa: BLE001 — a kernel that does not compile is the result
                    row[f"kernel_{label}_{qf}_error"] = f"{type(exc).__name__}: {exc}"
        os.environ.pop("PERF1_QF", None)
        _kernels.clear()
        row["qmm_bf16_ms"] = bench(lambda: _original(x, w, s, b, transpose=True, group_size=GS, bits=4))
        row["qmm_fp32_ms"] = bench(lambda: _original(x32, w, s32, b32, transpose=True, group_size=GS, bits=4))
        # The prefill path at 512 rows: emulated bf16 against a float16 dequantise + GEMM.
        x512 = mx.random.normal((512, k)).astype(mx.bfloat16)
        ref512 = x512.astype(mx.float32) @ mx.dequantize(w, s32, b32, group_size=GS, bits=4).T
        routes.add("p16")
        y512 = _patched(x512, w, s, b, transpose=True, group_size=GS, bits=4).astype(mx.float32)
        routes.discard("p16")
        row["p16_rel_err"] = float(mx.max(mx.abs(y512 - ref512)) / mx.max(mx.abs(ref512)))
        row["p16_512_ms"] = bench(lambda: mx.matmul(x512.astype(mx.float16), mx.dequantize(
            w, s.astype(mx.float16), b.astype(mx.float16), group_size=GS, bits=4).T).astype(mx.bfloat16), 20)
        row["qmm_bf16_512_ms"] = bench(lambda: _original(x512, w, s, b, transpose=True, group_size=GS, bits=4), 5)
        report["shapes"][name] = row
        print(name, row, flush=True)
    keys = [f"kernel_{l}_{q}_ms" for l in ("bf16", "f32") for q in QF]
    keys += [f"kernel_bf16_{q}_m{m}_ms" for q in QF for m in (2, 3, 4, 5)]
    for key in keys + ["qmm_bf16_ms", "qmm_fp32_ms", "p16_512_ms", "qmm_bf16_512_ms"]:
        rows = report["shapes"]
        if all(key in rows[name] for name in shapes):
            report[f"step_{key}"] = round(sum(rows[n][key] * c for n, c in per_layer.items()) * 36
                                          + rows["lm_head"][key], 3)
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("step")}), flush=True)


SERVER_PROMPTS = (
    "Explain how a refrigerator moves heat out of its interior, step by step.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Summarise the causes of the French Revolution in one paragraph.",
    "What is the difference between TCP and UDP? Give two examples where each is the better choice.",
    "Draft a polite email asking a landlord to repair a broken heating system before winter.",
    "List five practical tips for learning a new language as an adult, and explain why each works.",
    "Describe the water cycle to a ten-year-old.",
    "Compare electric cars and petrol cars on cost, maintenance and environmental impact.",
)


def server(model, tokenizer, arm, widths):
    """Eight different chat requests through mlx-lm's continuous-batching BatchGenerator at
    several concurrency widths, like a server holding that many requests open. Greedy, exactly
    NEW_TOKENS each (no stop tokens, so every width does the same work). Width 1 is the
    sequential server; its tokens are the reference each request must reproduce when batched."""
    from mlx_lm.generate import BatchGenerator

    apply_arm(model, arm)
    prompts = [chat_ids(tokenizer, text) for text in SERVER_PROMPTS]

    def serve(width):
        gen = BatchGenerator(model, max_tokens=NEW_TOKENS, completion_batch_size=width,
                             prefill_batch_size=width)
        began = time.perf_counter()
        uids = gen.insert(prompts, [NEW_TOKENS] * len(prompts))
        tokens, first = {u: [] for u in uids}, {}
        while responses := gen.next_generated():
            for r in responses:
                first.setdefault(r.uid, time.perf_counter() - began)
                tokens[r.uid].append(int(r.token))
        wall = time.perf_counter() - began
        gen.close()
        return [tokens[u] for u in uids], [first[u] * 1000 for u in uids], wall

    serve(max(widths))  # warmup: JIT for every row count this run will see
    reference, rows = None, {}
    for width in widths:
        outs, ttft, wall = serve(width)
        reference = reference or outs
        rows[str(width)] = {
            "wall_s": round(wall, 3), "aggregate_tps": round(len(prompts) * NEW_TOKENS / wall, 3),
            "ttft_ms": [round(t, 1) for t in ttft], "ttft_ms_median": round(st.median(ttft), 1),
            "identical_to_width1": sum(a == b for a, b in zip(outs, reference))}
        print(width, rows[str(width)], flush=True)
    return {"arm": arm, "requests": len(prompts), "new_tokens": NEW_TOKENS, "widths": rows,
            "arith": os.environ.get("PERF1_ARITH", "free"), "routed": dict(routed),
            "peak_memory_bytes": int(mx.get_peak_memory()), "performance_claim": False}


def chain_probe(out):
    """GPU time of one Qwen 3 8B decode step's matmuls alone: 36 layers x 7 projections plus the
    head, queued without a sync and evaluated once — against the end-to-end step time, it says
    how much of a token is matmul and how much is everything else."""
    shapes = {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (12288, 4096),
              "down": (4096, 12288), "lm_head": (151936, 4096)}
    per_layer = {"q_o": 2, "k_v": 2, "gate_up": 2, "down": 1}
    mx.random.seed(0)
    mats, xs = {}, {}
    for name, (n, k) in shapes.items():
        mats[name] = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16), group_size=GS, bits=4)
        xs[name] = mx.random.normal((1, k)).astype(mx.bfloat16)
    mx.eval(mats, xs)
    report = {"schema": "ironmule.perf1-chain.v1", "device": str(mx.default_device()), "performance_claim": False}
    for label, fn in (("kernel", lambda x, w, s, b: qmv(x, w, s, b)),
                      ("qmm_bf16", lambda x, w, s, b: _original(x, w, s, b, transpose=True, group_size=GS, bits=4))):
        def step():
            outs = [fn(xs[name], *mats[name]) for _ in range(36) for name, c in per_layer.items() for _ in range(c)]
            return outs + [fn(xs["lm_head"], *mats["lm_head"])]
        for _ in range(3):
            mx.eval(step())
        times = []
        for _ in range(10):
            began = time.perf_counter()
            mx.eval(step())
            times.append((time.perf_counter() - began) * 1000)
        report[f"{label}_step_ms"] = [round(t, 3) for t in times]
        report[f"{label}_step_ms_median"] = round(st.median(times), 3)
        print(label, report[f"{label}_step_ms_median"], flush=True)
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1)


def mma_probe(out):
    """The tensor-core kernel against the row kernel: correctness per Qwen 3 8B shape at M = 1, 8,
    16, and one decode step's matmuls (36 layers x 7 + head, one eval) at M = 1, 4, 8, 16."""
    shapes = {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (12288, 4096),
              "down": (4096, 12288), "lm_head": (151936, 4096)}
    per_layer = {"q_o": 2, "k_v": 2, "gate_up": 2, "down": 1}
    mx.random.seed(0)
    mats = {}
    for name, (n, k) in shapes.items():
        mats[name] = mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16), group_size=GS, bits=4)
    mx.eval(mats)
    report = {"schema": "ironmule.perf1-mma.v1", "device": str(mx.default_device()), "shapes": {}, "steps": {},
              "performance_claim": False}
    for name, (n, k) in shapes.items():
        w, sc, bi = mats[name]
        ref_w = mx.dequantize(w, sc.astype(mx.float32), bi.astype(mx.float32), group_size=GS, bits=4)
        row = {}
        for m in (1, 8, 16):
            x = mx.random.normal((m, k)).astype(mx.bfloat16)
            ref = x.astype(mx.float32) @ ref_w.T
            for label, fn in (("mma", mma_qmm), ("mma2", mma2_qmm)):
                try:
                    y = fn(x, w, sc, bi).astype(mx.float32)
                    row[f"{label}_m{m}_rel_err"] = float(mx.max(mx.abs(y - ref)) / mx.max(mx.abs(ref)))
                except Exception as exc:  # noqa: BLE001 — a kernel that does not compile is the result
                    row[f"{label}_m{m}_error"] = f"{type(exc).__name__}: {exc}"
        report["shapes"][name] = row
        print(name, row, flush=True)
    fns = {"mma2": mma2_qmm, "mma": mma_qmm, "kernel": lambda x, w, s_, b_: qmv(x, w, s_, b_)}
    for label, fn in fns.items():
        for m in (1, 4, 8, 16):
            if (label == "kernel" and m > 1) or (label == "mma" and m not in (1, 8)):
                continue  # measured in run 8; only the controls this comparison needs
            xs = {name: mx.random.normal((m, k)).astype(mx.bfloat16) for name, (n, k) in shapes.items()}
            def step():
                outs = [fn(xs[name], *mats[name]) for _ in range(36) for name, c in per_layer.items() for _ in range(c)]
                return outs + [fn(xs["lm_head"], *mats["lm_head"])]
            try:
                for _ in range(3):
                    mx.eval(step())
                times = []
                for _ in range(8):
                    began = time.perf_counter()
                    mx.eval(step())
                    times.append((time.perf_counter() - began) * 1000)
                report["steps"][f"{label}_m{m}_ms"] = round(st.median(times), 3)
            except Exception as exc:  # noqa: BLE001
                report["steps"][f"{label}_m{m}_error"] = f"{type(exc).__name__}: {exc}"
            print(label, m, report["steps"].get(f"{label}_m{m}_ms"), flush=True)
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1)


def main():
    mode = sys.argv[1]
    if mode == "kernel":
        return kernel_probe(sys.argv[2])
    if mode == "chain":
        return chain_probe(sys.argv[2])
    if mode == "mma":
        return mma_probe(sys.argv[2])
    if mode == "ironmule":
        model_id, revision, arm, knobs, runtime_mode, out = sys.argv[2:8]
        report = ironmule_arm(model_id, revision, arm, json.loads(knobs), runtime_mode)
    else:
        from mlx_lm import load

        path, arm = sys.argv[2:4]
        model, tokenizer = load(path)
        if mode == "e2e":
            out = sys.argv[4]
            report = measure(model, tokenizer, arm)
        elif mode == "nll":
            path_mode, text_path, out = sys.argv[4:7]
            report = nll(model, tokenizer, arm, path_mode, text_path)
        elif mode == "server":
            widths, out = sys.argv[4:6]
            report = server(model, tokenizer, arm, [int(w) for w in widths.split(",")])
        elif mode == "spec":
            draft_path, drafts, out = sys.argv[4:7]
            draft, _ = load(draft_path)
            report = spec(model, draft, tokenizer, arm, [int(d) for d in drafts.split(",")])
        else:
            raise SystemExit(__doc__)
        report["model_path"] = path
    report["mode"] = mode
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("tokens", "prompts")}), flush=True)


if __name__ == "__main__":
    main()
