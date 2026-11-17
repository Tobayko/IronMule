"""KG1: CUDA kernels generated and tuned for the GPU they run on, for llama.cpp's Q4_K weights.

llama.cpp computes a quantized matmul of 1-8 tokens with one fixed kernel per weight type (`mul_mat_vec_q`), tuned once
for all GPUs. Here a family of kernels is generated from parameters, every member is checked and timed on this GPU with
this model's own weights, and the fastest one for each matrix shape and token count goes into a small library that
llama.cpp calls through the `kg1` hook (`native/ggml_patches/kg1`, `IRONMULE_KG_LIB`); for anything the library has no
kernel for, llama.cpp's own kernels run as before.

A kernel is (L, R, W, B): L lanes share one 256-weight block (each reads 128 / L bytes of quants at once), a warp
computes R output rows (the activations it loaded serve all of them), W warps per thread block, B tokens at once. The
activations are rounded to 8 bits per 32 values, as llama.cpp's own kernels do, and four multiply-adds run per
`__dp4a`. A candidate serves only if its error against a float64 reference (which decodes the format element by
element) is within 1.25x llama.cpp's own kernel's on the same data, and only where it is faster than llama.cpp's
kernel in the engine's own step (the engine measures that at startup and switches the library off per token count).

Measured on a Kaggle T4 (docs/LAB.md, KG1 v2): all Q4_K matmuls of a Qwen3-4B token 1.06x / 1.50x / 1.69x llama.cpp's
time at 1 / 4 / 8 tokens.

Needs `nvcc` (the CUDA toolkit llama.cpp was built with) on the machine; without it nothing is generated.
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

KG_VERSION = 5  # 2 (ST2): MQ1 kernels for one token; 3 (KG4): tensor-core kernels for 1-32 tokens; 4: and Q6_K;
# 5 (KG5): decode attention through the kg5 hook
TYPE_Q4_K = 12  # GGML_TYPE_Q4_K
BATCHES = tuple(range(1, 9))  # from 9 tokens on llama.cpp's tensor-core kernel is faster (KG1 v2: 16.8 ms vs 42.4)
LANES, ROWS, WARPS = (8, 16, 32), (1, 2, 4, 8), (2, 4, 8)
ERROR_MARGIN = 1.25  # a kernel may be at most this much less exact than llama.cpp's own on the same data
# KG4: Q4_K x 8-bit activations on tensor cores (mma.sync m8n8k16, sm_75 and later), T tiles of 8 tokens; measured
# at the largest token count of each tile count and serving every count of that tile count
MMA_BATCHES = (8, 16, 24, 32)
MMA_SHAPES = ((1, 4, 2, 1), (2, 4, 2, 1), (1, 8, 2, 1), (2, 8, 2, 1), (1, 4, 4, 1), (2, 4, 4, 1), (1, 8, 4, 1),
              (1, 4, 2, 2), (2, 4, 2, 2), (1, 8, 2, 2), (1, 4, 2, 4), (1, 8, 2, 4), (1, 4, 4, 2), (1, 4, 1, 1),
              (1, 4, 1, 4), (2, 2, 2, 1), (1, 4, 8, 1), (1, 8, 8, 1), (2, 4, 8, 1), (1, 4, 8, 2))  # R, W, KB, SK
# T, R, W, KB, SK; T * KB <= 16 keeps a thread block's staged activations under 40 KB
MMA_VARIANTS = tuple((T,) + v for T in (1, 2, 3, 4) for v in MMA_SHAPES if T * v[2] <= 16)
MMA6_SHAPES = ((1, 4, 2, 1), (2, 4, 2, 1), (1, 8, 2, 1), (2, 8, 2, 1), (1, 4, 4, 1), (1, 8, 4, 1), (1, 4, 2, 2),
               (1, 8, 2, 2), (1, 4, 2, 4), (1, 8, 2, 4), (2, 4, 4, 1), (1, 4, 8, 1))  # R, W, KB, SK
MMA6_VARIANTS = tuple((T,) + v for T in (1, 2, 3, 4) for v in MMA6_SHAPES if T * v[2] <= 16)
TYPE_Q6_K = 14  # GGML_TYPE_Q6_K

DEVICE = r"""
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <mutex>
#include <vector>

struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(block_q4_K) == 144, "Q4_K block");

__device__ __forceinline__ uint32_t sbyte(uint32_t y, uint32_t z, uint32_t w, int k) {
    const uint32_t word = k < 4 ? y : (k < 8 ? z : w);
    return (word >> (8 * (k & 3))) & 0xFF;
}
// the 6-bit scale and min of sub-block j (get_scale_min_k4 of ggml-quants.c) from bytes 4..15 of a block
__device__ __forceinline__ void scale_min(uint32_t y, uint32_t z, uint32_t w, int j, float &sc, float &m) {
    if (j < 4) {
        sc = (float)(sbyte(y, z, w, j) & 63); m = (float)(sbyte(y, z, w, j + 4) & 63);
    } else {
        sc = (float)((sbyte(y, z, w, j + 4) & 0xF) | ((sbyte(y, z, w, j - 4) >> 6) << 4));
        m = (float)((sbyte(y, z, w, j + 4) >> 4) | ((sbyte(y, z, w, j) >> 6) << 4));
    }
}

// activations to 8 bits per 32 values: one warp per 32 values
__global__ void quant_x(const float *__restrict__ x, int8_t *__restrict__ xq, float *__restrict__ xd, int n) {
    const int blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (blk * 32 >= n) return;
    const float v = x[blk * 32 + lane];
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const float d = a / 127.f;
    xq[blk * 32 + lane] = (int8_t)(d == 0.f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

template <int L, int R, int W, int B>
__global__ void __launch_bounds__(W * 32) kq4k(const block_q4_K *__restrict__ wq, const int8_t *__restrict__ xq,
                                               const float *__restrict__ xd, float *__restrict__ y, int N, int K) {
    constexpr int G = 32 / L, BYTES = 128 / L, LPC = L / 4, WORDS = BYTES / 4;
    const int nb = K / 256;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * W + warp) * R;
    if (row0 >= N) return;
    const int g = lane / L, t = lane % L;
    const int c = t / LPC, l0 = (t % LPC) * BYTES;
    float acc[R][B];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < B; b++) acc[r][b] = 0.f;
    for (int ib = g; ib < nb; ib += G) {
        int xl[B][WORDS], xh[B][WORDS];
        float dl[B], dh[B], sl[B], sh[B];
#pragma unroll
        for (int b = 0; b < B; b++) {
            const int8_t *pl = xq + (size_t)b * K + ib * 256 + 64 * c + l0, *ph = pl + 32;
            if constexpr (WORDS == 4) {
                const int4 a = __ldg(reinterpret_cast<const int4 *>(pl)), h = __ldg(reinterpret_cast<const int4 *>(ph));
                xl[b][0] = a.x; xl[b][1] = a.y; xl[b][2] = a.z; xl[b][3] = a.w;
                xh[b][0] = h.x; xh[b][1] = h.y; xh[b][2] = h.z; xh[b][3] = h.w;
            } else if constexpr (WORDS == 2) {
                const int2 a = __ldg(reinterpret_cast<const int2 *>(pl)), h = __ldg(reinterpret_cast<const int2 *>(ph));
                xl[b][0] = a.x; xl[b][1] = a.y; xh[b][0] = h.x; xh[b][1] = h.y;
            } else {
                xl[b][0] = __ldg(reinterpret_cast<const int *>(pl)); xh[b][0] = __ldg(reinterpret_cast<const int *>(ph));
            }
            const size_t sb = (size_t)b * (K / 32) + ib * 8 + 2 * c;
            dl[b] = __ldg(xd + sb); dh[b] = __ldg(xd + sb + 1);
            int il = 0, ih = 0;
#pragma unroll
            for (int w = 0; w < WORDS; w++) { il = __dp4a(0x01010101, xl[b][w], il); ih = __dp4a(0x01010101, xh[b][w], ih); }
            sl[b] = dl[b] * (float)il; sh[b] = dh[b] * (float)ih;
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r >= N) break;
            const block_q4_K *blk = wq + (size_t)(row0 + r) * nb + ib;
            const uint4 hdr = __ldg(reinterpret_cast<const uint4 *>(blk));
            const float d = __half2float(__ushort_as_half((unsigned short)(hdr.x & 0xFFFF)));
            const float dm = __half2float(__ushort_as_half((unsigned short)(hdr.x >> 16)));
            float sc1, m1, sc2, m2;
            scale_min(hdr.y, hdr.z, hdr.w, 2 * c, sc1, m1);
            scale_min(hdr.y, hdr.z, hdr.w, 2 * c + 1, sc2, m2);
            const float d1 = d * sc1, n1 = dm * m1, d2 = d * sc2, n2 = dm * m2;
            int q[WORDS];
            const uint8_t *qp = blk->qs + 32 * c + l0;
            if constexpr (WORDS == 4) {
                const int4 v = __ldg(reinterpret_cast<const int4 *>(qp)); q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
            } else if constexpr (WORDS == 2) {
                const int2 v = __ldg(reinterpret_cast<const int2 *>(qp)); q[0] = v.x; q[1] = v.y;
            } else {
                q[0] = __ldg(reinterpret_cast<const int *>(qp));
            }
#pragma unroll
            for (int b = 0; b < B; b++) {
                int il = 0, ih = 0;
#pragma unroll
                for (int w = 0; w < WORDS; w++) {
                    il = __dp4a(q[w] & 0x0F0F0F0F, xl[b][w], il);
                    ih = __dp4a((q[w] >> 4) & 0x0F0F0F0F, xh[b][w], ih);
                }
                acc[r][b] += (d1 * dl[b] * (float)il - n1 * sl[b]) + (d2 * dh[b] * (float)ih - n2 * sh[b]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < B; b++) {
            float v = acc[r][b];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0 && row0 + r < N) y[(size_t)b * N + row0 + r] = v;
        }
}

// MQ1 (ST2): Q4_K repacked once so that lane j of a warp reads sub-block j's 16 bytes (512 contiguous bytes per warp
// and iteration), float32 activations as they are; one token. repack_q4k: one thread per 256-weight block.
__global__ void repack_q4k(const block_q4_K *__restrict__ src, uint4 *__restrict__ qs, uint4 *__restrict__ hdr,
                           size_t nblocks) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const block_q4_K &b = src[i];
    hdr[i] = *reinterpret_cast<const uint4 *>(&b);
    for (int j = 0; j < 8; j++) {  // sub-block j: weights 32j .. 32j + 31, chunk c = j / 2, low or high nibbles
        const int c = j / 2, high = j & 1;
        uint32_t words[4];
        for (int k = 0; k < 4; k++) {
            uint32_t v = 0;
            for (int t = 0; t < 4; t++) {
                const int i0 = 2 * (4 * k + t);  // weights i0 and i0 + 1 of the sub-block: bytes l = i0, i0 + 1
                const uint32_t a = high ? (b.qs[32 * c + i0] >> 4) : (b.qs[32 * c + i0] & 0xF);
                const uint32_t bb = high ? (b.qs[32 * c + i0 + 1] >> 4) : (b.qs[32 * c + i0 + 1] & 0xF);
                v |= (a | (bb << 4)) << (8 * t);
            }
            words[k] = v;
        }
        qs[8 * i + j] = make_uint4(words[0], words[1], words[2], words[3]);
    }
}

template <int W, int R, int U>
__global__ void __launch_bounds__(W * 32) mq1(const uint4 *__restrict__ qs, const uint4 *__restrict__ hdr,
                                              const float *__restrict__ x, float *__restrict__ y, int N, int K) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * W + warp) * R;
    if (row0 >= N) return;
    const int nsub = K / 32, nblk = K / 256;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.f;
    for (int base = 0; base < nsub; base += 32 * U) {
        uint4 q[U][R], h[U][R];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int j = base + 32 * u + lane;
#pragma unroll
            for (int r = 0; r < R; r++)
                if (j < nsub && row0 + r < N) {
                    q[u][r] = __ldg(qs + (size_t)(row0 + r) * nsub + j);
                    h[u][r] = __ldg(hdr + (size_t)(row0 + r) * nblk + j / 8);
                }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int j = base + 32 * u + lane;
            if (j >= nsub) break;
            float xs[32], sx = 0.f;
            const float4 *px = reinterpret_cast<const float4 *>(x + 32 * (size_t)j);
#pragma unroll
            for (int v = 0; v < 8; v++) {
                const float4 a = __ldg(px + v);
                xs[4 * v] = a.x; xs[4 * v + 1] = a.y; xs[4 * v + 2] = a.z; xs[4 * v + 3] = a.w;
                sx += (a.x + a.y) + (a.z + a.w);
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (row0 + r >= N) break;
                const uint4 hh = h[u][r];
                const float d = __half2float(__ushort_as_half((unsigned short)(hh.x & 0xFFFF)));
                const float dm = __half2float(__ushort_as_half((unsigned short)(hh.x >> 16)));
                float sc, mn;
                scale_min(hh.y, hh.z, hh.w, j % 8, sc, mn);
                const uint32_t words[4] = {q[u][r].x, q[u][r].y, q[u][r].z, q[u][r].w};
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    const uint32_t byte = (words[i / 4] >> (8 * (i & 3))) & 0xFF;
                    s = fmaf((float)(byte & 0xF), xs[2 * i], s);
                    s = fmaf((float)(byte >> 4), xs[2 * i + 1], s);
                }
                acc[r] += d * sc * s - dm * mn * sx;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && row0 + r < N) y[row0 + r] = v;
    }
}
typedef void (*mq_launch_t)(const uint4 *, const uint4 *, const float *, float *, int, int, cudaStream_t);
template <int W, int R, int U>
void launch_mq(const uint4 *qs, const uint4 *hdr, const float *x, float *y, int N, int K, cudaStream_t s) {
    mq1<W, R, U><<<(N + W * R - 1) / (W * R), W * 32, 0, s>>>(qs, hdr, x, y, N, K);
}
struct MqVariant { int W, R, U; mq_launch_t fn; };

typedef void (*launch_t)(const block_q4_K *, const int8_t *, const float *, float *, int, int, cudaStream_t);
template <int L, int R, int W, int B>
void launch(const block_q4_K *w, const int8_t *xq, const float *xd, float *y, int N, int K, cudaStream_t s) {
    const int rows = R * W;
    kq4k<L, R, W, B><<<(N + rows - 1) / rows, W * 32, 0, s>>>(w, xq, xd, y, N, K);
}
static void quantize(const float *x, int8_t *xq, float *xd, int n, cudaStream_t s) {
    quant_x<<<(n / 32 + 7) / 8, 256, 0, s>>>(x, xq, xd, n);
}
struct Variant { int L, R, W, B; launch_t fn; };

// KG4 (lab/kg4/kmma.cu): Q4_K x 8-bit activations on tensor cores, 1-32 tokens
__device__ __forceinline__ void mma8816(int &c0, int &c1, uint32_t a, uint32_t b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}

// an int32 of magnitude below 2^22 as float, exactly: an integer add and a float add instead of I2F (a quarter-rate
// instruction on Turing)
__device__ __forceinline__ float i2f(int v) { return __int_as_float(0x4B400000 + v) - 12582912.0f; }

// x [B][K] float -> fragment order for T tiles of 8 tokens (tokens past B are zero):
//   xf, 8 bytes per (tile tt, sub-block s, lane): xq[8 tt + lane / 4][32 s + 8 (lane % 4) .. + 7]
//   xs, per (tt, s, token in tile): (d, d * sum of the 32 quants) as half2, as llama.cpp's q8_1 keeps them
//   layout 1 (Q6_K, whose scales cover 16 weights): lane c holds 32 s + 4c .. + 3 and 32 s + 16 + 4c .. + 3 instead
__global__ void quant_frag(const float *__restrict__ x, int8_t *__restrict__ xf, half2 *__restrict__ xs, int B, int K,
                           int T, int layout) {
    const int S = K / 32;
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (wid >= T * 8 * S) return;
    const int tok = wid / S, s = wid % S;
    const float v = tok < B ? x[(size_t)tok * K + 32 * s + lane] : 0.f;
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const float d = a / 127.f;
    const int q = d == 0.f ? 0 : __float2int_rn(v / d);
    int sum = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    const int tt = tok / 8, g = tok % 8;
    const int at = layout ? (lane % 16) / 4 : lane / 8, byte = layout ? (lane / 16) * 4 + lane % 4 : lane % 8;
    xf[(((size_t)tt * S + s) * 32 + g * 4 + at) * 8 + byte] = (int8_t)q;
    if (lane == 0) xs[((size_t)tt * S + s) * 8 + g] = __floats2half2_rn(d, d * (float)sum);
}

// one 144-byte Q4_K block of the row this lane serves: its header and its 4 x 8 bytes of quants
struct Q4Row { uint4 h; uint2 q[4]; };
__device__ __forceinline__ Q4Row load_row(const block_q4_K *__restrict__ wq, int row, int N, int nb, int ib, int c) {
    Q4Row r;
    if (row < N) {
        const block_q4_K *blk = wq + (size_t)row * nb + ib;
        r.h = __ldg(reinterpret_cast<const uint4 *>(blk));
#pragma unroll
        for (int grp = 0; grp < 4; grp++) r.q[grp] = __ldg(reinterpret_cast<const uint2 *>(blk->qs + 32 * grp + 8 * c));
    } else {
        r.h = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int grp = 0; grp < 4; grp++) r.q[grp] = make_uint2(0, 0);
    }
    return r;
}

template <int T, int R, int W, int KB, int SK>
__global__ void __launch_bounds__(W * 32) kmma(const block_q4_K *__restrict__ wq, const uint2 *__restrict__ xf,
                                               const half2 *__restrict__ xs, float *__restrict__ y, int N, int K,
                                               int B) {
    __shared__ uint2 sx[T * KB * 256];
    __shared__ half2 ss[T * KB * 64];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int S = K / 32, nb = K / 256;
    const int rbase = (blockIdx.x * W + warp) * R * 8;
    const int per = (nb + SK - 1) / SK, kb0 = blockIdx.y * per, kb1 = min(nb, kb0 + per);
    float acc[R][T][2];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t][0] = acc[r][t][1] = 0.f;
    Q4Row cur[R];  // the next block's weights are loaded while this one is computed
#pragma unroll
    for (int r = 0; r < R; r++) cur[r] = load_row(wq, rbase + r * 8 + g, N, nb, kb0 < kb1 ? kb0 : 0, c);
    for (int sb0 = kb0; sb0 < kb1; sb0 += KB) {
        const int nk = min(KB, kb1 - sb0);
        __syncthreads();
        for (int i = threadIdx.x; i < T * nk * 256; i += W * 32) {
            const int tt = i / (nk * 256), j = i % (nk * 256);
            sx[tt * KB * 256 + j] = xf[((size_t)tt * S + sb0 * 8) * 32 + j];
        }
        for (int i = threadIdx.x; i < T * nk * 64; i += W * 32) {
            const int tt = i / (nk * 64), j = i % (nk * 64);
            ss[tt * KB * 64 + j] = xs[((size_t)tt * S + sb0 * 8) * 8 + j];
        }
        __syncthreads();
        for (int kb = 0; kb < nk; kb++) {
            const int ib = sb0 + kb;
            Q4Row nxt[R];
#pragma unroll
            for (int r = 0; r < R; r++) nxt[r] = load_row(wq, rbase + r * 8 + g, N, nb, ib + 1 < kb1 ? ib + 1 : ib, c);
#pragma unroll
            for (int s = 0; s < 8; s++) {
                const int grp = s / 2, sh = (s & 1) * 4;
                float dsc[R], dmn[R];
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float sc, m;
                    scale_min(cur[r].h.y, cur[r].h.z, cur[r].h.w, s, sc, m);
                    dsc[r] = __half2float(__ushort_as_half((unsigned short)(cur[r].h.x & 0xFFFF))) * sc;
                    dmn[r] = __half2float(__ushort_as_half((unsigned short)(cur[r].h.x >> 16))) * m;
                }
#pragma unroll
                for (int tt = 0; tt < T; tt++) {
                    const int at = (tt * KB + kb) * 8 + s;
                    const uint2 b = sx[at * 32 + lane];
                    const float2 s0 = __half22float2(ss[at * 8 + 2 * c]), s1 = __half22float2(ss[at * 8 + 2 * c + 1]);
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        int c0 = 0, c1 = 0;
                        mma8816(c0, c1, (cur[r].q[grp].x >> sh) & 0x0F0F0F0Fu, b.x);
                        mma8816(c0, c1, (cur[r].q[grp].y >> sh) & 0x0F0F0F0Fu, b.y);
                        acc[r][tt][0] = fmaf(dsc[r], s0.x * i2f(c0), fmaf(-dmn[r], s0.y, acc[r][tt][0]));
                        acc[r][tt][1] = fmaf(dsc[r], s1.x * i2f(c1), fmaf(-dmn[r], s1.y, acc[r][tt][1]));
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; r++) cur[r] = nxt[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = rbase + r * 8 + g;
        if (row >= N) continue;
#pragma unroll
        for (int tt = 0; tt < T; tt++)
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int tok = tt * 8 + 2 * c + i;
                if (tok >= B) continue;
                if (SK == 1) y[(size_t)tok * N + row] = acc[r][tt][i];
                else atomicAdd(y + (size_t)tok * N + row, acc[r][tt][i]);
            }
    }
}

typedef void (*mma_launch_t)(const block_q4_K *, const uint2 *, const half2 *, float *, int, int, int, cudaStream_t);
template <int T, int R, int W, int KB, int SK>
void launch_mma(const block_q4_K *w, const uint2 *xf, const half2 *xs, float *y, int N, int K, int B, cudaStream_t s) {
    if (SK > 1) cudaMemsetAsync(y, 0, (size_t)B * N * sizeof(float), s);
    kmma<T, R, W, KB, SK><<<dim3((N + W * R * 8 - 1) / (W * R * 8), SK), W * 32, 0, s>>>(w, xf, xs, y, N, K, B);
}
static void quantize_frag(const float *x, int8_t *xf, half2 *xs, int B, int K, int T, cudaStream_t s, int layout = 0) {
    const int warps = T * 8 * (K / 32);
    quant_frag<<<(warps + 7) / 8, 256, 0, s>>>(x, xf, xs, B, K, T, layout);
}
struct MmaVariant { int T, R, W, KB, SK; mma_launch_t fn; };

// KG4 for Q6_K (the output matrix and some FFN layers): 210-byte blocks of 256 weights, 6-bit quants (4 low bits in ql,
// 2 high in qh) less 32, one int8 scale per 16 weights, one half scale per block. A tensor-core chunk of 16 weights is
// one scale group, so the two chunks of a 32-weight group are scaled in int32 and turned into float once.
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(block_q6_K) == 210, "Q6_K block");
__device__ __forceinline__ uint32_t ld32a2(const uint8_t *p) {  // blocks of 210 bytes: only 2-byte aligned
    const uint16_t *q = reinterpret_cast<const uint16_t *>(p);
    return (uint32_t)__ldg(q) | ((uint32_t)__ldg(q + 1) << 16);
}
struct Q6Row { uint32_t qa[2][2], qb[2][2], qh[2][2], sc[4]; float d; };
// Q6_K repacked once for the tensor cores (KG5): 224-byte blocks (14 x 16 bytes): lane c's 12 words at 48 c (qa, qb, qh
// for n = 0, 1 and w = 0, 1), the 16 scales at 192, d at 208. 5 vector loads per lane and block instead of 33 2-byte
// loads from the 210-byte blocks.
__global__ void repack_q6k(const block_q6_K *__restrict__ src, uint4 *__restrict__ dst, size_t nblocks) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const block_q6_K &b = src[i];
    uint32_t out[56];
    auto w32 = [&](const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); };
    for (int c = 0; c < 4; c++)
        for (int n = 0; n < 2; n++)
            for (int w = 0; w < 2; w++) {
                out[12 * c + 2 * n + w] = w32(b.ql + 64 * n + 16 * w + 4 * c);
                out[12 * c + 4 + 2 * n + w] = w32(b.ql + 64 * n + 32 + 16 * w + 4 * c);
                out[12 * c + 8 + 2 * n + w] = w32(b.qh + 32 * n + 16 * w + 4 * c);
            }
    for (int k = 0; k < 4; k++) out[48 + k] = w32(reinterpret_cast<const uint8_t *>(b.scales) + 4 * k);
    out[52] = b.d; out[53] = out[54] = out[55] = 0;
    for (int k = 0; k < 14; k++) dst[14 * i + k] = make_uint4(out[4 * k], out[4 * k + 1], out[4 * k + 2], out[4 * k + 3]);
}
__device__ __forceinline__ Q6Row load_row6(const uint4 *__restrict__ wr, int row, int N, int nb, int ib, int c) {
    Q6Row r;
    if (row < N) {
        const uint4 *blk = wr + ((size_t)row * nb + ib) * 14;
        const uint4 a = __ldg(blk + 3 * c), b = __ldg(blk + 3 * c + 1), e = __ldg(blk + 3 * c + 2), sc = __ldg(blk + 12);
        r.qa[0][0] = a.x; r.qa[0][1] = a.y; r.qa[1][0] = a.z; r.qa[1][1] = a.w;
        r.qb[0][0] = b.x; r.qb[0][1] = b.y; r.qb[1][0] = b.z; r.qb[1][1] = b.w;
        r.qh[0][0] = e.x; r.qh[0][1] = e.y; r.qh[1][0] = e.z; r.qh[1][1] = e.w;
        r.sc[0] = sc.x; r.sc[1] = sc.y; r.sc[2] = sc.z; r.sc[3] = sc.w;
        r.d = __half2float(__ushort_as_half((unsigned short)(__ldg(reinterpret_cast<const uint32_t *>(blk + 13)) & 0xFFFF)));
    } else {
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int w = 0; w < 2; w++) r.qa[n][w] = r.qb[n][w] = r.qh[n][w] = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = 0;
        r.d = 0.f;
    }
    return r;
}
__device__ __forceinline__ int sc8(const uint32_t *sc, int i) { return (int)(int8_t)((sc[i >> 2] >> (8 * (i & 3))) & 0xFF); }

template <int T, int R, int W, int KB, int SK>
__global__ void __launch_bounds__(W * 32) kmma6(const uint4 *__restrict__ wq, const uint2 *__restrict__ xf,
                                                const half2 *__restrict__ xs, float *__restrict__ y, int N, int K,
                                                int B) {
    __shared__ uint2 sx[T * KB * 256];
    __shared__ half2 ss[T * KB * 64];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int S = K / 32, nb = K / 256;
    const int rbase = (blockIdx.x * W + warp) * R * 8;
    const int per = (nb + SK - 1) / SK, kb0 = blockIdx.y * per, kb1 = min(nb, kb0 + per);
    float acc[R][T][2];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t][0] = acc[r][t][1] = 0.f;
    Q6Row cur[R];
#pragma unroll
    for (int r = 0; r < R; r++) cur[r] = load_row6(wq, rbase + r * 8 + g, N, nb, kb0 < kb1 ? kb0 : 0, c);
    for (int sb0 = kb0; sb0 < kb1; sb0 += KB) {
        const int nk = min(KB, kb1 - sb0);
        __syncthreads();
        for (int i = threadIdx.x; i < T * nk * 256; i += W * 32) {
            const int tt = i / (nk * 256), j = i % (nk * 256);
            sx[tt * KB * 256 + j] = xf[((size_t)tt * S + sb0 * 8) * 32 + j];
        }
        for (int i = threadIdx.x; i < T * nk * 64; i += W * 32) {
            const int tt = i / (nk * 64), j = i % (nk * 64);
            ss[tt * KB * 64 + j] = xs[((size_t)tt * S + sb0 * 8) * 8 + j];
        }
        __syncthreads();
        for (int kb = 0; kb < nk; kb++) {
            const int ib = sb0 + kb;
            Q6Row nxt[R];
#pragma unroll
            for (int r = 0; r < R; r++) nxt[r] = load_row6(wq, rbase + r * 8 + g, N, nb, ib + 1 < kb1 ? ib + 1 : ib, c);
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int s = 4 * n + j;
                    uint32_t a0[R], a1[R];
                    int s0w[R], s1w[R];
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        const uint32_t src0 = (j & 1) ? cur[r].qb[n][0] : cur[r].qa[n][0];
                        const uint32_t src1 = (j & 1) ? cur[r].qb[n][1] : cur[r].qa[n][1];
                        const uint32_t lo0 = (j >= 2 ? src0 >> 4 : src0) & 0x0F0F0F0Fu;
                        const uint32_t lo1 = (j >= 2 ? src1 >> 4 : src1) & 0x0F0F0F0Fu;
                        a0[r] = __vsub4(lo0 | (((cur[r].qh[n][0] >> (2 * j)) & 0x03030303u) << 4), 0x20202020u);
                        a1[r] = __vsub4(lo1 | (((cur[r].qh[n][1] >> (2 * j)) & 0x03030303u) << 4), 0x20202020u);
                        s0w[r] = sc8(cur[r].sc, 8 * n + 2 * j);
                        s1w[r] = sc8(cur[r].sc, 8 * n + 2 * j + 1);
                    }
#pragma unroll
                    for (int tt = 0; tt < T; tt++) {
                        const int at = (tt * KB + kb) * 8 + s;
                        const uint2 b = sx[at * 32 + lane];
                        const float d0 = __low2float(ss[at * 8 + 2 * c]), d1 = __low2float(ss[at * 8 + 2 * c + 1]);
#pragma unroll
                        for (int r = 0; r < R; r++) {
                            int c00 = 0, c01 = 0, c10 = 0, c11 = 0;
                            mma8816(c00, c01, a0[r], b.x);
                            mma8816(c10, c11, a1[r], b.y);
                            acc[r][tt][0] = fmaf(cur[r].d * d0, (float)(s0w[r] * c00 + s1w[r] * c10), acc[r][tt][0]);
                            acc[r][tt][1] = fmaf(cur[r].d * d1, (float)(s0w[r] * c01 + s1w[r] * c11), acc[r][tt][1]);
                        }
                    }
                }
#pragma unroll
            for (int r = 0; r < R; r++) cur[r] = nxt[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = rbase + r * 8 + g;
        if (row >= N) continue;
#pragma unroll
        for (int tt = 0; tt < T; tt++)
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int tok = tt * 8 + 2 * c + i;
                if (tok >= B) continue;
                if (SK == 1) y[(size_t)tok * N + row] = acc[r][tt][i];
                else atomicAdd(y + (size_t)tok * N + row, acc[r][tt][i]);
            }
    }
}
typedef void (*mma6_launch_t)(const uint4 *, const uint2 *, const half2 *, float *, int, int, int, cudaStream_t);
template <int T, int R, int W, int KB, int SK>
void launch_mma6(const uint4 *w, const uint2 *xf, const half2 *xs, float *y, int N, int K, int B, cudaStream_t s) {
    if (SK > 1) cudaMemsetAsync(y, 0, (size_t)B * N * sizeof(float), s);
    kmma6<T, R, W, KB, SK><<<dim3((N + W * R * 8 - 1) / (W * R * 8), SK), W * 32, 0, s>>>(w, xf, xs, y, N, K, B);
}
struct Mma6Variant { int T, R, W, KB, SK; mma6_launch_t fn; };

// KG5: decode attention for one query token per KV stream (llama.cpp's kg5 hook): f32 Q, f16 K and V, an f16 mask.
// A thread block serves one KV head of one stream and its G query heads together, so every K and V row is read once.
// Per chunk of 32 cells: Q.K with 8 lanes per cell (4 cells at once, each row read as 256 contiguous bytes), the
// online softmax with one lane per cell, then P.V with the lanes over the head dimension. Splits of the cells keep the
// GPU busy when few streams run; a second kernel combines them. Everything after the f16 loads is fp32.
template <int D> struct FaLoad;
template <> struct FaLoad<64> { typedef uint4 K; typedef uint32_t V; };
template <> struct FaLoad<128> { typedef uint4 K; typedef uint2 V; };
template <> struct FaLoad<256> { typedef uint4 K; typedef uint4 V; };
__device__ __forceinline__ float fa_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float fa_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
template <int D, int G, int W>
__global__ void __launch_bounds__(W * 32) kfa(const float *__restrict__ q, const char *__restrict__ k,
                                              const char *__restrict__ v, const char *__restrict__ mask,
                                              float *__restrict__ dst, float *__restrict__ ws, int n_kv, int n_head,
                                              int64_t qnb2, int64_t qnb3, int64_t knb1, int64_t knb2, int64_t knb3,
                                              int64_t vnb1, int64_t vnb2, int64_t vnb3, int mne3, int64_t mnb3,
                                              float scale, int splits) {
    constexpr int KH = D / 8, VH = D / 32;  // halfs per lane: Q.K (8 lanes per cell), P.V (32 lanes per cell)
    __shared__ float qs[G][D];
    __shared__ float sc[W][G][32];
    __shared__ float pm[W][G], pl[W][G];
    __shared__ float pacc[W][G][D];
    const int h = blockIdx.x, s = blockIdx.y, z = blockIdx.z;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, grp = lane / 8, sub = lane % 8;
    for (int i = threadIdx.x; i < G * D; i += W * 32) {
        const int g = i / D, d = i % D;
        qs[g][d] = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(q) + s * qnb3 + (h * G + g) * qnb2
                                                     + 4 * d) * scale;
    }
    __syncthreads();
    const half *mrow = reinterpret_cast<const half *>(mask + (s % mne3) * mnb3);
    const char *kb = k + s * knb3 + h * knb2, *vb = v + s * vnb3 + h * vnb2;
    const int chunks = (n_kv + 31) / 32, per = (chunks + splits - 1) / splits;
    const int c0 = z * per, c1 = min(chunks, c0 + per);
    float m[G], l[G], acc[G][VH];
#pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -INFINITY; l[g] = 0.f;
#pragma unroll
        for (int e = 0; e < VH; e++) acc[g][e] = 0.f;
    }
    for (int ch = c0 + warp; ch < c1; ch += W) {
        const int cell = ch * 32 + lane;
        const float mv = cell < n_kv ? __half2float(mrow[cell]) : -INFINITY;
        const unsigned valid = __ballot_sync(0xffffffffu, mv != -INFINITY);
        if (valid == 0) continue;
#pragma unroll
        for (int it = 0; it < 8; it++) {  // Q.K: lanes 8 grp .. 8 grp + 7 share cell 32 ch + 4 it + grp
            const int ci = ch * 32 + it * 4 + grp;
            float part[G];
#pragma unroll
            for (int g = 0; g < G; g++) part[g] = 0.f;
            if ((valid >> (it * 4 + grp)) & 1u) {
                const typename FaLoad<D>::K *row = reinterpret_cast<const typename FaLoad<D>::K *>(kb + ci * knb1) + sub * (KH / 8);
#pragma unroll
                for (int u = 0; u < KH / 8; u++) {
                    const uint4 w4 = __ldg(row + u);
                    const uint32_t ws4[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const float2 kf = __half22float2(*reinterpret_cast<const half2 *>(&ws4[e]));
                        const int d = sub * KH + u * 8 + 2 * e;
#pragma unroll
                        for (int g = 0; g < G; g++) part[g] = fmaf(qs[g][d], kf.x, fmaf(qs[g][d + 1], kf.y, part[g]));
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < G; g++) {
#pragma unroll
                for (int off = 4; off > 0; off >>= 1) part[g] += __shfl_xor_sync(0xffffffffu, part[g], off);
                if (sub == 0) sc[warp][g][it * 4 + grp] = part[g];
            }
        }
        __syncwarp();
#pragma unroll
        for (int g = 0; g < G; g++) {  // the online softmax, one lane per cell
            const float sv = mv == -INFINITY ? -INFINITY : sc[warp][g][lane] + mv;
            const float nm = fmaxf(m[g], fa_max(sv));
            const float corr = m[g] == -INFINITY ? 0.f : __expf(m[g] - nm);
            const float p = sv == -INFINITY ? 0.f : __expf(sv - nm);
            l[g] = l[g] * corr + fa_sum(p);
            m[g] = nm;
#pragma unroll
            for (int e = 0; e < VH; e++) acc[g][e] *= corr;
            sc[warp][g][lane] = p;
        }
        __syncwarp();
        for (int j = 0; j < 32; j++) {  // P.V: the lanes over the head dimension
            if (!((valid >> j) & 1u)) continue;
            const typename FaLoad<D>::V raw = __ldg(reinterpret_cast<const typename FaLoad<D>::V *>(vb + (ch * 32 + j) * vnb1) + lane);
            const uint32_t *words = reinterpret_cast<const uint32_t *>(&raw);
#pragma unroll
            for (int e = 0; e < VH / 2; e++) {
                const float2 vf = __half22float2(*reinterpret_cast<const half2 *>(&words[e]));
#pragma unroll
                for (int g = 0; g < G; g++) {
                    const float pj = sc[warp][g][j];
                    acc[g][2 * e] = fmaf(pj, vf.x, acc[g][2 * e]);
                    acc[g][2 * e + 1] = fmaf(pj, vf.y, acc[g][2 * e + 1]);
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
        for (int e = 0; e < VH; e++) pacc[warp][g][lane * VH + e] = acc[g][e];
        if (lane == 0) { pm[warp][g] = m[g]; pl[warp][g] = l[g]; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * D; i += W * 32) {  // the warps' parts combined
        const int g = i / D, d = i % D, hq = h * G + g;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < W; w++) M = fmaxf(M, pm[w][g]);
        float L = 0.f, A = 0.f;
        if (M != -INFINITY) {
#pragma unroll
            for (int w = 0; w < W; w++) {
                const float f = pm[w][g] == -INFINITY ? 0.f : __expf(pm[w][g] - M);
                L += f * pl[w][g]; A += f * pacc[w][g][d];
            }
        }
        if (splits == 1) {
            dst[((size_t)s * n_head + hq) * D + d] = L > 0.f ? A / L : 0.f;
        } else {
            float *o = ws + (((size_t)s * n_head + hq) * splits + z) * (D + 2);
            o[d] = A;
            if (d == 0) { o[D] = M; o[D + 1] = L; }
        }
    }
}
template <int D>
__global__ void kfa_combine(const float *__restrict__ ws, float *__restrict__ dst, int n_head, int splits) {
    const int hq = blockIdx.x, s = blockIdx.y, d = threadIdx.x;
    const float *o = ws + ((size_t)s * n_head + hq) * splits * (D + 2);
    float M = -INFINITY;
    for (int z = 0; z < splits; z++) M = fmaxf(M, o[z * (D + 2) + D]);
    float L = 0.f, A = 0.f;
    if (M != -INFINITY)
        for (int z = 0; z < splits; z++) {
            const float mz = o[z * (D + 2) + D];
            const float f = mz == -INFINITY ? 0.f : __expf(mz - M);
            L += f * o[z * (D + 2) + D + 1]; A += f * o[z * (D + 2) + d];
        }
    dst[((size_t)s * n_head + hq) * D + d] = L > 0.f ? A / L : 0.f;
}
template <int D, int G>
static void launch_fa(const float *q, const char *k, const char *v, const char *mask, float *dst, float *ws, int n_kv,
                      int n_head, int n_head_kv, int n_stream, const int64_t *qnb, const int64_t *knb, const int64_t *vnb,
                      int mne3, int64_t mnb3, float scale, int splits, cudaStream_t s) {
    constexpr int W = 4;
    kfa<D, G, W><<<dim3(n_head_kv, n_stream, splits), W * 32, 0, s>>>(q, k, v, mask, dst, ws, n_kv, n_head, qnb[2], qnb[3],
                                                                      knb[1], knb[2], knb[3], vnb[1], vnb[2], vnb[3],
                                                                      mne3, mnb3, scale, splits);
    if (splits > 1) kfa_combine<D><<<dim3(n_head, n_stream), D, 0, s>>>(ws, dst, n_head, splits);
}
// 0 when this library computed the attention, non-zero for llama.cpp's own kernels
static int run_fa(const float *q, const void *k, const void *v, const void *mask, float *dst, const int64_t *qne,
                  const int64_t *qnb, const int64_t *kne, const int64_t *knb, const int64_t *vnb, const int64_t *mne,
                  const int64_t *mnb, float scale, float *ws, size_t ws_floats, cudaStream_t s) {
    const int D = (int)qne[0], n_head = (int)qne[2], n_stream = (int)qne[3], n_kv = (int)kne[1], n_head_kv = (int)kne[2];
    if (qne[1] != 1 || kne[0] != D || n_head_kv < 1 || n_head % n_head_kv || kne[3] != n_stream || mne[2] != 1) return 1;
    if (qnb[0] != 4 || knb[0] != 2 || vnb[0] != 2 || mnb[0] != 2 || mne[0] < n_kv || n_stream % mne[3]) return 1;
    if ((((uintptr_t)k | (uintptr_t)v | (uintptr_t)knb[1] | (uintptr_t)knb[2] | (uintptr_t)knb[3] | (uintptr_t)vnb[1] |
          (uintptr_t)vnb[2] | (uintptr_t)vnb[3]) & 15) != 0) return 1;
    const int G = n_head / n_head_kv, blocks = n_head_kv * n_stream, chunks = (n_kv + 31) / 32;
    int splits = (160 + blocks - 1) / blocks;
    splits = max(1, min(splits, chunks / 4));
    if ((size_t)n_stream * n_head * splits * (D + 2) > ws_floats) splits = 1;
    const char *kc = (const char *)k, *vc = (const char *)v, *mc = (const char *)mask;
#define FA(DD, GG) if (D == DD && G == GG) { launch_fa<DD, GG>(q, kc, vc, mc, dst, ws, n_kv, n_head, n_head_kv, n_stream, qnb, knb, vnb, (int)mne[3], mnb[3], scale, splits, s); return cudaPeekAtLastError() == cudaSuccess ? 0 : 1; }
    FA(64, 1) FA(64, 2) FA(64, 4) FA(64, 8) FA(128, 1) FA(128, 2) FA(128, 4) FA(128, 8) FA(256, 1) FA(256, 2) FA(256, 4)
#undef FA
    return 1;
}
"""

TUNE = r"""
// float64 reference: one warp per output row, element by element (dequantize_row_q4_K of ggml-quants.c)
__global__ void ref64(const block_q4_K *__restrict__ wq, const float *__restrict__ x, double *__restrict__ y, int N,
                      int K, int B) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const int nb = K / 256;
    for (int b = 0; b < B; b++) {
        double s = 0.0;
        for (int k = lane; k < K; k += 32) {
            const block_q4_K *blk = wq + (size_t)row * nb + k / 256;
            const int e = k % 256, chunk = e / 64, within = e % 64, j = 2 * chunk + (within >= 32), l = within % 32;
            const uint8_t *sc = blk->scales;
            int s6, m6;
            if (j < 4) { s6 = sc[j] & 63; m6 = sc[j + 4] & 63; }
            else { s6 = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m6 = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
            const uint8_t byte = blk->qs[32 * chunk + l];
            const int q = within >= 32 ? byte >> 4 : byte & 0xF;
            s += ((double)__half2float(__ushort_as_half(blk->d)) * s6 * q -
                  (double)__half2float(__ushort_as_half(blk->dmin)) * m6) * (double)x[(size_t)b * K + k];
        }
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[(size_t)b * N + row] = s;
    }
}

extern "C" int kg_mq_count() { return (int)MQ_VARIANTS.size(); }
extern "C" void kg_mq_info(int i, int *out) { out[0] = MQ_VARIANTS[i].W; out[1] = MQ_VARIANTS[i].R; out[2] = MQ_VARIANTS[i].U; }

// MQ1 variant i for one token: repacked on the GPU first, then timed and checked like kg_measure
extern "C" int kg_measure_mq(int i, const void *w_host, int N, int K, const float *x_host, int copies, int reps,
                             double *us, double *rel_err) {
    const MqVariant &v = MQ_VARIANTS[i];
    const size_t nblocks = (size_t)N * (K / 256), wbytes = nblocks * 144;
    std::vector<uint4 *> dq(copies), dh(copies);
    block_q4_K *dw; float *dx, *dy; double *dr;
    if (cudaMalloc(&dw, wbytes) || cudaMalloc(&dx, (size_t)K * 4) || cudaMalloc(&dy, (size_t)N * 4) ||
        cudaMalloc(&dr, (size_t)N * 8)) return 1;
    cudaMemcpy(dw, w_host, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x_host, (size_t)K * 4, cudaMemcpyHostToDevice);
    for (int c = 0; c < copies; c++) {
        if (cudaMalloc(&dq[c], nblocks * 128) || cudaMalloc(&dh[c], nblocks * 16)) return 2;
        repack_q4k<<<(unsigned)((nblocks + 127) / 128), 128>>>(dw, dq[c], dh[c], nblocks);
    }
    ref64<<<(N + 7) / 8, 256>>>(dw, dx, dr, N, K, 1);
    v.fn(dq[0], dh[0], dx, dy, N, K, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> hy(N); std::vector<double> hr(N);
    cudaMemcpy(hy.data(), dy, (size_t)N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), dr, (size_t)N * 8, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int k = 0; k < N; k++) { num += (hy[k] - hr[k]) * (hy[k] - hr[k]); den += hr[k] * hr[k]; }
    *rel_err = sqrt(num / den);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 3; r++) for (int c = 0; c < copies; c++) v.fn(dq[c], dh[c], dx, dy, N, K, 0);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) for (int c = 0; c < copies; c++) v.fn(dq[c], dh[c], dx, dy, N, K, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    *us = 1000.0 * ms / ((double)reps * copies);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int c = 0; c < copies; c++) { cudaFree(dq[c]); cudaFree(dh[c]); }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dr);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

// float64 reference for Q6_K (dequantize_row_q6_K of ggml-quants.c), the first `rows` rows
__global__ void ref64_q6(const block_q6_K *__restrict__ wq, const float *__restrict__ x, double *__restrict__ y, int N,
                         int K, int B, int rows) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows || row >= N) return;
    const int nb = K / 256;
    for (int b = 0; b < B; b++) {
        double s = 0.0;
        for (int k = lane; k < K; k += 32) {
            const block_q6_K *blk = wq + (size_t)row * nb + k / 256;
            const int e = k % 256, n = e / 128, r = e % 128, j = r / 32, l = r % 32;
            const uint8_t byte = blk->ql[64 * n + ((j & 1) ? 32 : 0) + l];
            const int nib = j >= 2 ? byte >> 4 : byte & 0xF;
            const int h = (blk->qh[32 * n + l] >> (2 * j)) & 3;
            const int q = (nib | (h << 4)) - 32;
            s += (double)__half2float(__ushort_as_half(blk->d)) * blk->scales[8 * n + l / 16 + 2 * j] * q *
                 (double)x[(size_t)b * K + k];
        }
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[(size_t)b * rows + row] = s;
    }
}

extern "C" int kg_mma6_count() { return (int)MMA6_VARIANTS.size(); }

// KG4 Q6_K variant i at B tokens; the error is checked on the first 4,096 rows (the output matrix has 151,936)
extern "C" int kg_measure_mma6(int i, const void *w_host, int N, int K, const float *x_host, int B, int copies,
                               int reps, double *us, double *rel_err) {
    const Mma6Variant &v = MMA6_VARIANTS[i];
    if (B < 1 || B > 8 * v.T) return 5;
    const int rows = N < 4096 ? N : 4096;
    const size_t wbytes = (size_t)N * (K / 256) * 210;
    std::vector<block_q6_K *> dw(copies);
    std::vector<uint4 *> dp(copies);
    const size_t nblocks = (size_t)N * (K / 256);
    float *dx, *dy; double *dr; int8_t *dxf; half2 *dxs;
    if (cudaMalloc(&dx, (size_t)B * K * 4) || cudaMalloc(&dy, (size_t)B * N * 4) || cudaMalloc(&dr, (size_t)B * rows * 8) ||
        cudaMalloc(&dxf, (size_t)v.T * 8 * K) || cudaMalloc(&dxs, (size_t)v.T * (K / 32) * 8 * sizeof(half2))) return 1;
    for (int c = 0; c < copies; c++) {
        if (cudaMalloc(&dw[c], wbytes) || cudaMalloc(&dp[c], nblocks * 224)) return 2;
        cudaMemcpy(dw[c], w_host, wbytes, cudaMemcpyHostToDevice);
        repack_q6k<<<(unsigned)((nblocks + 127) / 128), 128>>>(dw[c], dp[c], nblocks);
        if (c > 0) { cudaFree(dw[c]); dw[c] = nullptr; }  // the timing reads the repacked copies only
    }
    cudaMemcpy(dx, x_host, (size_t)B * K * 4, cudaMemcpyHostToDevice);
    ref64_q6<<<(rows + 7) / 8, 256>>>(dw[0], dx, dr, N, K, B, rows);
    quantize_frag(dx, dxf, dxs, B, K, v.T, 0, 1);
    v.fn(dp[0], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> hy((size_t)B * N); std::vector<double> hr((size_t)B * rows);
    cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), dr, hr.size() * 8, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int b = 0; b < B; b++)
        for (int k = 0; k < rows; k++) {
            const double e = hy[(size_t)b * N + k] - hr[(size_t)b * rows + k];
            num += e * e; den += hr[(size_t)b * rows + k] * hr[(size_t)b * rows + k];
        }
    *rel_err = sqrt(num / den);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 3; r++) for (int c = 0; c < copies; c++) {
        quantize_frag(dx, dxf, dxs, B, K, v.T, 0, 1); v.fn(dp[c], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
    }
    cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) for (int c = 0; c < copies; c++) {
        quantize_frag(dx, dxf, dxs, B, K, v.T, 0, 1); v.fn(dp[c], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    *us = 1000.0 * ms / ((double)reps * copies);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int c = 0; c < copies; c++) { if (dw[c]) cudaFree(dw[c]); cudaFree(dp[c]); }
    cudaFree(dx); cudaFree(dy); cudaFree(dr); cudaFree(dxf); cudaFree(dxs);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

extern "C" int kg_mma_count() { return (int)MMA_VARIANTS.size(); }
extern "C" void kg_mma_info(int i, int *out) {
    const MmaVariant &v = MMA_VARIANTS[i];
    out[0] = v.T; out[1] = v.R; out[2] = v.W; out[3] = v.KB; out[4] = v.SK;
}

// KG4 variant i at B tokens (B <= 8 T): timed with its activation quantization and checked like kg_measure
extern "C" int kg_measure_mma(int i, const void *w_host, int N, int K, const float *x_host, int B, int copies, int reps,
                              double *us, double *rel_err) {
    const MmaVariant &v = MMA_VARIANTS[i];
    if (B < 1 || B > 8 * v.T) return 5;
    const size_t wbytes = (size_t)N * (K / 256) * 144;
    std::vector<block_q4_K *> dw(copies);
    float *dx, *dy; double *dr; int8_t *dxf; half2 *dxs;
    if (cudaMalloc(&dx, (size_t)B * K * 4) || cudaMalloc(&dy, (size_t)B * N * 4) || cudaMalloc(&dr, (size_t)B * N * 8) ||
        cudaMalloc(&dxf, (size_t)v.T * 8 * K) || cudaMalloc(&dxs, (size_t)v.T * (K / 32) * 8 * sizeof(half2))) return 1;
    for (int c = 0; c < copies; c++) {
        if (cudaMalloc(&dw[c], wbytes)) return 2;
        cudaMemcpy(dw[c], w_host, wbytes, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dx, x_host, (size_t)B * K * 4, cudaMemcpyHostToDevice);
    ref64<<<(N + 7) / 8, 256>>>(dw[0], dx, dr, N, K, B);
    quantize_frag(dx, dxf, dxs, B, K, v.T, 0);
    v.fn(dw[0], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> hy((size_t)B * N); std::vector<double> hr((size_t)B * N);
    cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), dr, hr.size() * 8, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t k = 0; k < hy.size(); k++) { num += (hy[k] - hr[k]) * (hy[k] - hr[k]); den += hr[k] * hr[k]; }
    *rel_err = sqrt(num / den);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 3; r++) for (int c = 0; c < copies; c++) {
        quantize_frag(dx, dxf, dxs, B, K, v.T, 0); v.fn(dw[c], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
    }
    cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) for (int c = 0; c < copies; c++) {
        quantize_frag(dx, dxf, dxs, B, K, v.T, 0); v.fn(dw[c], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    *us = 1000.0 * ms / ((double)reps * copies);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int c = 0; c < copies; c++) cudaFree(dw[c]);
    cudaFree(dx); cudaFree(dy); cudaFree(dr); cudaFree(dxf); cudaFree(dxs);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

extern "C" int kg_count() { return (int)VARIANTS.size(); }
extern "C" void kg_info(int i, int *out) { out[0] = VARIANTS[i].L; out[1] = VARIANTS[i].R; out[2] = VARIANTS[i].W; out[3] = VARIANTS[i].B; }

// time (us per matmul over `copies` weight copies, `reps` rounds) and error against float64 of variant i
extern "C" int kg_measure(int i, const void *w_host, int N, int K, const float *x_host, int copies, int reps,
                          double *us, double *rel_err) {
    const Variant &v = VARIANTS[i];
    const size_t wbytes = (size_t)N * (K / 256) * 144;
    std::vector<block_q4_K *> dw(copies);
    float *dx, *dy, *dxd; int8_t *dxq; double *dr;
    if (cudaMalloc(&dx, (size_t)v.B * K * 4) || cudaMalloc(&dy, (size_t)v.B * N * 4) || cudaMalloc(&dr, (size_t)v.B * N * 8) ||
        cudaMalloc(&dxq, (size_t)v.B * K) || cudaMalloc(&dxd, (size_t)v.B * K / 32 * 4)) return 1;
    for (int c = 0; c < copies; c++) {
        if (cudaMalloc(&dw[c], wbytes)) return 2;
        cudaMemcpy(dw[c], w_host, wbytes, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dx, x_host, (size_t)v.B * K * 4, cudaMemcpyHostToDevice);
    ref64<<<(N + 7) / 8, 256>>>(dw[0], dx, dr, N, K, v.B);
    quantize(dx, dxq, dxd, v.B * K, 0);
    v.fn(dw[0], dxq, dxd, dy, N, K, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    std::vector<float> hy((size_t)v.B * N); std::vector<double> hr((size_t)v.B * N);
    cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hr.data(), dr, hr.size() * 8, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (size_t k = 0; k < hy.size(); k++) { num += (hy[k] - hr[k]) * (hy[k] - hr[k]); den += hr[k] * hr[k]; }
    *rel_err = sqrt(num / den);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 3; r++) for (int c = 0; c < copies; c++) { quantize(dx, dxq, dxd, v.B * K, 0); v.fn(dw[c], dxq, dxd, dy, N, K, 0); }
    cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) for (int c = 0; c < copies; c++) { quantize(dx, dxq, dxd, v.B * K, 0); v.fn(dw[c], dxq, dxd, dy, N, K, 0); }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    *us = 1000.0 * ms / ((double)reps * copies);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (int c = 0; c < copies; c++) cudaFree(dw[c]);
    cudaFree(dx); cudaFree(dy); cudaFree(dr); cudaFree(dxq); cudaFree(dxd);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}
"""

LIBRARY = r"""
static uint32_t enabled_mask = 0xFFFFFFFFu;  // bit B-1: token count B is served here (Q4_K)
static uint32_t enabled_mask6 = 0xFFFFFFFFu;  // the same for Q6_K, chosen on its own
extern "C" void ironmule_kg_set_mask(uint32_t mask) { enabled_mask = mask; }
extern "C" uint32_t ironmule_kg_get_mask() { return enabled_mask; }
extern "C" void ironmule_kg_set_mask6(uint32_t mask) { enabled_mask6 = mask; }
static int fa_enabled = 0;  // KG5: the decode attention, off until the engine measures it faster (KV streams only)
extern "C" void ironmule_kg_set_fa(int on) { fa_enabled = on; }
extern "C" int ironmule_kg_get_fa() { return fa_enabled; }
extern "C" int ironmule_kg_flash_attn(const float *q, const void *k, const void *v, const void *mask, float *dst,
                                      const int64_t *qne, const int64_t *qnb, const int64_t *kne, const int64_t *knb,
                                      const int64_t *vnb, const int64_t *mne, const int64_t *mnb, float scale,
                                      cudaStream_t s) {
    if (!fa_enabled) return 1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 1;
    static float *ws[16]; static size_t wcap[16];
    const size_t need = (size_t)qne[3] * qne[2] * 64 * (qne[0] + 2);  // up to 64 splits
    if (wcap[dev] < need) {
        if (ws[dev]) cudaFree(ws[dev]);
        if (cudaMalloc(&ws[dev], need * sizeof(float)) != cudaSuccess) { ws[dev] = nullptr; wcap[dev] = 0; return 1; }
        wcap[dev] = need;
    }
    return run_fa(q, k, v, mask, dst, qne, qnb, kne, knb, vnb, mne, mnb, scale, ws[dev], wcap[dev], s);
}
extern "C" uint32_t ironmule_kg_get_mask6() { return enabled_mask6; }
extern "C" const char *ironmule_kg_table() { return TABLE_JSON; }

struct Repacked { uint4 *qs, *hdr; };
static std::map<std::pair<int, const void *>, Repacked> repacked;  // (device, weights) -> MQ1's copy, made once
static std::mutex repack_lock;
static std::map<std::pair<int, const void *>, uint4 *> repacked6;  // (device, Q6_K weights) -> KG5's aligned copy

// the hook's entry: 0 when this library computed y = w x, non-zero for llama.cpp's own kernels
extern "C" int ironmule_kg_mul_mat(int type, const void *w, const float *x, float *y, int64_t N, int64_t K, int64_t B,
                                   cudaStream_t s) {
    if ((type != 12 && type != 14) || K % 256 != 0 || B < 1 || B > 32) return 1;
    if (!(((type == 14 ? enabled_mask6 : enabled_mask) >> (B - 1)) & 1u)) return 1;
    const Entry *hit = nullptr;
    for (const Entry &e : ENTRIES)
        if (e.type == type && e.K == K && e.N == N && e.B == B) { hit = &e; break; }
    if (hit == nullptr) return 1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 1;
    if (hit->mq != nullptr) {  // MQ1: the weights repacked on the GPU the first time this matrix is used
        Repacked r;
        {
            std::lock_guard<std::mutex> guard(repack_lock);
            auto it = repacked.find({dev, w});
            if (it == repacked.end()) {
                const size_t nblocks = (size_t)N * (K / 256);
                if (cudaMalloc(&r.qs, nblocks * 128) != cudaSuccess) return 1;
                if (cudaMalloc(&r.hdr, nblocks * 16) != cudaSuccess) { cudaFree(r.qs); return 1; }
                repack_q4k<<<(unsigned)((nblocks + 127) / 128), 128, 0, s>>>((const block_q4_K *)w, r.qs, r.hdr, nblocks);
                repacked[{dev, w}] = r;
            } else {
                r = it->second;
            }
        }
        hit->mq(r.qs, r.hdr, x, y, (int)N, (int)K, s);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }
    if (hit->mm != nullptr || hit->mm6 != nullptr) {  // KG4: the activations in fragment order, T tiles of 8 tokens
        static int8_t *xf[16]; static half2 *xs[16]; static size_t fcap[16];
        const size_t need = (size_t)hit->T * 8 * K;
        if (fcap[dev] < need) {
            if (xf[dev]) { cudaFree(xf[dev]); cudaFree(xs[dev]); }
            if (cudaMalloc(&xf[dev], need) != cudaSuccess ||
                cudaMalloc(&xs[dev], need / 32 * sizeof(half2)) != cudaSuccess) {
                fcap[dev] = 0; xf[dev] = nullptr; return 1;
            }
            fcap[dev] = need;
        }
        const uint4 *w6 = nullptr;
        if (hit->mm6 != nullptr) {  // KG5: Q6_K repacked on the GPU the first time this matrix is used
            std::lock_guard<std::mutex> guard(repack_lock);
            auto it = repacked6.find({dev, w});
            if (it == repacked6.end()) {
                const size_t nblocks = (size_t)N * (K / 256);
                uint4 *p = nullptr;
                if (cudaMalloc(&p, nblocks * 224) != cudaSuccess) return 1;
                repack_q6k<<<(unsigned)((nblocks + 127) / 128), 128, 0, s>>>((const block_q6_K *)w, p, nblocks);
                repacked6[{dev, w}] = p;
                w6 = p;
            } else {
                w6 = it->second;
            }
        }
        quantize_frag(x, xf[dev], xs[dev], (int)B, (int)K, hit->T, s, hit->mm6 != nullptr);
        if (hit->mm6 != nullptr) hit->mm6(w6, (const uint2 *)xf[dev], xs[dev], y, (int)N, (int)K, (int)B, s);
        else hit->mm((const block_q4_K *)w, (const uint2 *)xf[dev], xs[dev], y, (int)N, (int)K, (int)B, s);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }
    launch_t fn = hit->fn;
    static int8_t *xq[16]; static float *xd[16]; static size_t cap[16];
    const size_t need = (size_t)B * K;
    if (cap[dev] < need) {
        if (xq[dev]) { cudaFree(xq[dev]); cudaFree(xd[dev]); }
        if (cudaMalloc(&xq[dev], need) != cudaSuccess || cudaMalloc(&xd[dev], need / 32 * sizeof(float)) != cudaSuccess) {
            cap[dev] = 0; xq[dev] = nullptr; return 1;
        }
        cap[dev] = need;
    }
    quantize(x, xq[dev], xd[dev], (int)need, s);
    fn((const block_q4_K *)w, xq[dev], xd[dev], y, (int)N, (int)K, s);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}
"""


MQ_VARIANTS = tuple((W, R, U) for W in (2, 4, 8) for R in (1, 2, 4) for U in (1, 2))  # MQ1, one token


def variants(batches=BATCHES) -> list[tuple[int, int, int, int]]:
    return [(L, R, W, B) for B in batches for L, R, W in itertools.product(LANES, ROWS, WARPS) if R * B <= 64]


def tune_source(vs: list[tuple[int, int, int, int]]) -> str:
    rows = "\n".join(f"    {{{L}, {R}, {W}, {B}, launch<{L}, {R}, {W}, {B}>}}," for L, R, W, B in vs)
    mq = "\n".join(f"    {{{W}, {R}, {U}, launch_mq<{W}, {R}, {U}>}}," for W, R, U in MQ_VARIANTS)
    mma = "\n".join(f"    {{{', '.join(map(str, v))}, launch_mma<{', '.join(map(str, v))}>}}," for v in MMA_VARIANTS)
    mma6 = "\n".join(f"    {{{', '.join(map(str, v))}, launch_mma6<{', '.join(map(str, v))}>}}," for v in MMA6_VARIANTS)
    return (DEVICE + "static const std::vector<Variant> VARIANTS = {\n" + rows + "\n};\n"
            + "static const std::vector<MqVariant> MQ_VARIANTS = {\n" + mq + "\n};\n"
            + "static const std::vector<MmaVariant> MMA_VARIANTS = {\n" + mma + "\n};\n"
            + "static const std::vector<Mma6Variant> MMA6_VARIANTS = {\n" + mma6 + "\n};\n" + TUNE)


def library_source(table: dict[str, Any]) -> str:
    def row(e: dict[str, Any]) -> str:
        head = f"    {{{e.get('type', TYPE_Q4_K)}, {e['K']}, {e['N']}, {e['B']}, "
        v = ", ".join(str(e.get(k)) for k in ("T", "R", "W", "KB", "SK"))
        if e.get("kind") == "mq1":
            return head + f"nullptr, launch_mq<{e['W']}, {e['R']}, {e['U']}>, nullptr, nullptr, 0}},"
        if e.get("kind") == "mma":
            return head + f"nullptr, nullptr, launch_mma<{v}>, nullptr, {e['T']}}},"
        if e.get("kind") == "mma6":
            return head + f"nullptr, nullptr, nullptr, launch_mma6<{v}>, {e['T']}}},"
        return head + f"launch<{e['L']}, {e['R']}, {e['W']}, {e['B']}>, nullptr, nullptr, nullptr, 0}},"

    rows = "\n".join(row(e) for e in table["entries"])
    text = json.dumps(table, sort_keys=True).replace("\\", "\\\\").replace('"', '\\"')
    return (DEVICE + "struct Entry { int type, K, N, B; launch_t fn; mq_launch_t mq; mma_launch_t mm; mma6_launch_t mm6; int T; };\n"
            + "static const std::vector<Entry> ENTRIES = {\n" + rows + "\n};\n"
            + f'static const char *TABLE_JSON = "{text}";\n' + LIBRARY)


def nvcc_path() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    for path in sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True):
        return path
    return None


def compute_capability() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0].replace(".", "") if out else None


def fingerprint() -> dict[str, Any]:
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        gpu = ""
    nvcc = nvcc_path()
    version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] \
        if nvcc else ""
    return {"gpu": gpu, "cc": compute_capability(), "nvcc": version, "kg": KG_VERSION}


def compile_cuda(source: str, out: Path, arch: str) -> None:
    nvcc = nvcc_path()
    if nvcc is None:
        raise RuntimeError("no nvcc: KG1 needs the CUDA toolkit")
    out.parent.mkdir(parents=True, exist_ok=True)
    src = out.with_suffix(".cu")
    src.write_text(source)
    done = subprocess.run([nvcc, "-O3", "-std=c++17", f"-arch=sm_{arch}", "-shared", "-Xcompiler", "-fPIC",
                           "--cudart", "shared", "-o", str(out), str(src)], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"nvcc failed: {done.stderr[-2000:]}")


def q4k_tensors(gguf_path: str, bindings: str | None = None, kind: str = "Q4_K") -> dict[tuple[int, int], dict[str, Any]]:
    """The model's Q4_K (or `kind`) matrices by shape (K inputs, N outputs): how many, and one of them to tune on."""
    root = Path(bindings or os.environ.get("IRONMULE_GGML_BINDINGS", "")) / "vendor" / "llama.cpp" / "gguf-py"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gguf  # llama.cpp's own reader

    shapes: dict[tuple[int, int], dict[str, Any]] = {}
    for t in gguf.GGUFReader(gguf_path).tensors:
        if t.tensor_type.name != kind or len(t.shape) != 2:
            continue
        key = (int(t.shape[0]), int(t.shape[1]))
        entry = shapes.setdefault(key, {"count": 0, "name": t.name, "data": t.data})
        entry["count"] += 1
    return shapes


def tune(gguf_path: str, work: Path, *, reference_error: float = 6e-3, copies: int = 8, reps: int = 30,
         log=print) -> dict[str, Any]:
    """Every variant on this GPU with this model's own Q4_K matrices: the fastest correct one per shape and token count.
    `reference_error` is llama.cpp's own kernel's error against float64 on such data (KG1 v2 on a T4: 4.8-5.6e-3)."""
    import numpy as np

    arch = compute_capability()
    if arch is None:
        raise RuntimeError("no NVIDIA GPU")
    vs = variants()
    lib_path = work / f"kg_tune_sm{arch}_v{KG_VERSION}.so"
    started = time.time()
    if not lib_path.exists():
        compile_cuda(tune_source(vs), lib_path, arch)
    compile_s = time.time() - started
    lib = ctypes.CDLL(str(lib_path))
    lib.kg_measure_mq.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_double)]
    lib.kg_measure.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.kg_measure_mma.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]
    rng = np.random.default_rng(0)
    bound = ERROR_MARGIN * reference_error
    entries, measured = [], []
    for (K, N), t in sorted(q4k_tensors(gguf_path).items()):
        w = np.ascontiguousarray(np.asarray(t["data"]))
        x = rng.standard_normal((max(MMA_BATCHES), K)).astype(np.float32)
        mma_best: dict[int, dict[str, Any]] = {}
        for B in MMA_BATCHES:  # KG4: the tensor-core kernels at the largest token count of each tile count
            for i, (Tt, R, W, KB, SK) in enumerate(MMA_VARIANTS):
                if Tt != (B + 7) // 8:
                    continue
                us, err = ctypes.c_double(), ctypes.c_double()
                code = lib.kg_measure_mma(i, w.ctypes.data, N, K, x.ctypes.data, B, copies, reps, ctypes.byref(us),
                                          ctypes.byref(err))
                row = {"K": K, "N": N, "B": B, "kind": "mma", "T": Tt, "R": R, "W": W, "KB": KB, "SK": SK,
                       "us": round(us.value, 2), "rel_err": err.value, "code": code}
                measured.append(row)
                if code == 0 and err.value <= bound and (B not in mma_best or us.value < mma_best[B]["us"]):
                    mma_best[B] = row
        for B in range(max(BATCHES) + 1, max(MMA_BATCHES) + 1):
            top = mma_best.get(-(-B // 8) * 8)
            if top is not None:
                entries.append({**{k: top[k] for k in ("K", "N", "kind", "T", "R", "W", "KB", "SK", "us")}, "B": B})
        for B in BATCHES:
            best = None
            for i, (L, R, W, b) in enumerate(vs):
                if b != B:
                    continue
                us, err = ctypes.c_double(), ctypes.c_double()
                code = lib.kg_measure(i, w.ctypes.data, N, K, x.ctypes.data, copies, reps, ctypes.byref(us),
                                      ctypes.byref(err))
                row = {"K": K, "N": N, "B": B, "L": L, "R": R, "W": W, "us": round(us.value, 2),
                       "rel_err": err.value, "code": code}
                measured.append(row)
                if code == 0 and err.value <= bound and (best is None or us.value < best["us"]):
                    best = row
            if B == 1:  # MQ1 (ST2): Q4_K repacked for warp streaming, float32 activations
                for i, (Wm, Rm, Um) in enumerate(MQ_VARIANTS):
                    us, err = ctypes.c_double(), ctypes.c_double()
                    code = lib.kg_measure_mq(i, w.ctypes.data, N, K, x.ctypes.data, copies, reps, ctypes.byref(us),
                                             ctypes.byref(err))
                    row = {"K": K, "N": N, "B": 1, "kind": "mq1", "W": Wm, "R": Rm, "U": Um, "us": round(us.value, 2),
                           "rel_err": err.value, "code": code}
                    measured.append(row)
                    if code == 0 and err.value <= bound and (best is None or us.value < best["us"]):
                        best = row
            top = mma_best.get(8)  # one tile of 8 tokens serves 1-8 at about the price measured at 8
            if top is not None and (best is None or top["us"] < best["us"]):
                best = {**top, "B": B}
            if best is not None:
                keys = ("K", "N", "B", "L", "R", "W", "U", "KB", "SK", "T", "kind", "us")
                entries.append({k: best[k] for k in keys if k in best})
                log(json.dumps({"kg_tuned": entries[-1]}))
    lib.kg_measure_mma6.argtypes = lib.kg_measure_mma.argtypes
    for (K, N), t in sorted(q4k_tensors(gguf_path, kind="Q6_K").items()):  # KG4: Q6_K on the tensor cores
        w = np.ascontiguousarray(np.asarray(t["data"]))
        x = rng.standard_normal((max(MMA_BATCHES), K)).astype(np.float32)
        copies6 = 2 if w.nbytes > 100e6 else copies  # the output matrix: 319 MB a copy
        best6: dict[int, dict[str, Any]] = {}
        for B in MMA_BATCHES:
            for i, (Tt, R, W, KB, SK) in enumerate(MMA6_VARIANTS):
                if Tt != (B + 7) // 8:
                    continue
                us, err = ctypes.c_double(), ctypes.c_double()
                code = lib.kg_measure_mma6(i, w.ctypes.data, N, K, x.ctypes.data, B, copies6, reps, ctypes.byref(us),
                                           ctypes.byref(err))
                row = {"K": K, "N": N, "B": B, "kind": "mma6", "type": TYPE_Q6_K, "T": Tt, "R": R, "W": W, "KB": KB,
                       "SK": SK, "us": round(us.value, 2), "rel_err": err.value, "code": code}
                measured.append(row)
                if code == 0 and err.value <= bound and (B not in best6 or us.value < best6[B]["us"]):
                    best6[B] = row
        for B in range(1, max(MMA_BATCHES) + 1):  # one tile of 8 tokens serves 1-8 at about the price measured at 8
            top = best6.get(-(-B // 8) * 8)
            if top is not None:
                entries.append({**{k: top[k] for k in ("K", "N", "kind", "type", "T", "R", "W", "KB", "SK", "us")},
                                "B": B})
                log(json.dumps({"kg_tuned": entries[-1]}))
    return {"version": KG_VERSION, "fingerprint": fingerprint(), "arch": arch, "entries": entries,
            "error_bound": bound, "compile_s": round(compile_s, 1), "tune_s": round(time.time() - started, 1),
            "measured": measured}


def build(table: dict[str, Any], out: Path) -> Path:
    compile_cuda(library_source({k: v for k, v in table.items() if k != "measured"}), out, table["arch"])
    return out


def ensure(gguf_path: str, cache: Path, log=print) -> Path | None:
    """The library for this GPU, nvcc and model shapes, tuned and built once and reused after; None where it cannot be."""
    if nvcc_path() is None or compute_capability() is None:
        return None
    fp = fingerprint()
    shapes = sorted(q4k_tensors(gguf_path)) + sorted(q4k_tensors(gguf_path, kind="Q6_K"))
    if not shapes:
        return None
    key = hashlib.sha256(json.dumps({"fp": fp, "shapes": shapes}, sort_keys=True).encode()).hexdigest()[:16]
    out = cache / f"kg_{key}.so"
    if out.exists():
        return out
    table = tune(gguf_path, cache, log=log)
    (cache / f"kg_{key}.json").write_text(json.dumps(table, indent=1))
    if not table["entries"]:
        return None
    return build(table, out)


__all__ = ["BATCHES", "KG_VERSION", "MMA6_VARIANTS", "MMA_BATCHES", "MMA_VARIANTS", "build", "ensure", "fingerprint", "library_source", "q4k_tensors", "tune",
           "tune_source", "variants"]
