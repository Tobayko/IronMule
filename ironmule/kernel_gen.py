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

KG_VERSION = 3  # 2 (ST2): MQ1 kernels for one token; 3 (KG4): tensor-core kernels for 1-32 tokens
TYPE_Q4_K = 12  # GGML_TYPE_Q4_K
BATCHES = tuple(range(1, 9))  # from 9 tokens on llama.cpp's tensor-core kernel is faster (KG1 v2: 16.8 ms vs 42.4)
LANES, ROWS, WARPS = (8, 16, 32), (1, 2, 4, 8), (2, 4, 8)
ERROR_MARGIN = 1.25  # a kernel may be at most this much less exact than llama.cpp's own on the same data
# KG4: Q4_K x 8-bit activations on tensor cores (mma.sync m8n8k16, sm_75 and later), T tiles of 8 tokens; measured
# at the largest token count of each tile count and serving every count of that tile count
MMA_BATCHES = (8, 16, 24, 32)
MMA_SHAPES = ((1, 2, 1, 1), (1, 4, 1, 1), (1, 8, 1, 1), (2, 2, 1, 1), (2, 4, 1, 1), (2, 8, 1, 1), (4, 2, 1, 1),
              (4, 4, 1, 1), (1, 4, 2, 1), (2, 4, 2, 1), (4, 4, 2, 1), (2, 8, 2, 1), (1, 4, 1, 2), (2, 4, 1, 2),
              (1, 4, 1, 4), (2, 4, 1, 4), (1, 8, 1, 2), (1, 8, 1, 4), (2, 2, 1, 2), (4, 2, 1, 2))  # R, W, KB, SK
MMA_VARIANTS = tuple((T,) + v for T in (1, 2, 3, 4) for v in MMA_SHAPES)  # T, R, W, KB, SK

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

// x [B][K] float -> fragment order for T tiles of 8 tokens (tokens past B are zero):
//   xf, 8 bytes per (tile tt, sub-block s, lane): xq[8 tt + lane / 4][32 s + 8 (lane % 4) .. + 7]
//   xs, per (tt, s, token in tile): (d, d * sum of the 32 quants)
__global__ void quant_frag(const float *__restrict__ x, int8_t *__restrict__ xf, float2 *__restrict__ xs, int B, int K,
                           int T) {
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
    xf[(((size_t)tt * S + s) * 32 + g * 4 + lane / 8) * 8 + lane % 8] = (int8_t)q;
    if (lane == 0) xs[((size_t)tt * S + s) * 8 + g] = make_float2(d, d * (float)sum);
}

template <int T, int R, int W, int KB, int SK>
__global__ void __launch_bounds__(W * 32) kmma(const block_q4_K *__restrict__ wq, const uint2 *__restrict__ xf,
                                               const float2 *__restrict__ xs, float *__restrict__ y, int N, int K,
                                               int B) {
    __shared__ uint2 sx[T * KB * 256];
    __shared__ float2 ss[T * KB * 64];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int S = K / 32, nb = K / 256;
    const int rbase = (blockIdx.x * W + warp) * R * 8;
    const int per = (nb + SK - 1) / SK, kb0 = blockIdx.y * per, kb1 = min(nb, kb0 + per);
    float acc[R][T][2];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[r][t][0] = acc[r][t][1] = 0.f;
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
            uint2 q[R][4];
            uint4 h[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const int row = rbase + r * 8 + g;
                if (row < N) {
                    const block_q4_K *blk = wq + (size_t)row * nb + sb0 + kb;
                    h[r] = __ldg(reinterpret_cast<const uint4 *>(blk));
#pragma unroll
                    for (int grp = 0; grp < 4; grp++) q[r][grp] = __ldg(reinterpret_cast<const uint2 *>(blk->qs + 32 * grp + 8 * c));
                } else {
                    h[r] = make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int grp = 0; grp < 4; grp++) q[r][grp] = make_uint2(0, 0);
                }
            }
#pragma unroll
            for (int s = 0; s < 8; s++) {
                const int grp = s / 2, sh = (s & 1) * 4;
                float dsc[R], dmn[R];
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float sc, m;
                    scale_min(h[r].y, h[r].z, h[r].w, s, sc, m);
                    dsc[r] = __half2float(__ushort_as_half((unsigned short)(h[r].x & 0xFFFF))) * sc;
                    dmn[r] = __half2float(__ushort_as_half((unsigned short)(h[r].x >> 16))) * m;
                }
#pragma unroll
                for (int tt = 0; tt < T; tt++) {
                    const int at = (tt * KB + kb) * 8 + s;
                    const uint2 b = sx[at * 32 + lane];
                    const float2 s0 = ss[at * 8 + 2 * c], s1 = ss[at * 8 + 2 * c + 1];
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        int c0 = 0, c1 = 0;
                        mma8816(c0, c1, (q[r][grp].x >> sh) & 0x0F0F0F0Fu, b.x);
                        mma8816(c0, c1, (q[r][grp].y >> sh) & 0x0F0F0F0Fu, b.y);
                        acc[r][tt][0] = fmaf(dsc[r], s0.x * (float)c0, fmaf(-dmn[r], s0.y, acc[r][tt][0]));
                        acc[r][tt][1] = fmaf(dsc[r], s1.x * (float)c1, fmaf(-dmn[r], s1.y, acc[r][tt][1]));
                    }
                }
            }
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

typedef void (*mma_launch_t)(const block_q4_K *, const uint2 *, const float2 *, float *, int, int, int, cudaStream_t);
template <int T, int R, int W, int KB, int SK>
void launch_mma(const block_q4_K *w, const uint2 *xf, const float2 *xs, float *y, int N, int K, int B, cudaStream_t s) {
    if (SK > 1) cudaMemsetAsync(y, 0, (size_t)B * N * sizeof(float), s);
    kmma<T, R, W, KB, SK><<<dim3((N + W * R * 8 - 1) / (W * R * 8), SK), W * 32, 0, s>>>(w, xf, xs, y, N, K, B);
}
static void quantize_frag(const float *x, int8_t *xf, float2 *xs, int B, int K, int T, cudaStream_t s) {
    const int warps = T * 8 * (K / 32);
    quant_frag<<<(warps + 7) / 8, 256, 0, s>>>(x, xf, xs, B, K, T);
}
struct MmaVariant { int T, R, W, KB, SK; mma_launch_t fn; };
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
    float *dx, *dy; double *dr; int8_t *dxf; float2 *dxs;
    if (cudaMalloc(&dx, (size_t)B * K * 4) || cudaMalloc(&dy, (size_t)B * N * 4) || cudaMalloc(&dr, (size_t)B * N * 8) ||
        cudaMalloc(&dxf, (size_t)v.T * 8 * K) || cudaMalloc(&dxs, (size_t)v.T * (K / 32) * 8 * sizeof(float2))) return 1;
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
static uint32_t enabled_mask = 0xFFFFFFFFu;  // bit B-1: token count B is served here
extern "C" void ironmule_kg_set_mask(uint32_t mask) { enabled_mask = mask; }
extern "C" uint32_t ironmule_kg_get_mask() { return enabled_mask; }
extern "C" const char *ironmule_kg_table() { return TABLE_JSON; }

struct Repacked { uint4 *qs, *hdr; };
static std::map<std::pair<int, const void *>, Repacked> repacked;  // (device, weights) -> MQ1's copy, made once
static std::mutex repack_lock;

// the hook's entry: 0 when this library computed y = w x, non-zero for llama.cpp's own kernels
extern "C" int ironmule_kg_mul_mat(int type, const void *w, const float *x, float *y, int64_t N, int64_t K, int64_t B,
                                   cudaStream_t s) {
    if (type != 12 || K % 256 != 0 || B < 1 || B > 32 || !((enabled_mask >> (B - 1)) & 1u)) return 1;
    const Entry *hit = nullptr;
    for (const Entry &e : ENTRIES)
        if (e.K == K && e.N == N && e.B == B) { hit = &e; break; }
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
    if (hit->mm != nullptr) {  // KG4: the activations in fragment order, T tiles of 8 tokens
        static int8_t *xf[16]; static float2 *xs[16]; static size_t fcap[16];
        const size_t need = (size_t)hit->T * 8 * K;
        if (fcap[dev] < need) {
            if (xf[dev]) { cudaFree(xf[dev]); cudaFree(xs[dev]); }
            if (cudaMalloc(&xf[dev], need) != cudaSuccess ||
                cudaMalloc(&xs[dev], need / 32 * sizeof(float2)) != cudaSuccess) {
                fcap[dev] = 0; xf[dev] = nullptr; return 1;
            }
            fcap[dev] = need;
        }
        quantize_frag(x, xf[dev], xs[dev], (int)B, (int)K, hit->T, s);
        hit->mm((const block_q4_K *)w, (const uint2 *)xf[dev], xs[dev], y, (int)N, (int)K, (int)B, s);
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
    return (DEVICE + "static const std::vector<Variant> VARIANTS = {\n" + rows + "\n};\n"
            + "static const std::vector<MqVariant> MQ_VARIANTS = {\n" + mq + "\n};\n"
            + "static const std::vector<MmaVariant> MMA_VARIANTS = {\n" + mma + "\n};\n" + TUNE)


def library_source(table: dict[str, Any]) -> str:
    def row(e: dict[str, Any]) -> str:
        if e.get("kind") == "mq1":
            return f"    {{{e['K']}, {e['N']}, {e['B']}, nullptr, launch_mq<{e['W']}, {e['R']}, {e['U']}>, nullptr, 0}},"
        if e.get("kind") == "mma":
            v = ", ".join(str(e[k]) for k in ("T", "R", "W", "KB", "SK"))
            return f"    {{{e['K']}, {e['N']}, {e['B']}, nullptr, nullptr, launch_mma<{v}>, {e['T']}}},"
        return f"    {{{e['K']}, {e['N']}, {e['B']}, launch<{e['L']}, {e['R']}, {e['W']}, {e['B']}>, nullptr, nullptr, 0}},"

    rows = "\n".join(row(e) for e in table["entries"])
    text = json.dumps(table, sort_keys=True).replace("\\", "\\\\").replace('"', '\\"')
    return (DEVICE + "struct Entry { int K, N, B; launch_t fn; mq_launch_t mq; mma_launch_t mm; int T; };\n"
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


def q4k_tensors(gguf_path: str, bindings: str | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    """The model's Q4_K matrices by shape (K inputs, N outputs): how many, and one of them to tune on."""
    root = Path(bindings or os.environ.get("IRONMULE_GGML_BINDINGS", "")) / "vendor" / "llama.cpp" / "gguf-py"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gguf  # llama.cpp's own reader

    shapes: dict[tuple[int, int], dict[str, Any]] = {}
    for t in gguf.GGUFReader(gguf_path).tensors:
        if t.tensor_type.name != "Q4_K" or len(t.shape) != 2:
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
    shapes = sorted(q4k_tensors(gguf_path))
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


__all__ = ["BATCHES", "KG_VERSION", "MMA_BATCHES", "MMA_VARIANTS", "build", "ensure", "fingerprint", "library_source", "q4k_tensors", "tune",
           "tune_source", "variants"]
