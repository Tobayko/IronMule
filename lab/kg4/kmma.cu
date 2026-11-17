// KG4: Q4_K weights x 8-bit activations on a Turing GPU's tensor cores (mma.sync m8n8k16 s8), 1-32 tokens.
//
// llama.cpp computes a Q4_K matmul of 9-32 tokens with mul_mat_q (tensor cores, stream-k); on a T4 at 32 tokens it
// reads the weights at about 97 GB/s, under a third of the card's 320 (KG4 probe). Here one warp computes R tiles of
// 8 output rows for T tiles of 8 tokens over its share of K, reading each row's 144-byte blocks once, straight from
// global memory: 4 lanes per row, 8 contiguous bytes each per 64 weights. The activations are rounded to 8 bits per
// 32 values as llama.cpp's q8_1 does, written once in the order the mma fragments want, and staged in shared memory
// per KB blocks of 256 weights for all warps of a thread block. Inside a 32-weight sub-block (one 6-bit scale and min)
// two mma of 16 weights accumulate in int32; the k order inside a sub-block is permuted the same way for weights and
// activations (lane c holds weights 8c .. 8c + 7 of the sub-block), which the integer dot product does not see.
//
// usage: kmma WEIGHTS.bin N K X.bin GGML_PREFIX COPIES REPS B...
//   X.bin holds at least max(B) rows of K floats; GGML_PREFIX_B<B>.bin is llama.cpp's result for the same data
//   (lab/kg1/ref_ggml.c), compared against float64 as this kernel's is. One JSON line per (B, variant).
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
__global__ void quant_frag(const float *__restrict__ x, int8_t *__restrict__ xf, half2 *__restrict__ xs, int B, int K,
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
static void quantize_frag(const float *x, int8_t *xf, half2 *xs, int B, int K, int T, cudaStream_t s) {
    const int warps = T * 8 * (K / 32);
    quant_frag<<<(warps + 7) / 8, 256, 0, s>>>(x, xf, xs, B, K, T);
}
struct MmaVariant { int T, R, W, KB, SK; mma_launch_t fn; };

#define V(T, R, W, KB, SK) {T, R, W, KB, SK, launch_mma<T, R, W, KB, SK>}
#define VT(T) V(T, 1, 4, 2, 1), V(T, 2, 4, 2, 1), V(T, 1, 8, 2, 1), V(T, 2, 8, 2, 1), V(T, 1, 4, 4, 1), \
    V(T, 2, 4, 4, 1), V(T, 1, 8, 4, 1), V(T, 1, 4, 2, 2), V(T, 2, 4, 2, 2), V(T, 1, 8, 2, 2), V(T, 1, 4, 2, 4), \
    V(T, 1, 8, 2, 4), V(T, 1, 4, 4, 2), V(T, 1, 4, 1, 1), V(T, 1, 4, 1, 4), V(T, 2, 2, 2, 1)
#define VT8(T) VT(T), V(T, 1, 4, 8, 1), V(T, 1, 8, 8, 1), V(T, 2, 4, 8, 1), V(T, 1, 4, 8, 2)
static const std::vector<MmaVariant> VARIANTS = {VT8(1), VT8(2), VT(3), VT(4)};  // T * KB <= 16

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

static void *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(*n);
    if (fread(p, 1, *n, f) != *n) exit(1);
    fclose(f);
    return p;
}

static double rel(const float *a, const double *r, size_t n) {
    double num = 0, den = 0;
    for (size_t k = 0; k < n; k++) { num += (a[k] - r[k]) * (a[k] - r[k]); den += r[k] * r[k]; }
    return sqrt(num / den);
}

int main(int argc, char **argv) {
    if (argc < 9) { fprintf(stderr, "usage: kmma W.bin N K X.bin GGML_PREFIX COPIES REPS B...\n"); return 2; }
    const int N = atoi(argv[2]), K = atoi(argv[3]), C = atoi(argv[6]), REPS = atoi(argv[7]);
    size_t wn, xn;
    void *w = slurp(argv[1], &wn);
    float *x = (float *)slurp(argv[4], &xn);
    if (wn != (size_t)N * (K / 256) * 144) { fprintf(stderr, "weights: %zu bytes\n", wn); return 2; }
    std::vector<block_q4_K *> dw(C);
    for (int c = 0; c < C; c++) {
        if (cudaMalloc(&dw[c], wn)) return 3;
        cudaMemcpy(dw[c], w, wn, cudaMemcpyHostToDevice);
    }
    float *dx, *dy; double *dr; int8_t *dxf; half2 *dxs;
    if (cudaMalloc(&dx, xn) || cudaMalloc(&dy, (size_t)32 * N * 4) || cudaMalloc(&dr, (size_t)32 * N * 8) ||
        cudaMalloc(&dxf, (size_t)32 * K) || cudaMalloc(&dxs, (size_t)4 * (K / 32) * 8 * sizeof(half2))) return 3;
    cudaMemcpy(dx, x, xn, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int a = 8; a < argc; a++) {
        const int B = atoi(argv[a]), T = (B + 7) / 8;
        if ((size_t)B * K * 4 > xn || B > 32) continue;
        ref64<<<(N + 7) / 8, 256>>>(dw[0], dx, dr, N, K, B);
        std::vector<double> hr((size_t)B * N);
        std::vector<float> hy((size_t)B * N);
        cudaMemcpy(hr.data(), dr, hr.size() * 8, cudaMemcpyDeviceToHost);
        char path[600]; snprintf(path, sizeof path, "%s_B%d.bin", argv[5], B);
        double gerr = -1;
        if (FILE *f = fopen(path, "rb")) {
            if (fread(hy.data(), 4, hy.size(), f) == hy.size()) gerr = rel(hy.data(), hr.data(), hr.size());
            fclose(f);
        }
        for (const MmaVariant &v : VARIANTS) {
            if (v.T != T) continue;
            cudaMemset(dy, 0xFF, (size_t)B * N * 4);
            quantize_frag(dx, dxf, dxs, B, K, T, 0);
            v.fn(dw[0], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
            const cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("{\"N\": %d, \"K\": %d, \"B\": %d, \"T\": %d, \"R\": %d, \"W\": %d, \"KB\": %d, \"SK\": %d, \"error\": \"%s\"}\n",
                       N, K, B, v.T, v.R, v.W, v.KB, v.SK, cudaGetErrorString(err));
                return 4;
            }
            cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);
            const double e = rel(hy.data(), hr.data(), hr.size());
            for (int r = 0; r < 3; r++) for (int c = 0; c < C; c++) {
                quantize_frag(dx, dxf, dxs, B, K, T, 0); v.fn(dw[c], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
            }
            cudaEventRecord(e0);
            for (int r = 0; r < REPS; r++) for (int c = 0; c < C; c++) {
                quantize_frag(dx, dxf, dxs, B, K, T, 0); v.fn(dw[c], (const uint2 *)dxf, dxs, dy, N, K, B, 0);
            }
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1000.0 * ms / ((double)REPS * C);
            printf("{\"N\": %d, \"K\": %d, \"B\": %d, \"T\": %d, \"R\": %d, \"W\": %d, \"KB\": %d, \"SK\": %d, "
                   "\"us\": %.2f, \"GBps\": %.1f, \"rel_err\": %.3e, \"ggml_rel_err\": %.3e}\n",
                   N, K, B, v.T, v.R, v.W, v.KB, v.SK, us, wn / us / 1e3, e, gerr);
            fflush(stdout);
        }
    }
    return 0;
}
