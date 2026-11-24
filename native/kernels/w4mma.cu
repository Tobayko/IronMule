// W4R (IronMule, 2026-11-19): llama.cpp's Q4_K and Q6_K weights times 1-32 tokens on Ampere+ tensor cores.
//
// A decode step of a served model is bound by reading its weights once. GGUF stores each row's 256-weight blocks one
// after another, so a 16-row mma tile gathers 16 short runs that lie a row apart; measured on an A100 that pattern
// streamed at 1.44 TB/s at best and could not overlap the arithmetic. W4R repacks every matrix once, on the GPU, into
// tile order: for each 16-row tile and 256-weight super-block, the quants of all 32 lanes lie in one contiguous run in
// the exact order of the mma fragments (one 32-bit word = a thread's 8 weights of one row and one 32-weight sub-block),
// followed by the 16 rows' scale headers. A warp then reads 2 KB contiguous per super-block with 128-bit loads and walks
// along K sequentially. The tensor cores multiply 1024 + q exactly (fp16 in, fp32 accumulation); scales and minimums
// are applied per sub-block afterwards in fp32 (acc += S * dot + NB * sum(x)), so no weight is dequantized one by one.
//
// A variant is (WARPS, KS, NT): warps per block, warps sharing one tile along K (reduced in shared memory), and 8-token
// tiles; a call also chooses a grid split of K (partials reduced by a second kernel). Which variant serves a matrix
// shape and token count is measured on the GPU itself (ironmule/w4_kernels.py) and set at run time.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#ifndef IRONMULE_NO_MQ1
#include "mq1/mq1.cuh"  // MQ1: vLLM's Marlin for Q4_K (native/kernels/mq1)
#else
// T4W (2026-11-24): Marlin needs sm_80; a Turing build (lab/w4bw on Kaggle's T4) leaves MQ1 out
namespace mq1 {
struct Prepared {};
struct Scratch { void *a = nullptr, *c = nullptr, *c_tmp = nullptr, *locks = nullptr; };
inline int prepare(const void *, long, long, int, cudaStream_t, Prepared &) { return 1; }
inline int run(const Prepared &, const float *, float *, int, int, cudaStream_t, Scratch &, bool) { return 1; }
inline void release(Prepared &) {}
}  // namespace mq1
#endif

namespace {

constexpr int TYPE_Q4_K = 12, TYPE_Q6_K = 14;
constexpr int BS4 = 144, BS6 = 210;
constexpr int T4 = 2304;  // bytes per (tile, super-block), Q4_K: 2048 quants + 16 rows x 16-byte headers
constexpr int T4P = 2560;  // PS (v12): 2048 quants + 16 rows x (8 fp16 sub-block scales + 8 fp16 negated minimums)
constexpr int T6 = 3584;  // Q6_K: 32 lanes x 96 bytes (16 nibble words + 8 two-bit words) + 16 rows x 32-byte headers

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // T4W: Turing has m16n8k8 only; k 0-7 is (a0, a1; b0), k 8-15 is (a2, a3; b1) in the same fragment layout
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a2), "r"(a3), "r"(b1));
#else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t h2_bits(half2 v) { return *reinterpret_cast<uint32_t *>(&v); }
__device__ __forceinline__ half2 bits_h2(uint32_t u) { return *reinterpret_cast<half2 *>(&u); }

// FP: a lane's 8 activations of token n at k (float, straight from llama.cpp), as 4 half2 (saturated), and the sums of
// the 32 rounded values of sub-block j for tokens 2c and 2c + 1 of the 8-token tile (shuffled across the warp; every
// lane must call this)
__device__ __forceinline__ void fused_x(const float *__restrict__ x, int K, int B, int n, size_t k, int c,
                                        uint4 &hx, float2 &sv) {
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
    if (n < B) {
        a = __ldg(reinterpret_cast<const float4 *>(x + (size_t)n * K + k));
        b = __ldg(reinterpret_cast<const float4 *>(x + (size_t)n * K + k + 4));
    }
    auto sat = [](float v) { return fminf(fmaxf(v, -65000.f), 65000.f); };
    const half2 h0 = __floats2half2_rn(sat(a.x), sat(a.y)), h1 = __floats2half2_rn(sat(a.z), sat(a.w));
    const half2 h2 = __floats2half2_rn(sat(b.x), sat(b.y)), h3 = __floats2half2_rn(sat(b.z), sat(b.w));
    hx = make_uint4(*reinterpret_cast<const uint32_t *>(&h0), *reinterpret_cast<const uint32_t *>(&h1),
                    *reinterpret_cast<const uint32_t *>(&h2), *reinterpret_cast<const uint32_t *>(&h3));
    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1), f2 = __half22float2(h2), f3 = __half22float2(h3);
    float t = ((f0.x + f0.y) + (f1.x + f1.y)) + ((f2.x + f2.y) + (f3.x + f3.y));
    t += __shfl_xor_sync(0xffffffffu, t, 1);
    t += __shfl_xor_sync(0xffffffffu, t, 2);  // token n's sum over the 4 lanes of its quad
    sv.x = __shfl_sync(0xffffffffu, t, (2 * c) * 4);
    sv.y = __shfl_sync(0xffffffffu, t, (2 * c + 1) * 4);
}

// byte b of w as a float, exactly: the byte becomes the low mantissa bits of 2^23, then 2^23 is subtracted (two
// full-rate instructions instead of a shift, a mask and a quarter-rate I2F; ptxas -v / SASS, A100). The constant
// comes from blockDim.y (1 at run time) so that ptxas keeps it in a register and the PRMT selector is the immediate.
__device__ __forceinline__ float byte_f(uint32_t w, int b) {
    const uint32_t k = 0x4AFFFFFFu + blockDim.y;  // 0x4B000000
    return __int_as_float(__byte_perm(w, k, 0x7540u | (uint32_t)b)) - 8388608.f;
}

// word i of a register array of uint4 (i a compile-time constant after unrolling: stays in registers)
template <int NQ>
__device__ __forceinline__ uint32_t wget(const uint4 (&q)[NQ], int i) {
    const uint4 v = q[i / 4];
    return (i & 3) == 0 ? v.x : (i & 3) == 1 ? v.y : (i & 3) == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint4 ldg_stream(const void *p) {  // read once: do not keep in L1
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// ---------------------------------------------------------------------------------------------------------------- repack
// Q4_K: one thread per (tile, super-block, lane); writes the lane's 16 words and, lanes 0-15, one row header
__global__ void repack_q4k(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int N, int nb) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int tiles = (N + 15) / 16;
    if (id >= (size_t)tiles * nb * 32) return;
    const int lane = (int)(id % 32), sb = (int)((id / 32) % nb), tile = (int)(id / 32 / nb);
    const int g = lane / 4, c = lane % 4;
    uint8_t *out = dst + ((size_t)tile * nb + sb) * T4;
    uint32_t words[16];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int row = tile * 16 + g + 8 * h;
        const uint8_t *blk = src + ((size_t)row * nb + sb) * BS4;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t wd = 0;
            if (row < N) {
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int grp = j / 2;
                    const uint8_t lo = blk[16 + 32 * grp + 8 * c + b], hi = blk[16 + 32 * grp + 8 * c + 4 + b];
                    const uint32_t qlo = (j & 1) ? lo >> 4 : lo & 0xF, qhi = (j & 1) ? hi >> 4 : hi & 0xF;
                    wd |= (qlo | (qhi << 4)) << (8 * b);  // byte b: weight b (low nibble), weight 4 + b (high)
                }
            }
            words[2 * j + h] = wd;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)  // interleaved: piece i of every lane is contiguous (conflict-free 16-byte reads)
        *reinterpret_cast<uint4 *>(out + (i * 32 + lane) * 16) =
            make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
    if (lane < 16) {
        const int row = tile * 16 + lane;
        uint4 hd = make_uint4(0, 0, 0, 0);
        if (row < N) memcpy(&hd, src + ((size_t)row * nb + sb) * BS4, 16);
        *reinterpret_cast<uint4 *>(out + 2048 + lane * 16) = hd;
    }
}

// PS (v12, 2026-11-24): the same quant order with stride T4P, and per row the eight sub-block scales s_j * d and the
// eight negated minimums m_j * -dmin as fp16, rounded once exactly as w4d's half2 products round them in the kernel
__global__ void repack_q4k_pre(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int N, int nb) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int tiles = (N + 15) / 16;
    if (id >= (size_t)tiles * nb * 32) return;
    const int lane = (int)(id % 32), sb = (int)((id / 32) % nb), tile = (int)(id / 32 / nb);
    const int g = lane / 4, c = lane % 4;
    uint8_t *out = dst + ((size_t)tile * nb + sb) * T4P;
    uint32_t words[16];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int row = tile * 16 + g + 8 * h;
        const uint8_t *blk = src + ((size_t)row * nb + sb) * BS4;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t wd = 0;
            if (row < N) {
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int grp = j / 2;
                    const uint8_t lo = blk[16 + 32 * grp + 8 * c + b], hi = blk[16 + 32 * grp + 8 * c + 4 + b];
                    const uint32_t qlo = (j & 1) ? lo >> 4 : lo & 0xF, qhi = (j & 1) ? hi >> 4 : hi & 0xF;
                    wd |= (qlo | (qhi << 4)) << (8 * b);
                }
            }
            words[2 * j + h] = wd;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
        *reinterpret_cast<uint4 *>(out + (i * 32 + lane) * 16) =
            make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
    if (lane < 16) {
        const int row = tile * 16 + lane;
        half sv[8], mv[8];
#pragma unroll
        for (int j = 0; j < 8; j++) sv[j] = mv[j] = __ushort_as_half(0);
        if (row < N) {
            const uint8_t *blk = src + ((size_t)row * nb + sb) * BS4;
            const half d = __ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8)));
            const half nm = __hneg(__ushort_as_half((uint16_t)(blk[2] | (blk[3] << 8))));
            const uint8_t *q = blk + 4;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int sc = j < 4 ? (q[j] & 63) : ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
                const int mn = j < 4 ? (q[j + 4] & 63) : ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
                sv[j] = __hmul(__ushort2half_rn((unsigned short)sc), d);   // = w4d's hmul2((1024+s)-1024, D)
                mv[j] = __hmul(__ushort2half_rn((unsigned short)mn), nm);  // = w4d's hmul2((1024+m)-1024, -dmin)
            }
        }
        uint4 a, b;
        memcpy(&a, sv, 16);
        memcpy(&b, mv, 16);
        *reinterpret_cast<uint4 *>(out + 2048 + lane * 32) = a;
        *reinterpret_cast<uint4 *>(out + 2048 + lane * 32 + 16) = b;
    }
}

// Q6_K: thread (g, c) takes l = 4c..4c+3 of each 16-weight scale group: word (j, h) holds k = 32j + 4c + b (low
// nibble of byte b) and 32j + 16 + 4c + b (high nibble); the two-bit word of sub-block j holds, at bit 8b + 2m, the
// high bits of weight b of group m = 2h + t
__global__ void repack_q6k(const uint8_t *__restrict__ src, uint8_t *__restrict__ dst, int N, int nb) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int tiles = (N + 15) / 16;
    if (id >= (size_t)tiles * nb * 32) return;
    const int lane = (int)(id % 32), sb = (int)((id / 32) % nb), tile = (int)(id / 32 / nb);
    const int g = lane / 4, c = lane % 4;
    uint8_t *out = dst + ((size_t)tile * nb + sb) * T6;
    uint32_t lo[16], hw[8];
#pragma unroll
    for (int j = 0; j < 8; j++) hw[j] = 0;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int row = tile * 16 + g + 8 * h;
        const uint8_t *blk = src + ((size_t)row * nb + sb) * BS6;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t wd = 0;
            const int half_ = j / 4, quarter = j % 4;
#pragma unroll
            for (int t = 0; t < 2; t++)
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    uint32_t q = 0;
                    if (row < N) {
                        const int l = 16 * t + 4 * c + b;  // within the quarter's 32
                        const uint8_t qlb = blk[64 * half_ + 32 * (quarter & 1) + l];
                        const uint8_t qhb = blk[128 + 32 * half_ + l];
                        const uint32_t low = quarter >= 2 ? qlb >> 4 : qlb & 0xF;
                        const uint32_t high = (qhb >> (2 * quarter)) & 3;
                        q = low | (high << 4);
                    }
                    wd |= (q & 0xF) << (8 * b + 4 * t);
                    hw[j] |= (q >> 4) << (8 * b + 2 * (2 * h + t));
                }
            lo[2 * j + h] = wd;
        }
    }
    // interleaved like Q4_K: piece i (4 words) of every lane is contiguous across the warp
    uint32_t all[24];
#pragma unroll
    for (int i = 0; i < 16; i++) all[i] = lo[i];
#pragma unroll
    for (int j = 0; j < 8; j++) all[16 + j] = hw[j];
#pragma unroll
    for (int i = 0; i < 6; i++)
        *reinterpret_cast<uint4 *>(out + (i * 32 + lane) * 16) = make_uint4(all[4 * i], all[4 * i + 1], all[4 * i + 2], all[4 * i + 3]);
    if (lane < 16) {
        const int row = tile * 16 + lane;
        uint8_t hd[32] = {0};
        if (row < N) {
            memcpy(hd, src + ((size_t)row * nb + sb) * BS6 + 192, 16);      // 16 int8 scales
            memcpy(hd + 16, src + ((size_t)row * nb + sb) * BS6 + 208, 2);  // d
        }
        memcpy(out + 3072 + lane * 32, hd, 32);
    }
}

// ------------------------------------------------------------------------------------------------------------ activations
// x [B][K] float -> xh [rows][K] half (rows >= B zero, saturated to the half range) and xs [K / 16][rows]: the sum of
// every 16 rounded values; one warp per (token row, 32 values)
__global__ void prep_x(const float *__restrict__ x, half *__restrict__ xh, float *__restrict__ xs, int B, int K,
                       int rows) {
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    const int groups = K / 32;
    if (wid >= rows * groups) return;
    const int n = wid / groups, grp = wid % groups;
    const float v = n < B ? x[(size_t)n * K + grp * 32 + lane] : 0.f;
    const half h = __float2half_rn(fminf(fmaxf(v, -65000.f), 65000.f));
    xh[(size_t)n * K + grp * 32 + lane] = h;
    float t = __half2float(h);
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0 || lane == 16) xs[(size_t)(grp * 2 + lane / 16) * rows + n] = t;
    const float t32 = t + __shfl_xor_sync(0xffffffffu, t, 16);
    if (lane == 0) xs[(size_t)(K / 16) * rows + (size_t)grp * rows + n] = t32;  // then the 32-sums [K/32][rows]
}

// PD1 check (IRONMULE_KG_PD1_CHECK=1, diagnostics only): count the halves of the first B rows where a producer's copy
// differs bit for bit from prep_x's
__global__ void pd1_compare(const half *__restrict__ a, const half *__restrict__ b, size_t n,
                            unsigned long long *__restrict__ counts) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (__half_as_ushort(a[i]) != __half_as_ushort(b[i])) atomicAdd(&counts[0], 1ull);
    if (threadIdx.x == 0) atomicAdd(&counts[1], (unsigned long long)min((size_t)blockDim.x, n - i));
}

// ----------------------------------------------------------------------------------------------------------------- compute
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// T4W: no cp.async before sm_80; a plain 16-byte copy (complete when issued), the kernels' barriers stay
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    *reinterpret_cast<uint4 *>(smem) = *reinterpret_cast<const uint4 *>(gmem);
}
__device__ __forceinline__ void cp_commit() {}
template <int N> __device__ __forceinline__ void cp_wait() {}
#else
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
#endif

template <int KS, int NT>
struct XCfg {
    static constexpr int ROWS = NT * 8;
    static constexpr int XROW = KS * 256 + 32;            // halves per token row (32 pad: conflict-free 16-byte reads)
    static constexpr int XB = ROWS * XROW * 2;            // bytes of one x buffer
    static constexpr int SB = KS * 16 * ROWS * 4;         // bytes of one buffer of 16-value sums [KS*16][ROWS]
    static constexpr int BUF = XB + SB;
    static constexpr int SMEM = 2 * BUF;
};

// grid (row tiles, K splits). The block's warps split into tiles x KS parts; at every step the block stages the x
// and 16-value sums of KS consecutive super-blocks into shared memory (cp.async, double buffered) and part p of every
// tile multiplies super-block step + p, its weights streamed straight from the tile-order copy into registers.
template <int TYPE, int WARPS, int KS, int NT, int DQ>
__global__ void __launch_bounds__(WARPS * 32)
w4r(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    using X = XCfg<KS, NT>;
    constexpr int TB = TYPE == TYPE_Q4_K ? T4 : T6;
    constexpr int TILES = WARPS / KS;
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ float red[KS > 1 ? WARPS : 1][KS > 1 ? NT * 4 * 32 : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256, rows8 = NT * 8;
    const int tile = blockIdx.x * TILES + warp / KS, part = warp % KS;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile * 16 < N;
    const uint8_t *base = wr + (size_t)tile * nb * TB;
    const int nsteps = (sb1 - sb0 + KS - 1) / KS;

    auto stage = [&](int buf, int step) {
        uint8_t *dst = smem + buf * X::BUF;
        const int sbs = sb0 + step * KS, kn = min(KS, sb1 - sbs);
        const int xr = kn * 256 / 8;  // 16-byte pieces per token row
        for (int i = threadIdx.x; i < X::ROWS * xr; i += WARPS * 32) {
            const int n = i / xr, piece = i % xr;
            cp_async16(dst + (n * X::XROW + piece * 8) * 2, xh + (size_t)n * K + (size_t)sbs * 256 + piece * 8);
        }
        const int sp = kn * 16 * X::ROWS / 4;  // 16-byte pieces of sums
        for (int i = threadIdx.x; i < sp; i += WARPS * 32)
            cp_async16(dst + X::XB + i * 16, xs + (size_t)sbs * 16 * X::ROWS + i * 4);
    };

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int i = 0; i < 4; i++) acc[nt][i] = 0.f;

    constexpr int NQ = TYPE == TYPE_Q4_K ? 4 : 6, NH = TYPE == TYPE_Q4_K ? 2 : 4;
    auto fetch = [&](int sb, uint4 (&q)[NQ], uint4 (&hd)[NH]) {
        const uint8_t *p = base + (size_t)sb * TB;
        if constexpr (TYPE == TYPE_Q4_K) {
#pragma unroll
            for (int i = 0; i < 4; i++) q[i] = ldg_stream(p + (i * 32 + lane) * 16);
            hd[0] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 16));
            hd[1] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16));
        } else {
#pragma unroll
            for (int i = 0; i < 6; i++) q[i] = ldg_stream(p + (i * 32 + lane) * 16);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                hd[2 * h] = __ldg(reinterpret_cast<const uint4 *>(p + 3072 + (g + 8 * h) * 32));
                hd[2 * h + 1] = __ldg(reinterpret_cast<const uint4 *>(p + 3072 + (g + 8 * h) * 32 + 16));
            }
        }
    };
    auto compute = [&](int buf, const uint4 (&Q)[NQ], const uint4 (&H)[NH]) {
        const half *xb = reinterpret_cast<const half *>(smem + buf * X::BUF) + part * 256;
        const float *sm = reinterpret_cast<const float *>(smem + buf * X::BUF + X::XB) + part * 16 * X::ROWS;
        if constexpr (TYPE == TYPE_Q4_K) {
            float dd[2], dm[2];
            uint32_t sw[2][2], mw[2][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint4 hv = H[h];
                dd[h] = __half2float(__ushort_as_half((unsigned short)(hv.x & 0xFFFF)));
                dm[h] = __half2float(__ushort_as_half((unsigned short)(hv.x >> 16)));
                sw[h][0] = hv.y & 0x3F3F3F3Fu;
                mw[h][0] = hv.z & 0x3F3F3F3Fu;
                sw[h][1] = (hv.w & 0x0F0F0F0Fu) | ((hv.y >> 2) & 0x30303030u);
                mw[h][1] = ((hv.w >> 4) & 0x0F0F0F0Fu) | ((hv.z >> 2) & 0x30303030u);
            }
            if constexpr (DQ) {
                // dequantize once (fp16, q * S - m) and reuse the fragment for every 8-token tile
                const half2 h1024 = __half2half2(__ushort_as_half(0x6400));
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint32_t a[2][4];
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int sh = 8 * (j & 3);
                        const half2 s2 = __float2half2_rn(dd[h] * byte_f(sw[h][j >> 2], j & 3));
                        const half2 m2 = __float2half2_rn(-dm[h] * byte_f(mw[h][j >> 2], j & 3));
                        const uint32_t wd = wget(Q, 2 * j + h);
#pragma unroll
                        for (int t = 0; t < 2; t++) {
                            const uint32_t v = (t ? wd >> 4 : wd) & 0x0F0F0F0Fu;
                            const uint32_t p0 = __byte_perm(v, 0x64u, 0x4140), p1 = __byte_perm(v, 0x64u, 0x4342);
                            const half2 w0 = __hfma2(__hsub2(*reinterpret_cast<const half2 *>(&p0), h1024), s2, m2);
                            const half2 w1 = __hfma2(__hsub2(*reinterpret_cast<const half2 *>(&p1), h1024), s2, m2);
                            a[t][h] = h2_bits(w0);
                            a[t][2 + h] = h2_bits(w1);
                        }
                    }
#pragma unroll
                    for (int nt = 0; nt < NT; nt++) {
                        const uint4 bx = *reinterpret_cast<const uint4 *>(xb + (nt * 8 + g) * X::XROW + 32 * j + 8 * c);
                        mma16816(acc[nt], a[0][0], a[0][1], a[0][2], a[0][3], bx.x, bx.y);
                        mma16816(acc[nt], a[1][0], a[1][1], a[1][2], a[1][3], bx.z, bx.w);
                    }
                }
                return;
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                float S[2], NB[2];
                uint32_t a[2][4];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int sh = 8 * (j & 3);
                    S[h] = dd[h] * byte_f(sw[h][j >> 2], j & 3);
                    NB[h] = -fmaf(1024.f, S[h], dm[h] * byte_f(mw[h][j >> 2], j & 3));
                    const uint32_t wd = wget(Q, 2 * j + h);
                    const uint32_t v0 = wd & 0x0F0F0F0Fu, v1 = (wd >> 4) & 0x0F0F0F0Fu;
                    a[0][h] = __byte_perm(v0, 0x64u, 0x4140);
                    a[0][2 + h] = __byte_perm(v0, 0x64u, 0x4342);
                    a[1][h] = __byte_perm(v1, 0x64u, 0x4140);
                    a[1][2 + h] = __byte_perm(v1, 0x64u, 0x4342);
                }
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    const uint4 bx = *reinterpret_cast<const uint4 *>(xb + (nt * 8 + g) * X::XROW + 32 * j + 8 * c);
                    const float2 s0 = *reinterpret_cast<const float2 *>(sm + (2 * j) * X::ROWS + nt * 8 + 2 * c);
                    const float2 s1 = *reinterpret_cast<const float2 *>(sm + (2 * j + 1) * X::ROWS + nt * 8 + 2 * c);
                    const float x0 = s0.x + s1.x, x1 = s0.y + s1.y;
                    float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(tmp, a[0][0], a[0][1], a[0][2], a[0][3], bx.x, bx.y);
                    mma16816(tmp, a[1][0], a[1][1], a[1][2], a[1][3], bx.z, bx.w);
                    acc[nt][0] = fmaf(S[0], tmp[0], fmaf(NB[0], x0, acc[nt][0]));
                    acc[nt][1] = fmaf(S[0], tmp[1], fmaf(NB[0], x1, acc[nt][1]));
                    acc[nt][2] = fmaf(S[1], tmp[2], fmaf(NB[1], x0, acc[nt][2]));
                    acc[nt][3] = fmaf(S[1], tmp[3], fmaf(NB[1], x1, acc[nt][3]));
                }
            }
        } else {
            float dd[2];
#pragma unroll
            for (int h = 0; h < 2; h++) dd[h] = __half2float(__ushort_as_half((unsigned short)(H[2 * h + 1].x & 0xFFFF)));
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint32_t hw = wget(Q, 16 + j);
                uint32_t a[2][4];
                float S[2][2], NB[2][2];  // [t][h]
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t wd = wget(Q, 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = ((t ? wd >> 4 : wd) & 0x0F0F0F0Fu) |
                                           (((hw >> (2 * (2 * h + t))) & 0x03030303u) << 4);
                        a[t][h] = __byte_perm(v, 0x64u, 0x4140);
                        a[t][2 + h] = __byte_perm(v, 0x64u, 0x4342);
                        const int si = 8 * (j / 4) + t + 2 * (j % 4);
                        const uint4 sv = H[2 * h];
                        const uint32_t sword = si < 4 ? sv.x : si < 8 ? sv.y : si < 12 ? sv.z : sv.w;
                        S[t][h] = dd[h] * (float)(int8_t)((sword >> (8 * (si & 3))) & 0xFF);
                        NB[t][h] = -1056.f * S[t][h];
                    }
                }
                if constexpr (DQ) {
                    const half2 h1056 = __float2half2_rn(1056.f);
#pragma unroll
                    for (int t = 0; t < 2; t++)
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const half2 s2 = __float2half2_rn(S[t][h]);
                            a[t][h] = h2_bits(__hmul2(__hsub2(bits_h2(a[t][h]), h1056), s2));
                            a[t][2 + h] = h2_bits(__hmul2(__hsub2(bits_h2(a[t][2 + h]), h1056), s2));
                        }
#pragma unroll
                    for (int nt = 0; nt < NT; nt++)
#pragma unroll
                        for (int t = 0; t < 2; t++) {
                            const uint2 bx = *reinterpret_cast<const uint2 *>(xb + (nt * 8 + g) * X::XROW + 32 * j + 16 * t + 4 * c);
                            mma16816(acc[nt], a[t][0], a[t][1], a[t][2], a[t][3], bx.x, bx.y);
                        }
                    continue;
                }
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint2 bx = *reinterpret_cast<const uint2 *>(xb + (nt * 8 + g) * X::XROW + 32 * j + 16 * t + 4 * c);
                        const float2 sv = *reinterpret_cast<const float2 *>(sm + (2 * j + t) * X::ROWS + nt * 8 + 2 * c);
                        float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                        mma16816(tmp, a[t][0], a[t][1], a[t][2], a[t][3], bx.x, bx.y);
                        acc[nt][0] = fmaf(S[t][0], tmp[0], fmaf(NB[t][0], sv.x, acc[nt][0]));
                        acc[nt][1] = fmaf(S[t][0], tmp[1], fmaf(NB[t][0], sv.y, acc[nt][1]));
                        acc[nt][2] = fmaf(S[t][1], tmp[2], fmaf(NB[t][1], sv.x, acc[nt][2]));
                        acc[nt][3] = fmaf(S[t][1], tmp[3], fmaf(NB[t][1], sv.y, acc[nt][3]));
                    }
            }
        }
    };

    uint4 qa[NQ], ha[NH], qb[NQ], hb[NH];
    if (nsteps > 0) {
        stage(0, 0);
        cp_commit();
        if (live && sb0 + part < sb1) fetch(sb0 + part, qa, ha);
    }
    for (int step = 0; step < nsteps; step += 2) {
        // even step: buffer 0, registers a; prefetch the next step's x (buffer 1) and weights (registers b)
        if (step + 1 < nsteps) stage(1, step + 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        {
            const int sb = sb0 + step * KS + part, nsb = sb + KS;
            if (live && nsb < sb1) fetch(nsb, qb, hb);
            if (live && sb < sb1) compute(0, qa, ha);
        }
        __syncthreads();
        if (step + 1 >= nsteps) break;
        // odd step: buffer 1, registers b
        if (step + 2 < nsteps) stage(0, step + 2);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        {
            const int sb = sb0 + (step + 1) * KS + part, nsb = sb + KS;
            if (live && nsb < sb1) fetch(nsb, qa, ha);
            if (live && sb < sb1) compute(1, qb, hb);
        }
        __syncthreads();
    }
    cp_wait<0>();

    if constexpr (KS > 1) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) red[warp][(nt * 4 + i) * 32 + lane] = acc[nt][i];
        __syncthreads();
        if (part != 0) return;
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int p = 1; p < KS; p++) acc[nt][i] += red[warp + p][(nt * 4 + i) * 32 + lane];
    }
    if (!live) return;
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = tile * 16 + g + 8 * (i / 2);
            const int tok = nt * 8 + 2 * c + (i % 2);
            if (row < N && tok < B) o[(size_t)tok * N + row] = acc[nt][i];
        }
}


// v7 (Q4_K, 1-16 tokens): no shared memory and no barriers; every warp prefetches, one super-block ahead and into
// registers, the weights of its MT 16-row tiles (and with PX its tokens' x fragments and their 32-value sums), so no
// load sits in the arithmetic's path; each x fragment serves all MT tiles
template <int WARPS, int KS, int NT, int PX, int MT, int FP>
__global__ void __launch_bounds__(WARPS * 32)
w4p(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    constexpr int TILES = WARPS / KS;
    __shared__ float red[KS > 1 ? WARPS : 1][KS > 1 ? MT * NT * 4 * 32 : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256, rows8 = NT * 8;
    const int tile0 = (blockIdx.x * TILES + warp / KS) * MT, part = warp % KS;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile0 * 16 < N;
    const int tiles = (N + 15) / 16;
    const float *xs32 = xs + (size_t)(K / 16) * rows8;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct XB { uint4 x[NT][8]; float2 s[NT][8]; };
    struct Buf { uint4 q[MT][4], h[MT][2]; XB xb[1]; int sb; };
    auto fetch_x = [&](int sb, XB &b) {
        if constexpr (FP) {  // xs carries llama.cpp's float activations; no prep kernel ran
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int j = 0; j < 8; j++)
                    fused_x(xs, K, B, nt * 8 + g, (size_t)sb * 256 + 32 * j + 8 * c, c, b.x[nt][j], b.s[nt][j]);
            return;
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const int n = nt * 8 + g;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                b.x[nt][j] = n < B ? __ldg(reinterpret_cast<const uint4 *>(xh + (size_t)n * K + (size_t)sb * 256 + 32 * j + 8 * c))
                                   : make_uint4(0, 0, 0, 0);
                b.s[nt][j] = __ldg(reinterpret_cast<const float2 *>(xs32 + ((size_t)sb * 8 + j) * rows8 + nt * 8 + 2 * c));
            }
        }
    };
    auto fetch = [&](int sb, Buf &b) {
        b.sb = sb;
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const uint8_t *p = wr + ((size_t)t * nb + sb) * T4;
#pragma unroll
            for (int i = 0; i < 4; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
            b.h[m][0] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 16));
            b.h[m][1] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16));
        }
        if constexpr (PX) fetch_x(sb, b.xb[0]);
    };
    auto compute = [&](const Buf &b) {
        XB local;
        if constexpr (!PX) fetch_x(b.sb, local);
        const XB &X = PX ? b.xb[0] : local;
        float dd[MT][2], dm[MT][2];
        uint32_t sw[MT][2][2], mw[MT][2][2];
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint4 hv = b.h[m][h];
                dd[m][h] = __half2float(__ushort_as_half((unsigned short)(hv.x & 0xFFFF)));
                dm[m][h] = __half2float(__ushort_as_half((unsigned short)(hv.x >> 16)));
                sw[m][h][0] = hv.y & 0x3F3F3F3Fu;
                mw[m][h][0] = hv.z & 0x3F3F3F3Fu;
                sw[m][h][1] = (hv.w & 0x0F0F0F0Fu) | ((hv.y >> 2) & 0x30303030u);
                mw[m][h][1] = ((hv.w >> 4) & 0x0F0F0F0Fu) | ((hv.z >> 2) & 0x30303030u);
            }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float S[MT][2], NB[MT][2];
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int sh = 8 * (j & 3);
                    S[m][h] = dd[m][h] * byte_f(sw[m][h][j >> 2], j & 3);
                    NB[m][h] = -fmaf(1024.f, S[m][h], dm[m][h] * byte_f(mw[m][h][j >> 2], j & 3));
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
                    const uint32_t v0 = wd & 0x0F0F0F0Fu, v1 = (wd >> 4) & 0x0F0F0F0Fu;
                    a[m][0][h] = __byte_perm(v0, 0x64u, 0x4140);
                    a[m][0][2 + h] = __byte_perm(v0, 0x64u, 0x4342);
                    a[m][1][h] = __byte_perm(v1, 0x64u, 0x4140);
                    a[m][1][2 + h] = __byte_perm(v1, 0x64u, 0x4342);
                }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint4 bx = X.x[nt][j];
                const float2 sv = X.s[nt][j];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    float tmp[4] = {0.f, 0.f, 0.f, 0.f};
                    mma16816(tmp, a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], bx.x, bx.y);
                    mma16816(tmp, a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], bx.z, bx.w);
                    acc[m][nt][0] = fmaf(S[m][0], tmp[0], fmaf(NB[m][0], sv.x, acc[m][nt][0]));
                    acc[m][nt][1] = fmaf(S[m][0], tmp[1], fmaf(NB[m][0], sv.y, acc[m][nt][1]));
                    acc[m][nt][2] = fmaf(S[m][1], tmp[2], fmaf(NB[m][1], sv.x, acc[m][nt][2]));
                    acc[m][nt][3] = fmaf(S[m][1], tmp[3], fmaf(NB[m][1], sv.y, acc[m][nt][3]));
                }
            }
        }
    };
    if (live) {
        Buf ba, bb;
        int sb = sb0 + part;
        if (sb < sb1) fetch(sb, ba);
        for (; sb < sb1; sb += 2 * KS) {
            if (sb + KS < sb1) fetch(sb + KS, bb);
            compute(ba);
            if (sb + KS >= sb1) break;
            if (sb + 2 * KS < sb1) fetch(sb + 2 * KS, ba);
            compute(bb);
        }
    }
    if constexpr (KS > 1) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) red[warp][((m * NT + nt) * 4 + i) * 32 + lane] = acc[m][nt][i];
        __syncthreads();
        if (part != 0) return;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int p = 1; p < KS; p++) acc[m][nt][i] += red[warp + p][((m * NT + nt) * 4 + i) * 32 + lane];
    }
    if (!live) return;
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                const int tok = nt * 8 + 2 * c + (i % 2);
                if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
            }
}

__global__ void reduce_split(const float *__restrict__ part, float *__restrict__ y, size_t n, int splits) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    for (int k = 0; k < splits; k++) s += part[(size_t)k * n + i];
    y[i] = s;
}

typedef int (*launch_fn)(const uint8_t *, const half *, const float *, float *, float *, int, int, int, int,
                         cudaStream_t);
// RF1: this call's per-range counters (zeroed, RF1_COUNTERS ints in the stream's scratch), nullptr: reduce_split
thread_local int *rf1_counters = nullptr;
constexpr unsigned RF1_COUNTERS = 4096;

template <int TYPE, int WARPS, int KS, int NT, int DQ>
int launch(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
           cudaStream_t s) {
    constexpr int TILES = WARPS / KS;
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    const int tiles = (N + 15) / 16;
    dim3 grid((tiles + TILES - 1) / TILES, splits);
    using X = XCfg<KS, NT>;
    auto kern = w4r<TYPE, WARPS, KS, NT, DQ>;
    static bool attr = false;
    if (!attr) {
        if (X::SMEM > 48 * 1024 &&
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, X::SMEM) != cudaSuccess)
            return 2;
        attr = true;
    }
    kern<<<grid, WARPS * 32, X::SMEM, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int WARPS, int KS, int NT, int PX, int MT, int FP>
int launch_p(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
             cudaStream_t s) {
    constexpr int TILES = WARPS / KS;
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    dim3 grid((((N + 15) / 16 + MT - 1) / MT + TILES - 1) / TILES, splits);
    w4p<WARPS, KS, NT, PX, MT, FP><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// DK1 (2026-11-24, lab only): IRONMULE_W4_ABL removes one part of v9 (w4d) to locate its limit; wrong results by design,
// never set in a product build. 1: weights streamed, no x, no dequantisation, no mma; 2: every super-block reads the
// first super-block's weights again (loads issued, served from L2, no DRAM stream; x, dequantisation and mma real);
// 3: x read from the first sub-block only (weights and math real); 4: no mma (weights, x and dequantisation real);
// 5: no dequantisation (raw nibble words into the mma; weights, x and mma real).
// IRONMULE_W4_PIN: dynamic shared memory per block, so that an ablation with fewer registers keeps the full kernel's
// blocks per SM; IRONMULE_W4_CARVE: the shared-memory carveout it asks for (the rest of the 192 KB is L1, which holds x).
#ifndef IRONMULE_W4_ABL
#define IRONMULE_W4_ABL 0
#endif
#ifndef IRONMULE_W4_PIN
#define IRONMULE_W4_PIN 0
#endif
#ifdef IRONMULE_W4_CARVE  // percent of the unified 192 KB that becomes shared memory (the rest is L1, which holds x)
#define IRONMULE_W4_CARVE_PCT IRONMULE_W4_CARVE
#else
#define IRONMULE_W4_CARVE_PCT 100
#endif
// v9 (Q4_K, 9-32 tokens): v7's barrier-free register prefetch of MT 16-row tiles per warp, with the dequantize-once
// arithmetic of w4r's DQ path (fp16 weights q * S - m, so the tensor cores accumulate straight into the result and no
// 32-value sums are read). Each x fragment, read through L1 one sub-block ahead, serves all MT tiles: at 32 tokens a
// one-tile warp reads 7 bytes of x per weight byte, an MT-tile warp 7 / MT.
template <int WARPS, int KS, int NT, int MT, int MINB = 1>  // MINB: blocks per SM the compiler must fit (registers)
__global__ void __launch_bounds__(WARPS * 32, MINB)
w4d(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    constexpr int TILES = WARPS / KS;
    __shared__ float red[KS > 1 ? WARPS : 1][KS > 1 ? MT * NT * 4 * 32 : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * TILES + warp / KS) * MT, part = warp % KS;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile0 * 16 < N;
    const int tiles = (N + 15) / 16;
    float acc[MT][NT][4];
    uint32_t sink = 0;  // DK1 ablations: keeps what a removed part would have used alive
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct Buf { uint4 q[MT][4], h[MT][2]; };
    auto fetch = [&](int sb, Buf &b) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const uint8_t *p = wr + ((size_t)t * nb + (IRONMULE_W4_ABL == 2 ? sb0 + part : sb)) * T4;
#pragma unroll
            for (int i = 0; i < 4; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
            b.h[m][0] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 16));
            b.h[m][1] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16));
        }
    };
    // prep_x zero-fills token rows B..NT*8-1, so every row of the tile may be read
    auto xload = [&](int sb, int j, uint4 (&xv)[NT]) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
            xv[nt] = __ldg(reinterpret_cast<const uint4 *>(xh + (size_t)(nt * 8 + g) * K +
                                                           (IRONMULE_W4_ABL == 3 ? 0 : (size_t)sb * 256 + 32 * j) + 8 * c));
    };
    const half2 h1024 = __half2half2(__ushort_as_half(0x6400));
    // 0x64 in a register ptxas cannot fold (blockDim.y is 1 at run time, unknown when compiling): a PRMT takes one
    // immediate, and with 0x64 as that immediate ptxas rebuilt the selector 0x4140 / 0x4342 in a register before
    // every PRMT (SASS of v9: about 300 IMAD.U32 of 2,400 instructions); this way the selector is the immediate
    const uint32_t k64 = 0x63u + blockDim.y;
    auto compute = [&](const Buf &b, int sb) {
        if constexpr (IRONMULE_W4_ABL == 1) {
#pragma unroll
            for (int m = 0; m < MT; m++) {
#pragma unroll
                for (int i = 0; i < 4; i++) sink ^= b.q[m][i].x ^ b.q[m][i].y ^ b.q[m][i].z ^ b.q[m][i].w;
                sink ^= b.h[m][0].x ^ b.h[m][0].y ^ b.h[m][0].z ^ b.h[m][0].w ^ b.h[m][1].x ^ b.h[m][1].y ^ b.h[m][1].z ^
                        b.h[m][1].w;
            }
            return;
        }
        // the two rows' (g, g + 8) super-block scale and minimum as half2 {row g, row g + 8}; the 6-bit sub-block
        // scales and minimums become half2 {s_g, s_g8} by the 1024 + s trick (exact), and their products with d are
        // rounded once to half, as the float path's rounding of its exact product: no int-to-float conversion
        uint32_t D[MT], NM[MT], sw[MT][2][2], mw[MT][2][2];
#pragma unroll
        for (int m = 0; m < MT; m++) {
            D[m] = __byte_perm(b.h[m][0].x, b.h[m][1].x, 0x5410);
            NM[m] = h2_bits(__hneg2(bits_h2(__byte_perm(b.h[m][0].x, b.h[m][1].x, 0x7632))));
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint4 hv = b.h[m][h];
                sw[m][h][0] = hv.y & 0x3F3F3F3Fu;
                mw[m][h][0] = hv.z & 0x3F3F3F3Fu;
                sw[m][h][1] = (hv.w & 0x0F0F0F0Fu) | ((hv.y >> 2) & 0x30303030u);
                mw[m][h][1] = ((hv.w >> 4) & 0x0F0F0F0Fu) | ((hv.z >> 2) & 0x30303030u);
            }
        }
        uint4 xv[2][NT];
        xload(sb, 0, xv[0]);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j < 7) xload(sb, j + 1, xv[(j + 1) & 1]);
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint32_t pick = (uint32_t)(j & 3) | ((uint32_t)(4 + (j & 3)) << 4);
                const half2 sc = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(sw[m][0][j >> 2], sw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(D[m]));
                const half2 mn = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(mw[m][0][j >> 2], mw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(NM[m]));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const half2 s2 = h ? __high2half2(sc) : __low2half2(sc);
                    const half2 m2 = h ? __high2half2(mn) : __low2half2(mn);
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        if constexpr (IRONMULE_W4_ABL == 5) {  // raw nibble words into the mma: no dequantisation
                            a[m][t][h] = t ? wd >> 4 : wd;
                            a[m][t][2 + h] = t ? wd >> 12 : wd >> 8;
                            continue;
                        }
                        const uint32_t v = (t ? wd >> 4 : wd) & 0x0F0F0F0Fu;
                        const uint32_t p0 = __byte_perm(v, k64, 0x4140), p1 = __byte_perm(v, k64, 0x4342);
                        a[m][t][h] = h2_bits(__hfma2(__hsub2(bits_h2(p0), h1024), s2, m2));
                        a[m][t][2 + h] = h2_bits(__hfma2(__hsub2(bits_h2(p1), h1024), s2, m2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint4 bx = xv[j & 1][nt];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    if constexpr (IRONMULE_W4_ABL == 4) {
                        if (nt == 0)
#pragma unroll
                            for (int t = 0; t < 2; t++) sink ^= a[m][t][0] ^ a[m][t][1] ^ a[m][t][2] ^ a[m][t][3];
                        sink ^= bx.x ^ bx.y ^ bx.z ^ bx.w;
                    } else {
                        mma16816(acc[m][nt], a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], bx.x, bx.y);
                        mma16816(acc[m][nt], a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], bx.z, bx.w);
                    }
                }
            }
        }
    };
    if (live) {
        Buf ba, bb;
        int sb = sb0 + part;
        if (sb < sb1) fetch(sb, ba);
        for (; sb < sb1; sb += 2 * KS) {
            if (sb + KS < sb1) fetch(sb + KS, bb);
            compute(ba, sb);
            if (sb + KS >= sb1) break;
            if (sb + 2 * KS < sb1) fetch(sb + 2 * KS, ba);
            compute(bb, sb + KS);
        }
    }
    if constexpr (KS > 1) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) red[warp][((m * NT + nt) * 4 + i) * 32 + lane] = acc[m][nt][i];
        __syncthreads();
        if (part != 0) return;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int p = 1; p < KS; p++) acc[m][nt][i] += red[warp + p][((m * NT + nt) * 4 + i) * 32 + lane];
    }
    if (!live) return;
    if (IRONMULE_W4_ABL != 0 && sink == 0x9E3779B9u) out[lane] = 1.f;  // never true in practice; keeps sink's inputs
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                const int tok = nt * 8 + 2 * c + (i % 2);
                if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
            }
}

// v13 "QKV" (Q4_K, 2026-11-24, experimental): v9 (w4d) over up to three matrices that read the same activations (a
// layer's q, k and v projections) as one virtual matrix, their 16-row tiles one after another: one launch and one
// prepared x instead of three, so that the 1024-row k and v (9 % of the bandwidth alone at 32 tokens, A100 pod 26)
// fill the GPU together with q. Each segment keeps its own weights and output; a K split reduces per segment. Every
// segment's row count is a multiple of 16.
struct Seg3 {
    const uint8_t *wr[3];
    float *out[3];
    int tend[3];    // cumulative tile counts
    int n[3];       // rows of each segment
    int before[3];  // rows of the segments before it
    int nseg, ntot;
};

__global__ void reduce_split_seg(const float *__restrict__ part, const Seg3 sg, int B, int splits) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, n = (size_t)B * sg.ntot;
    if (i >= n) return;
    float s = 0.f;
    for (int k = 0; k < splits; k++) s += part[(size_t)k * n + i];
    int sgi = 0;
    if (sg.nseg > 1 && i >= (size_t)B * sg.before[1]) sgi = 1;
    if (sg.nseg > 2 && i >= (size_t)B * sg.before[2]) sgi = 2;
    sg.out[sgi][i - (size_t)B * sg.before[sgi]] = s;
}

template <int WARPS, int KS, int NT, int MT, int MINB = 1>
__global__ void __launch_bounds__(WARPS * 32, MINB)
w4dq(const Seg3 sg, const half *__restrict__ xh, const float *__restrict__ xs, float *__restrict__ pbuf, int K, int B,
     int sb_per_split) {
    constexpr int TILES = WARPS / KS;
    __shared__ float red[KS > 1 ? WARPS : 1][KS > 1 ? MT * NT * 4 * 32 : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * TILES + warp / KS) * MT, part = warp % KS;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const int tiles = sg.tend[sg.nseg - 1];  // QKV: the segments' 16-row tiles one after another
    const bool live = tile0 < tiles;
    const int N = sg.ntot;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct Buf { uint4 q[MT][4], h[MT][2]; };
    auto fetch = [&](int sb, Buf &b) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const int sgi = t < sg.tend[0] ? 0 : (t < sg.tend[1] ? 1 : 2);
            const uint8_t *p = sg.wr[sgi] + ((size_t)(t - (sgi ? sg.tend[sgi - 1] : 0)) * nb + sb) * T4;
#pragma unroll
            for (int i = 0; i < 4; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
            b.h[m][0] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 16));
            b.h[m][1] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16));
        }
    };
    // prep_x zero-fills token rows B..NT*8-1, so every row of the tile may be read
    auto xload = [&](int sb, int j, uint4 (&xv)[NT]) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
            xv[nt] = __ldg(reinterpret_cast<const uint4 *>(xh + (size_t)(nt * 8 + g) * K + (size_t)sb * 256 + 32 * j + 8 * c));
    };
    const half2 h1024 = __half2half2(__ushort_as_half(0x6400));
    // 0x64 in a register ptxas cannot fold (blockDim.y is 1 at run time, unknown when compiling): a PRMT takes one
    // immediate, and with 0x64 as that immediate ptxas rebuilt the selector 0x4140 / 0x4342 in a register before
    // every PRMT (SASS of v9: about 300 IMAD.U32 of 2,400 instructions); this way the selector is the immediate
    const uint32_t k64 = 0x63u + blockDim.y;
    auto compute = [&](const Buf &b, int sb) {
        // the two rows' (g, g + 8) super-block scale and minimum as half2 {row g, row g + 8}; the 6-bit sub-block
        // scales and minimums become half2 {s_g, s_g8} by the 1024 + s trick (exact), and their products with d are
        // rounded once to half, as the float path's rounding of its exact product: no int-to-float conversion
        uint32_t D[MT], NM[MT], sw[MT][2][2], mw[MT][2][2];
#pragma unroll
        for (int m = 0; m < MT; m++) {
            D[m] = __byte_perm(b.h[m][0].x, b.h[m][1].x, 0x5410);
            NM[m] = h2_bits(__hneg2(bits_h2(__byte_perm(b.h[m][0].x, b.h[m][1].x, 0x7632))));
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint4 hv = b.h[m][h];
                sw[m][h][0] = hv.y & 0x3F3F3F3Fu;
                mw[m][h][0] = hv.z & 0x3F3F3F3Fu;
                sw[m][h][1] = (hv.w & 0x0F0F0F0Fu) | ((hv.y >> 2) & 0x30303030u);
                mw[m][h][1] = ((hv.w >> 4) & 0x0F0F0F0Fu) | ((hv.z >> 2) & 0x30303030u);
            }
        }
        uint4 xv[2][NT];
        xload(sb, 0, xv[0]);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j < 7) xload(sb, j + 1, xv[(j + 1) & 1]);
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint32_t pick = (uint32_t)(j & 3) | ((uint32_t)(4 + (j & 3)) << 4);
                const half2 sc = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(sw[m][0][j >> 2], sw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(D[m]));
                const half2 mn = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(mw[m][0][j >> 2], mw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(NM[m]));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const half2 s2 = h ? __high2half2(sc) : __low2half2(sc);
                    const half2 m2 = h ? __high2half2(mn) : __low2half2(mn);
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = (t ? wd >> 4 : wd) & 0x0F0F0F0Fu;
                        const uint32_t p0 = __byte_perm(v, k64, 0x4140), p1 = __byte_perm(v, k64, 0x4342);
                        a[m][t][h] = h2_bits(__hfma2(__hsub2(bits_h2(p0), h1024), s2, m2));
                        a[m][t][2 + h] = h2_bits(__hfma2(__hsub2(bits_h2(p1), h1024), s2, m2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint4 bx = xv[j & 1][nt];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    mma16816(acc[m][nt], a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], bx.x, bx.y);
                    mma16816(acc[m][nt], a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], bx.z, bx.w);
                }
            }
        }
    };
    if (live) {
        Buf ba, bb;
        int sb = sb0 + part;
        if (sb < sb1) fetch(sb, ba);
        for (; sb < sb1; sb += 2 * KS) {
            if (sb + KS < sb1) fetch(sb + KS, bb);
            compute(ba, sb);
            if (sb + KS >= sb1) break;
            if (sb + 2 * KS < sb1) fetch(sb + 2 * KS, ba);
            compute(bb, sb + KS);
        }
    }
    if constexpr (KS > 1) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) red[warp][((m * NT + nt) * 4 + i) * 32 + lane] = acc[m][nt][i];
        __syncthreads();
        if (part != 0) return;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int p = 1; p < KS; p++) acc[m][nt][i] += red[warp + p][((m * NT + nt) * 4 + i) * 32 + lane];
    }
    if (!live) return;
#pragma unroll
    for (int m = 0; m < MT; m++) {
        const int t = tile0 + m;
        if (t >= tiles) continue;
        const int sgi = t < sg.tend[0] ? 0 : (t < sg.tend[1] ? 1 : 2);
        const int n = sg.n[sgi], first = sgi ? sg.tend[sgi - 1] : 0;
        float *o = gridDim.y == 1 ? sg.out[sgi] : pbuf + (size_t)blockIdx.y * B * N + (size_t)B * sg.before[sgi];
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (t - first) * 16 + g + 8 * (i / 2);
                const int tok = nt * 8 + 2 * c + (i % 2);
                if (row < n && tok < B) o[(size_t)tok * n + row] = acc[m][nt][i];
            }
    }
}

// v12 "PS" (Q4_K, 9-32 tokens, 2026-11-24, experimental): v9 (w4d) on a tile-order copy whose rows carry their eight
// sub-block scales and negated minimums precomputed in fp16 (repack_q4k_pre, T4P = 2,560 bytes a tile and super-block
// against 2,304): 11 % more weight bytes for fewer instructions per sub-block. The products are rounded as w4d rounds
// them, so the results equal w4d's bit for bit. Opt-in (IRONMULE_W4_PRE=1): a matrix it serves keeps a second copy.
__device__ __forceinline__ uint32_t wget1(const uint4 v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int WARPS, int KS, int NT, int MT, int MINB = 1>  // MINB: blocks per SM the compiler must fit (registers)
__global__ void __launch_bounds__(WARPS * 32, MINB)
w4dp(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    constexpr int TILES = WARPS / KS;
    __shared__ float red[KS > 1 ? WARPS : 1][KS > 1 ? MT * NT * 4 * 32 : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * TILES + warp / KS) * MT, part = warp % KS;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile0 * 16 < N;
    const int tiles = (N + 15) / 16;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct Buf { uint4 q[MT][4], h[MT][4]; };  // PS: S and M of rows g and g + 8
    auto fetch = [&](int sb, Buf &b) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const uint8_t *p = wr + ((size_t)t * nb + sb) * T4P;
#pragma unroll
            for (int i = 0; i < 4; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
            b.h[m][0] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 32));             // S, row g
            b.h[m][1] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 32 + 16));        // M, row g
            b.h[m][2] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 32));       // S, row g + 8
            b.h[m][3] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 32 + 16));  // M, row g + 8
        }
    };
    // prep_x zero-fills token rows B..NT*8-1, so every row of the tile may be read
    auto xload = [&](int sb, int j, uint4 (&xv)[NT]) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
            xv[nt] = __ldg(reinterpret_cast<const uint4 *>(xh + (size_t)(nt * 8 + g) * K + (size_t)sb * 256 + 32 * j + 8 * c));
    };
    const half2 h1024 = __half2half2(__ushort_as_half(0x6400));
    // 0x64 in a register ptxas cannot fold (blockDim.y is 1 at run time, unknown when compiling): a PRMT takes one
    // immediate, and with 0x64 as that immediate ptxas rebuilt the selector 0x4140 / 0x4342 in a register before
    // every PRMT (SASS of v9: about 300 IMAD.U32 of 2,400 instructions); this way the selector is the immediate
    const uint32_t k64 = 0x63u + blockDim.y;
    auto compute = [&](const Buf &b, int sb) {
        // the two rows' (g, g + 8) super-block scale and minimum as half2 {row g, row g + 8}; the 6-bit sub-block
        // scales and minimums become half2 {s_g, s_g8} by the 1024 + s trick (exact), and their products with d are
        // rounded once to half, as the float path's rounding of its exact product: no int-to-float conversion
        uint4 xv[2][NT];
        xload(sb, 0, xv[0]);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j < 7) xload(sb, j + 1, xv[(j + 1) & 1]);
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                // PS: {row g, row g + 8} of sub-block j, one byte permute each (w4d: two, a subtract, a multiply)
                const uint32_t sel = (j & 1) ? 0x7632u : 0x5410u;
                const half2 sc = bits_h2(__byte_perm(wget1(b.h[m][0], j >> 1), wget1(b.h[m][2], j >> 1), sel));
                const half2 mn = bits_h2(__byte_perm(wget1(b.h[m][1], j >> 1), wget1(b.h[m][3], j >> 1), sel));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const half2 s2 = h ? __high2half2(sc) : __low2half2(sc);
                    const half2 m2 = h ? __high2half2(mn) : __low2half2(mn);
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = (t ? wd >> 4 : wd) & 0x0F0F0F0Fu;
                        const uint32_t p0 = __byte_perm(v, k64, 0x4140), p1 = __byte_perm(v, k64, 0x4342);
                        a[m][t][h] = h2_bits(__hfma2(__hsub2(bits_h2(p0), h1024), s2, m2));
                        a[m][t][2 + h] = h2_bits(__hfma2(__hsub2(bits_h2(p1), h1024), s2, m2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint4 bx = xv[j & 1][nt];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    mma16816(acc[m][nt], a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], bx.x, bx.y);
                    mma16816(acc[m][nt], a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], bx.z, bx.w);
                }
            }
        }
    };
    if (live) {
        Buf ba, bb;
        int sb = sb0 + part;
        if (sb < sb1) fetch(sb, ba);
        for (; sb < sb1; sb += 2 * KS) {
            if (sb + KS < sb1) fetch(sb + KS, bb);
            compute(ba, sb);
            if (sb + KS >= sb1) break;
            if (sb + 2 * KS < sb1) fetch(sb + 2 * KS, ba);
            compute(bb, sb + KS);
        }
    }
    if constexpr (KS > 1) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) red[warp][((m * NT + nt) * 4 + i) * 32 + lane] = acc[m][nt][i];
        __syncthreads();
        if (part != 0) return;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int p = 1; p < KS; p++) acc[m][nt][i] += red[warp + p][((m * NT + nt) * 4 + i) * 32 + lane];
    }
    if (!live) return;
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                const int tok = nt * 8 + 2 * c + (i % 2);
                if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
            }
}

// v10 (Q4_K, 9-32 tokens): v9's arithmetic with v8's weight path. ptxas -v (sm_80) put v9's two-tile warps at
// 238-245 registers, 8 warps per SM; here each warp streams its MT tiles' super-blocks with cp.async into its own
// shared-memory ring of STAGES super-blocks and holds only the current one in registers.
template <int WARPS, int STAGES, int NT, int MT>
__global__ void __launch_bounds__(WARPS * 32)
w4e(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * WARPS + warp) * MT;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    if (tile0 * 16 >= N) return;
    const int tiles = (N + 15) / 16;
    uint8_t *ring = smem + warp * STAGES * MT * T4;
    const int ns = sb1 - sb0;
    auto issue = [&](int k) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const uint8_t *src = wr + ((size_t)min(tile0 + m, tiles - 1) * nb + sb0 + k) * T4;
            uint8_t *dst = ring + ((k % STAGES) * MT + m) * T4;
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int piece = lane + 32 * i;
                if (piece < T4 / 16) cp_async16(dst + piece * 16, src + piece * 16);
            }
        }
    };
    auto xload = [&](int sb, int j, uint4 (&xv)[NT]) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
            xv[nt] = __ldg(reinterpret_cast<const uint4 *>(xh + (size_t)(nt * 8 + g) * K + (size_t)sb * 256 + 32 * j + 8 * c));
    };
    const half2 h1024 = __half2half2(__ushort_as_half(0x6400));
    // 0x64 in a register ptxas cannot fold (blockDim.y is 1 at run time, unknown when compiling): a PRMT takes one
    // immediate, and with 0x64 as that immediate ptxas rebuilt the selector 0x4140 / 0x4342 in a register before
    // every PRMT (SASS of v9: about 300 IMAD.U32 of 2,400 instructions); this way the selector is the immediate
    const uint32_t k64 = 0x63u + blockDim.y;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
#pragma unroll
    for (int k = 0; k < STAGES - 1; k++) {
        if (k < ns) issue(k);
        cp_commit();
    }
    for (int k = 0; k < ns; k++) {
        const int sb = sb0 + k;
        uint4 xv[2][NT];
        xload(sb, 0, xv[0]);  // x first: its latency overlaps the wait for the weights
        cp_wait<STAGES - 2>();
        __syncwarp();
        if (k + STAGES - 1 < ns) issue(k + STAGES - 1);
        cp_commit();
        const uint8_t *base = ring + (k % STAGES) * MT * T4;
        uint32_t D[MT], NM[MT], sw[MT][2][2], mw[MT][2][2];
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const uint8_t *p = base + m * T4;
            const uint4 h0 = *reinterpret_cast<const uint4 *>(p + 2048 + g * 16);
            const uint4 h1 = *reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16);
            D[m] = __byte_perm(h0.x, h1.x, 0x5410);
            NM[m] = h2_bits(__hneg2(bits_h2(__byte_perm(h0.x, h1.x, 0x7632))));
            const uint4 hd[2] = {h0, h1};
#pragma unroll
            for (int h = 0; h < 2; h++) {
                sw[m][h][0] = hd[h].y & 0x3F3F3F3Fu;
                mw[m][h][0] = hd[h].z & 0x3F3F3F3Fu;
                sw[m][h][1] = (hd[h].w & 0x0F0F0F0Fu) | ((hd[h].y >> 2) & 0x30303030u);
                mw[m][h][1] = ((hd[h].w >> 4) & 0x0F0F0F0Fu) | ((hd[h].z >> 2) & 0x30303030u);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j < 7) xload(sb, j + 1, xv[(j + 1) & 1]);
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                // piece i of the interleaved tile holds words 4i..4i+3 of every lane; word 2j + h is in piece j / 2
                const uint4 qv = *reinterpret_cast<const uint4 *>(base + m * T4 + ((j >> 1) * 32 + lane) * 16);
                const uint32_t pick = (uint32_t)(j & 3) | ((uint32_t)(4 + (j & 3)) << 4);
                const half2 sc = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(sw[m][0][j >> 2], sw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(D[m]));
                const half2 mn = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(mw[m][0][j >> 2], mw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(NM[m]));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const half2 s2 = h ? __high2half2(sc) : __low2half2(sc);
                    const half2 m2 = h ? __high2half2(mn) : __low2half2(mn);
                    const int w = (2 * j + h) & 3;
                    const uint32_t wd = w == 0 ? qv.x : w == 1 ? qv.y : w == 2 ? qv.z : qv.w;
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = (t ? wd >> 4 : wd) & 0x0F0F0F0Fu;
                        const uint32_t p0 = __byte_perm(v, k64, 0x4140), p1 = __byte_perm(v, k64, 0x4342);
                        a[m][t][h] = h2_bits(__hfma2(__hsub2(bits_h2(p0), h1024), s2, m2));
                        a[m][t][2 + h] = h2_bits(__hfma2(__hsub2(bits_h2(p1), h1024), s2, m2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint4 bx = xv[j & 1][nt];
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    mma16816(acc[m][nt], a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], bx.x, bx.y);
                    mma16816(acc[m][nt], a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], bx.z, bx.w);
                }
            }
        }
    }
    cp_wait<0>();
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                const int tok = nt * 8 + 2 * c + (i % 2);
                if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
            }
}

template <int WARPS, int STAGES, int NT, int MT>
int launch_e(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
             cudaStream_t s) {
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    const int smem = WARPS * STAGES * MT * T4;
    auto kern = w4e<WARPS, STAGES, NT, MT>;
    static bool attr = false;
    if (!attr) {
        if (smem > 48 * 1024 && cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr = true;
    }
    dim3 grid((((N + 15) / 16 + MT - 1) / MT + WARPS - 1) / WARPS, splits);
    kern<<<grid, WARPS * 32, smem, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// RF1 (2026-11-24, opt-in IRONMULE_KG_RF1=1): a grid K-split kernel reduces its own partial sums. Every block writes
// its partials as before; the last block of a row range to finish (an atomic count per range, after a fence, nothing
// waits) adds them in reduce_split's order (0 + p0 + p1 + ...) into y and resets the count for the next launch.
__device__ __forceinline__ void split_finish(const float *part, float *__restrict__ y, int *__restrict__ counters, int r0,
                                             int rows, int N, int B) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&counters[blockIdx.x], 1) == (int)gridDim.y - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int nr = min(rows, N - r0);
    const size_t n = (size_t)B * N;
    for (int i = threadIdx.x; i < nr * B; i += blockDim.x) {
        const int tok = i / nr, row = r0 + i % nr;
        const size_t idx = (size_t)tok * N + row;
        float sum = 0.f;
        for (int k = 0; k < (int)gridDim.y; k++) sum += __ldcg(part + (size_t)k * n + idx);
        y[idx] = sum;
    }
    if (threadIdx.x == 0) counters[blockIdx.x] = 0;
}

// v11 (Q4_K, 9-32 tokens, long K): v9 with the x of each super-block staged once per block in shared memory
// (cp.async, double buffered) and read by all WARPS x MT tiles of the block. A diagnostic with x folded onto two
// super-blocks (L1-resident) took the 25600 x 5120 down matrix at 32 tokens from 99 to 87 us: v9 re-read the 1.6 MB of
// x from L2 for every one or two tiles.
template <int WARPS, int NT, int MT, int FR = 0>  // FR: RF1, the K-split partials reduced in the kernel
__global__ void __launch_bounds__(WARPS * 32)
w4f(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split, float *__restrict__ y, int *__restrict__ counters) {
    constexpr int ROWS = NT * 8, XROW = 256 + 32;  // halves per staged token row (64-byte pad: conflict-free)
    __shared__ __align__(16) half xbuf[2][ROWS * XROW];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * WARPS + warp) * MT;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile0 * 16 < N;
    const int tiles = (N + 15) / 16;
    const int ns = sb1 - sb0;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct Buf { uint4 q[MT][4], h[MT][2]; };
    auto fetch = [&](int sb, Buf &b) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const uint8_t *p = wr + ((size_t)t * nb + sb) * T4;
#pragma unroll
            for (int i = 0; i < 4; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
            b.h[m][0] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + g * 16));
            b.h[m][1] = __ldg(reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16));
        }
    };
    auto stage = [&](int buf, int sb) {  // ROWS x 256 halves = ROWS x 32 pieces of 16 bytes
        for (int i = threadIdx.x; i < ROWS * 32; i += WARPS * 32) {
            const int n = i / 32, piece = i % 32;
            cp_async16(&xbuf[buf][n * XROW + piece * 8], xh + (size_t)n * K + (size_t)sb * 256 + piece * 8);
        }
    };
    const half2 h1024 = __half2half2(__ushort_as_half(0x6400));
    const uint32_t k64 = 0x63u + blockDim.y;  // see w4d
    auto compute = [&](const Buf &b, int buf) {
        uint32_t D[MT], NM[MT], sw[MT][2][2], mw[MT][2][2];
#pragma unroll
        for (int m = 0; m < MT; m++) {
            D[m] = __byte_perm(b.h[m][0].x, b.h[m][1].x, 0x5410);
            NM[m] = h2_bits(__hneg2(bits_h2(__byte_perm(b.h[m][0].x, b.h[m][1].x, 0x7632))));
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint4 hv = b.h[m][h];
                sw[m][h][0] = hv.y & 0x3F3F3F3Fu;
                mw[m][h][0] = hv.z & 0x3F3F3F3Fu;
                sw[m][h][1] = (hv.w & 0x0F0F0F0Fu) | ((hv.y >> 2) & 0x30303030u);
                mw[m][h][1] = ((hv.w >> 4) & 0x0F0F0F0Fu) | ((hv.z >> 2) & 0x30303030u);
            }
        }
        const half *xb = xbuf[buf];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint32_t pick = (uint32_t)(j & 3) | ((uint32_t)(4 + (j & 3)) << 4);
                const half2 sc = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(sw[m][0][j >> 2], sw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(D[m]));
                const half2 mn = __hmul2(__hsub2(bits_h2(__byte_perm(__byte_perm(mw[m][0][j >> 2], mw[m][1][j >> 2], pick), k64, 0x4140)), h1024), bits_h2(NM[m]));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const half2 s2 = h ? __high2half2(sc) : __low2half2(sc);
                    const half2 m2 = h ? __high2half2(mn) : __low2half2(mn);
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = (t ? wd >> 4 : wd) & 0x0F0F0F0Fu;
                        const uint32_t p0 = __byte_perm(v, k64, 0x4140), p1 = __byte_perm(v, k64, 0x4342);
                        a[m][t][h] = h2_bits(__hfma2(__hsub2(bits_h2(p0), h1024), s2, m2));
                        a[m][t][2 + h] = h2_bits(__hfma2(__hsub2(bits_h2(p1), h1024), s2, m2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint4 bx = *reinterpret_cast<const uint4 *>(xb + (nt * 8 + g) * XROW + 32 * j + 8 * c);
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    mma16816(acc[m][nt], a[m][0][0], a[m][0][1], a[m][0][2], a[m][0][3], bx.x, bx.y);
                    mma16816(acc[m][nt], a[m][1][0], a[m][1][1], a[m][1][2], a[m][1][3], bx.z, bx.w);
                }
            }
        }
    };
    // every warp takes part in staging and barriers, live or not
    Buf ba, bb;
    if (ns > 0) {
        stage(0, sb0);
        cp_commit();
        fetch(sb0, ba);
    }
    for (int k = 0; k < ns; k += 2) {
        if (k + 1 < ns) stage(1, sb0 + k + 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (k + 1 < ns) fetch(sb0 + k + 1, bb);
        if (live) compute(ba, 0);
        __syncthreads();
        if (k + 1 >= ns) break;
        if (k + 2 < ns) stage(0, sb0 + k + 2);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (k + 2 < ns) fetch(sb0 + k + 2, ba);
        if (live) compute(bb, 1);
        __syncthreads();
    }
    cp_wait<0>();
    if constexpr (FR == 0)
        if (!live) return;
    float *o = out + (size_t)blockIdx.y * B * N;
    if (FR == 0 || live) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                    const int tok = nt * 8 + 2 * c + (i % 2);
                    if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
                }
    }
    if constexpr (FR != 0) split_finish(out, y, counters, blockIdx.x * WARPS * MT * 16, WARPS * MT * 16, N, B);
}

template <int WARPS, int NT, int MT, int FR = 0>  // FR: RF1, the K-split partials reduced in the kernel
__global__ void __launch_bounds__(WARPS * 32)
w4f6(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split, float *__restrict__ y, int *__restrict__ counters) {
    constexpr int ROWS = NT * 8, XROW = 256 + 32;  // halves per staged token row (64-byte pad: conflict-free)
    __shared__ __align__(16) half xbuf[2][ROWS * XROW];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * WARPS + warp) * MT;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile0 * 16 < N;
    const int tiles = (N + 15) / 16;
    const int ns = sb1 - sb0;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct Buf { uint4 q[MT][6], h[MT][4]; };
    auto fetch = [&](int sb, Buf &b) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const uint8_t *p = wr + ((size_t)t * nb + sb) * T6;
#pragma unroll
            for (int i = 0; i < 6; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                b.h[m][2 * h] = __ldg(reinterpret_cast<const uint4 *>(p + 3072 + (g + 8 * h) * 32));
                b.h[m][2 * h + 1] = __ldg(reinterpret_cast<const uint4 *>(p + 3072 + (g + 8 * h) * 32 + 16));
            }
        }
    };
    auto stage = [&](int buf, int sb) {  // ROWS x 256 halves = ROWS x 32 pieces of 16 bytes
        for (int i = threadIdx.x; i < ROWS * 32; i += WARPS * 32) {
            const int n = i / 32, piece = i % 32;
            cp_async16(&xbuf[buf][n * XROW + piece * 8], xh + (size_t)n * K + (size_t)sb * 256 + piece * 8);
        }
    };
    const half2 h1056 = __float2half2_rn(1056.f), h1152 = __float2half2_rn(1152.f);
    const uint32_t k64 = 0x63u + blockDim.y;  // see w4d
    auto compute = [&](const Buf &b, int buf) {
        uint32_t D[MT];
#pragma unroll
        for (int m = 0; m < MT; m++) D[m] = __byte_perm(b.h[m][1].x, b.h[m][3].x, 0x5410);
        const half *xb = xbuf[buf];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint32_t hw = wget(b.q[m], 16 + j);
                half2 sc[2];
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const int si = 8 * (j / 4) + t + 2 * (j % 4);
                    const uint4 s0 = b.h[m][0], s1 = b.h[m][2];
                    const uint32_t w0 = si < 4 ? s0.x : si < 8 ? s0.y : si < 12 ? s0.z : s0.w;
                    const uint32_t w1 = si < 4 ? s1.x : si < 8 ? s1.y : si < 12 ? s1.z : s1.w;
                    const uint32_t pair = __byte_perm(w0, w1, (uint32_t)(si & 3) | ((uint32_t)(4 + (si & 3)) << 4)) ^ 0x8080u;
                    sc[t] = __hmul2(__hsub2(bits_h2(__byte_perm(pair, k64, 0x4140)), h1152), bits_h2(D[m]));
                }
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = ((t ? wd >> 4 : wd) & 0x0F0F0F0Fu) |
                                           (((hw >> (2 * (2 * h + t))) & 0x03030303u) << 4);
                        const half2 s2 = h ? __high2half2(sc[t]) : __low2half2(sc[t]);
                        a[m][t][h] = h2_bits(__hmul2(__hsub2(bits_h2(__byte_perm(v, k64, 0x4140)), h1056), s2));
                        a[m][t][2 + h] = h2_bits(__hmul2(__hsub2(bits_h2(__byte_perm(v, k64, 0x4342)), h1056), s2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const uint2 bx = *reinterpret_cast<const uint2 *>(xb + (nt * 8 + g) * XROW + 32 * j + 16 * t + 4 * c);
#pragma unroll
                    for (int m = 0; m < MT; m++)
                        mma16816(acc[m][nt], a[m][t][0], a[m][t][1], a[m][t][2], a[m][t][3], bx.x, bx.y);
                }
        }
    };
    // every warp takes part in staging and barriers, live or not
    Buf ba, bb;
    if (ns > 0) {
        stage(0, sb0);
        cp_commit();
        fetch(sb0, ba);
    }
    for (int k = 0; k < ns; k += 2) {
        if (k + 1 < ns) stage(1, sb0 + k + 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (k + 1 < ns) fetch(sb0 + k + 1, bb);
        if (live) compute(ba, 0);
        __syncthreads();
        if (k + 1 >= ns) break;
        if (k + 2 < ns) stage(0, sb0 + k + 2);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        if (k + 2 < ns) fetch(sb0 + k + 2, ba);
        if (live) compute(bb, 1);
        __syncthreads();
    }
    cp_wait<0>();
    if constexpr (FR == 0)
        if (!live) return;
    float *o = out + (size_t)blockIdx.y * B * N;
    if (FR == 0 || live) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                    const int tok = nt * 8 + 2 * c + (i % 2);
                    if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
                }
    }
    if constexpr (FR != 0) split_finish(out, y, counters, blockIdx.x * WARPS * MT * 16, WARPS * MT * 16, N, B);
}

template <int TYPE, int WARPS, int NT, int MT>
int launch_f(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
             cudaStream_t s) {
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    dim3 grid((((N + 15) / 16 + MT - 1) / MT + WARPS - 1) / WARPS, splits);
    int *cnt = rf1_counters;  // RF1: set by run_repacked when on and room for this grid
    if (splits > 1 && cnt != nullptr && grid.x <= RF1_COUNTERS) {
        if constexpr (TYPE == TYPE_Q4_K)
            w4f<WARPS, NT, MT, 1><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, part, N, K, B, per, y, cnt);
        else
            w4f6<WARPS, NT, MT, 1><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, part, N, K, B, per, y, cnt);
        return cudaGetLastError() == cudaSuccess ? 0 : 3;
    }
    if constexpr (TYPE == TYPE_Q4_K)
        w4f<WARPS, NT, MT><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per, nullptr, nullptr);
    else
        w4f6<WARPS, NT, MT><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per, nullptr,
                                                        nullptr);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// v9 for Q6_K: the same structure over w4r's Q6_K tile layout (16 nibble words, 8 two-bit words, 16 int8 scales)
template <int WARPS, int KS, int NT, int MT>
__global__ void __launch_bounds__(WARPS * 32)
w4d6(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
     float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    constexpr int TILES = WARPS / KS;
    __shared__ float red[KS > 1 ? WARPS : 1][KS > 1 ? MT * NT * 4 * 32 : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256;
    const int tile0 = (blockIdx.x * TILES + warp / KS) * MT, part = warp % KS;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    const bool live = tile0 * 16 < N;
    const int tiles = (N + 15) / 16;
    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[m][nt][i] = 0.f;
    struct Buf { uint4 q[MT][6], h[MT][4]; };
    auto fetch = [&](int sb, Buf &b) {
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t = min(tile0 + m, tiles - 1);
            const uint8_t *p = wr + ((size_t)t * nb + sb) * T6;
#pragma unroll
            for (int i = 0; i < 6; i++) b.q[m][i] = ldg_stream(p + (i * 32 + lane) * 16);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                b.h[m][2 * h] = __ldg(reinterpret_cast<const uint4 *>(p + 3072 + (g + 8 * h) * 32));
                b.h[m][2 * h + 1] = __ldg(reinterpret_cast<const uint4 *>(p + 3072 + (g + 8 * h) * 32 + 16));
            }
        }
    };
    auto xload = [&](int sb, int j, uint2 (&xv)[NT][2]) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int t = 0; t < 2; t++)
                xv[nt][t] = __ldg(reinterpret_cast<const uint2 *>(xh + (size_t)(nt * 8 + g) * K + (size_t)sb * 256 + 32 * j + 16 * t + 4 * c));
    };
    const half2 h1056 = __float2half2_rn(1056.f), h1152 = __float2half2_rn(1152.f);
    auto compute = [&](const Buf &b, int sb) {
        uint32_t D[MT];  // half2 {d of row g, d of row g + 8}
#pragma unroll
        for (int m = 0; m < MT; m++) D[m] = __byte_perm(b.h[m][1].x, b.h[m][3].x, 0x5410);
        uint2 xv[2][NT][2];
        xload(sb, 0, xv[0]);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if (j < 7) xload(sb, j + 1, xv[(j + 1) & 1]);
            uint32_t a[MT][2][4];
#pragma unroll
            for (int m = 0; m < MT; m++) {
                const uint32_t hw = wget(b.q[m], 16 + j);
                half2 sc[2];  // per t: {scale of row g, of row g + 8}, int8 scales by the 1024 + 128 + s trick (exact)
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const int si = 8 * (j / 4) + t + 2 * (j % 4);
                    const uint4 s0 = b.h[m][0], s1 = b.h[m][2];
                    const uint32_t w0 = si < 4 ? s0.x : si < 8 ? s0.y : si < 12 ? s0.z : s0.w;
                    const uint32_t w1 = si < 4 ? s1.x : si < 8 ? s1.y : si < 12 ? s1.z : s1.w;
                    const uint32_t pair = __byte_perm(w0, w1, (uint32_t)(si & 3) | ((uint32_t)(4 + (si & 3)) << 4)) ^ 0x8080u;
                    sc[t] = __hmul2(__hsub2(bits_h2(__byte_perm(pair, 0x64u, 0x4140)), h1152), bits_h2(D[m]));
                }
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t wd = wget(b.q[m], 2 * j + h);
#pragma unroll
                    for (int t = 0; t < 2; t++) {
                        const uint32_t v = ((t ? wd >> 4 : wd) & 0x0F0F0F0Fu) |
                                           (((hw >> (2 * (2 * h + t))) & 0x03030303u) << 4);
                        const half2 s2 = h ? __high2half2(sc[t]) : __low2half2(sc[t]);
                        a[m][t][h] = h2_bits(__hmul2(__hsub2(bits_h2(__byte_perm(v, 0x64u, 0x4140)), h1056), s2));
                        a[m][t][2 + h] = h2_bits(__hmul2(__hsub2(bits_h2(__byte_perm(v, 0x64u, 0x4342)), h1056), s2));
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    const uint2 bx = xv[j & 1][nt][t];
#pragma unroll
                    for (int m = 0; m < MT; m++)
                        mma16816(acc[m][nt], a[m][t][0], a[m][t][1], a[m][t][2], a[m][t][3], bx.x, bx.y);
                }
        }
    };
    if (live) {
        Buf ba, bb;
        int sb = sb0 + part;
        if (sb < sb1) fetch(sb, ba);
        for (; sb < sb1; sb += 2 * KS) {
            if (sb + KS < sb1) fetch(sb + KS, bb);
            compute(ba, sb);
            if (sb + KS >= sb1) break;
            if (sb + 2 * KS < sb1) fetch(sb + 2 * KS, ba);
            compute(bb, sb + KS);
        }
    }
    if constexpr (KS > 1) {
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) red[warp][((m * NT + nt) * 4 + i) * 32 + lane] = acc[m][nt][i];
        __syncthreads();
        if (part != 0) return;
#pragma unroll
        for (int m = 0; m < MT; m++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int p = 1; p < KS; p++) acc[m][nt][i] += red[warp + p][((m * NT + nt) * 4 + i) * 32 + lane];
    }
    if (!live) return;
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (tile0 + m) * 16 + g + 8 * (i / 2);
                const int tok = nt * 8 + 2 * c + (i % 2);
                if (row < N && tok < B) o[(size_t)tok * N + row] = acc[m][nt][i];
            }
}

template <int TYPE, int WARPS, int KS, int NT, int MT, int MINB = 1>
int launch_d(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
             cudaStream_t s) {
    constexpr int TILES = WARPS / KS;
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    dim3 grid((((N + 15) / 16 + MT - 1) / MT + TILES - 1) / TILES, splits);
#if defined(IRONMULE_W4_CARVE) || IRONMULE_W4_PIN > 0
    if constexpr (TYPE == TYPE_Q4_K) {  // DK1: an explicit carveout (and room for the pin)
        static bool once = (cudaFuncSetAttribute(w4d<WARPS, KS, NT, MT, MINB>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                 IRONMULE_W4_CARVE_PCT),
                            cudaFuncSetAttribute(w4d<WARPS, KS, NT, MT, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 IRONMULE_W4_PIN), true);
        (void)once;
    }
#endif
    if constexpr (TYPE == TYPE_Q4_K)
        w4d<WARPS, KS, NT, MT, MINB><<<grid, WARPS * 32, IRONMULE_W4_PIN, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B,
                                                                            per);
    else
        w4d6<WARPS, KS, NT, MT><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}


template <int WARPS, int KS, int NT, int MT, int MINB = 1>
int launch_dp(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
              cudaStream_t s) {
    constexpr int TILES = WARPS / KS;
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    dim3 grid((((N + 15) / 16 + MT - 1) / MT + TILES - 1) / TILES, splits);
    w4dp<WARPS, KS, NT, MT, MINB><<<grid, WARPS * 32, 0, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int WARPS, int KS, int NT, int MT>
int launch_dq(const Seg3 &sg, const half *xh, const float *xs, float *part, int K, int B, int ksplit, cudaStream_t s) {
    constexpr int TILES = WARPS / KS;
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    const int tiles = sg.tend[sg.nseg - 1];
    dim3 grid(((tiles + MT - 1) / MT + TILES - 1) / TILES, splits);
    w4dq<WARPS, KS, NT, MT><<<grid, WARPS * 32, 0, s>>>(sg, xh, xs, part, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * sg.ntot;
        reduce_split_seg<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, sg, B, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

// v8 (Q4_K, 1-8 tokens): each warp streams its tile's super-blocks with cp.async into its own shared-memory ring of
// STAGES super-blocks (2304 contiguous bytes each), so about three super-blocks per warp are in flight while it
// computes; registers hold only the current super-block and the x fragments, which leaves room for more warps
template <int WARPS, int STAGES, int FP>
__global__ void __launch_bounds__(WARPS * 32)
w4q(const uint8_t *__restrict__ wr, const half *__restrict__ xh, const float *__restrict__ xs,
    float *__restrict__ out, int N, int K, int B, int sb_per_split) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int nb = K / 256, rows8 = 8;
    const int tile = blockIdx.x * WARPS + warp;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nb, sb0 + sb_per_split);
    if (tile * 16 >= N) return;
    const uint8_t *base = wr + (size_t)tile * nb * T4;
    const float *xs32 = xs + (size_t)(K / 16) * rows8;
    uint8_t *ring = smem + warp * STAGES * T4;
    const int ns = sb1 - sb0;
    auto issue = [&](int k) {
        const uint8_t *src = base + (size_t)(sb0 + k) * T4;
        uint8_t *dst = ring + (k % STAGES) * T4;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const int piece = lane + 32 * i;
            if (piece < T4 / 16) cp_async16(dst + piece * 16, src + piece * 16);
        }
    };
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < STAGES - 1; k++) {
        if (k < ns) issue(k);
        cp_commit();
    }
    for (int k = 0; k < ns; k++) {
        const int sb = sb0 + k;
        uint4 xv[8];  // x first: its latency overlaps the wait for the weights
        float2 sv[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            if constexpr (FP) {
                fused_x(xs, K, B, g, (size_t)sb * 256 + 32 * j + 8 * c, c, xv[j], sv[j]);
            } else {
                xv[j] = g < B ? __ldg(reinterpret_cast<const uint4 *>(xh + (size_t)g * K + (size_t)sb * 256 + 32 * j + 8 * c))
                              : make_uint4(0, 0, 0, 0);
                sv[j] = __ldg(reinterpret_cast<const float2 *>(xs32 + ((size_t)sb * 8 + j) * rows8 + 2 * c));
            }
        }
        cp_wait<STAGES - 2>();
        __syncwarp();
        if (k + STAGES - 1 < ns) issue(k + STAGES - 1);
        cp_commit();
        const uint8_t *p = ring + (k % STAGES) * T4;
        uint4 q[4];
#pragma unroll
        for (int i = 0; i < 4; i++) q[i] = *reinterpret_cast<const uint4 *>(p + (i * 32 + lane) * 16);
        const uint4 h0 = *reinterpret_cast<const uint4 *>(p + 2048 + g * 16);
        const uint4 h1 = *reinterpret_cast<const uint4 *>(p + 2048 + (g + 8) * 16);
        float dd[2], dm[2];
        uint32_t sw[2][2], mw[2][2];
        const uint4 hd[2] = {h0, h1};
#pragma unroll
        for (int h = 0; h < 2; h++) {
            dd[h] = __half2float(__ushort_as_half((unsigned short)(hd[h].x & 0xFFFF)));
            dm[h] = __half2float(__ushort_as_half((unsigned short)(hd[h].x >> 16)));
            sw[h][0] = hd[h].y & 0x3F3F3F3Fu;
            mw[h][0] = hd[h].z & 0x3F3F3F3Fu;
            sw[h][1] = (hd[h].w & 0x0F0F0F0Fu) | ((hd[h].y >> 2) & 0x30303030u);
            mw[h][1] = ((hd[h].w >> 4) & 0x0F0F0F0Fu) | ((hd[h].z >> 2) & 0x30303030u);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float S[2], NB[2];
            uint32_t a[2][4];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int sh = 8 * (j & 3);
                S[h] = dd[h] * byte_f(sw[h][j >> 2], j & 3);
                NB[h] = -fmaf(1024.f, S[h], dm[h] * byte_f(mw[h][j >> 2], j & 3));
                const uint32_t wd = wget(q, 2 * j + h);
                const uint32_t v0 = wd & 0x0F0F0F0Fu, v1 = (wd >> 4) & 0x0F0F0F0Fu;
                a[0][h] = __byte_perm(v0, 0x64u, 0x4140);
                a[0][2 + h] = __byte_perm(v0, 0x64u, 0x4342);
                a[1][h] = __byte_perm(v1, 0x64u, 0x4140);
                a[1][2 + h] = __byte_perm(v1, 0x64u, 0x4342);
            }
            float tmp[4] = {0.f, 0.f, 0.f, 0.f};
            mma16816(tmp, a[0][0], a[0][1], a[0][2], a[0][3], xv[j].x, xv[j].y);
            mma16816(tmp, a[1][0], a[1][1], a[1][2], a[1][3], xv[j].z, xv[j].w);
            acc[0] = fmaf(S[0], tmp[0], fmaf(NB[0], sv[j].x, acc[0]));
            acc[1] = fmaf(S[0], tmp[1], fmaf(NB[0], sv[j].y, acc[1]));
            acc[2] = fmaf(S[1], tmp[2], fmaf(NB[1], sv[j].x, acc[2]));
            acc[3] = fmaf(S[1], tmp[3], fmaf(NB[1], sv[j].y, acc[3]));
        }
    }
    cp_wait<0>();
    float *o = out + (size_t)blockIdx.y * B * N;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = tile * 16 + g + 8 * (i / 2);
        const int tok = 2 * c + (i % 2);
        if (row < N && tok < B) o[(size_t)tok * N + row] = acc[i];
    }
}

template <int WARPS, int STAGES, int FP>
int launch_q(const uint8_t *wr, const half *xh, const float *xs, float *y, float *part, int N, int K, int B, int ksplit,
             cudaStream_t s) {
    const int nb = K / 256;
    const int per = (nb + ksplit - 1) / ksplit;
    const int splits = (nb + per - 1) / per;
    const int smem = WARPS * STAGES * T4;
    auto kern = w4q<WARPS, STAGES, FP>;
    static bool attr = false;
    if (!attr) {
        if (smem > 48 * 1024 && cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr = true;
    }
    dim3 grid(((N + 15) / 16 + WARPS - 1) / WARPS, splits);
    kern<<<grid, WARPS * 32, smem, s>>>(wr, xh, xs, splits == 1 ? y : part, N, K, B, per);
    if (splits > 1) {
        const size_t n = (size_t)B * N;
        reduce_split<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(part, y, n, splits);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

struct Variant {
    int type, warps, ks, nt;
    launch_fn fn;
    int fp = 0;   // 1: the kernel converts llama.cpp's float activations itself (no prep kernel)
    int mq1 = 0;  // 1: MQ1 (Marlin) on its own copy of the weights, not the tile-order copy
    int pre = 0;  // 1: PS (v12) on the tile-order copy with precomputed fp16 scales and minimums (T4P)
    int sums = 0; // 1: the kernel reads prep_x's 16- or 32-value sums of x, not only its fp16 copy (w4r, v7, v8)
};
// PD1: the variants that read prep_x's sums
constexpr Variant with_sums(Variant v) { v.sums = 1; return v; }

int mq1_stub(const uint8_t *, const half *, const float *, float *, float *, int, int, int, int, cudaStream_t) {
    return 1;
}
#define VMQ(N) Variant{TYPE_Q4_K, 8, 60000, N, mq1_stub, 0, 1}

#define V(T, W, S, N) with_sums(Variant{T, W, S, N, launch<T, W, S, N, 0>})
#define VD(T, W, S, N) with_sums(Variant{T, W, S + 10, N, launch<T, W, S, N, 1>})
#define VDN(T, W, S) VD(T, W, S, 1), VD(T, W, S, 2), VD(T, W, S, 3), VD(T, W, S, 4)
#define VN(T, W, S) V(T, W, S, 1), V(T, W, S, 2), V(T, W, S, 3), V(T, W, S, 4)
#define VT(T) VN(T, 4, 1), VN(T, 8, 1), VN(T, 4, 2), VN(T, 8, 2), VN(T, 16, 2), VN(T, 8, 4), VN(T, 16, 4), VN(T, 16, 1)
#define VP(W, S, N, P, M) with_sums(Variant{TYPE_Q4_K, W, S + 100 * (1 + P) + 1000 * M, N, launch_p<W, S, N, P, M, 0>})
#define VPF(W, S, N, P) Variant{TYPE_Q4_K, W, S + 100 * (1 + P) + 20000, N, launch_p<W, S, N, P, 1, 1>, 1}
#define VPN(W, S) VP(W, S, 1, 1, 1), VP(W, S, 2, 1, 1), VP(W, S, 1, 0, 1), VP(W, S, 2, 0, 1), VP(W, S, 1, 0, 2), \
                  VP(W, S, 2, 0, 2)
#define VQ(W, S) with_sums(Variant{TYPE_Q4_K, W, 5000 + S, 1, launch_q<W, S, 0>})
#define VQF(W, S) Variant{TYPE_Q4_K, W, 25000 + S, 1, launch_q<W, S, 1>, 1}
#define VW(W, S, N, M) Variant{TYPE_Q4_K, W, 30000 + 100 * M + S, N, launch_d<TYPE_Q4_K, W, S, N, M>}
#define VE(W, ST, N, M) Variant{TYPE_Q4_K, W, 40000 + 100 * M + ST, N, launch_e<W, ST, N, M>}
#define VEN(N) VE(4, 4, N, 1), VE(4, 3, N, 1), VE(8, 3, N, 1), VE(8, 4, N, 1), VE(2, 4, N, 1), VE(4, 3, N, 2), \
               VE(2, 4, N, 2), VE(4, 2, N, 2)
#define VF(W, N, M) Variant{TYPE_Q4_K, W, 50000 + 100 * M, N, launch_f<TYPE_Q4_K, W, N, M>}
#define VF6(W, N, M) Variant{TYPE_Q6_K, W, 50000 + 100 * M, N, launch_f<TYPE_Q6_K, W, N, M>}
// v9 with a minimum of B blocks per SM: fewer registers, more warps to hide the dequantize -> mma chains (2026-11-22)
#define VWP(W, S, N, M) Variant{TYPE_Q4_K, W, 80000 + 100 * M + S, N, launch_dp<W, S, N, M>, 0, 0, 1}
#define VWPN(N) VWP(2, 1, N, 2), VWP(4, 1, N, 2), VWP(8, 1, N, 2), VWP(4, 2, N, 2), VWP(8, 2, N, 2), VWP(8, 4, N, 2), \
                VWP(4, 1, N, 1), VWP(8, 2, N, 1)
// PS with a minimum of B blocks per SM (fewer registers), the occupancy VWB gives v9
#define VWPB(W, S, N, M, B) Variant{TYPE_Q4_K, W, 90000 + 100 * M + 10 * B + S, N, launch_dp<W, S, N, M, B>, 0, 0, 1}
#define VWB(W, S, N, M, B) Variant{TYPE_Q4_K, W, 70000 + 100 * M + 10 * B + S, N, launch_d<TYPE_Q4_K, W, S, N, M, B>}
#define VW6(W, S, N, M) Variant{TYPE_Q6_K, W, 30000 + 100 * M + S, N, launch_d<TYPE_Q6_K, W, S, N, M>}
#define VW6N(N) VW6(4, 1, N, 1), VW6(8, 1, N, 1), VW6(8, 2, N, 1), VW6(8, 4, N, 1), VW6(2, 1, N, 2), VW6(4, 1, N, 2), \
                VW6(4, 2, N, 2), VW6(8, 4, N, 2)
#define VWN(N) VW(2, 1, N, 2), VW(4, 1, N, 2), VW(8, 1, N, 2), VW(4, 2, N, 2), VW(8, 2, N, 2), VW(8, 4, N, 2), \
               VW(4, 1, N, 1), VW(8, 2, N, 1)
const Variant VARIANTS[] = {VT(TYPE_Q4_K), VT(TYPE_Q6_K),
                            VDN(TYPE_Q4_K, 4, 1), VDN(TYPE_Q4_K, 8, 1), VDN(TYPE_Q4_K, 8, 2), VDN(TYPE_Q4_K, 4, 2),
                            VDN(TYPE_Q6_K, 4, 1), VDN(TYPE_Q6_K, 8, 1), VDN(TYPE_Q6_K, 8, 2), VDN(TYPE_Q6_K, 4, 2), VPN(2, 1), VPN(4, 1), VPN(4, 2), VPN(8, 2), VPN(8, 4),
                            VPN(4, 4), VPN(2, 2), VPN(8, 1), VQ(2, 4), VQ(4, 4), VQ(4, 3), VQ(8, 3), VQ(2, 6), VQ(4, 5),
                            VQ(1, 8), VQ(2, 8), VQF(2, 4), VQF(4, 4), VQF(4, 3), VQF(2, 6), VQF(1, 8), VQF(2, 8),
                            VPF(2, 1, 1, 1), VPF(4, 1, 1, 1), VPF(4, 2, 1, 1), VPF(8, 2, 1, 1), VPF(2, 1, 1, 0),
                            VPF(4, 1, 1, 0), VPF(4, 2, 1, 0), VPF(2, 1, 2, 1), VPF(4, 1, 2, 1), VPF(4, 2, 2, 0),
                            VWN(2), VWN(3), VWN(4), VW6N(2), VW6N(3), VW6N(4), VEN(2), VEN(4),  // 3: 17-24 tokens
                            // few output rows (5120): K split inside the block, one tile per 4-16 warps, no reduce kernel
                            VW(4, 4, 2, 1), VW(8, 8, 2, 1), VW(16, 8, 2, 1), VW(16, 16, 2, 1),
                            VW(4, 4, 4, 1), VW(8, 8, 4, 1), VW(16, 8, 4, 1), VW(16, 16, 4, 1), VW(8, 4, 4, 1),
                            VW6(4, 4, 2, 1), VW6(8, 8, 2, 1), VW6(16, 16, 2, 1),
                            VW6(4, 4, 4, 1), VW6(8, 8, 4, 1), VW6(16, 16, 4, 1), VW6(8, 4, 4, 1),
                            // long K (down, 25600): x of 32 tokens is 1.6 MB, read from L2 once per tile group; three or
                            // four tiles per warp (weights through the shared-memory ring) cut that L2 traffic 3-4x
                            VE(4, 2, 4, 4), VE(8, 2, 4, 4), VE(4, 3, 4, 4), VE(4, 2, 4, 3), VE(8, 2, 4, 3), VE(4, 3, 4, 3),
                            VE(2, 3, 4, 4), VE(4, 2, 2, 4),
                            VF(4, 4, 1), VF(8, 4, 1), VF(16, 4, 1), VF(4, 4, 2), VF(8, 4, 2), VF(2, 4, 2),
                            VF(4, 2, 1), VF(8, 2, 1), VF(4, 2, 2), VF(8, 2, 2),
                            VF6(4, 4, 1), VF6(8, 4, 1), VF6(2, 4, 1), VF6(4, 4, 2), VF6(4, 2, 1), VF6(8, 2, 1),
                            VMQ(1), VMQ(2), VMQ(3), VMQ(4),
                            VWB(2, 1, 4, 2, 5), VWB(2, 1, 4, 2, 6), VWB(2, 1, 4, 2, 8), VWB(4, 1, 4, 2, 3), VWB(4, 1, 4, 2, 4),
                            VWB(8, 4, 4, 2, 2), VWB(4, 1, 4, 1, 4), VWB(2, 1, 2, 2, 6), VWB(4, 1, 2, 1, 4),
                            VWPN(2), VWPN(3), VWPN(4),  // PS (v12), the configurations of VWN
                            VWPB(2, 1, 4, 2, 5), VWPB(2, 1, 4, 2, 6), VWPB(4, 1, 4, 2, 3), VWPB(4, 1, 4, 2, 4),
                            VWPB(8, 4, 4, 2, 2), VWPB(4, 1, 4, 1, 4), VWPB(4, 1, 3, 1, 4), VWPB(4, 1, 2, 1, 4)};
constexpr int NV = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

struct Scratch {
    uint8_t *x = nullptr;  // activations as half, then their 16-value sums
    int *cnt = nullptr;    // RF1: RF1_COUNTERS zeroed counts, made outside a capture
    float *part = nullptr;
    size_t xcap = 0, pcap = 0;
};
// SG1 (2026-11-23): one scratch per CUDA stream, never freed. Each llama.cpp context computes on its own stream and keeps
// its own captured CUDA graph, which holds the scratch's address. With one scratch per device, a second engine in the
// process (SD1's draft model) replayed a graph whose scratch the served model had since grown, i.e. freed and moved
// (drafts accepted 2-4 % under multi-row graphs against 0.64-0.70 without them, A100 pod 20).
std::map<cudaStream_t, Scratch> scratches;
std::vector<void *> retired;  // grown-out buffers stay allocated: an earlier captured graph may still read them
std::mutex lock_;
std::map<std::tuple<int, long, long, int>, std::pair<int, int>> choice;  // (type, N, K, B) -> (variant, ksplit)
std::map<std::tuple<int, const void *, int>, uint8_t *> repacked;  // (device, weights, PS) -> tile-order copy
size_t repacked_total = 0;
uint32_t mask4 = 0xFFFFFFFFu, mask6 = 0xFFFFFFFFu;
std::string table_json = "{\"entries\": []}";

int grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*p) retired.push_back(*p);  // SG1: not freed
    *p = nullptr;
    *cap = 0;
    if (cudaMalloc(p, need) != cudaSuccess) return 1;
    *cap = need;
    return 0;
}

size_t repacked_bytes(int type, long N, long K, int pre = 0) {
    return (size_t)((N + 15) / 16) * (K / 256) * (type == TYPE_Q4_K ? (pre ? T4P : T4) : T6);
}

void launch_repack(int type, const uint8_t *w, uint8_t *dst, long N, long K, cudaStream_t s, int pre = 0) {
    const size_t threads = (size_t)((N + 15) / 16) * (K / 256) * 32;
    if (type == TYPE_Q4_K && pre)
        repack_q4k_pre<<<(unsigned)((threads + 255) / 256), 256, 0, s>>>(w, dst, (int)N, (int)(K / 256));
    else if (type == TYPE_Q4_K)
        repack_q4k<<<(unsigned)((threads + 255) / 256), 256, 0, s>>>(w, dst, (int)N, (int)(K / 256));
    else
        repack_q6k<<<(unsigned)((threads + 255) / 256), 256, 0, s>>>(w, dst, (int)N, (int)(K / 256));
}

// the tile-order copy of w, made the first time (call that first time outside CUDA graph capture)
uint8_t *repack(int dev, int type, const void *w, long N, long K, cudaStream_t s, int pre = 0) {
    std::lock_guard<std::mutex> guard(lock_);
    auto it = repacked.find({dev, w, pre});
    if (it != repacked.end()) return it->second;
    uint8_t *dst = nullptr;
    const size_t bytes = repacked_bytes(type, N, K, pre);
    if (cudaMalloc(&dst, bytes) != cudaSuccess) return nullptr;
    launch_repack(type, (const uint8_t *)w, dst, N, K, s, pre);
    if (cudaGetLastError() != cudaSuccess) { cudaFree(dst); return nullptr; }
    repacked[{dev, w, pre}] = dst;
    repacked_total += bytes;
    return dst;
}

// PK3: the src1 node and graph epoch of the matmul llama.cpp offers next (0 / nullptr without the hook)
const void *cur_node = nullptr;
unsigned long long cur_epoch = 0;
struct Prepared {
    const void *node = nullptr;
    unsigned long long epoch = 0;
    const float *x = nullptr;
    long B = 0, K = 0;
    int rows = 0;
    cudaStream_t s = nullptr;
    const void *buf = nullptr;  // the scratch it lies in (a grown scratch is a new buffer)
    bool sums = true;  // PD1: false when a producer kernel wrote the fp16 copy without prep_x's sums (not in the key)
    bool operator==(const Prepared &o) const {
        return node == o.node && epoch == o.epoch && x == o.x && B == o.B && K == o.K && rows == o.rows && s == o.s &&
               buf == o.buf;
    }
};
std::map<cudaStream_t, Prepared> prepared;  // SG1: per stream, as the scratch it describes

// RF1 switch: IRONMULE_KG_RF1=1, or ironmule_kg_rf1_set at run time (-1: the environment again)
std::atomic<int> rf1_override{-1};
bool ironmule_kg_rf1_on() {
    const int o = rf1_override.load();
    if (o >= 0) return o != 0;
    static const bool env = std::getenv("IRONMULE_KG_RF1") && std::atoi(std::getenv("IRONMULE_KG_RF1")) != 0;
    return env;
}

// PD1 check: prep_x's copy of a producer-prepared x, compared bit for bit (counts[0] differing halves, [1] compared)
bool pd1_check_on() {
    static const bool on = std::getenv("IRONMULE_KG_PD1_CHECK") && std::atoi(std::getenv("IRONMULE_KG_PD1_CHECK")) != 0;
    return on;
}
uint8_t *pd1_check_x = nullptr;
size_t pd1_check_cap = 0;
unsigned long long *pd1_counts = nullptr;

int run_repacked(const Variant &v, const uint8_t *wr, const float *x, float *y, long N, long K, long B, int ksplit,
                 int dev, cudaStream_t s) {
    (void)dev;
    Scratch *scp;
    {
        std::lock_guard<std::mutex> guard(lock_);
        scp = &scratches[s];  // std::map: the element stays where it is when others are added
    }
    Scratch &sc = *scp;
    if (v.fp) {  // the kernel reads x itself
        if (ksplit > 1) {
            std::lock_guard<std::mutex> guard(lock_);
            if (grow((void **)&sc.part, &sc.pcap, (size_t)ksplit * B * N * sizeof(float))) return 1;
        }
        return v.fn(wr, nullptr, x, y, sc.part, (int)N, (int)K, (int)B, ksplit, s);
    }
    const int rows = v.nt * 8;
    const size_t xbytes = ((size_t)rows * K * sizeof(half) + 255) / 256 * 256;
    {
        std::lock_guard<std::mutex> guard(lock_);
        if (grow((void **)&sc.x, &sc.xcap, xbytes + (size_t)K / 16 * rows * sizeof(float) * 3 / 2)) return 1;
        if (ksplit > 1 && grow((void **)&sc.part, &sc.pcap, (size_t)ksplit * B * N * sizeof(float))) return 1;
    }
    half *xh = reinterpret_cast<half *>(sc.x);
    float *xs = reinterpret_cast<float *>(sc.x + xbytes);
    // PK3: q, k and v (gate and up) read the same src1 node; within one graph its prepared copy is still in the scratch
    // when nothing else was prepared since. The key is llama.cpp's src1 node and graph epoch (ironmule_kg_src_node), x,
    // B, K and the row count, so another tensor at a reused address never matches.
    const Prepared want{cur_node, cur_epoch, x, B, K, rows, s, sc.x};
    Prepared *last;
    {
        std::lock_guard<std::mutex> guard(lock_);
        last = &prepared[s];
    }
    // PD1: a producer (llama.cpp's fused norm or SWIGLU, patch pd1) may have written the fp16 copy under this key
    // already, without the sums; a variant that reads the sums then still runs prep_x
    if (cur_node == nullptr || !(want == *last) || (v.sums && !last->sums)) {
        const int warps = rows * (int)(K / 32);
        prep_x<<<(warps + 7) / 8, 256, 0, s>>>(x, xh, xs, (int)B, (int)K, rows);
        *last = want;
    } else if (!last->sums && pd1_check_on()) {  // the producer's copy against prep_x's (never in a capture's growth)
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &capture);
        const size_t need = xbytes + (size_t)K / 16 * rows * sizeof(float) * 3 / 2;
        bool ready;
        {
            std::lock_guard<std::mutex> guard(lock_);
            if (capture == cudaStreamCaptureStatusNone) {
                grow((void **)&pd1_check_x, &pd1_check_cap, need);
                if (pd1_counts == nullptr && cudaMalloc(&pd1_counts, 2 * sizeof(unsigned long long)) == cudaSuccess)
                    cudaMemset(pd1_counts, 0, 2 * sizeof(unsigned long long));
            }
            ready = pd1_counts != nullptr && pd1_check_cap >= need;
        }
        if (ready) {
            half *ch = reinterpret_cast<half *>(pd1_check_x);
            const int warps = rows * (int)(K / 32);
            prep_x<<<(warps + 7) / 8, 256, 0, s>>>(x, ch, reinterpret_cast<float *>(pd1_check_x + xbytes), (int)B, (int)K,
                                                    rows);
            const size_t n = (size_t)B * K;
            pd1_compare<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(xh, ch, n, pd1_counts);
        }
    }
    // RF1: the launcher reduces a grid K-split inside the kernel where it can (v11) and these counts exist
    rf1_counters = nullptr;
    if (ksplit > 1 && ironmule_kg_rf1_on()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &capture);
        std::lock_guard<std::mutex> guard(lock_);
        if (sc.cnt == nullptr && capture == cudaStreamCaptureStatusNone &&
            cudaMalloc((void **)&sc.cnt, RF1_COUNTERS * sizeof(int)) == cudaSuccess)
            cudaMemsetAsync(sc.cnt, 0, RF1_COUNTERS * sizeof(int), s);
        rf1_counters = sc.cnt;
    }
    const int rc = v.fn(wr, xh, xs, y, sc.part, (int)N, (int)K, (int)B, ksplit, s);
    rf1_counters = nullptr;
    return rc;
}

}  // namespace

extern "C" {

// PK3: llama.cpp (patch pk3) names the src1 node and graph epoch of the matmul it offers next
void ironmule_kg_src_node(const void *node, unsigned long long epoch) {
    cur_node = node;
    cur_epoch = epoch;
}


int im_w4_variant_count() { return NV; }

// type, warps, ks, nt, 0, 0 of variant i (the tuner's view)
void im_w4_variant(int i, int *out) {
    const Variant &v = VARIANTS[i];
    out[0] = v.type; out[1] = v.warps; out[2] = v.ks; out[3] = v.nt; out[4] = v.pre; out[5] = 0;
}

// y[B][N] = w[N][K] (GGUF blocks at w; repacked once) x[B][K]^T with variant i: 0 on success
// MQ1: Marlin's copy of a Q4_K matrix, made the first time outside a CUDA graph capture
static std::map<std::pair<int, const void *>, mq1::Prepared> mq1_prepared;
static mq1::Scratch mq1_scratch[16];

// MQ1 is opt-in (IRONMULE_MQ1=1): on Qwen3-32B Q4_K_M it never beat W4R beyond noise, and its weight copies beside
// W4R's pushed the quality gate's tuning child out of an A100's 80 GB (2026-11-22)
static bool mq1_enabled() {
    static const bool on = getenv("IRONMULE_MQ1") != nullptr && atoi(getenv("IRONMULE_MQ1")) != 0;
    return on;
}

static int mq1_serve(int dev, const void *w, const float *x, float *y, long N, long K, long B, int ksplit, cudaStream_t s) {
    if (ksplit != 1 || !mq1_enabled()) return 1;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &capture);
    const bool capturing = capture != cudaStreamCaptureStatusNone;
    mq1::Prepared p;
    {
        std::lock_guard<std::mutex> guard(lock_);
        auto it = mq1_prepared.find({dev, w});
        if (it == mq1_prepared.end()) {
            if (capturing || mq1::prepare(w, N, K, dev, s, p) != 0) return 4;
            mq1_prepared[{dev, w}] = p;
        } else {
            p = it->second;
        }
    }
    return mq1::run(p, x, y, (int)B, dev, s, mq1_scratch[dev], capturing);
}

int im_w4_run(int i, const void *w, const float *x, float *y, long N, long K, long B, int ksplit, cudaStream_t s) {
    if (i < 0 || i >= NV || K % 256 != 0 || B < 1 || B > 32 || ksplit < 1) return 1;
    const Variant &v = VARIANTS[i];
    if ((B + 7) / 8 != v.nt) return 1;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 1;
    if (v.mq1) return mq1_serve(dev, w, x, y, N, K, B, ksplit, s);
    uint8_t *wr = repack(dev, v.type, w, N, K, s, v.pre);
    if (wr == nullptr) return 4;
    return run_repacked(v, wr, x, y, N, K, B, ksplit, dev, s);
}

// bytes the tile-order copies hold
size_t im_w4_repacked_bytes() {
    std::lock_guard<std::mutex> guard(lock_);
    return repacked_total;
}

void im_w4_set_choice(int type, long N, long K, int B, int variant, int ksplit) {
    std::lock_guard<std::mutex> guard(lock_);
    if (variant < 0) choice.erase({type, N, K, B});
    else choice[{type, N, K, B}] = {variant, ksplit};
}

void ironmule_kg_set_mask(uint32_t m) { mask4 = m; }
uint32_t ironmule_kg_get_mask() { return mask4; }
void ironmule_kg_set_mask6(uint32_t m) { mask6 = m; }
uint32_t ironmule_kg_get_mask6() { return mask6; }
const char *ironmule_kg_table() { return table_json.c_str(); }

// IRONMULE_W4_TABLE: the tuner's choices, one "type N K B variant ksplit" line each (ironmule/w4_kernels.py)
static void load_table_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char *path = getenv("IRONMULE_W4_TABLE");
        if (path == nullptr || *path == 0) return;
        FILE *f = fopen(path, "r");
        if (f == nullptr) return;
        int t, b, v, k;
        long n, kk;
        std::lock_guard<std::mutex> guard(lock_);
        while (fscanf(f, "%d %ld %ld %d %d %d", &t, &n, &kk, &b, &v, &k) == 6)
            if (v >= 0 && v < NV && k >= 1) choice[{t, n, kk, b}] = {v, k};
        fclose(f);
        std::string list = "[";
        for (size_t i = 0; i < choice.size(); i++) list += i ? ",0" : "0";
        table_json = std::string("{\"source\": \"") + path + "\", \"entries\": " + list + "]}";
    });
}

// PD1 (decode): instead of prep_x converting a matmul's float activations x [B][K] into the fp16 copy, the kernel that
// produces x (llama.cpp's fused add + RMS norm + mul, or SWIGLU; patch pd1) writes that copy itself, with prep_x's
// conversion (saturated to +-65000, rounded to nearest). This returns where: this stream's scratch, rows [0, B) of the
// [ceil(B / 8) * 8][K] layout, and records it under PK3's key (node, graph epoch, x, B, K, rows, stream, buffer), so
// the matmuls on that node skip prep_x; prep_x's 16- and 32-value sums are not written, so a variant that reads them
// still runs prep_x. nullptr (the producer writes no copy) when PD1 is off, for a batch or K the library takes no
// matmul for, and inside a CUDA graph capture when the scratch would have to grow. IRONMULE_KG_PD1=1 turns it on;
// ironmule_kg_pd1_set overrides that at run time (-1: the environment again), for A/B measurements in one process.
static std::atomic<int> pd1_override{-1};
void ironmule_kg_pd1_set(int on) { pd1_override.store(on); }
void ironmule_kg_rf1_set(int on) { rf1_override.store(on); }
int ironmule_kg_pd1_enabled() {
    const int o = pd1_override.load();
    if (o >= 0) return o != 0;
    static const bool env = [] {
        const char *v = std::getenv("IRONMULE_KG_PD1");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return env ? 1 : 0;
}
// PD1 check: halves compared and halves that differed so far (synchronises the device)
void ironmule_kg_pd1_counts(unsigned long long *out) {
    out[0] = out[1] = 0;
    if (pd1_counts == nullptr) return;
    cudaDeviceSynchronize();
    unsigned long long h[2] = {0, 0};
    cudaMemcpy(h, pd1_counts, sizeof(h), cudaMemcpyDeviceToHost);
    out[0] = h[1];
    out[1] = h[0];
}

void *ironmule_kg_prepare(const void *node, unsigned long long epoch, const float *x, int64_t K, int64_t B,
                          cudaStream_t s) {
    if (!ironmule_kg_pd1_enabled() || node == nullptr || x == nullptr || K % 256 != 0 || B < 1 || B > 32) return nullptr;
    if (!(((mask4 | mask6) >> (B - 1)) & 1u)) return nullptr;
    load_table_once();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &capture);
    const int rows = (int)((B + 7) / 8 * 8);
    const size_t xbytes = ((size_t)rows * K * sizeof(half) + 255) / 256 * 256;
    const size_t need = xbytes + (size_t)K / 16 * rows * sizeof(float) * 3 / 2;  // as run_repacked grows it
    std::lock_guard<std::mutex> guard(lock_);
    // a matmul with this K and batch that the table gives to this library, and none whose variant reads prep_x's sums
    // (prep_x would run anyway, and the producer's copy would be extra work: +0.04 ms per 8-row step, A100, pod 31)
    bool any = false;
    for (const auto &kv : choice)
        if (std::get<2>(kv.first) == (long)K && std::get<3>(kv.first) == (int)B && !VARIANTS[kv.second.first].fp) {
            if (VARIANTS[kv.second.first].sums) return nullptr;
            any = true;
        }
    if (!any) return nullptr;
    if (capture != cudaStreamCaptureStatusNone) {
        auto it = scratches.find(s);
        if (it == scratches.end() || it->second.xcap < need) return nullptr;
    }
    Scratch &sc = scratches[s];
    if (grow((void **)&sc.x, &sc.xcap, need)) return nullptr;
    prepared[s] = Prepared{node, epoch, x, (long)B, (long)K, rows, s, sc.x, false};
    return sc.x;
}

// llama.cpp's kg1 hook: 0 when this library computed y = w x, non-zero for llama.cpp's own kernels
int ironmule_kg_mul_mat(int type, const void *w, const float *x, float *y, int64_t N, int64_t K, int64_t B,
                        cudaStream_t s) {
    load_table_once();
    if (B > 32 && B <= 64 && std::getenv("IRONMULE_W4_SPLIT64")) {  // SD1 (experiment): 33-64 rows as two calls
        const int64_t first = 32;
        if (ironmule_kg_mul_mat(type, w, x, y, N, K, first, s) != 0) return 1;
        return ironmule_kg_mul_mat(type, w, x + first * K, y + first * N, N, K, B - first, s);
    }
    if ((type != TYPE_Q4_K && type != TYPE_Q6_K) || K % 256 != 0 || B < 1 || B > 32) return 1;
    if (!(((type == TYPE_Q6_K ? mask6 : mask4) >> (B - 1)) & 1u)) return 1;
    std::pair<int, int> pick;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &capture);
    int dev = 0;
    cudaGetDevice(&dev);
    {
        std::lock_guard<std::mutex> guard(lock_);
        auto it = choice.find({type, (long)N, (long)K, (int)B});
        if (it == choice.end()) return 1;
        pick = it->second;
        if (capture != cudaStreamCaptureStatusNone) {
            // inside a CUDA graph capture nothing may be allocated or repacked: only what an earlier call prepared
            const Variant &v = VARIANTS[pick.first];
            if (v.mq1) {  // MQ1 checks its own copy here and its scratch in mq1::run
                if (dev < 0 || dev >= 16 || mq1_prepared.find({dev, w}) == mq1_prepared.end()) return 1;
            } else {
                const size_t rows = (size_t)v.nt * 8;
                const size_t xbytes = (rows * K * sizeof(half) + 255) / 256 * 256 + (size_t)K / 16 * rows * sizeof(float) * 3 / 2;
                auto sit = scratches.find(s);  // SG1: this stream's scratch, prepared by an earlier uncaptured call
                const size_t xcap = sit == scratches.end() ? 0 : sit->second.xcap;
                const size_t pcap = sit == scratches.end() ? 0 : sit->second.pcap;
                if (dev < 0 || dev >= 16 || repacked.find({dev, w, v.pre}) == repacked.end() || (!v.fp && xcap < xbytes) ||
                    (pick.second > 1 && pcap < (size_t)pick.second * B * N * sizeof(float)))
                    return 1;
            }
        }
    }
    return im_w4_run(pick.first, w, x, y, N, K, B, pick.second, s);
}

// MQ1's measurement: the same protocol as the W4R variants below, over `copies` Marlin copies of w
static int mq1_bench(const void *w_host, size_t wbytes, long N, long K, const float *x_host, long B, int ksplit, int copies,
              int reps, double *us, float *y_host) {
    if (ksplit != 1 || N % 64 != 0 || !mq1_enabled()) return 1;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetLastError();
    uint8_t *w = nullptr;
    float *x = nullptr, *y = nullptr;
    if (cudaMalloc(&w, wbytes) != cudaSuccess) return 10;
    if (cudaMalloc(&x, sizeof(float) * B * K) != cudaSuccess || cudaMalloc(&y, sizeof(float) * B * N) != cudaSuccess) {
        cudaFree(w); cudaFree(x); return 11;
    }
    cudaMemcpy(w, w_host, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(x, x_host, sizeof(float) * B * K, cudaMemcpyHostToDevice);
    std::vector<mq1::Prepared> ps(copies);
    int rc = 0;
    for (int c = 0; c < copies && rc == 0; c++) rc = mq1::prepare(w, N, K, dev, 0, ps[c]) ? 14 : 0;
    mq1::Scratch sc;
    if (rc == 0) rc = mq1::run(ps[0], x, y, (int)B, dev, 0, sc, false);
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = 12;
    if (rc == 0) cudaMemcpy(y_host, y, sizeof(float) * B * N, cudaMemcpyDeviceToHost);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float total = 0.f;
    if (rc == 0) {
        for (int r = 0; r < 3; r++) rc |= mq1::run(ps[r % copies], x, y, (int)B, dev, 0, sc, false);
        cudaEventRecord(e0, 0);
        for (int r = 0; rc == 0 && r < reps; r++) rc |= mq1::run(ps[r % copies], x, y, (int)B, dev, 0, sc, false);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&total, e0, e1);
    }
    if (rc == 0 && cudaGetLastError() != cudaSuccess) rc = 13;
    if (rc == 0) *us = 1000.0 * total / reps;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    for (auto &p : ps) mq1::release(p);
    cudaFree(sc.a); cudaFree(sc.c); cudaFree(sc.c_tmp); cudaFree(sc.locks);
    cudaFree(w); cudaFree(x); cudaFree(y);
    return rc;
}

// Tuning: time variant i over `copies` repacked copies of w (beyond the L2 cache), `reps` back-to-back calls between
// two events as a served step issues them; returns y of the first call. 0 on success; *us = microseconds per call.
int im_w4_bench(int i, const void *w_host, size_t wbytes, long N, long K, const float *x_host, long B, int ksplit,
                int copies, int reps, double *us, float *y_host) {
    if (i < 0 || i >= NV) return 1;
    const Variant &v = VARIANTS[i];
    if ((B + 7) / 8 != v.nt) return 1;
    if (v.mq1) return mq1_bench(w_host, wbytes, N, K, x_host, B, ksplit, copies, reps, us, y_host);
    uint8_t *w = nullptr, *wr = nullptr;
    float *x = nullptr, *y = nullptr;
    const size_t rb = repacked_bytes(v.type, N, K, v.pre);
    int rc = 0;
    cudaGetLastError();  // a previous variant's failed launch must not fail this one
    if (cudaMalloc(&w, wbytes) != cudaSuccess) return 10;
    if (cudaMalloc(&wr, rb * copies) != cudaSuccess) { cudaFree(w); return 10; }
    if (cudaMalloc(&x, sizeof(float) * B * K) != cudaSuccess || cudaMalloc(&y, sizeof(float) * B * N) != cudaSuccess) {
        cudaFree(w); cudaFree(wr); cudaFree(x); return 11;
    }
    cudaMemcpy(w, w_host, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(x, x_host, sizeof(float) * B * K, cudaMemcpyHostToDevice);
    for (int c = 0; c < copies; c++) launch_repack(v.type, w, wr + c * rb, N, K, 0, v.pre);
    int dev = 0;
    cudaGetDevice(&dev);
    rc = run_repacked(v, wr, x, y, N, K, B, ksplit, dev, 0);
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = 12;
    if (rc == 0) cudaMemcpy(y_host, y, sizeof(float) * B * N, cudaMemcpyDeviceToHost);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float total = 0.f;
    if (rc == 0) {
        for (int r = 0; r < 3; r++) rc |= run_repacked(v, wr + (r % copies) * rb, x, y, N, K, B, ksplit, dev, 0);
        cudaEventRecord(e0, 0);
        for (int r = 0; rc == 0 && r < reps; r++) rc |= run_repacked(v, wr + (r % copies) * rb, x, y, N, K, B, ksplit, dev, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&total, e0, e1);
    }
    if (rc == 0 && cudaGetLastError() != cudaSuccess) rc = 13;
    if (rc == 0) *us = 1000.0 * total / reps;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(w); cudaFree(wr); cudaFree(x); cudaFree(y);
    return rc;
}


// ---------------------------------------------------------------------------------------------- v13 QKV (lab bench)
typedef int (*launch_dq_fn)(const Seg3 &, const half *, const float *, float *, int, int, int, cudaStream_t);
struct FusedVariant { int warps, ks, nt, mt; launch_dq_fn fn; };
#define FQ(W, S, N, M) FusedVariant{W, S, N, M, launch_dq<W, S, N, M>}
#define FQN(N) FQ(2, 1, N, 2), FQ(4, 1, N, 2), FQ(8, 1, N, 2), FQ(4, 2, N, 2), FQ(8, 2, N, 2), FQ(8, 4, N, 2), \
               FQ(4, 1, N, 1), FQ(8, 2, N, 1)
static const FusedVariant FVARIANTS[] = {FQN(1), FQN(2), FQN(3), FQN(4)};
static constexpr int NFV = sizeof(FVARIANTS) / sizeof(FVARIANTS[0]);

int im_w4_qkv_variant_count() { return NFV; }
void im_w4_qkv_variant(int i, int *out) {
    const FusedVariant &v = FVARIANTS[i];
    out[0] = v.warps; out[1] = v.ks; out[2] = v.nt; out[3] = v.mt;
}

// one prepared x, then the fused launch over the segments in sg (their weights repacked, Q4_K)
static int run_fused(const FusedVariant &v, const Seg3 &sg, const float *x, long K, long B, int ksplit, cudaStream_t s) {
    const int rows = v.nt * 8;
    const size_t xbytes = ((size_t)rows * K * sizeof(half) + 255) / 256 * 256;
    Scratch *scp;
    {
        std::lock_guard<std::mutex> guard(lock_);
        scp = &scratches[s];
        if (grow((void **)&scp->x, &scp->xcap, xbytes + (size_t)K / 16 * rows * sizeof(float) * 3 / 2)) return 1;
        if (ksplit > 1 && grow((void **)&scp->part, &scp->pcap, (size_t)ksplit * B * sg.ntot * sizeof(float))) return 1;
    }
    half *xh = reinterpret_cast<half *>(scp->x);
    float *xs = reinterpret_cast<float *>(scp->x + xbytes);
    const int warps = rows * (int)(K / 32);
    prep_x<<<(warps + 7) / 8, 256, 0, s>>>(x, xh, xs, (int)B, (int)K, rows);
    return v.fn(sg, xh, xs, scp->part, (int)K, (int)B, ksplit, s);
}

// QKV bench: the segments' separate best variants back to back on one stream, sharing one prepared x as PK3 does in
// a served step, against the fused variant fv; weights in `copies` repacked copies (beyond the L2 cache), two rounds
// alternating, the best round of each. y_sep / y_fused: [segment][B][N_s] of the first call. 0 on success.
int im_w4_qkv_bench(int nseg, const void *const *w_host, const size_t *wbytes, const long *N, long K, const float *x_host,
                    long B, const int *sep_variant, const int *sep_ksplit, int fv, int fks, int copies, int reps,
                    double *us_sep, double *us_fused, float *y_sep, float *y_fused) {
    if (nseg < 1 || nseg > 3 || fv < 0 || fv >= NFV || (B + 7) / 8 != FVARIANTS[fv].nt || K % 256) return 1;
    for (int i = 0; i < nseg; i++) {
        const Variant &v = VARIANTS[sep_variant[i]];
        if (v.type != TYPE_Q4_K || v.pre || v.mq1 || (B + 7) / 8 != v.nt || N[i] % 16) return 1;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetLastError();
    std::vector<uint8_t *> wr((size_t)nseg * copies, nullptr);
    std::vector<float *> y(nseg, nullptr);
    float *x = nullptr;
    long ntot = 0;
    int rc = 0;
    if (cudaMalloc(&x, sizeof(float) * B * K) != cudaSuccess) return 10;
    cudaMemcpy(x, x_host, sizeof(float) * B * K, cudaMemcpyHostToDevice);
    for (int i = 0; i < nseg && rc == 0; i++) {
        uint8_t *w = nullptr;
        const size_t rb = repacked_bytes(TYPE_Q4_K, N[i], K);
        if (cudaMalloc(&w, wbytes[i]) != cudaSuccess || cudaMalloc(&y[i], sizeof(float) * B * N[i]) != cudaSuccess) { rc = 10; break; }
        cudaMemcpy(w, w_host[i], wbytes[i], cudaMemcpyHostToDevice);
        for (int c = 0; c < copies; c++) {
            if (cudaMalloc(&wr[(size_t)c * nseg + i], rb) != cudaSuccess) { rc = 10; break; }
            launch_repack(TYPE_Q4_K, (const uint8_t *)w, wr[(size_t)c * nseg + i], N[i], K, 0);
        }
        cudaFree(w);
        ntot += N[i];
    }
    auto seg = [&](int c) {
        Seg3 sg{};
        sg.nseg = nseg;
        sg.ntot = (int)ntot;
        int tiles = 0, before = 0;
        for (int i = 0; i < nseg; i++) {
            sg.wr[i] = wr[(size_t)c * nseg + i];
            sg.out[i] = y[i];
            tiles += (int)(N[i] / 16);
            sg.tend[i] = tiles;
            sg.n[i] = (int)N[i];
            sg.before[i] = before;
            before += (int)N[i];
        }
        return sg;
    };
    static int marker;  // PK3's key for the shared prepared x: one node per step, as llama.cpp names q's src1
    unsigned long long epoch = 1ull << 40;
    auto separate = [&](int c) {
        int r = 0;
        ironmule_kg_src_node(&marker, ++epoch);
        for (int i = 0; i < nseg; i++)
            r |= run_repacked(VARIANTS[sep_variant[i]], wr[(size_t)c * nseg + i], x, y[i], N[i], K, B, sep_ksplit[i], dev, 0);
        ironmule_kg_src_node(nullptr, 0);
        return r;
    };
    auto fused = [&](int c) { return run_fused(FVARIANTS[fv], seg(c), x, K, B, fks, 0); };
    auto copy_out = [&](float *host) {
        size_t off = 0;
        for (int i = 0; i < nseg; i++) {
            cudaMemcpy(host + off, y[i], sizeof(float) * B * N[i], cudaMemcpyDeviceToHost);
            off += (size_t)B * N[i];
        }
    };
    if (rc == 0) rc = separate(0);
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = 12;
    if (rc == 0) copy_out(y_sep);
    if (rc == 0) rc = fused(0);
    if (rc == 0 && cudaDeviceSynchronize() != cudaSuccess) rc = 12;
    if (rc == 0) copy_out(y_fused);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double best_sep = 1e30, best_fused = 1e30;
    for (int round = 0; rc == 0 && round < 2; round++) {
        for (int which = 0; rc == 0 && which < 2; which++) {
            for (int r = 0; r < 3 && rc == 0; r++) rc |= which ? fused(r % copies) : separate(r % copies);
            cudaEventRecord(e0, 0);
            for (int r = 0; r < reps && rc == 0; r++) rc |= which ? fused(r % copies) : separate(r % copies);
            cudaEventRecord(e1, 0);
            cudaEventSynchronize(e1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            double &best = which ? best_fused : best_sep;
            best = std::min(best, 1000.0 * ms / reps);
        }
    }
    if (rc == 0 && cudaGetLastError() != cudaSuccess) rc = 13;
    if (rc == 0) { *us_sep = best_sep; *us_fused = best_fused; }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    for (auto p : wr) cudaFree(p);
    for (auto p : y) cudaFree(p);
    cudaFree(x);
    return rc;
}


// QK1: llama.cpp's group hook (native/ggml_patches/qk1). IRONMULE_KG_QKV_TABLE: lines "nseg ntot K B variant ksplit",
// the fused configuration measured faster than the separate ones (lab/w4bw/qkv_bench.py); a group without a line, a
// token count off the mask, or (inside a CUDA graph capture) one not yet repacked or prepared returns 1: llama.cpp then
// runs each matmul as before.
static std::map<std::tuple<int, long, long, int>, std::pair<int, int>> group_choice;
static void load_group_table_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char *path = getenv("IRONMULE_KG_QKV_TABLE");
        FILE *f = path ? fopen(path, "r") : nullptr;
        if (!f) return;
        int ns, b, v, ks;
        long nt, k;
        while (fscanf(f, "%d %ld %ld %d %d %d", &ns, &nt, &k, &b, &v, &ks) == 6)
            if (v >= 0 && v < NFV && FVARIANTS[v].nt == (b + 7) / 8) group_choice[{ns, nt, k, b}] = {v, ks};
        fclose(f);
    });
}

int ironmule_kg_mul_mat_group(int n, const int *types, const void *const *w, const float *x, float *const *y,
                              const int64_t *n_rows, int64_t K, int64_t B, cudaStream_t s) {
    load_group_table_once();
    if (n < 2 || n > 3 || B < 1 || B > 32 || K % 256 != 0 || !((mask4 >> (B - 1)) & 1u)) return 1;
    long ntot = 0;
    for (int i = 0; i < n; i++) {
        if (types[i] != TYPE_Q4_K || n_rows[i] % 16 != 0) return 1;
        ntot += (long)n_rows[i];
    }
    auto it = group_choice.find({n, ntot, (long)K, (int)B});
    if (it == group_choice.end()) return 1;
    const FusedVariant &v = FVARIANTS[it->second.first];
    const int ks = it->second.second;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return 1;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &capture);
    Seg3 sg{};
    sg.nseg = n;
    sg.ntot = (int)ntot;
    int tiles = 0, before = 0;
    for (int i = 0; i < n; i++) {
        uint8_t *wr = nullptr;
        if (capture != cudaStreamCaptureStatusNone) {  // inside a capture: only what an earlier call prepared
            std::lock_guard<std::mutex> guard(lock_);
            auto r = repacked.find({dev, w[i], 0});
            auto sc = scratches.find(s);
            const size_t rows = (size_t)v.nt * 8;
            const size_t xbytes = (rows * K * sizeof(half) + 255) / 256 * 256 + (size_t)K / 16 * rows * sizeof(float) * 3 / 2;
            if (r == repacked.end() || sc == scratches.end() || sc->second.xcap < xbytes ||
                (ks > 1 && sc->second.pcap < (size_t)ks * B * ntot * sizeof(float)))
                return 1;
            wr = r->second;
        } else {
            wr = repack(dev, TYPE_Q4_K, w[i], n_rows[i], K, s);
            if (wr == nullptr) return 4;
        }
        sg.wr[i] = wr;
        sg.out[i] = y[i];
        tiles += (int)(n_rows[i] / 16);
        sg.tend[i] = tiles;
        sg.n[i] = (int)n_rows[i];
        sg.before[i] = before;
        before += (int)n_rows[i];
    }
    return run_fused(v, sg, x, K, B, ks, s);
}

}  // extern "C"
