// MQ1 (IronMule, 2026-11-20): vLLM's Marlin kernel (v0.11.0, Apache-2.0, vendored unchanged in mq1/vllm/ apart
// from the torch includes) for llama.cpp's Q4_K matrices at 1-32 tokens.
//
// Why: a profile of vLLM serving the same Qwen3-32B wave on an A100 (docs/LAB.md, 2026-11-20) showed Marlin's 4-bit
// matmuls costing 12.0 ms per 32-row decode step against 11.6 ms per one-row step, while W4R slowed from about 13 to
// about 20 ms. Marlin keeps the weights in shared memory (a 4-stage cp.async pipeline), reuses each activation fragment
// for 64 output columns per warp and splits K across blocks so that every SM works.
//
// Format: Q4_K stores w = d*sc*q - dmin*m per 32-weight sub-block (q 0..15). Marlin's float-zero-point path (vLLM uses
// it for HQQ) computes w = q*s - z*s per group. MQ1 takes s = fp16(d*sc) (as W4R) and z = fp16(dmin*m / s), so q is
// exact and the minimum carries an fp16 rounding of z and of z*s; a sub-block with sc = 0 gets a tiny s that keeps
// z in the fp16 range. Marlin writes fp16 output; MQ1 converts it to llama.cpp's float. The tuner admits MQ1 per shape
// and token count only within ERROR_BOUND of the float64 reference and only where it is measured faster.
#pragma once

#include "torch_shim.h"
#include "vllm/kernel.h"
#include "vllm/marlin_template.h"
#include "vllm/gptq_marlin_repack.cuh"

#include <map>
#include <mutex>
#include <utility>

namespace mq1 {

using marlin::div_ceil;

constexpr int GROUP = 32;

// ------------------------------------------------------------------------------------------------- Q4_K -> GPTQ u4
// one thread per (32-weight group, output row n), n fastest: qw [K/8][N] u32 (8 consecutive k per word, nibble i =
// k 8w+i), s and z [K/32][N] half
__global__ void q4k_to_gptq(const uint8_t *__restrict__ src, uint32_t *__restrict__ qw, half *__restrict__ s,
                            half *__restrict__ z, int N, int K) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int G = K / GROUP, nb = K / 256;
    if (id >= (size_t)N * G) return;
    const int n = (int)(id % N), gi = (int)(id / N);
    const int sb = gi / 8, j = gi % 8;
    const uint8_t *blk = src + ((size_t)n * nb + sb) * 144;
    const float d = __half2float(*reinterpret_cast<const half *>(blk));
    const float dmin = __half2float(*reinterpret_cast<const half *>(blk + 2));
    const uint8_t *sc = blk + 4;
    int scv, mv;
    if (j < 4) {
        scv = sc[j] & 63;
        mv = sc[j + 4] & 63;
    } else {
        scv = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        mv = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const float M = dmin * (float)mv;
    float S = __half2float(__float2half_rn(d * (float)scv));
    if (S == 0.f && M != 0.f) S = __half2float(__float2half_rn(fabsf(M) / 60000.f));  // keeps z = M / S in range
    s[(size_t)gi * N + n] = __float2half_rn(S);
    z[(size_t)gi * N + n] = __float2half_rn(S != 0.f ? M / S : 0.f);
    const uint8_t *q = blk + 16 + 32 * (j / 2);
#pragma unroll
    for (int w = 0; w < 4; w++) {
        uint32_t v = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int l = 8 * w + i;
            const uint32_t nib = (j & 1) ? (q[l] >> 4) : (q[l] & 0xF);
            v |= nib << (4 * i);
        }
        qw[((size_t)gi * 4 + w) * N + n] = v;
    }
}

// marlin_permute_scales for grouped scales (vLLM marlin_utils.py): every 64 consecutive values of the row-major
// [K/32][N] array are permuted by scale_perm[8 i + j] = i + 8 j
__global__ void permute64(const half *__restrict__ in, half *__restrict__ out, size_t n) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const size_t base = id / 64 * 64;
    const int t = (int)(id % 64), i = t / 8, j = t % 8;
    out[id] = in[base + i + 8 * j];
}

__global__ void f32_to_f16(const float *__restrict__ x, half *__restrict__ y, size_t n) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) y[id] = __float2half_rn(x[id]);
}

__global__ void f16_to_f32(const half *__restrict__ x, float *__restrict__ y, size_t n) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) y[id] = __half2float(x[id]);
}

struct Prepared {
    uint32_t *qm = nullptr;  // Marlin-ordered weights, K * N / 8 words
    half *s = nullptr, *z = nullptr;  // permuted [K/32][N]
    long N = 0, K = 0;
};

inline int sm_count(int dev) {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    return v;
}

inline int max_smem(int dev) {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return v;
}

// raw: device Q4_K bytes of an [N][K] matrix. 0 on success; never inside a CUDA graph capture
inline int prepare(const void *raw, long N, long K, int dev, cudaStream_t st, Prepared &out) {
    if (K % 256 != 0 || N % 64 != 0) return 1;
    const size_t G = (size_t)K / GROUP, words = (size_t)K * N / 8;
    uint32_t *qw = nullptr;
    half *s0 = nullptr, *z0 = nullptr;
    Prepared p;
    p.N = N;
    p.K = K;
    if (cudaMalloc(&qw, words * 4) != cudaSuccess || cudaMalloc(&s0, G * N * 2) != cudaSuccess ||
        cudaMalloc(&z0, G * N * 2) != cudaSuccess || cudaMalloc(&p.qm, words * 4) != cudaSuccess ||
        cudaMalloc(&p.s, G * N * 2) != cudaSuccess || cudaMalloc(&p.z, G * N * 2) != cudaSuccess) {
        cudaFree(qw); cudaFree(s0); cudaFree(z0); cudaFree(p.qm); cudaFree(p.s); cudaFree(p.z);
        return 2;
    }
    const size_t th = G * N;
    q4k_to_gptq<<<(unsigned)((th + 255) / 256), 256, 0, st>>>((const uint8_t *)raw, qw, s0, z0, (int)N, (int)K);
    permute64<<<(unsigned)((th + 255) / 256), 256, 0, st>>>(s0, p.s, th);
    permute64<<<(unsigned)((th + 255) / 256), 256, 0, st>>>(z0, p.z, th);
    const int smem = max_smem(dev);
    auto rk = marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false>;
    cudaFuncSetAttribute(rk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    rk<<<sm_count(dev), marlin::repack_threads, smem, st>>>(qw, nullptr, p.qm, (int)K, (int)N);
    const bool ok = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(st) == cudaSuccess;
    cudaFree(qw); cudaFree(s0); cudaFree(z0);
    if (!ok) {
        cudaFree(p.qm); cudaFree(p.s); cudaFree(p.z);
        return 3;
    }
    out = p;
    return 0;
}

inline void release(Prepared &p) {
    cudaFree(p.qm); cudaFree(p.s); cudaFree(p.z);
    p = Prepared{};
}

// ------------------------------------------------------------------------- vLLM's marlin_mm, reduced to MQ1's case
// (fp16, kU4 with float zero points, group 32, no act order, no bias): its thread configurations and checks
struct thread_config_t { int thread_k, thread_n, num_threads; };
inline const thread_config_t small_batch[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
inline const thread_config_t large_batch[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};
constexpr int STAGES = marlin::pipe_stages;

inline int cache_size(const thread_config_t &c, int thread_m_blocks) {
    const int tb_k = c.thread_k, tb_n = c.thread_n, tb_m = thread_m_blocks * 16;
    const int sh_a = STAGES * (tb_m * tb_k) * 2;
    const int sh_b = STAGES * (tb_k * tb_n / 8) * 4;
    const int sh_red = tb_m * (tb_n + 8) * 2;
    const int sh_bias = tb_n * 2;
    int tmp = (sh_b > sh_red ? sh_red : sh_b) + sh_bias;
    tmp = std::max(std::max(sh_b, sh_red), tmp);
    const int sh_s = div_ceil(tb_k, GROUP) * tb_n * 2 * STAGES;
    return tmp + sh_a + sh_s + sh_s;  // float zero points take as much as the scales
}

inline bool valid(const thread_config_t &c, int thread_m_blocks, int n, int k, int smem) {
    if (k % c.thread_k != 0 || n % c.thread_n != 0) return false;
    if (c.thread_n < marlin::min_thread_n || c.thread_k < marlin::min_thread_k || c.num_threads < 128) return false;
    return cache_size(c, thread_m_blocks) + 512 <= smem;
}

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

template <int M, int NB, int KB, bool M8, int THREADS>
MarlinFuncPtr kern() {
    return marlin::Marlin<half, vllm::kU4.id(), vllm::kFloat16.id(), THREADS, M, NB, KB, M8, STAGES, GROUP / 16, true>;
}

inline MarlinFuncPtr pick(int m_blocks, int n_blocks, int k_blocks, bool m8, int threads) {
#define MQ1_IF(M, NB, KB, M8, T) \
    if (m_blocks == M && n_blocks == NB && k_blocks == KB && m8 == M8 && threads == T) return kern<M, NB, KB, M8, T>();
    MQ1_IF(1, 8, 8, true, 256) MQ1_IF(1, 8, 4, true, 128) MQ1_IF(1, 4, 8, true, 128)
    MQ1_IF(1, 8, 8, false, 256) MQ1_IF(1, 8, 4, false, 128) MQ1_IF(1, 4, 8, false, 128)
    MQ1_IF(2, 16, 4, false, 256) MQ1_IF(2, 8, 4, false, 128) MQ1_IF(2, 4, 8, false, 128)
#undef MQ1_IF
    return nullptr;
}

struct Scratch {
    half *a = nullptr, *c = nullptr;
    float *c_tmp = nullptr;
    int *locks = nullptr;
    size_t a_cap = 0, c_cap = 0, t_cap = 0, l_cap = 0;
};

inline int grow(void **p, size_t *cap, size_t need, bool zero = false) {
    if (*cap >= need) return 0;
    if (*p) cudaFree(*p);
    *p = nullptr;
    *cap = 0;
    if (cudaMalloc(p, need) != cudaSuccess) return 1;
    if (zero) cudaMemset(*p, 0, need);
    *cap = need;
    return 0;
}

// y[m][n] (float) = x[m][k] (float) times the prepared matrix; m <= 32
inline int run(const Prepared &p, const float *x, float *y, int m, int dev, cudaStream_t st, Scratch &sc,
               bool capturing) {
    const int n = (int)p.N, k = (int)p.K;
    if (m < 1 || m > 64) return 1;
    const int sms = sm_count(dev), smem = max_smem(dev);
    const int m_blocks = std::min(div_ceil(m, 16), 4);
    const size_t a_need = (size_t)m * k * 2, c_need = (size_t)m * n * 2;
    const size_t t_need = (size_t)sms * std::min((m + 15) / 16 * 16, 64) * marlin::max_thread_n * 4;
    const size_t l_need = (size_t)(n / 64 * 16 + sms) * sizeof(int);
    if (capturing) {
        if (sc.a_cap < a_need || sc.c_cap < c_need || sc.t_cap < t_need || sc.l_cap < l_need) return 5;
    } else if (grow((void **)&sc.a, &sc.a_cap, a_need) || grow((void **)&sc.c, &sc.c_cap, c_need) ||
               grow((void **)&sc.c_tmp, &sc.t_cap, t_need) || grow((void **)&sc.locks, &sc.l_cap, l_need, true)) {
        return 2;
    }
    const thread_config_t *cfgs = m_blocks > 1 ? large_batch : small_batch;
    const bool m8 = m <= 8;
    MarlinFuncPtr kernel = nullptr;
    thread_config_t c{};
    for (int i = 0; i < 3 && kernel == nullptr; i++) {
        if (!valid(cfgs[i], m_blocks, n, k, smem)) continue;
        kernel = pick(m_blocks, cfgs[i].thread_n / 16, cfgs[i].thread_k / 16, m8, cfgs[i].num_threads);
        c = cfgs[i];
    }
    if (kernel == nullptr || m_blocks > 2) return 3;  // only 1-32 rows are instantiated
    static std::mutex attr_lock;
    static std::map<const void *, bool> attr_set;
    {
        std::lock_guard<std::mutex> g(attr_lock);
        if (!attr_set[(const void *)kernel]) {
            if (capturing) return 6;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            attr_set[(const void *)kernel] = true;
        }
    }
    const size_t an = (size_t)m * k, cn = (size_t)m * n;
    f32_to_f16<<<(unsigned)((an + 255) / 256), 256, 0, st>>>(x, sc.a, an);
    kernel<<<sms, c.num_threads, smem, st>>>((const int4 *)sc.a, (const int4 *)p.qm, (int4 *)sc.c, (int4 *)sc.c_tmp,
                                              nullptr, (const int4 *)p.s, nullptr, (const int4 *)p.z, nullptr,
                                              k / GROUP, m, n, k, k, sc.locks, false, false, true, smem);
    f16_to_f32<<<(unsigned)((cn + 255) / 256), 256, 0, st>>>(sc.c, y, cn);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace mq1
