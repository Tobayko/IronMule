// MQ1: llama.cpp's Q4_K weights, repacked once so that a warp streams them (the Marlin principle, arXiv 2408.11743,
// on a T4: Turing has no cp.async, so the streaming is in registers), times float32 activations kept as they are
// (llama.cpp rounds them to 8 bits first), for 1-4 tokens.
//
// Layout (repacked from Q4_K by the harness): per row, the qs of every 32-weight sub-block in order, byte i holding
// weights 2i (low nibble) and 2i+1 (high), 16 bytes per sub-block; beside it each 256-weight block's 16-byte header
// (d, dmin, the 12 bytes of 6-bit scales and mins) as in Q4_K. Same bytes as Q4_K (4.5 bits a weight), only reordered:
// lane j of a warp reads sub-block j's 16 bytes, so a warp reads 512 contiguous bytes per iteration.
//
// Kernel <W, R, U, B>: W warps per thread block, R rows per warp (their activations loaded once), U iterations
// unrolled (loads in flight before the arithmetic), B tokens. Checked against a float64 reference on the original Q4_K
// bytes and timed against llama.cpp's own kernel (ref_ggml.c of lab/kg1).
//
// usage: mq1 Q4K.bin QS.bin HDR.bin N K X.bin GGML_Y_PREFIX COPIES REPS NAME
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };

__device__ __forceinline__ uint32_t sbyte(uint32_t y, uint32_t z, uint32_t w, int k) {
    const uint32_t word = k < 4 ? y : (k < 8 ? z : w);
    return (word >> (8 * (k & 3))) & 0xFF;
}
__device__ __forceinline__ void scale_min(uint4 h, int j, float &sc, float &m) {
    if (j < 4) {
        sc = (float)(sbyte(h.y, h.z, h.w, j) & 63); m = (float)(sbyte(h.y, h.z, h.w, j + 4) & 63);
    } else {
        sc = (float)((sbyte(h.y, h.z, h.w, j + 4) & 0xF) | ((sbyte(h.y, h.z, h.w, j - 4) >> 6) << 4));
        m = (float)((sbyte(h.y, h.z, h.w, j + 4) >> 4) | ((sbyte(h.y, h.z, h.w, j) >> 6) << 4));
    }
}

template <int W, int R, int U, int B>
__global__ void __launch_bounds__(W * 32) mq1(const uint4 *__restrict__ qs, const uint4 *__restrict__ hdr,
                                              const float *__restrict__ x, float *__restrict__ y, int N, int K) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row0 = (blockIdx.x * W + warp) * R;
    if (row0 >= N) return;
    const int nsub = K / 32, nblk = K / 256;
    float acc[R][B];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < B; b++) acc[r][b] = 0.f;
    for (int base = 0; base < nsub; base += 32 * U) {
        uint4 q[U][R], h[U][R];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int j = base + 32 * u + lane;
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (j < nsub && row0 + r < N) {
                    q[u][r] = __ldg(qs + (size_t)(row0 + r) * nsub + j);
                    h[u][r] = __ldg(hdr + (size_t)(row0 + r) * nblk + j / 8);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int j = base + 32 * u + lane;
            if (j >= nsub) break;
            float xs[B][32], sx[B];
#pragma unroll
            for (int b = 0; b < B; b++) {
                const float4 *px = reinterpret_cast<const float4 *>(x + (size_t)b * K + 32 * j);
                sx[b] = 0.f;
#pragma unroll
                for (int v = 0; v < 8; v++) {
                    const float4 a = __ldg(px + v);
                    xs[b][4 * v] = a.x; xs[b][4 * v + 1] = a.y; xs[b][4 * v + 2] = a.z; xs[b][4 * v + 3] = a.w;
                    sx[b] += (a.x + a.y) + (a.z + a.w);
                }
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (row0 + r >= N) break;
                const float d = __half2float(__ushort_as_half((unsigned short)(h[u][r].x & 0xFFFF)));
                const float dm = __half2float(__ushort_as_half((unsigned short)(h[u][r].x >> 16)));
                float sc, mn;
                scale_min(h[u][r], j % 8, sc, mn);
                const float ds = d * sc, ms = dm * mn;
                const uint32_t words[4] = {q[u][r].x, q[u][r].y, q[u][r].z, q[u][r].w};
#pragma unroll
                for (int b = 0; b < B; b++) {
                    float s = 0.f;
#pragma unroll
                    for (int i = 0; i < 16; i++) {
                        const uint32_t byte = (words[i / 4] >> (8 * (i & 3))) & 0xFF;
                        s = fmaf((float)(byte & 0xF), xs[b][2 * i], s);
                        s = fmaf((float)(byte >> 4), xs[b][2 * i + 1], s);
                    }
                    acc[r][b] += ds * s - ms * sx[b];
                }
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

typedef void (*launch_t)(const uint4 *, const uint4 *, const float *, float *, int, int);
template <int W, int R, int U, int B>
void launch(const uint4 *qs, const uint4 *hdr, const float *x, float *y, int N, int K) {
    mq1<W, R, U, B><<<(N + W * R - 1) / (W * R), W * 32>>>(qs, hdr, x, y, N, K);
}
struct Variant { int W, R, U, B; launch_t fn; };
#define V(W, R, U, B) {W, R, U, B, launch<W, R, U, B>}
static const std::vector<Variant> VARIANTS = {
    V(2,1,1,1), V(4,1,1,1), V(8,1,1,1), V(2,1,2,1), V(4,1,2,1), V(8,1,2,1), V(4,1,4,1), V(8,1,4,1),
    V(2,2,1,1), V(4,2,1,1), V(4,2,2,1), V(8,2,2,1), V(4,4,1,1), V(4,4,2,1),
    V(2,1,1,2), V(4,1,1,2), V(4,1,2,2), V(8,1,2,2), V(4,2,1,2), V(4,2,2,2), V(8,2,2,2),
    V(2,1,1,4), V(4,1,1,4), V(4,1,2,4), V(4,2,1,4), V(8,2,1,4),
};

static std::vector<char> slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "no %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return v;
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

int main(int argc, char **argv) {
    std::vector<char> wq = slurp(argv[1]), qsb = slurp(argv[2]), hb = slurp(argv[3]);
    const int N = atoi(argv[4]), K = atoi(argv[5]);
    std::vector<char> xb = slurp(argv[6]);
    const char *gpref = argv[7]; const int C = atoi(argv[8]), REPS = atoi(argv[9]); const char *name = argv[10];
    const size_t bytes = wq.size();
    block_q4_K *dq; CK(cudaMalloc(&dq, bytes)); CK(cudaMemcpy(dq, wq.data(), bytes, cudaMemcpyHostToDevice));
    std::vector<uint4 *> dqs(C), dh(C);
    for (int c = 0; c < C; c++) {
        CK(cudaMalloc(&dqs[c], qsb.size())); CK(cudaMemcpy(dqs[c], qsb.data(), qsb.size(), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dh[c], hb.size())); CK(cudaMemcpy(dh[c], hb.data(), hb.size(), cudaMemcpyHostToDevice));
    }
    float *dx, *dy; double *dr;
    CK(cudaMalloc(&dx, xb.size())); CK(cudaMemcpy(dx, xb.data(), xb.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, (size_t)4 * N * 4)); CK(cudaMalloc(&dr, (size_t)4 * N * 8));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> hy((size_t)4 * N), gy; std::vector<double> hr((size_t)4 * N);
    int lastB = -1;
    for (const Variant &v : VARIANTS) {
        if (v.B != lastB) {
            ref64<<<(N + 7) / 8, 256>>>(dq, dx, dr, N, K, v.B); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hr.data(), dr, (size_t)v.B * N * 8, cudaMemcpyDeviceToHost));
            std::string gp = std::string(gpref) + "_B" + std::to_string(v.B) + ".bin";
            gy.assign((size_t)v.B * N, NAN);
            FILE *f = fopen(gp.c_str(), "rb");
            if (f) { if (fread(gy.data(), 4, gy.size(), f) != gy.size()) gy.assign((size_t)v.B * N, NAN); fclose(f); }
            double num = 0, den = 0;
            for (size_t i = 0; i < gy.size(); i++) { num += (gy[i] - hr[i]) * (gy[i] - hr[i]); den += hr[i] * hr[i]; }
            printf("{\"tensor\": \"%s\", \"N\": %d, \"K\": %d, \"B\": %d, \"ggml_rel_err\": %.3e}\n", name, N, K, v.B, sqrt(num / den));
            lastB = v.B;
        }
        v.fn(dqs[0], dh[0], dx, dy, N, K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hy.data(), dy, (size_t)v.B * N * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < (size_t)v.B * N; i++) { num += (hy[i] - hr[i]) * (hy[i] - hr[i]); den += hr[i] * hr[i]; }
        for (int i = 0; i < 3; i++) for (int c = 0; c < C; c++) v.fn(dqs[c], dh[c], dx, dy, N, K);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < REPS; i++) for (int c = 0; c < C; c++) v.fn(dqs[c], dh[c], dx, dy, N, K);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1000.0 * ms / ((double)REPS * C);
        printf("{\"tensor\": \"%s\", \"N\": %d, \"K\": %d, \"B\": %d, \"W\": %d, \"R\": %d, \"U\": %d, \"us\": %.2f, "
               "\"GBps\": %.1f, \"rel_err\": %.3e}\n", name, N, K, v.B, v.W, v.R, v.U, us, bytes / us / 1e3,
               sqrt(num / den));
        fflush(stdout);
    }
    return 0;
}
