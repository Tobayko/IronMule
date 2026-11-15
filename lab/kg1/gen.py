"""KG1: generate CUDA kernels for llama.cpp's Q4_K weights times float32 activations (the matmuls that take most of a
decode step), one variant per parameter set, so that a machine can pick (and later learn) the fastest for its GPU.

A variant is (L, R, W, B):
  L  lanes that share one 256-weight block (8, 16 or 32): each lane reads 128 / L bytes of 4-bit quants at once
  R  output rows one warp computes (the activations it loaded serve all R rows)
  W  warps per thread block
  B  activation rows (tokens of one step) the kernel takes at once, a compile-time constant

The arithmetic: per sub-block of 32 weights, d * sc * sum(q * x) - dmin * m * sum(x), in float32 with the
activations as they are (llama.cpp's own kernels first round them to 8 bits). Every variant is checked against a
float64 reference that dequantises element by element (an independent decoding of the format), and against
llama.cpp's result on the same weights.

    python gen.py > bench.cu          # all variants that fit the register budget, with the benchmark's main()
"""
import itertools
import sys

LANES, ROWS, WARPS, BATCHES = (8, 16, 32), (1, 2, 4), (2, 4, 8), (1, 2, 4, 8, 16)


def variants():
    for L, R, W, B in itertools.product(LANES, ROWS, WARPS, BATCHES):
        per_lane = 128 // L
        if R * B > 32 or 2 * B * per_lane > 128:  # accumulators and activations held in registers
            continue
        yield L, R, W, B


KERNEL = r"""
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(block_q4_K) == 144, "Q4_K block");

// the 6-bit scale and min of sub-block j (0..7) from the 12 packed bytes in words y, z, w (bytes 4..15 of a block)
__device__ __forceinline__ uint32_t sbyte(uint32_t y, uint32_t z, uint32_t w, int k) {
    const uint32_t word = k < 4 ? y : (k < 8 ? z : w);
    return (word >> (8 * (k & 3))) & 0xFF;
}
__device__ __forceinline__ void scale_min(uint32_t y, uint32_t z, uint32_t w, int j, float &sc, float &m) {
    if (j < 4) {
        sc = (float)(sbyte(y, z, w, j) & 63); m = (float)(sbyte(y, z, w, j + 4) & 63);
    } else {
        sc = (float)((sbyte(y, z, w, j + 4) & 0xF) | ((sbyte(y, z, w, j - 4) >> 6) << 4));
        m = (float)((sbyte(y, z, w, j + 4) >> 4) | ((sbyte(y, z, w, j) >> 6) << 4));
    }
}

template <int L, int R, int W, int B>
__global__ void __launch_bounds__(W * 32) kq4k(const block_q4_K *__restrict__ wq, const float *__restrict__ x,
                                               float *__restrict__ y, int N, int K) {
    constexpr int G = 32 / L, BYTES = 128 / L, LPC = L / 4;
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
        float xl[B][BYTES], xh[B][BYTES], sl[B], sh[B];
#pragma unroll
        for (int b = 0; b < B; b++) {
            const float4 *pl = reinterpret_cast<const float4 *>(x + (size_t)b * K + ib * 256 + 64 * c + l0);
            const float4 *ph = reinterpret_cast<const float4 *>(x + (size_t)b * K + ib * 256 + 64 * c + 32 + l0);
            sl[b] = 0.f; sh[b] = 0.f;
#pragma unroll
            for (int v = 0; v < BYTES / 4; v++) {
                const float4 a = __ldg(pl + v), h = __ldg(ph + v);
                xl[b][4 * v] = a.x; xl[b][4 * v + 1] = a.y; xl[b][4 * v + 2] = a.z; xl[b][4 * v + 3] = a.w;
                xh[b][4 * v] = h.x; xh[b][4 * v + 1] = h.y; xh[b][4 * v + 2] = h.z; xh[b][4 * v + 3] = h.w;
                sl[b] += (a.x + a.y) + (a.z + a.w); sh[b] += (h.x + h.y) + (h.z + h.w);
            }
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
            uint32_t q[BYTES / 4];
            const uint8_t *qp = blk->qs + 32 * c + l0;
            if constexpr (BYTES == 16) {
                const uint4 v = __ldg(reinterpret_cast<const uint4 *>(qp)); q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
            } else if constexpr (BYTES == 8) {
                const uint2 v = __ldg(reinterpret_cast<const uint2 *>(qp)); q[0] = v.x; q[1] = v.y;
            } else {
                q[0] = __ldg(reinterpret_cast<const uint32_t *>(qp));
            }
#pragma unroll
            for (int b = 0; b < B; b++) {
                float lo = 0.f, hi = 0.f;
#pragma unroll
                for (int i = 0; i < BYTES; i++) {
                    const uint32_t byte = (q[i / 4] >> (8 * (i & 3))) & 0xFF;
                    lo = fmaf((float)(byte & 0xF), xl[b][i], lo);
                    hi = fmaf((float)(byte >> 4), xh[b][i], hi);
                }
                acc[r][b] += (d1 * lo - n1 * sl[b]) + (d2 * hi - n2 * sh[b]);
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

// the float64 reference: one warp per output row, element by element (dequantize_row_q4_K of ggml-quants.c)
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
            const double d = (double)__half2float(__ushort_as_half(blk->d));
            const double dm = (double)__half2float(__ushort_as_half(blk->dmin));
            s += (d * s6 * q - dm * m6) * (double)x[(size_t)b * K + k];
        }
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) y[(size_t)b * N + row] = s;
    }
}

typedef void (*launch_t)(const block_q4_K *, const float *, float *, int, int, cudaStream_t);
template <int L, int R, int W, int B>
void launch(const block_q4_K *w, const float *x, float *y, int N, int K, cudaStream_t s) {
    const int rows = R * W;
    kq4k<L, R, W, B><<<(N + rows - 1) / rows, W * 32, 0, s>>>(w, x, y, N, K);
}
struct Variant { int L, R, W, B; launch_t fn; };
"""

MAIN = r"""
static std::vector<char> slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "no %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> v(n); if (fread(v.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return v;
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// usage: bench WEIGHTS.bin N K X.bin GGML_Y_PREFIX COPIES REPS NAME
int main(int argc, char **argv) {
    const char *wpath = argv[1]; const int N = atoi(argv[2]), K = atoi(argv[3]);
    const char *xpath = argv[4], *gpref = argv[5]; const int C = atoi(argv[6]), REPS = atoi(argv[7]);
    const char *name = argv[8];
    std::vector<char> wb = slurp(wpath), xb = slurp(xpath);
    const size_t wbytes = (size_t)N * (K / 256) * 144;
    if (wb.size() != wbytes) { fprintf(stderr, "weights: %zu != %zu\n", wb.size(), wbytes); return 1; }
    std::vector<block_q4_K *> dw(C);
    for (int c = 0; c < C; c++) { CK(cudaMalloc(&dw[c], wbytes)); CK(cudaMemcpy(dw[c], wb.data(), wbytes, cudaMemcpyHostToDevice)); }
    float *dx, *dy; double *dr;
    CK(cudaMalloc(&dx, xb.size())); CK(cudaMemcpy(dx, xb.data(), xb.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy, (size_t)16 * N * 4)); CK(cudaMalloc(&dr, (size_t)16 * N * 8));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> hy((size_t)16 * N); std::vector<double> hr((size_t)16 * N);
    int lastB = -1;
    std::vector<float> gy;
    for (const Variant &v : VARIANTS) {
        if (v.B != lastB) {
            ref64<<<(N + 7) / 8, 256>>>(dw[0], dx, dr, N, K, v.B); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hr.data(), dr, (size_t)v.B * N * 8, cudaMemcpyDeviceToHost));
            std::string gp = std::string(gpref) + "_B" + std::to_string(v.B) + ".bin";
            FILE *f = fopen(gp.c_str(), "rb"); gy.assign((size_t)v.B * N, NAN);
            if (f) { if (fread(gy.data(), 4, gy.size(), f) != gy.size()) gy.assign((size_t)v.B * N, NAN); fclose(f); }
            double num = 0, den = 0;
            for (size_t i = 0; i < gy.size(); i++) { num += (gy[i] - hr[i]) * (gy[i] - hr[i]); den += hr[i] * hr[i]; }
            printf("{\"tensor\": \"%s\", \"N\": %d, \"K\": %d, \"B\": %d, \"ggml_rel_err\": %.3e}\n", name, N, K, v.B, sqrt(num / den));
            lastB = v.B;
        }
        v.fn(dw[0], dx, dy, N, K, 0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hy.data(), dy, (size_t)v.B * N * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, gnum = 0;
        for (size_t i = 0; i < (size_t)v.B * N; i++) {
            num += (hy[i] - hr[i]) * (hy[i] - hr[i]); den += hr[i] * hr[i]; gnum += (hy[i] - gy[i]) * (hy[i] - gy[i]);
        }
        for (int i = 0; i < 3; i++) for (int c = 0; c < C; c++) v.fn(dw[c], dx, dy, N, K, 0);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < REPS; i++) for (int c = 0; c < C; c++) v.fn(dw[c], dx, dy, N, K, 0);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1000.0 * ms / (REPS * C);
        printf("{\"tensor\": \"%s\", \"N\": %d, \"K\": %d, \"B\": %d, \"L\": %d, \"R\": %d, \"W\": %d, \"us\": %.2f, "
               "\"GBps\": %.1f, \"rel_err\": %.3e, \"rel_to_ggml\": %.3e}\n", name, N, K, v.B, v.L, v.R, v.W, us,
               wbytes / us / 1e3, sqrt(num / den), sqrt(gnum / den));
        fflush(stdout);
    }
    return 0;
}
"""


def source() -> str:
    rows = [f"    {{{L}, {R}, {W}, {B}, launch<{L}, {R}, {W}, {B}>}}," for L, R, W, B in
            sorted(variants(), key=lambda v: (v[3], v[0], v[1], v[2]))]
    return KERNEL + "static const std::vector<Variant> VARIANTS = {\n" + "\n".join(rows) + "\n};\n" + MAIN


if __name__ == "__main__":
    sys.stdout.write(source())
