// ROW1 Step 4: the ROW1 variant-A AMX kernel (exact Q8_0 semantics) inside ggml's AMX mul_mat, for 2 <= M <= 48.
// Included into ggml/src/ggml-cpu/amx/mmq.cpp. Weights are packed a second time at load (ggml's own AMX layout stays
// for its M == 1 VNNI kernel). Disable at run time with ROW1_GGML=0.
#pragma once
#include <immintrin.h>
#include <sys/mman.h>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <cstdlib>

namespace row1 {
struct tcfg { uint8_t pal, start; uint8_t r[14]; uint16_t cb[16]; uint8_t rows[16]; } __attribute__((packed));
static std::unordered_map<const void *, uint8_t *> g_packed;
static std::mutex g_mu;
static int enabled() { static int e = -1; if (e < 0) { const char *v = getenv("ROW1_GGML"); e = v ? atoi(v) : 1; } return e; }
static int pf() { static int p = -1; if (p < 0) { const char *v = getenv("ROW1_PF"); p = v ? atoi(v) : 16384; } return p; }

static void pack_a(const uint8_t *src, int N, int K, uint8_t *dst) {  // per (pair, kb): [512 B0][512 B1][32 s0][32 s1]
    int KB = K / 32;
    for (int j = 0; j < N / 16; j++)
        for (int kb = 0; kb < KB; kb++) {
            uint8_t *o = dst + ((size_t)(j >> 1) * KB + kb) * 1088;
            for (int n = 0; n < 16; n++) {
                const uint8_t *blk = src + ((size_t)(j * 16 + n) * KB + kb) * 34;
                for (int r = 0; r < 8; r++)
                    for (int q = 0; q < 4; q++) o[(j & 1) * 512 + r * 64 + n * 4 + q] = blk[2 + r * 4 + q];
                memcpy(o + 1024 + (j & 1) * 32 + n * 2, blk, 2);
            }
        }
}
static void register_weight(const void *key, const void *src, int N, int K) {
    if (N % 32 || K % 32) return;
    size_t n = (size_t)N * K / 32 * 34;
    void *p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
    madvise(p, n, MADV_HUGEPAGE);
    pack_a((const uint8_t *)src, N, K, (uint8_t *)p);
    std::lock_guard<std::mutex> g(g_mu);
    g_packed[key] = (uint8_t *)p;
}
static const uint8_t *lookup(const void *key) {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_packed.find(key);
    return it == g_packed.end() ? nullptr : it->second;
}
static void config(int R) {
    tcfg c; memset(&c, 0, sizeof c); c.pal = 1;
    int mt = (R + 15) / 16;
    if (mt <= 2) {
        int r0 = mt == 1 ? R : 16, r1 = R - 16;
        c.rows[0] = c.rows[1] = c.rows[4] = r0; c.cb[0] = c.cb[1] = 64; c.cb[4] = 32;
        if (mt > 1) { c.rows[2] = c.rows[3] = c.rows[5] = r1; c.cb[2] = c.cb[3] = 64; c.cb[5] = 32; }
        c.rows[6] = c.rows[7] = 8; c.cb[6] = c.cb[7] = 64;
    } else {
        for (int i = 0; i < mt; i++) {
            int rows = (i == mt - 1) ? R - 16 * i : 16;
            c.rows[i] = rows; c.cb[i] = 64; c.rows[3 + i] = rows; c.cb[3 + i] = 32;
        }
        c.rows[6] = 8; c.cb[6] = 64;
    }
    _tile_loadconfig(&c);
}
static inline void post_a(int rows, const int32_t *buf, __m512 sw, const float *xs, int KB, int kb, __m512 *acc) {
    for (int m = 0; m < rows; m++) {
        __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_load_si512(buf + m * 16)), sw);
        acc[m] = _mm512_fmadd_ps(v, _mm512_set1_ps(xs[m * KB + kb]), acc[m]);
    }
}
// X packed [mt][kb][16][32], xs [R][KB]; pairs [j0, j1) of 32 output columns; Y row stride ldc
static void kern_a(const uint8_t *W, int K, int j0, int j1, const int8_t *X, const float *xs, int R, float *Y, int ldc) {
    int mt = (R + 15) / 16, KB = K / 32, PF = pf();
    alignas(64) int32_t buf[4][16 * 16];
    __m512 acc[2][48];
    const int8_t *X1 = X + (size_t)KB * 512, *X2 = X + (size_t)2 * KB * 512;
    int r0 = mt == 1 ? R : 16, r1 = mt == 2 ? R - 16 : 16, r2 = R - 32;
    for (int jp = j0; jp < j1; jp++) {
        const uint8_t *p = W + (size_t)jp * KB * 1088;
        if (mt <= 2) {
            for (int m = 0; m < R; m++) acc[0][m] = acc[1][m] = _mm512_setzero_ps();
            for (int kb = 0; kb < KB; kb++, p += 1088) {
                if (PF) for (int l = 0; l < 1088; l += 64) _mm_prefetch((const char *)p + PF + l, _MM_HINT_T1);
                _tile_loadd(6, p, 64); _tile_loadd(7, p + 512, 64);
                _tile_zero(0); _tile_zero(1);
                _tile_loadd(4, X + (size_t)kb * 512, 32); _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7);
                if (mt > 1) { _tile_zero(2); _tile_zero(3); _tile_loadd(5, X1 + (size_t)kb * 512, 32); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7); }
                _tile_stored(0, buf[0], 64); _tile_stored(1, buf[1], 64);
                if (mt > 1) { _tile_stored(2, buf[2], 64); _tile_stored(3, buf[3], 64); }
                __m512 s0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p + 1024)));
                __m512 s1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p + 1056)));
                post_a(r0, buf[0], s0, xs, KB, kb, acc[0]); post_a(r0, buf[1], s1, xs, KB, kb, acc[1]);
                if (mt > 1) { post_a(r1, buf[2], s0, xs + 16 * KB, KB, kb, acc[0] + 16); post_a(r1, buf[3], s1, xs + 16 * KB, KB, kb, acc[1] + 16); }
            }
            for (int m = 0; m < R; m++) { _mm512_storeu_ps(Y + (size_t)m * ldc + jp * 32, acc[0][m]); _mm512_storeu_ps(Y + (size_t)m * ldc + jp * 32 + 16, acc[1][m]); }
        } else {
            for (int b = 0; b < 2; b++) {
                const uint8_t *q = p;
                for (int m = 0; m < R; m++) acc[0][m] = _mm512_setzero_ps();
                for (int kb = 0; kb < KB; kb++, q += 1088) {
                    if (PF && b == 0) for (int l = 0; l < 1088; l += 64) _mm_prefetch((const char *)q + PF + l, _MM_HINT_T1);
                    _tile_loadd(6, q + b * 512, 64);
                    _tile_zero(0); _tile_loadd(3, X + (size_t)kb * 512, 32); _tile_dpbssd(0, 3, 6);
                    _tile_zero(1); _tile_loadd(4, X1 + (size_t)kb * 512, 32); _tile_dpbssd(1, 4, 6);
                    _tile_zero(2); _tile_loadd(5, X2 + (size_t)kb * 512, 32); _tile_dpbssd(2, 5, 6);
                    _tile_stored(0, buf[0], 64); _tile_stored(1, buf[1], 64); _tile_stored(2, buf[2], 64);
                    __m512 sw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(q + 1024 + b * 32)));
                    post_a(16, buf[0], sw, xs, KB, kb, acc[0]); post_a(16, buf[1], sw, xs + 16 * KB, KB, kb, acc[0] + 16);
                    post_a(r2, buf[2], sw, xs + 32 * KB, KB, kb, acc[0] + 32);
                }
                for (int m = 0; m < R; m++) _mm512_storeu_ps(Y + (size_t)m * ldc + jp * 32 + b * 16, acc[0][m]);
            }
        }
    }
}
// workspace beyond ggml's own (quantized A rows): packed X and float scales
static size_t extra_wsize(int M, int K) { return (size_t)((M + 15) / 16) * 16 * K + (size_t)M * (K / 32) * 4 + 128; }
}  // namespace row1
