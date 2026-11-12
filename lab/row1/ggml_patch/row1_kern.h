// ROW1 Step 4: the ROW1 variant-A AMX kernel (exact Q8_0 semantics) inside ggml's AMX mul_mat, for 2 <= M <= 48.
// Included into ggml/src/ggml-cpu/amx/mmq.cpp. Weights are packed a second time at load (ggml's own AMX layout stays
// for its M == 1 VNNI kernel). Variant chosen at load and run time by ROW1_GGML = 0 | A | B | B16 (default 0).
// A: exact Q8_0 semantics, 2 <= M (chunks of 48 rows beyond 48). B: per-row int8 weights requantized from Q8_0,
// int8 activations with one scale per row, int32 accumulation over K. B16: B weights, 16-bit activations split
// into a high and a low int8 kernel row, combined after accumulation. B and B16 serve every M >= 1.
#pragma once
#include <immintrin.h>
#include <sys/mman.h>
#include <mutex>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <atomic>

namespace row1 {
struct tcfg { uint8_t pal, start; uint8_t r[14]; uint16_t cb[16]; uint8_t rows[16]; } __attribute__((packed));
static std::unordered_map<const void *, uint8_t *> g_packed;
static std::mutex g_mu;
enum { OFF = 0, VA = 1, VB = 2, VB16 = 3, VB8S = 4 };
static int seg() { static int g = -1; if (g < 0) { const char *v = getenv("ROW1_SEG"); g = v ? atoi(v) : 256; } return g; }
static int dyn() { static int d = -1; if (d < 0) { const char *v = getenv("ROW1_DYN"); d = v ? atoi(v) : 1; } return d; }
static std::atomic<int> g_next{0};
static int rbkb() { static int r = -1; if (r < 0) { const char *v = getenv("ROW1_RBKB"); r = v ? atoi(v) : 768; } return r; }
static int variant() {
    static int e = -1;
    if (e < 0) {
        const char *v = getenv("ROW1_GGML"); e = OFF;
        if (v && !strcmp(v, "A")) e = VA; else if (v && !strcmp(v, "B")) e = VB; else if (v && !strcmp(v, "B16")) e = VB16;
        else if (v && !strcmp(v, "B8S")) e = VB8S;
    }
    return e;
}
static int enabled() { return variant() == VA; }
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
// B layout: per (pair, 64-chunk c): [1 KiB member 0][1 KiB member 1], B[r][n*4+q] = W[n][c*64+r*4+q]; then N floats
static void pack_b(const uint8_t *src, int N, int K, int8_t *dst, float *scale) {
    int KB = K / 32;
    std::vector<float> w(K); std::vector<int8_t> qq(K);
    for (int row = 0; row < N; row++) {
        int j = row / 16, n = row % 16; float mx = 0;
        for (int kb = 0; kb < KB; kb++) {
            const uint8_t *blk = src + ((size_t)row * KB + kb) * 34;
            float d = _cvtsh_ss(*(const uint16_t *)blk);
            for (int i = 0; i < 32; i++) { w[kb * 32 + i] = d * (int8_t)blk[2 + i]; mx = std::max(mx, std::fabs(w[kb * 32 + i])); }
        }
        float s = mx / 127.f; if (s == 0) s = 1; scale[row] = s;
        for (int k = 0; k < K; k++) qq[k] = (int8_t)lrintf(w[k] / s);
        for (int c = 0; c < K / 64; c++) {
            int8_t *o = dst + (((size_t)(j >> 1) * (K / 64) + c) * 2 + (j & 1)) * 1024;
            for (int r = 0; r < 16; r++) for (int q = 0; q < 4; q++) o[r * 64 + n * 4 + q] = qq[c * 64 + r * 4 + q];
        }
    }
}
static size_t g_resident = 0;
static void register_weight(const void *key, const void *src, int N, int K) {
    int v = variant();
    if (v == OFF || N % 32 || K % 512) return;
    size_t n = v == VA ? (size_t)N * K / 32 * 34 : (size_t)N * K + (size_t)N * 4;
    void *p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
    madvise(p, n, MADV_HUGEPAGE);
    if (v == VA) pack_a((const uint8_t *)src, N, K, (uint8_t *)p);
    else pack_b((const uint8_t *)src, N, K, (int8_t *)p, (float *)((char *)p + (size_t)N * K));
    std::lock_guard<std::mutex> g(g_mu);
    g_packed[key] = (uint8_t *)p; g_resident += n;
    static int said = 0;
    if (N == 151936 || (++said % 50) == 0) fprintf(stderr, "row1: variant %d, %.0f MiB packed so far\n", v, g_resident / 1048576.0);
}
static const uint8_t *lookup(const void *key) {
    std::lock_guard<std::mutex> g(g_mu);
    auto it = g_packed.find(key);
    return it == g_packed.end() ? nullptr : it->second;
}
// tiles: mt <= 2 -> C 0..3, A 4,5, B 6,7; mt == 3 -> C 0..2, A 3..5, B 6. A cols 32 (variant A) or 64 (B).
static void config(int R, int acb = 32) {
    tcfg c; memset(&c, 0, sizeof c); c.pal = 1;
    int mt = (R + 15) / 16, brows = acb / 4;
    if (mt <= 2) {
        int r0 = mt == 1 ? R : 16, r1 = R - 16;
        c.rows[0] = c.rows[1] = c.rows[4] = r0; c.cb[0] = c.cb[1] = 64; c.cb[4] = acb;
        if (mt > 1) { c.rows[2] = c.rows[3] = c.rows[5] = r1; c.cb[2] = c.cb[3] = 64; c.cb[5] = acb; }
        c.rows[6] = c.rows[7] = brows; c.cb[6] = c.cb[7] = 64;
    } else {
        for (int i = 0; i < mt; i++) {
            int rows = (i == mt - 1) ? R - 16 * i : 16;
            c.rows[i] = rows; c.cb[i] = 64; c.rows[3 + i] = rows; c.cb[3 + i] = acb;
        }
        c.rows[6] = brows; c.cb[6] = 64;
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
// ---- B / B16 ----
// Kernel rows: B -> one per activation row; B16 -> 2m (high), 2m+1 (low). Rows are cut into chunks of CH kernel rows
// (CH = all rows when <= 48, else 32); X packed per chunk as [mt][K/64][16][64], padded rows zero.
struct bplan { int Mk, CH, nch; };
static bplan plan_b(int M, bool b16, bool b8s = false) { bplan p; p.Mk = b16 ? 2 * M : M; p.CH = p.Mk <= (b8s ? 32 : 48) ? std::max(p.Mk, 1) : 32; p.nch = (p.Mk + p.CH - 1) / p.CH; return p; }
static size_t wsize_b(int M, int K, bool b16) { bplan p = plan_b(M, b16); int chp = (p.CH + 15) / 16 * 16; return (size_t)p.nch * chp * K + (size_t)p.nch * chp * 4 * (K / 64) + 256; }
static inline int8_t *xrow(int8_t *Xp, const bplan &p, int K, int r, int c) {
    int ch = r / p.CH, rr = r % p.CH, chp = (p.CH + 15) / 16 * 16;
    return Xp + (size_t)ch * chp * K + (((size_t)(rr >> 4) * (K / 64) + c) * 16 + (rr & 15)) * 64;
}
static void quant_rows_b(const float *x, int ldx, int m0, int m1, int K, bool b16, const bplan &p, int8_t *Xp, float *xs) {
    for (int m = m0; m < m1; m++) {
        const float *v = x + (size_t)m * ldx;
        __m512 mx = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(v + k)));
        float amax = _mm512_reduce_max_ps(mx);
        float lim = b16 ? 32639.f : 127.f, d = amax / lim, id = d ? 1.f / d : 0.f;
        __m512 vid = _mm512_set1_ps(id);
        if (!b16) {
            xs[m] = d;
            for (int c = 0; c < K / 64; c++) {
                int8_t *o = xrow(Xp, p, K, m, c);
                for (int h = 0; h < 4; h++) {
                    __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(v + c * 64 + h * 16), vid));
                    _mm_storeu_si128((__m128i *)(o + h * 16), _mm512_cvtsepi32_epi8(q));
                }
            }
        } else {
            xs[2 * m] = xs[2 * m + 1] = d;
            for (int c = 0; c < K / 64; c++) {
                int8_t *oh = xrow(Xp, p, K, 2 * m, c), *ol = xrow(Xp, p, K, 2 * m + 1, c);
                for (int h = 0; h < 4; h++) {
                    __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(v + c * 64 + h * 16), vid));
                    __m512i hi = _mm512_srai_epi32(_mm512_add_epi32(q, _mm512_set1_epi32(128)), 8);
                    __m512i lo = _mm512_sub_epi32(q, _mm512_slli_epi32(hi, 8));
                    _mm_storeu_si128((__m128i *)(oh + h * 16), _mm512_cvtsepi32_epi8(hi));
                    _mm_storeu_si128((__m128i *)(ol + h * 16), _mm512_cvtsepi32_epi8(lo));
                }
            }
        }
    }
}
static void quant_rows_b8s(const float *x, int ldx, int m0, int m1, int K, int SEG, const bplan &p, int8_t *Xp, float *xs) {
    int NS = K / SEG;
    for (int m = m0; m < m1; m++) {
        const float *v = x + (size_t)m * ldx;
        for (int sg = 0; sg < NS; sg++) {
            __m512 mx = _mm512_setzero_ps();
            for (int k = sg * SEG; k < (sg + 1) * SEG; k += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(v + k)));
            float d = _mm512_reduce_max_ps(mx) / 127.f, id = d ? 1.f / d : 0.f; xs[(size_t)m * NS + sg] = d;
            __m512 vid = _mm512_set1_ps(id);
            for (int c = sg * SEG / 64; c < (sg + 1) * SEG / 64; c++) {
                int8_t *o = xrow(Xp, p, K, m, c);
                for (int h = 0; h < 4; h++) {
                    __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(v + c * 64 + h * 16), vid));
                    _mm_storeu_si128((__m128i *)(o + h * 16), _mm512_cvtsepi32_epi8(q));
                }
            }
        }
    }
}
// B8s kernel: CH <= 32 (mt <= 2); every SEG/64 chunks the C tiles are scaled by the activation segment scale and
// added to float accumulators.
static void kern_b8s(const int8_t *W, const float *ws, int K, int jp0, int jp1, int8_t *Xp, const float *xs, const bplan &p, int SEG, float *Y, int ldc) {
    int KC = K / 64, PF = pf(), CH = p.CH, mt = (CH + 15) / 16, chp = (CH + 15) / 16 * 16, NS = K / SEG, SC = SEG / 64;
    alignas(64) int32_t buf[4][16 * 16];
    __m512 acc[2][32];
    for (int jp = jp0; jp < jp1; jp++) {
        const int8_t *pw = W + (size_t)jp * KC * 2048;
        __m512 sw0 = _mm512_loadu_ps(ws + jp * 32), sw1 = _mm512_loadu_ps(ws + jp * 32 + 16);
        for (int ch = 0; ch < p.nch; ch++) {
            const int8_t *X = Xp + (size_t)ch * chp * K, *X1 = X + (size_t)KC * 1024;
            int rb = ch * CH, rows = std::min(CH, p.Mk - rb);
            for (int i = 0; i < rows; i++) acc[0][i] = acc[1][i] = _mm512_setzero_ps();
            const int8_t *q = pw;
            for (int sg = 0; sg < NS; sg++) {
                _tile_zero(0); _tile_zero(1); if (mt > 1) { _tile_zero(2); _tile_zero(3); }
                for (int c = sg * SC; c < (sg + 1) * SC; c++, q += 2048) {
                    if (PF && ch == 0) for (int l = 0; l < 2048; l += 64) _mm_prefetch((const char *)q + PF + l, _MM_HINT_T1);
                    _tile_loadd(6, q, 64); _tile_loadd(7, q + 1024, 64);
                    _tile_loadd(4, X + (size_t)c * 1024, 64); _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7);
                    if (mt > 1) { _tile_loadd(5, X1 + (size_t)c * 1024, 64); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7); }
                }
                _tile_stored(0, buf[0], 64); _tile_stored(1, buf[1], 64);
                if (mt > 1) { _tile_stored(2, buf[2], 64); _tile_stored(3, buf[3], 64); }
                for (int i = 0; i < rows; i++) {
                    const int32_t *b0 = buf[(i >> 4) * 2] + (i & 15) * 16, *b1 = buf[(i >> 4) * 2 + 1] + (i & 15) * 16;
                    __m512 sa = _mm512_set1_ps(xs[(size_t)(rb + i) * NS + sg]);
                    acc[0][i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(b0)), sa, acc[0][i]);
                    acc[1][i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(b1)), sa, acc[1][i]);
                }
            }
            for (int i = 0; i < rows; i++) {
                _mm512_storeu_ps(Y + (size_t)(rb + i) * ldc + jp * 32, _mm512_mul_ps(acc[0][i], sw0));
                _mm512_storeu_ps(Y + (size_t)(rb + i) * ldc + jp * 32 + 16, _mm512_mul_ps(acc[1][i], sw1));
            }
        }
    }
}
// post: tile buffer rows are kernel rows r0 .. r0+rows-1 (r0 even); write Y for output columns n0..n0+15
static inline void post_b(const int32_t *buf, int r0, int rows, int Mk, bool b16, const float *xs, __m512 sw, float *Y, int ldc, int n0) {
    if (!b16) {
        for (int i = 0; i < rows && r0 + i < Mk; i++) {
            __m512 v = _mm512_cvtepi32_ps(_mm512_load_si512(buf + i * 16));
            _mm512_storeu_ps(Y + (size_t)(r0 + i) * ldc + n0, _mm512_mul_ps(v, _mm512_mul_ps(sw, _mm512_set1_ps(xs[r0 + i]))));
        }
    } else {
        for (int i = 0; i + 1 < rows + 1 && r0 + i < Mk; i += 2) {
            __m512 h = _mm512_cvtepi32_ps(_mm512_load_si512(buf + i * 16)), l = _mm512_cvtepi32_ps(_mm512_load_si512(buf + (i + 1) * 16));
            __m512 v = _mm512_fmadd_ps(h, _mm512_set1_ps(256.f), l);
            _mm512_storeu_ps(Y + (size_t)((r0 + i) / 2) * ldc + n0, _mm512_mul_ps(v, _mm512_mul_ps(sw, _mm512_set1_ps(xs[r0 + i]))));
        }
    }
}
static void kern_b(const int8_t *W, const float *ws, int K, int jp0, int jp1, int8_t *Xp, const float *xs, const bplan &p, bool b16, float *Y, int ldc, int ch0 = 0, int ch1 = -1) {
    if (ch1 < 0) ch1 = p.nch;
    int KC = K / 64, PF = pf(), CH = p.CH, mt = (CH + 15) / 16, chp = (CH + 15) / 16 * 16;
    alignas(64) int32_t buf[16 * 16];
    for (int jp = jp0; jp < jp1; jp++) {
        const int8_t *pw = W + (size_t)jp * KC * 2048;
        __m512 sw0 = _mm512_loadu_ps(ws + jp * 32), sw1 = _mm512_loadu_ps(ws + jp * 32 + 16);
        for (int ch = ch0; ch < ch1; ch++) {
            const int8_t *X = Xp + (size_t)ch * chp * K, *X1 = X + (size_t)KC * 1024, *X2 = X + (size_t)2 * KC * 1024;
            int rb = ch * CH;  // first kernel row of this chunk
            if (mt <= 2) {
                const int8_t *q = pw;
                _tile_zero(0); _tile_zero(1); if (mt > 1) { _tile_zero(2); _tile_zero(3); }
                for (int c = 0; c < KC; c++, q += 2048) {
                    if (PF && ch == ch0) for (int l = 0; l < 2048; l += 64) _mm_prefetch((const char *)q + PF + l, _MM_HINT_T1);
                    _tile_loadd(6, q, 64); _tile_loadd(7, q + 1024, 64);
                    _tile_loadd(4, X + (size_t)c * 1024, 64); _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7);
                    if (mt > 1) { _tile_loadd(5, X1 + (size_t)c * 1024, 64); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7); }
                }
                int r0 = mt == 1 ? CH : 16;
                _tile_stored(0, buf, 64); post_b(buf, rb, r0, p.Mk, b16, xs, sw0, Y, ldc, jp * 32);
                _tile_stored(1, buf, 64); post_b(buf, rb, r0, p.Mk, b16, xs, sw1, Y, ldc, jp * 32 + 16);
                if (mt > 1) {
                    _tile_stored(2, buf, 64); post_b(buf, rb + 16, CH - 16, p.Mk, b16, xs, sw0, Y, ldc, jp * 32);
                    _tile_stored(3, buf, 64); post_b(buf, rb + 16, CH - 16, p.Mk, b16, xs, sw1, Y, ldc, jp * 32 + 16);
                }
            } else {
                for (int b = 0; b < 2; b++) {
                    const int8_t *q = pw + b * 1024;
                    _tile_zero(0); _tile_zero(1); _tile_zero(2);
                    for (int c = 0; c < KC; c++, q += 2048) {
                        if (PF && b == 0) for (int l = 0; l < 2048; l += 64) _mm_prefetch((const char *)q + PF + l, _MM_HINT_T1);
                        _tile_loadd(6, q, 64);
                        _tile_loadd(3, X + (size_t)c * 1024, 64); _tile_dpbssd(0, 3, 6);
                        _tile_loadd(4, X1 + (size_t)c * 1024, 64); _tile_dpbssd(1, 4, 6);
                        _tile_loadd(5, X2 + (size_t)c * 1024, 64); _tile_dpbssd(2, 5, 6);
                    }
                    __m512 sw = b ? sw1 : sw0;
                    _tile_stored(0, buf, 64); post_b(buf, rb, 16, p.Mk, b16, xs, sw, Y, ldc, jp * 32 + b * 16);
                    _tile_stored(1, buf, 64); post_b(buf, rb + 16, 16, p.Mk, b16, xs, sw, Y, ldc, jp * 32 + b * 16);
                    _tile_stored(2, buf, 64); post_b(buf, rb + 32, CH - 32, p.Mk, b16, xs, sw, Y, ldc, jp * 32 + b * 16);
                }
            }
        }
    }
}
// ---- A4: exact Q4_0 semantics (Q4_0 weights x Q8_0 activations), variant-A arithmetic ----
// packed per (pair, kb): [256 B member 0 nibbles][256 B member 1][32 B scales 0][32 B scales 1] = 576 B.
// nibble byte (r, i), i < 32: low = B-tile byte (r, i), high = B-tile byte (r, i + 32), both as raw 0..15.
static void pack_a4(const uint8_t *src, int N, int K, uint8_t *dst) {
    int KB = K / 32;
    for (int j = 0; j < N / 16; j++)
        for (int kb = 0; kb < KB; kb++) {
            uint8_t *o = dst + ((size_t)(j >> 1) * KB + kb) * 576;
            uint8_t tile[8][64];
            for (int n = 0; n < 16; n++) {
                const uint8_t *blk = src + ((size_t)(j * 16 + n) * KB + kb) * 18;
                for (int e = 0; e < 32; e++) {
                    uint8_t q = e < 16 ? (blk[2 + e] & 15) : (blk[2 + e - 16] >> 4);
                    tile[e / 4][n * 4 + e % 4] = q;
                }
                memcpy(o + 512 + (j & 1) * 32 + n * 2, blk, 2);
            }
            for (int r = 0; r < 8; r++) for (int i = 0; i < 32; i++)
                o[(j & 1) * 256 + r * 32 + i] = (uint8_t)(tile[r][i] | (tile[r][i + 32] << 4));
        }
}
static inline void unpack4(const uint8_t *p, int8_t *out) {  // 256 B nibbles -> 512 B B tile (8 x 64)
    const __m512i m = _mm512_set1_epi8(15), e8 = _mm512_set1_epi8(8);
    for (int h = 0; h < 4; h++) {
        __m512i v = _mm512_loadu_si512(p + h * 64);
        __m512i lo = _mm512_and_si512(v, m), hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m);
        _mm512_store_si512(out + h * 128, _mm512_sub_epi8(_mm512_shuffle_i64x2(lo, hi, 0x44), e8));
        _mm512_store_si512(out + h * 128 + 64, _mm512_sub_epi8(_mm512_shuffle_i64x2(lo, hi, 0xEE), e8));
    }
}
static void kern_a4(const uint8_t *W, int K, int j0, int j1, const int8_t *X, const float *xs, int R, float *Y, int ldc) {
    int mt = (R + 15) / 16, KB = K / 32, PF = pf();
    alignas(64) int32_t buf[4][16 * 16];
    alignas(64) int8_t bt[2][512];
    __m512 acc[2][48];
    const int8_t *X1 = X + (size_t)KB * 512, *X2 = X + (size_t)2 * KB * 512;
    int r0 = mt == 1 ? R : 16, r1 = mt == 2 ? R - 16 : 16, r2 = R - 32;
    for (int jp = j0; jp < j1; jp++) {
        const uint8_t *p = W + (size_t)jp * KB * 576;
        if (mt <= 2) {
            for (int m = 0; m < R; m++) acc[0][m] = acc[1][m] = _mm512_setzero_ps();
            for (int kb = 0; kb < KB; kb++, p += 576) {
                if (PF) for (int l = 0; l < 576; l += 64) _mm_prefetch((const char *)p + PF + l, _MM_HINT_T1);
                unpack4(p, bt[0]); unpack4(p + 256, bt[1]);
                _tile_loadd(6, bt[0], 64); _tile_loadd(7, bt[1], 64);
                _tile_zero(0); _tile_zero(1);
                _tile_loadd(4, X + (size_t)kb * 512, 32); _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7);
                if (mt > 1) { _tile_zero(2); _tile_zero(3); _tile_loadd(5, X1 + (size_t)kb * 512, 32); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7); }
                _tile_stored(0, buf[0], 64); _tile_stored(1, buf[1], 64);
                if (mt > 1) { _tile_stored(2, buf[2], 64); _tile_stored(3, buf[3], 64); }
                __m512 s0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p + 512)));
                __m512 s1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(p + 544)));
                post_a(r0, buf[0], s0, xs, KB, kb, acc[0]); post_a(r0, buf[1], s1, xs, KB, kb, acc[1]);
                if (mt > 1) { post_a(r1, buf[2], s0, xs + 16 * KB, KB, kb, acc[0] + 16); post_a(r1, buf[3], s1, xs + 16 * KB, KB, kb, acc[1] + 16); }
            }
            for (int m = 0; m < R; m++) { _mm512_storeu_ps(Y + (size_t)m * ldc + jp * 32, acc[0][m]); _mm512_storeu_ps(Y + (size_t)m * ldc + jp * 32 + 16, acc[1][m]); }
        } else {
            for (int b = 0; b < 2; b++) {
                const uint8_t *q = p;
                for (int m = 0; m < R; m++) acc[0][m] = _mm512_setzero_ps();
                for (int kb = 0; kb < KB; kb++, q += 576) {
                    if (PF && b == 0) for (int l = 0; l < 576; l += 64) _mm_prefetch((const char *)q + PF + l, _MM_HINT_T1);
                    unpack4(q + b * 256, bt[0]);
                    _tile_loadd(6, bt[0], 64);
                    _tile_zero(0); _tile_loadd(3, X + (size_t)kb * 512, 32); _tile_dpbssd(0, 3, 6);
                    _tile_zero(1); _tile_loadd(4, X1 + (size_t)kb * 512, 32); _tile_dpbssd(1, 4, 6);
                    _tile_zero(2); _tile_loadd(5, X2 + (size_t)kb * 512, 32); _tile_dpbssd(2, 5, 6);
                    _tile_stored(0, buf[0], 64); _tile_stored(1, buf[1], 64); _tile_stored(2, buf[2], 64);
                    __m512 sw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(q + 512 + b * 32)));
                    post_a(16, buf[0], sw, xs, KB, kb, acc[0]); post_a(16, buf[1], sw, xs + 16 * KB, KB, kb, acc[0] + 16);
                    post_a(r2, buf[2], sw, xs + 32 * KB, KB, kb, acc[0] + 32);
                }
                for (int m = 0; m < R; m++) _mm512_storeu_ps(Y + (size_t)m * ldc + jp * 32 + b * 16, acc[0][m]);
            }
        }
    }
}
static void register_weight_q4(const void *key, const void *src, int N, int K) {
    if (variant() != VA || N % 32 || K % 512) return;
    size_t n = (size_t)N * K / 32 * 18;
    void *p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
    madvise(p, n, MADV_HUGEPAGE);
    pack_a4((const uint8_t *)src, N, K, (uint8_t *)p);
    std::lock_guard<std::mutex> g(g_mu);
    g_packed[key] = (uint8_t *)p; g_resident += n;
}
static int a4m1() { static int e = -1; if (e < 0) { const char *v = getenv("ROW1_A4M1"); e = v ? atoi(v) : 0; } return e; }
// workspace beyond ggml's own (quantized A rows): packed X and float scales
static size_t extra_wsize(int M, int K) {  // A: chunks of 48 rows; B/B16 per plan
    size_t a = (size_t)((M + 47) / 48) * 48 * K + (size_t)((M + 47) / 48) * 48 * (K / 32) * 4 + 256;
    return std::max(a, wsize_b(M, K, true));
}
}  // namespace row1
