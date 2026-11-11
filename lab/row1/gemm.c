// ROW1 E2: row-cheap int8 GEMM with AMX.  Y[R x N] = X[R x K] . W[N x K]^T, R <= 48.
// Variant A: exact Q8_0 semantics (per-32 weight scale f16, per-32 activation scale), K step 32.
// Variant B: one float scale per weight row and per activation row, long int32 accumulation, K step 64.
// Packed weights are streamed strictly sequentially; one B tile load serves up to 3 A/C tile pairs.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

typedef struct { uint8_t pal, start; uint8_t r[14]; uint16_t cb[16]; uint8_t rows[16]; } __attribute__((packed)) tcfg;
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

// ---------------- packing ----------------
// Q8_0 block: f16 d, int8 q[32]  (34 bytes)
// Variant A packed, per n-tile (16 rows), per 32-block kb: 512 B int8 in tdpbssd B layout (8 rows x 64 B:
// B[r][n*4+q] = W[n][kb*32 + r*4 + q]) followed by 16 f16 scales (32 B). 544 B per (n-tile, kb).
size_t row1_size_a(int N, int K) { return (size_t)(N / 32) * (K / 32) * 1088; }
void row1_pack_a(const uint8_t *src, int N, int K, uint8_t *dst) {
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
// activations for A: [mt][kb][16 rows][32 B]
void row1_packx_a(const int8_t *x, int R, int K, int8_t *xp) {
    int KB = K / 32;
    for (int m = 0; m < R; m++) for (int kb = 0; kb < KB; kb++)
        memcpy(xp + (((size_t)(m >> 4) * KB + kb) * 16 + (m & 15)) * 32, x + (size_t)m * K + kb * 32, 32);
}
// Variant B packed, per n-tile, per 64-chunk c: 1024 B (16 rows x 64 B, B[r][n*4+q] = W[n][c*64+r*4+q]);
// row scales separately (float per row). Requantized from Q8_0: s = max|w|/127, q = round(w/s).
size_t row1_size_b(int N, int K) { return (size_t)N * K; }
void row1_pack_b(const uint8_t *src, int N, int K, int8_t *dst, float *scale) {
    int KB = K / 32;
    float *w = malloc(sizeof(float) * K);
    int8_t *qq = malloc(K);
    for (int j = 0; j < N / 16; j++)
        for (int n = 0; n < 16; n++) {
            int row = j * 16 + n; float mx = 0;
            for (int kb = 0; kb < KB; kb++) {
                const uint8_t *blk = src + ((size_t)row * KB + kb) * 34;
                float d = h2f(*(const uint16_t *)blk);
                for (int i = 0; i < 32; i++) { w[kb * 32 + i] = d * (int8_t)blk[2 + i]; mx = fmaxf(mx, fabsf(w[kb * 32 + i])); }
            }
            float s = mx / 127.f; if (s == 0) s = 1; scale[row] = s;
            for (int k = 0; k < K; k++) qq[k] = (int8_t)lrintf(w[k] / s);
            for (int c = 0; c < K / 64; c++) {  // n-tile pairs interleaved: [pair][c][member][1 KiB]
                int8_t *o = dst + (((size_t)(j >> 1) * (K / 64) + c) * 2 + (j & 1)) * 1024;
                for (int r = 0; r < 16; r++) for (int q = 0; q < 4; q++) o[r * 64 + n * 4 + q] = qq[c * 64 + r * 4 + q];
            }
        }
    free(w); free(qq);
}
void *row1_alloc(size_t n, int huge) {
    void *p = mmap(0, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
    madvise(p, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    return p;
}
void row1_free(void *p, size_t n) { munmap(p, n); }

// ---------------- activation quantization ----------------
// A: per row, per 32-block: q int8, s float.   B: per row: q int8, one scale.
void row1_quant_a(const float *x, int R, int K, int8_t *q, float *s) {
    for (int m = 0; m < R; m++) for (int kb = 0; kb < K / 32; kb++) {
        const float *v = x + (size_t)m * K + kb * 32; float mx = 0;
        for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(v[i]));
        float d = mx / 127.f, id = d ? 1.f / d : 0;
        d = h2f(_cvtss_sh(d, 0));  // ggml stores the Q8_0 activation scale as f16
        s[m * (K / 32) + kb] = d;
        for (int i = 0; i < 32; i++) q[(size_t)m * K + kb * 32 + i] = (int8_t)lrintf(v[i] * id);
    }
}
void row1_quant_b(const float *x, int R, int K, int8_t *q, float *s) {
    for (int m = 0; m < R; m++) {
        const float *v = x + (size_t)m * K; float mx = 0;
        for (int k = 0; k < K; k++) mx = fmaxf(mx, fabsf(v[k]));
        float d = mx / 127.f, id = d ? 1.f / d : 0; s[m] = d;
        for (int k = 0; k < K; k++) q[(size_t)m * K + k] = (int8_t)lrintf(v[k] * id);
    }
}

void row1_packx_b(const int8_t *x, int R, int K, int8_t *xp) {
    int KC = K / 64;
    for (int m = 0; m < R; m++) for (int c = 0; c < KC; c++)
        memcpy(xp + (((size_t)(m >> 4) * KC + c) * 16 + (m & 15)) * 64, x + (size_t)m * K + c * 64, 64);
}
// ---------------- kernels (one thread, n-tiles [j0, j1)) ----------------
// tiles: 0..2 C, 3..5 A, 6 B
static void config(int R, int variant) {
    tcfg c; memset(&c, 0, sizeof c); c.pal = 1;
    int mt = (R + 15) / 16;
    if (mt <= 2) {
        int r0 = mt == 1 ? R : 16, r1 = R - 16, acb = variant ? 64 : 32;
        c.rows[0] = c.rows[1] = c.rows[4] = r0; c.cb[0] = c.cb[1] = 64; c.cb[4] = acb;
        if (mt > 1) { c.rows[2] = c.rows[3] = c.rows[5] = r1; c.cb[2] = c.cb[3] = 64; c.cb[5] = acb; }
        c.rows[6] = c.rows[7] = variant ? 16 : 8; c.cb[6] = c.cb[7] = 64;
    } else {
        for (int i = 0; i < mt; i++) {
            int rows = (i == mt - 1) ? R - 16 * i : 16;
            c.rows[i] = rows; c.cb[i] = 64;
            c.rows[3 + i] = rows; c.cb[3 + i] = variant ? 64 : 32;
        }
        c.rows[6] = variant ? 16 : 8; c.cb[6] = 64;
    }
    _tile_loadconfig(&c);
}
static int PF = 0;
static void post_b(int t, int rows, int32_t *buf, const float *ws, const float *xs, float *Y, int N, int j) {
    __m512 sw = _mm512_loadu_ps(ws + j * 16);
    for (int m = 0; m < rows; m++) {
        __m512 v = _mm512_cvtepi32_ps(_mm512_load_si512(buf + m * 16));
        _mm512_storeu_ps(Y + (size_t)(16 * t + m) * N + j * 16, _mm512_mul_ps(v, _mm512_mul_ps(sw, _mm512_set1_ps(xs[16 * t + m]))));
    }
}
#define PREF(q) do { if (PF) { const char *pq = (const char *)(q) + PF; for (int l = 0; l < 2048; l += 64) _mm_prefetch(pq + l, PFH); } } while (0)
static int PFH_T1 = 0;
#define PFH _MM_HINT_T1
static void kern_b(const int8_t *W, const float *ws, int K, int j0, int j1, const int8_t *X, const float *xs, int R, float *Y, int N) {
    // j0, j1 are n-tile pair indices
    int mt = (R + 15) / 16, KC = K / 64;
    int32_t buf[16 * 16] __attribute__((aligned(64)));
    const int8_t *X1 = X + (size_t)KC * 1024, *X2 = X + (size_t)2 * KC * 1024;
    int r0 = mt == 1 ? R : 16, r1 = mt == 2 ? R - 16 : 16, r2 = R - 32;
    for (int jp = j0; jp < j1; jp++) {
        const int8_t *p = W + (size_t)jp * KC * 2048;
        if (mt <= 2) {  // tiles: C 0..3 (C[t][member] = t*2+member), A 4,5, B 6,7
            _tile_zero(0); _tile_zero(1); if (mt > 1) { _tile_zero(2); _tile_zero(3); }
            for (int c = 0; c < KC; c++, p += 2048) {
                PREF(p);
                _tile_loadd(6, p, 64); _tile_loadd(7, p + 1024, 64);
                _tile_loadd(4, X + (size_t)c * 1024, 64);
                _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7);
                if (mt > 1) { _tile_loadd(5, X1 + (size_t)c * 1024, 64); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7); }
            }
            _tile_stored(0, buf, 64); post_b(0, r0, buf, ws, xs, Y, N, 2 * jp);
            _tile_stored(1, buf, 64); post_b(0, r0, buf, ws, xs, Y, N, 2 * jp + 1);
            if (mt > 1) { _tile_stored(2, buf, 64); post_b(1, r1, buf, ws, xs, Y, N, 2 * jp);
                          _tile_stored(3, buf, 64); post_b(1, r1, buf, ws, xs, Y, N, 2 * jp + 1); }
        } else {  // mt == 3: tiles C 0..2, A 3..5, B 6; the pair's two members one after the other (second from L2)
            for (int b = 0; b < 2; b++) {
                const int8_t *q = p + b * 1024;
                _tile_zero(0); _tile_zero(1); _tile_zero(2);
                for (int c = 0; c < KC; c++, q += 2048) {
                    if (b == 0) PREF(q);
                    _tile_loadd(6, q, 64);
                    _tile_loadd(3, X + (size_t)c * 1024, 64); _tile_dpbssd(0, 3, 6);
                    _tile_loadd(4, X1 + (size_t)c * 1024, 64); _tile_dpbssd(1, 4, 6);
                    _tile_loadd(5, X2 + (size_t)c * 1024, 64); _tile_dpbssd(2, 5, 6);
                }
                _tile_stored(0, buf, 64); post_b(0, 16, buf, ws, xs, Y, N, 2 * jp + b);
                _tile_stored(1, buf, 64); post_b(1, 16, buf, ws, xs, Y, N, 2 * jp + b);
                _tile_stored(2, buf, 64); post_b(2, r2, buf, ws, xs, Y, N, 2 * jp + b);
            }
        }
    }
}
static inline void post_a(int rows, const int32_t *buf, __m512 sw, const float *xs, int KB, int kb, __m512 *acc) {
    for (int m = 0; m < rows; m++) {
        __m512 v = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_load_si512(buf + m * 16)), sw);
        acc[m] = _mm512_fmadd_ps(v, _mm512_set1_ps(xs[m * KB + kb]), acc[m]);
    }
}
static void kern_a(const uint8_t *W, int K, int j0, int j1, const int8_t *X, const float *xs, int R, float *Y, int N) {
    // j0, j1: n-tile pairs. X packed [mt][kb][16][32].
    int mt = (R + 15) / 16, KB = K / 32;
    int32_t buf[4][16 * 16] __attribute__((aligned(64)));
    __m512 acc[2][48];
    const int8_t *X1 = X + (size_t)KB * 512, *X2 = X + (size_t)2 * KB * 512;
    int r0 = mt == 1 ? R : 16, r1 = mt == 2 ? R - 16 : 16, r2 = R - 32;
    for (int jp = j0; jp < j1; jp++) {
        const uint8_t *p = W + (size_t)jp * KB * 1088;
        if (mt <= 2) {  // C 0..3 (t*2+member), A 4,5 (rows x 32 B), B 6,7 (8 x 64 B)
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
            for (int m = 0; m < R; m++) { _mm512_storeu_ps(Y + (size_t)m * N + jp * 32, acc[0][m]); _mm512_storeu_ps(Y + (size_t)m * N + jp * 32 + 16, acc[1][m]); }
        } else {  // mt == 3: C 0..2, A 3..5, B 6; members one after the other
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
                for (int m = 0; m < R; m++) _mm512_storeu_ps(Y + (size_t)m * N + jp * 32 + b * 16, acc[0][m]);
            }
        }
    }
}

// ---------------- thread pool with spin barrier ----------------
typedef struct { const void *W; const float *ws; int N, K; } mat_t;
static int T; static pthread_t th[16];
static atomic_int go_gen, done_cnt;
static struct { const mat_t *mats; int nm, variant, R, reps; const int8_t *X; const float *xs; float *Y; } job;
static atomic_int bar_cnt, bar_gen;
static void barrier(void) {
    int g = atomic_load(&bar_gen);
    if (atomic_fetch_add(&bar_cnt, 1) == T - 1) { atomic_store(&bar_cnt, 0); atomic_fetch_add(&bar_gen, 1); }
    else while (atomic_load(&bar_gen) == g) _mm_pause();
}
static void work(int id) {
    config(job.R, job.variant);
    for (int rep = 0; rep < job.reps; rep++)
        for (int i = 0; i < job.nm; i++) {
            const mat_t *m = &job.mats[i]; int NT = m->N / 32;
            int j0 = (int)((long)NT * id / T), j1 = (int)((long)NT * (id + 1) / T);
            if (job.variant) kern_b(m->W, m->ws, m->K, j0, j1, job.X, job.xs, job.R, job.Y, m->N);
            else kern_a(m->W, m->K, j0, j1, job.X, job.xs, job.R, job.Y, m->N);
            barrier();
        }
}
static void *worker(void *a) {
    long id = (long)a; syscall(SYS_arch_prctl, 0x1023, 18);
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(id, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    int seen = 0;
    for (;;) { while (atomic_load(&go_gen) == seen) _mm_pause(); seen = atomic_load(&go_gen); work((int)id); atomic_fetch_add(&done_cnt, 1); }
    return 0;
}
int row1_init(int threads) {
    if (syscall(SYS_arch_prctl, 0x1023, 18)) return -1;
    T = threads; char *e = getenv("ROW1_PF"); PF = e ? atoi(e) : 0;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    for (long i = 1; i < T; i++) pthread_create(&th[i], 0, worker, (void *)i);
    return 0;
}
// Runs `reps` passes over nm matrices; returns seconds. The caller thread is worker 0.
double row1_run(const mat_t *mats, int nm, int variant, int R, const int8_t *X, const float *xs, float *Y, int reps) {
    job.mats = mats; job.nm = nm; job.variant = variant; job.R = R; job.X = X; job.xs = xs; job.Y = Y; job.reps = reps;
    atomic_store(&done_cnt, 0);
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    atomic_fetch_add(&go_gen, 1);
    work(0);
    while (atomic_load(&done_cnt) < T - 1) _mm_pause();
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
}
