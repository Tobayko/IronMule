// FA1 (2026-11-13): a decode-and-prefill flash attention for f16 K/V on AVX-512, included into
// ggml/src/ggml-cpu/ops.cpp in front of ggml_compute_forward_flash_attn_ext_f16, which calls fa1_attn first.
//
// ggml's CPU kernel walks the keys one at a time: one f16 dot product, one expf, one f16 multiply-add of V per key
// and query head, and it accumulates V*softmax in f16. With grouped-query attention every key/value head is read
// again for each query head that shares it. FA1 computes the same attention (scale, logit softcap, mask, online
// softmax) with these changes:
//   * the rk2 query heads that share a KV head are done together, so each K and V row is converted once per block;
//   * keys go in blocks of FA1_BLOCK: scores for the block, one running-max update and one rescale per block, the
//     exponentials vectorized (ggml_v_expf);
//   * V*p is accumulated in f32 (ggml's path accumulates it in f16), the dot products in f32.
// Results differ from ggml's in rounding only (order of summation, f32 instead of f16 accumulation), so it serves
// behind the quality gate like every other path that may change tokens. FA1=0 in the environment turns it off.
// Taken only for: f32 Q, f16 K and V, head sizes a multiple of 64 up to 256, no ALiBi (max_bias 0), no sinks,
// the mask (if any) shared by all heads, and no broadcast over the fourth dimension.
#pragma once
#if defined(__AVX512F__)
#include <immintrin.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>

#define FA1_BLOCK 32


static int fa1_enabled() {
    static int e = -1;
    if (e < 0) { const char * s = getenv("FA1"); e = s ? atoi(s) : 1; }
    return e;
}

// The engine asks the library for this to know FA1 and FA1u are there (ironmule.ggml_engine.fa1_version); a unified KV
// cache serves only with them, since llama.cpp's own CPU kernel scans every cell of the pool for every query.
extern "C" __attribute__((visibility("default"))) int ironmule_fa1_version(void) {
    const char * u = getenv("FA1U");
    return fa1_enabled() && !(u && atoi(u) == 0) ? 3 : 0;
}

static inline float fa1_hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

// The 16 horizontal sums of v[0..15] in one vector, in order (FA1 v2). Two 256-bit and one 128-bit fold across
// vectors, then a 4x4 transpose-add inside each 128-bit lane; the inputs are taken as v0-3, v8-11, v4-7, v12-15 so the
// sums come out as 0..15.
static inline __m512 fa1_hsum16(const __m512 * v) {
    static const int order[16] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};
    __m512 w[8], x[4];
    for (int i = 0; i < 8; ++i) {
        const __m512 a = v[order[i]], b = v[order[i + 8]];
        w[i] = _mm512_add_ps(_mm512_shuffle_f32x4(a, b, 0x44), _mm512_shuffle_f32x4(a, b, 0xEE));
    }
    for (int i = 0; i < 4; ++i) {
        x[i] = _mm512_add_ps(_mm512_shuffle_f32x4(w[i], w[i + 4], 0x88), _mm512_shuffle_f32x4(w[i], w[i + 4], 0xDD));
    }
    const __m512 y0 = _mm512_add_ps(_mm512_unpacklo_ps(x[0], x[1]), _mm512_unpackhi_ps(x[0], x[1]));
    const __m512 y1 = _mm512_add_ps(_mm512_unpacklo_ps(x[2], x[3]), _mm512_unpackhi_ps(x[2], x[3]));
    return _mm512_add_ps(_mm512_shuffle_ps(y0, y1, 0x44), _mm512_shuffle_ps(y0, y1, 0xEE));
}

static bool fa1_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    if (!fa1_enabled() || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || sinks) {
        return false;
    }
    const int64_t DK = k->ne[0], DV = v->ne[0];
    if (DK % 64 || DV % 64 || DK > 256 || DV > 256) return false;
    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f) return false;
    if (softcap != 0.0f) scale /= softcap;
    const int64_t neq1 = q->ne[1], neq2 = q->ne[2], neq3 = q->ne[3];
    const int64_t nek1 = k->ne[1], nek2 = k->ne[2], nek3 = k->ne[3];
    if (v->ne[2] != nek2 || v->ne[3] != nek3 || neq3 != nek3 || neq2 % nek2) return false;
    const int64_t rk2 = neq2 / nek2;
    if (rk2 > 8) return false;
    if (mask && mask->ne[2] != 1) return false;
    if (q->nb[0] != sizeof(float) || k->nb[0] != 2 || v->nb[0] != 2 || dst->nb[0] != sizeof(float)) return false;
    const int64_t ne1 = dst->ne[1], ne2 = dst->ne[2];

    const int ith = params->ith, nth = params->nth;
    const int64_t units = neq1 * nek2 * neq3;  // (query position, KV head, stream)
    const int64_t per = (units + nth - 1) / nth;
    const int64_t u0 = per * ith, u1 = std::min(units, u0 + per);

    alignas(64) float Q[8][256];
    alignas(64) float A[8][256];
    alignas(64) float P[8][FA1_BLOCK];
    alignas(64) float KF[FA1_BLOCK][256];
    float M[8], S[8];
    const int nd = (int) (DK / 16), nv = (int) (DV / 16);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t iq1 = u % neq1, g = (u / neq1) % nek2, i3 = u / (neq1 * nek2);
        for (int h = 0; h < rk2; ++h) {
            const float * pq = (const float *) ((const char *) q->data + iq1 * q->nb[1] + (g * rk2 + h) * q->nb[2] + i3 * q->nb[3]);
            memcpy(Q[h], pq, DK * sizeof(float));
            memset(A[h], 0, DV * sizeof(float));
            M[h] = -INFINITY; S[h] = 0.0f;
        }
        const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + iq1 * mask->nb[1] + (i3 % mask->ne[3]) * mask->nb[3]) : nullptr;
        const char * kb = (const char *) k->data + g * k->nb[2] + i3 * k->nb[3];
        const char * vb = (const char *) v->data + g * v->nb[2] + i3 * v->nb[3];

        for (int64_t c0 = 0; c0 < nek1; c0 += FA1_BLOCK) {
            const int nb = (int) std::min<int64_t>(FA1_BLOCK, nek1 - c0);
            float mval[FA1_BLOCK];
            int live = 0;
            for (int j = 0; j < nb; ++j) {
                mval[j] = mp ? GGML_CPU_FP16_TO_FP32(mp[c0 + j]) : 0.0f;
                live += mval[j] != -INFINITY;
            }
            if (!live) continue;
            // scores of the block for every query head of the group: the block's K rows converted to f32 once, then
            // per head 16 keys at a time in 16 accumulators, summed across lanes together (fa1_hsum16)
            for (int j = 0; j < nb; ++j) {
                const ggml_fp16_t * kr = (const ggml_fp16_t *) (kb + (c0 + j) * k->nb[1]);
                for (int d = 0; d < nd; ++d) {
                    _mm512_store_ps(KF[j] + 16 * d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (kr + 16 * d))));
                }
            }
            for (int j = nb; j < FA1_BLOCK; ++j) memset(KF[j], 0, DK * sizeof(float));
            for (int h = 0; h < rk2; ++h) {
                for (int j0 = 0; j0 < FA1_BLOCK; j0 += 16) {
                    __m512 acc[16];
                    for (int j = 0; j < 16; ++j) acc[j] = _mm512_setzero_ps();
                    for (int d = 0; d < nd; ++d) {
                        const __m512 qd = _mm512_load_ps(Q[h] + 16 * d);
                        for (int j = 0; j < 16; ++j) acc[j] = _mm512_fmadd_ps(_mm512_load_ps(KF[j0 + j] + 16 * d), qd, acc[j]);
                    }
                    _mm512_store_ps(P[h] + j0, _mm512_mul_ps(fa1_hsum16(acc), _mm512_set1_ps(scale)));
                }
                for (int j = 0; j < nb; ++j) {
                    if (mval[j] == -INFINITY) { P[h][j] = -INFINITY; continue; }
                    float s = P[h][j];
                    if (softcap != 0.0f) s = softcap * tanhf(s);
                    P[h][j] = s + mval[j];
                }
            }
            for (int j = nb; j < FA1_BLOCK; ++j) for (int h = 0; h < rk2; ++h) P[h][j] = -INFINITY;
            // one running-max update and rescale per block; exponentials vectorized
            for (int h = 0; h < rk2; ++h) {
                float bmax = -INFINITY;
                for (int j = 0; j < nb; ++j) bmax = std::max(bmax, P[h][j]);
                const float Mn = std::max(M[h], bmax);
                if (Mn == -INFINITY) { for (int j = 0; j < FA1_BLOCK; ++j) P[h][j] = 0.0f; continue; }
                const float ms = M[h] == -INFINITY ? 0.0f : expf(M[h] - Mn);
                if (ms != 1.0f) {
                    const __m512 vms = _mm512_set1_ps(ms);
                    for (int d = 0; d < nv; ++d) _mm512_store_ps(A[h] + 16 * d, _mm512_mul_ps(_mm512_load_ps(A[h] + 16 * d), vms));
                }
                const __m512 vM = _mm512_set1_ps(Mn);
                float sum = 0.0f;
                for (int j = 0; j < FA1_BLOCK; j += 16) {
                    __m512 x = _mm512_sub_ps(_mm512_load_ps(P[h] + j), vM);
                    const __mmask16 ok = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-INFINITY), _CMP_NEQ_OQ);
                    __m512 e = _mm512_maskz_mov_ps(ok, ggml_v_expf(_mm512_maskz_mov_ps(ok, x)));
                    _mm512_store_ps(P[h] + j, e);
                    sum += _mm512_reduce_add_ps(e);
                }
                S[h] = S[h] * ms + sum;
                M[h] = Mn;
            }
            // V * p, accumulated in f32, V rows converted once for the whole group
            for (int d0 = 0; d0 < nv; d0 += 4) {
                const int dn = std::min(4, nv - d0);
                for (int h0 = 0; h0 < rk2; h0 += 4) {
                    const int hn = (int) std::min<int64_t>(4, rk2 - h0);
                    __m512 acc[4][4];
                    for (int h = 0; h < hn; ++h) for (int d = 0; d < dn; ++d) acc[h][d] = _mm512_load_ps(A[h0 + h] + 16 * (d0 + d));
                    for (int j = 0; j < nb; ++j) {
                        if (mval[j] == -INFINITY) continue;
                        const ggml_fp16_t * vr = (const ggml_fp16_t *) (vb + (c0 + j) * v->nb[1]);
                        __m512 vd[4];
                        for (int d = 0; d < dn; ++d) vd[d] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (vr + 16 * (d0 + d))));
                        for (int h = 0; h < hn; ++h) {
                            const __m512 p = _mm512_set1_ps(P[h0 + h][j]);
                            for (int d = 0; d < dn; ++d) acc[h][d] = _mm512_fmadd_ps(vd[d], p, acc[h][d]);
                        }
                    }
                    for (int h = 0; h < hn; ++h) for (int d = 0; d < dn; ++d) _mm512_store_ps(A[h0 + h] + 16 * (d0 + d), acc[h][d]);
                }
            }
        }
        for (int h = 0; h < rk2; ++h) {
            const float inv = S[h] > 0.0f ? 1.0f / S[h] : 0.0f;
            const int64_t i2 = g * rk2 + h;
            float * out = (float *) ((char *) dst->data + (i3 * ne2 * ne1 + i2 + iq1 * ne1) * dst->nb[1]);
            for (int64_t d = 0; d < DV; ++d) out[d] = A[h][d] * inv;
        }
    }
    return true;
}

// FA1u (2026-11-13): several query tokens of one stream together -- a unified KV cache, where all sequences share one
// pool of cells, or a prompt. One unit is a KV head and up to FA1U_TOK tokens (all their query heads of that group): it
// walks the key blocks once, converting each block's K and V rows to f32 once for all its rows, so a key block that
// many sequences can see (a shared prompt prefix) is read once per step instead of once per sequence. Per row and block
// it scores 16 keys at a time where most are visible to the row and key by key where few are (another sequence's cells
// in the same block), then one online-softmax update and V*p accumulation in f32, as FA1. Same results as FA1 per row
// up to summation order. FA1=0 turns it off together with FA1; FA1U=0 alone.
#define FA1U_TOK 16

static int fa1u_enabled() {
    static int e = -1;
    if (e < 0) { const char * s = getenv("FA1U"); e = s ? atoi(s) : 1; }
    return e;
}

static bool fa1u_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    if (!fa1_enabled() || !fa1u_enabled() || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || sinks) {
        return false;
    }
    const int64_t DK = k->ne[0], DV = v->ne[0];
    if (DK % 64 || DV % 64 || DK > 256 || DV > 256) return false;
    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f) return false;
    if (softcap != 0.0f) scale /= softcap;
    const int64_t neq1 = q->ne[1], neq2 = q->ne[2], neq3 = q->ne[3];
    const int64_t nek1 = k->ne[1], nek2 = k->ne[2], nek3 = k->ne[3];
    if (neq1 < 2 || v->ne[2] != nek2 || v->ne[3] != nek3 || neq3 != nek3 || neq2 % nek2) return false;
    const int64_t rk2 = neq2 / nek2;
    if (rk2 > 8) return false;
    if (mask && mask->ne[2] != 1) return false;
    if (q->nb[0] != sizeof(float) || k->nb[0] != 2 || v->nb[0] != 2 || dst->nb[0] != sizeof(float)) return false;
    const int64_t ne1 = dst->ne[1], ne2 = dst->ne[2];

    const int64_t chunks = (neq1 + FA1U_TOK - 1) / FA1U_TOK;
    const int64_t units = chunks * nek2 * neq3;
    const int ith = params->ith, nth = params->nth;
    const int64_t per = (units + nth - 1) / nth;
    const int64_t u0 = per * ith, u1 = std::min(units, u0 + per);
    const int nd = (int) (DK / 16), nv = (int) (DV / 16);
    const int maxrows = FA1U_TOK * 8;

    static thread_local std::vector<float> buf;
    const size_t need = (size_t) maxrows * (256 + 256 + FA1_BLOCK) + 2 * FA1_BLOCK * 256 + 16 * 4;
    if (buf.size() < need) buf.resize(need);
    float * base = (float *) (((uintptr_t) buf.data() + 63) & ~(uintptr_t) 63);
    float * Q = base;                                   // [rows][256]
    float * A = Q + (size_t) maxrows * 256;             // [rows][256]
    float * P = A + (size_t) maxrows * 256;             // [rows][FA1_BLOCK]
    float * KF = P + (size_t) maxrows * FA1_BLOCK;      // [FA1_BLOCK][256]
    float * VF = KF + FA1_BLOCK * 256;                  // [FA1_BLOCK][256]
    float M[FA1U_TOK * 8], S[FA1U_TOK * 8];
    float mval[FA1U_TOK][FA1_BLOCK];

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t ch = u % chunks, g = (u / chunks) % nek2, i3 = u / (chunks * nek2);
        const int64_t t0 = ch * FA1U_TOK;
        const int nt = (int) std::min<int64_t>(FA1U_TOK, neq1 - t0);
        const int R = nt * (int) rk2;
        for (int t = 0; t < nt; ++t) {
            for (int h = 0; h < rk2; ++h) {
                const int r = t * (int) rk2 + h;
                const float * pq = (const float *) ((const char *) q->data + (t0 + t) * q->nb[1] + (g * rk2 + h) * q->nb[2] + i3 * q->nb[3]);
                memcpy(Q + (size_t) r * 256, pq, DK * sizeof(float));
                memset(A + (size_t) r * 256, 0, DV * sizeof(float));
                M[r] = -INFINITY; S[r] = 0.0f;
            }
        }
        const char * kb = (const char *) k->data + g * k->nb[2] + i3 * k->nb[3];
        const char * vb = (const char *) v->data + g * v->nb[2] + i3 * v->nb[3];

        for (int64_t c0 = 0; c0 < nek1; c0 += FA1_BLOCK) {
            const int nb = (int) std::min<int64_t>(FA1_BLOCK, nek1 - c0);
            int seen[FA1_BLOCK] = {0};
            int live_tok[FA1U_TOK];
            int any = 0;
            for (int t = 0; t < nt; ++t) {
                const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + (t0 + t) * mask->nb[1] + (i3 % mask->ne[3]) * mask->nb[3]) : nullptr;
                live_tok[t] = 0;
                for (int j = 0; j < FA1_BLOCK; ++j) {
                    const float m = (j < nb) ? (mp ? GGML_CPU_FP16_TO_FP32(mp[c0 + j]) : 0.0f) : -INFINITY;
                    mval[t][j] = m;
                    if (m != -INFINITY) { live_tok[t]++; seen[j] = 1; }
                }
                any += live_tok[t];
            }
            if (!any) continue;
            // the block's K and V rows that anyone sees, converted once for every row of the unit
            for (int j = 0; j < nb; ++j) {
                if (!seen[j]) continue;
                const ggml_fp16_t * kr = (const ggml_fp16_t *) (kb + (c0 + j) * k->nb[1]);
                const ggml_fp16_t * vr = (const ggml_fp16_t *) (vb + (c0 + j) * v->nb[1]);
                for (int d = 0; d < nd; ++d) _mm512_store_ps(KF + j * 256 + 16 * d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (kr + 16 * d))));
                for (int d = 0; d < nv; ++d) _mm512_store_ps(VF + j * 256 + 16 * d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (vr + 16 * d))));
            }
            for (int j = 0; j < FA1_BLOCK; ++j) if (j >= nb || !seen[j]) memset(KF + j * 256, 0, DK * sizeof(float));
            for (int t = 0; t < nt; ++t) {
                if (!live_tok[t]) {
                    for (int h = 0; h < rk2; ++h) { float * pr = P + (size_t) (t * rk2 + h) * FA1_BLOCK; for (int j = 0; j < FA1_BLOCK; ++j) pr[j] = 0.0f; }
                    continue;
                }
                for (int h = 0; h < rk2; ++h) {
                    const int r = t * (int) rk2 + h;
                    const float * qr = Q + (size_t) r * 256;
                    float * pr = P + (size_t) r * FA1_BLOCK;
                    if (live_tok[t] > 8) {  // most keys visible: 16 at a time
                        for (int j0 = 0; j0 < FA1_BLOCK; j0 += 16) {
                            __m512 acc[16];
                            for (int j = 0; j < 16; ++j) acc[j] = _mm512_setzero_ps();
                            for (int d = 0; d < nd; ++d) {
                                const __m512 qd = _mm512_load_ps(qr + 16 * d);
                                for (int j = 0; j < 16; ++j) acc[j] = _mm512_fmadd_ps(_mm512_load_ps(KF + (j0 + j) * 256 + 16 * d), qd, acc[j]);
                            }
                            _mm512_storeu_ps(pr + j0, _mm512_mul_ps(fa1_hsum16(acc), _mm512_set1_ps(scale)));
                        }
                    } else {  // few visible (other sequences' cells): key by key
                        for (int j = 0; j < FA1_BLOCK; ++j) {
                            if (mval[t][j] == -INFINITY) continue;
                            __m512 acc = _mm512_setzero_ps();
                            for (int d = 0; d < nd; ++d) acc = _mm512_fmadd_ps(_mm512_load_ps(KF + j * 256 + 16 * d), _mm512_load_ps(qr + 16 * d), acc);
                            pr[j] = _mm512_reduce_add_ps(acc) * scale;
                        }
                    }
                    float bmax = -INFINITY;
                    for (int j = 0; j < FA1_BLOCK; ++j) {
                        if (mval[t][j] == -INFINITY) { pr[j] = -INFINITY; continue; }
                        float sc = pr[j];
                        if (softcap != 0.0f) sc = softcap * tanhf(sc);
                        pr[j] = sc + mval[t][j];
                        bmax = std::max(bmax, pr[j]);
                    }
                    const float Mn = std::max(M[r], bmax);
                    if (Mn == -INFINITY) { for (int j = 0; j < FA1_BLOCK; ++j) pr[j] = 0.0f; continue; }
                    const float ms = M[r] == -INFINITY ? 0.0f : expf(M[r] - Mn);
                    float * ar = A + (size_t) r * 256;
                    if (ms != 1.0f) {
                        const __m512 vms = _mm512_set1_ps(ms);
                        for (int d = 0; d < nv; ++d) _mm512_store_ps(ar + 16 * d, _mm512_mul_ps(_mm512_load_ps(ar + 16 * d), vms));
                    }
                    const __m512 vM = _mm512_set1_ps(Mn);
                    float sum = 0.0f;
                    for (int j = 0; j < FA1_BLOCK; j += 16) {
                        __m512 x = _mm512_sub_ps(_mm512_loadu_ps(pr + j), vM);
                        const __mmask16 ok = _mm512_cmp_ps_mask(x, _mm512_set1_ps(-INFINITY), _CMP_NEQ_OQ);
                        __m512 e = _mm512_maskz_mov_ps(ok, ggml_v_expf(_mm512_maskz_mov_ps(ok, x)));
                        _mm512_storeu_ps(pr + j, e);
                        sum += _mm512_reduce_add_ps(e);
                    }
                    S[r] = S[r] * ms + sum;
                    M[r] = Mn;
                    // V * p for this row: the visible keys only
                    for (int d0 = 0; d0 < nv; d0 += 4) {
                        const int dn = std::min(4, nv - d0);
                        __m512 acc[4];
                        for (int d = 0; d < dn; ++d) acc[d] = _mm512_load_ps(ar + 16 * (d0 + d));
                        for (int j = 0; j < FA1_BLOCK; ++j) {
                            if (mval[t][j] == -INFINITY) continue;
                            const __m512 pj = _mm512_set1_ps(pr[j]);
                            for (int d = 0; d < dn; ++d) acc[d] = _mm512_fmadd_ps(_mm512_load_ps(VF + j * 256 + 16 * (d0 + d)), pj, acc[d]);
                        }
                        for (int d = 0; d < dn; ++d) _mm512_store_ps(ar + 16 * (d0 + d), acc[d]);
                    }
                }
            }
        }
        for (int t = 0; t < nt; ++t) {
            for (int h = 0; h < rk2; ++h) {
                const int r = t * (int) rk2 + h;
                const float inv = S[r] > 0.0f ? 1.0f / S[r] : 0.0f;
                const int64_t i2 = g * rk2 + h, i1 = t0 + t;
                float * out = (float *) ((char *) dst->data + (i3 * ne2 * ne1 + i2 + i1 * ne1) * dst->nb[1]);
                const float * ar = A + (size_t) r * 256;
                for (int64_t d = 0; d < DV; ++d) out[d] = ar[d] * inv;
            }
        }
    }
    return true;
}

// FA1x (2026-11-13): FA1u's attention on the AMX matrix unit. The roofline (docs/LAB.md, RL-MATH) puts a chat step's
// attention on the AVX-512 vector units at about a third of their FP32 peak while the AMX unit, ten times faster in
// bf16, idles. FA1x computes S = Q K^T and O += P V with AMX bf16 tiles and fp32 accumulation: Q, K, the softmax's P and
// V are rounded to bf16 (8 bits of mantissa) before each product; the scores, the running max, the sum and O stay
// fp32. The block's K and V rows are converted to bf16 in the tile (VNNI) layout once for all rows of the unit. Masking
// and the online softmax are FA1u's. The tile configuration of the calling thread (ggml's AMX matmul keeps its own) is
// saved and restored. FA1X=0 turns it off; it serves behind the quality gate like FA1u.
#if defined(__AMX_TILE__) && defined(__AMX_BF16__) && defined(__AVX512BF16__)
#include <unistd.h>
#include <sys/syscall.h>

struct fa1x_cfg { uint8_t pal, start; uint8_t r[14]; uint16_t cb[16]; uint8_t rows[16]; } __attribute__((packed));

static int fa1x_enabled() {
    static int e = -1;
    if (e < 0) {
        const char * s = getenv("FA1X");
        e = s ? atoi(s) : 1;
        if (e && syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XTILEDATA */) != 0) e = 0;
    }
    return e;
}

static inline uint16_t fa1x_bf16(float x) {  // round to nearest even, as _mm512_cvtneps_pbh
    uint32_t u; memcpy(&u, &x, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}

// x = hi + lo, both bf16 (hi = bf16(x), lo = bf16(x - hi)): an f16 value is exact in hi + lo (8 + 8 bits of mantissa
// against f16's 11), an f32 value to about 16 bits. Products then take three bf16 passes, hi*hi + hi*lo + lo*hi, the
// lo*lo term (about 2^-16 relative) dropped: the error-compensated split of the Ozaki scheme, on AMX.
static inline void fa1x_split(__m512 x, __m256i * hi, __m256i * lo) {
    const __m256i h = (__m256i) _mm512_cvtneps_pbh(x);
    const __m512 back = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
    *hi = h;
    *lo = (__m256i) _mm512_cvtneps_pbh(_mm512_sub_ps(x, back));
}

#define FA1X_TOK 32
#define FA1X_DENSE 480  // visible (row, key) pairs from which a block goes to AMX (cost model in docs/LAB.md, RL-MATH)
static int64_t fa1x_dense() {
    static int64_t d = -1;
    if (d < 0) { const char * s = getenv("FA1X_DENSE"); d = s ? atoll(s) : FA1X_DENSE; }
    return d;
}

static bool fa1x_attn(const ggml_compute_params * params, ggml_tensor * dst) {
    // v2 orientation: S^T = K Q^T (K in its own row layout as the A tile; Q^T in VNNI pairs once per unit), the softmax
    // down the columns of S^T, 16 rows per vector; P gathered to rows; O += P V (V rows interleaved in pairs per block).
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    if (!fa1_enabled() || !fa1u_enabled() || !fa1x_enabled() || q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || sinks) {
        return false;
    }
    const int64_t D = k->ne[0];
    if (v->ne[0] != D || D % 64 || D > 256) return false;
    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f) return false;
    if (softcap != 0.0f) scale /= softcap;
    const int64_t neq1 = q->ne[1], neq2 = q->ne[2], neq3 = q->ne[3];
    const int64_t nek1 = k->ne[1], nek2 = k->ne[2], nek3 = k->ne[3];
    if (neq1 < 2 || v->ne[2] != nek2 || v->ne[3] != nek3 || neq3 != nek3 || neq2 % nek2) return false;
    const int64_t rk2 = neq2 / nek2;
    if (rk2 > 8) return false;
    if (mask && mask->ne[2] != 1) return false;
    if (q->nb[0] != sizeof(float) || k->nb[0] != 2 || v->nb[0] != 2 || dst->nb[0] != sizeof(float)) return false;
    const int64_t ne1 = dst->ne[1], ne2 = dst->ne[2];

    const int64_t chunks = (neq1 + FA1X_TOK - 1) / FA1X_TOK;
    const int64_t units = chunks * nek2 * neq3;
    const int ith = params->ith, nth = params->nth;
    const int64_t per = (units + nth - 1) / nth;
    const int64_t u0 = per * ith, u1 = std::min(units, u0 + per);
    if (u0 >= u1) return true;
    const int64_t RMAX = FA1X_TOK * 8;

    static thread_local std::vector<uint8_t> buf;
    const size_t need = 2 * (size_t) (D / 2) * RMAX * 4      // Q^T hi, lo: [D/2][RMAX] pairs
                        + (size_t) RMAX * D * 4                // O
                        + (size_t) 32 * RMAX * 4               // S^T, then P^T
                        + (size_t) 32 * RMAX * 4               // the mask, expanded per row
                        + 2 * (size_t) RMAX * 32 * 2           // P hi, lo
                        + 4 * (size_t) 32 * D * 2              // K hi, lo; V pairs hi, lo
                        + 64 * 8;
    if (buf.size() < need) buf.resize(need);
    uint8_t * p8 = (uint8_t *) (((uintptr_t) buf.data() + 63) & ~(uintptr_t) 63);
    uint32_t * QT = (uint32_t *) p8;  p8 += (size_t) (D / 2) * RMAX * 4;
    uint32_t * QTL = (uint32_t *) p8; p8 += (size_t) (D / 2) * RMAX * 4;
    float * O = (float *) p8;         p8 += (size_t) RMAX * D * 4;
    float * ST = (float *) p8;        p8 += (size_t) 32 * RMAX * 4;
    float * MT = (float *) p8;        p8 += (size_t) 32 * RMAX * 4;
    uint16_t * PB = (uint16_t *) p8;  p8 += (size_t) RMAX * 32 * 2;
    uint16_t * PL = (uint16_t *) p8;  p8 += (size_t) RMAX * 32 * 2;
    uint16_t * KA = (uint16_t *) p8;  p8 += (size_t) 32 * D * 2;
    uint16_t * KAL = (uint16_t *) p8; p8 += (size_t) 32 * D * 2;
    uint16_t * VV = (uint16_t *) p8;  p8 += (size_t) 32 * D * 2;
    uint16_t * VL = (uint16_t *) p8;
    alignas(64) float Mv[FA1X_TOK * 8], Sv[FA1X_TOK * 8], msv[FA1X_TOK * 8];
    float mval[FA1X_TOK][FA1_BLOCK];
    alignas(64) uint16_t ilo[32], ihi[32];
    for (int i = 0; i < 16; ++i) { ilo[2 * i] = i; ilo[2 * i + 1] = 32 + i; ihi[2 * i] = 16 + i; ihi[2 * i + 1] = 48 + i; }
    const __m512i idx_lo = _mm512_loadu_si512(ilo), idx_hi = _mm512_loadu_si512(ihi);
    alignas(64) int gidx[16];

    fa1x_cfg saved, cfg;
    _tile_storeconfig(&saved);
    memset(&cfg, 0, sizeof cfg);
    cfg.pal = 1;
    for (int t = 0; t < 8; ++t) { cfg.rows[t] = 16; cfg.cb[t] = 64; }
    _tile_loadconfig(&cfg);

    for (int64_t u = u0; u < u1; ++u) {
        const int64_t ch = u % chunks, g = (u / chunks) % nek2, i3 = u / (chunks * nek2);
        const int64_t t0 = ch * FA1X_TOK;
        const int nt = (int) std::min<int64_t>(FA1X_TOK, neq1 - t0);
        const int R = nt * (int) rk2;
        const int64_t RP = (R + 15) / 16 * 16;
        for (int i = 0; i < 16; ++i) gidx[i] = (int) (i * RP);
        const __m512i gv = _mm512_load_si512(gidx);
        // Q^T in VNNI pairs: QT[p][r] = (Q[r][2p], Q[r][2p+1]), hi and lo
        for (int64_t r = 0; r < RP; ++r) {
            alignas(64) uint16_t qh[256], ql[256];
            if (r < R) {
                const int t = (int) r / (int) rk2, h = (int) r % (int) rk2;
                const float * pq = (const float *) ((const char *) q->data + (t0 + t) * q->nb[1] + (g * rk2 + h) * q->nb[2] + i3 * q->nb[3]);
                for (int64_t d = 0; d < D; d += 16) {
                    __m256i hi, lo;
                    fa1x_split(_mm512_loadu_ps(pq + d), &hi, &lo);
                    _mm256_store_si256((__m256i *) (qh + d), hi);
                    _mm256_store_si256((__m256i *) (ql + d), lo);
                }
            } else {
                memset(qh, 0, D * 2); memset(ql, 0, D * 2);
            }
            for (int64_t pr = 0; pr < D / 2; ++pr) {
                QT[pr * RP + r] = ((const uint32_t *) qh)[pr];
                QTL[pr * RP + r] = ((const uint32_t *) ql)[pr];
            }
            memset(O + r * D, 0, D * 4);
            Mv[r] = -INFINITY; Sv[r] = 0.0f;
        }
        const char * kb = (const char *) k->data + g * k->nb[2] + i3 * k->nb[3];
        const char * vb = (const char *) v->data + g * v->nb[2] + i3 * v->nb[3];

        for (int64_t c0 = 0; c0 < nek1; c0 += FA1_BLOCK) {
            const int nb = (int) std::min<int64_t>(FA1_BLOCK, nek1 - c0);
            int any = 0, all = (nb == FA1_BLOCK), live_tok[FA1X_TOK], seen[FA1_BLOCK] = {0};
            for (int t = 0; t < nt; ++t) {
                const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + (t0 + t) * mask->nb[1] + (i3 % mask->ne[3]) * mask->nb[3]) : nullptr;
                live_tok[t] = 0;
                for (int j = 0; j < FA1_BLOCK; ++j) {
                    const float m = (j < nb) ? (mp ? GGML_CPU_FP16_TO_FP32(mp[c0 + j]) : 0.0f) : -INFINITY;
                    mval[t][j] = m;
                    if (m != -INFINITY) { live_tok[t]++; seen[j] = 1; }
                    all &= m == 0.0f;
                }
                any += live_tok[t];
            }
            if (!any) continue;
            if ((int64_t) any * rk2 < fa1x_dense()) {
                // a sparse block (other sequences' cells of a unified pool): FA1u's way, the visible pairs only, fp32
                float * KF = ST, * VF = MT;  // free in a sparse block: [32][D] each
                for (int j = 0; j < nb; ++j) {
                    if (!seen[j]) continue;
                    const ggml_fp16_t * kr = (const ggml_fp16_t *) (kb + (c0 + j) * k->nb[1]);
                    const ggml_fp16_t * vr = (const ggml_fp16_t *) (vb + (c0 + j) * v->nb[1]);
                    for (int64_t d = 0; d < D; d += 16) {
                        _mm512_storeu_ps(KF + j * D + d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (kr + d))));
                        _mm512_storeu_ps(VF + j * D + d, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (vr + d))));
                    }
                }
                for (int t = 0; t < nt; ++t) {
                    if (!live_tok[t]) continue;
                    for (int h = 0; h < rk2; ++h) {
                        const int r = t * (int) rk2 + h;
                        const float * pq = (const float *) ((const char *) q->data + (t0 + t) * q->nb[1] + (g * rk2 + h) * q->nb[2] + i3 * q->nb[3]);
                        float sc[FA1_BLOCK], bmax = -INFINITY;
                        for (int j = 0; j < FA1_BLOCK; ++j) {
                            if (mval[t][j] == -INFINITY) { sc[j] = -INFINITY; continue; }
                            __m512 acc = _mm512_setzero_ps();
                            for (int64_t d = 0; d < D; d += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(KF + j * D + d), _mm512_loadu_ps(pq + d), acc);
                            float x = _mm512_reduce_add_ps(acc) * scale;
                            if (softcap != 0.0f) x = softcap * tanhf(x);
                            sc[j] = x + mval[t][j];
                            bmax = std::max(bmax, sc[j]);
                        }
                        const float Mn = std::max(Mv[r], bmax);
                        if (Mn == -INFINITY) continue;
                        const float ms = Mv[r] == -INFINITY ? 0.0f : expf(Mv[r] - Mn);
                        float * orow = O + (size_t) r * D;
                        if (ms != 1.0f) {
                            const __m512 m = _mm512_set1_ps(ms);
                            for (int64_t d = 0; d < D; d += 16) _mm512_store_ps(orow + d, _mm512_mul_ps(_mm512_load_ps(orow + d), m));
                        }
                        float sum = 0.0f;
                        for (int j = 0; j < FA1_BLOCK; ++j) {
                            if (sc[j] == -INFINITY) continue;
                            const float pj = expf(sc[j] - Mn);
                            sum += pj;
                            const __m512 pp = _mm512_set1_ps(pj);
                            for (int64_t d = 0; d < D; d += 16) _mm512_store_ps(orow + d, _mm512_fmadd_ps(_mm512_loadu_ps(VF + j * D + d), pp, _mm512_load_ps(orow + d)));
                        }
                        Sv[r] = Sv[r] * ms + sum;
                        Mv[r] = Mn;
                    }
                }
                continue;
            }
            // K rows and V pairs of the block, hi and lo (keys past the block's end as zeros)
            for (int j = 0; j < FA1_BLOCK; ++j) {
                uint16_t * ka = KA + (size_t) j * D, * kl = KAL + (size_t) j * D;
                if (j >= nb) { memset(ka, 0, D * 2); memset(kl, 0, D * 2); continue; }
                const ggml_fp16_t * kr = (const ggml_fp16_t *) (kb + (c0 + j) * k->nb[1]);
                for (int64_t d = 0; d < D; d += 16) {
                    __m256i hi, lo;
                    fa1x_split(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (kr + d))), &hi, &lo);
                    _mm256_storeu_si256((__m256i *) (ka + d), hi);
                    _mm256_storeu_si256((__m256i *) (kl + d), lo);
                }
            }
            for (int pr = 0; pr < 16; ++pr) {
                alignas(64) uint16_t h0[256], l0[256], h1[256], l1[256];
                for (int e = 0; e < 2; ++e) {
                    const int j = 2 * pr + e;
                    uint16_t * hh = e ? h1 : h0, * ll = e ? l1 : l0;
                    if (j >= nb) { memset(hh, 0, D * 2); memset(ll, 0, D * 2); continue; }
                    const ggml_fp16_t * vr = (const ggml_fp16_t *) (vb + (c0 + j) * v->nb[1]);
                    for (int64_t d = 0; d < D; d += 16) {
                        __m256i hi, lo;
                        fa1x_split(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (vr + d))), &hi, &lo);
                        _mm256_store_si256((__m256i *) (hh + d), hi);
                        _mm256_store_si256((__m256i *) (ll + d), lo);
                    }
                }
                for (int64_t d = 0; d < D; d += 32) {
                    const __m512i a = _mm512_load_si512(h0 + d), b = _mm512_load_si512(h1 + d);
                    _mm512_storeu_si512(VV + ((size_t) pr * D + d) * 2, _mm512_permutex2var_epi16(a, idx_lo, b));
                    _mm512_storeu_si512(VV + ((size_t) pr * D + d + 16) * 2, _mm512_permutex2var_epi16(a, idx_hi, b));
                    const __m512i c = _mm512_load_si512(l0 + d), e2 = _mm512_load_si512(l1 + d);
                    _mm512_storeu_si512(VL + ((size_t) pr * D + d) * 2, _mm512_permutex2var_epi16(c, idx_lo, e2));
                    _mm512_storeu_si512(VL + ((size_t) pr * D + d + 16) * 2, _mm512_permutex2var_epi16(c, idx_hi, e2));
                }
            }
            // S^T[j][r] = K[j] . Q[r]
            for (int kh = 0; kh < 2; ++kh) {
                for (int64_t rt = 0; rt < RP; rt += 16) {
                    _tile_zero(0);
                    for (int64_t s0 = 0; s0 < D; s0 += 32) {
                        _tile_loadd(1, KA + (size_t) kh * 16 * D + s0, D * 2);
                        _tile_loadd(2, KAL + (size_t) kh * 16 * D + s0, D * 2);
                        _tile_loadd(3, QT + (s0 / 2) * RP + rt, RP * 4);
                        _tile_loadd(4, QTL + (s0 / 2) * RP + rt, RP * 4);
                        _tile_dpbf16ps(0, 1, 3);
                        _tile_dpbf16ps(0, 1, 4);
                        _tile_dpbf16ps(0, 2, 3);
                    }
                    _tile_stored(0, ST + (size_t) kh * 16 * RP + rt, RP * 4);
                }
            }
            if (!all) {  // the mask, one value per (key, row)
                for (int j = 0; j < FA1_BLOCK; ++j) {
                    float * mt = MT + (size_t) j * RP;
                    for (int64_t r = 0; r < RP; ++r) mt[r] = r < R ? mval[r / rk2][j] : -INFINITY;
                }
            }
            // the online softmax down the columns, 16 rows at a time
            const __m512 vs = _mm512_set1_ps(scale), ninf = _mm512_set1_ps(-INFINITY);
            for (int64_t rt = 0; rt < RP; rt += 16) {
                const __mmask16 valid = (rt + 16 <= R) ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (R > rt ? R - rt : 0)) - 1);
                __m512 bmax = ninf;
                for (int j = 0; j < FA1_BLOCK; ++j) {
                    __m512 sc = _mm512_mul_ps(_mm512_loadu_ps(ST + (size_t) j * RP + rt), vs);
                    if (softcap != 0.0f) {
                        alignas(64) float tmp[16];
                        _mm512_store_ps(tmp, sc);
                        for (int i = 0; i < 16; ++i) tmp[i] = softcap * tanhf(tmp[i]);
                        sc = _mm512_load_ps(tmp);
                    }
                    if (!all) sc = _mm512_add_ps(sc, _mm512_loadu_ps(MT + (size_t) j * RP + rt));
                    sc = _mm512_mask_mov_ps(ninf, valid, sc);
                    _mm512_storeu_ps(ST + (size_t) j * RP + rt, sc);
                    bmax = _mm512_max_ps(bmax, sc);
                }
                const __m512 mold = _mm512_loadu_ps(Mv + rt);
                const __m512 mn = _mm512_max_ps(mold, bmax);
                const __mmask16 live = _mm512_cmp_ps_mask(mn, ninf, _CMP_NEQ_OQ);
                const __mmask16 had = _mm512_cmp_ps_mask(mold, ninf, _CMP_NEQ_OQ) & live;
                const __m512 ms = _mm512_mask_mov_ps(_mm512_setzero_ps(), had, ggml_v_expf(_mm512_maskz_sub_ps(had, mold, mn)));
                __m512 sum = _mm512_setzero_ps();
                for (int j = 0; j < FA1_BLOCK; ++j) {
                    const __m512 x = _mm512_sub_ps(_mm512_loadu_ps(ST + (size_t) j * RP + rt), mn);
                    const __mmask16 ok = _mm512_cmp_ps_mask(x, ninf, _CMP_NEQ_OQ) & live;
                    const __m512 e = _mm512_maskz_mov_ps(ok, ggml_v_expf(_mm512_maskz_mov_ps(ok, x)));
                    _mm512_storeu_ps(ST + (size_t) j * RP + rt, e);
                    sum = _mm512_add_ps(sum, e);
                }
                const __m512 sold = _mm512_loadu_ps(Sv + rt);
                _mm512_storeu_ps(Sv + rt, _mm512_mask_mov_ps(sold, live, _mm512_add_ps(_mm512_mul_ps(sold, ms), sum)));
                _mm512_storeu_ps(Mv + rt, _mm512_mask_mov_ps(mold, live, mn));
                _mm512_storeu_ps(msv + rt, _mm512_mask_mov_ps(_mm512_set1_ps(1.0f), live, ms));
            }
            // rescale O where the max moved; P to rows (gathered), hi and lo
            for (int64_t r = 0; r < RP; ++r) {
                if (msv[r] != 1.0f) {
                    const __m512 m = _mm512_set1_ps(msv[r]);
                    float * orow = O + r * D;
                    for (int64_t d = 0; d < D; d += 16) _mm512_store_ps(orow + d, _mm512_mul_ps(_mm512_load_ps(orow + d), m));
                }
                for (int j0 = 0; j0 < 32; j0 += 16) {
                    const __m512 pv = _mm512_i32gather_ps(gv, ST + (size_t) j0 * RP + r, 4);
                    __m256i hi, lo;
                    fa1x_split(pv, &hi, &lo);
                    _mm256_storeu_si256((__m256i *) (PB + r * 32 + j0), hi);
                    _mm256_storeu_si256((__m256i *) (PL + r * 32 + j0), lo);
                }
            }
            // O += P V
            for (int64_t rt = 0; rt < RP; rt += 16) {
                _tile_loadd(5, PB + rt * 32, 64);
                _tile_loadd(6, PL + rt * 32, 64);
                for (int64_t c = 0; c < D; c += 16) {
                    _tile_loadd(7, O + rt * D + c, D * 4);
                    _tile_loadd(1, VV + c * 2, D * 4);
                    _tile_loadd(2, VL + c * 2, D * 4);
                    _tile_dpbf16ps(7, 5, 1);
                    _tile_dpbf16ps(7, 5, 2);
                    _tile_dpbf16ps(7, 6, 1);
                    _tile_stored(7, O + rt * D + c, D * 4);
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            const int t = r / (int) rk2, h = r % (int) rk2;
            const float inv = Sv[r] > 0.0f ? 1.0f / Sv[r] : 0.0f;
            const int64_t i2 = g * rk2 + h, i1 = t0 + t;
            float * out = (float *) ((char *) dst->data + (i3 * ne2 * ne1 + i2 + i1 * ne1) * dst->nb[1]);
            const float * orow = O + (size_t) r * D;
            for (int64_t d = 0; d < D; ++d) out[d] = orow[d] * inv;
        }
    }
    if (saved.pal) _tile_loadconfig(&saved); else _tile_release();
    return true;
}
#else
static bool fa1x_attn(const ggml_compute_params *, ggml_tensor *) { return false; }
#endif
#endif
