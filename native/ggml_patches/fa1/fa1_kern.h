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
#endif
