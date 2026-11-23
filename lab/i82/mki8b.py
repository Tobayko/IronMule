# I82: W8A8 prompt matmuls with diagnostics and per-channel smoothing (after pk4), plus PK5 (fast Q4_K -> fp16 for PK2)
import sys
from pathlib import Path
s = Path(sys.argv[1]).read_text()

anchor = '''extern "C" GGML_BACKEND_API long ironmule_pk2_set_min_rows(long rows) {'''
assert s.count(anchor) == 1
kernels = r'''// I82 (IronMule, experiment): IRONMULE_I8 selects how PK2's prompt matmuls multiply. 0: fp16 (default). 2: int8 x int8
// -> int32 on the tensor cores, weights per output row, activations per token (scale = absmax / 127). 1 / 3 / 4: round
// both / only the weights / only the activations to those int8 levels and multiply in fp16 (to locate the error).
// 5: fp16, and record each weight's per-input-channel absmax of the activations (calibration). 6 / 7: int8 / rounding
// only, after smoothing with s_k = sqrt(max|x_k|) / sqrt(max|w_k|) from that calibration (x / s, w * s; unchanged
// product). IRONMULE_I8_MAX_K keeps matmuls with a longer inner dimension on fp16. ironmule_i8_set_mode switches.
static std::atomic<int> ironmule_i8_mode_v{-1};
static int ironmule_i8_mode() {
    int mode = ironmule_i8_mode_v.load(std::memory_order_relaxed);
    if (mode < 0) {
        const char * v = std::getenv("IRONMULE_I8");
        mode = v ? std::atoi(v) : 0;
        ironmule_i8_mode_v.store(mode, std::memory_order_relaxed);
    }
    return mode;
}
extern "C" GGML_BACKEND_API int ironmule_i8_set_mode(int mode);
extern "C" GGML_BACKEND_API int ironmule_i8_set_mode(int mode) {
    const int previous = ironmule_i8_mode();
    ironmule_i8_mode_v.store(mode, std::memory_order_relaxed);
    return previous;
}

static long ironmule_i8_max_k() {
    static long k = -1;
    if (k < 0) {
        const char * v = std::getenv("IRONMULE_I8_MAX_K");
        k = v ? std::atol(v) : 0L;
    }
    return k;
}

struct ironmule_i8_stats {
    float * ax = nullptr;  // max |x_k| over the calibration rows
    float * aw = nullptr;  // max |w_k| over the weight's rows
    float * sk = nullptr;  // smoothing factor, made from ax and aw on first use
    bool aw_done = false, sk_done = false;
};
static std::map<const void *, ironmule_i8_stats> ironmule_i8_table;

// column absmax of a [rows][n] half matrix into amax[n] (atomic max of non-negative floats as ints)
static __global__ void ironmule_col_absmax(const half * __restrict__ x, const int64_t rows, const int64_t n,
                                           float * __restrict__ amax) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= n) {
        return;
    }
    float m = 0.0f;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        m = fmaxf(m, fabsf(__half2float(x[r*n + k])));
    }
    atomicMax((int *) (amax + k), __float_as_int(m));
}

static __global__ void ironmule_make_smooth(const float * __restrict__ ax, const float * __restrict__ aw,
                                            float * __restrict__ sk, const int64_t n) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k < n) {
        float v = (ax[k] > 0.0f && aw[k] > 0.0f) ? sqrtf(ax[k]) / sqrtf(aw[k]) : 1.0f;
        sk[k] = fminf(fmaxf(v, 1e-4f), 1e4f);
    }
}

// one block per row of n halves, optionally times col[k] (MUL) or divided by it: absmax, scale (absmax / 127), and the
// row as int8 (q) or rounded back to half (fq); ROUND false copies the (smoothed) row to fq unrounded
template <bool SIM, int COL, bool ROUND>
static __global__ void ironmule_q8_rows(const half * __restrict__ x, const int64_t n, int8_t * __restrict__ q,
                                        half * __restrict__ fq, float * __restrict__ scale, const float * __restrict__ col) {
    const int64_t row = blockIdx.x;
    const half * xr = x + row*n;
    auto val = [&](int64_t i) {
        float v = __half2float(xr[i]);
        if (COL == 1) v *= col[i];
        if (COL == 2) v /= col[i];
        return v;
    };
    float m = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        m = fmaxf(m, fabsf(val(i)));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    __shared__ float red[32];
    const int warp = threadIdx.x/32, lane = threadIdx.x%32;
    if (lane == 0) {
        red[warp] = m;
    }
    __syncthreads();
    m = lane < (int) (blockDim.x/32) ? red[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    const float s = m/127.0f;
    const float inv = m > 0.0f ? 127.0f/m : 0.0f;
    if (threadIdx.x == 0 && scale) {
        scale[row] = s;
    }
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = val(i);
        if (!ROUND) {
            fq[row*n + i] = __float2half(v);
            continue;
        }
        const int a = max(-127, min(127, __float2int_rn(v*inv)));
        if (SIM) {
            fq[row*n + i] = __float2half(a*s);
        } else {
            q[row*n + i] = (int8_t) a;
        }
    }
}

// dst[j*ldc + i] = acc[j*rows + i] * ws[i] * xs[j]
static __global__ void ironmule_i8_scale(const int32_t * __restrict__ acc, const int64_t rows, const int64_t cols,
                                         const float * __restrict__ ws, const float * __restrict__ xs,
                                         float * __restrict__ dst, const int64_t ldc) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t j = blockIdx.y;
    if (i < rows && j < cols) {
        dst[j*ldc + i] = (float) acc[j*rows + i] * ws[i] * xs[j];
    }
}

static inline __device__ void ironmule_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// PK5 (IronMule): Q4_K -> fp16 for PK2, the same float arithmetic as llama.cpp's dequantize_block_q4_K (bit-identical)
// with 8 contiguous values per thread (one 16-byte store) and 8 super-blocks per 256-thread block
static __global__ void ironmule_dequant_q4_K_f16(const block_q4_K * __restrict__ x, half * __restrict__ y, const int64_t nb) {
    const int64_t i = (int64_t) blockIdx.x*8 + threadIdx.x/32;
    if (i >= nb) {
        return;
    }
    const int t = threadIdx.x % 32;
    const int j = t/4;            // sub-block 0..7
    const int o = (t%4)*8;        // offset inside the sub-block
    const int il = j/2;
    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);
    uint8_t sc, m;
    ironmule_scale_min_k4(j, x[i].scales, sc, m);
    const float d = dall * sc;
    const float mn = dmin * m;
    const uint2 qv = *(const uint2 *) (x[i].qs + 32*il + o);
    const uint8_t * qb = (const uint8_t *) &qv;
    half h[8];
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int q = (j & 1) ? (qb[l] >> 4) : (qb[l] & 0xF);
        h[l] = d * q - mn;
    }
    *(uint4 *) (y + i*QK_K + 32*j + o) = *(const uint4 *) h;
}

// I82: int8 rows straight from Q4_K (no fp16 copy): one 256-thread block per output row; each lane dequantizes 8
// values per super-block as PK5 does (in float), times the smoothing factor sk[k] when given; the row's absmax gives
// its scale, then the values are rounded to int8 (one 8-byte store per lane)
static __global__ void ironmule_q4k_to_i8(const block_q4_K * __restrict__ x, const int64_t nbr,
                                          const float * __restrict__ sk, int8_t * __restrict__ q, float * __restrict__ scale) {
    const int64_t row = blockIdx.x;
    const block_q4_K * xr = x + row*nbr;
    const int warp = threadIdx.x/32, lane = threadIdx.x%32;
    const int j = lane/4, o = (lane%4)*8, il = j/2;
    auto vals = [&](int64_t b, float (&v)[8]) {
        const float dall = __low2half(xr[b].dm);
        const float dmin = __high2half(xr[b].dm);
        uint8_t sc, m;
        ironmule_scale_min_k4(j, xr[b].scales, sc, m);
        const float d = dall * sc;
        const float mn = dmin * m;
        const uint2 qv = *(const uint2 *) (xr[b].qs + 32*il + o);
        const uint8_t * qb = (const uint8_t *) &qv;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int qq = (j & 1) ? (qb[l] >> 4) : (qb[l] & 0xF);
            v[l] = d * qq - mn;
            if (sk) v[l] *= sk[b*QK_K + 32*j + o + l];
        }
    };
    float mx = 0.0f;
    for (int64_t b = warp; b < nbr; b += blockDim.x/32) {
        float v[8];
        vals(b, v);
#pragma unroll
        for (int l = 0; l < 8; ++l) mx = fmaxf(mx, fabsf(v[l]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    }
    __shared__ float red[32];
    if (lane == 0) {
        red[warp] = mx;
    }
    __syncthreads();
    mx = lane < (int) (blockDim.x/32) ? red[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
    }
    const float inv = mx > 0.0f ? 127.0f/mx : 0.0f;
    if (threadIdx.x == 0) {
        scale[row] = mx/127.0f;
    }
    int8_t * qr = q + row*nbr*QK_K;
    for (int64_t b = warp; b < nbr; b += blockDim.x/32) {
        float v[8];
        vals(b, v);
        char4 lo, hi;
        lo.x = (signed char) max(-127, min(127, __float2int_rn(v[0]*inv)));
        lo.y = (signed char) max(-127, min(127, __float2int_rn(v[1]*inv)));
        lo.z = (signed char) max(-127, min(127, __float2int_rn(v[2]*inv)));
        lo.w = (signed char) max(-127, min(127, __float2int_rn(v[3]*inv)));
        hi.x = (signed char) max(-127, min(127, __float2int_rn(v[4]*inv)));
        hi.y = (signed char) max(-127, min(127, __float2int_rn(v[5]*inv)));
        hi.z = (signed char) max(-127, min(127, __float2int_rn(v[6]*inv)));
        hi.w = (signed char) max(-127, min(127, __float2int_rn(v[7]*inv)));
        *(char4 *) (qr + b*QK_K + 32*j + o) = lo;
        *(char4 *) (qr + b*QK_K + 32*j + o + 4) = hi;
    }
}

// the int8 modes that read a Q4_K matrix directly (no fp16 copy), for this matmul's shape
static bool ironmule_i8_direct(const ggml_tensor * src0, int64_t ne00, int64_t ne10, int64_t row_diff) {
    static int on = -1;
    if (on < 0) {
        const char * v = std::getenv("IRONMULE_I8_DIRECT");
        on = v ? std::atoi(v) : 1;
    }
    if (!on || ironmule_pk2_min_rows() <= 0 || src0->type != GGML_TYPE_Q4_K) {
        return false;
    }
    const int mode = ironmule_i8_mode();
    return (mode == 2 || mode == 6) && ne00 == ne10 && ne00 % 4 == 0 && row_diff % 4 == 0
        && (ironmule_i8_max_k() == 0 || ne00 <= ironmule_i8_max_k())
        && (mode == 2 || ironmule_i8_table.count(src0->data) == 1);
}

static bool ironmule_pk5_on() {
    static int on = -1;
    if (on < 0) {
        const char * v = std::getenv("IRONMULE_PK5");
        on = v ? std::atoi(v) : 1;
    }
    return on != 0;
}

'''
s = s.replace(anchor, kernels + anchor, 1)

# PK5: the dequantization of src0 for the fp16 cuBLAS path
old = '''            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);'''
assert s.count(old) == 1, s.count(old)
new = '''            size_t ne = row_diff*ne00;
            if (ironmule_i8_direct(src0, ne00, ne10, row_diff)) {
                // I82: the int8 path below reads the Q4_K blocks itself
            } else if (src0->type == GGML_TYPE_Q4_K && ironmule_pk2_min_rows() > 0 && ironmule_pk5_on()) {  // PK5
                src0_as_f16.alloc(ne);
                const int64_t nb = ne / QK_K;
                ironmule_dequant_q4_K_f16<<<(nb + 7)/8, 256, 0, stream>>>((const block_q4_K *) src0_dd_i, src0_as_f16.get(), nb);
            } else {
                src0_as_f16.alloc(ne);
                to_fp16_cuda(src0_dd_i, src0_as_f16.get(), ne, stream);
            }'''
s = s.replace(old, new, 1)

old = '''        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc) || ironmule_pk2_min_rows() > 0) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {'''
assert s.count(old) == 1
new = '''        const int i8 = ironmule_pk2_min_rows() > 0 ? ironmule_i8_mode() : 0;
        if (i8 > 0 && src0->type != GGML_TYPE_F16 && ne00 == ne10 && ne00 % 4 == 0 && row_diff % 4 == 0
                && (ironmule_i8_max_k() == 0 || ne00 <= ironmule_i8_max_k())) {
            // I82
            const float alpha = 1.0f;
            const float beta = 0.0f;
            ggml_cuda_pool_alloc<float> ws(ctx.pool(id), row_diff);
            ggml_cuda_pool_alloc<float> xs(ctx.pool(id), src1_ncols);
            const float * sk = nullptr;
            if (i8 >= 5) {
                ironmule_i8_stats & st = ironmule_i8_table[(const void *) src0->data];
                if (st.ax == nullptr) {
                    CUDA_CHECK(cudaMalloc((void **) &st.ax, 3*ne00*sizeof(float)));
                    CUDA_CHECK(cudaMemsetAsync(st.ax, 0, 3*ne00*sizeof(float), stream));
                    st.aw = st.ax + ne00;
                    st.sk = st.ax + 2*ne00;
                }
                if (!st.aw_done && src0_ptr != nullptr && src0_as_f16.get() != nullptr) {
                    ironmule_col_absmax<<<dim3((ne00 + 255)/256, 64), 256, 0, stream>>>(src0_ptr, row_diff, ne00, st.aw);
                    st.aw_done = true;
                }
                if (i8 == 5) {
                    ironmule_col_absmax<<<dim3((ne10 + 255)/256, 64), 256, 0, stream>>>(src1_ptr, src1_ncols, ne10, st.ax);
                } else if (!st.sk_done) {
                    ironmule_make_smooth<<<(ne00 + 255)/256, 256, 0, stream>>>(st.ax, st.aw, st.sk, ne00);
                    st.sk_done = true;
                }
                sk = st.sk;
            }
            if (i8 == 5) {
                CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                            row_diff, src1_ncols, ne10,
                            &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                    src1_ptr,  CUDA_R_16F, ne10,
                            &beta,   dst_dd_i, CUDA_R_32F, ldc,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else if (i8 != 2 && i8 != 6) {
                ggml_cuda_pool_alloc<half> w16(ctx.pool(id), row_diff*ne00);
                ggml_cuda_pool_alloc<half> x16(ctx.pool(id), src1_ncols*ne10);
                const bool rw = i8 == 1 || i8 == 3 || i8 == 7, rx = i8 == 1 || i8 == 4 || i8 == 7;
                if (i8 == 7) {
                    ironmule_q8_rows<true, 1, true><<<row_diff, 256, 0, stream>>>(src0_ptr, ne00, nullptr, w16.get(), ws.get(), sk);
                    ironmule_q8_rows<true, 2, true><<<src1_ncols, 256, 0, stream>>>(src1_ptr, ne10, nullptr, x16.get(), xs.get(), sk);
                } else {
                    if (rw) {
                        ironmule_q8_rows<true, 0, true><<<row_diff, 256, 0, stream>>>(src0_ptr, ne00, nullptr, w16.get(), ws.get(), nullptr);
                    } else {
                        ironmule_q8_rows<true, 0, false><<<row_diff, 256, 0, stream>>>(src0_ptr, ne00, nullptr, w16.get(), nullptr, nullptr);
                    }
                    if (rx) {
                        ironmule_q8_rows<true, 0, true><<<src1_ncols, 256, 0, stream>>>(src1_ptr, ne10, nullptr, x16.get(), xs.get(), nullptr);
                    } else {
                        ironmule_q8_rows<true, 0, false><<<src1_ncols, 256, 0, stream>>>(src1_ptr, ne10, nullptr, x16.get(), nullptr, nullptr);
                    }
                }
                CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                            row_diff, src1_ncols, ne10,
                            &alpha, w16.get(), CUDA_R_16F, ne00,
                                    x16.get(), CUDA_R_16F, ne10,
                            &beta,   dst_dd_i, CUDA_R_32F, ldc,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                ggml_cuda_pool_alloc<int8_t> w8(ctx.pool(id), row_diff*ne00);
                ggml_cuda_pool_alloc<int8_t> x8(ctx.pool(id), src1_ncols*ne10);
                ggml_cuda_pool_alloc<int32_t> acc(ctx.pool(id), row_diff*src1_ncols);
                const bool direct = ironmule_i8_direct(src0, ne00, ne10, row_diff);
                if (direct) {
                    ironmule_q4k_to_i8<<<row_diff, 256, 0, stream>>>((const block_q4_K *) src0_dd_i, ne00/QK_K,
                                                                     i8 == 6 ? sk : nullptr, w8.get(), ws.get());
                } else if (i8 == 6) {
                    ironmule_q8_rows<false, 1, true><<<row_diff, 256, 0, stream>>>(src0_ptr, ne00, w8.get(), nullptr, ws.get(), sk);
                } else {
                    ironmule_q8_rows<false, 0, true><<<row_diff, 256, 0, stream>>>(src0_ptr, ne00, w8.get(), nullptr, ws.get(), nullptr);
                }
                if (i8 == 6) {
                    ironmule_q8_rows<false, 2, true><<<src1_ncols, 256, 0, stream>>>(src1_ptr, ne10, x8.get(), nullptr, xs.get(), sk);
                } else {
                    ironmule_q8_rows<false, 0, true><<<src1_ncols, 256, 0, stream>>>(src1_ptr, ne10, x8.get(), nullptr, xs.get(), nullptr);
                }
                const int32_t ialpha = 1;
                const int32_t ibeta = 0;
                CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                            row_diff, src1_ncols, ne10,
                            &ialpha, w8.get(),  CUDA_R_8I, ne00,
                                     x8.get(),  CUDA_R_8I, ne10,
                            &ibeta,  acc.get(), CUDA_R_32I, row_diff,
                            CUBLAS_COMPUTE_32I,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                const dim3 grid((row_diff + 255)/256, src1_ncols);
                ironmule_i8_scale<<<grid, 256, 0, stream>>>(acc.get(), row_diff, src1_ncols, ws.get(), xs.get(), dst_dd_i, ldc);
            }
        } else if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc) || ironmule_pk2_min_rows() > 0) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha, src0_ptr,  CUDA_R_16F, ne00,
                                src1_ptr,  CUDA_R_16F, ne10,
                        &beta,   dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {'''
s = s.replace(old, new, 1)
if '#include <map>' not in s:
    s = s.replace('#include <cstdlib>\n', '#include <cstdlib>\n#include <map>\n', 1)
Path(sys.argv[2]).write_text(s)
print("i82 + pk5 applied")
