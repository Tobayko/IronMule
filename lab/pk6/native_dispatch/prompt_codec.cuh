// Numeric representation for the Ada GGUF prompt plugin. All per-channel scales are FP32.
#pragma once
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifdef IM_PROMPT_INT8
using Accumulated = int32_t;
using BlasScale = int32_t;
constexpr auto packed_type = CUDA_R_8I;
constexpr auto accumulated_type = CUDA_R_32I;
constexpr auto compute_type = CUBLAS_COMPUTE_32I;
constexpr auto blas_scale_type = CUDA_R_32I;
constexpr float packed_max = 127.f;
__device__ unsigned char encode(float x) {
    int q = __float2int_rn(fminf(127.f, fmaxf(-127.f, x)));
    return static_cast<unsigned char>(static_cast<signed char>(q));
}
__device__ float decode_accumulated(Accumulated x) { return static_cast<float>(x); }
#else
using Accumulated = __nv_bfloat16;
using BlasScale = float;
constexpr auto packed_type = CUDA_R_8F_E4M3;
constexpr auto accumulated_type = CUDA_R_16BF;
constexpr auto compute_type = CUBLAS_COMPUTE_32F;
constexpr auto blas_scale_type = CUDA_R_32F;
constexpr float packed_max = 448.f;
__device__ unsigned char encode(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ float decode_accumulated(Accumulated x) { return __bfloat162float(x); }
#endif
