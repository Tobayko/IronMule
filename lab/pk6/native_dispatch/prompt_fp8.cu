// Research KG1 plugin for Ada prompt matmuls. GGUF blocks remain the source of every weight.
// Convert only the current matrix into bounded scratch, multiply with FP32 accumulation,
// then undo the per-output-channel and per-input-row scales. Decode keeps its incumbent kernels.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <tuple>
#include "prompt_codec.cuh"

namespace {
constexpr size_t scratch_limit = size_t{1536} << 20;
constexpr size_t lt_bytes = size_t{8} << 20;
std::atomic<uint64_t> calls{0};
std::atomic<int> last_status{0};
__device__ float block_half(const unsigned char *p) {
    return __half2float(*reinterpret_cast<const half *>(p));
}

__device__ float warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, d));
    return v;
}

template<int Type> __global__ void weight_scales(const unsigned char *weights, float *scale, int blocks) {
    int n = blockIdx.x, lane = threadIdx.x;
    constexpr int bytes = Type == 12 ? 144 : 210;
    float largest = 0;
    for (int i = lane; i < blocks; i += 32) {
        const auto *p = weights + (size_t(n) * blocks + i) * bytes;
        float bound;
#ifdef IM_PROMPT_INT8
        // Bound each actual GGUF group independently so large outliers elsewhere in
        // the block do not waste the signed byte's precision.
        bound = 0.f;
        if constexpr (Type == 12) {
            const auto *s = p + 4;
            for (int g = 0; g < 8; ++g) {
                int sc = g < 4 ? s[g] & 63 : (s[g+4] & 15) | ((s[g-4] >> 6) << 4);
                int mn = g < 4 ? s[g+4] & 63 : (s[g+4] >> 4) | ((s[g] >> 6) << 4);
                float minimum = block_half(p+2) * mn;
                float maximum = 15.f * block_half(p) * sc - minimum;
                bound = fmaxf(bound, fmaxf(fabsf(minimum), fabsf(maximum)));
            }
        } else {
            for (int g = 0; g < 16; ++g) {
                float scale = block_half(p+208) * reinterpret_cast<const int8_t *>(p+192)[g];
                bound = fmaxf(bound, 32.f * fabsf(scale));
            }
        }
#else
        if constexpr (Type == 12) bound = fabsf(block_half(p)) * 945.f + fabsf(block_half(p+2)) * 63.f;
        else bound = fabsf(block_half(p+208)) * 4096.f;
#endif
        largest = fmaxf(largest, bound);
    }
    largest = warp_max(largest);
    if (!lane) scale[n] = largest > 0.f ? largest / packed_max : 1.f;
}

template<int Type> __global__ void expand_weights(const unsigned char *weights, const float *scales,
                                                   unsigned char *dst, int blocks_per_row, int64_t blocks) {
    constexpr int bytes = Type == 12 ? 144 : 210;
    int lane = threadIdx.x & 31;
    int64_t index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (index >= blocks) return;
    const unsigned char *p = weights + index * bytes;
    float inv = 1.f / scales[index / blocks_per_row];
    int pos = lane * 8;
    union { unsigned char b[8]; uint2 pair; } encoded;
    if constexpr (Type == 12) {
        int g = pos >> 5, off = pos & 31;
        const auto *s = p + 4;
        int sc = g < 4 ? s[g] & 63 : (s[g+4] & 15) | ((s[g-4] >> 6) << 4);
        int mn = g < 4 ? s[g+4] & 63 : (s[g+4] >> 4) | ((s[g] >> 6) << 4);
        float delta = block_half(p) * sc, bias = block_half(p+2) * mn;
        const auto *q = p + 16 + 32 * (g / 2) + off;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int v = g & 1 ? q[i] >> 4 : q[i] & 15;
            encoded.b[i] = encode((delta * v - bias) * inv);
        }
    } else {
        int halfblock = pos >> 7, g = (pos & 127) >> 5, off = pos & 31;
        const auto *ql = p + halfblock * 64 + (g & 1) * 32 + off;
        const auto *qh = p + 128 + halfblock * 32 + off;
        float delta = block_half(p+208) * reinterpret_cast<const int8_t *>(p+192)[pos / 16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int low = g < 2 ? ql[i] & 15 : ql[i] >> 4;
            int v = (low | (((qh[i] >> (2*g)) & 3) << 4)) - 32;
            encoded.b[i] = encode(delta * v * inv);
        }
    }
    reinterpret_cast<uint2 *>(dst)[index * 32 + lane] = encoded.pair;
}

__global__ void pack_input(const float *x, unsigned char *packed, float *scale, int k, int rows) {
    int row = blockIdx.x, tid = threadIdx.x;
    float vmax = 0;
    if (row < rows)
        for (int i=tid; i<k; i+=blockDim.x) vmax = fmaxf(vmax, fabsf(x[size_t(row)*k+i]));
    __shared__ float parts[8], factor;
    vmax = warp_max(vmax);
    if (!(tid & 31)) parts[tid / 32] = vmax;
    __syncthreads();
    if (tid == 0) {
        float high=0;
        for (float v : parts) high=fmaxf(v,high);
        factor = high > 0 ? high / packed_max : 1.f;
        scale[row]=factor;
    }
    __syncthreads();
    float inv=1.f/factor;
    for (int i=tid; i<k; i+=blockDim.x)
        packed[size_t(row)*k+i] = row < rows ?
            encode(x[size_t(row)*k+i]*inv) : 0;
}

__global__ void finish(const Accumulated *src, const float *wscale, const float *xscale,
                       float *y, int n, size_t count) {
    size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) y[i] = decode_accumulated(src[i]) * wscale[i % n] * xscale[i/n];
}

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t w = nullptr, x = nullptr, y = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool ready = false;
    ~Plan() {
        if (y) cublasLtMatrixLayoutDestroy(y);
        if (x) cublasLtMatrixLayoutDestroy(x);
        if (w) cublasLtMatrixLayoutDestroy(w);
        if (op) cublasLtMatmulDescDestroy(op);
    }
    void init(cublasLtHandle_t h, int n, int k, int b) {
        cublasOperation_t transpose=CUBLAS_OP_T;
        if (cublasLtMatmulDescCreate(&op,compute_type,blas_scale_type) ||
            cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)) ||
            cublasLtMatrixLayoutCreate(&w,packed_type,k,n,k) ||
            cublasLtMatrixLayoutCreate(&x,packed_type,k,b,k) ||
            cublasLtMatrixLayoutCreate(&y,accumulated_type,n,b,n)) return;
        cublasLtMatmulPreference_t pref;
        if (cublasLtMatmulPreferenceCreate(&pref)) return;
        cublasLtMatmulHeuristicResult_t result{};
        int found=0;
        auto status=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&lt_bytes,sizeof(lt_bytes));
        if (!status) status=cublasLtMatmulAlgoGetHeuristic(h,op,w,x,y,y,pref,1,&result,&found);
        cublasLtMatmulPreferenceDestroy(pref);
        if (!status && found == 1 && result.state==CUBLAS_STATUS_SUCCESS) { algo=result.algo; ready=true; }
        else last_status.store(status ? int(status) : -2);
    }
};

struct Context {
    cublasLtHandle_t h=nullptr;
    unsigned char *storage=nullptr;
    size_t capacity=0;
    std::map<std::tuple<int,int,int>,Plan> plans;
    ~Context() { if (storage) cudaFree(storage); if (h) cublasLtDestroy(h); }
    bool reserve(size_t needed) {
        if (needed <= capacity) return true;
        if (needed > scratch_limit) return false;
        unsigned char *next=nullptr;
        if (cudaMalloc(&next,needed) != cudaSuccess) { cudaGetLastError(); return false; }
        if (storage) cudaFree(storage);
        storage=next; capacity=needed;
        return true;
    }
};
size_t aligned(size_t n) { return (n+255) & ~size_t{255}; }
}

extern "C" uint64_t im_prompt_dispatches() { return calls.load(); }
extern "C" int im_prompt_status() { return last_status.load(); }

extern "C" int ironmule_kg_mul_mat(int type,const void *w,const float *x,float *y,int64_t n64,int64_t k64,
                                   int64_t b64,cudaStream_t stream) {
    static const int threshold=[](){ const char *s=std::getenv("IRONMULE_LAB_FP8"); return s?std::atoi(s):0; }();
    if (threshold<=0 || (type!=12 && type!=14) || b64<threshold || b64>4096 ||
        k64<=0 || k64>65536 || k64%256 || n64<=0 || n64>262144 || n64%16) return 1;
    int device;
    cudaStreamCaptureStatus capture;
    if (cudaGetDevice(&device)!=cudaSuccess || cudaStreamIsCapturing(stream,&capture)!=cudaSuccess ||
        capture!=cudaStreamCaptureStatusNone) return 1;
    thread_local std::map<std::pair<int,cudaStream_t>, Context> contexts;
    auto &ctx=contexts[{device,stream}];
    if (!ctx.h) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop,device)!=cudaSuccess || prop.major!=8 || prop.minor!=9 ||
            cublasLtCreate(&ctx.h)!=CUBLAS_STATUS_SUCCESS) return 1;
    }
    int n=n64,k=k64,b=b64, padded=(b+15)&~15;
    auto [it,added]=ctx.plans.try_emplace({n,k,padded});
    auto &plan=it->second;
    if (added) plan.init(ctx.h,n,k,padded);
    if (!plan.ready) return 1;
    size_t wb=aligned(size_t(n)*k), xb=aligned(size_t(k)*padded), yb=aligned(size_t(n)*padded*sizeof(Accumulated));
    size_t ws=aligned(size_t(n)*4), xs=aligned(size_t(padded)*4);
    if (!ctx.reserve(wb+xb+yb+ws+xs+lt_bytes)) return 1;
    auto *weights=ctx.storage, *input=weights+wb;
    auto *result=reinterpret_cast<Accumulated *>(input+xb);
    auto *sw=reinterpret_cast<float *>(input+xb+yb);
    auto *sx=reinterpret_cast<float *>(input+xb+yb+ws);
    void *workspace=input+xb+yb+ws+xs;
    int blocks=k/256;
    int64_t count=int64_t(n)*blocks;
    auto *raw=static_cast<const unsigned char *>(w);
    if (type==12) {
        weight_scales<12><<<n,32,0,stream>>>(raw,sw,blocks);
        expand_weights<12><<<(count+7)/8,256,0,stream>>>(raw,sw,weights,blocks,count);
    } else {
        weight_scales<14><<<n,32,0,stream>>>(raw,sw,blocks);
        expand_weights<14><<<(count+7)/8,256,0,stream>>>(raw,sw,weights,blocks,count);
    }
    pack_input<<<padded,256,0,stream>>>(x,input,sx,k,b);
    const BlasScale alpha=1,beta=0;
    auto status=cublasLtMatmul(ctx.h,plan.op,&alpha,weights,plan.w,input,plan.x,&beta,result,plan.y,
                              result,plan.y,&plan.algo,workspace,lt_bytes,stream);
    if (status!=CUBLAS_STATUS_SUCCESS) { last_status.store(int(status)); return 1; }
    size_t outputs=size_t(n)*b;
    finish<<<(outputs+255)/256,256,0,stream>>>(result,sw,sx,y,n,outputs);
    ++calls;
    return 0;
}

extern "C" int im_prompt_threshold() {
    const char *s = std::getenv("IRONMULE_LAB_FP8");
    return s ? std::atoi(s) : 0;
}
