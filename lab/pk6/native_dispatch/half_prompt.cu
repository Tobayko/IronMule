// A bounded half-precision FFN weight cache with unscaled FP32 GEMM output for Ada.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <tuple>
#include "block_values.cuh"
#include "weight_cache.cuh"

namespace {
std::atomic<uint64_t> launches{0};
std::atomic<int> status{0};

struct State {
    cublasHandle_t handle=nullptr;
    half *x=nullptr;
    size_t elements=0;
    ~State() { if (x) cudaFree(x); if (handle) cublasDestroy(handle); }
    bool initialize(cudaStream_t stream, size_t count) {
        if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
        if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return false;
        if (count <= elements) return true;
        half *next=nullptr;
        if (cudaMalloc(&next, count*sizeof(half)) != cudaSuccess) { cudaGetLastError(); return false; }
        if (x) cudaFree(x);
        x=next; elements=count;
        return true;
    }
};
__global__ void to_half(const float *src, half *dst, size_t elements) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<elements) dst[i]=__float2half_rn(src[i]);
}
}

extern "C" int im_prompt_threshold() {
    const char *s=std::getenv("IRONMULE_LAB_FP8");
    return s ? std::atoi(s) : 0;
}
extern "C" uint64_t im_prompt_dispatches() { return launches.load(); }
extern "C" int im_prompt_status() { return status.load(); }

extern "C" int ironmule_kg_mul_mat(int type,const void *w,const float *x,float *y,int64_t n,int64_t k,
                                    int64_t b,cudaStream_t stream) {
    static const int minimum=im_prompt_threshold();
    if (minimum<=0 || b<minimum || b>2048 || n<=0 || n>25600 || k<=0 || k>25600 ||
        (type!=12 && type!=14) || (n!=25600 && k!=25600) || n%16 || k%256) return 1;
    std::lock_guard<std::mutex> lifetime(persistent_prompt::mutex);
    if (persistent_prompt::sessions!=1) return 1;
    int device;
    cudaStreamCaptureStatus capture;
    if (cudaGetDevice(&device)!=cudaSuccess || cudaStreamIsCapturing(stream,&capture)!=cudaSuccess ||
        capture!=cudaStreamCaptureStatusNone) return 1;
    thread_local std::map<std::pair<int,cudaStream_t>,State> states;
    auto &state=states[{device,stream}];
    if (!state.handle) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop,device)!=cudaSuccess || prop.major!=8 || prop.minor!=9) return 1;
    }
    if (!state.initialize(stream,size_t(b)*k)) return 1;
    bool fresh=true;
    auto *entry=persistent_prompt::get(w,type,n,k,device,stream,fresh,sizeof(half));
    if (!entry) return 1;
    auto *packed=reinterpret_cast<half *>(entry->weights);
    if (fresh) {
        auto *raw=static_cast<const unsigned char *>(w);
        const int blocks=int(n*k/256);
        if (type==12) half_weights<12><<<blocks,256,0,stream>>>(raw,packed,blocks);
        else half_weights<14><<<blocks,256,0,stream>>>(raw,packed,blocks);
    }
    size_t elements=b*k;
    to_half<<<(elements+255)/256,256,0,stream>>>(x,state.x,elements);
    const float alpha=1.0f,beta=0.0f;
    auto rc=cublasGemmEx(state.handle,CUBLAS_OP_T,CUBLAS_OP_N,n,b,k,&alpha,packed,CUDA_R_16F,k,
                         state.x,CUDA_R_16F,k,&beta,y,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (rc!=CUBLAS_STATUS_SUCCESS) {status.store(int(rc)); return 1;}
    ++launches;
    return 0;
}
