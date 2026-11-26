// Extend the vendored MQ1 conversion to real multi-tile prompt batches on Ada.
// Single-device experimental ABI. The caller explicitly delimits immutable model weight lifetimes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include "mq1/mq1.cuh"

namespace {
std::mutex guard;
std::atomic<uint64_t> launches{0};
std::map<const void *, mq1::Prepared> matrices;
mq1::Scratch scratch;
int sessions = 0, device = -1;
size_t prepared_bytes = 0;
cudaStream_t owner_stream = nullptr;
bool stream_bound = false;
mq1::MarlinFuncPtr wide = nullptr;
int threads = 0, shared_bytes = 0, sms = 0;
const size_t max_cache = size_t{7} << 30;

void clear() {
    if (device >= 0) {
        int previous = -1;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        for (auto &it : matrices) mq1::release(it.second);
        cudaFree(scratch.a); cudaFree(scratch.c); cudaFree(scratch.c_tmp); cudaFree(scratch.locks);
        if (previous >= 0) cudaSetDevice(previous);
    }
    matrices.clear();
    scratch = {};
    device = -1;
    stream_bound = false;
    prepared_bytes = 0;
    wide = nullptr;
}

bool initialize(int dev) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop,dev) != cudaSuccess || prop.major!=8 || prop.minor!=9) return false;
    int shared = mq1::max_smem(dev);
    for (auto cfg : mq1::large_batch) {
        if (!mq1::valid(cfg, 4, 25600, 5120, shared)) continue;
        mq1::MarlinFuncPtr fn = nullptr;
        if (cfg.thread_n == 256) fn = mq1::kern<4,16,4,false,256>();
        else if (cfg.thread_n == 128) fn = mq1::kern<4,8,4,false,128>();
        else continue;
        if (cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,shared) != cudaSuccess) continue;
        device=dev; wide=fn; threads=cfg.num_threads; shared_bytes=shared; sms=prop.multiProcessorCount;
        return true;
    }
    return false;
}

__global__ void convert_pad(const float *input, half *dst, size_t actual, size_t padded) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < padded) dst[i] = i < actual ? __float2half_rn(input[i]) : __float2half_rn(0.f);
}
}

extern "C" void im_prompt_begin() {
    std::lock_guard<std::mutex> lock(guard);
    clear();
    ++sessions;
}
extern "C" void im_prompt_end() {
    std::lock_guard<std::mutex> lock(guard);
    clear();
    if (sessions > 0) --sessions;
}
extern "C" int im_prompt_threshold() { return 16; }
extern "C" int im_prompt_status() { return 0; }
extern "C" uint64_t im_prompt_dispatches() { return launches.load(); }
extern "C" size_t im_prompt_bytes() { std::lock_guard<std::mutex> lock(guard); return prepared_bytes; }

extern "C" int ironmule_kg_mul_mat(int type,const void *w,const float *x,float *y,int64_t n,int64_t k,
                                   int64_t b,cudaStream_t stream) {
    if (type != 12 || b < 16 || b > 2048 || n < 256 || n > 25600 || k <= 0 || k > 25600 ||
        n%256 || k%256) return 1;
    cudaStreamCaptureStatus capture;
    int dev;
    if (cudaGetDevice(&dev)!=cudaSuccess || cudaStreamIsCapturing(stream,&capture)!=cudaSuccess ||
        capture!=cudaStreamCaptureStatusNone) return 1;
    std::lock_guard<std::mutex> lock(guard);
    if (sessions != 1 || (stream_bound && owner_stream != stream) ||
        (device >= 0 && dev!=device) || (device<0 && !initialize(dev))) return 1;
    owner_stream=stream; stream_bound=true;
    auto it = matrices.find(w);
    if (it==matrices.end()) {
        const size_t bytes = size_t(n)*k*5/8;  // nibble plus two fp16 values per 32 weights
        if (prepared_bytes+bytes > max_cache) return 1;
        size_t free_bytes=0, total=0;
        if (cudaMemGetInfo(&free_bytes,&total)!=cudaSuccess || free_bytes < 2*bytes+(size_t{2}<<30)) return 1;
        mq1::Prepared weight;
        if (mq1::prepare(w,n,k,dev,stream,weight)) return 1;
        it = matrices.emplace(w,weight).first;
        prepared_bytes+=bytes;
    }
    auto &p=it->second;
    if (p.N!=n || p.K!=k) return 1;
    if (b<=32) {
        if (mq1::run(p,x,y,b,dev,stream,scratch,false)) return 1;
        ++launches;
        return 0;
    }
    const int padded=(b+63)/64*64;
    const size_t ac=size_t(padded)*k, cc=size_t(padded)*n;
    if (mq1::grow((void **)&scratch.a,&scratch.a_cap,ac*2) ||
        mq1::grow((void **)&scratch.c,&scratch.c_cap,cc*2) ||
        mq1::grow((void **)&scratch.c_tmp,&scratch.t_cap,size_t(sms)*64*marlin::max_thread_n*4) ||
        mq1::grow((void **)&scratch.locks,&scratch.l_cap,size_t(n/64*16+sms)*sizeof(int),true)) return 1;
    convert_pad<<<(ac+255)/256,256,0,stream>>>(x,scratch.a,b*k,ac);
    for (int offset=0;offset<padded;offset+=1024) {
        int m=std::min(1024,padded-offset);
        wide<<<sms,threads,shared_bytes,stream>>>(
            (const int4 *)(scratch.a+size_t(offset)*k),(const int4 *)p.qm,
            (int4 *)(scratch.c+size_t(offset)*n),(int4 *)scratch.c_tmp,nullptr,
            (const int4 *)p.s,nullptr,(const int4 *)p.z,nullptr,k/mq1::GROUP,m,n,k,k,
            scratch.locks,false,false,true,shared_bytes);
    }
    const size_t actual_out=size_t(b)*n;
    mq1::f16_to_f32<<<(actual_out+255)/256,256,0,stream>>>(scratch.c,y,actual_out);
    if (cudaGetLastError()!=cudaSuccess) return 1;
    ++launches;
    return 0;
}
