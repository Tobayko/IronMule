// Optional immutable GGUF weight cache for the dedicated one-model Ada study.
// All use and teardown share one lock. Foreign streams and nested model lifetimes use stock.
#pragma once
#include <mutex>
namespace persistent_prompt {
struct Entry { unsigned char *weights; float *scales; };
static std::mutex mutex;
static std::map<std::tuple<const void*,int,int,int>,Entry> entries;
static size_t bytes=0;
static int sessions=0, owner=-1;
static cudaStream_t only_stream=nullptr;
static void reset() {
    if (owner>=0) {
        int previous=-1;
        cudaGetDevice(&previous);
        cudaSetDevice(owner);
        cudaDeviceSynchronize();
        for (auto &item:entries) { cudaFree(item.second.weights); cudaFree(item.second.scales); }
        if (previous>=0) cudaSetDevice(previous);
    }
    entries.clear(); bytes=0; owner=-1; only_stream=nullptr;
}
static Entry *get(const void *raw,int type,int n,int k,int device,cudaStream_t stream,bool &fresh,int element_bytes=1) {
    if (sessions!=1 || (owner>=0 && (device!=owner || stream!=only_stream))) return nullptr;
    auto key=std::make_tuple(raw,type,n,k);
    auto found=entries.find(key);
    if (found!=entries.end()) { fresh=false; return &found->second; }
    // Only cache FFN projections. Small attention kernels have a different bandwidth crossover.
    if (n!=25600 && k!=25600) return nullptr;
    const size_t matrix_bytes=size_t(n)*k*element_bytes;
    const size_t scale_bytes=element_bytes==1 ? size_t(n)*sizeof(float) : 0;
    const size_t requested=matrix_bytes+scale_bytes;
    constexpr size_t cap=size_t{7}<<30;
    size_t free=0,total=0;
    if (bytes+requested>cap || cudaMemGetInfo(&free,&total)!=cudaSuccess || free<requested+(size_t{4}<<30)) return nullptr;
    Entry entry{};
    if (cudaMalloc(&entry.weights,matrix_bytes)!=cudaSuccess || (scale_bytes && cudaMalloc(&entry.scales,scale_bytes)!=cudaSuccess)) {
        cudaFree(entry.weights); cudaFree(entry.scales); cudaGetLastError(); return nullptr;
    }
    fresh=true; owner=device; only_stream=stream; bytes+=requested;
    return &entries.emplace(key,entry).first->second;
}
}
extern "C" void im_prompt_begin() {
    std::lock_guard<std::mutex> lock(persistent_prompt::mutex);
    persistent_prompt::reset(); ++persistent_prompt::sessions;
}
extern "C" void im_prompt_end() {
    std::lock_guard<std::mutex> lock(persistent_prompt::mutex);
    persistent_prompt::reset();
    if (persistent_prompt::sessions>0) --persistent_prompt::sessions;
}
extern "C" size_t im_prompt_bytes() {
    std::lock_guard<std::mutex> lock(persistent_prompt::mutex);
    return persistent_prompt::bytes;
}
