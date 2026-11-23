// The GPU's achievable read bandwidth: every byte of a large buffer read once with 16-byte streaming loads.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void read_all(const uint4 *__restrict__ p, size_t n, unsigned *sink) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keeps the loads
}

extern "C" double bw_probe(size_t bytes, int reps) {  // GB/s, best of reps
    uint4 *p = nullptr;
    unsigned *sink = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess || cudaMalloc(&sink, 4) != cudaSuccess) return -1;
    cudaMemset(p, 1, bytes);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    double best = 0;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a);
        read_all<<<sms * 8, 512>>>(p, bytes / 16, sink);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (r > 0 && bytes / (ms * 1e6) > best) best = bytes / (ms * 1e6);
    }
    cudaFree(p);
    cudaFree(sink);
    return best;
}
