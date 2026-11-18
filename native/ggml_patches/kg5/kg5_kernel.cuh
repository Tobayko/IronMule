// KG5: one decode query per stream, four query heads sharing one F16 KV head,
// 128 components. A warp handles 32 KV rows. Splitting along context supplies
// enough independent warps to fill a T4, even for just a few KV heads.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace ironmule_kg5 {

struct view {
    const char *data;
    size_t row, head, stream;  // byte strides; components are packed
};

struct args {
    view q, k, v, mask;
    int rows, kv_heads, streams, mask_streams;
    float scale;
    float *dst; // ggml permutes query/head axes in its contiguous output
};

struct alignas(8) four_halves { half2 lo, hi; };

struct partial {
    float weighted[128], peak, mass;
};

static constexpr int rows_per_warp = 32;
static constexpr int warps_per_block = 4;

static __global__ void partials(args a, partial *tmp) {
    const int lane = threadIdx.x % 32;
    const int part = blockIdx.x * warps_per_block + threadIdx.x / 32;
    const int parts = (a.rows + rows_per_warp - 1) / rows_per_warp;
    if (part >= parts) return;
    const int kv = blockIdx.y, stream = blockIdx.z;
    const char *q = a.q.data + stream * a.q.stream + 4 * kv * a.q.head;
    const char *k = a.k.data + stream * a.k.stream + kv * a.k.head;
    const char *v = a.v.data + stream * a.v.stream + kv * a.v.head;
    const half *mask = reinterpret_cast<const half *>(a.mask.data + (stream % a.mask_streams) * a.mask.stream);
    float4 queries[4], acc[4] = {};
    float high[4], mass[4] = {};
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        queries[h] = reinterpret_cast<const float4 *>(q + h * a.q.head)[lane];
        high[h] = -CUDART_INF_F;
    }
    const int stop = min(a.rows, (part + 1) * rows_per_warp);
    for (int row = part * rows_per_warp; row < stop; ++row) {
        const float bias = __half2float(mask[row]);
        if (bias == -CUDART_INF_F) continue;
        // One coalesced load of four half values per lane, reused by every query.
        const four_halves key = reinterpret_cast<const four_halves *>(k + row * a.k.row)[lane];
        const four_halves val = reinterpret_cast<const four_halves *>(v + row * a.v.row)[lane];
        const float2 k01 = __half22float2(key.lo), k23 = __half22float2(key.hi);
        const float2 v01 = __half22float2(val.lo), v23 = __half22float2(val.hi);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float4 query = queries[h];
            float dot = query.x*k01.x + query.y*k01.y + query.z*k23.x + query.w*k23.y;
#pragma unroll
            for (int step = 16; step >= 1; step /= 2) dot += __shfl_xor_sync(0xffffffff, dot, step);
            const float score = dot * a.scale + bias;
            const float next = fmaxf(high[h], score);
            const float old = __expf(high[h] - next), weight = __expf(score - next);
            acc[h].x = acc[h].x * old + weight * v01.x;
            acc[h].y = acc[h].y * old + weight * v01.y;
            acc[h].z = acc[h].z * old + weight * v23.x;
            acc[h].w = acc[h].w * old + weight * v23.y;
            mass[h] = mass[h] * old + weight;
            high[h] = next;
        }
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        partial &p = tmp[(((stream * a.kv_heads + kv) * parts + part) * 4) + h];
        // Write four coalesced stripes; the merge reverses this permutation.
        p.weighted[lane] = acc[h].x;
        p.weighted[32 + lane] = acc[h].y;
        p.weighted[64 + lane] = acc[h].z;
        p.weighted[96 + lane] = acc[h].w;
        if (lane == 0) { p.peak = high[h]; p.mass = mass[h]; }
    }
}

static __global__ void merge(args a, const partial *tmp) {
    const int h = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int group = blockIdx.y * a.kv_heads + blockIdx.x;
    const int parts = (a.rows + rows_per_warp - 1) / rows_per_warp;
    const partial *base = tmp + group * parts * 4 + h;
    float peak = -CUDART_INF_F;
    for (int i = 0; i < parts; ++i) peak = fmaxf(peak, base[i * 4].peak);
    float4 acc = {};
    float mass = 0;
    for (int i = 0; i < parts; ++i) {
        const partial &p = base[4 * i];
        if (p.mass == 0) continue;
        const float w = __expf(p.peak - peak);
        mass += p.mass * w;
        acc.x += w * p.weighted[lane];
        acc.y += w * p.weighted[32 + lane];
        acc.z += w * p.weighted[64 + lane];
        acc.w += w * p.weighted[96 + lane];
    }
    reinterpret_cast<float4 *>(a.dst + (4 * group + h) * 128)[lane] =
        make_float4(acc.x / mass, acc.y / mass, acc.z / mass, acc.w / mass);
}

static inline size_t workspace(const args &a) {
    return size_t(a.streams) * a.kv_heads * 4 * ((a.rows + rows_per_warp - 1) / rows_per_warp);
}

static inline cudaError_t launch(const args &a, partial *tmp, cudaStream_t stream) {
    const int rows_per_block = rows_per_warp * warps_per_block;
    partials<<<dim3((a.rows + rows_per_block - 1) / rows_per_block, a.kv_heads, a.streams),
               32 * warps_per_block, 0, stream>>>(a, tmp);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    merge<<<dim3(a.kv_heads, a.streams), 4 * 32, 0, stream>>>(a, tmp);
    return cudaGetLastError();
}
} // namespace ironmule_kg5
