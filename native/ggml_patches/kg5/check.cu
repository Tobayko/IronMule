// Fast device correctness gate. Build on a CPU host with the CUDA 12 toolkit
// or directly on a T4: nvcc -std=c++17 -O3 -arch=sm_75 check.cu -o check
#include "kg5_kernel.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

static void checked(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "KG5: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

template<typename T> static T *managed(size_t n) {
    T *p;
    checked(cudaMallocManaged(&p, n * sizeof(T)));
    return p;
}

// Two streams, independent or broadcast masks, a completely masked tile,
// a partial last tile, nonzero attention bias, padded/interleaved head strides.
static bool check(int rows, bool broadcast, unsigned seed) {
    using namespace ironmule_kg5;
    constexpr int streams = 2, heads = 2, width = 128, pitch = 144;
    const int shape = streams * heads * 4 * width;
    float *q = managed<float>(shape), *out = managed<float>(shape);
    half *k = managed<half>(streams * rows * heads * pitch);
    half *v = managed<half>(streams * rows * heads * pitch);
    half *mask = managed<half>(streams * rows);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < shape; ++i) q[i] = dist(rng) * 4;
    for (int i = 0; i < streams * rows * heads * pitch; ++i) {
        k[i] = __float2half(dist(rng)); v[i] = __float2half(dist(rng));
    }
    for (int stream = 0; stream < streams; ++stream)
        for (int row = 0; row < rows; ++row) {
            bool hidden = (row >= 32 && row < 64) || row % (3 + stream) == 2;
            mask[stream*rows+row] = __float2half(hidden ? -INFINITY : dist(rng));
        }
    view qv{reinterpret_cast<char *>(q), 0, width * sizeof(float), 4 * heads * width * sizeof(float)};
    view kv{reinterpret_cast<char *>(k), heads * pitch * sizeof(half), pitch * sizeof(half), rows * heads * pitch * sizeof(half)};
    view vv = kv;
    vv.data = reinterpret_cast<char *>(v);
    view mv{reinterpret_cast<char *>(mask), 0, 0, rows * sizeof(half)};
    args a{qv, kv, vv, mv, rows, heads, streams, broadcast ? 1 : streams, 0.125f, out};
    partial *tmp = managed<partial>(workspace(a));
    checked(launch(a, tmp, nullptr));
    checked(cudaDeviceSynchronize());
    double max_error = 0;
    for (int stream = 0; stream < streams; ++stream) {
        for (int h = 0; h < 4 * heads; ++h) {
            const int offset = (stream * 4 * heads + h) * width;
            std::vector<double> scores(rows);
            double peak = -INFINITY;
            for (int row = 0; row < rows; ++row) {
                int at = ((stream * rows + row) * heads + h / 4) * pitch;
                double dot = 0;
                for (int d = 0; d < width; ++d) dot += double(q[offset+d]) * __half2float(k[at+d]);
                scores[row] = dot * a.scale + __half2float(mask[(broadcast ? 0 : stream) * rows + row]);
                peak = std::fmax(peak, scores[row]);
            }
            double denominator = 0, numerator[width] = {};
            for (int row = 0; row < rows; ++row) {
                double w = std::exp(scores[row] - peak);
                denominator += w;
                int at = ((stream * rows + row) * heads + h / 4) * pitch;
                for (int d = 0; d < width; ++d) numerator[d] += w * __half2float(v[at+d]);
            }
            for (int d = 0; d < width; ++d) {
                double error = std::abs(out[offset+d] - numerator[d] / denominator);
                max_error = std::isfinite(error) ? std::fmax(max_error, error) : INFINITY;
            }
        }
    }
    for (void *ptr : {static_cast<void *>(q), (void *)out, (void *)k, (void *)v, (void *)mask, (void *)tmp})
        checked(cudaFree(ptr));
    std::printf("KG5 reference rows=%d broadcast=%d max_abs_error=%.9g\n", rows, broadcast, max_error);
    return max_error <= 2e-4;
}

int main() {
    return check(301, false, 17) && check(1024, true, 23) ? 0 : 1;
}
