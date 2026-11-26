// Information experiment only. Reproduce the dense PK2 GEMM contract outside the serving process.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CUDA_OK(call) do { auto rc = (call); if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); } while (0)
#define BLAS_OK(call) do { auto rc = (call); if (rc != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(#call); } while (0)
constexpr size_t workspace_bytes = 32u << 20;

__global__ void fill(half *dst, size_t n, uint32_t seed) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (i < n) {
        uint32_t v = uint32_t(i) + seed;
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        v *= 0x846ca68bu;
        v ^= v >> 16;
        dst[i] = __float2half_rn(float(int(v & 0xffffff) - 0x800000) * 0x1p-23f);
    }
}
__global__ void compare(const float *ref, const float *test, size_t n, unsigned *errors) {
    const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    bool bad = i < n && (!isfinite(ref[i]) || !isfinite(test[i]) || __float_as_uint(ref[i]) != __float_as_uint(test[i]));
    unsigned votes = __ballot_sync(0xffffffffu, bad);
    if (votes && (threadIdx.x & 31) == 0) atomicAdd(errors, __popc(votes));
}

template<class F> static float elapsed(cudaEvent_t begin, cudaEvent_t end, F work) {
    CUDA_OK(cudaEventRecord(begin));
    BLAS_OK(work());
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
    if (!(ms > 0)) throw std::runtime_error("invalid CUDA duration");
    return ms;
}

int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("expected M N K seed");
        int m = std::stoi(argv[1]), n = std::stoi(argv[2]), k = std::stoi(argv[3]);
        unsigned seed = std::stoul(argv[4]);
        if (m <= 0 || n <= 0 || k <= 0 || m > 32768 || n > 8192 || k > 32768) throw std::runtime_error("shape bound");
        const size_t ak = size_t(m) * k, bk = size_t(n) * k, cells = size_t(m) * n;
        const size_t need = (ak + bk) * sizeof(half) + 2 * cells * sizeof(float) + workspace_bytes;
        size_t free_bytes, total;
        CUDA_OK(cudaMemGetInfo(&free_bytes, &total));
        if (need > (size_t(3) << 30) || need + (size_t(1) << 30) > free_bytes) throw std::runtime_error("memory bound");
        cublasHandle_t h;
        cublasLtHandle_t lt;
        BLAS_OK(cublasCreate(&h));
        BLAS_OK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
        BLAS_OK(cublasLtCreate(&lt));
        cublasLtMatmulDesc_t op;
        BLAS_OK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans = CUBLAS_OP_T;
        BLAS_OK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
        cublasLtMatrixLayout_t a_layout, b_layout, out_layout;
        BLAS_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16F, k, m, k));
        BLAS_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16F, k, n, k));
        BLAS_OK(cublasLtMatrixLayoutCreate(&out_layout, CUDA_R_32F, m, n, m));
        cublasLtMatmulPreference_t preference;
        BLAS_OK(cublasLtMatmulPreferenceCreate(&preference));
        BLAS_OK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_bytes, sizeof(workspace_bytes)));
        half *weights, *activations;
        float *ref, *test;
        void *workspace;
        unsigned *mismatches;
        CUDA_OK(cudaMalloc(&weights, ak * sizeof(half)));
        CUDA_OK(cudaMalloc(&activations, bk * sizeof(half)));
        CUDA_OK(cudaMalloc(&ref, cells * sizeof(float)));
        CUDA_OK(cudaMalloc(&test, cells * sizeof(float)));
        CUDA_OK(cudaMalloc(&workspace, workspace_bytes));
        CUDA_OK(cudaMalloc(&mismatches, sizeof(unsigned)));
        fill<<<(ak + 255) / 256, 256>>>(weights, ak, seed);
        fill<<<(bk + 255) / 256, 256>>>(activations, bk, seed ^ 0x8ec614a3u);
        CUDA_OK(cudaGetLastError());
        const float alpha = 1, beta = 0;
        auto reference = [&] { return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                weights, CUDA_R_16F, k, activations, CUDA_R_16F, k, &beta, ref, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
        cublasLtMatmulHeuristicResult_t choices[4];
        int count = 0, version;
        BLAS_OK(cublasGetVersion(h, &version));
        BLAS_OK(cublasLtMatmulAlgoGetHeuristic(lt, op, a_layout, b_layout, out_layout, out_layout,
                                               preference, 4, choices, &count));
        cudaEvent_t begin, end;
        CUDA_OK(cudaEventCreate(&begin));
        CUDA_OK(cudaEventCreate(&end));
        elapsed(begin, end, reference);
        std::printf("{\"kind\":\"info\",\"heuristics\":%d,\"cublas_version\":%d,\"requested_bytes\":%zu}\n",
                    count, version, need);
        for (int j = 0; j < count; j++) {
            const auto &choice = choices[j];
            if (choice.state != CUBLAS_STATUS_SUCCESS || choice.workspaceSize > workspace_bytes) {
                std::printf("{\"index\":%d,\"status\":\"ineligible\",\"rc\":%d}\n", j, int(choice.state));
                continue;
            }
            auto candidate = [&] { return cublasLtMatmul(lt, op, &alpha, weights, a_layout, activations, b_layout,
                    &beta, test, out_layout, test, out_layout, &choice.algo, workspace, workspace_bytes, nullptr); };
            int id;
            size_t written;
            BLAS_OK(cublasLtMatmulAlgoConfigGetAttribute(&choice.algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written));
            // NaN-fill catches a partial writer even when another algorithm ran before it.
            CUDA_OK(cudaMemset(test, 0xff, cells * sizeof(float)));
            auto rc = candidate();
            if (rc != CUBLAS_STATUS_SUCCESS) {
                std::printf("{\"index\":%d,\"algo_id\":%d,\"status\":\"declined\",\"rc\":%d}\n", j, id, int(rc));
                continue;
            }
            auto errors = [&] {
                CUDA_OK(cudaMemset(mismatches, 0, sizeof(unsigned)));
                compare<<<(cells + 255) / 256, 256>>>(ref, test, cells, mismatches);
                unsigned wrong;
                CUDA_OK(cudaMemcpy(&wrong, mismatches, sizeof(wrong), cudaMemcpyDeviceToHost));
                return wrong;
            };
            auto wrong = errors();
            if (wrong) {
                std::printf("{\"index\":%d,\"algo_id\":%d,\"status\":\"changed_output\",\"mismatches\":%u}\n", j, id, wrong);
                continue;
            }
            for (int p = 0; p < 4; p++) {
                float r, t;
                if (p % 2 == 0) {
                    r = elapsed(begin, end, reference);
                    t = elapsed(begin, end, candidate);
                } else {
                    t = elapsed(begin, end, candidate);
                    r = elapsed(begin, end, reference);
                }
                if (errors()) throw std::runtime_error("nondeterministic or changed timed output");
                std::printf("{\"index\":%d,\"algo_id\":%d,\"workspace\":%zu,\"status\":\"paired\","
                            "\"pair\":%d,\"reference_ms\":%.9g,\"candidate_ms\":%.9g}\n", j, id,
                            choice.workspaceSize, p, r, t);
            }
        }
        // One bounded process per shape. Its CUDA context owns all buffers and library handles until exit.
        CUDA_OK(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
