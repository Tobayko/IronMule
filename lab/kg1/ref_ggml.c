// KG1: llama.cpp's own CUDA kernel for the same Q4_K weights and activations (ggml_mul_mat on the CUDA backend), its
// time per matmul and its result, for gen.py's benchmark to compare against.
// usage: ref_ggml WEIGHTS.bin N K X.bin OUT_PREFIX COPIES REPS B...   (REF_TYPE: the ggml type, default 12 = Q4_K)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

static void *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(*n);
    if (fread(p, 1, *n, f) != *n) exit(1);
    fclose(f);
    return p;
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    const char *wpath = argv[1]; const int N = atoi(argv[2]), K = atoi(argv[3]);
    const char *xpath = argv[4], *prefix = argv[5]; const int C = atoi(argv[6]), REPS = atoi(argv[7]);
    size_t wn, xn;
    void *w = slurp(wpath, &wn); float *x = slurp(xpath, &xn);
    ggml_backend_t be = ggml_backend_cuda_init(0);
    const enum ggml_type type = getenv("REF_TYPE") ? (enum ggml_type)atoi(getenv("REF_TYPE")) : GGML_TYPE_Q4_K;
    for (int a = 8; a < argc; a++) {
        const int B = atoi(argv[a]);
        struct ggml_init_params ip = { ggml_tensor_overhead() * (2 * C + 8) + ggml_graph_overhead(), NULL, true };
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, B);
        struct ggml_tensor *W[64], *Y[64];
        for (int c = 0; c < C; c++) W[c] = ggml_new_tensor_2d(ctx, type, K, N);
        for (int c = 0; c < C; c++) Y[c] = ggml_mul_mat(ctx, W[c], X);
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        for (int c = 0; c < C; c++) ggml_build_forward_expand(gf, Y[c]);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        for (int c = 0; c < C; c++) ggml_backend_tensor_set(W[c], w, 0, wn);
        ggml_backend_tensor_set(X, x, 0, (size_t)B * K * sizeof(float));
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
        const double t0 = now();
        for (int i = 0; i < REPS; i++) ggml_backend_graph_compute(be, gf);
        const double us = (now() - t0) * 1e6 / ((double)REPS * C);
        float *y = malloc((size_t)N * B * sizeof(float));
        ggml_backend_tensor_get(Y[0], y, 0, (size_t)N * B * sizeof(float));
        char path[512]; snprintf(path, sizeof path, "%s_B%d.bin", prefix, B);
        FILE *f = fopen(path, "wb"); fwrite(y, sizeof(float), (size_t)N * B, f); fclose(f);
        printf("{\"ggml\": true, \"type\": %d, \"N\": %d, \"K\": %d, \"B\": %d, \"us\": %.2f, \"GBps\": %.1f}\n",
               (int)type, N, K, B, us, wn / us / 1e3);
        fflush(stdout);
        free(y);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
