// Included by the vendored CUDA backend or the standalone Linux measurement probe.
#pragma once
#include "kg5_kernel.cuh"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>

static std::atomic<uint64_t> kg5_counter{0};
extern "C" __attribute__((visibility("default"))) uint64_t ironmule_kg5_launches() {
    return kg5_counter.load(std::memory_order_relaxed);
}

static bool kg5_packed(const ggml_tensor *t, size_t alignment, size_t element) {
    return t->nb[0] == element && (reinterpret_cast<uintptr_t>(t->data) |
           t->nb[1] | t->nb[2] | t->nb[3]) % alignment == 0;
}

static bool kg5_try(ggml_backend_cuda_context &ctx, ggml_tensor *dst, int cc) {
    const char *enabled = std::getenv("IRONMULE_KG5");
    if (!enabled || std::strcmp(enabled, "1") != 0 || cc != GGML_CUDA_CC_TURING) return false;
    const char *shape = std::getenv("IRONMULE_KG5_SHAPE");
    if (!shape || std::strcmp(shape, "1024,32,8") != 0) return false;
    const ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mask = dst->src[3];
    if (!q || !k || !v || !mask || dst->src[4] ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
        q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128 || q->ne[1] != 1 ||
        k->ne[1] != 1024 || v->ne[1] != k->ne[1] ||
        k->ne[2] != 8 || v->ne[2] != k->ne[2] || q->ne[2] != 4*k->ne[2] ||
        q->ne[3] != 32 || q->ne[3] != k->ne[3] || v->ne[3] != k->ne[3] ||
        mask->ne[0] < k->ne[1] || mask->ne[2] != 1 || mask->ne[3] < 1 || q->ne[3] % mask->ne[3] ||
        !kg5_packed(q, 16, sizeof(float)) || !kg5_packed(k, 8, sizeof(half)) || !kg5_packed(v, 8, sizeof(half)) ||
        !kg5_packed(mask, 2, sizeof(half)) || !kg5_packed(dst, 16, sizeof(float)) || !ggml_is_contiguous(dst))
        return false;

    float params[3];
    std::memcpy(params, dst->op_params, sizeof(params));
    if (params[1] != 0.0f || params[2] != 0.0f) return false; // ALiBi and logit soft-capping use the stock path
    auto view_of = [](const ggml_tensor *t) -> ironmule_kg5::view {
        return {static_cast<const char *>(t->data), t->nb[1], t->nb[2], t->nb[3]};
    };
    ironmule_kg5::args a = {view_of(q), view_of(k), view_of(v), view_of(mask), int(k->ne[1]), int(k->ne[2]),
                            int(q->ne[3]), int(mask->ne[3]), params[0], static_cast<float *>(dst->data)};
    if (ironmule_kg5::workspace(a) * sizeof(ironmule_kg5::partial) > 64 * 1024 * 1024)
        return false;
    ggml_cuda_pool_alloc<ironmule_kg5::partial> tmp(ctx.pool(), ironmule_kg5::workspace(a));
    CUDA_CHECK(ironmule_kg5::launch(a, tmp.ptr, ctx.stream()));
    if (kg5_counter.fetch_add(1, std::memory_order_relaxed) == 0) {
        const char *trace = std::getenv("IRONMULE_KG5_TRACE");
        if (trace && std::strcmp(trace, "1") == 0)
            std::fprintf(stderr, "KG5 launch pid=%ld rows=%d streams=%d\n", long(getpid()), a.rows, a.streams);
    }
    return true;
}
