// Count successful entries to the original generated W4 library during a request.
// CUDA graph replays can reuse captured work without re-entering these host functions.
#include <atomic>
#include <cstdint>
#include <dlfcn.h>

namespace {
std::atomic<uint64_t> calls{0};
}

extern "C" uint64_t im_prompt_dispatches() {
    return calls.load(std::memory_order_relaxed);
}

extern "C" int ironmule_kg_mul_mat(int type, const void *w, const float *x, float *y,
                                   int64_t n, int64_t k, int64_t b, void *stream) {
    static const auto original = reinterpret_cast<decltype(&ironmule_kg_mul_mat)>(
        dlsym(RTLD_NEXT, "ironmule_kg_mul_mat"));
    const int rc = original ? original(type, w, x, y, n, k, b, stream) : 1;
    if (rc == 0) calls.fetch_add(1, std::memory_order_relaxed);
    return rc;
}

extern "C" int ironmule_kg_mul_mat_group(int count, const int *types, const void *const *w,
                                         const float *x, float *const *y, const int64_t *rows,
                                         int64_t k, int64_t b, void *stream) {
    static const auto original = reinterpret_cast<decltype(&ironmule_kg_mul_mat_group)>(
        dlsym(RTLD_NEXT, "ironmule_kg_mul_mat_group"));
    const int rc = original ? original(count, types, w, x, y, rows, k, b, stream) : 1;
    if (rc == 0) calls.fetch_add(1, std::memory_order_relaxed);
    return rc;
}
