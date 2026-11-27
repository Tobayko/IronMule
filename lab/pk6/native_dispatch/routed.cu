// Bind a CPU Pilot decision to each qualified interval before the server builds its GPU graphs.
// target_rows.h is generated from a measurement archive by learn.py.
#define ironmule_kg_mul_mat im_prompt_unrouted
#include "prompt_fp8.cu"
#undef ironmule_kg_mul_mat
#include "target_rows.h"

extern "C" int ironmule_kg_mul_mat(int type, const void *w, const float *x, float *y,
                                  int64_t n, int64_t k, int64_t rows, cudaStream_t stream) {
    for (const auto &entry : qualified_prompt_rows) {
        if (entry.type == type && entry.n == n && entry.k == k && entry.low <= rows && rows <= entry.high) {
            return im_prompt_unrouted(type, w, x, y, n, k, rows, stream);
        }
    }
    return 1;
}
