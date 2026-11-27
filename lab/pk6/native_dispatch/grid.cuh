// A single Q4_K translation unit. Native tile validation is still authoritative.
// Explore column tiling and the upstream XY grid versus its persistent Stream-K grid.
#include <atomic>
#include <cstdlib>
#include <cstring>
#include "grid_ranges.h"

static std::atomic<uint64_t> grid_calls{0};
extern "C" uint64_t im_mmq_tile_calls() { return grid_calls.load(std::memory_order_relaxed); }

static int grid_mode(const mmq_args &args, int cc) {
    if (cc != GGML_CUDA_CC_ADA_LOVELACE || args.ids_dst || args.nchannels_y != 1 || args.nsamples_y != 1) return 0;
    static const int option = []() {
        const char *p = std::getenv("IRONMULE_LAB_MMQ_TILE");
        if (!p) return 0;
        if (std::strcmp(p, "grid") == 0) return -1;
        if (std::strcmp(p, "xy") == 0) return 1;
        if (std::strcmp(p, "xy64") == 0) return 2;
        if (std::strcmp(p, "sk64") == 0) return 3;
        return 0;
    }();
    if (option != -1) return option;
    for (const auto &r : measured_grid_ranges) {
        if (args.nrows_x == r.n && args.ncols_x == r.k && args.ncols_y >= r.lo && args.ncols_y <= r.hi)
            return r.mode;
    }
    return 0;
}
