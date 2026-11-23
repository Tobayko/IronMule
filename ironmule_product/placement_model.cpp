// The controller trains and scores on a CPU core even when the LLM runs on CUDA.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace {
constexpr int width = 5;
using Features = std::array<double, width>;

Features features(double chars, double limit, double queue) {
    const double prompt = std::log1p(chars) / 8.0;
    const double output = std::log1p(limit) / 5.0;
    return {1.0, prompt, output, std::log1p(queue), prompt * output};
}
}

// Rows contain prompt characters, output limit, outstanding requests and service milliseconds.
// Fit log latency with ridge regression. A bounded problem of five coefficients needs no BLAS.
extern "C" int placement_fit(const double* rows, std::size_t count, double* weights) {
    if (!rows || !weights || count == 0 || count > 1000000) return 0;
    double system[width][width + 1]{};
    for (std::size_t i = 0; i < count; ++i) {
        const double* row = rows + 4 * i;
        for (int j = 0; j < 4; ++j) if (!std::isfinite(row[j]) || row[j] < 0) return 0;
        if (row[0] > 4194304 || row[1] < 1 || row[1] > 8192 ||
            row[2] > 65536 || row[3] <= 0 || row[3] > 86400000) return 0;
        const auto x = features(row[0], row[1], row[2]);
        for (int a = 0; a < width; ++a) {
            for (int b = 0; b < width; ++b) system[a][b] += x[a] * x[b];
            system[a][width] += x[a] * std::log(row[3]);
        }
    }
    for (int a = 0; a < width; ++a) system[a][a] += a == 0 ? 1e-8 : 0.02;
    for (int c = 0; c < width; ++c) {
        int pivot = c;
        for (int a = c + 1; a < width; ++a)
            if (std::abs(system[a][c]) > std::abs(system[pivot][c])) pivot = a;
        if (pivot != c) std::swap_ranges(system[c], system[c] + width + 1, system[pivot]);
        const double divisor = system[c][c];
        if (!std::isfinite(divisor) || std::abs(divisor) < 1e-14) return 0;
        for (int b = c; b <= width; ++b) system[c][b] /= divisor;
        for (int a = 0; a < width; ++a) {
            if (a == c) continue;
            const double factor = system[a][c];
            for (int b = c; b <= width; ++b) system[a][b] -= factor * system[c][b];
        }
    }
    for (int a = 0; a < width; ++a) weights[a] = system[a][width];
    return 1;
}

extern "C" double placement_ms(const double* weights, double chars, double limit, double queue) {
    const auto x = features(chars, limit, queue);
    double value = 0.0;
    for (int a = 0; a < width; ++a) value += x[a] * weights[a];
    return std::exp(std::clamp(value, -6.0, 18.0));
}

// Price one more arrival on each possible lane. The measured queue domain is empty or one ahead.
// Keeping the two best unmodified next-hop costs makes a full counterfactual pass linear in lane count.
extern "C" int placement_plan(const double* weights, const double* queues, int lanes,
                              double chars, double limit, double next_chars, double next_limit,
                              double future_weight) {
    if (!weights || !queues || lanes < 1) return -1;
    int first = -1;
    double best_next = INFINITY, second_next = INFINITY;
    for (int i = 0; i < lanes; ++i) {
        const double next = placement_ms(weights + width * i, next_chars, next_limit, queues[i]);
        if (next < best_next) {
            second_next = best_next;
            best_next = next;
            first = i;
        } else {
            second_next = std::min(second_next, next);
        }
    }
    int choice = 0;
    double best = INFINITY;
    for (int i = 0; i < lanes; ++i) {
        const double* lane = weights + width * i;
        const double current = placement_ms(lane, chars, limit, queues[i]);
        const double occupied = placement_ms(lane, next_chars, next_limit, 1.0);
        const double other = i == first ? second_next : best_next;
        const double cost = current + future_weight * std::min(occupied, other);
        if (cost < best) {
            choice = i;
            best = cost;
        }
    }
    return choice;
}
