// Contextual bandit for GGUF placement. Idle comparisons observe every eligible
// action for the same fixture, so the gradient of expected reward is available
// without sampling an action on a caller's request.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {
using Features = std::array<double, 5>;

Features features(double chars, double limit, double queue) {
    const double prompt = std::log1p(chars) / 8.0;
    const double output = std::log1p(limit) / 5.0;
    return {1.0, prompt, output, queue, prompt * output};
}

double score(const double* weights, const Features& x) {
    double result = 0;
    for (size_t k = 0; k < x.size(); ++k) result += weights[k] * x[k];
    return result;
}
}

// For each measured case and worker: chars, output limit, queue depth, completion
// milliseconds, first-token milliseconds, and agreement with the reference response.
extern "C" int placement_reward_fit(const double* rows, size_t cases, size_t lanes, double* weights) {
    if (!rows || !weights || !lanes || !cases || cases > 1000000 / lanes) return 0;
    std::fill_n(weights, lanes * 5, 0.0);
    std::vector<Features> x(cases * lanes);
    std::vector<double> rewards(cases * lanes), probabilities(lanes);
    for (size_t c = 0; c < cases; ++c) {
        const double* row = rows + 6 * c * lanes;
        size_t reference = 0;
        for (size_t i = 0; i < lanes; ++i) {
            const double* arm = row + 6 * i;
            for (int k = 0; k < 6; ++k) if (!std::isfinite(arm[k])) return 0;
            if (arm[0] < 1 || arm[1] < 1 || arm[2] < 0 || arm[4] <= 0 || arm[3] < arm[4]
                    || (arm[5] != 0 && arm[5] != 1)) return 0;
            x[c * lanes + i] = features(arm[0], arm[1], arm[2]);
            if (arm[2] < row[6 * reference + 2]) reference = i;
        }
        const double* base = row + 6 * reference;
        for (size_t i = 0; i < lanes; ++i) {
            const double* arm = row + 6 * i;
            const double gain = std::log(base[3]) - std::log(arm[3]);
            const double wait = std::max(0.0, std::log(arm[4]) - std::log(base[4]));
            rewards[c * lanes + i] = arm[5] ? std::clamp(gain - wait, -1.0, 1.0) : -2.0;
        }
    }
    for (int epoch = 0; epoch < 32; ++epoch) {
        for (size_t c = 0; c < cases; ++c) {
            for (size_t i = 0; i < lanes; ++i)
                probabilities[i] = score(weights + 5 * i, x[c * lanes + i]);
            const double top = *std::max_element(probabilities.begin(), probabilities.end());
            double sum = 0, mean = 0;
            for (size_t i = 0; i < lanes; ++i) {
                probabilities[i] = std::exp(probabilities[i] - top);
                sum += probabilities[i];
                mean += probabilities[i] * rewards[c * lanes + i];
            }
            mean /= sum;
            for (size_t i = 0; i < lanes; ++i) {
                const double update = 0.08 * probabilities[i] / sum * (rewards[c * lanes + i] - mean);
                for (size_t k = 0; k < 5; ++k) weights[5 * i + k] += update * x[c * lanes + i][k];
            }
        }
    }
    return 1;
}

extern "C" int placement_reward_choose(const double* weights, const double* queues, int lanes,
                                        double chars, double limit) {
    if (!weights || !queues || lanes < 1) return -1;
    int choice = 0;
    double best = score(weights, features(chars, limit, queues[0]));
    for (int i = 1; i < lanes; ++i) {
        const double value = score(weights + 5 * i, features(chars, limit, queues[i]));
        if (value > best) {
            best = value;
            choice = i;
        }
    }
    return choice;
}
