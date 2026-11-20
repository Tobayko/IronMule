// IronMule MQ1: the parts of torch the vendored vLLM Marlin sources use (TORCH_CHECK), without torch
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <variant>

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...)                                              \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "MQ1 check failed: %s (%s:%d)\n", #cond,   \
                         __FILE__, __LINE__);                               \
            std::abort();                                                   \
        }                                                                   \
    } while (0)
#endif
