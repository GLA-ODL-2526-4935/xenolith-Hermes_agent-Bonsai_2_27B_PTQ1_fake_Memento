/* Bounded compute-buffer policy; does not estimate RSS or quantize state.
 * SPDX-License-Identifier: MIT */
#ifndef GGML_RUNTIME_MEMORY_POLICY_H
#define GGML_RUNTIME_MEMORY_POLICY_H
#include <stddef.h>
#include <stdint.h>

static inline int ggml_runtime_parse_mib(const char * text, size_t * bytes) {
    *bytes = 0;
    if (!text) return 1;
    if (!*text) return 0;
    size_t n = 0;
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return 0;
        const size_t digit = (size_t) (*p - '0');
        if (n > (SIZE_MAX - digit) / 10) return 0;
        n = n * 10 + digit;
    }
    if (n > SIZE_MAX / (1024 * 1024)) return 0;
    *bytes = n * 1024 * 1024;
    return 1;
}

static inline int ggml_runtime_budget_add(size_t * total, size_t value, size_t limit) {
    if (value > SIZE_MAX - *total || (limit && (*total > limit || value > limit - *total))) return 0;
    *total += value;
    return 1;
}
#endif
