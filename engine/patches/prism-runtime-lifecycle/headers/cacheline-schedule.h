/* Xenolith CPU partition experiment. Independent implementation informed by
 * llama.cpp-ng's cache-line scheduling; arithmetic and packing are unchanged.
 * SPDX-License-Identifier: MIT */
#ifndef GGML_CACHELINE_SCHEDULE_H
#define GGML_CACHELINE_SCHEDULE_H
#include <stddef.h>
#include <stdint.h>
#ifndef GGML_CPU_CACHELINE_MIN_BYTES
#define GGML_CPU_CACHELINE_MIN_BYTES 65536
#endif

/* One complete F32 output column only. Return 0 for unsupported alignment.
 * Partition physical lines, including a partial first line. No tensor storage,
 * scratch, threadpool state or reduction order is changed here. */
static inline int ggml_cacheline_range(uintptr_t base, int64_t rows,
        size_t weight_row_bytes, int nth, int ith, int64_t * begin, int64_t * end) {
    if (!begin || !end || rows < 0 || nth < 1 || ith < 0 || ith >= nth ||
        base % sizeof(float) || weight_row_bytes == 0) return 0;
    *begin = *end = 0;
    if (!rows) return 1;
    const uint64_t r = (uint64_t) rows;
    const uint64_t head = ((64 - base % 64) % 64) / sizeof(float);
    const uint64_t first = head < r ? head : r;
    const uint64_t remaining = r - first;
    const uint64_t units = (first ? 1 : 0) + remaining / 16 + (remaining % 16 != 0);
    const uint64_t minimum = GGML_CPU_CACHELINE_MIN_BYTES > 0 ? GGML_CPU_CACHELINE_MIN_BYTES : 1;
    const uint64_t rows_per_worker = minimum / weight_row_bytes + (minimum % weight_row_bytes != 0);
    uint64_t workers = r / rows_per_worker;
    if (!workers) workers = 1;
    if (workers > (uint64_t) nth) workers = (uint64_t) nth;
    int snap = 1;
    if (units < workers) {
        if (weight_row_bytes >= 16384) snap = 0; /* few long rows: keep parallel work */
        else workers = units;
    }
    if ((uint64_t) ith >= workers) return 1;
    /* quotient/remainder avoids multiplication overflow for huge row counts */
    const uint64_t count = snap ? units : r;
    const uint64_t lo = count / workers * (uint64_t) ith +
        (count % workers * (uint64_t) ith) / workers;
    const uint64_t hi = count / workers * (uint64_t) (ith + 1) +
        (count % workers * (uint64_t) (ith + 1)) / workers;
    if (!snap) { *begin = (int64_t) lo; *end = (int64_t) hi; return 1; }
    const uint64_t a = lo == 0 ? 0 : first ? first + (lo - 1) * 16 : lo * 16;
    const uint64_t b = hi == 0 ? 0 : first ? first + (hi - 1) * 16 : hi * 16;
    *begin = (int64_t) (a < r ? a : r);
    *end = (int64_t) (b < r ? b : r);
    return 1;
}
#endif
