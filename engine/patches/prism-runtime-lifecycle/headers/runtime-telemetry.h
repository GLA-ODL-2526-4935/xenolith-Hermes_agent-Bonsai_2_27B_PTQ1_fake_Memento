/* Opt-in fixed-size aggregate counters: no prompts, tokens or unbounded traces.
 * SPDX-License-Identifier: MIT */
#ifndef GGML_RUNTIME_TELEMETRY_H
#define GGML_RUNTIME_TELEMETRY_H
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct ggml_runtime_stats {
    uint64_t alloc_calls = 0, alloc_requested_bytes = 0, alloc_failures = 0;
    uint64_t copy_calls = 0, copy_payload_bytes = 0, sync_calls = 0;
    uint64_t verify_cycles = 0, drafted = 0, matched = 0, rollback_tokens = 0, replay_cycles = 0;
};

class ggml_runtime_telemetry {
    std::mutex mutex;
    ggml_runtime_stats stats;
    const char * label;
    uint64_t events = 0;
    static void add(uint64_t & dst, uint64_t n) { dst = n > UINT64_MAX - dst ? UINT64_MAX : dst + n; }
    void log_locked() const {
        std::fprintf(stderr, "xeno_runtime[%s] alloc_calls=%llu alloc_requested_bytes=%llu alloc_failures=%llu "
            "copy_calls=%llu copy_payload_bytes=%llu sync_calls=%llu verify_cycles=%llu drafted=%llu matched=%llu "
            "rollback_tokens=%llu replay_cycles=%llu\n", label,
            (unsigned long long)stats.alloc_calls, (unsigned long long)stats.alloc_requested_bytes,
            (unsigned long long)stats.alloc_failures, (unsigned long long)stats.copy_calls,
            (unsigned long long)stats.copy_payload_bytes, (unsigned long long)stats.sync_calls,
            (unsigned long long)stats.verify_cycles, (unsigned long long)stats.drafted,
            (unsigned long long)stats.matched, (unsigned long long)stats.rollback_tokens,
            (unsigned long long)stats.replay_cycles);
    }
    void event_locked() {
        add(events, 1);
        if ((events & (events - 1)) == 0) log_locked();
    }
public:
    const bool enabled;
    explicit ggml_runtime_telemetry(const char * name) : label(name), enabled([] {
        const char * p = std::getenv("GGML_RUNTIME_TELEMETRY");
        return p && std::strcmp(p, "1") == 0;
    }()) {}
    void allocation(uint64_t bytes, bool failed) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        add(stats.alloc_calls, 1); add(stats.alloc_requested_bytes, bytes);
        if (failed) add(stats.alloc_failures, 1);
        event_locked();
    }
    void copy(uint64_t bytes) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        add(stats.copy_calls, 1); add(stats.copy_payload_bytes, bytes); event_locked();
    }
    void sync() {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        add(stats.sync_calls, 1); event_locked();
    }
    void verify(uint64_t n, uint64_t accepted, uint64_t rollback, bool replay) {
        if (!enabled) return;
        std::lock_guard<std::mutex> lock(mutex);
        add(stats.verify_cycles, 1); add(stats.drafted, n); add(stats.matched, accepted);
        add(stats.rollback_tokens, rollback); if (replay) add(stats.replay_cycles, 1);
        event_locked();
    }
    ggml_runtime_stats snapshot() { std::lock_guard<std::mutex> lock(mutex); return stats; }
    void report() { if (enabled) { std::lock_guard<std::mutex> lock(mutex); log_locked(); } }
};
#endif
