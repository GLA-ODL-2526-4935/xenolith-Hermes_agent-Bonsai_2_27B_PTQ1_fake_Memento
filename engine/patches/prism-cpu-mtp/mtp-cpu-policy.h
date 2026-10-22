#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

// One sequence, one request. Bounded exploration, measured microseconds per
// emitted token (accepted drafts + the target token), not acceptance ratio.
struct common_mtp_cpu_policy {
    int cap = 1;
    uint64_t cycles = 0;
    std::array<int, 5> samples{};
    std::array<double, 5> cost{};

    void reset(int maximum) {
        cap = std::max(1, std::min(4, maximum));
        cycles = 0;
        samples.fill(0);
        cost.fill(0.0);
    }

    int choose() const {
        for (int n = 1; n <= cap; ++n) {
            if (samples[n] < 4) {
                return n;
            }
        }
        // Periodically revisit alternatives when the workload changes.
        if (cycles % 32 == 0) {
            return 1 + (int) ((cycles / 32) % cap);
        }
        int best = 1;
        for (int n = 2; n <= cap; ++n) {
            if (cost[n] < cost[best]) {
                best = n;
            }
        }
        return best;
    }

    void observe(int window, int drafted, int accepted, int64_t elapsed_us) {
        // Context/output caps and confidence truncation are not full samples.
        if (window < 1 || window > cap || drafted != window || accepted < 0 ||
                accepted > drafted || elapsed_us <= 0) {
            return;
        }
        const double value = (double) elapsed_us / (accepted + 1);
        cost[window] = samples[window] ? 0.75 * cost[window] + 0.25 * value : value;
        if (samples[window] < std::numeric_limits<int>::max()) {
            ++samples[window];
        }
        ++cycles;
    }
};
