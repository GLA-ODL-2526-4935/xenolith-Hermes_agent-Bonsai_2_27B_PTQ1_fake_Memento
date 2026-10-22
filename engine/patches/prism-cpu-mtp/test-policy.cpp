#include "mtp-cpu-policy.h"
#include <cassert>

int main() {
    common_mtp_cpu_policy p;
    p.reset(4);
    // A longer window can have higher acceptance and still cost more per token.
    for (int n = 1; n <= 4; ++n) {
        for (int sample = 0; sample < 4; ++sample) {
            assert(p.choose() == n);
            p.observe(n, n, n, n == 2 ? 60 : (n + 1) * 100);
        }
    }
    assert(p.choose() == 2);
    const auto old = p.cycles;
    p.observe(4, 2, 2, 100); // shortened by context: do not bias window4
    p.observe(2, 2, 3, 100); // impossible acceptance
    p.observe(2, 2, 2, 0);
    assert(p.cycles == old);
    for (int i = 0; i < 500; ++i) {
        int n = p.choose();
        assert(n >= 1 && n <= 4);
        p.observe(n, n, n, n == 2 ? 60 : (n + 1) * 100);
    }
    assert(p.samples[1] > 4 && p.samples[3] > 4 && p.samples[4] > 4);
    // Changing workload: repeated slow observations move selection away from2.
    for (int i = 0; i < 32; ++i) {
        p.observe(2, 2, 0, 100000);
    }
    assert(p.choose() != 2);
    p.reset(1);
    assert(p.choose() == 1 && p.cycles == 0 && p.samples[2] == 0);
    p.reset(100);
    assert(p.cap == 4);
    p.reset(0);
    assert(p.cap == 1);
    // Copy-elision predicate: a replay can only occur if a clone was retained.
    for (int draft = 1; draft <= 64; ++draft) {
        for (int window = 0; window <= 64; ++window) {
            for (int accepted = 1; accepted <= draft + 1; ++accepted) {
                const int rollback = draft + 1 - accepted;
                assert(!(rollback > window) || draft > window);
            }
        }
    }
}
