// Source-local draft selection adaptation. Upstream llama.cpp MIT license.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// 0 = warm off, 1..4 = MTP, 5..8 = suffix chain of length arm-4.
// Request-local and deterministic: never consumes target sampler randomness.
struct common_source_policy {
    struct stat { uint64_t tokens=0, us=0, samples=0; };
    std::array<stat,9> stats{};
    uint64_t cycles=0;
    int arm=0;
    int64_t started=0;
    bool active=false;
    int context_bucket=-1;
    void reset() { *this=common_source_policy{}; }
    int choose(int maximum, bool suffix, bool explore=true) const {
        const int cap=std::max(0,std::min(4,maximum));
        if (!cap || stats[0].samples<2) return 0;
        // One bounded exploration cycle per 16 completed cycles. Short outputs
        // stay mostly ordinary; cooled-down losers get a deterministic recheck.
        if (explore && cycles%16==2) {
            const int count=cap*(suffix?2:1);
            const int probe=(int)((cycles/16)%(count+1));
            if (probe==count) return 0; // periodically refresh the off baseline
            return probe<cap ? probe+1 : probe-cap+5;
        }
        int best=0;
        const double base=(double)stats[0].tokens/stats[0].us;
        double best_score=base;
        for (int a=1;a<9;++a) {
            if ((a>4 && !suffix) || (a>4?a-4:a)>cap || stats[a].samples<2) continue;
            const double rate=(double)stats[a].tokens/stats[a].us;
            // Small bounded optimistic bonus, never enough to promote a
            // measured arm slower than off. This is not a regret guarantee.
            const double bonus=std::min(0.02,std::sqrt(std::log((double)cycles+1)/stats[a].samples)*0.005);
            if (rate>base*1.05 && rate*(1+bonus)>best_score) { best=a; best_score=rate*(1+bonus); }
        }
        return best;
    }
    void begin(int cap, bool suffix, int64_t now, int bucket=-1) {
        if (active) return; // full checkpoint replay belongs to the same cycle
        if (bucket!=context_bucket) { reset(); context_bucket=bucket; }
        arm=choose(cap,suffix); started=now; active=true;
    }
    bool finish(int committed, int64_t now) {
        if (!active) return false;
        active=false;
        if (committed<=0 || now<=started) return false; // censored failure/cancel
        auto & s=stats[arm];
        s.tokens+=(uint64_t)committed; s.us+=(uint64_t)(now-started); ++s.samples; ++cycles;
        return true;
    }
};

// No persistent second context or suffix tree. Borrow the current committed
// prompt and scan its last 8191 tokens plus id_last, in one request only.
// Longest exact suffix, majority continuation, then conditional prefix counts.
// Bounded scratch: <=8192 int32 tokens +8192 size_t positions +8192 int32 votes.
// This is a scored chain adaptation, not the paper's tree implementation.
template<class Token>
static std::vector<Token> common_suffix_chain(const std::vector<Token> & prompt,
        Token last, int maximum) {
    maximum=std::max(0,std::min(4,maximum));
    if (!maximum) return {};
    const size_t retained=std::min<size_t>(8191,prompt.size());
    std::vector<Token> h; h.reserve(retained+1);
    h.insert(h.end(),prompt.end()-retained,prompt.end()); h.push_back(last);
    const size_t n=h.size();
    std::vector<size_t> matches;
    matches.reserve(n);
    // Minimum match is two exact tokens; require a historical continuation.
    for (size_t width=std::min<size_t>(16,n);width>=2;--width) {
        for (size_t end=width;end<n;++end) {
            if (std::equal(h.begin()+end-width,h.begin()+end,h.end()-width)) matches.push_back(end);
        }
        if (!matches.empty()) break;
    }
    std::vector<Token> result, votes;
    result.reserve(4); votes.reserve(n);
    for (int depth=0;depth<maximum && !matches.empty();++depth) {
        votes.clear();
        for (auto pos:matches) if (pos+(size_t)depth<n) votes.push_back(h[pos+depth]);
        if (votes.empty()) break;
        std::sort(votes.begin(),votes.end());
        Token best=votes.front(); size_t hits=0;
        for (size_t i=0;i<votes.size();) {
            size_t j=i+1; while (j<votes.size() && votes[j]==votes[i]) ++j;
            if (j-i>hits) { hits=j-i; best=votes[i]; }
            i=j;
        }
        // Ambiguous chains truncate, rather than spend a full verifier window.
        // Include exhausted occurrences in denominator (no optimistic censoring).
        if (hits*2<matches.size()) break;
        result.push_back(best);
        matches.erase(std::remove_if(matches.begin(),matches.end(),[&](size_t pos) {
            return pos+(size_t)depth>=n || h[pos+depth]!=best;
        }),matches.end());
    }
    return result;
}
