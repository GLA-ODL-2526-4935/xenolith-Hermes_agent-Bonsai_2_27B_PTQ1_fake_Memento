// Sandwich-inspired explicit phase selection. MIT, no model or tuning workload.
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <cstddef>

enum class common_cpu_phase { original, prefill, decode, verify };

// Prompt provenance wins over token count. Mixed batches use the original plan.
template<class Tokens>
common_cpu_phase common_sandwich_classify(const Tokens & tokens,int off,int count) {
    if (off<0 || count<=0 || (size_t)off>tokens.size() || (size_t)count>tokens.size()-(size_t)off)
        return common_cpu_phase::original;
    bool prompt=false,generated=false;
    const auto slot=tokens[(size_t)off].id_slot;
    for (int i=off;i<off+count;++i) {
        if (tokens[(size_t)i].id_slot!=slot) return common_cpu_phase::original;
        prompt|=tokens[(size_t)i].is_prompt;
        generated|=!tokens[(size_t)i].is_prompt;
    }
    if (prompt && generated) return common_cpu_phase::original;
    if (prompt) return common_cpu_phase::prefill;
    return count==1 ? common_cpu_phase::decode : common_cpu_phase::verify;
}

static inline int common_sandwich_threads(const char * text) {
    if (!text || !*text) return 0;
    int count=0;
    for (const char * p=text;*p;++p) {
        if (*p<'0' || *p>'9' || count>64) throw std::invalid_argument("Sandwich threads must be 1..64 decimal");
        count=count*10+(*p-'0');
    }
    if (count<1 || count>64) throw std::invalid_argument("Sandwich threads must be 1..64");
    return count;
}

template<size_t N>
static bool common_sandwich_mask(const char * hex,bool (&mask)[N],int threads) {
    if (!hex || !*hex) return false;
    size_t length=0;
    while (hex[length]) {
        if (++length>(N+3)/4) throw std::invalid_argument("Sandwich mask exceeds CPU mask capacity");
    }
    for (size_t i=0;i<N;++i) mask[i]=false;
    int bits=0;
    for (size_t i=0;i<length;++i) {
        const char ch=hex[length-1-i];
        int v=ch>='0' && ch<='9' ? ch-'0' : ch>='a' && ch<='f' ? ch-'a'+10 : ch>='A' && ch<='F' ? ch-'A'+10 : -1;
        if (v<0) throw std::invalid_argument("Sandwich mask requires plain hexadecimal");
        for (size_t j=0;j<4;++j) if (v&(1<<j)) {
            if (4*i+j>=N) throw std::invalid_argument("Sandwich mask exceeds CPU mask capacity");
            mask[4*i+j]=true; ++bits;
        }
    }
    if (bits<threads) throw std::invalid_argument("Sandwich mask has fewer CPUs than threads");
    return true;
}
