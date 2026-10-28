// SPDX-License-Identifier: MIT
#pragma once
#include "source-policy.h"
#include <utility>

template<class Token>
struct common_suffix_workspace {
    std::vector<size_t> matches;
    std::vector<Token> votes;
    void release() {
        std::vector<size_t>().swap(matches);
        std::vector<Token>().swap(votes);
    }
};

// Borrow the prompt only for this synchronous call. Keep scratch until request end.
template<class Token>
static void common_suffix_chain_reuse(const std::vector<Token> & prompt, Token last,
        int maximum, common_suffix_workspace<Token> & scratch, std::vector<Token> & result) {
    if (&prompt==&result || &result==&scratch.votes || &prompt==&scratch.votes) {
        auto value=common_suffix_chain(prompt,last,maximum);
        result=std::move(value);
        return;
    }
    result.clear(); scratch.matches.clear(); scratch.votes.clear();
    maximum=std::max(0,std::min(4,maximum));
    const size_t retained=std::min<size_t>(8191,prompt.size());
    const size_t n=retained+1, offset=prompt.size()-retained;
    if (!maximum || n<3) return;
    scratch.matches.reserve(8192); scratch.votes.reserve(8192); result.reserve(4);
    auto & matches=scratch.matches;
    auto & votes=scratch.votes;
    const auto token=[&](size_t i) { return i<retained?prompt[offset+i]:last; };
    size_t longest=1;
    for (size_t end=2;end<n;++end) {
        if (prompt[offset+end-1]!=last) continue;
        const size_t limit=std::min<size_t>(16,end);
        size_t width=1;
        while (width<limit && prompt[offset+end-1-width]==token(n-1-width)) ++width;
        if (width<2 || width<longest) continue;
        if (width>longest) { matches.clear(); longest=width; }
        matches.push_back(end);
    }
    for (int depth=0;depth<maximum && !matches.empty();++depth) {
        votes.clear();
        for (auto pos:matches) if (pos+(size_t)depth<n) votes.push_back(token(pos+depth));
        if (votes.empty()) break;
        // Two counters retain every candidate above one third of the votes.
        // This includes both possible half-vote ties; verify using all matches.
        Token candidates[2]={}; size_t counts[2]={};
        for (Token value:votes) {
            if (counts[0] && value==candidates[0]) ++counts[0];
            else if (counts[1] && value==candidates[1]) ++counts[1];
            else if (!counts[0]) { candidates[0]=value; counts[0]=1; }
            else if (!counts[1]) { candidates[1]=value; counts[1]=1; }
            else { --counts[0]; --counts[1]; }
        }
        Token best={}; size_t hits=0;
        for (int i=0;i<2;++i) if (counts[i]) {
            const size_t count=(size_t)std::count(votes.begin(),votes.end(),candidates[i]);
            if (count>hits || (count==hits && candidates[i]<best)) {
                hits=count; best=candidates[i];
            }
        }
        if (hits*2<matches.size()) break;
        result.push_back(best);
        matches.erase(std::remove_if(matches.begin(),matches.end(),[&](size_t pos) {
            return pos+(size_t)depth>=n || token(pos+depth)!=best;
        }),matches.end());
    }
}
