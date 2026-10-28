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
    for (size_t width=std::min<size_t>(16,n);width>=2;--width) {
        for (size_t end=width;end<n;++end) {
            if (prompt[offset+end-1]==last &&
                    std::equal(prompt.begin()+offset+end-width,prompt.begin()+offset+end-1,
                        prompt.end()-(width-1))) matches.push_back(end);
        }
        if (!matches.empty()) break;
    }
    for (int depth=0;depth<maximum && !matches.empty();++depth) {
        votes.clear();
        for (auto pos:matches) if (pos+(size_t)depth<n) votes.push_back(token(pos+depth));
        if (votes.empty()) break;
        std::sort(votes.begin(),votes.end());
        Token best=votes.front(); size_t hits=0;
        for (size_t i=0;i<votes.size();) {
            size_t j=i+1; while (j<votes.size() && votes[j]==votes[i]) ++j;
            if (j-i>hits) { hits=j-i; best=votes[i]; }
            i=j;
        }
        if (hits*2<matches.size()) break;
        result.push_back(best);
        matches.erase(std::remove_if(matches.begin(),matches.end(),[&](size_t pos) {
            return pos+(size_t)depth>=n || token(pos+depth)!=best;
        }),matches.end());
    }
}
