#include "suffix-workspace.h"
#include "suffix-previous.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <thread>

static std::atomic<size_t> allocations{0};
static std::atomic<bool> counting{false};
void * operator new(size_t n) {
    if (counting) ++allocations;
    if (void * p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p,size_t) noexcept { std::free(p); }

using Token=int32_t;
static size_t cases=0;
static void check(const std::vector<Token> & prompt,Token last,int cap,
        common_suffix_workspace<Token> & workspace,std::vector<Token> & result) {
    const auto old=common_suffix_chain(prompt,last,cap);
    common_suffix_chain_reuse(prompt,last,cap,workspace,result);
    assert(old==result);
    assert(result.size()<=4);
    assert(workspace.matches.size()<=8192 && workspace.votes.size()<=8192);
    ++cases;
}
static void parity() {
    common_suffix_workspace<Token> w;
    std::vector<Token> result;
    for (size_t length=0,power=1;length<=9;++length,power*=3) {
        std::vector<Token> p(length);
        for (size_t pattern=0;pattern<power;++pattern) {
            size_t x=pattern;
            for (auto & t:p) { t=(Token)(x%3); x/=3; }
            for (Token last:{-1,0,1,2}) for (int cap=-1;cap<=5;++cap) check(p,last,cap,w,result);
        }
    }
    std::mt19937 gen(3407);
    for (int i=0;i<2000;++i) {
        const size_t n=i<1000?gen()%100:(gen()%17000);
        std::vector<Token> p(n);
        for (auto & t:p) t=(Token)(gen()%(i%3==0?3:100000));
        for (int cap:{0,1,3,4,8}) check(p,(Token)(gen()%3),cap,w,result);
    }
    for (size_t n:{2,3,15,16,17,8190,8191,8192,16384}) {
        std::vector<Token> p(n,7);
        for (int cap:{-2,0,1,2,3,4,9}) check(p,7,cap,w,result);
        for (size_t i=0;i<n;++i) p[i]=(Token)(i%3);
        for (int cap:{1,2,3,4}) check(p,0,cap,w,result);
    }
    std::vector<Token> alias{1,2,1,2,1,2};
    auto expected=common_suffix_chain(alias,1,4);
    common_suffix_chain_reuse(alias,1,4,w,alias); assert(alias==expected);
    w.votes={1,2,1,2}; expected=common_suffix_chain(w.votes,1,4);
    common_suffix_chain_reuse(w.votes,1,4,w,result); assert(result==expected);
    std::vector<Token> p{1,2,1,2}; expected=common_suffix_chain(p,1,4);
    common_suffix_chain_reuse(p,1,4,w,w.votes); assert(w.votes==expected);
    w.release(); assert(w.matches.capacity()==0 && w.votes.capacity()==0);
    check({8,9,8,9},8,4,w,result);
    std::cout<<"parity_cases,"<<cases<<"\n";
}
static void benchmark() {
    volatile size_t sink=0;
    std::cout<<"length,pattern,legacy_us,reuse_us,legacy_allocations,reuse_allocations\n";
    for (size_t n:{128,2048,8192}) for (int pattern:{0,1}) {
        std::vector<Token> p(n);std::mt19937 gen(3407);
        for (size_t i=0;i<n;++i) p[i]=pattern?(Token)(gen()%100000):(Token)(i%5);
        const Token last=pattern?91000:(Token)(n%5);
        common_suffix_workspace<Token> w;std::vector<Token> result;
        common_suffix_chain_reuse(p,last,4,w,result);
        auto run=[&](bool reuse) {
            allocations=0;counting=true;auto start=std::chrono::steady_clock::now();
            for (int i=0;i<1000;++i) {
                if (reuse) common_suffix_chain_reuse(p,last,4,w,result);
                else common_suffix_chain_previous(p,last,4,w,result);
                sink+=result.size();
            }
            auto end=std::chrono::steady_clock::now();counting=false;
            return std::make_pair(std::chrono::duration<double,std::micro>(end-start).count()/1000,allocations.load());
        };
        const auto old=run(false),fresh=run(true);
        assert(fresh.second==0);
        std::cout<<n<<","<<(pattern?"random":"repeat")<<","<<old.first<<","<<fresh.first<<","<<old.second<<","<<fresh.second<<"\n";
    }
    std::cout<<"sink,"<<sink<<"\n";
}
int main() {
    parity();benchmark();
    std::thread threads[4];
    for (auto & t:threads) t=std::thread([] {
        common_suffix_workspace<Token> w; std::vector<Token> out;
        const std::vector<Token> p{1,2,3,1,2,3,1,2};
        for(int i=0;i<1000;++i) {
            common_suffix_chain_reuse(p,3,4,w,out);
            assert(out==common_suffix_chain(p,3,4));
        }
        w.release();
    });
    for(auto & t:threads) t.join();
    std::cout<<"independent_workspaces,PASS\n";
}
