#include "ggml-cpu/cacheline-schedule.h"
#include "runtime-memory-policy.h"
#include "runtime-telemetry.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <thread>
#include <vector>

int main() {
    // All supported physical F32 offsets, including views inside a line.
    for (int offset=0; offset<64; offset+=4) for (int n=0;n<=513;++n)
    for (int nth : {1,2,3,6,8,17,64}) for (size_t bytes : {size_t(1),size_t(28),size_t(1024),size_t(16384),size_t(65536),SIZE_MAX}) {
        int64_t cursor=0;
        std::vector<unsigned char> seen(n,0);
        for (int ith=0;ith<nth;++ith) {
            int64_t a=-1,b=-1;
            assert(ggml_cacheline_range(1024+offset,n,bytes,nth,ith,&a,&b));
            assert(0<=a && a<=b && b<=n);
            if(a!=b) {
                assert(a==cursor); cursor=b;
                for (int64_t i=a;i<b;++i) assert(++seen[i]==1);
                // Unless explicitly choosing few long rows, internal boundaries
                // are physical cache-line boundaries, not just logical multiples.
                const uint64_t head=((64-offset)%64)/4;
                const uint64_t units=(head && n ? 1:0)+(n>head?(n-head+15)/16:0);
                const size_t rp=65536/bytes+(65536%bytes!=0);
                const uint64_t workers=std::min<uint64_t>(nth,std::max<uint64_t>(1,n/rp));
                if (!(units<workers && bytes>=16384) && a>0) assert((1024+offset+a*4)%64==0);
            }
        }
        assert(cursor==n);
        assert(std::all_of(seen.begin(),seen.end(),[](unsigned char x){return x==1;}));
    }
    int64_t a,b;
    assert(!ggml_cacheline_range(3,10,28,2,0,&a,&b));
    assert(!ggml_cacheline_range(0,-1,28,2,0,&a,&b));
    assert(!ggml_cacheline_range(0,10,28,0,0,&a,&b));
    assert(!ggml_cacheline_range(0,10,0,2,0,&a,&b));
    for(int ith=0;ith<64;++ith) {
        assert(ggml_cacheline_range(4,INT64_MAX,SIZE_MAX,64,ith,&a,&b));
        assert(0<=a && a<=b && b<=INT64_MAX);
    }
    size_t cap=1;
    assert(ggml_runtime_parse_mib(nullptr,&cap) && cap==0);
    assert(ggml_runtime_parse_mib("0",&cap) && cap==0);
    assert(ggml_runtime_parse_mib("2048",&cap) && cap==size_t(2048)*1024*1024);
    for (const char * s : {"", "-1", "+1", " 1", "1 ", "1.5", "garbage", "18446744073709551615", "99999999999999999999999999"})
        assert(!ggml_runtime_parse_mib(s,&cap));
    size_t total=90;
    assert(!ggml_runtime_budget_add(&total,11,100) && total==90);
    assert(ggml_runtime_budget_add(&total,10,100) && total==100);
    assert(!ggml_runtime_budget_add(&total,1,100) && total==100);
    total=SIZE_MAX-1;
    assert(!ggml_runtime_budget_add(&total,2,0) && total==SIZE_MAX-1);
    ggml_runtime_telemetry telemetry("unit-test");
    assert(telemetry.enabled); // runner explicitly sets environment
    std::vector<std::thread> threads;
    for(int t=0;t<4;++t) threads.emplace_back([&] {
        for(int i=0;i<1000;++i) { telemetry.allocation(128,false); telemetry.copy(64); telemetry.verify(4,2,2,true); }
    });
    for(auto & t:threads)t.join();
    auto s=telemetry.snapshot();
    assert(s.alloc_calls==4000 && s.alloc_requested_bytes==512000 && s.copy_calls==4000 && s.copy_payload_bytes==256000);
    assert(s.verify_cycles==4000 && s.drafted==16000 && s.matched==8000 && s.rollback_tokens==8000 && s.replay_cycles==4000);
    telemetry.copy(UINT64_MAX); telemetry.copy(1);
    assert(telemetry.snapshot().copy_payload_bytes==UINT64_MAX);
    telemetry.report();
    std::puts("policy, physical cache-line coverage, budget parsing and concurrent telemetry: PASS");
}
