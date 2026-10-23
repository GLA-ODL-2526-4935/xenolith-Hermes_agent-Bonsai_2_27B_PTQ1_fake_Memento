// Compile here; execute later on the target before runtime activation.
#include "source-policy.h"
#include <cassert>
#include <set>
int main() {
    common_source_policy p;
    std::set<int> seen;
    for (int i=0;i<600;++i) {
        p.begin(4,true,i*1000+1);
        seen.insert(p.arm);
        const int arm=p.arm;
        // Synthetic losing speculation must return to off between probes.
        assert(p.finish(1,i*1000+(arm?900:101)));
        assert(!p.finish(1,i*1000+999)); // no double credit
    }
    assert(seen.size()==9);
    p.reset();p.begin(4,true,100);p.arm=7;
    p.begin(1,false,500,1);assert(p.started==100 && p.arm==7); // replay
    assert(p.finish(3,1100));assert(p.stats[7].tokens==3 && p.stats[7].us==1000);
    p.begin(4,true,1200,1);assert(p.cycles==0 && p.arm==0); // bucket reset
    p.reset();p.begin(4,false,10);assert(!p.finish(0,100));assert(p.cycles==0);
    for (int cap=0;cap<=4;++cap) for (bool suffix:{false,true}) {
        p.reset();
        for (int i=0;i<200;++i) {
            p.begin(cap,suffix,i*1000+1);
            assert(p.arm==0 || (p.arm<=4?p.arm:p.arm-4)<=cap);
            assert(suffix || p.arm<=4);p.finish(1,i*1000+100);
        }
    }
    p.reset();p.stats[0]={20,2000,20};p.stats[4]={80,2000,20};p.cycles=20;
    assert(p.choose(4,false)==4);assert(p.choose(3,false)==0);
    assert(common_suffix_chain(std::vector<int>{1,2,3,4,1},2,4)==std::vector<int>({3,4,1,2}));
    assert(common_suffix_chain(std::vector<int>{1,2,3,4},5,4).empty());
    assert(common_suffix_chain(std::vector<int>{1},2,4).empty());
    assert(common_suffix_chain(std::vector<int>{},2,4).empty());
    assert(common_suffix_chain(std::vector<int>{1,2,3,4,1},2,0).empty());
    std::vector<int> huge(20000,77);huge.push_back(1);huge.push_back(2);
    assert(common_suffix_chain(huge,3,4).empty());
}
