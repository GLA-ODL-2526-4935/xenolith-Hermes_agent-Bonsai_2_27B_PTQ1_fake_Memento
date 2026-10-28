#include "comparison-prefix.inc"
#define ptq1_block_codes previous_codes
#define ptq1_pair_planes previous_planes
#define ggml_ptq1_pair_lut_tile previous_tile
#include "pair-before.h"
#undef GGML_PTQ1_PAIR_LUT_H
#undef ptq1_block_codes
#undef ptq1_pair_planes
#undef ggml_ptq1_pair_lut_tile
#define GGML_PTQ1_PAIR_LUT_VECTOR_SCALE
#include "pair-after.h"
#include <algorithm>
#include <chrono>
#include <vector>

static uint32_t state=1234567;
static uint32_t next() { state^=state<<13;state^=state>>17;state^=state<<5;return state; }
static uint16_t scale() { uint16_t h=(uint16_t)next();return (h&0x7c00)==0x7c00?(uint16_t)(h&0xfbff):h; }
int main() {
    size_t cases=0;
    for (int blocks:{0,1,3,17}) for (int rows=1;rows<=16;++rows) for (int cols=1;cols<=5;++cols) {
        const int ws=blocks+2,as=blocks*4+3,os=rows+3;
        std::vector<block_ptq1_0> w(rows*ws);
        std::vector<block_q8_0> a(cols*as);
        std::vector<float> old(cols*os+1),now(old.size());
        for (int trial=0;trial<128;++trial) {
            for (auto & x:w) {x.d=scale();for(auto & v:x.qs)v=(uint8_t)next();for(auto & v:x.qh)v=(uint8_t)next();}
            for (auto & x:a) {x.d=scale();for(auto & v:x.qs)v=trial<2?(trial?-128:127):(int8_t)next();}
            std::fill(old.begin(),old.end(),123456.0f);now=old;
            previous_tile(blocks*128,old.data(),os*4,w.data(),ws*sizeof(w[0]),a.data(),as*sizeof(a[0]),rows,cols);
            ggml_ptq1_pair_lut_tile(blocks*128,now.data(),os*4,w.data(),ws*sizeof(w[0]),a.data(),as*sizeof(a[0]),rows,cols);
            assert(!std::memcmp(old.data(),now.data(),old.size()*4));++cases;
        }
    }
    std::printf("bitwise_cases,%zu\n",cases);
    using clock=std::chrono::steady_clock;
    volatile float sink=0;
    for(int rows:{1,8,16}) for(int cols:{1,3,5}) {
        const int blocks=32;
        std::vector<block_ptq1_0> w(rows*blocks);std::vector<block_q8_0> a(cols*blocks*4);
        for(auto & x:w){x.d=0x3c00;for(auto & v:x.qs)v=(uint8_t)next();for(auto & v:x.qh)v=(uint8_t)next();}
        for(auto & x:a){x.d=0x2800;for(auto & v:x.qs)v=(int8_t)next();}
        float result[80];std::vector<double> times[2];
        for(int trial=0;trial<9;++trial)for(int j=0;j<2;++j){
            int which=(trial+j)%2;auto start=clock::now();
            for(int repeat=0;repeat<100;++repeat){
                if(which)ggml_ptq1_pair_lut_tile(blocks*128,result,16*4,w.data(),blocks*sizeof(w[0]),a.data(),blocks*4*sizeof(a[0]),rows,cols);
                else previous_tile(blocks*128,result,16*4,w.data(),blocks*sizeof(w[0]),a.data(),blocks*4*sizeof(a[0]),rows,cols);
                sink+=result[0];
            }
            times[which].push_back(std::chrono::duration<double,std::micro>(clock::now()-start).count()/100);
        }
        for(auto & t:times)std::sort(t.begin(),t.end());
        std::printf("rows,%d,cols,%d,previous_us,%.3f,vector_us,%.3f\n",rows,cols,times[0][4],times[1][4]);
    }
    return sink==12345.0f;
}
