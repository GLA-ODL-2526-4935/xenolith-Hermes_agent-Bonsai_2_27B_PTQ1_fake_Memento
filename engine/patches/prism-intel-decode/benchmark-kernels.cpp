// Future target only. Includes LUT setup in each timed pass; BITCOS conversion
// and offset bytes reported separately. Synthetic buffers, not a second model.
#include "comparison-prefix.inc"
#include "ptq1-pair-lut.h"
#include "ptq1-bitcos.h"
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstdlib>

int main(int argc,char ** argv) {
    const int rows=argc>1 ? std::atoi(argv[1]) : 4096;
    const int k=argc>2 ? std::atoi(argv[2]) : 8192;
    const int cols=argc>3 ? std::atoi(argv[3]) : 5;
    const int repeats=argc>4 ? std::atoi(argv[4]) : 5;
    const int encoding=argc>5 ? std::atoi(argv[5]) : -1; // -1: deterministic mixed bytes
    if (rows<1 || rows>32768 || k<128 || k>32768 || k%128 || cols<1 || cols>5 || repeats<3 || repeats>100 || encoding<-1 || encoding>255) return 2;
    const int blocks=k/128;
    std::vector<block_ptq1_0> x((size_t)rows*blocks);
    std::vector<block_q8_0> y((size_t)cols*blocks*4);
    std::vector<float> expected((size_t)rows*cols),actual(expected.size());
    uint32_t state=0x12345678;
    for (auto & w:x) {
        auto byte=[&]() { state^=state<<13; state^=state>>17; state^=state<<5; return encoding<0 ? (uint8_t)state : (uint8_t)encoding; };
        for (auto & v:w.qs) v=byte();
        for (auto & v:w.qh) v=byte();
        w.d=0x3c00;
    }
    for (size_t i=0;i<y.size();++i) { y[i].d=0x2800; for (int j=0;j<32;++j) y[i].qs[j]=(int8_t)((i*31+j*7)%256-128); }
    std::vector<uint8_t> packed((size_t)rows*blocks*34);
    std::vector<uint32_t> offsets((size_t)rows*(blocks+1));
    std::vector<size_t> starts(rows+1);
    std::vector<ptq1_bitcos_row_view> views(rows);
    using clock=std::chrono::steady_clock;
    auto start=clock::now();
    for (int r=0;r<rows;++r) {
        uint32_t * off=offsets.data()+(size_t)r*(blocks+1);
        for (int b=0;b<blocks;++b) off[b+1]=off[b]+(uint32_t)ptq1_bitcos_pack(&x[(size_t)r*blocks+b],packed.data()+starts[r]+off[b],34);
        starts[r+1]=starts[r]+off[blocks];
        if (!ptq1_bitcos_view_init(k,&views[r],packed.data()+starts[r],off[blocks],off,blocks+1)) return 4;
    }
    const auto pack_us=std::chrono::duration<double,std::micro>(clock::now()-start).count();
    auto run=[&](int kernel,std::vector<float> & result) {
        if (kernel==2) {
            for (int r=0;r<rows;r+=16) ggml_ptq1_pair_lut_tile(k,result.data()+r,rows*sizeof(float),x.data()+(size_t)r*blocks,blocks*sizeof(block_ptq1_0),
                y.data(),blocks*4*sizeof(block_q8_0),std::min(16,rows-r),cols);
        } else if (kernel==4 && cols>1) {
            for (int r=0;r<rows;++r) ggml_vec_dot_ptq1_0_q8_0_batch(k,result.data()+r,rows*sizeof(float),x.data()+(size_t)r*blocks,y.data(),blocks*4*sizeof(block_q8_0),cols);
        } else for (int r=0;r<rows;++r) for (int c=0;c<cols;++c) {
            float * out=&result[(size_t)c*rows+r];
            if (kernel==0) baseline_dot(k,out,0,x.data()+(size_t)r*blocks,0,y.data()+(size_t)c*blocks*4,0,1);
            if (kernel==1) ggml_vec_dot_ptq1_0_q8_0_avx2(k,out,x.data()+(size_t)r*blocks,y.data()+(size_t)c*blocks*4);
            if (kernel==4) ggml_vec_dot_ptq1_0_q8_0_avx2(k,out,x.data()+(size_t)r*blocks,y.data()+(size_t)c*blocks*4);
            if (kernel==3) *out=ptq1_bitcos_row_dot_view(&views[r],y.data()+(size_t)c*blocks*4);
        }
    };
    run(0,expected);
    for (int kernel=0;kernel<5;++kernel) { run(kernel,actual); if (std::memcmp(actual.data(),expected.data(),actual.size()*4)) return 3; }
    std::vector<double> times[5];
    for (int trial=0;trial<repeats;++trial) for (int i=0;i<5;++i) {
        const int kernel=(trial+i)%5; start=clock::now(); run(kernel,actual);
        times[kernel].push_back(std::chrono::duration<double,std::micro>(clock::now()-start).count());
    }
    std::printf("raw_bytes=%zu bitcos_payload=%zu offsets=%zu conversion_us=%.3f synthetic_encoding=%d\n",x.size()*28,starts.back(),offsets.size()*4,pack_us,encoding);
    for (int i=0;i<5;++i) {
        std::sort(times[i].begin(),times[i].end());
        std::printf("kernel=%d median_us=%.3f min_us=%.3f max_us=%.3f\n",i,times[i][times[i].size()/2],times[i].front(),times[i].back());
    }
    return 0;
}
