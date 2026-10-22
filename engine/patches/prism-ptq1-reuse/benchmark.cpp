// Kernel microbenchmark, NOT model throughput or measured DRAM traffic.
// Future target use only, after test-reuse passes. Includes LUT preparation cost.
#include "comparison-prefix.inc"
#include <vector>
#include <chrono>
#include <cstdlib>
#include <algorithm>

int main(int argc, char ** argv) {
    const int rows=argc>1?std::atoi(argv[1]):32768;
    const int n=argc>2?std::atoi(argv[2]):8192;
    const int columns=argc>3?std::atoi(argv[3]):5;
    const int repetitions=argc>4?std::atoi(argv[4]):3;
    if (rows<1 || n<128 || n%128 || columns<2 || columns>5 || repetitions<1) return 2;
    const int blocks=n/128;
    const size_t ys=(size_t)blocks*4;
    std::vector<block_ptq1_0> weights((size_t)rows*blocks);
    std::vector<block_q8_0> activations(ys*columns);
    std::vector<ptq1_exact_lut_block> tables((size_t)blocks*columns);
    std::vector<float> result((size_t)rows*columns), reference(result.size());
    uint32_t state=0x83a96541;
    auto random=[&]() { state^=state<<13; state^=state>>17; state^=state<<5; return state; };
    for (auto & w:weights) { w.d=0x3400; for(auto & v:w.qs)v=(uint8_t)random(); for(auto & v:w.qh)v=(uint8_t)random(); }
    for (auto & a:activations) { a.d=0x3000; for(auto & v:a.qs)v=(int8_t)random(); }
    auto run=[&](int mode) {
        if (mode==3) for(int c=0;c<columns;++c) ptq1_build_exact_lut(activations.data()+c*ys,blocks,tables.data()+c*blocks);
        for (int r=0;r<rows;++r) {
            const block_ptq1_0 * w=weights.data()+(size_t)r*blocks;
            if(mode==2) ggml_vec_dot_ptq1_0_q8_0_batch(n,result.data()+r,rows*sizeof(float),w,activations.data(),ys*sizeof(block_q8_0),columns);
            else for(int c=0;c<columns;++c) {
                float * target=result.data()+(size_t)c*rows+r;
                if(mode==0) baseline_dot(n,target,0,w,0,activations.data()+c*ys,0,1);
                if(mode==1) ggml_vec_dot_ptq1_0_q8_0_avx2(n,target,w,activations.data()+c*ys);
                if(mode==3) ptq1_dot_exact_lut(n,target,w,tables.data()+c*blocks);
            }
        }
    };
    run(0); reference=result;
    for(int mode=1;mode<4;++mode) {
        run(mode);
        if(std::memcmp(reference.data(),result.data(),result.size()*sizeof(float))) {
            std::fprintf(stderr,"Parity failure mode=%d\n",mode); return 1;
        }
    }
    const char * names[]={"pinned-SSE-SSSE3","single-AVX2","reuse-AVX2","experimental-exact-LUT"};
    std::printf("single-thread; rows=%d K=%d columns=%d weights_bytes=%zu LUT_bytes=%zu; no DRAM counters\n",rows,n,columns,weights.size()*sizeof(weights[0]),tables.size()*sizeof(tables[0]));
    for(int repeat=0;repeat<repetitions;++repeat) for(int index=0;index<4;++index) {
        const int mode=(index+repeat)%4; // Rotate order to reduce systematic warmup/thermal bias.
        const auto begin=std::chrono::steady_clock::now(); run(mode);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        double checksum=0; for(float v:result)checksum+=v;
        std::printf("%s repeat=%d seconds=%.9f checksum=%.9g\n",names[mode],repeat,seconds,checksum);
    }
}
