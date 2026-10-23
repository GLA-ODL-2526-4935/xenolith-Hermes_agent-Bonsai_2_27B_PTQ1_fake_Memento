// Target execution required; object compilation alone proves no numeric parity.
#include "comparison-prefix.inc"
#include "ptq1-pair-lut.h"
#include "ptq1-qigen.h"
#include <algorithm>
#include <vector>
static void env(const char * k,const char * v) {
#ifdef _WIN32
    _putenv_s(k,v);
#else
    setenv(k,v,1);
#endif
}
int main() {
    env("GGML_PTQ1_QIGEN","0");env("GGML_PTQ1_QIGEN_SCHEDULE","12344");
    assert(ggml_ptq1_qigen_mode(1)==0);
    env("GGML_PTQ1_QIGEN","1");
    for (int c=1;c<=5;++c) assert(ggml_ptq1_qigen_mode(c)=="12344"[c-1]-'0');
    assert(ggml_ptq1_qigen_mode(0)==0 && ggml_ptq1_qigen_mode(6)==0);
    for (auto bad:{"","1234","123456","1234x","12345"}) {
        env("GGML_PTQ1_QIGEN_SCHEDULE",bad);
        for(int c=1;c<=5;++c) assert(ggml_ptq1_qigen_mode(c)==0);
    }
    for (int rows=1;rows<=33;++rows) for (int cols=1;cols<=5;++cols) {
        const int blocks=3,k=128*blocks,stride=rows+3;
        std::vector<block_ptq1_0> w(rows*blocks);
        std::vector<block_q8_0> a(cols*blocks*4);
        for (size_t i=0;i<w.size();++i) {
            w[i].d=(i%2)?0x3c00:0xbc00;
            for(int j=0;j<24;++j)w[i].qs[j]=(uint8_t)(i*17+j*31);
            w[i].qh[0]=(uint8_t)i;w[i].qh[1]=(uint8_t)~i;
        }
        for (size_t i=0;i<a.size();++i) {
            a[i].d=(i%2)?0x2800:0xb000;
            for(int j=0;j<32;++j)a[i].qs[j]=(int8_t)(i*31+j*17);
        }
        std::vector<float> ref(cols*stride,12345.0f),actual=ref;
        for(int c=0;c<cols;++c)for(int r=0;r<rows;++r)
            baseline_dot(k,&ref[c*stride+r],0,&w[r*blocks],0,&a[c*blocks*4],0,1);
        for(int tile:{8,16}) {
            std::fill(actual.begin(),actual.end(),12345.0f);
            for(int r=0;r<rows;r+=tile)ggml_ptq1_pair_lut_tile(k,actual.data()+r,stride*4,
                &w[r*blocks],blocks*sizeof(block_ptq1_0),a.data(),blocks*4*sizeof(block_q8_0),std::min(tile,rows-r),cols);
            assert(!std::memcmp(ref.data(),actual.data(),ref.size()*4));
        }
    }
}
