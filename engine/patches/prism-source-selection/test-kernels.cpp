// Future deployment-target parity test. Compile-only on preparation PC.
#include "comparison-prefix.inc"
#include "ptq1-pair-lut.h"
#include "ptq1-bitcos.h"
#include <vector>

static uint32_t state=0x5eda719u;
static uint32_t random32() { state^=state<<13; state^=state>>17; state^=state<<5; return state; }
static uint16_t finite_half() {
    uint16_t h=(uint16_t)random32();
    return (h&0x7c00)==0x7c00 ? (uint16_t)(h&0xfbff) : h;
}
static bool same(float a,float b) { return std::memcmp(&a,&b,4)==0; }
int main() {
    for (int blocks:{0,1,3,17}) for (int rows:{1,3,8,15,16}) for (int cols:{1,2,4,5}) {
        const int wstride=blocks+3, astride=4*blocks+3, ostride=rows+3;
        std::vector<block_ptq1_0> x(rows*wstride);
        std::vector<block_q8_0> y(cols*astride);
        std::vector<float> output(cols*ostride+1);
        for (int trial=0;trial<512;++trial) {
            for (auto & w:x) {
                w.d=finite_half();
                for (auto & byte:w.qs) byte=trial<256 ? (uint8_t)trial : (uint8_t)random32();
                for (auto & byte:w.qh) byte=trial<256 ? (uint8_t)trial : (uint8_t)random32();
            }
            for (auto & a:y) {
                a.d=finite_half();
                for (auto & value:a.qs) value=trial<256 ? (trial%2 ? -128 : 127) : (int8_t)random32();
            }
            std::fill(output.begin(),output.end(),123456.0f);
            ggml_ptq1_pair_lut_tile(blocks*128,output.data(),ostride*sizeof(float),
                x.data(),wstride*sizeof(block_ptq1_0),y.data(),astride*sizeof(block_q8_0),rows,cols);
            for (int r=0;r<rows;++r) {
                std::vector<uint8_t> packed(34*blocks+1,0xcd);
                std::vector<uint32_t> offsets(blocks+1,0);
                for (int b=0;b<blocks;++b) {
                    const size_t size=ptq1_bitcos_pack(&x[r*wstride+b],packed.data()+offsets[b],34);
                    assert(size && ptq1_bitcos_validate(packed.data()+offsets[b],size));
                    offsets[b+1]=offsets[b]+(uint32_t)size;
                }
                assert(packed[offsets[blocks]]==0xcd);
                if (blocks) {
                    std::vector<uint8_t> row_copy(34*blocks+1,0xcd);
                    std::vector<uint32_t> row_offsets(blocks+1,0);
                    const size_t copied=ptq1_bitcos_pack_row(blocks*128,x.data()+r*wstride,
                        row_copy.data(),34*blocks,row_offsets.data(),row_offsets.size(),0);
                    assert(copied==offsets[blocks] && row_offsets==offsets);
                    assert(std::memcmp(packed.data(),row_copy.data(),copied)==0 && row_copy[copied]==0xcd);
                    const size_t selected=ptq1_bitcos_pack_row(blocks*128,x.data()+r*wstride,
                        row_copy.data(),34*blocks,row_offsets.data(),row_offsets.size(),1);
                    assert((selected!=0)==(copied+row_offsets.size()*4<(size_t)blocks*28));
                }
                for (int c=0;c<cols;++c) {
                    float expected=0,bits=0;
                    baseline_dot(blocks*128,&expected,0,x.data()+r*wstride,0,y.data()+c*astride,0,1);
                    assert(ptq1_bitcos_row_dot(blocks*128,&bits,packed.data(),offsets[blocks],offsets.data(),offsets.size(),y.data()+c*astride));
                    if (!same(expected,output[c*ostride+r]) || !same(expected,bits)) {
                        std::fprintf(stderr,"Parity failure b=%d r=%d c=%d trial=%d\n",blocks,r,c,trial);
                        return 1;
                    }
                }
                if (blocks) {
                    const uint32_t saved=offsets[1];
                    offsets[1]=UINT32_MAX;
                    float sentinel=42;
                    assert(!ptq1_bitcos_row_dot(blocks*128,&sentinel,packed.data(),offsets[blocks],offsets.data(),offsets.size(),y.data()));
                    assert(sentinel==42); offsets[1]=saved;
                    uint8_t guard[34]; std::memset(guard,0xab,sizeof(guard));
                    assert(ptq1_bitcos_pack(x.data(),guard,17)==0);
                    for (auto v:guard) assert(v==0xab);
                }
            }
            for (int c=0;c<cols;++c) for (int r=rows;r<ostride;++r) assert(output[c*ostride+r]==123456.0f);
            assert(output.back()==123456.0f);
        }
    }
    std::puts("PASS tested baseline/pair-LUT/BITCOS bitwise parity and bounded outputs; full backend/model acceptance still required");
}
