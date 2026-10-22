// Future target test. Compile-only on the preparation machine.
#include "comparison-prefix.inc"
#include <vector>

static uint32_t rng_state = 0x61a9702du;
static uint32_t random_u32() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}
static bool same(float a, float b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }
static uint16_t finite_half() {
    const uint16_t value = (uint16_t)random_u32();
    return (value & 0x7c00) == 0x7c00 ? (uint16_t)(value & 0xfbff) : value;
}
int main() {
    for (int blocks : {0,1,3,16}) {
        for (int columns=2; columns<=5; ++columns) {
            for (int padding : {0,4}) {
                const size_t stride = (blocks*4+padding)*sizeof(block_q8_0);
                std::vector<block_ptq1_0> x(blocks+1);
                std::vector<block_q8_0> storage((blocks*4+padding)*columns+1);
                std::vector<ptq1_exact_lut_block> lut(blocks+1);
                for (int trial=0; trial<4096; ++trial) {
                    for (auto & w : x) {
                        w.d = finite_half();
                        for (auto & q : w.qs) q = (uint8_t)random_u32();
                        for (auto & q : w.qh) q = (uint8_t)random_u32();
                    }
                    for (auto & a : storage) {
                        a.d = finite_half();
                        for (auto & q : a.qs) q = (int8_t)random_u32();
                    }
                    // Distinct all-encodings and extreme-activation cases, not only random inputs.
                    if (blocks && trial<256) {
                        for (auto & w : x) {
                            for (auto & q : w.qs) q = (uint8_t)trial;
                            for (auto & q : w.qh) q = (uint8_t)trial;
                        }
                        for (auto & a : storage) for (auto & q : a.qs) q = trial&1 ? -128 : 127;
                    }
                    float output[5][3];
                    for (auto & row : output) for (auto & v : row) v = 98765.0f;
                    ggml_vec_dot_ptq1_0_q8_0_batch(blocks*128,&output[0][1],sizeof(output[0]),x.data(),storage.data(),stride,columns);
                    for (int c=0; c<columns; ++c) {
                        const block_q8_0 * y = (const block_q8_0 *)((const char *)storage.data()+c*stride);
                        float expected=0, single=0, table=0;
                        baseline_dot(blocks*128,&expected,0,x.data(),0,y,0,1);
                        ggml_vec_dot_ptq1_0_q8_0_avx2(blocks*128,&single,x.data(),y);
                        ptq1_build_exact_lut(y,blocks,lut.data());
                        ptq1_dot_exact_lut(blocks*128,&table,x.data(),lut.data());
                        if (!same(expected,single) || !same(expected,output[c][1]) || !same(expected,table)) {
                            std::fprintf(stderr,"Mismatch blocks=%d cols=%d padding=%d trial=%d column=%d\n",blocks,columns,padding,trial,c);
                            return 1;
                        }
                    }
                    for (int c=0;c<5;++c) {
                        if (output[c][0]!=98765.0f || output[c][2]!=98765.0f || (c>=columns && output[c][1]!=98765.0f)) return 2;
                    }
                }
            }
        }
    }
    std::puts("PASS: batch/single/LUT agree bitwise with pinned SSE/SSSE3 for tested finite scales; strides and guards intact");
}
