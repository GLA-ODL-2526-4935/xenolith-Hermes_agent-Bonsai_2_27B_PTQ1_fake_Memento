// BITCOS-inspired exact experimental codec + dot. MIT. Not a new GGUF type.
// Compact block: original scale (2 B), nonzero bitmap (16 B), compact signs.
// Row indexing is caller-owned and must be counted separately. No allocation.
#ifndef GGML_PTQ1_BITCOS_H
#define GGML_PTQ1_BITCOS_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static int ptq1_bitcos_trit(const block_ptq1_0 * x, int position) {
    unsigned v; int times;
    if (position < 80) { v=x->qs[position%16]; times=position/16; }
    else if (position < 120) { v=x->qs[16+(position-80)%8]; times=(position-80)/8; }
    else { v=x->qh[(position-120)%2]; times=(position-120)/2; }
    for (int i=0; i<times; ++i) v=(v*3u)&255u;
    return (int)((v*3u)>>8)-1;
}
static unsigned ptq1_popcount32(uint32_t x) {
    x -= (x>>1)&0x55555555u;
    x = (x&0x33333333u)+((x>>2)&0x33333333u);
    x = (x+(x>>4))&0x0f0f0f0fu;
    return (x*0x01010101u)>>24;
}
static unsigned ptq1_bitcos_nonzeros(const block_ptq1_0 * x) {
    unsigned n=0;
    for (int i=0;i<128;++i) n += ptq1_bitcos_trit(x,i)!=0;
    return n;
}
static size_t ptq1_bitcos_size(const block_ptq1_0 * x) {
    return 18u+(ptq1_bitcos_nonzeros(x)+7u)/8u;
}
// Returns 0 without writes when capacity is insufficient. Scale bits are copied.
static size_t ptq1_bitcos_pack(const block_ptq1_0 * x, uint8_t * out, size_t cap) {
    const size_t bytes=ptq1_bitcos_size(x);
    if (cap<bytes) return 0;
    memset(out,0,bytes); memcpy(out,&x->d,2);
    unsigned n=0;
    for (unsigned i=0;i<128;++i) {
        const int w=ptq1_bitcos_trit(x,(int)i);
        if (w) {
            out[2+i/8] |= (uint8_t)(1u<<(i%8));
            if (w>0) out[18+n/8] |= (uint8_t)(1u<<(n%8));
            ++n;
        }
    }
    return bytes;
}
// Conditional row conversion. Include the complete offset array in the cost gate.
// Two-pass sizing avoids a partial write on insufficient buffers or a loss.
// An empty row stays on the raw path. No persistent model cache is allocated.
static size_t ptq1_bitcos_pack_row(int n,const block_ptq1_0 * x,uint8_t * out,size_t cap,
        uint32_t * offsets,size_t offset_count,int require_smaller) {
    if (n<=0 || n%128 || offset_count!=(size_t)n/128+1) return 0;
    const size_t blocks=(size_t)n/128;
    size_t needed=0;
    for (size_t b=0;b<blocks;++b) needed+=ptq1_bitcos_size(x+b);
    if (needed>UINT32_MAX || cap<needed || !out || !offsets ||
        (require_smaller && needed+offset_count*sizeof(uint32_t)>=blocks*sizeof(block_ptq1_0))) return 0;
    offsets[0]=0;
    for (size_t b=0;b<blocks;++b)
        offsets[b+1]=offsets[b]+(uint32_t)ptq1_bitcos_pack(x+b,out+offsets[b],cap-offsets[b]);
    return needed;
}
// Validate an untrusted view before unchecked hot-loop access. No overreads.
static int ptq1_bitcos_validate(const uint8_t * data,size_t bytes) {
    if (bytes<18) return 0;
    unsigned n=0;
    for (int i=2;i<18;++i) n+=ptq1_popcount32(data[i]);
    if (bytes!=18+(n+7)/8) return 0;
    if (n%8 && (data[bytes-1]>>(n%8))) return 0;
    return 1;
}
static uint32_t ptq1_read_le32(const uint8_t * p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static uint32_t ptq1_deposit_signs(uint32_t signs,uint32_t mask) {
#if defined(__BMI2__) && !defined(GGML_PTQ1_BITCOS_NO_BMI2)
    return _pdep_u32(signs,mask);
#else
    uint32_t out=0;
    while (mask) {
        const uint32_t lowest=mask&(~mask+1u);
        if (signs&1u) out|=lowest;
        signs>>=1; mask&=mask-1;
    }
    return out;
#endif
}
static uint32_t ptq1_bitcos_sign_word(const uint8_t * signs,unsigned byte_count,
                                    unsigned offset,unsigned count) {
    // Up to 39 significant bits straddle five bytes. Bound the final read.
    uint64_t word=0;
    const unsigned begin=offset/8;
    const unsigned need=(offset%8+count+7)/8;
    for (unsigned j=0;j<need && begin+j<byte_count;++j) word|=(uint64_t)signs[begin+j]<<(8*j);
    word>>=offset%8;
    return count==32 ? (uint32_t)word : (uint32_t)word&((1u<<count)-1u);
}
#if defined(__AVX2__) && !defined(GGML_PTQ1_DISABLE_AVX2)
static __m256i ptq1_bitmap_bytes(uint32_t mask) {
    const __m256i order=_mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,
                                       2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const __m256i bits=_mm256_setr_epi8(1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128,
                                      1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128);
    const __m256i expanded=_mm256_shuffle_epi8(_mm256_set1_epi32((int32_t)mask),order);
    return _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(expanded,bits),_mm256_setzero_si256()),_mm256_set1_epi8(1));
}
// Caller validates every packed block/offset once before dispatch. Four Q8 scales
// remain independent. The result is one block contribution, not a row reduction.
static float ptq1_bitcos_block_dot(const uint8_t * data,size_t bytes,const block_q8_0 * y) {
    assert(bytes>=18 && bytes<=34); // contents validated before hot-loop entry
    float partial=0.0f; unsigned cursor=0;
    for (int g=0;g<4;++g) {
        const uint32_t mask=ptq1_read_le32(data+2+4*g);
        const unsigned count=ptq1_popcount32(mask);
        const uint32_t packed=ptq1_bitcos_sign_word(data+18,(unsigned)(bytes-18),cursor,count);
        const __m256i present=ptq1_bitmap_bytes(mask);
        const __m256i positive=ptq1_bitmap_bytes(ptq1_deposit_signs(packed,mask));
        const __m256i codes=_mm256_sub_epi8(_mm256_add_epi8(_mm256_set1_epi8(1),
                            _mm256_add_epi8(positive,positive)),present);
        const __m256i sums=ptq1_dot_codes_avx2(codes,_mm256_loadu_si256((const __m256i *)y[g].qs));
        const int total=hsum_i32_4_sse2(_mm_add_epi32(_mm256_castsi256_si128(sums),_mm256_extracti128_si256(sums,1)));
        partial+=GGML_CPU_FP16_TO_FP32(y[g].d)*total;
        cursor+=count;
    }
    uint16_t scale; memcpy(&scale,data,2);
    return GGML_CPU_FP16_TO_FP32(scale)*partial;
}
// uint32 offsets relative to a single row; larger rows must keep raw PTQ1.
// Validates before writing any result. Indices are supplied, never hidden heap RAM.
struct ptq1_bitcos_row_view {
    const uint8_t * data;
    const uint32_t * offsets;
    size_t blocks;
};
static int ptq1_bitcos_view_init(int n,struct ptq1_bitcos_row_view * view,
        const uint8_t * data,size_t bytes,const uint32_t * offsets,size_t count) {
    if (n<0 || n%128 || count!=(size_t)n/128+1 || !view || (bytes && !data) ||
        !offsets || offsets[0]!=0 || offsets[count-1]!=bytes) return 0;
    for (size_t b=0;b+1<count;++b)
        if (offsets[b]>bytes || offsets[b+1]>bytes || offsets[b+1]<offsets[b] ||
            !ptq1_bitcos_validate(data+offsets[b],offsets[b+1]-offsets[b])) return 0;
    view->data=data; view->offsets=offsets; view->blocks=count-1;
    return 1;
}
// A validated view borrows immutable caller buffers; mutation/lifetime violations
// invalidate it. No repeated whole-row metadata validation in the hot dot loop.
static float ptq1_bitcos_row_dot_view(const struct ptq1_bitcos_row_view * view,const block_q8_0 * y) {
    float total=0.0f;
    for (size_t b=0;b<view->blocks;++b) {
        const uint32_t * offsets=view->offsets;
        total+=ptq1_bitcos_block_dot(view->data+offsets[b],offsets[b+1]-offsets[b],y+4*b);
    }
    return total;
}
static int ptq1_bitcos_row_dot(int n,float * result,const uint8_t * data,size_t bytes,
                               const uint32_t * offsets,size_t count,const block_q8_0 * y) {
    struct ptq1_bitcos_row_view view;
    if (!ptq1_bitcos_view_init(n,&view,data,bytes,offsets,count)) return 0;
    *result=ptq1_bitcos_row_dot_view(&view,y); return 1;
}
#endif
#endif
