// Exact TL1_1-style byte-plane lookup candidate, MIT; no requantization.
// Included after ptq1-avx2.h. Runtime selection is opt-in, never automatic.
#ifndef GGML_PTQ1_PAIR_LUT_H
#define GGML_PTQ1_PAIR_LUT_H
#if defined(__AVX2__) && !defined(GGML_PTQ1_DISABLE_AVX2)
// Decode only one 128-weight block per row; bounded scratch, no row/tensor copy.
static void ptq1_block_codes(const block_ptq1_0 * x, uint8_t codes[128]) {
    __m256i state = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) x->qs));
    state = _mm256_inserti128_si256(state, _mm_loadl_epi64((const __m128i *) (x->qs+16)), 1);
    for (int j = 0; j < 5; ++j) {
        uint8_t tmp[32];
        _mm256_storeu_si256((__m256i *) tmp, ptq1_decode_trits_avx2(&state));
        memcpy(codes+j*16, tmp, 16);
        memcpy(codes+80+j*8, tmp+16, 8);
    }
#if defined(GGML_PTQ1_PAIR_LUT_COMPACT)
    // Same eight base-3 tail trits, interleaved qh[0]/qh[1]. Load exactly
    // two packed bytes and write exactly eight output bytes (no padding).
    uint16_t qh;
    memcpy(&qh, x->qh, sizeof(qh));
    __m128i tail = _mm_unpacklo_epi8(_mm_set1_epi16((int16_t) qh), _mm_setzero_si128());
    tail = _mm_and_si128(_mm_mullo_epi16(tail,
        _mm_setr_epi16(1,1,3,3,9,9,27,27)), _mm_set1_epi16(255));
    const __m128i trits = _mm_add_epi16(decode_trits_sse2(&tail), _mm_set1_epi16(1));
    _mm_storel_epi64((__m128i *) (codes+120), _mm_packus_epi16(trits, _mm_setzero_si128()));
#else
    for (int j = 0; j < 8; ++j) {
        unsigned v = x->qh[j%2];
        for (int i = 0; i < j/2; ++i) v = (v*3u)&255u;
        codes[120+j] = (uint8_t) ((v*3u)>>8);
    }
#endif
}

// A pair has nine exact signed sums, each in [-256,256]. Two byte planes
// preserve all int16 bits, including -128 activations and sums outside int8.
static __m256i ptq1_pair_planes(int a, int b) {
#if defined(GGML_PTQ1_PAIR_LUT_COMPACT)
    // Algebraic construction, no scalar 32-byte table staging. Values stay
    // in [-256,256]. Pack unsigned byte planes only after masking/shifting.
    const __m128i first = _mm_setr_epi16(-a-b,-b,a-b,-a,0,a,b-a,b);
    const __m128i last = _mm_cvtsi32_si128((int) (uint16_t) (a+b));
    const __m128i mask = _mm_set1_epi16(255);
    const __m128i low = _mm_packus_epi16(_mm_and_si128(first,mask), _mm_and_si128(last,mask));
    const __m128i high = _mm_packus_epi16(_mm_srli_epi16(first,8), _mm_srli_epi16(last,8));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
#else
    uint8_t bytes[32] = {0};
    for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
        const uint16_t v = (uint16_t) ((i-1)*a+(j-1)*b);
        bytes[i+3*j] = (uint8_t) v;
        bytes[16+i+3*j] = (uint8_t) (v>>8);
    }
    return _mm256_loadu_si256((const __m256i *) bytes);
#endif
}

void ggml_ptq1_pair_lut_tile(int n, float * result, size_t output_col_stride,
        const void * vx, size_t weight_row_stride, const void * vy,
        size_t activation_col_stride, int rows, int columns) {
    assert(n >= 0 && n%128 == 0 && rows >= 1 && rows <= 16 && columns >= 1 && columns <= 5);
    float totals[5][16] = {{0}};
    for (int b = 0; b < n/128; ++b) {
#if defined(GGML_PTQ1_PAIR_LUT_COMPACT)
        uint8_t indices[64][16];
        // Full tiles overwrite every byte. Tail lanes still need defined
        // indices for the 16-lane loads, though their outputs are discarded.
        if (rows < 16) memset(indices,0,sizeof(indices));
#else
        uint8_t indices[64][16] = {{0}};
#endif
        float weight_scale[16];
        for (int r = 0; r < rows; ++r) {
            const block_ptq1_0 * x = (const block_ptq1_0 *) ((const char *) vx+r*weight_row_stride)+b;
            uint8_t codes[128];
            ptq1_block_codes(x, codes);
            for (int p = 0; p < 64; ++p) indices[p][r] = codes[2*p]+3*codes[2*p+1];
            weight_scale[r] = GGML_CPU_FP16_TO_FP32(x->d);
        }
        for (int c = 0; c < columns; ++c) {
            const block_q8_0 * y = (const block_q8_0 *) ((const char *) vy+c*activation_col_stride)+4*b;
            float partial[16] = {0};
            for (int g = 0; g < 4; ++g) {
#if defined(GGML_PTQ1_PAIR_LUT_COMPACT)
                __m256i sum16 = _mm256_setzero_si256();
#else
                __m256i sum0 = _mm256_setzero_si256(), sum1 = sum0;
#endif
                for (int p = 0; p < 16; ++p) {
                    const __m128i index = _mm_loadu_si128((const __m128i *) indices[g*16+p]);
                    const __m256i looked = _mm256_shuffle_epi8(
                        ptq1_pair_planes(y[g].qs[2*p], y[g].qs[2*p+1]),
                        _mm256_broadcastsi128_si256(index));
                    const __m128i low = _mm256_castsi256_si128(looked);
                    const __m128i high = _mm256_extracti128_si256(looked,1);
#if defined(GGML_PTQ1_PAIR_LUT_COMPACT)
                    // At most 16 exact pair sums, abs(pair)<=256:
                    // abs(sum)<=4096, safely inside signed int16. This is
                    // integer-only reassociation; FP32 scale order is unchanged.
                    const __m256i pair16 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_unpacklo_epi8(low,high)),
                        _mm_unpackhi_epi8(low,high),1);
                    sum16 = _mm256_add_epi16(sum16,pair16);
#else
                    sum0 = _mm256_add_epi32(sum0, _mm256_cvtepi16_epi32(_mm_unpacklo_epi8(low,high)));
                    sum1 = _mm256_add_epi32(sum1, _mm256_cvtepi16_epi32(_mm_unpackhi_epi8(low,high)));
#endif
                }
#if defined(GGML_PTQ1_PAIR_LUT_COMPACT)
                const __m256i sum0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16));
                const __m256i sum1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16,1));
#endif
                int32_t sums[16];
                _mm256_storeu_si256((__m256i *) sums,sum0);
                _mm256_storeu_si256((__m256i *) (sums+8),sum1);
                const float scale = GGML_CPU_FP16_TO_FP32(y[g].d);
                // Same four scale applications and block reduction as upstream.
                for (int r = 0; r < rows; ++r) partial[r] += scale*sums[r];
            }
            for (int r = 0; r < rows; ++r) totals[c][r] += weight_scale[r]*partial[r];
        }
    }
    for (int c = 0; c < columns; ++c)
        memcpy((char *) result+c*output_col_stride,totals[c],rows*sizeof(float));
}
#endif
#endif
