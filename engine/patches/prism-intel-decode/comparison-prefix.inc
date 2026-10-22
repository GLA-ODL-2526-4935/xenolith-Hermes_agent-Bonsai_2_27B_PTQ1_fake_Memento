#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#define GGML_RESTRICT __restrict__
#define UNUSED(x) (void)(x)
#define QK_PTQ1_0 128
struct block_ptq1_0 { uint8_t qs[24], qh[2]; uint16_t d; };
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(block_ptq1_0)==28, "PTQ1 layout");
static_assert(sizeof(block_q8_0)==34, "Q8 layout");
static float half_to_float(uint16_t h) {
    const int exponent = (h >> 10) & 31;
    const int fraction = h & 1023;
    const float value = exponent == 0 ? std::ldexp((float)fraction, -24) :
        exponent == 31 ? (fraction ? NAN : INFINITY) : std::ldexp((float)(1024+fraction), exponent-25);
    return h & 32768 ? -value : value;
}
#define GGML_CPU_FP16_TO_FP32(x) half_to_float(x)
#if defined(__SSE2__)
static inline int hsum_i32_4_sse2(__m128i x) {
    x = _mm_add_epi32(x, _mm_srli_si128(x, 8));
    x = _mm_add_epi32(x, _mm_srli_si128(x, 4));
    return _mm_cvtsi128_si32(x);
}

static inline __m128i decode_trits_sse2(__m128i * x) {
    const __m128i triple = _mm_add_epi16(*x, _mm_add_epi16(*x, *x));
    *x = _mm_and_si128(triple, _mm_set1_epi16(255));
    return _mm_sub_epi16(_mm_srli_epi16(triple, 8), _mm_set1_epi16(1));
}
#endif

#if defined(__SSSE3__)
static inline __m128i mul_sum_ternary_pairs_ssse3(const __m128i codes, const __m128i y) {
    // Subtract the code offset without negating -128 activations.
    const __m128i dot = _mm_maddubs_epi16(codes, y);
    const __m128i sum = _mm_maddubs_epi16(_mm_set1_epi8(1), y);
    return _mm_madd_epi16(_mm_sub_epi16(dot, sum), _mm_set1_epi16(1));
}

static inline __m128i decode_trits_ssse3(__m128i * x) {
    // Extract floor(3*x/256) and keep the low byte for the next trit.
    const __m128i biased = _mm_xor_si128(*x, _mm_set1_epi8(-128));
    const __m128i ge1 = _mm_cmpgt_epi8(biased, _mm_set1_epi8(85 - 128));
    const __m128i ge2 = _mm_cmpgt_epi8(biased, _mm_set1_epi8(170 - 128));
    *x = _mm_add_epi8(*x, _mm_add_epi8(*x, *x));
    return _mm_sub_epi8(_mm_setzero_si128(), _mm_add_epi8(ge1, ge2));
}
#endif


void baseline_dot(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    assert(n % QK_PTQ1_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);

    const block_ptq1_0 * GGML_RESTRICT x = (const block_ptq1_0 *) vx;
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *) vy;
    const __m128i zero = _mm_setzero_si128();
    float sumf = 0.0f;
    for (int i = 0; i < n / QK_PTQ1_0; ++i) {
        const block_q8_0 * yb = &y[i * 4];
        __m128i sums[4] = {zero, zero, zero, zero};
        const __m128i qs = _mm_loadu_si128((const __m128i *) x[i].qs);
#if defined(__SSSE3__)
        __m128i packed = qs;
#else
        __m128i lo = _mm_unpacklo_epi8(qs, zero);
        __m128i hi = _mm_unpackhi_epi8(qs, zero);
#endif

        // The first 16 packed bytes decode to five groups of 16 weights.
        for (int j = 0; j < 5; ++j) {
            const __m128i qy = _mm_loadu_si128((const __m128i *) &yb[j / 2].qs[(j % 2) * 16]);
#if defined(__SSSE3__)
            const __m128i dot = mul_sum_ternary_pairs_ssse3(decode_trits_ssse3(&packed), qy);
            sums[j / 2] = _mm_add_epi32(sums[j / 2], dot);
#else
            const __m128i ql = decode_trits_sse2(&lo);
            const __m128i qh = decode_trits_sse2(&hi);
            const __m128i sy = _mm_cmpgt_epi8(zero, qy);
            const __m128i dot0 = _mm_madd_epi16(ql, _mm_unpacklo_epi8(qy, sy));
            const __m128i dot1 = _mm_madd_epi16(qh, _mm_unpackhi_epi8(qy, sy));
            sums[j / 2] = _mm_add_epi32(sums[j / 2], _mm_add_epi32(dot0, dot1));
#endif
        }

        // The next eight bytes hold weights 80..119; qh holds weights 120..127.
        __m128i tail = _mm_loadl_epi64((const __m128i *) &x[i].qs[16]);
#if !defined(__SSSE3__)
        tail = _mm_unpacklo_epi8(tail, zero);
#endif
        for (int j = 0; j < 5; ++j) {
            const int offset = 80 + j * 8;
            const __m128i qy = _mm_loadl_epi64((const __m128i *) &yb[offset / 32].qs[offset % 32]);
#if defined(__SSSE3__)
            const __m128i dot = mul_sum_ternary_pairs_ssse3(decode_trits_ssse3(&tail), qy);
#else
            const __m128i q = decode_trits_sse2(&tail);
            const __m128i sy = _mm_cmpgt_epi8(zero, qy);
            const __m128i dot = _mm_madd_epi16(q, _mm_unpacklo_epi8(qy, sy));
#endif
            sums[offset / 32] = _mm_add_epi32(sums[offset / 32], dot);
        }
        uint16_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        tail = _mm_unpacklo_epi8(_mm_set1_epi16((int16_t) qh), zero);
        tail = _mm_and_si128(_mm_mullo_epi16(tail, _mm_setr_epi16(1, 1, 3, 3, 9, 9, 27, 27)), _mm_set1_epi16(255));
        const __m128i q = decode_trits_sse2(&tail);
        const __m128i qy = _mm_loadl_epi64((const __m128i *) &yb[3].qs[24]);
        const __m128i sy = _mm_cmpgt_epi8(zero, qy);
        sums[3] = _mm_add_epi32(sums[3], _mm_madd_epi16(q, _mm_unpacklo_epi8(qy, sy)));

        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            sumi += GGML_CPU_FP16_TO_FP32(yb[k].d) * hsum_i32_4_sse2(sums[k]);
        }
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * sumi;
    }
    *s = sumf;
#else
    ggml_vec_dot_ptq1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}


#include "ptq1-avx2.h"

#define GGML_PTQ1_ENABLE_BATCH
#include "ptq1-reuse.h"
#include "ptq1-exact-lut.h"
