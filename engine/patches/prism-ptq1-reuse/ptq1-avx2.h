// PTQ1 AVX2 source candidate. Upstream llama.cpp MIT license applies.
// Included after the existing SSE helpers in arch/x86/quants.c.
#ifndef GGML_PTQ1_AVX2_H
#define GGML_PTQ1_AVX2_H
#if defined(__AVX2__) && !defined(GGML_PTQ1_DISABLE_AVX2)
static inline __m256i ptq1_decode_trits_avx2(__m256i * state) {
    const __m256i biased = _mm256_xor_si256(*state, _mm256_set1_epi8(-128));
    const __m256i ge1 = _mm256_cmpgt_epi8(biased, _mm256_set1_epi8(85 - 128));
    const __m256i ge2 = _mm256_cmpgt_epi8(biased, _mm256_set1_epi8(170 - 128));
    *state = _mm256_add_epi8(*state, _mm256_add_epi8(*state, *state));
    return _mm256_sub_epi8(_mm256_setzero_si256(), _mm256_add_epi8(ge1, ge2));
}

static inline __m256i ptq1_dot_codes_avx2(__m256i codes, __m256i activation) {
    // Unsigned 0..2 codes avoid the int8 -128 negation trap. The intermediate
    // pair sums are bounded by 512 in magnitude, so maddubs does not saturate.
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i dot = _mm256_maddubs_epi16(codes, activation);
    const __m256i offset = _mm256_maddubs_epi16(ones8, activation);
    return _mm256_madd_epi16(_mm256_sub_epi16(dot, offset), _mm256_set1_epi16(1));
}

static void ggml_vec_dot_ptq1_0_q8_0_avx2(
        int n, float * GGML_RESTRICT result,
        const block_ptq1_0 * GGML_RESTRICT x,
        const block_q8_0 * GGML_RESTRICT y) {
    assert(n % QK_PTQ1_0 == 0);
    const __m128i zero = _mm_setzero_si128();
    float sumf = 0.0f;
    for (int i = 0; i < n / QK_PTQ1_0; ++i) {
        const block_q8_0 * yb = &y[i * 4];
        __m128i sums[4] = {zero, zero, zero, zero};
        // Bounded 16 + 8 byte loads, never a 32-byte read of a 24-byte payload.
        // Low lane: weights 0..79. High lane: weights 80..119, eight at a time.
        __m256i packed = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) x[i].qs));
        packed = _mm256_inserti128_si256(packed, _mm_loadl_epi64((const __m128i *) &x[i].qs[16]), 1);
        for (int j = 0; j < 5; ++j) {
            const int offset = 80 + j * 8;
            const __m128i main_y = _mm_loadu_si128((const __m128i *) &yb[j / 2].qs[(j % 2) * 16]);
            const __m128i tail_y = _mm_loadl_epi64((const __m128i *) &yb[offset / 32].qs[offset % 32]);
            __m256i qy = _mm256_castsi128_si256(main_y);
            qy = _mm256_inserti128_si256(qy, tail_y, 1);
            const __m256i dot = ptq1_dot_codes_avx2(ptq1_decode_trits_avx2(&packed), qy);
            sums[j / 2] = _mm_add_epi32(sums[j / 2], _mm256_castsi256_si128(dot));
            sums[offset / 32] = _mm_add_epi32(sums[offset / 32], _mm256_extracti128_si256(dot, 1));
        }
        // Eight final trits retain upstream element ordering and bounded loads.
        uint16_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        __m128i tail = _mm_unpacklo_epi8(_mm_set1_epi16((int16_t) qh), zero);
        tail = _mm_and_si128(_mm_mullo_epi16(tail, _mm_setr_epi16(1, 1, 3, 3, 9, 9, 27, 27)), _mm_set1_epi16(255));
        const __m128i q = decode_trits_sse2(&tail);
        const __m128i qy = _mm_loadl_epi64((const __m128i *) &yb[3].qs[24]);
        const __m128i sy = _mm_cmpgt_epi8(zero, qy);
        sums[3] = _mm_add_epi32(sums[3], _mm_madd_epi16(q, _mm_unpacklo_epi8(qy, sy)));
        // Preserve upstream floating accumulation order and four Q8 scales.
        float sumi = 0.0f;
        for (int k = 0; k < 4; ++k) {
            sumi += GGML_CPU_FP16_TO_FP32(yb[k].d) * hsum_i32_4_sse2(sums[k]);
        }
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * sumi;
    }
    *result = sumf;
}
#endif
#endif
