// Source-only PTQ1 tiny-batch candidate; llama.cpp MIT license applies.
// Included after ptq1-avx2.h in arch/x86/quants.c.
#ifndef GGML_PTQ1_REUSE_H
#define GGML_PTQ1_REUSE_H
#if defined(__AVX2__) && !defined(GGML_PTQ1_DISABLE_AVX2) && (defined(GGML_PTQ1_ENABLE_BATCH) || defined(GGML_PTQ1_ENABLE_QIGEN))
void ggml_vec_dot_ptq1_0_q8_0_batch(
        int n, float * GGML_RESTRICT result, size_t result_stride,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        size_t column_stride, int columns) {
    assert(n >= 0 && n % QK_PTQ1_0 == 0);
    assert(columns >= 2 && columns <= 5);
    const block_ptq1_0 * x = (const block_ptq1_0 *) vx;
    const __m128i zero = _mm_setzero_si128();
    float totals[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < n / QK_PTQ1_0; ++i) {
        // Keep one decoded block in registers/stack, not an expanded weight row.
        // Decode and read the packed payload once for all verification columns.
        __m256i codes[5];
        __m256i packed = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) x[i].qs));
        packed = _mm256_inserti128_si256(packed, _mm_loadl_epi64((const __m128i *) &x[i].qs[16]), 1);
        for (int j = 0; j < 5; ++j) {
            codes[j] = ptq1_decode_trits_avx2(&packed);
        }
        uint16_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        __m128i tail = _mm_unpacklo_epi8(_mm_set1_epi16((int16_t) qh), zero);
        tail = _mm_and_si128(_mm_mullo_epi16(tail, _mm_setr_epi16(1, 1, 3, 3, 9, 9, 27, 27)), _mm_set1_epi16(255));
        const __m128i last_codes = decode_trits_sse2(&tail);
        const float scale = GGML_CPU_FP16_TO_FP32(x[i].d);
        for (int c = 0; c < columns; ++c) {
            const block_q8_0 * yb = (const block_q8_0 *) ((const char *) vy + c * column_stride) + i * 4;
            __m128i sums[4] = {zero, zero, zero, zero};
            for (int j = 0; j < 5; ++j) {
                const int offset = 80 + j * 8;
                __m256i qy = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *) &yb[j / 2].qs[(j % 2) * 16]));
                qy = _mm256_inserti128_si256(qy, _mm_loadl_epi64((const __m128i *) &yb[offset / 32].qs[offset % 32]), 1);
                const __m256i dot = ptq1_dot_codes_avx2(codes[j], qy);
                sums[j / 2] = _mm_add_epi32(sums[j / 2], _mm256_castsi256_si128(dot));
                sums[offset / 32] = _mm_add_epi32(sums[offset / 32], _mm256_extracti128_si256(dot, 1));
            }
            const __m128i qy = _mm_loadl_epi64((const __m128i *) &yb[3].qs[24]);
            sums[3] = _mm_add_epi32(sums[3], _mm_madd_epi16(last_codes, _mm_unpacklo_epi8(qy, _mm_cmpgt_epi8(zero, qy))));
            // Keep the single-column scale and accumulation order unchanged.
            float sumi = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sumi += GGML_CPU_FP16_TO_FP32(yb[k].d) * hsum_i32_4_sse2(sums[k]);
            }
            totals[c] += scale * sumi;
        }
    }
    for (int c = 0; c < columns; ++c) {
        memcpy((char *) result + c * result_stride, &totals[c], sizeof(float));
    }
}
#endif
#endif
