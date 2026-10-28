// Experimental exact LUT baseline for comparison ONLY; not runtime-dispatched.
// A four-activation subset table, positive-mask minus negative-mask.
// No table quantization or approximate averaging. MIT; see LICENSE.llama-cpp.
struct ptq1_exact_lut_block {
    int16_t subset[32][16]; // 1024 B per 128 activations, versus Q8's 136 B.
    float scales[4];
};
static void ptq1_build_exact_lut(const block_q8_0 * y, int blocks, ptq1_exact_lut_block * lut) {
    for (int b = 0; b < blocks; ++b) {
        for (int k = 0; k < 4; ++k) lut[b].scales[k] = GGML_CPU_FP16_TO_FP32(y[b*4+k].d);
        for (int g = 0; g < 32; ++g) {
            const int8_t * a = &y[b*4+g/8].qs[(g%8)*4];
            for (int mask = 0; mask < 16; ++mask) {
                int s = 0;
                for (int j = 0; j < 4; ++j) if (mask & (1 << j)) s += a[j];
                lut[b].subset[g][mask] = (int16_t) s;
            }
        }
    }
}
static int ptq1_lut_trit(const block_ptq1_0 & x, int position) {
    int v, iterations;
    if (position < 80) { v = x.qs[position%16]; iterations = position/16; }
    else if (position < 120) { v = x.qs[16+(position-80)%8]; iterations = (position-80)/8; }
    else { v = x.qh[(position-120)%2]; iterations = (position-120)/2; }
    for (int j = 0; j < iterations; ++j) v = (v*3) & 255;
    return (v*3)/256 - 1;
}
static void ptq1_dot_exact_lut(int n, float * result, const block_ptq1_0 * x, const ptq1_exact_lut_block * lut) {
    float total = 0.0f;
    for (int b = 0; b < n/128; ++b) {
        int sums[4] = {0,0,0,0};
        for (int g = 0; g < 32; ++g) {
            int positive = 0, negative = 0;
            for (int j = 0; j < 4; ++j) {
                const int w = ptq1_lut_trit(x[b], g*4+j);
                positive |= (w == 1) << j;
                negative |= (w == -1) << j;
            }
            sums[g/8] += lut[b].subset[g][positive] - lut[b].subset[g][negative];
        }
        float partial = 0.0f;
        for (int k = 0; k < 4; ++k) partial += lut[b].scales[k] * sums[k];
        total += GGML_CPU_FP16_TO_FP32(x[b].d) * partial;
    }
    *result = total;
}
