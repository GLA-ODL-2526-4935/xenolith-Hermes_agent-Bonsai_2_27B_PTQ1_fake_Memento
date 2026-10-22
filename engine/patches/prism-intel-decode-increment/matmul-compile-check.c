#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#define GGML_RESTRICT restrict
#define GGML_UNUSED(x) (void)(x)
#define MIN(a,b) ((a)<(b)?(a):(b))
enum ggml_type { GGML_TYPE_PTQ1_0, GGML_TYPE_Q8_0 };
struct ggml_tensor { enum ggml_type type; int64_t ne[4]; size_t nb[4]; void * data; struct ggml_tensor * src[2]; };
struct ggml_compute_params { void * wdata; bool use_ref; };
typedef void (*ggml_vec_dot_t)(int,float *,size_t,const void *,size_t,const void *,size_t,int);
bool ggml_is_contiguous(const struct ggml_tensor *);
ggml_vec_dot_t ggml_cpu_vec_dot(const struct ggml_tensor *);
enum ggml_type ggml_cpu_vec_dot_type(const struct ggml_tensor *);
size_t ggml_row_size(enum ggml_type,int64_t);
void ggml_vec_dot_ptq1_0_q8_0_batch(int,float *,size_t,const void *,const void *,size_t,int);
#define GGML_TENSOR_BINARY_OP_LOCALS const int64_t ne00 __attribute__((unused)) = src0->ne[0]; const size_t nb00 __attribute__((unused)) = src0->nb[0]; const int64_t ne01 __attribute__((unused)) = src0->ne[1]; const size_t nb01 __attribute__((unused)) = src0->nb[1]; const int64_t ne02 __attribute__((unused)) = src0->ne[2]; const size_t nb02 __attribute__((unused)) = src0->nb[2]; const int64_t ne03 __attribute__((unused)) = src0->ne[3]; const size_t nb03 __attribute__((unused)) = src0->nb[3]; const int64_t ne10 __attribute__((unused)) = src1->ne[0]; const size_t nb10 __attribute__((unused)) = src1->nb[0]; const int64_t ne11 __attribute__((unused)) = src1->ne[1]; const size_t nb11 __attribute__((unused)) = src1->nb[1]; const int64_t ne12 __attribute__((unused)) = src1->ne[2]; const size_t nb12 __attribute__((unused)) = src1->nb[2]; const int64_t ne13 __attribute__((unused)) = src1->ne[3]; const size_t nb13 __attribute__((unused)) = src1->nb[3]; const int64_t ne0 __attribute__((unused)) = dst->ne[0]; const size_t nb0 __attribute__((unused)) = dst->nb[0]; const int64_t ne1 __attribute__((unused)) = dst->ne[1]; const size_t nb1 __attribute__((unused)) = dst->nb[1]; const int64_t ne2 __attribute__((unused)) = dst->ne[2]; const size_t nb2 __attribute__((unused)) = dst->nb[2]; const int64_t ne3 __attribute__((unused)) = dst->ne[3]; const size_t nb3 __attribute__((unused)) = dst->nb[3];
#include <limits.h>
size_t ggml_type_size(enum ggml_type);
void ggml_ptq1_pair_lut_tile(int,float *,size_t,const void *,size_t,const void *,size_t,int,int);
static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_UNUSED(type);

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = ggml_cpu_vec_dot(src0);
    enum ggml_type const vec_dot_type = ggml_cpu_vec_dot_type(src0);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // Exact byte-plane pair LUT: table build is amortized across up to 16 rows.
    // Opt-in, ordinary 2-D skinny matrices only; all other shapes keep fallback.
#if defined(__AVX2__) && !defined(GGML_PTQ1_DISABLE_AVX2) && defined(GGML_PTQ1_ENABLE_PAIR_LUT)
    const int64_t pair_columns = ir1_end-ir1_start;
    if (!params->use_ref && src0->type == GGML_TYPE_PTQ1_0 &&
        vec_dot_type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 &&
        ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        ne00 >= 0 && ne00 <= INT_MAX && ne00 % 128 == 0 &&
        nb00 == ggml_type_size(GGML_TYPE_PTQ1_0) &&
        ne11 >= 1 && ne11 <= 5 && pair_columns >= 1 && pair_columns <= 5 &&
        ir1_start >= 0 && ir1_end <= ne11 && nb0 == sizeof(float)) {
        for (int64_t r=ir0_start;r<ir0_end;r+=16) {
            ggml_ptq1_pair_lut_tile((int)ne00,
                (float *)((char *)dst->data+ir1_start*nb1)+r,nb1,
                (const char *)src0->data+r*nb01,nb01,
                (const char *)wdata+ir1_start*src1_col_stride,src1_col_stride,
                (int)MIN(16,ir0_end-r),(int)pair_columns);
        }
        return;
    }
#endif

    // Opt-in tiny-batch verification: share packed-weight unpack across columns.
    // Higher dimensions/broadcasts and reference requests keep the original path.
#if defined(__AVX2__) && !defined(GGML_PTQ1_DISABLE_AVX2) && defined(GGML_PTQ1_ENABLE_BATCH)
    const int64_t ptq1_columns = ir1_end - ir1_start;
    if (!params->use_ref && src0->type == GGML_TYPE_PTQ1_0 &&
        vec_dot_type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 &&
        ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
        ne11 >= 2 && ne11 <= 5 && ptq1_columns >= 2 && ptq1_columns <= 5 &&
        ir1_start >= 0 && ir1_end <= ne11 && nb0 == sizeof(float)) {
        const char * ptq1_y = (const char *) wdata + ir1_start * src1_col_stride;
        float * ptq1_dst = (float *) ((char *) dst->data + ir1_start * nb1);
        for (int64_t ir0 = ir0_start; ir0 < ir0_end; ++ir0) {
            ggml_vec_dot_ptq1_0_q8_0_batch(ne00, ptq1_dst + ir0, nb1,
                (const char *) src0->data + ir0 * nb01, ptq1_y,
                src1_col_stride, (int) ptq1_columns);
        }
        return;
    }
#endif

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}
