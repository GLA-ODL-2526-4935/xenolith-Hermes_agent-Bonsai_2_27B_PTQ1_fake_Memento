/* E4B baseline: generic row-interleaved Q4_0 x Q8 projection.
 * No expert selection or MoE kernels. Token-serial prefill is intentional
 * until independent numerical validation precedes batching/optimization. */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void xe_probe(__global uint *value) {
    if (get_global_id(0) == 0) value[0] = 0x78656e6fu;
}
__kernel void xe_e4b_q4q8(__global const uchar *wq,
                         __global const half *wd,
                         __global const char *aq,
                         __global const half *ad,
                         __global float *out, int blocks, int rows) {
    int row=(int)get_global_id(0);
    if(row>=rows) return;
    float sum=0.0f;
    for(int b=0;b<blocks;b++) {
        size_t group=((size_t)(row>>3)*blocks+b);
        size_t start=group*128+(row&7)*4;
        int dot=0;
        for(int j=0;j<16;j++) {
            uchar packed=wq[start+(j/4)*32+j%4];
            dot+=((int)(packed&15)-8)*(int)aq[b*32+j];
            dot+=((int)(packed>>4)-8)*(int)aq[b*32+j+16];
        }
        sum+=(float)dot*convert_float(wd[group*8+(row&7)])*convert_float(ad[b]);
    }
    out[row]=sum;
}
