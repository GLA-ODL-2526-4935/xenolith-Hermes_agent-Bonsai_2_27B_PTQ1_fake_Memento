// Target only; object compilation here is not executed parity.
#include "comparison-prefix.inc"
#define ptq1_block_codes ptq1_block_codes_v3
#define ptq1_pair_planes ptq1_pair_planes_v3
#define ggml_ptq1_pair_lut_tile ggml_ptq1_pair_lut_tile_v3
#include "ptq1-pair-lut-v3.h"
#undef GGML_PTQ1_PAIR_LUT_H
#undef ptq1_block_codes
#undef ptq1_pair_planes
#undef ggml_ptq1_pair_lut_tile
#define GGML_PTQ1_PAIR_LUT_COMPACT
#include "ptq1-pair-lut.h"
#include "mtp-verify-copy.h"
#include "mtp-hook-compile.cpp"
#include <algorithm>
int main() {
 for(int a=-128;a<=127;++a) for(int b=-128;b<=127;++b) {
  uint8_t old[32],now[32];
  _mm256_storeu_si256((__m256i *)old,ptq1_pair_planes_v3(a,b));
  _mm256_storeu_si256((__m256i *)now,ptq1_pair_planes(a,b));
  assert(!std::memcmp(old,now,32));
 }
 for(unsigned value=0;value<65536;++value) {
  block_ptq1_0 w={}; w.qh[0]=(uint8_t)value;w.qh[1]=(uint8_t)(value>>8);
  for(int i=0;i<24;++i) w.qs[i]=(uint8_t)(value+i*31);
  uint8_t old[129],now[129];old[128]=now[128]=0xaf;
  ptq1_block_codes_v3(&w,old);ptq1_block_codes(&w,now);
  assert(!std::memcmp(old,now,129));
 }
 // Defined scratch tails for every possible row count, and all skinny columns.
 for(int rows=1;rows<=16;++rows) for(int columns=1;columns<=5;++columns) {
  block_ptq1_0 w[16*3]={}; block_q8_0 a[5*12]={};
  for(int i=0;i<rows*3;++i) {w[i].d=0x3c00;for(int j=0;j<24;++j)w[i].qs[j]=(uint8_t)(i*7+j*31);w[i].qh[0]=(uint8_t)i;w[i].qh[1]=(uint8_t)~i;}
  for(int i=0;i<columns*12;++i) {a[i].d=0x2800;for(int j=0;j<32;++j)a[i].qs[j]=(int8_t)(i*17+j*13);}
  float old[5*19],now[5*19];std::fill_n(old,95,12345.0f);std::fill_n(now,95,12345.0f);
  ggml_ptq1_pair_lut_tile_v3(384,old,19*4,w,3*28,a,12*34,rows,columns);
  ggml_ptq1_pair_lut_tile(384,now,19*4,w,3*28,a,12*34,rows,columns);
  assert(!std::memcmp(old,now,sizeof(old)));
 }
 for(int width:{1,7,64}) {
  std::vector<float> earlier,pending(width);
  for(int rows:{1,2,3,5,128,513,1,5}) {
  std::vector<float> source(rows*width);
  const size_t capacity=earlier.capacity();
  // Includes signed zero and NaN payloads: copies must preserve bits.
  for(size_t i=0;i<source.size();++i){uint32_t bits=0x7fc00000u+(uint32_t)i;std::memcpy(&source[i],&bits,4);}
  std::vector<int> reads(rows,0);
  common_mtp_capture_verify(earlier,pending,rows,width,[&](int i){++reads[i];return source.data()+i*width;});
  assert(earlier.size()==(size_t)(rows-1)*width);
  assert(earlier.capacity()>=capacity);
  for(int i=0;i<rows;++i) assert(reads[i]==1);
  assert(!std::memcmp(pending.data(),source.data()+(rows-1)*width,width*4));
  for(int i=0;i<rows-1;++i) assert(!std::memcmp(earlier.data()+i*width,source.data()+i*width,width*4));
  std::vector<float> saved=source; std::fill(source.begin(),source.end(),0.0f);
  for(int accepted=0;accepted<=rows+1;++accepted) {
   const int index=std::min(accepted,rows-1);
   const float * selected=index==rows-1?pending.data():earlier.data()+index*width;
   assert(!std::memcmp(selected,saved.data()+index*width,width*4));
   actual_hook hook;hook.n_embd=width;hook.verify_h={earlier};hook.pending_h={pending};hook.verify_h_rows={rows};
   hook.accept(0,(uint16_t)accepted);
   assert(!std::memcmp(hook.pending_h[0].data(),saved.data()+index*width,width*4));
  }
  }
 }
}
