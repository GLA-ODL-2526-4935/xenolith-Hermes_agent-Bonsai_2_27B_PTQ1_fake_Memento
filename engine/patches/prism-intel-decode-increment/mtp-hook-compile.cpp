#include "mtp-verify-copy.h"
#include <algorithm>
#include <cstdint>
using llama_seq_id=int;
static const float * llama_get_embeddings_nextn_ith(void *,int) {static float row[64];return row;}
struct actual_hook {
 std::vector<std::vector<float>> verify_h,pending_h;
 std::vector<int32_t> verify_h_rows,i_batch_beg; int n_embd=64;
 void capture(llama_seq_id seq_id,int32_t n_rows,void * ctx_tgt) {
 const size_t row_bytes=(size_t)n_embd*sizeof(float);
 (void)row_bytes;
            verify_h_rows[seq_id] = n_rows;
#if defined(LLAMA_MTP_COMPACT_VERIFY)
            common_mtp_capture_verify(verify_h[seq_id],pending_h[seq_id],n_rows,n_embd,
                [&](int32_t i) { return llama_get_embeddings_nextn_ith(ctx_tgt,i_batch_beg[seq_id]+i); });
#else
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
#endif
 }
 void accept(llama_seq_id seq_id,uint16_t n_accepted) {
        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        // process() already saved the final row. Full acceptance needs no copy.
        if (i_h != n_rows - 1) {
            std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
        }
 }
};
