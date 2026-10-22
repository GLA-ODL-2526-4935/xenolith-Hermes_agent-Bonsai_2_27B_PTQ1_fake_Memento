// Exact owned hidden-row capture; upstream llama.cpp MIT license applies.
// Does not change KV/recurrent state, row selection or acceptance policy.
#ifndef COMMON_MTP_VERIFY_COPY_H
#define COMMON_MTP_VERIFY_COPY_H
#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

template<class RowGetter>
static void common_mtp_capture_verify(std::vector<float> & earlier,
        std::vector<float> & pending, int rows, int width, RowGetter row) {
    assert(rows > 0 && width > 0 && pending.size() == (std::size_t) width);
    const std::size_t bytes = (std::size_t) width * sizeof(float);
    // Own every row needed after the next decode. Never borrow context output
    // pointers across a graph call. Keep earlier.capacity() for future cycles.
    earlier.resize((std::size_t) (rows-1) * width);
    for (int i=0;i<rows-1;++i) {
        std::memcpy(earlier.data()+(std::size_t) i*width,row(i),bytes);
    }
    // The final row is already the full-acceptance carryover: copy it directly
    // once. accept() reads earlier only when i_h != rows-1.
    std::memcpy(pending.data(),row(rows-1),bytes);
}
#endif
