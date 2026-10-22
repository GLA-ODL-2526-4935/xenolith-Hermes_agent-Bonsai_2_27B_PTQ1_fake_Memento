#include "models/gemma4_arch.h"

#include <stdio.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "model-profile check failed at line %d: %s\n", \
                __LINE__, #expr); \
        return 1; \
    } \
} while (0)

static int check_global_pattern(const xe_gemma4_model *model,
                                uint32_t expected_global) {
    uint32_t global = 0;
    for (uint32_t layer = 0; layer < model->block_count; layer++)
        global += xe_gemma4_is_global(model, layer) != 0;
    CHECK(global == expected_global);
    return 0;
}

int main(void) {
    const xe_gemma4_model *e4b = xe_gemma4_model_for_metadata(42, 2560);

    CHECK(e4b == &xe_gemma4_e4b);
    CHECK(xe_gemma4_model_for_metadata(30, 2816) == NULL);
    CHECK(xe_gemma4_model_for_metadata(42, 2816) == NULL);

    CHECK(!e4b->runtime_ready);
    CHECK(e4b->block_count == 42);
    CHECK(e4b->feed_forward_length == 10240);
    CHECK(e4b->shared_kv_layers == 18);
    CHECK(e4b->per_layer_embedding_length == 256);
    CHECK(e4b->embedding_length == 2560);
    CHECK(e4b->attention_head_count == 8);
    CHECK(e4b->swa_kv_head_count == 2 && e4b->global_kv_head_count == 2);
    CHECK(e4b->swa_head_dim == 256 && e4b->global_head_dim == 512);
    CHECK(check_global_pattern(e4b, 7) == 0);

    for (uint32_t layer = 0; layer < 24; layer++)
        CHECK(xe_gemma4_layer_has_own_kv(e4b, layer));
    for (uint32_t layer = 24; layer < e4b->block_count; layer++) {
        CHECK(!xe_gemma4_layer_has_own_kv(e4b, layer));
        CHECK(xe_gemma4_kv_donor(e4b, layer) ==
              (xe_gemma4_is_global(e4b, layer) ? 23u : 22u));
    }
    for (uint32_t layer = 0; layer < 24; layer++)
        CHECK(xe_gemma4_kv_donor(e4b, layer) == layer);
    CHECK(xe_gemma4_kv_donor(e4b, 42) == XE_GEMMA4_MAX_LAYERS);

    puts("E4B-only model metadata/layout checks PASS");
    return 0;
}
