#ifndef XE_GEMMA4_ARCH_H
#define XE_GEMMA4_ARCH_H

#include <stdint.h>
#include <stddef.h>

/* E4B is the only model profile accepted by this fork. */

typedef struct {
    const char *name;
    uint32_t tensor_count;
    uint32_t block_count;
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t sliding_window;
    uint32_t swa_kv_head_count;
    uint32_t swa_head_dim;
    uint32_t global_kv_head_count;
    uint32_t global_head_dim;
    uint32_t shared_kv_layers;
    uint32_t per_layer_embedding_length;
    uint32_t vocab_size;
    uint32_t merge_count;
    float swa_rope_base;
    float global_rope_base;
    float rms_epsilon;
    float final_logit_softcap;
    float partial_rotary_factor;
    int attention_k_equals_v;
    int add_bos_token;
    int requires_eot_metadata;
    int runtime_ready;
    uint32_t chat_template_length;
    uint64_t chat_template_hash;
} xe_gemma4_model;

enum { XE_GEMMA4_MAX_LAYERS = 42 };

/* Text-backbone values from google/gemma-4-E4B. Optional tied output and
 * unused shared-layer KV tensors make tensor count converter-dependent.
 * The loader validates names/shapes/types and required tensors instead. */
static const xe_gemma4_model xe_gemma4_e4b = {
    .name = "gemma-4-e4b",
    .tensor_count = 0, /* No fixed-count contract. */
    .block_count = 42,
    .context_length = 131072,
    .embedding_length = 2560,
    .feed_forward_length = 10240,
    .attention_head_count = 8,
    .sliding_window = 512,
    .swa_kv_head_count = 2,
    .swa_head_dim = 256,
    .global_kv_head_count = 2,
    .global_head_dim = 512,
    .shared_kv_layers = 18,
    .per_layer_embedding_length = 256,
    .vocab_size = 262144,
    .merge_count = 514906,
    .swa_rope_base = 10000.0f,
    .global_rope_base = 1000000.0f,
    .rms_epsilon = 1e-6f,
    .final_logit_softcap = 30.0f,
    .partial_rotary_factor = 0.25f,
    .attention_k_equals_v = 0,
    .add_bos_token = 1,
    .requires_eot_metadata = 0,
    .runtime_ready = 0, /* Execution exists; external validation is pending. */
    /* Filled after the fine-tuned checkpoint/chat template is pinned. */
    .chat_template_length = 0,
    .chat_template_hash = 0,
};

static inline const xe_gemma4_model *
xe_gemma4_model_for_metadata(uint32_t block_count, uint32_t embedding_length) {
    if (block_count == xe_gemma4_e4b.block_count &&
        embedding_length == xe_gemma4_e4b.embedding_length)
        return &xe_gemma4_e4b;
    return NULL;
}

static inline int xe_gemma4_is_global(const xe_gemma4_model *model,
                                      uint32_t layer) {
    return model && layer < model->block_count && layer % 6 == 5;
}

static inline int xe_gemma4_layer_has_own_kv(const xe_gemma4_model *model,
                                             uint32_t layer) {
    return model && layer < model->block_count &&
           layer < model->block_count - model->shared_kv_layers;
}

static inline uint32_t xe_gemma4_kv_head_count(const xe_gemma4_model *model,
                                               uint32_t layer) {
    return xe_gemma4_is_global(model, layer) ? model->global_kv_head_count
                                             : model->swa_kv_head_count;
}

/* A shared layer uses the last physical donor of its attention type. */
static inline uint32_t xe_gemma4_kv_donor(const xe_gemma4_model *model,
                                        uint32_t layer) {
    if (!model || layer >= model->block_count) return XE_GEMMA4_MAX_LAYERS;
    if (xe_gemma4_layer_has_own_kv(model, layer)) return layer;
    uint32_t donor = model->block_count - model->shared_kv_layers;
    while (donor > 0) {
        --donor;
        if (xe_gemma4_is_global(model, donor) == xe_gemma4_is_global(model, layer))
            return donor;
    }
    return XE_GEMMA4_MAX_LAYERS;
}

#endif
