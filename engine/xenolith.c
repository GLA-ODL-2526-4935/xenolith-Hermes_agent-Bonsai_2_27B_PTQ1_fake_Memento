#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "xenolith.h"
#include "xenolith_internal.h"
#include "format.h"
#include "models/gemma4_arch.h"

#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <level_zero/ze_api.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef XE_NUMERIC_SOURCE_HASH
#define XE_NUMERIC_SOURCE_HASH "unknown"
#endif

#ifndef XE_NUMERIC_CFLAGS_HASH
#define XE_NUMERIC_CFLAGS_HASH "unknown"
#endif

static _Noreturn void xe_fatal(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "xenolith: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

typedef enum { XE_MEM_HOST, XE_MEM_SHARED } xe_mem_kind;
typedef enum { XE_RELAXED_NONE, XE_RELAXED_EXP, XE_RELAXED_EXT } xe_relaxed_limits;

/* We centralize the allocation here because when implementing GPU prefill
 * the buffer touched by GPU should be allocated with zeMemAllocShared().
 * Memory allocated with plain malloc would not be visible by GPU. */
static void *xe_alloc(const xe_engine *e, size_t n, xe_mem_kind kind);
static void xe_free(const xe_engine *e, void *p, xe_mem_kind kind);

#ifdef XE_TEST_ALLOC
size_t xe_test_allocations;
#endif

#ifdef XE_TEST_OUTPUT_COUNT
size_t xe_test_output_calls;
#endif

/* E4B text backbone. Physical KV storage belongs only to the first 24 layers. */
#define XE_SOFTCAP_SECOND_LOOP 0
#define XE_LAYERS 42
#define XE_KV_LAYERS 24
#define XE_PLE 256
#define XE_PLE_WIDTH (XE_LAYERS * XE_PLE)
#define XE_EMBD 2560
#define XE_VOCAB 262144
#define XE_MODEL_CTX XE_CONTEXT_MAX
#define XE_Q_HEADS 8
#define XE_DENSE_FFN 10240
#define XE_SWA_WINDOW 512
#define XE_SWA_KV_HEADS 2
#define XE_SWA_HEAD_DIM 256
#define XE_SWA_ROPE_BASE 10000.0f
#define XE_REPACK_THREADS 6
#ifndef XE_WORKERS
#define XE_WORKERS 6
#endif
#define XE_GLOBAL_KV_HEADS 2
#define XE_GLOBAL_HEAD_DIM 512
#define XE_GLOBAL_ROPE_BASE 1000000.0f
#define XE_REPACK_PARTS 64
#define XE_REPACK_PART_LIMIT (UINT64_C(512) * 1024 * 1024)
#define XE_RMS_EPS 1e-6f
#define XE_LOGIT_SOFTCAP 30.0f
#define XE_TENSOR_COUNT 768
#define XE_MERGES 514906
#define XE_BYTE0_ID 238
#define XE_TOK_HASH 524288
#define XE_MERGE_HASH 1048576
#define XE_TT_NORMAL 1
#define XE_TT_BYTE 6
#define XE_CHANNEL_BEGIN_ID 100
#define XE_CHANNEL_END_ID 101
#define XE_TURN_BEGIN_ID 105
#define XE_TURN_END_ID 106
#define XE_SWA_LAYER_ELEMS ((size_t)XE_SWA_KV_HEADS * XE_SWA_WINDOW * XE_SWA_HEAD_DIM)
#define XE_SWA_SLAB_ELEMS ((size_t)XE_SWA_LAYERS * XE_SWA_LAYER_ELEMS)
#define XE_GLOBAL_LAYERS 4
#define XE_SWA_LAYERS 20

#define XE_IS_GLOBAL(i) ((i) % 6 == 5)

#define XE_GGUF_MAGIC 0x46554747u
#define XE_GGUF_VERSION 3

enum { XE_T_U8, XE_T_I8, XE_T_U16, XE_T_I16, XE_T_U32, XE_T_I32, XE_T_F32,
       XE_T_BOOL, XE_T_STRING, XE_T_ARRAY, XE_T_U64, XE_T_I64, XE_T_F64 };

#define XE_GGML_TYPE_F32 0
#define XE_GGML_TYPE_Q4_0 2

/* The binding loop in xe_bind_tensors() relies on this exact field order
 * to write the mmap pointer to this struct with a simple plain memcpy().
 * Do not reorder the fields. */
typedef struct {
    const uint8_t *qs;
    const uint16_t *d;
    int blocks;
} xe_q4;

typedef struct {
    int8_t *qs;
    _Float16 *d;
    int16_t *sigma;
    int n;
} xe_q8;

typedef struct {
    int32_t id;
    float logit;
    float prob;
} xe_sample_candidate;

typedef void (*xe_phase_fn)(xe_session *, const void *, int, int);

typedef struct {
    xe_phase_fn fn;
    xe_session *session;
    const void *arg;
    int caller_participates;
} xe_phase;

typedef struct {
    _Alignas(64) atomic_uint_fast64_t value;
} xe_phase_signal;

typedef struct {
    xe_engine *engine;
    int lane;
} xe_worker_arg;

typedef struct xe_layer {
    xe_q4 attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down, inp_gate, proj;
    const float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *post_norm, *layer_out_scale;
} xe_layer;

typedef struct {
    const char *p;
    uint64_t len;
} xe_str;

typedef struct {
    uint64_t key;
    uint32_t rank;
    int32_t res;
} xe_merge;

typedef struct {
    ze_driver_handle_t driver;
    ze_device_handle_t device;
    ze_context_handle_t context;
    uint64_t max_mem_alloc_size;
    xe_relaxed_limits relaxed_limits;
    ze_command_list_handle_t commands;
    ze_module_handle_t module;
    ze_kernel_handle_t e4b_q4q8;
} xe_gpu;

struct xe_engine {
    const xe_gemma4_model *model;
    int context;
    void *map;
    size_t map_len;
    int map_fd;
    const void *data;
    size_t header_len;   /* bytes before the tensor data section: metadata + tensor directory */
    uint32_t gguf_version;
    char *name;

    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;

    xe_q4 tok_embd, output, per_layer_token_embd, per_layer_model_proj;
    const float *per_layer_proj_norm;
    const float *out_norm;
    const float *rope_freqs;

    xe_layer layers[XE_LAYERS];

    const void *tok_tokens; uint64_t tok_tokens_count;
    const void *tok_scores; uint64_t tok_scores_count;
    const void *tok_token_type; uint64_t tok_token_type_count;
    const void *tok_merges; uint64_t tok_merges_count;
    xe_str chat_template;

    int32_t bos_id, eos_id, eot_id, unk_id, pad_id;

    xe_str *tok_piece;
    int32_t *tok_hash;
    xe_merge *tok_merge;
    uint32_t tok_max_piece;
    double tok_build_seconds;
    int vocab_only;

    uint64_t q4_0_count, q4_0_bytes;
    uint64_t f32_count, f32_bytes;
    uint64_t dense_bytes, ple_bytes, tok_embd_bytes;

    void *repack_parts[XE_REPACK_PARTS];
    uint64_t repack_part_sizes[XE_REPACK_PARTS];
    int repack_part_count;
    uint64_t repack_slab_size;
    uint64_t repack_nibble_bytes, repack_scale_bytes;
    double repack_seconds;
    uint64_t repack_verified_blocks;
    void *f32_slab;
    uint64_t f32_slab_size;

    xe_gpu gpu;

    pthread_t followers[XE_WORKERS - 1];
    xe_worker_arg worker_args[XE_WORKERS - 1];
    pthread_mutex_t worker_mutex;
    pthread_cond_t worker_cv;
    pthread_cond_t ready_cv;
    pthread_t owner;
    cpu_set_t owner_affinity;
    uint64_t worker_ticket;
    int worker_ready;
    int worker_stop;
    int pool_initialized;
    int worker_pinned;
    int token_active;
    xe_phase phase;
    atomic_uint_fast64_t phase_epoch;
    xe_phase_signal phase_done[XE_WORKERS];
    float rope_swa_inv[XE_SWA_HEAD_DIM / 2];
    float rope_global_inv[XE_GLOBAL_HEAD_DIM / 2];
    int scalar_rms;
    uint8_t snapshot_fingerprint[6][32];
    int snapshot_fingerprint_ready;
    uint8_t model_content_hash[32];
    int model_content_hash_ready;
};

struct xe_session {
    xe_engine *engine;
    _Float16 *swa_k;
    _Float16 *swa_v;
    _Float16 *swa_spare_k;
    _Float16 *swa_spare_v;
    uint8_t swa_dirty[XE_SWA_WINDOW];
    _Float16 *global_k;
    _Float16 *global_v;
    _Float16 *cow_global_k[XE_GLOBAL_LAYERS];
    _Float16 *cow_global_v[XE_GLOBAL_LAYERS];
    void *workspace;
    size_t workspace_size;
    float *hidden;
    float *q;
    float *k;
    float *v;
    float *attn_heads;
    float *attn_proj;
    float *attn_out;
    float *dense_out;
    float *combined;
    float *normed, *gate, *up, *activation;
    float *ple_inputs, *ple_lookup, *ple_gate;
    int gpu_prefill;
    int reference_mode;
    const char *dump_layers;
    int dump_position;
    float *scores;
    float *rope_swa_cos;
    float *rope_swa_sin;
    float *rope_global_cos;
    float *rope_global_sin;
    float *logits;
    xe_sample_candidate *sample_candidates;
    xe_q8 q8_main;
    int32_t *tokens;
    int n_tokens;
    _Float16 *anchor_swa_k;
    _Float16 *anchor_swa_v;
    float *anchor_logits;
    int anchor_position;
    int anchor_valid;
    xe_session *cow_source;
    int cow;
    int cow_split;
    int cow_capacity;
    int prefill_pending;
    int prefill_pending_start;
    int prefill_pending_rows;
    float *prefill_pending_hidden;
};

static size_t xe_global_layer_elems(const xe_engine *e) {
    return (size_t)XE_GLOBAL_KV_HEADS * e->context * XE_GLOBAL_HEAD_DIM;
}

static size_t xe_global_slab_elems(const xe_engine *e) {
    return (size_t)XE_GLOBAL_LAYERS * xe_global_layer_elems(e);
}

extern const unsigned char _binary_xenolith_gpu_spv_start[];
extern const unsigned char _binary_xenolith_gpu_spv_end[];

static _Noreturn void xe_ze_fatal(const char *operation, ze_result_t result) {
    xe_fatal("%s failed: 0x%x", operation, result);
}

static void xe_ze_check(const char *operation, ze_result_t result) {
    if (result != ZE_RESULT_SUCCESS) xe_ze_fatal(operation, result);
}

static void *xe_alloc(const xe_engine *e, size_t n, xe_mem_kind kind) {
#ifdef XE_TEST_ALLOC
    extern size_t xe_test_allocations;
    xe_test_allocations++;
#endif
    if (n > SIZE_MAX - 63) xe_fatal("allocation size overflow: %zu", n);
    size_t rounded = (n + 63) / 64 * 64;
    if (kind == XE_MEM_HOST) {
        void *p = aligned_alloc(64, rounded);
        if (!p) xe_fatal("out of memory allocating %zu bytes", rounded);
        return p;
    }
    if (kind != XE_MEM_SHARED) xe_fatal("invalid memory kind %d", kind);
    if (!e || !e->gpu.context) xe_fatal("shared allocation without a GPU context");
    ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    };
    ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
    };
    ze_relaxed_allocation_limits_exp_desc_t relaxed_exp = {
        .stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC,
        .flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE
    };
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
    ze_relaxed_allocation_limits_ext_desc_t relaxed_ext = {
        .stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC,
        .flags = ZE_RELAXED_ALLOCATION_LIMITS_EXT_FLAG_MAX_SIZE
    };
#endif
    if (rounded > e->gpu.max_mem_alloc_size) {
        if (e->gpu.relaxed_limits == XE_RELAXED_EXP)
            device_desc.pNext = &relaxed_exp;
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
        else if (e->gpu.relaxed_limits == XE_RELAXED_EXT)
            device_desc.pNext = &relaxed_ext;
#endif
        else
            xe_fatal("shared allocation of %zu bytes exceeds device limit %llu; "
                     "driver does not support relaxed allocation limits",
                     rounded, (unsigned long long)e->gpu.max_mem_alloc_size);
    }
    void *p = NULL;
    ze_result_t result = zeMemAllocShared(e->gpu.context, &device_desc, &host_desc,
                                        rounded, 64, e->gpu.device, &p);
    if (result != ZE_RESULT_SUCCESS)
        xe_fatal("zeMemAllocShared failed: 0x%x allocating %zu bytes "
                 "(device limit %llu, relaxed limits %s)", result, rounded,
                 (unsigned long long)e->gpu.max_mem_alloc_size,
                 device_desc.pNext ? "enabled" : "disabled");
    return p;
}

static void xe_free(const xe_engine *e, void *p, xe_mem_kind kind) {
    if (!p) return;
    if (kind == XE_MEM_HOST) {
        free(p);
        return;
    }
    if (kind != XE_MEM_SHARED) xe_fatal("invalid memory kind %d", kind);
    if (!e || !e->gpu.context) xe_fatal("shared free without a GPU context");
    xe_ze_check("zeMemFree", zeMemFree(e->gpu.context, p));
}

typedef enum {
    XE_GPU_NOT_ENABLED,
    XE_GPU_TESTED,
    XE_GPU_UNTESTED
} xe_gpu_status;

static xe_gpu_status xe_gpu_classify(uint32_t id) {
    /* Integrated GPUs only. Keep these IDs and test status in sync with README.md.
     * IDs: intel/compute-runtime shared/source/dll/devices/devices_base.inl;
     * Wildcat Lake grouping: Linux include/drm/intel/pciids.h. */
    switch (id) {
        case 0xa7a0: /* Maintainer's Core i7-13700H. */
        case 0x7d51: /* Core Ultra 7 255H, reported by aziis98 in PR #2. */
        case 0x64a0: /* Core Ultra 7 268V; Raffaele's WSL report in PR #6, 64k capacity. */
            return XE_GPU_TESTED;
        /* Tiger Lake. */
        case 0x9a40: case 0x9a49: case 0x9a59: case 0x9a60:
        case 0x9a68: case 0x9a70: case 0x9a78:
        /* Rocket Lake. */
        case 0x4c80: case 0x4c8a: case 0x4c8b: case 0x4c8c:
        case 0x4c90: case 0x4c9a:
        /* Alder Lake mobile. */
        case 0x4626: case 0x4628: case 0x462a: case 0x46a0:
        case 0x46a1: case 0x46a3: case 0x46a6: case 0x46a8:
        case 0x46aa: case 0x46b0: case 0x46b1: case 0x46b3:
        case 0x46c0: case 0x46c1: case 0x46c3:
        /* Alder Lake desktop. */
        case 0x4680: case 0x4682: case 0x4688: case 0x468a:
        case 0x468b: case 0x4690: case 0x4692: case 0x4693:
        /* Alder Lake-N / Twin Lake. */
        case 0x46d0: case 0x46d1: case 0x46d2: case 0x46d3: case 0x46d4:
        /* Raptor Lake mobile / refresh. */
        case 0xa720: case 0xa721: case 0xa7a1: case 0xa7a8:
        case 0xa7a9: case 0xa7aa: case 0xa7ab: case 0xa7ac: case 0xa7ad:
        /* Raptor Lake desktop / refresh. */
        case 0xa780: case 0xa781: case 0xa782: case 0xa783:
        case 0xa788: case 0xa789: case 0xa78a: case 0xa78b:
        /* Meteor Lake and Arrow Lake. */
        case 0x7d40: case 0x7d45: case 0x7d55: case 0x7dd5:
        case 0x7d41: case 0x7d67: case 0x7dd1:
        /* Lunar Lake. */
        case 0x6420: case 0x64b0:
        /* Panther Lake. */
        case 0xb080: case 0xb081: case 0xb082: case 0xb083:
        case 0xb084: case 0xb085: case 0xb086: case 0xb087:
        case 0xb08f: case 0xb090: case 0xb0a0: case 0xb0b0:
        /* Wildcat Lake. */
        case 0xfd80: case 0xfd81:
        /* Nova Lake, Xe3 (IGFX_NVL_XE3G). */
        case 0xd740: case 0xd741: case 0xd742: case 0xd743:
        case 0xd744: case 0xd745:
        /* Nova Lake, Xe3P (IGFX_NVL). */
        case 0xd74a: case 0xd74b:
        case 0xd750: case 0xd751: case 0xd752: case 0xd753:
        case 0xd754: case 0xd755: case 0xd756: case 0xd757: case 0xd75f:
            return XE_GPU_UNTESTED;
        default:
            return XE_GPU_NOT_ENABLED;
    }
}

static xe_relaxed_limits xe_gpu_get_relaxed_limits(ze_driver_handle_t driver) {
    uint32_t count = 0;
    xe_ze_check("zeDriverGetExtensionProperties count",
                zeDriverGetExtensionProperties(driver, &count, NULL));
    if (!count) return XE_RELAXED_NONE;
    ze_driver_extension_properties_t *extensions =
        xe_alloc(NULL, (size_t)count * sizeof(*extensions), XE_MEM_HOST);
    xe_ze_check("zeDriverGetExtensionProperties",
                zeDriverGetExtensionProperties(driver, &count, extensions));
    xe_relaxed_limits limits = XE_RELAXED_NONE;
    for (uint32_t i = 0; i < count; i++) {
        if (extensions[i].version < ZE_MAKE_VERSION(1, 0)) continue;
        if (limits == XE_RELAXED_NONE &&
            strcmp(extensions[i].name, ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME) == 0)
            limits = XE_RELAXED_EXP;
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
        if (strcmp(extensions[i].name, ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME) == 0)
            limits = XE_RELAXED_EXT;
#endif
    }
    xe_free(NULL, extensions, XE_MEM_HOST);
    return limits;
}

static void xe_gpu_init(xe_engine *e) {
    xe_ze_check("zeInit", zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count = 0;
    xe_ze_check("zeDriverGet count", zeDriverGet(&driver_count, NULL));
    if (!driver_count) xe_fatal("no Level Zero GPU driver");
    driver_count = 1;
    xe_ze_check("zeDriverGet", zeDriverGet(&driver_count, &e->gpu.driver));

    uint32_t device_count = 0;
    xe_ze_check("zeDeviceGet count", zeDeviceGet(e->gpu.driver, &device_count, NULL));
    if (!device_count) xe_fatal("no Level Zero GPU device");
    device_count = 1;
    xe_ze_check("zeDeviceGet", zeDeviceGet(e->gpu.driver, &device_count, &e->gpu.device));

    ze_device_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
    };
    xe_ze_check("zeDeviceGetProperties", zeDeviceGetProperties(e->gpu.device, &properties));
    e->gpu.max_mem_alloc_size = properties.maxMemAllocSize;
    e->gpu.relaxed_limits = xe_gpu_get_relaxed_limits(e->gpu.driver);
    xe_gpu_status status = xe_gpu_classify(properties.deviceId);
    if (properties.vendorId != 0x8086 || status == XE_GPU_NOT_ENABLED)
        xe_fatal("GPU %04x:%04x is not enabled; see README.md for Intel Xe integrated GPU device IDs",
                 properties.vendorId, properties.deviceId);
    if (status == XE_GPU_UNTESTED)
        fprintf(stderr, "xenolith: %s (%04x:%04x) is enabled but has not yet been tested with Xenolith; "
                        "using existing kernels without XMX.\n"
                        "xenolith: if inference works, please report your results so this device can be marked tested; "
                        "if it fails, please report the problem. Include your device ID, system/driver versions, "
                        "Xenolith commit, command and output: https://github.com/simoneiacomino/xenolith/issues\n",
                properties.name, properties.vendorId, properties.deviceId);

    ze_context_desc_t context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC
    };
    xe_ze_check("zeContextCreate",
                zeContextCreate(e->gpu.driver, &context_desc, &e->gpu.context));

    ze_command_queue_desc_t queue_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = 0,
        .index = 0,
        .flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    xe_ze_check("zeCommandListCreateImmediate",
                zeCommandListCreateImmediate(e->gpu.context, e->gpu.device,
                                             &queue_desc, &e->gpu.commands));
    size_t spv_size = (size_t)(_binary_xenolith_gpu_spv_end -
                               _binary_xenolith_gpu_spv_start);
    ze_module_desc_t module_desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = spv_size,
        .pInputModule = _binary_xenolith_gpu_spv_start,
        .pBuildFlags = ""
    };
    ze_module_build_log_handle_t log = NULL;
    ze_result_t result = zeModuleCreate(e->gpu.context, e->gpu.device, &module_desc,
                                        &e->gpu.module, &log);
    if (result != ZE_RESULT_SUCCESS) {
        if (log) {
            size_t log_size = 0;
            zeModuleBuildLogGetString(log, &log_size, NULL);
            char *message = xe_alloc(e, log_size + 1, XE_MEM_HOST);
            zeModuleBuildLogGetString(log, &log_size, message);
            message[log_size] = 0;
            fprintf(stderr, "%s\n", message);
            xe_free(e, message, XE_MEM_HOST);
        }
        xe_ze_fatal("zeModuleCreate", result);
    }
    if (log) xe_ze_check("zeModuleBuildLogDestroy", zeModuleBuildLogDestroy(log));

    ze_kernel_desc_t e4b_desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pKernelName = "xe_e4b_q4q8"
    };
    xe_ze_check("zeKernelCreate E4B", zeKernelCreate(e->gpu.module, &e4b_desc, &e->gpu.e4b_q4q8));
    xe_ze_check("zeKernelSetGroupSize E4B", zeKernelSetGroupSize(e->gpu.e4b_q4q8, 64, 1, 1));
    ze_kernel_desc_t kernel_desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pKernelName = "xe_probe"
    };
    ze_kernel_handle_t probe;
    xe_ze_check("zeKernelCreate probe",
                zeKernelCreate(e->gpu.module, &kernel_desc, &probe));
    xe_ze_check("zeKernelSetGroupSize probe", zeKernelSetGroupSize(probe, 1, 1, 1));
    uint32_t *value = xe_alloc(e, sizeof(*value), XE_MEM_SHARED);
    *value = 0;
    xe_ze_check("zeKernelSetArgumentValue probe",
                zeKernelSetArgumentValue(probe, 0, sizeof(value), &value));
    ze_group_count_t groups = { 1, 1, 1 };
    xe_ze_check("zeCommandListAppendLaunchKernel probe",
                zeCommandListAppendLaunchKernel(e->gpu.commands, probe, &groups,
                                                NULL, 0, NULL));
    xe_ze_check("zeCommandListHostSynchronize probe",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    if (*value != UINT32_C(0x78656e6f))
        xe_fatal("GPU probe returned 0x%08x", *value);
    xe_free(e, value, XE_MEM_SHARED);
    xe_ze_check("zeKernelDestroy probe", zeKernelDestroy(probe));
}

static void xe_gpu_destroy(xe_engine *e) {
    if (e->gpu.commands)
        xe_ze_check("zeCommandListHostSynchronize close",
                    zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    if (e->gpu.e4b_q4q8)
        xe_ze_check("zeKernelDestroy E4B", zeKernelDestroy(e->gpu.e4b_q4q8));
    if (e->gpu.module) xe_ze_check("zeModuleDestroy", zeModuleDestroy(e->gpu.module));
    if (e->gpu.commands)
        xe_ze_check("zeCommandListDestroy", zeCommandListDestroy(e->gpu.commands));
    if (e->gpu.context) xe_ze_check("zeContextDestroy", zeContextDestroy(e->gpu.context));
    memset(&e->gpu, 0, sizeof e->gpu);
}

static void xe_gpu_pointer_arg(ze_kernel_handle_t kernel, uint32_t index,
                               const void *pointer) {
    xe_ze_check("zeKernelSetArgumentValue pointer",
                zeKernelSetArgumentValue(kernel, index, sizeof(pointer),
                                         &pointer));
}

static void xe_gpu_int_arg(ze_kernel_handle_t kernel, uint32_t index,
                           int value) {
    xe_ze_check("zeKernelSetArgumentValue int",
                zeKernelSetArgumentValue(kernel, index, sizeof(value), &value));
}

#ifdef XE_BENCH_TG_PROFILE
typedef struct {
    double embed_rope;
    double attention;
    double dense;
    double glue;
    double output;
    double worker_end;
} xe_tg_profile;

static xe_tg_profile xe_tg_profile_data;

static double xe_tg_profile_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

typedef struct {
    uint8_t *base;
    size_t pos;
} xe_workspace_arena;




static void xe_constants_init(xe_engine *e) {
    for (int i=0;i<XE_SWA_HEAD_DIM/2;i++)
        e->rope_swa_inv[i]=powf(XE_SWA_ROPE_BASE,-2.0f*i/XE_SWA_HEAD_DIM);
    /* Proportional RoPE keeps full-head NeoX pairing and exponent spacing,
     * rotating only the first quarter of the pairs. */
    int active=(int)(XE_GLOBAL_HEAD_DIM/2*e->model->partial_rotary_factor);
    for (int i=0;i<XE_GLOBAL_HEAD_DIM/2;i++)
        e->rope_global_inv[i]=i<active ?
            powf(XE_GLOBAL_ROPE_BASE,-2.0f*i/XE_GLOBAL_HEAD_DIM)/e->rope_freqs[i] : 0.0f;
}


static void xe_pin_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    if (rc != 0) xe_fatal("pthread_setaffinity_np CPU %d: %s", cpu, strerror(rc));
}

static int xe_worker_cpu(int lane) {
#ifdef XE_WORKER_ECORES
    return 12 + lane;
#else
    return 2 * lane;
#endif
}

static int xe_worker_pin_available(const cpu_set_t *allowed) {
    int packages[XE_WORKERS];
    int cores[XE_WORKERS];
    for (int lane = 0; lane < XE_WORKERS; lane++) {
        int cpu = xe_worker_cpu(lane);
        if (!CPU_ISSET(cpu, allowed)) return 0;
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        FILE *f = fopen(path, "r");
        if (!f) return 0;
        int package;
        int read = fscanf(f, "%d", &package);
        fclose(f);
        if (read != 1) return 0;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        f = fopen(path, "r");
        if (!f) return 0;
        int core;
        read = fscanf(f, "%d", &core);
        fclose(f);
        if (read != 1) return 0;
        for (int other = 0; other < lane; other++)
            if (packages[other] == package && cores[other] == core) return 0;
        packages[lane] = package;
        cores[lane] = core;
    }
    return 1;
}

static void xe_require_owner(const xe_engine *e) {
    if (!e->pool_initialized || !pthread_equal(e->owner, pthread_self()))
        xe_fatal("decode called from a thread other than the engine owner");
}

static uint64_t xe_wait_changed(atomic_uint_fast64_t *value, uint64_t seen) {
    uint64_t current;
    do {
        current = atomic_load_explicit(value, memory_order_acquire);
        if (current == seen) {
#ifdef XE_LOW_POWER_WAIT
            _umonitor((void *)value);
            if (atomic_load_explicit(value, memory_order_acquire) == seen)
                _umwait(0, __rdtsc() + UINT64_C(100000));
#else
            _mm_pause();
#endif
        }
    } while (current == seen);
    return current;
}

static void xe_wait_equal(atomic_uint_fast64_t *value, uint64_t expected) {
    for (;;) {
        if (atomic_load_explicit(value, memory_order_acquire) == expected) return;
#ifdef XE_LOW_POWER_WAIT
        _umonitor((void *)value);
        if (atomic_load_explicit(value, memory_order_acquire) != expected)
            _umwait(0, __rdtsc() + UINT64_C(100000));
#else
        _mm_pause();
#endif
    }
}

static void *xe_worker_main(void *opaque) {
    xe_worker_arg *wa = opaque;
    xe_engine *e = wa->engine;
    int lane = wa->lane;
    uint64_t seen = 0;

    if (e->worker_pinned) xe_pin_thread(xe_worker_cpu(lane));

    pthread_mutex_lock(&e->worker_mutex);
    e->worker_ready++;
    pthread_cond_signal(&e->ready_cv);
    uint64_t ticket = e->worker_ticket;

    for (;;) {
        while (!e->worker_stop && ticket == e->worker_ticket)
            pthread_cond_wait(&e->worker_cv, &e->worker_mutex);
        if (e->worker_stop) break;
        ticket = e->worker_ticket;
        pthread_mutex_unlock(&e->worker_mutex);

        for (;;) {
            uint64_t epoch = xe_wait_changed(&e->phase_epoch, seen);

            xe_phase phase = e->phase;
            if (phase.fn) {
                int worker = phase.caller_participates ? lane : lane - 1;
                int workers = phase.caller_participates ? XE_WORKERS : XE_WORKERS - 1;
                phase.fn(phase.session, phase.arg, worker, workers);
            }
            atomic_store_explicit(&e->phase_done[lane].value, epoch, memory_order_release);
            seen = epoch;
            if (!phase.fn) break;
        }

        pthread_mutex_lock(&e->worker_mutex);
    }

    pthread_mutex_unlock(&e->worker_mutex);
    return NULL;
}

static void xe_worker_pool_init(xe_engine *e) {
    if (e->pool_initialized) {
        xe_require_owner(e);
        return;
    }

    e->owner = pthread_self();
    e->worker_pinned = pthread_getaffinity_np(e->owner, sizeof e->owner_affinity,
                                                &e->owner_affinity) == 0 &&
                       xe_worker_pin_available(&e->owner_affinity);
    if (pthread_mutex_init(&e->worker_mutex, NULL) != 0 ||
        pthread_cond_init(&e->worker_cv, NULL) != 0 ||
        pthread_cond_init(&e->ready_cv, NULL) != 0)
        xe_fatal("worker synchronization initialization failed");

    e->worker_ticket = 0;
    e->worker_ready = 0;
    e->worker_stop = 0;
    e->token_active = 0;
    e->phase.fn = NULL;
    e->phase.session = NULL;
    e->phase.arg = NULL;
    e->phase.caller_participates = 0;
    atomic_init(&e->phase_epoch, 0);
    for (int lane = 0; lane < XE_WORKERS; lane++)
        atomic_init(&e->phase_done[lane].value, 0);
    e->pool_initialized = 1;

    xe_constants_init(e);

    if (e->worker_pinned) xe_pin_thread(xe_worker_cpu(0));

    pthread_mutex_lock(&e->worker_mutex);
    for (int lane = 1; lane < XE_WORKERS; lane++) {
        xe_worker_arg *wa = &e->worker_args[lane - 1];
        wa->engine = e;
        wa->lane = lane;
        int rc = pthread_create(&e->followers[lane - 1], NULL, xe_worker_main, wa);
        if (rc != 0) xe_fatal("pthread_create worker %d: %s", lane, strerror(rc));
    }
    while (e->worker_ready != XE_WORKERS - 1)
        pthread_cond_wait(&e->ready_cv, &e->worker_mutex);
    pthread_mutex_unlock(&e->worker_mutex);
}

static void xe_worker_pool_destroy(xe_engine *e) {
    if (!e->pool_initialized) return;
    xe_require_owner(e);
    if (e->token_active) xe_fatal("engine closed during an active decode token");

    pthread_mutex_lock(&e->worker_mutex);
    e->worker_stop = 1;
    pthread_cond_broadcast(&e->worker_cv);
    pthread_mutex_unlock(&e->worker_mutex);

    for (int i = 0; i < XE_WORKERS - 1; i++)
        pthread_join(e->followers[i], NULL);

    pthread_cond_destroy(&e->ready_cv);
    pthread_cond_destroy(&e->worker_cv);
    pthread_mutex_destroy(&e->worker_mutex);
    e->pool_initialized = 0;
    if (e->worker_pinned) {
        int rc = pthread_setaffinity_np(e->owner, sizeof e->owner_affinity, &e->owner_affinity);
        if (rc != 0) xe_fatal("pthread_setaffinity_np restore: %s", strerror(rc));
    }
}

static void xe_workers_begin(xe_engine *e) {
    xe_require_owner(e);
    if (e->token_active) xe_fatal("nested decode token");
    e->token_active = 1;
    pthread_mutex_lock(&e->worker_mutex);
    e->worker_ticket++;
    pthread_cond_broadcast(&e->worker_cv);
    pthread_mutex_unlock(&e->worker_mutex);
}

static uint64_t xe_publish_phase(xe_engine *e, xe_session *s, xe_phase_fn fn,
                                 const void *arg, int caller_participates) {
    e->phase.fn = fn;
    e->phase.session = s;
    e->phase.arg = arg;
    e->phase.caller_participates = caller_participates;
    uint64_t epoch = atomic_load_explicit(&e->phase_epoch, memory_order_relaxed) + 1;
    atomic_store_explicit(&e->phase_epoch, epoch, memory_order_release);
    return epoch;
}

static void xe_wait_followers(xe_engine *e, uint64_t epoch) {
    for (int lane = 1; lane < XE_WORKERS; lane++)
        xe_wait_equal(&e->phase_done[lane].value, epoch);
}

static void xe_dispatch(xe_engine *e, xe_session *s, xe_phase_fn fn,
                        const void *arg, int caller_participates) {
    xe_require_owner(e);
    if (!e->token_active || !fn) xe_fatal("invalid worker phase");
    uint64_t epoch = xe_publish_phase(e, s, fn, arg, caller_participates);
    if (caller_participates) fn(s, arg, 0, XE_WORKERS);
    xe_wait_followers(e, epoch);
}

static void xe_workers_end(xe_engine *e) {
    xe_require_owner(e);
    if (!e->token_active) xe_fatal("decode token is not active");
    uint64_t epoch = xe_publish_phase(e, NULL, NULL, NULL, 0);
    xe_wait_followers(e, epoch);
    e->token_active = 0;
}

static void *xe_workspace_take(xe_workspace_arena *a, size_t bytes) {
    a->pos = (a->pos + 63) & ~(size_t)63;
    void *p = a->base ? a->base + a->pos : NULL;
    a->pos += bytes;
    return p;
}

static void xe_workspace_q8(xe_workspace_arena *a, xe_q8 *q, int n) {
    q->qs = xe_workspace_take(a, (size_t)n * sizeof(*q->qs));
    q->d = xe_workspace_take(a, (size_t)(n / 32) * sizeof(*q->d));
    q->sigma = xe_workspace_take(a, (size_t)(n / 32) * sizeof(*q->sigma));
    q->n = n;
}

static _Float16 *xe_kv_layer_ptr(xe_session *s, int layer, int value);

static size_t xe_workspace_layout(xe_session *s, void *base) {
    xe_workspace_arena a = { base, 0 };
    s->hidden = xe_workspace_take(&a, XE_EMBD * sizeof(*s->hidden));
    s->q = xe_workspace_take(&a, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*s->q));
    s->k = xe_workspace_take(&a, (size_t)XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*s->k));
    s->v = xe_workspace_take(&a, (size_t)XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*s->v));
    s->attn_heads = xe_workspace_take(&a, (size_t)XE_Q_HEADS * XE_GLOBAL_HEAD_DIM * sizeof(*s->attn_heads));
    s->attn_proj = xe_workspace_take(&a, XE_EMBD * sizeof(*s->attn_proj));
    s->attn_out = xe_workspace_take(&a, XE_EMBD * sizeof(*s->attn_out));
    s->dense_out = xe_workspace_take(&a, XE_EMBD * sizeof(*s->dense_out));
    s->combined = xe_workspace_take(&a, XE_EMBD * sizeof(*s->combined));
    s->scores = xe_workspace_take(&a, (size_t)XE_Q_HEADS * s->engine->context * sizeof(*s->scores));
    s->rope_swa_cos = xe_workspace_take(&a, (XE_SWA_HEAD_DIM / 2) * sizeof(*s->rope_swa_cos));
    s->rope_swa_sin = xe_workspace_take(&a, (XE_SWA_HEAD_DIM / 2) * sizeof(*s->rope_swa_sin));
    s->rope_global_cos = xe_workspace_take(&a, (XE_GLOBAL_HEAD_DIM / 2) * sizeof(*s->rope_global_cos));
    s->rope_global_sin = xe_workspace_take(&a, (XE_GLOBAL_HEAD_DIM / 2) * sizeof(*s->rope_global_sin));
    s->logits = xe_workspace_take(&a, XE_VOCAB * sizeof(*s->logits));
    s->sample_candidates = xe_workspace_take(&a, XE_VOCAB * sizeof(*s->sample_candidates));
    xe_workspace_q8(&a, &s->q8_main, XE_PLE_WIDTH);
    s->normed = xe_workspace_take(&a, XE_PLE_WIDTH * sizeof(float));
    s->gate = xe_workspace_take(&a, XE_DENSE_FFN * sizeof(float));
    s->up = xe_workspace_take(&a, XE_DENSE_FFN * sizeof(float));
    s->activation = xe_workspace_take(&a, XE_DENSE_FFN * sizeof(float));
    s->ple_inputs = xe_workspace_take(&a, XE_PLE_WIDTH * sizeof(float));
    s->ple_lookup = xe_workspace_take(&a, XE_PLE_WIDTH * sizeof(float));
    s->ple_gate = xe_workspace_take(&a, XE_PLE * sizeof(float));
    return (a.pos + 63) & ~(size_t)63;
}

static float xe_hsum8_ps(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static int xe_hsum8_i32(__m256i x) {
    __m128i lo = _mm256_castsi256_si128(x);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static uint16_t xe_half_bits(_Float16 x);

static float xe_absmax32(const float *x) {
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 m0 = _mm256_and_ps(_mm256_loadu_ps(x), mask);
    __m256 m1 = _mm256_and_ps(_mm256_loadu_ps(x + 8), mask);
    __m256 m2 = _mm256_and_ps(_mm256_loadu_ps(x + 16), mask);
    __m256 m3 = _mm256_and_ps(_mm256_loadu_ps(x + 24), mask);
    __m256 m = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));
    __m128 lo = _mm256_castps256_ps128(m);
    __m128 hi = _mm256_extractf128_ps(m, 1);
    __m128 r = _mm_max_ps(lo, hi);
    r = _mm_max_ps(r, _mm_movehl_ps(r, r));
    r = _mm_max_ss(r, _mm_shuffle_ps(r, r, 1));
    return _mm_cvtss_f32(r);
}

static void xe_quantize_q8_blocks(const float *x, xe_q8 *q, int first, int end) {
    for (int b = first; b < end; b++) {
        const float *xb = x + 32 * b;
        int8_t *qb = q->qs + 32 * b;
        float amax = xe_absmax32(xb);
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        int sigma = 0;
        q->d[b] = (_Float16)d;
        for (int i = 0; i < 32; i++) {
            int v = (int)roundf(xb[i] * id);
            qb[i] = (int8_t)v;
            sigma += v;
        }
        q->sigma[b] = (int16_t)sigma;
    }
}

static void xe_quantize_q8_block_at(const float *x, xe_q8 *q, int b) {
    float amax = xe_absmax32(x);
    float d = amax / 127.0f;
    float id = d ? 1.0f / d : 0.0f;
    int sigma = 0;
    q->d[b] = (_Float16)d;
    for (int i = 0; i < 32; i++) {
        int v = (int)roundf(x[i] * id);
        q->qs[32 * b + i] = (int8_t)v;
        sigma += v;
    }
    q->sigma[b] = (int16_t)sigma;
}

static void xe_quantize_q8(const float *x, int n, xe_q8 *q) {
    if ((n & 31) || n > q->n) xe_fatal("q8: invalid length %d for capacity %d", n, q->n);
    xe_quantize_q8_blocks(x, q, 0, n / 32);
}

static inline int xe_q4_q8_raw_integer_q(const uint8_t *packed_weights, __m256i q) {
    __m128i packed = _mm_loadu_si128((const __m128i *)packed_weights);
    __m128i mask = _mm_set1_epi8(0x0f);
    __m128i low = _mm_and_si128(packed, mask);
    __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    __m256i nibbles = _mm256_set_m128i(high, low);
    __m256i dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), nibbles, q);
    return xe_hsum8_i32(dot);
}

static inline int xe_q4_q8_raw_integer(const uint8_t *packed_weights, const int8_t *activation) {
    return xe_q4_q8_raw_integer_q(packed_weights,
                                  _mm256_load_si256((const __m256i *)activation));
}

static inline uint16_t xe_q4_scale(const xe_q4 *w, uint64_t row, int blocks, int block) {
    if (!w->blocks) return w->d[row * (uint64_t)blocks + block];
    if (w->blocks != blocks) xe_fatal("q4: layout has %d blocks, matvec expects %d",
                                      w->blocks, blocks);
    uint64_t group = row >> 3;
    int lane = (int)(row & 7);
    return w->d[(group * (uint64_t)blocks + block) * 8 + lane];
}

static inline void xe_q4_copy_block(const xe_q4 *w, uint64_t row, int blocks,
                                    int block, uint8_t packed[16]) {
    if (!w->blocks) {
        memcpy(packed, w->qs + 16 * (row * (uint64_t)blocks + block), 16);
        return;
    }
    if (w->blocks != blocks) xe_fatal("q4: layout has %d blocks, matvec expects %d",
                                      w->blocks, blocks);
    uint64_t group = row >> 3;
    int lane = (int)(row & 7);
    const uint8_t *src = w->qs + (group * (uint64_t)blocks + block) * 128 + lane * 4;
    for (int chunk = 0; chunk < 4; chunk++)
        memcpy(packed + chunk * 4, src + chunk * 32, 4);
}

static float xe_q4_q8_dot_vector(const xe_q4 *w, uint64_t row, int n, const xe_q8 *q) {
    int blocks = n / 32;
    __m256 sum = _mm256_setzero_ps();
    int b = 0;

    for (; b + 7 < blocks; b += 8) {
        _Alignas(32) int integers[8];
        _Alignas(16) uint8_t packed[16];
        _Alignas(16) uint16_t wd_bits[8];
        for (int j = 0; j < 8; j++) {
            xe_q4_copy_block(w, row, blocks, b + j, packed);
            integers[j] = xe_q4_q8_raw_integer(packed, q->qs + 32 * (b + j));
            wd_bits[j] = xe_q4_scale(w, row, blocks, b + j);
        }
        __m256i vi = _mm256_load_si256((const __m256i *)integers);
        __m128i sigma16 = _mm_loadu_si128((const __m128i *)(q->sigma + b));
        __m256i correction = _mm256_slli_epi32(_mm256_cvtepi16_epi32(sigma16), 3);
        vi = _mm256_sub_epi32(vi, correction);
        __m256 wd = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)wd_bits));
        __m256 qd = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)(q->d + b)));
        __m256 scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(vi), _mm256_mul_ps(wd, qd));
        sum = _mm256_add_ps(sum, scaled);
    }

    float result = xe_hsum8_ps(sum);
    for (; b < blocks; b++) {
        _Alignas(16) uint8_t packed[16];
        xe_q4_copy_block(w, row, blocks, b, packed);
        int integer = xe_q4_q8_raw_integer(packed, q->qs + 32 * b) -
                      8 * (int)q->sigma[b];
        result += (float)integer * _cvtsh_ss(xe_q4_scale(w, row, blocks, b)) *
                  _cvtsh_ss(xe_half_bits(q->d[b]));
    }
    return result;
}

static float xe_q4_q8_dot(const xe_q4 *w, uint64_t row, int n, const xe_q8 *q) {
    return xe_q4_q8_dot_vector(w, row, n, q);
}

static void xe_q4_q8_dot4(const xe_q4 *w, uint64_t first_row, int n,
                          const xe_q8 *q, float *out) {
    for (int r = 0; r < 4; r++)
        out[r] = xe_q4_q8_dot_vector(w, first_row + (uint64_t)r, n, q);
}

static inline uint32_t xe_load_u32(const void *p) {
    uint32_t value;
    memcpy(&value, p, sizeof(value));
    return value;
}

static inline __attribute__((always_inline)) __m256 xe_q4_q8_block8(
        const uint8_t *packed, const uint16_t *wd_bits, const xe_q8 *q, int block) {
    const __m256i mask = _mm256_set1_epi8(0x0f);
    __m256i integer = _mm256_setzero_si256();
    for (int chunk = 0; chunk < 4; chunk++) {
        __m256i weights = _mm256_load_si256((const __m256i *)(packed + chunk * 32));
        __m256i low = _mm256_and_si256(weights, mask);
        __m256i high = _mm256_and_si256(_mm256_srli_epi16(weights, 4), mask);
        __m256i qlow = _mm256_set1_epi32((int)xe_load_u32(q->qs + block * 32 + chunk * 4));
        __m256i qhigh = _mm256_set1_epi32((int)xe_load_u32(q->qs + block * 32 + 16 + chunk * 4));
        integer = _mm256_dpbusd_epi32(integer, low, qlow);
        integer = _mm256_dpbusd_epi32(integer, high, qhigh);
    }
    integer = _mm256_sub_epi32(integer,
        _mm256_set1_epi32(8 * (int)q->sigma[block]));
    __m256 wd = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)wd_bits));
    __m256 qd = _mm256_set1_ps(_cvtsh_ss(xe_half_bits(q->d[block])));
    return _mm256_mul_ps(_mm256_cvtepi32_ps(integer), _mm256_mul_ps(wd, qd));
}

#define XE_DEFINE_Q4_Q8_DOT8(NAME, BLOCKS)                                      \
static __attribute__((noinline)) void NAME(const xe_q4 *w, uint64_t first_row,  \
                                            const xe_q8 *q, float out[8]) {      \
    const uint8_t *packed = w->qs + (first_row >> 3) * (uint64_t)(BLOCKS) * 128;\
    const uint16_t *wd = w->d + (first_row >> 3) * (uint64_t)(BLOCKS) * 8;       \
    __m256 s0 = _mm256_setzero_ps();                                             \
    __m256 s1 = _mm256_setzero_ps();                                             \
    __m256 s2 = _mm256_setzero_ps();                                             \
    __m256 s3 = _mm256_setzero_ps();                                             \
    __m256 s4 = _mm256_setzero_ps();                                             \
    __m256 s5 = _mm256_setzero_ps();                                             \
    __m256 s6 = _mm256_setzero_ps();                                             \
    __m256 s7 = _mm256_setzero_ps();                                             \
    int block = 0;                                                               \
    for (; block + 7 < (BLOCKS); block += 8) {                                   \
        s0 = _mm256_add_ps(s0, xe_q4_q8_block8(packed + (block + 0) * 128,       \
                                                wd + (block + 0) * 8, q, block + 0));\
        s1 = _mm256_add_ps(s1, xe_q4_q8_block8(packed + (block + 1) * 128,       \
                                                wd + (block + 1) * 8, q, block + 1));\
        s2 = _mm256_add_ps(s2, xe_q4_q8_block8(packed + (block + 2) * 128,       \
                                                wd + (block + 2) * 8, q, block + 2));\
        s3 = _mm256_add_ps(s3, xe_q4_q8_block8(packed + (block + 3) * 128,       \
                                                wd + (block + 3) * 8, q, block + 3));\
        s4 = _mm256_add_ps(s4, xe_q4_q8_block8(packed + (block + 4) * 128,       \
                                                wd + (block + 4) * 8, q, block + 4));\
        s5 = _mm256_add_ps(s5, xe_q4_q8_block8(packed + (block + 5) * 128,       \
                                                wd + (block + 5) * 8, q, block + 5));\
        s6 = _mm256_add_ps(s6, xe_q4_q8_block8(packed + (block + 6) * 128,       \
                                                wd + (block + 6) * 8, q, block + 6));\
        s7 = _mm256_add_ps(s7, xe_q4_q8_block8(packed + (block + 7) * 128,       \
                                                wd + (block + 7) * 8, q, block + 7));\
    }                                                                            \
    __m256 result = _mm256_add_ps(_mm256_add_ps(s0, s4), _mm256_add_ps(s1, s5));\
    result = _mm256_add_ps(result,                                                \
                           _mm256_add_ps(_mm256_add_ps(s2, s6),                  \
                                         _mm256_add_ps(s3, s7)));                \
    for (; block < (BLOCKS); block++)                                             \
        result = _mm256_add_ps(result, xe_q4_q8_block8(packed + block * 128,     \
                                                        wd + block * 8, q, block));\
    _mm256_storeu_ps(out, result);                                                \
}

XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b8, 8)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b64, 64)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b80, 80)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b128, 128)
XE_DEFINE_Q4_Q8_DOT8(xe_q4_q8_dot8_b320, 320)

#undef XE_DEFINE_Q4_Q8_DOT8

static inline void xe_q4_q8_dot8(const xe_q4 *w, uint64_t first_row, int n,
                                 const xe_q8 *q, float out[8]) {
    int blocks = n / 32;
    if (w->blocks == blocks && !(first_row & 7)) {
        switch (blocks) {
            case 8:  xe_q4_q8_dot8_b8(w, first_row, q, out); return;
            case 64:  xe_q4_q8_dot8_b64(w, first_row, q, out); return;
            case 80:  xe_q4_q8_dot8_b80(w, first_row, q, out); return;
            case 128: xe_q4_q8_dot8_b128(w, first_row, q, out); return;
            case 320: xe_q4_q8_dot8_b320(w, first_row, q, out); return;
        }
    }
    for (int row = 0; row < 8; row++)
        out[row] = xe_q4_q8_dot_vector(w, first_row + (uint64_t)row, n, q);
}

static void xe_matvec_q4_q8_rows4(const xe_q4 *w, uint64_t first_row, int n_in,
                                  int first_out, int end_out, const xe_q8 *q, float *y) {
    int row = first_out;
    while (row < end_out && ((first_row + (uint64_t)row) & 3)) {
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
        row++;
    }
    for (; row + 3 < end_out; row += 4)
        xe_q4_q8_dot4(w, first_row + (uint64_t)row, n_in, q, y + row);
    for (; row < end_out; row++)
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
}

static void xe_matvec_q4_q8_rows(const xe_q4 *w, uint64_t first_row, int n_in,
                                 int first_out, int end_out, const xe_q8 *q, float *y) {
    int row = first_out;
    while (row < end_out && ((first_row + (uint64_t)row) & 7)) {
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
        row++;
    }
    for (; row + 7 < end_out; row += 8)
        xe_q4_q8_dot8(w, first_row + (uint64_t)row, n_in, q, y + row);
    for (; row < end_out; row++)
        y[row] = xe_q4_q8_dot(w, first_row + (uint64_t)row, n_in, q);
}

static void xe_matvec_q4_q8_rows_vector(const xe_q4 *w, uint64_t first_row, int n_in,
                                        int first_out, int end_out, const xe_q8 *q, float *y) {
    for (int row = first_out; row < end_out; row++)
        y[row] = xe_q4_q8_dot_vector(w, first_row + (uint64_t)row, n_in, q);
}

static float xe_sumsq(const float *x, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);
        s0 = _mm256_fmadd_ps(x0, x0, s0);
        s1 = _mm256_fmadd_ps(x1, x1, s1);
        s2 = _mm256_fmadd_ps(x2, x2, s2);
        s3 = _mm256_fmadd_ps(x3, x3, s3);
    }
    float sum = xe_hsum8_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) sum += x[i] * x[i];
    return sum;
}

static float xe_rms_scale(const float *x, int n) {
    return 1.0f / sqrtf(xe_sumsq(x, n) / (float)n + XE_RMS_EPS);
}

static float xe_rms_scale_scalar(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return 1.0f / sqrtf(sum / (float)n + XE_RMS_EPS);
}

static float xe_rms_scale_engine(const xe_engine *e, const float *x, int n) {
    return e->scalar_rms ? xe_rms_scale_scalar(x, n) : xe_rms_scale(x, n);
}

static void xe_rmsnorm_scale(const float *x, const float *weight, int n, float scale, float *y) {
    int i = 0;
    __m256 vs = _mm256_set1_ps(scale);
    if (weight) {
        for (; i + 7 < n; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(vx, vs), vw));
        }
    } else {
        for (; i + 7 < n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    }
    for (; i < n; i++) y[i] = x[i] * scale * (weight ? weight[i] : 1.0f);
}

static void __attribute__((unused)) xe_rmsnorm(const float *x, const float *weight, int n, float *y) {
    xe_rmsnorm_scale(x, weight, n, xe_rms_scale(x, n), y);
}

static void xe_rmsnorm_engine(const xe_engine *e, const float *x, const float *weight,
                              int n, float *y) {
    xe_rmsnorm_scale(x, weight, n, xe_rms_scale_engine(e, x, n), y);
}

static void xe_rope_prepare(xe_session *s, int pos) {
    xe_engine *e = s->engine;
    for (int i = 0; i < XE_SWA_HEAD_DIM / 2; i++) {
        float theta = (float)pos * e->rope_swa_inv[i];
        s->rope_swa_cos[i] = cosf(theta);
        s->rope_swa_sin[i] = sinf(theta);
    }
    for (int i = 0; i < XE_GLOBAL_HEAD_DIM / 2; i++) {
        float theta = (float)pos * e->rope_global_inv[i];
        s->rope_global_cos[i] = cosf(theta);
        s->rope_global_sin[i] = sinf(theta);
    }
}

static void xe_rope_apply(float *heads, int n_heads, int head_dim,
                          const float *cosines, const float *sines) {
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float *head = heads + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float lo = head[i];
            float hi = head[i + half];
            head[i] = lo * cosines[i] - hi * sines[i];
            head[i + half] = lo * sines[i] + hi * cosines[i];
        }
    }
}

typedef struct {
    const xe_q4 *w;
    uint64_t first_row;
    int n_in;
    int n_out;
    const xe_q8 *q;
    float *y;
} xe_matvec_arg;

typedef struct {
    xe_matvec_arg op[3];
    int n_ops;
} xe_matvec_batch_arg;

static void xe_matvec_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    const xe_matvec_arg *a = opaque;
    int begin, end;
    if (!(a->n_out & 7) && !(a->first_row & 7)) {
        int groups = a->n_out / 8;
        begin = 8 * (groups * worker / workers);
        end = 8 * (groups * (worker + 1) / workers);
    } else {
        begin = a->n_out * worker / workers;
        end = a->n_out * (worker + 1) / workers;
    }
    xe_matvec_q4_q8_rows(a->w, a->first_row, a->n_in, begin, end, a->q, a->y);
}

static void xe_matvec_batch_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    const xe_matvec_batch_arg *a = opaque;
    int total = 0;
    for (int i = 0; i < a->n_ops; i++) total += a->op[i].n_out;
    int begin, end;
    if (!(total & 7)) {
        int groups = total / 8;
        begin = 8 * (groups * worker / workers);
        end = 8 * (groups * (worker + 1) / workers);
    } else {
        begin = total * worker / workers;
        end = total * (worker + 1) / workers;
    }
    int offset = 0;
    for (int i = 0; i < a->n_ops; i++) {
        const xe_matvec_arg *op = &a->op[i];
        int lo = begin > offset ? begin - offset : 0;
        int hi = end < offset + op->n_out ? end - offset : op->n_out;
        if (lo < hi)
            xe_matvec_q4_q8_rows(op->w, op->first_row, op->n_in, lo, hi, op->q, op->y);
        offset += op->n_out;
    }
}

typedef struct {
    const float *x;
    const float *weight;
    float scale;
    int n;
    xe_q8 *q;
} xe_norm_q8_arg;

static void xe_norm_q8_phase(xe_session *s, const void *opaque, int worker, int workers) {
    (void)s;
    const xe_norm_q8_arg *a = opaque;
    int blocks = a->n / 32;
    int begin = blocks * worker / workers;
    int end = blocks * (worker + 1) / workers;
    for (int b = begin; b < end; b++) {
        float values[32];
        for (int i = 0; i < 32; i++) {
            int j = 32 * b + i;
            values[i] = a->x[j] * a->scale * a->weight[j];
        }
        xe_quantize_q8_block_at(values, a->q, b);
    }
}






static uint16_t xe_half_bits(_Float16 x) {
    uint16_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}









static float xe_dot_f32_avx(const float *a, const float *b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    float sum = xe_hsum8_ps(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static void xe_softmax_f32(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
} xe_cur;

static void xe_need(xe_cur *c, size_t n) {
    if (n > c->size - c->pos)
        xe_fatal("truncated GGUF at byte %zu (need %zu bytes, %zu remain)", c->pos, n, c->size - c->pos);
}

static uint8_t xe_u8(xe_cur *c) {
    xe_need(c, 1);
    return c->base[c->pos++];
}

static uint32_t xe_u32(xe_cur *c) {
    xe_need(c, 4);
    uint32_t v;
    memcpy(&v, c->base + c->pos, 4);
    c->pos += 4;
    return v;
}

static uint64_t xe_u64(xe_cur *c) {
    xe_need(c, 8);
    uint64_t v;
    memcpy(&v, c->base + c->pos, 8);
    c->pos += 8;
    return v;
}

static float xe_f32(xe_cur *c) {
    xe_need(c, 4);
    float v;
    memcpy(&v, c->base + c->pos, 4);
    c->pos += 4;
    return v;
}

static xe_str xe_read_str(xe_cur *c) {
    uint64_t len = xe_u64(c);
    xe_need(c, len);
    xe_str s = { (const char *)(c->base + c->pos), len };
    c->pos += len;
    return s;
}

static int xe_str_eq(xe_str s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.p, lit, n) == 0;
}

static size_t xe_type_size(uint32_t t) {
    switch (t) {
        case XE_T_U8: case XE_T_I8: case XE_T_BOOL: return 1;
        case XE_T_U16: case XE_T_I16: return 2;
        case XE_T_U32: case XE_T_I32: case XE_T_F32: return 4;
        case XE_T_U64: case XE_T_I64: case XE_T_F64: return 8;
        default: xe_fatal("gguf: bad scalar value type %u", t);
    }
}

static void xe_skip_value(xe_cur *c, uint32_t type) {
    if (type == XE_T_STRING) { xe_read_str(c); return; }
    if (type == XE_T_ARRAY) {
        uint32_t etype = xe_u32(c);
        uint64_t n = xe_u64(c);
        if (etype == XE_T_STRING) {
            for (uint64_t i = 0; i < n; i++) xe_read_str(c);
        } else if (etype == XE_T_ARRAY) {
            xe_fatal("gguf: nested arrays not supported");
        } else {
            size_t sz = xe_type_size(etype);
            xe_need(c, sz * n);
            c->pos += sz * n;
        }
        return;
    }
    size_t sz = xe_type_size(type);
    xe_need(c, sz);
    c->pos += sz;
}

static void xe_capture_array(xe_cur *c, uint32_t vtype, const char *key, const void **ptr, uint64_t *count) {
    if (vtype != XE_T_ARRAY) xe_fatal("%s: expected array, found value type %u", key, vtype);
    uint32_t etype = xe_u32(c);
    uint64_t n = xe_u64(c);
    *ptr = c->base + c->pos;
    *count = n;
    if (etype == XE_T_STRING) {
        for (uint64_t i = 0; i < n; i++) xe_read_str(c);
    } else {
        size_t sz = xe_type_size(etype);
        xe_need(c, sz * n);
        c->pos += sz * n;
    }
}

static uint32_t xe_expect_u32(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_U32) xe_fatal("%s: expected u32, found value type %u", key, type);
    return xe_u32(c);
}

static float xe_expect_f32(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_F32) xe_fatal("%s: expected f32, found value type %u", key, type);
    return xe_f32(c);
}

static int xe_expect_bool(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_BOOL) xe_fatal("%s: expected bool, found value type %u", key, type);
    return xe_u8(c) != 0;
}

static xe_str xe_expect_string(xe_cur *c, uint32_t type, const char *key) {
    if (type != XE_T_STRING) xe_fatal("%s: expected string, found value type %u", key, type);
    return xe_read_str(c);
}

/* Identify the model before applying profile-specific metadata rules.  GGUF
 * metadata ordering is not significant, so this deliberately performs a
 * read-only first pass over the mapped header. */
static const xe_gemma4_model *xe_detect_model(const xe_engine *e, xe_cur metadata,
                                               const char *gguf_path) {
    uint32_t block_count = 0;
    uint32_t embedding_length = 0;
    int architecture_seen = 0;
    int block_count_seen = 0;
    int embedding_length_seen = 0;

    for (uint64_t i = 0; i < e->n_kv; i++) {
        xe_str key = xe_read_str(&metadata);
        uint32_t vtype = xe_u32(&metadata);
        if (xe_str_eq(key, "general.architecture")) {
            xe_str value = xe_expect_string(&metadata, vtype,
                                            "general.architecture");
            if (!xe_str_eq(value, "gemma4"))
                xe_fatal("%s: unsupported architecture %.*s", gguf_path,
                         (int)value.len, value.p);
            architecture_seen = 1;
            continue;
        }
        if (xe_str_eq(key, "gemma4.block_count")) {
            block_count = xe_expect_u32(&metadata, vtype,
                                        "gemma4.block_count");
            block_count_seen = 1;
            continue;
        }
        if (xe_str_eq(key, "gemma4.embedding_length")) {
            embedding_length = xe_expect_u32(&metadata, vtype,
                                              "gemma4.embedding_length");
            embedding_length_seen = 1;
            continue;
        }
        xe_skip_value(&metadata, vtype);
    }

    if (!architecture_seen || !block_count_seen || !embedding_length_seen)
        xe_fatal("%s: insufficient Gemma 4 metadata for model detection",
                 gguf_path);
    const xe_gemma4_model *model =
        xe_gemma4_model_for_metadata(block_count, embedding_length);
    if (!model)
        xe_fatal("%s: unsupported Gemma 4 layout (layers=%u, embedding=%u)",
                 gguf_path, block_count, embedding_length);
    return model;
}

static uint64_t xe_fnv1a(const char *p, uint32_t n);

static void xe_expect_array_header(xe_cur *c, uint32_t vtype, uint32_t want_etype, uint64_t want_n, const char *key) {
    if (vtype != XE_T_ARRAY) xe_fatal("%s: expected array, found value type %u", key, vtype);
    uint32_t etype = xe_u32(c);
    uint64_t n = xe_u64(c);
    if (etype != want_etype || n != want_n)
        xe_fatal("%s: expected array[%u] of length %llu, found type %u length %llu",
                 key, want_etype, (unsigned long long)want_n, etype, (unsigned long long)n);
}

enum {
    K_ARCH, K_BLOCK_COUNT, K_CTX_LEN, K_EMBD_LEN, K_FFN_LEN, K_HEAD_COUNT,
    K_SWA_WIN, K_KEY_LEN, K_VAL_LEN,
    K_KEY_LEN_SWA, K_VAL_LEN_SWA, K_ROPE_BASE, K_ROPE_BASE_SWA, K_SOFTCAP, K_RMS_EPS,
    K_HEAD_COUNT_KV, K_SWA_PATTERN, K_SHARED_KV, K_PLE_LEN, K_TOK_MODEL, K_BOS,
    K_EOS, K_EOT, K_UNK, K_PAD, K_ADD_SPACE, K_ADD_BOS, K_CHAT_TEMPLATE,
    K_REQUIRED_COUNT
};

static const char *xe_req_keys[K_REQUIRED_COUNT] = {
    "general.architecture",
    "gemma4.block_count",
    "gemma4.context_length",
    "gemma4.embedding_length",
    "gemma4.feed_forward_length",
    "gemma4.attention.head_count",
    "gemma4.attention.sliding_window",
    "gemma4.attention.key_length",
    "gemma4.attention.value_length",
    "gemma4.attention.key_length_swa",
    "gemma4.attention.value_length_swa",
    "gemma4.rope.freq_base",
    "gemma4.rope.freq_base_swa",
    "gemma4.final_logit_softcapping",
    "gemma4.attention.layer_norm_rms_epsilon",
    "gemma4.attention.head_count_kv",
    "gemma4.attention.sliding_window_pattern",
    "gemma4.attention.shared_kv_layers",
    "gemma4.embedding_length_per_layer_input",
    "tokenizer.ggml.model",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.chat_template",
};

#define XE_KEY(lit) xe_str_eq(key, lit)

#define XE_REQ_U32(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        uint32_t v = xe_expect_u32(c, vtype, lit); \
        if (v != (uint32_t)(expect)) xe_fatal(lit ": expected %u, found %u", (unsigned)(expect), v); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_F32(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        float v = xe_expect_f32(c, vtype, lit); \
        if (v != (float)(expect)) xe_fatal(lit ": expected %g, found %g", (double)(expect), (double)v); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_BOOL(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        int v = xe_expect_bool(c, vtype, lit); \
        if (v != (expect)) xe_fatal(lit ": expected %d, found %d", (int)(expect), v); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_STR(lit, kidx, expect) \
    if (XE_KEY(lit)) { \
        xe_str v = xe_expect_string(c, vtype, lit); \
        size_t n = strlen(expect); \
        if (v.len != n || memcmp(v.p, expect, n)) xe_fatal(lit ": expected \"%s\", found \"%.*s\"", expect, (int)v.len, v.p); \
        seen |= 1u << (kidx); \
        continue; \
    }

#define XE_REQ_U32_STORE(lit, kidx, expect, dest) \
    if (XE_KEY(lit)) { \
        uint32_t v = xe_expect_u32(c, vtype, lit); \
        if (v != (uint32_t)(expect)) xe_fatal(lit ": expected %u, found %u", (unsigned)(expect), v); \
        (dest) = (int32_t)v; \
        seen |= 1u << (kidx); \
        continue; \
    }

enum { XE_TK_SINGLETON, XE_TK_ALL, XE_TK_SWA, XE_TK_GLOBAL };
enum { XE_CAT_DENSE, XE_CAT_PLE, XE_CAT_EMBD };

typedef struct {
    const char *suffix;
    int kind;
    int qtype;
    uint64_t ne0, ne1, ne2;
    size_t off;
    int cat;
} xe_tspec;

static const xe_tspec xe_specs[] = {
#define S(n,t,a,b,f,cat) {n, XE_TK_SINGLETON,t,a,b,1,offsetof(struct xe_engine,f),cat}
#define L(n,t,a,b,f) {n,XE_TK_ALL,t,a,b,1,offsetof(xe_layer,f),XE_CAT_DENSE}
#define A(n,k,a,b,f) {n,k,XE_GGML_TYPE_Q4_0,a,b,1,offsetof(xe_layer,f),XE_CAT_DENSE}
#define N(n,k,d,f) {n,k,XE_GGML_TYPE_F32,d,1,1,offsetof(xe_layer,f),XE_CAT_DENSE}
    S("token_embd.weight",XE_GGML_TYPE_Q4_0,XE_EMBD,XE_VOCAB,tok_embd,XE_CAT_EMBD),
    S("output.weight",XE_GGML_TYPE_Q4_0,XE_EMBD,XE_VOCAB,output,XE_CAT_DENSE),
    S("output_norm.weight",XE_GGML_TYPE_F32,XE_EMBD,1,out_norm,XE_CAT_DENSE),
    S("rope_freqs.weight",XE_GGML_TYPE_F32,XE_GLOBAL_HEAD_DIM/2,1,rope_freqs,XE_CAT_DENSE),
    S("per_layer_token_embd.weight",XE_GGML_TYPE_Q4_0,XE_PLE_WIDTH,XE_VOCAB,per_layer_token_embd,XE_CAT_PLE),
    S("per_layer_model_proj.weight",XE_GGML_TYPE_Q4_0,XE_EMBD,XE_PLE_WIDTH,per_layer_model_proj,XE_CAT_DENSE),
    S("per_layer_proj_norm.weight",XE_GGML_TYPE_F32,XE_PLE,1,per_layer_proj_norm,XE_CAT_DENSE),
    L("attn_norm.weight",XE_GGML_TYPE_F32,XE_EMBD,1,attn_norm),
    L("post_attention_norm.weight",XE_GGML_TYPE_F32,XE_EMBD,1,post_attn_norm),
    L("ffn_norm.weight",XE_GGML_TYPE_F32,XE_EMBD,1,ffn_norm),
    L("post_ffw_norm.weight",XE_GGML_TYPE_F32,XE_EMBD,1,post_ffw_norm),
    L("layer_output_scale.weight",XE_GGML_TYPE_F32,1,1,layer_out_scale),
    L("ffn_gate.weight",XE_GGML_TYPE_Q4_0,XE_EMBD,XE_DENSE_FFN,ffn_gate),
    L("ffn_up.weight",XE_GGML_TYPE_Q4_0,XE_EMBD,XE_DENSE_FFN,ffn_up),
    L("ffn_down.weight",XE_GGML_TYPE_Q4_0,XE_DENSE_FFN,XE_EMBD,ffn_down),
    L("inp_gate.weight",XE_GGML_TYPE_Q4_0,XE_EMBD,XE_PLE,inp_gate),
    L("proj.weight",XE_GGML_TYPE_Q4_0,XE_PLE,XE_EMBD,proj),
    L("post_norm.weight",XE_GGML_TYPE_F32,XE_EMBD,1,post_norm),
    A("attn_q.weight",XE_TK_SWA,XE_EMBD,XE_Q_HEADS*XE_SWA_HEAD_DIM,attn_q),
    A("attn_k.weight",XE_TK_SWA,XE_EMBD,XE_SWA_KV_HEADS*XE_SWA_HEAD_DIM,attn_k),
    A("attn_v.weight",XE_TK_SWA,XE_EMBD,XE_SWA_KV_HEADS*XE_SWA_HEAD_DIM,attn_v),
    A("attn_output.weight",XE_TK_SWA,XE_Q_HEADS*XE_SWA_HEAD_DIM,XE_EMBD,attn_o),
    N("attn_q_norm.weight",XE_TK_SWA,XE_SWA_HEAD_DIM,q_norm),
    N("attn_k_norm.weight",XE_TK_SWA,XE_SWA_HEAD_DIM,k_norm),
    A("attn_q.weight",XE_TK_GLOBAL,XE_EMBD,XE_Q_HEADS*XE_GLOBAL_HEAD_DIM,attn_q),
    A("attn_k.weight",XE_TK_GLOBAL,XE_EMBD,XE_GLOBAL_KV_HEADS*XE_GLOBAL_HEAD_DIM,attn_k),
    A("attn_v.weight",XE_TK_GLOBAL,XE_EMBD,XE_GLOBAL_KV_HEADS*XE_GLOBAL_HEAD_DIM,attn_v),
    A("attn_output.weight",XE_TK_GLOBAL,XE_Q_HEADS*XE_GLOBAL_HEAD_DIM,XE_EMBD,attn_o),
    N("attn_q_norm.weight",XE_TK_GLOBAL,XE_GLOBAL_HEAD_DIM,q_norm),
    N("attn_k_norm.weight",XE_TK_GLOBAL,XE_GLOBAL_HEAD_DIM,k_norm),
#undef S
#undef L
#undef A
#undef N
};

#define XE_NSPECS (sizeof(xe_specs) / sizeof(xe_specs[0]))

static const xe_tspec *xe_find_spec(const char *suffix, int kind, int *suffix_known) {
    *suffix_known = 0;
    for (size_t i = 0; i < XE_NSPECS; i++) {
        const xe_tspec *s = &xe_specs[i];
        if (strcmp(s->suffix, suffix)) continue;
        *suffix_known = 1;
        if (s->kind == XE_TK_ALL || s->kind == kind) return s;
    }
    return NULL;
}

static uint64_t xe_align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

typedef struct {
    void *target;
    uint64_t offset;
    uint64_t size;
} xe_bind_rec;

typedef struct {
    xe_q4 *field;
    uint64_t nblocks;
    int blocks;
    uint64_t rows;
    const char *suffix;
    int layer_idx;
    uint8_t *qs_seg;
    uint16_t *d_seg;
    int part;
    uint64_t part_offset;
    uint64_t source_offset;
} xe_repack_job;

static int xe_repack_source_order(const void *a, const void *b) {
    const xe_repack_job *x = a, *y = b;
    return (x->source_offset > y->source_offset) -
           (x->source_offset < y->source_offset);
}

static void xe_repack_release(xe_engine *e, uint64_t *released,
                              uint64_t consumed, const char *gguf_path) {
    uint64_t end = consumed & ~UINT64_C(4095);
    if (end <= *released) return;
    size_t bytes = (size_t)(end - *released);
    if (madvise((uint8_t *)e->map + *released, bytes, MADV_DONTNEED) != 0)
        xe_fatal("%s: madvise consumed source: %s", gguf_path, strerror(errno));
    int error = posix_fadvise(e->map_fd, (off_t)*released, (off_t)bytes,
                             POSIX_FADV_DONTNEED);
    if (error)
        xe_fatal("%s: fadvise consumed source: %s", gguf_path, strerror(error));
    *released = end;
}

typedef struct {
    xe_repack_job *jobs;
    int njobs;
    atomic_int *next;
    const char *gguf_path;
    uint64_t verified;
} xe_repack_worker_arg;

static void *xe_repack_worker(void *arg_) {
    xe_repack_worker_arg *arg = arg_;
    for (;;) {
        int i = atomic_fetch_add(arg->next, 1);
        if (i >= arg->njobs) break;
        xe_repack_job *j = &arg->jobs[i];

        const uint8_t *src = j->field->qs;
        if (j->rows & 7) xe_fatal("%s: q4 repack rows not divisible by eight: %s",
                                  arg->gguf_path, j->suffix);
        for (uint64_t group = 0; group < j->rows / 8; group++) {
            for (int block = 0; block < j->blocks; block++) {
                uint8_t *dst_qs = j->qs_seg + (group * j->blocks + block) * 128;
                uint16_t *dst_d = j->d_seg + (group * j->blocks + block) * 8;
                for (int row = 0; row < 8; row++) {
                    uint64_t native = (group * 8 + row) * j->blocks + block;
                    const uint8_t *blk = src + (size_t)native * 18;
                    for (int chunk = 0; chunk < 4; chunk++)
                        memcpy(dst_qs + chunk * 32 + row * 4, blk + 2 + chunk * 4, 4);
                    memcpy(&dst_d[row], blk, 2);
                }
            }
        }

        j->field->qs = j->qs_seg;
        j->field->d = j->d_seg;
        j->field->blocks = j->blocks;

        uint64_t cand[3] = { 0, j->nblocks / 2, j->nblocks - 1 };
        uint64_t uniq[3];
        int nuniq = 0;
        for (int k = 0; k < 3; k++) {
            int dup = 0;
            for (int m = 0; m < nuniq; m++) if (uniq[m] == cand[k]) dup = 1;
            if (!dup) uniq[nuniq++] = cand[k];
        }
#ifdef XE_REPACK_VERIFY_ALL
        uint64_t verify_count = j->nblocks;
#else
        uint64_t verify_count = (uint64_t)nuniq;
#endif
        for (uint64_t k = 0; k < verify_count; k++) {
#ifdef XE_REPACK_VERIFY_ALL
            uint64_t b = k;
#else
            uint64_t b = uniq[k];
#endif
            uint64_t row = b / j->blocks;
            int block = (int)(b % j->blocks);
            uint8_t recon[18];
            memcpy(recon, &j->d_seg[((row >> 3) * j->blocks + block) * 8 + (row & 7)], 2);
            const uint8_t *packed = j->qs_seg + ((row >> 3) * j->blocks + block) * 128 +
                                    (row & 7) * 4;
            for (int chunk = 0; chunk < 4; chunk++)
                memcpy(recon + 2 + chunk * 4, packed + chunk * 32, 4);
            if (memcmp(recon, src + (size_t)b * 18, 18) != 0) {
                if (j->layer_idx < 0)
                    xe_fatal("%s: repack verify failed: %s block %llu", arg->gguf_path, j->suffix, (unsigned long long)b);
                else
                    xe_fatal("%s: repack verify failed: blk.%d.%s block %llu", arg->gguf_path, j->layer_idx, j->suffix, (unsigned long long)b);
            }
            arg->verified++;
        }
    }
    return NULL;
}

static void xe_copy_f32(xe_engine *e) {
    uint64_t total=0;
    for(size_t i=0;i<XE_NSPECS;i++) {
        const xe_tspec *t=&xe_specs[i]; if(t->qtype!=XE_GGML_TYPE_F32) continue;
        int count=t->kind==XE_TK_SINGLETON?1:XE_LAYERS;
        total+=xe_align_up(t->ne0*t->ne1*t->ne2*4,64)*count;
    }
    if(total>XE_REPACK_PART_LIMIT) xe_fatal("F32 slab exceeds allocation limit");
    e->f32_slab=xe_alloc(e,total,XE_MEM_SHARED); e->f32_slab_size=total;
    uint8_t *cursor=e->f32_slab;
    for(size_t i=0;i<XE_NSPECS;i++) {
        const xe_tspec *t=&xe_specs[i]; if(t->qtype!=XE_GGML_TYPE_F32) continue;
        for(int l=t->kind==XE_TK_SINGLETON?-1:0;l<XE_LAYERS;l++) {
            if(l>=0 && ((t->kind==XE_TK_SWA && XE_IS_GLOBAL(l)) ||
                        (t->kind==XE_TK_GLOBAL && !XE_IS_GLOBAL(l)))) continue;
            char *base=l<0?(char*)e:(char*)&e->layers[l];
            const void *src; memcpy(&src,base+t->off,sizeof src);
            if(src) {size_t bytes=t->ne0*t->ne1*t->ne2*4; memcpy(cursor,src,bytes);
                const void *dst=cursor; memcpy(base+t->off,&dst,sizeof dst); cursor+=xe_align_up(bytes,64);}
            if(l<0) break;
        }
    }
}


static xe_engine *xe_map_file(const char *gguf_path) {
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) xe_fatal("%s: %s", gguf_path, strerror(errno));

    struct stat st;
    if (fstat(fd, &st) < 0) xe_fatal("%s: fstat: %s", gguf_path, strerror(errno));
    if (st.st_size <= 0) xe_fatal("%s: empty file", gguf_path);

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) xe_fatal("%s: mmap: %s", gguf_path, strerror(errno));
    xe_engine *e = calloc(1, sizeof *e);
    if (!e) xe_fatal("out of memory allocating engine");
    e->map = map;
    e->map_len = (size_t)st.st_size;
    e->map_fd = fd;
    e->bos_id = e->eos_id = e->eot_id = e->unk_id = e->pad_id = -1;

    return e;
}

static xe_cur xe_parse_header(xe_engine *e, const char *gguf_path) {
    xe_cur c = { (const uint8_t *)e->map, e->map_len, 0 };

    uint32_t magic = xe_u32(&c);
    if (magic != XE_GGUF_MAGIC) xe_fatal("%s: bad GGUF magic 0x%08x", gguf_path, magic);

    uint32_t version = xe_u32(&c);
    if (version != XE_GGUF_VERSION) xe_fatal("%s: unsupported GGUF version %u (expected %u)", gguf_path, version, (unsigned)XE_GGUF_VERSION);
    e->gguf_version = version;

    e->n_tensors = xe_u64(&c);

    e->n_kv = xe_u64(&c);

    return c;
}

static void xe_parse_metadata(xe_engine *e, xe_cur *c, const char *gguf_path) {
    uint32_t seen = 0;
    uint64_t alignment = 32;
    const xe_gemma4_model *model = e->model;

    if (!model) xe_fatal("%s: model profile was not detected", gguf_path);
    if (model->shared_kv_layers == 0) seen |= 1u << K_SHARED_KV;
    if (model->per_layer_embedding_length == 0) seen |= 1u << K_PLE_LEN;
    if (!model->requires_eot_metadata) seen |= 1u << K_EOT;

    for (uint64_t i = 0; i < e->n_kv; i++) {
        xe_str key = xe_read_str(c);
        uint32_t vtype = xe_u32(c);

        XE_REQ_STR("general.architecture", K_ARCH, "gemma4")
        XE_REQ_U32("gemma4.block_count", K_BLOCK_COUNT, model->block_count)
        XE_REQ_U32("gemma4.context_length", K_CTX_LEN, model->context_length)
        XE_REQ_U32("gemma4.embedding_length", K_EMBD_LEN, model->embedding_length)
        XE_REQ_U32("gemma4.feed_forward_length", K_FFN_LEN, model->feed_forward_length)
        XE_REQ_U32("gemma4.attention.head_count", K_HEAD_COUNT, model->attention_head_count)
        XE_REQ_U32("gemma4.attention.sliding_window", K_SWA_WIN, model->sliding_window)
        XE_REQ_U32("gemma4.attention.key_length", K_KEY_LEN, model->global_head_dim)
        XE_REQ_U32("gemma4.attention.value_length", K_VAL_LEN, model->global_head_dim)
        XE_REQ_U32("gemma4.attention.key_length_swa", K_KEY_LEN_SWA, model->swa_head_dim)
        XE_REQ_U32("gemma4.attention.value_length_swa", K_VAL_LEN_SWA, model->swa_head_dim)
        XE_REQ_F32("gemma4.rope.freq_base", K_ROPE_BASE, model->global_rope_base)
        XE_REQ_F32("gemma4.rope.freq_base_swa", K_ROPE_BASE_SWA, model->swa_rope_base)
        XE_REQ_F32("gemma4.final_logit_softcapping", K_SOFTCAP, model->final_logit_softcap)
        XE_REQ_U32("gemma4.attention.shared_kv_layers", K_SHARED_KV, model->shared_kv_layers)
        XE_REQ_U32("gemma4.embedding_length_per_layer_input", K_PLE_LEN, model->per_layer_embedding_length)
        XE_REQ_STR("tokenizer.ggml.model", K_TOK_MODEL, "gemma4")
        XE_REQ_U32_STORE("tokenizer.ggml.bos_token_id", K_BOS, 2, e->bos_id)
        XE_REQ_U32_STORE("tokenizer.ggml.eos_token_id", K_EOS, 1, e->eos_id)
        XE_REQ_U32_STORE("tokenizer.ggml.eot_token_id", K_EOT, 106, e->eot_id)
        XE_REQ_U32_STORE("tokenizer.ggml.unknown_token_id", K_UNK, 3, e->unk_id)
        XE_REQ_U32_STORE("tokenizer.ggml.padding_token_id", K_PAD, 0, e->pad_id)
        XE_REQ_BOOL("tokenizer.ggml.add_space_prefix", K_ADD_SPACE, 0)
        XE_REQ_BOOL("tokenizer.ggml.add_bos_token", K_ADD_BOS, model->add_bos_token)

        if (XE_KEY("tokenizer.chat_template")) {
            xe_str v = xe_expect_string(c, vtype, "tokenizer.chat_template");
            if (v.len == 0 ||
                (model->chat_template_hash &&
                 (v.len != model->chat_template_length ||
                  xe_fnv1a(v.p, (uint32_t)v.len) != model->chat_template_hash)))
                xe_fatal("tokenizer.chat_template: unsupported Gemma 4 template");
            e->chat_template = v;
            seen |= 1u << K_CHAT_TEMPLATE;
            continue;
        }

        if (XE_KEY("gemma4.attention.layer_norm_rms_epsilon")) {
            float v = xe_expect_f32(c, vtype, "gemma4.attention.layer_norm_rms_epsilon");
            if (fabsf(v - model->rms_epsilon) >= 1e-9f)
                xe_fatal("gemma4.attention.layer_norm_rms_epsilon: expected ~%g, found %g", (double)model->rms_epsilon, (double)v);
            seen |= 1u << K_RMS_EPS;
            continue;
        }

        if (XE_KEY("gemma4.attention.head_count_kv")) {
            if (vtype == XE_T_U32) {
                uint32_t v = xe_u32(c);
                if (model->swa_kv_head_count != model->global_kv_head_count ||
                    v != model->swa_kv_head_count)
                    xe_fatal("gemma4.attention.head_count_kv: scalar %u is not valid for %s",
                             v, model->name);
            } else {
                xe_expect_array_header(c, vtype, XE_T_I32, model->block_count,
                                       "gemma4.attention.head_count_kv");
                for (uint32_t j = 0; j < model->block_count; j++) {
                    int32_t v = (int32_t)xe_u32(c);
                    int32_t want = (int32_t)xe_gemma4_kv_head_count(model, j);
                    if (v != want) xe_fatal("gemma4.attention.head_count_kv[%u]: expected %d, found %d", j, want, v);
                }
            }
            seen |= 1u << K_HEAD_COUNT_KV;
            continue;
        }

        if (XE_KEY("gemma4.attention.sliding_window_pattern")) {
            xe_expect_array_header(c, vtype, XE_T_BOOL, model->block_count, "gemma4.attention.sliding_window_pattern");
            for (uint32_t j = 0; j < model->block_count; j++) {
                uint8_t v = xe_u8(c);
                int want = !xe_gemma4_is_global(model, j);
                if ((v != 0) != want) xe_fatal("gemma4.attention.sliding_window_pattern[%u]: expected %d, found %u", j, want, v);
            }
            seen |= 1u << K_SWA_PATTERN;
            continue;
        }

        if (XE_KEY("general.alignment")) {
            alignment = xe_expect_u32(c, vtype, "general.alignment");
            if (alignment == 0 || (alignment & (alignment - 1)))
                xe_fatal("general.alignment: bad value %llu", (unsigned long long)alignment);
            continue;
        }

        if (XE_KEY("general.name")) {
            xe_str v = xe_expect_string(c, vtype, "general.name");
            e->name = malloc(v.len + 1);
            if (!e->name) xe_fatal("out of memory copying general.name");
            memcpy(e->name, v.p, v.len);
            e->name[v.len] = 0;
            continue;
        }

        if (XE_KEY("tokenizer.ggml.tokens")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.tokens", &e->tok_tokens, &e->tok_tokens_count);
            continue;
        }
        if (XE_KEY("tokenizer.ggml.scores")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.scores", &e->tok_scores, &e->tok_scores_count);
            continue;
        }
        if (XE_KEY("tokenizer.ggml.token_type")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.token_type", &e->tok_token_type, &e->tok_token_type_count);
            continue;
        }
        if (XE_KEY("tokenizer.ggml.merges")) {
            xe_capture_array(c, vtype, "tokenizer.ggml.merges", &e->tok_merges, &e->tok_merges_count);
            continue;
        }

        xe_skip_value(c, vtype);
    }

    for (int i = 0; i < K_REQUIRED_COUNT; i++)
        if (!(seen & (1u << i))) xe_fatal("%s: missing required key %s", gguf_path, xe_req_keys[i]);

    e->alignment = alignment;
}

static uint64_t xe_fnv1a(const char *p, uint32_t n) {
    uint64_t h = 14695981039346656037ull;
    for (uint32_t i = 0; i < n; i++) {
        h ^= (uint8_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t xe_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static int32_t xe_vocab_find(const xe_engine *e, const char *p, uint32_t n) {
    uint32_t i = (uint32_t)xe_fnv1a(p, n) & (XE_TOK_HASH - 1);
    for (;;) {
        int32_t id = e->tok_hash[i];
        if (id < 0) return -1;
        const xe_str *s = &e->tok_piece[id];
        if (s->len == n && memcmp(s->p, p, n) == 0) return id;
        i = (i + 1) & (XE_TOK_HASH - 1);
    }
}

static int xe_merge_find(const xe_engine *e, int32_t l, int32_t r, uint32_t *rank, int32_t *res) {
    uint64_t key = ((uint64_t)(uint32_t)l << 32) | (uint32_t)r;
    uint32_t i = (uint32_t)xe_mix64(key) & (XE_MERGE_HASH - 1);
    for (;;) {
        const xe_merge *m = &e->tok_merge[i];
        if (m->res < 0) return 0;
        if (m->key == key) {
            *rank = m->rank;
            *res = m->res;
            return 1;
        }
        i = (i + 1) & (XE_MERGE_HASH - 1);
    }
}

static xe_cur xe_cur_at(const xe_engine *e, const void *p) {
    xe_cur c = { (const uint8_t *)e->map, e->map_len, (size_t)((const uint8_t *)p - (const uint8_t *)e->map) };
    return c;
}

static void xe_tok_build(xe_engine *e, const char *gguf_path) {
    if (!e->tok_tokens) xe_fatal("%s: missing required key tokenizer.ggml.tokens", gguf_path);
    if (!e->tok_scores) xe_fatal("%s: missing required key tokenizer.ggml.scores", gguf_path);
    if (!e->tok_token_type) xe_fatal("%s: missing required key tokenizer.ggml.token_type", gguf_path);
    if (!e->tok_merges) xe_fatal("%s: missing required key tokenizer.ggml.merges", gguf_path);

    if (e->tok_tokens_count != XE_VOCAB)
        xe_fatal("tokenizer.ggml.tokens: expected %d entries, found %llu", XE_VOCAB, (unsigned long long)e->tok_tokens_count);
    if (e->tok_scores_count != XE_VOCAB)
        xe_fatal("tokenizer.ggml.scores: expected %d entries, found %llu", XE_VOCAB, (unsigned long long)e->tok_scores_count);
    if (e->tok_token_type_count != XE_VOCAB)
        xe_fatal("tokenizer.ggml.token_type: expected %d entries, found %llu", XE_VOCAB, (unsigned long long)e->tok_token_type_count);
    if (e->tok_merges_count != XE_MERGES)
        xe_fatal("tokenizer.ggml.merges: expected %d entries, found %llu", XE_MERGES, (unsigned long long)e->tok_merges_count);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    e->tok_piece = xe_alloc(e, XE_VOCAB * sizeof *e->tok_piece, XE_MEM_HOST);
    e->tok_hash = xe_alloc(e, XE_TOK_HASH * sizeof *e->tok_hash, XE_MEM_HOST);
    e->tok_merge = xe_alloc(e, XE_MERGE_HASH * sizeof *e->tok_merge, XE_MEM_HOST);

    for (uint32_t i = 0; i < XE_TOK_HASH; i++) e->tok_hash[i] = -1;
    for (uint32_t i = 0; i < XE_MERGE_HASH; i++) e->tok_merge[i].res = -1;

    xe_cur c = xe_cur_at(e, e->tok_tokens);
    uint32_t max_piece = 0;
    for (int32_t id = 0; id < XE_VOCAB; id++) {
        xe_str s = xe_read_str(&c);
        if (s.len > 0xffffffffull) xe_fatal("tokenizer.ggml.tokens: piece %d too long", id);
        e->tok_piece[id] = s;
        if ((uint32_t)s.len > max_piece) max_piece = (uint32_t)s.len;
    }
    e->tok_max_piece = max_piece;

    for (int32_t id = 0; id < XE_VOCAB; id++) {
        const xe_str *s = &e->tok_piece[id];
        uint32_t i = (uint32_t)xe_fnv1a(s->p, (uint32_t)s->len) & (XE_TOK_HASH - 1);
        for (;;) {
            int32_t cur = e->tok_hash[i];
            if (cur < 0) break;
            const xe_str *o = &e->tok_piece[cur];
            if (o->len == s->len && memcmp(o->p, s->p, s->len) == 0)
                xe_fatal("%s: tokenizer.ggml.tokens: ids %d and %d share the piece \"%.*s\"",
                         gguf_path, cur, id, (int)s->len, s->p);
            i = (i + 1) & (XE_TOK_HASH - 1);
        }
        e->tok_hash[i] = id;
    }

    static const char xe_hex[] = "0123456789ABCDEF";
    for (int b = 0; b < 256; b++) {
        char lit[6] = { '<', '0', 'x', xe_hex[b >> 4], xe_hex[b & 15], '>' };
        int32_t id = xe_vocab_find(e, lit, 6);
        if (id != XE_BYTE0_ID + b)
            xe_fatal("%s: tokenizer.ggml.tokens: expected \"%.*s\" at id %d, found %d",
                     gguf_path, 6, lit, XE_BYTE0_ID + b, id);
    }

    static const struct { const char *piece; int32_t id; } controls[] = {
        { "<|tool>", 46 },
        { "<tool|>", 47 },
        { "<|tool_call>", 48 },
        { "<tool_call|>", 49 },
        { "<|tool_response>", 50 },
        { "<tool_response|>", 51 },
        { "<|\"|>", 52 },
        { "<|think|>", 98 },
        { "<|channel>", XE_CHANNEL_BEGIN_ID },
        { "<channel|>", XE_CHANNEL_END_ID },
        { "<|turn>", XE_TURN_BEGIN_ID },
        { "<turn|>", XE_TURN_END_ID },
    };
    for (size_t i = 0; i < sizeof controls / sizeof controls[0]; i++) {
        int32_t id = xe_vocab_find(e, controls[i].piece, (uint32_t)strlen(controls[i].piece));
        if (id != controls[i].id)
            xe_fatal("%s: tokenizer.ggml.tokens: expected \"%s\" at id %d, found %d",
                     gguf_path, controls[i].piece, controls[i].id, id);
    }

    char *concat = xe_alloc(e, max_piece + 1, XE_MEM_HOST);
    xe_cur mc = xe_cur_at(e, e->tok_merges);
    for (uint32_t rank = 0; rank < XE_MERGES; rank++) {
        xe_str s = xe_read_str(&mc);
        uint64_t sp = 0;
        for (uint64_t k = 1; k < s.len; k++)
            if (s.p[k] == ' ') { sp = k; break; }
        if (sp == 0)
            xe_fatal("%s: tokenizer.ggml.merges[%u]: no separating space in \"%.*s\"",
                     gguf_path, rank, (int)s.len, s.p);

        uint32_t lnn = (uint32_t)sp, rnn = (uint32_t)(s.len - sp - 1);
        int32_t lid = xe_vocab_find(e, s.p, lnn);
        int32_t rid = xe_vocab_find(e, s.p + sp + 1, rnn);
        int32_t cid = -1;
        if (lnn + rnn <= max_piece) {
            memcpy(concat, s.p, lnn);
            memcpy(concat + lnn, s.p + sp + 1, rnn);
            cid = xe_vocab_find(e, concat, lnn + rnn);
        }
        if (lid < 0 || rid < 0 || cid < 0)
            xe_fatal("%s: tokenizer.ggml.merges[%u]: \"%.*s\" is not closed over the vocabulary (left %d right %d result %d)",
                     gguf_path, rank, (int)s.len, s.p, lid, rid, cid);

        uint64_t key = ((uint64_t)(uint32_t)lid << 32) | (uint32_t)rid;
        uint32_t i = (uint32_t)xe_mix64(key) & (XE_MERGE_HASH - 1);
        while (e->tok_merge[i].res >= 0) {
            if (e->tok_merge[i].key == key) break;
            i = (i + 1) & (XE_MERGE_HASH - 1);
        }
        if (e->tok_merge[i].res < 0)
            e->tok_merge[i] = (xe_merge){ key, rank, cid };
    }
    xe_free(e, concat, XE_MEM_HOST);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    e->tok_build_seconds = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
}

static void xe_bind_tensors(xe_engine *e, xe_cur *c, const char *gguf_path) {
    static xe_bind_rec recs[XE_TENSOR_COUNT];

    for (uint64_t i = 0; i < e->n_tensors; i++) {
        xe_str name = xe_read_str(c);
        char namebuf[192];
        if (name.len >= sizeof namebuf) xe_fatal("%s: tensor name too long (%llu bytes)", gguf_path, (unsigned long long)name.len);
        memcpy(namebuf, name.p, name.len);
        namebuf[name.len] = 0;

        uint32_t n_dims = xe_u32(c);
        if (n_dims < 1 || n_dims > 3) xe_fatal("tensor %s: unsupported n_dims %u", namebuf, n_dims);
        uint64_t ne[3] = { 1, 1, 1 };
        for (uint32_t d = 0; d < n_dims; d++) ne[d] = xe_u64(c);
        uint32_t ttype = xe_u32(c);
        uint64_t offset = xe_u64(c);

        int kind, layer_idx = -1;
        const char *suffix;
        if (!strncmp(namebuf, "blk.", 4)) {
            char *end;
            long idx = strtol(namebuf + 4, &end, 10);
            if (end == namebuf + 4 || *end != '.' || idx < 0 || idx >= XE_LAYERS)
                xe_fatal("tensor %s: bad blk index", namebuf);
            layer_idx = (int)idx;
            suffix = end + 1;
            kind = XE_IS_GLOBAL(layer_idx) ? XE_TK_GLOBAL : XE_TK_SWA;
        } else {
            suffix = namebuf;
            kind = XE_TK_SINGLETON;
        }

        int suffix_known;
        const xe_tspec *spec = xe_find_spec(suffix, kind, &suffix_known);
        if (!spec) {
            if (suffix_known) xe_fatal("tensor %s: not valid for this layer's kind", namebuf);
            xe_fatal("tensor %s: unknown tensor name", namebuf);
        }

        if (ttype != (uint32_t)spec->qtype)
            xe_fatal("tensor %s: expected type %d, found %u", namebuf, spec->qtype, ttype);
        if (ne[0] != spec->ne0 || ne[1] != spec->ne1 || ne[2] != spec->ne2)
            xe_fatal("tensor %s: expected shape [%llu,%llu,%llu], found [%llu,%llu,%llu]",
                     namebuf,
                     (unsigned long long)spec->ne0, (unsigned long long)spec->ne1, (unsigned long long)spec->ne2,
                     (unsigned long long)ne[0], (unsigned long long)ne[1], (unsigned long long)ne[2]);

        uint64_t size;
        if (spec->qtype == XE_GGML_TYPE_Q4_0) {
            if (ne[0] % 32 != 0) xe_fatal("tensor %s: ne0 %llu not a multiple of 32", namebuf, (unsigned long long)ne[0]);
            size = (ne[0] / 32) * 18 * ne[1] * ne[2];
        } else {
            size = 4ull * ne[0] * ne[1] * ne[2];
        }

        if (offset % e->alignment != 0) xe_fatal("tensor %s: offset %llu not aligned to %llu", namebuf, (unsigned long long)offset, (unsigned long long)e->alignment);

        void *target = layer_idx >= 0 ? (void *)((char *)&e->layers[layer_idx] + spec->off) : (void *)((char *)e + spec->off);
        const void *existing;
        memcpy(&existing, target, sizeof existing);
        if (existing) xe_fatal("tensor %s: duplicate", namebuf);
        const void *claimed = (const void *)1;
        memcpy(target, &claimed, sizeof claimed);

        recs[i].target = target;
        recs[i].offset = offset;
        recs[i].size = size;

        if (spec->cat == XE_CAT_PLE) e->ple_bytes += size;
        else if (spec->cat == XE_CAT_EMBD) e->tok_embd_bytes += size;
        else e->dense_bytes += size;

        if (ttype == XE_GGML_TYPE_Q4_0) { e->q4_0_count++; e->q4_0_bytes += size; }
        else { e->f32_count++; e->f32_bytes += size; }
    }

    uint64_t data_off = xe_align_up((uint64_t)c->pos, e->alignment);
    if (data_off > e->map_len) xe_fatal("%s: data section start %llu beyond file size %zu", gguf_path, (unsigned long long)data_off, e->map_len);
    const uint8_t *data_base = (const uint8_t *)e->map + data_off;
    e->data = data_base;
    e->header_len = (size_t)data_off;

    uint64_t avail = e->map_len - data_off;
    for (uint64_t i = 0; i < e->n_tensors; i++) {
        xe_bind_rec *r = &recs[i];
        if (r->offset > avail || r->size > avail - r->offset)
            xe_fatal("%s: tensor #%llu offset %llu size %llu exceeds file bounds", gguf_path, (unsigned long long)i, (unsigned long long)r->offset, (unsigned long long)r->size);
        const void *ptr = data_base + r->offset;
        memcpy(r->target, &ptr, sizeof ptr);
    }
}

static void xe_check_complete(xe_engine *e, const char *path) {
    if (!e->rope_freqs) xe_fatal("%s: missing rope_freqs.weight",path);
    int active=(int)(XE_GLOBAL_HEAD_DIM/2*e->model->partial_rotary_factor);
    for(int i=0;i<XE_GLOBAL_HEAD_DIM/2;i++) {
        float factor=e->rope_freqs[i];
        if(!isfinite(factor) || (i<active ? fabsf(factor-1.0f)>1e-6f : factor<1e20f))
            xe_fatal("%s: unexpected E4B proportional RoPE factor at %d",path,i);
    }
    /* Tail K/V/norm tensors may be exported or omitted; they are never executed.
     * output.weight is optional because the base model ties its output head. */
    for (size_t i=0;i<XE_NSPECS;i++) {
        const xe_tspec *t=&xe_specs[i];
        for(int layer=t->kind==XE_TK_SINGLETON ? -1:0;layer<XE_LAYERS;layer++) {
            if(layer<0 && !strcmp(t->suffix,"output.weight")) break;
            if(layer>=0) {
                if(t->kind==XE_TK_SWA && XE_IS_GLOBAL(layer)) continue;
                if(t->kind==XE_TK_GLOBAL && !XE_IS_GLOBAL(layer)) continue;
                if(!strcmp(t->suffix,"layer_output_scale.weight")) continue;
                if(layer>=XE_KV_LAYERS && (!strcmp(t->suffix,"attn_k.weight") ||
                   !strcmp(t->suffix,"attn_v.weight") || !strcmp(t->suffix,"attn_k_norm.weight"))) continue;
            }
            const void *ptr;
            const char *base=layer<0?(const char*)e:(const char*)&e->layers[layer];
            memcpy(&ptr,base+t->off,sizeof ptr);
            if(!ptr) xe_fatal("%s: missing %s (layer %d)",path,t->suffix,layer);
            if(layer<0) break;
        }
    }
}


/* Q4_0 blocks are 18 bytes total, 2 of fp16 scale and 16 of weight nibbles.
 * We split scales from nibbles because the 18 byte stride defeats coalesced loads.
 * We measured approximately 3.1x on GPU and 1.6x on CPU. */
static void xe_repack(xe_engine *e, const char *gguf_path) {
    madvise(e->map, e->map_len, MADV_SEQUENTIAL);
    e->per_layer_token_embd.blocks = XE_PLE_WIDTH / 32;

    static xe_repack_job jobs[XE_TENSOR_COUNT];
    int njobs = 0;
    uint64_t slab_size = 0, nibble_bytes = 0, scale_bytes = 0;

    for (size_t i = 0; i < XE_NSPECS; i++) {
        const xe_tspec *s = &xe_specs[i];
        if (s->qtype != XE_GGML_TYPE_Q4_0 || s->cat == XE_CAT_PLE) continue;
        int blocks = (int)(s->ne0 / 32);
        uint64_t rows = s->ne1 * s->ne2;
        uint64_t nblocks = (uint64_t)blocks * rows;
        uint64_t nb64 = xe_align_up(nblocks * 16, 64);
        uint64_t sb64 = xe_align_up(nblocks * 2, 64);

        if (s->kind == XE_TK_SINGLETON) {
            if (!((xe_q4 *)((char *)e + s->off))->qs) continue;
            jobs[njobs++] = (xe_repack_job){
                .field = (xe_q4 *)((char *)e + s->off), .nblocks = nblocks,
                .blocks = blocks, .rows = rows, .suffix = s->suffix, .layer_idx = -1 };
            nibble_bytes += nblocks * 16;
            scale_bytes += nblocks * 2;
            slab_size += nb64 + sb64;
            continue;
        }
        for (int l = 0; l < XE_LAYERS; l++) {
            if (s->kind == XE_TK_SWA && XE_IS_GLOBAL(l)) continue;
            if (s->kind == XE_TK_GLOBAL && !XE_IS_GLOBAL(l)) continue;
            if (!((xe_q4 *)((char *)&e->layers[l] + s->off))->qs) continue;
            jobs[njobs++] = (xe_repack_job){
                .field = (xe_q4 *)((char *)&e->layers[l] + s->off), .nblocks = nblocks,
                .blocks = blocks, .rows = rows, .suffix = s->suffix, .layer_idx = l };
            nibble_bytes += nblocks * 16;
            scale_bytes += nblocks * 2;
            slab_size += nb64 + sb64;
        }
    }

    if ((uint64_t)njobs + 1 != e->q4_0_count)
        xe_fatal("%s: repack job count %d mismatches Q4_0 tensor count %llu", gguf_path, njobs, (unsigned long long)e->q4_0_count);

    for (int i = 0; i < njobs; i++)
        jobs[i].source_offset = (uint64_t)(jobs[i].field->qs -
                                          (const uint8_t *)e->map);
    qsort(jobs, (size_t)njobs, sizeof *jobs, xe_repack_source_order);

    int part = 0;
    uint64_t part_used = 0;
    for (int i = 0; i < njobs; i++) {
        uint64_t nb64 = xe_align_up(jobs[i].nblocks * 16, 64);
        uint64_t sb64 = xe_align_up(jobs[i].nblocks * 2, 64);
        uint64_t bytes = nb64 + sb64;
        if (bytes > XE_REPACK_PART_LIMIT)
            xe_fatal("%s: repack tensor %s exceeds shared allocation limit",
                     gguf_path, jobs[i].suffix);
        if (part_used && part_used + bytes > XE_REPACK_PART_LIMIT) {
            e->repack_part_sizes[part++] = part_used;
            part_used = 0;
        }
        if (part >= XE_REPACK_PARTS)
            xe_fatal("%s: repack needs more than %d shared allocations",
                     gguf_path, XE_REPACK_PARTS);
        jobs[i].part = part;
        jobs[i].part_offset = part_used;
        part_used += bytes;
    }
    e->repack_part_sizes[part++] = part_used;
    e->repack_part_count = part;
    e->repack_slab_size = slab_size;
    e->repack_nibble_bytes = nibble_bytes;
    e->repack_scale_bytes = scale_bytes;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    xe_copy_f32(e);

    uint64_t released = xe_align_up(e->header_len, 4096);
    uint64_t verified = 0;
    int first = 0;
    for (int i = 0; i < e->repack_part_count; i++) {
        e->repack_parts[i] = xe_alloc(e, e->repack_part_sizes[i], XE_MEM_SHARED);
#if defined(XE_REPACK_HUGEPAGE) || defined(XE_REPACK_NOHUGEPAGE)
        size_t page = 4096;
        int huge_advice =
#ifdef XE_REPACK_HUGEPAGE
            MADV_HUGEPAGE;
#else
            MADV_NOHUGEPAGE;
#endif
        uintptr_t huge_begin = ((uintptr_t)e->repack_parts[i] + page - 1) &
                               ~(uintptr_t)(page - 1);
        uintptr_t huge_end = ((uintptr_t)e->repack_parts[i] +
                              e->repack_part_sizes[i]) & ~(uintptr_t)(page - 1);
        if (huge_end > huge_begin &&
            madvise((void *)huge_begin, huge_end - huge_begin, huge_advice) != 0)
            xe_fatal("%s: madvise repack slab: %s", gguf_path, strerror(errno));
#endif
        int end = first;
        while (end < njobs && jobs[end].part == i) {
            uint64_t nb64 = xe_align_up(jobs[end].nblocks * 16, 64);
            uint8_t *cursor = (uint8_t *)e->repack_parts[i] +
                              jobs[end].part_offset;
            jobs[end].qs_seg = cursor;
            jobs[end].d_seg = (uint16_t *)(cursor + nb64);
            end++;
        }

        atomic_int next_job;
        atomic_init(&next_job, 0);
        pthread_t threads[XE_REPACK_THREADS];
        xe_repack_worker_arg args[XE_REPACK_THREADS];
        for (int t = 0; t < XE_REPACK_THREADS; t++) {
            args[t] = (xe_repack_worker_arg){ jobs + first, end - first,
                                             &next_job, gguf_path, 0 };
            if (pthread_create(&threads[t], NULL, xe_repack_worker, &args[t]) != 0)
                xe_fatal("%s: pthread_create failed for repack worker %d", gguf_path, t);
        }
        for (int t = 0; t < XE_REPACK_THREADS; t++) {
            pthread_join(threads[t], NULL);
            verified += args[t].verified;
        }
        uint64_t consumed = end < njobs ? jobs[end].source_offset : e->map_len;
        xe_repack_release(e, &released, consumed, gguf_path);
        first = end;
    }
    close(e->map_fd);
    e->map_fd = -1;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    e->repack_seconds = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    e->repack_verified_blocks = verified;

#ifdef XE_REPACK_DROP_SOURCE
    if (madvise(e->map, e->map_len, MADV_DONTNEED) != 0)
        xe_fatal("%s: madvise source map: %s", gguf_path, strerror(errno));
#else
    madvise(e->map, e->map_len, MADV_NORMAL);
#endif
}

static xe_engine *xe_open_common(const char *gguf_path, int vocab_only, xe_cur *c) {
    xe_engine *e = xe_map_file(gguf_path);
    e->vocab_only = vocab_only;
    e->context = XE_CONTEXT_DEFAULT;
    e->scalar_rms = 1;
    *c = xe_parse_header(e, gguf_path);
    e->model = xe_detect_model(e, *c, gguf_path);
    if (!vocab_only && (e->n_tensors == 0 || e->n_tensors > XE_TENSOR_COUNT))
        xe_fatal("%s: unsupported tensor directory size", gguf_path);
    xe_parse_metadata(e, c, gguf_path);
    xe_tok_build(e, gguf_path);
    if (vocab_only) {
        close(e->map_fd);
        e->map_fd = -1;
    }
    return e;
}

static void xe_snapshot_fingerprints_init(xe_engine *e);

const uint8_t *xe_engine_model_fingerprint(xe_engine *e) {
    if (!e) return NULL;
    if (e->pool_initialized) xe_require_owner(e);
    if (!e->model_content_hash_ready) {
        format_sha256_bytes(e->map, e->map_len, e->model_content_hash);
        e->model_content_hash_ready = 1;
    }
    return e->model_content_hash;
}

static xe_engine *xe_engine_open_impl(const char *gguf_path, int context,
                                      int prepare_cache) {
    if (context < XE_CONTEXT_MIN || context > XE_CONTEXT_MAX)
        xe_fatal("context: capacity %d out of range [%d, %d]",
                 context, XE_CONTEXT_MIN, XE_CONTEXT_MAX);
    xe_cur c;
    xe_engine *e = xe_open_common(gguf_path, 0, &c);
    if (!e->model->runtime_ready) {
        const char *optin=getenv("XE_ALLOW_UNVALIDATED_E4B");
        if(!optin || strcmp(optin,"1")) xe_fatal(
            "%s: E4B execution is implemented but unvalidated. "
            "Use XE_ALLOW_UNVALIDATED_E4B=1 only in the separate validation environment.",gguf_path);
    }
    if ((uint32_t)context > e->model->context_length)
        xe_fatal("context: capacity %d exceeds %s model limit %u", context,
                 e->model->name, e->model->context_length);
    e->context = context;
    xe_bind_tensors(e, &c, gguf_path);
    xe_check_complete(e, gguf_path);
    if (prepare_cache) {
        /* Identify the original GGUF before repacking or dropping source pages.
         * This is cache identity, not an allowlist of supported weight values. */
        struct timespec hash_start, hash_end;
        fprintf(stderr, "xenolith: computing full model SHA-256...\n");
        clock_gettime(CLOCK_MONOTONIC, &hash_start);
        xe_engine_model_fingerprint(e);
        clock_gettime(CLOCK_MONOTONIC, &hash_end);
        fprintf(stderr, "xenolith: model SHA-256 ready (%.2f s)\n",
                (double)(hash_end.tv_sec - hash_start.tv_sec) +
                (double)(hash_end.tv_nsec - hash_start.tv_nsec) / 1e9);
    }
    xe_constants_init(e);
    xe_gpu_init(e);
    xe_repack(e, gguf_path);
    if (prepare_cache) xe_snapshot_fingerprints_init(e);
    return e;
}

xe_engine *xe_engine_open_with_context(const char *gguf_path, int context) {
    return xe_engine_open_impl(gguf_path, context, 0);
}

xe_engine *xe_engine_open_cached(const char *gguf_path, int context) {
    return xe_engine_open_impl(gguf_path, context, 1);
}

xe_engine *xe_engine_open(const char *gguf_path) {
    return xe_engine_open_with_context(gguf_path, XE_CONTEXT_DEFAULT);
}

xe_engine *xe_engine_open_vocab(const char *gguf_path) {
    xe_cur c;
    return xe_open_common(gguf_path, 1, &c);
}

void xe_engine_close(xe_engine *e) {
    if (!e) return;
    xe_worker_pool_destroy(e);
    xe_free(e, e->f32_slab, XE_MEM_SHARED);
    for (int i = 0; i < e->repack_part_count; i++)
        xe_free(e, e->repack_parts[i], XE_MEM_SHARED);
    xe_gpu_destroy(e);
    xe_free(e, e->tok_merge, XE_MEM_HOST);
    xe_free(e, e->tok_hash, XE_MEM_HOST);
    xe_free(e, e->tok_piece, XE_MEM_HOST);
    if (e->map && e->map_fd >= 0) close(e->map_fd);
    munmap(e->map, e->map_len);
    free(e->name);
    free(e);
}

void xe_engine_info(const xe_engine *e, FILE *out) {
    fprintf(out,"model            gemma-4-e4b (dense text backbone)\n");
    fprintf(out,"runtime          implemented; awaiting separate model validation\n");
    fprintf(out,"layers           %d; hidden %d; FFN %d; PLE %d\n",XE_LAYERS,XE_EMBD,XE_DENSE_FFN,XE_PLE);
    fprintf(out,"attention        8 query / 2 KV heads; local 256, global 512\n");
    fprintf(out,"KV storage       24 donors (20 local + 4 global); 18 shared layers\n");
    fprintf(out,"context          %d (maximum %d)\n",e->context,XE_MODEL_CTX);
    fprintf(out,"weights          Q4_0 matrices / F32 norms; PLE table mmap %.2f GiB\n",e->ple_bytes/1073741824.0);
    fprintf(out,"prefill          baseline token-serial GPU projections; CPU attention\n");
}


static uint32_t xe_utf8_len(uint8_t c) {
    static const uint8_t lookup[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    return lookup[c >> 4];
}

typedef struct {
    int32_t id;
    int32_t prev, next;
    uint32_t off, n;
} xe_sym;

typedef struct {
    uint32_t rank;
    uint32_t lpos, rpos;
    int32_t lid, rid, res;
} xe_big;

static int xe_big_less(const xe_big *a, const xe_big *b) {
    return a->rank < b->rank || (a->rank == b->rank && a->lpos < b->lpos);
}

static void xe_heap_push(xe_big *h, int *n, xe_big v) {
    int i = (*n)++;
    h[i] = v;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!xe_big_less(&h[i], &h[p])) break;
        xe_big t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static xe_big xe_heap_pop(xe_big *h, int *n) {
    xe_big top = h[0];
    h[0] = h[--(*n)];
    for (int i = 0;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && xe_big_less(&h[l], &h[m])) m = l;
        if (r < *n && xe_big_less(&h[r], &h[m])) m = r;
        if (m == i) break;
        xe_big t = h[m]; h[m] = h[i]; h[i] = t;
        i = m;
    }
    return top;
}

static void xe_try_bigram(const xe_engine *e, xe_sym *sym, int l, int r,
                          xe_big *heap, int *hn) {
    if (l < 0 || r < 0) return;
    if (sym[l].id < 0 || sym[r].id < 0) return;
    uint32_t rank; int32_t res;
    if (!xe_merge_find(e, sym[l].id, sym[r].id, &rank, &res)) return;
    xe_heap_push(heap, hn, (xe_big){ rank, (uint32_t)l, (uint32_t)r,
                                     sym[l].id, sym[r].id, res });
}

static int xe_bpe_chunk(const xe_engine *e, const char *s, uint32_t len,
                        int32_t *out) {
    if (len == 0) return 0;

    xe_sym *sym = xe_alloc(e, len * sizeof *sym, XE_MEM_HOST);
    xe_big *heap = xe_alloc(e, 3 * len * sizeof *heap, XE_MEM_HOST);
    int nsym = 0, hn = 0;

    for (uint32_t off = 0; off < len; ) {
        uint32_t n = xe_utf8_len((uint8_t)s[off]);
        sym[nsym] = (xe_sym){ xe_vocab_find(e, s + off, n),
                              nsym - 1, nsym + 1, off, n };
        nsym++; off += n;
    }
    sym[nsym - 1].next = -1;

    for (int i = 0; i + 1 < nsym; i++)
        xe_try_bigram(e, sym, i, i + 1, heap, &hn);

    while (hn > 0) {
        xe_big b = xe_heap_pop(heap, &hn);
        xe_sym *l = &sym[b.lpos], *r = &sym[b.rpos];
        if (l->n == 0 || r->n == 0) continue;
        if (l->id != b.lid || r->id != b.rid) continue;
        l->id = b.res;
        l->n += r->n;
        l->next = r->next;
        if (r->next >= 0) sym[r->next].prev = (int32_t)b.lpos;
        r->n = 0;
        xe_try_bigram(e, sym, l->prev, (int)b.lpos, heap, &hn);
        xe_try_bigram(e, sym, (int)b.lpos, l->next, heap, &hn);
    }

    int nout = 0;
    for (int i = 0; i >= 0; i = sym[i].next) {
        if (sym[i].id >= 0) { out[nout++] = sym[i].id; continue; }
        for (uint32_t k = 0; k < sym[i].n; k++)
            out[nout++] = XE_BYTE0_ID + (uint8_t)s[sym[i].off + k];
    }

    xe_free(e, heap, XE_MEM_HOST);
    xe_free(e, sym, XE_MEM_HOST);
    return nout;
}

static uint32_t xe_cpt_to_utf8(uint32_t cpt, char *dst) {
    if (cpt <= 0x7f) {
        dst[0] = (char)cpt;
        return 1;
    }
    if (cpt <= 0x7ff) {
        dst[0] = (char)(0xc0 | ((cpt >> 6) & 0x1f));
        dst[1] = (char)(0x80 | (cpt & 0x3f));
        return 2;
    }
    if (cpt <= 0xffff) {
        dst[0] = (char)(0xe0 | ((cpt >> 12) & 0x0f));
        dst[1] = (char)(0x80 | ((cpt >> 6) & 0x3f));
        dst[2] = (char)(0x80 | (cpt & 0x3f));
        return 3;
    }
    dst[0] = (char)(0xf0 | ((cpt >> 18) & 0x07));
    dst[1] = (char)(0x80 | ((cpt >> 12) & 0x3f));
    dst[2] = (char)(0x80 | ((cpt >> 6) & 0x3f));
    dst[3] = (char)(0x80 | (cpt & 0x3f));
    return 4;
}

static uint32_t xe_escape(const char *text, size_t n, char *dst) {
    const uint8_t *s = (const uint8_t *)text;
    uint32_t o = 0;
    size_t i = 0;
    while (i < n) {
        uint8_t c = s[i];
        if (!(c & 0x80)) {
            if (c == ' ') {
                dst[o++] = (char)0xe2; dst[o++] = (char)0x96; dst[o++] = (char)0x81;
            } else {
                dst[o++] = (char)c;
            }
            i++;
            continue;
        }
        uint32_t need = 0;
        if (!(c & 0x40)) need = 0;
        else if (!(c & 0x20)) need = 2;
        else if (!(c & 0x10)) need = 3;
        else if (!(c & 0x08)) need = 4;
        int ok = need != 0 && i + need <= n;
        for (uint32_t k = 1; ok && k < need; k++)
            if ((s[i + k] & 0xc0) != 0x80) ok = 0;
        if (!ok) {
            dst[o++] = (char)0xef; dst[o++] = (char)0xbf; dst[o++] = (char)0xbd;
            i++;
            continue;
        }
        uint32_t cpt = c & (0x7fu >> need);
        for (uint32_t k = 1; k < need; k++) cpt = (cpt << 6) | (s[i + k] & 0x3f);
        i += need;
        o += xe_cpt_to_utf8(cpt, dst + o);
    }
    return o;
}

int xe_encode_text_bounded(const xe_engine *e, const char *text, int32_t *out, int cap) {
    if (!e || !text || !out) xe_fatal("encode_text: missing input");
    if (cap < 0) xe_fatal("encode_text: negative output capacity %d", cap);
    size_t n = strlen(text);
    if (n == 0) return 0;
    if (n > UINT32_MAX / 3u) xe_fatal("encode_text: input too large");

    char *buf = xe_alloc(e, 3 * n, XE_MEM_HOST);
    int32_t *tmp = xe_alloc(e, 3 * n * sizeof(*tmp), XE_MEM_HOST);
    uint32_t m = xe_escape(text, n, buf);
    int nout = 0;

    for (uint32_t i = 0; i < m; ) {
        uint32_t j = i;
        if (buf[i] == '\n') {
            while (j < m && buf[j] == '\n') j++;
            int32_t id = xe_vocab_find(e, buf + i, j - i);
            if (id >= 0) tmp[nout++] = id;
            else nout += xe_bpe_chunk(e, buf + i, j - i, tmp + nout);
        } else {
            while (j < m && buf[j] != '\n') j++;
            nout += xe_bpe_chunk(e, buf + i, j - i, tmp + nout);
        }
        i = j;
    }

    xe_free(e, buf, XE_MEM_HOST);
    if (nout <= cap) memcpy(out, tmp, (size_t)nout * sizeof(*out));
    xe_free(e, tmp, XE_MEM_HOST);
    return nout;
}

int xe_encode_text(const xe_engine *e, const char *text, int32_t *out, int cap) {
    int n = xe_encode_text_bounded(e, text, out, cap);
    if (n > cap) xe_fatal("encode_text: output capacity %d too small (need %d)", cap, n);
    return n;
}

static int32_t xe_token_type(const xe_engine *e, int32_t tok) {
    int32_t t;
    memcpy(&t, (const uint8_t *)e->tok_token_type + 4 * (size_t)tok, 4);
    return t;
}

int xe_detokenize(const xe_engine *e, int32_t tok, char *buf, int cap) {
    if (tok < 0 || tok >= XE_VOCAB)
        xe_fatal("detokenize: token id %d out of range [0, %d]", tok, XE_VOCAB - 1);

    int32_t type = xe_token_type(e, tok);
    if (type == XE_TT_BYTE) {
        if (cap < 1) xe_fatal("detokenize: output capacity %d too small for token %d", cap, tok);
        buf[0] = (char)(uint8_t)(tok - XE_BYTE0_ID);
        return 1;
    }
    if (type != XE_TT_NORMAL) return 0;

    const xe_str *p = &e->tok_piece[tok];
    int nout = 0;
    for (uint64_t i = 0; i < p->len; ) {
        if (nout >= cap) xe_fatal("detokenize: output capacity %d too small for token %d", cap, tok);
        if (p->len - i >= 3 && (uint8_t)p->p[i] == 0xe2 && (uint8_t)p->p[i + 1] == 0x96 && (uint8_t)p->p[i + 2] == 0x81) {
            buf[nout++] = ' ';
            i += 3;
        } else {
            buf[nout++] = p->p[i++];
        }
    }
    return nout;
}

int32_t xe_token_id(const xe_engine *e, const char *piece) {
    return xe_vocab_find(e, piece, (uint32_t)strlen(piece));
}

const char *xe_token_piece(const xe_engine *e, int32_t tok, int *len) {
    if (tok < 0 || tok >= XE_VOCAB)
        xe_fatal("token_piece: token id %d out of range [0, %d]", tok, XE_VOCAB - 1);
    *len = (int)e->tok_piece[tok].len;
    return e->tok_piece[tok].p;
}

int32_t xe_bos_id(const xe_engine *e) {
    return e->bos_id;
}

int32_t xe_eos_id(const xe_engine *e) {
    return e->eos_id;
}

int32_t xe_eot_id(const xe_engine *e) {
    return e->eot_id;
}

int xe_context_size(const xe_engine *e) {
    if (!e) xe_fatal("context_size: missing engine");
    return e->context;
}

int xe_vocab_size(const xe_engine *e) {
    if (!e) xe_fatal("vocab_size: missing engine");
    return (int)e->tok_tokens_count;
}

uint64_t xe_engine_model_size(const xe_engine *e) {
    if (!e) xe_fatal("model_size: missing engine");
    return e->q4_0_bytes + e->f32_bytes;
}

int xe_engine_worker_count(const xe_engine *e) {
    if (!e) xe_fatal("worker_count: missing engine");
    return XE_WORKERS;
}

int xe_engine_worker_cpu(const xe_engine *e, int worker) {
    if (!e) xe_fatal("worker_cpu: missing engine");
    if (worker < 0 || worker >= XE_WORKERS)
        xe_fatal("worker_cpu: worker %d out of range [0, %d]", worker, XE_WORKERS - 1);
    return e->worker_pinned ? xe_worker_cpu(worker) : -1;
}

static int xe_tokens_valid(const xe_tokens *tokens) {
    return tokens && tokens->len >= 0 && tokens->cap >= 0 && tokens->cap <= XE_MODEL_CTX &&
        tokens->len <= tokens->cap && ((tokens->v != NULL) == (tokens->cap != 0));
}

static void xe_tokens_reserve(xe_tokens *tokens, int need) {
    if (!xe_tokens_valid(tokens)) xe_fatal("tokens: invalid vector");
    if (need < tokens->len || need > XE_MODEL_CTX)
        xe_fatal("tokens: length %d out of range [0, %d]", need, XE_MODEL_CTX);
    if (need <= tokens->cap) return;

    int cap = tokens->cap ? tokens->cap : 64;
    while (cap < need) {
        if (cap > XE_MODEL_CTX / 2) {
            cap = XE_MODEL_CTX;
            break;
        }
        cap *= 2;
    }
    int32_t *v = xe_alloc(NULL, (size_t)cap * sizeof(*v), XE_MEM_HOST);
    if (tokens->len)
        memcpy(v, tokens->v, (size_t)tokens->len * sizeof(*v));
    xe_free(NULL, tokens->v, XE_MEM_HOST);
    tokens->v = v;
    tokens->cap = cap;
}

void xe_tokens_push(xe_tokens *tokens, int32_t token) {
    if (!xe_tokens_valid(tokens)) xe_fatal("tokens_push: invalid vector");
    if (tokens->len == XE_MODEL_CTX) xe_fatal("tokens_push: context is full");
    xe_tokens_reserve(tokens, tokens->len + 1);
    tokens->v[tokens->len++] = token;
}

void xe_tokens_free(xe_tokens *tokens) {
    if (!tokens) return;
    xe_free(NULL, tokens->v, XE_MEM_HOST);
    memset(tokens, 0, sizeof(*tokens));
}

const char *xe_chat_template(const xe_engine *e, uint64_t *length) {
    if (!e) xe_fatal("chat_template: missing engine");
    if (length) *length = e->chat_template.len;
    return e->chat_template.p;
}

static inline __m256 xe_load_f16x8(const _Float16 *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}

static inline float xe_sum_f32x8(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static _Float16 *xe_kv_layer_ptr(xe_session *s, int layer, int value) {
    if (layer < 0 || layer >= XE_LAYERS) xe_fatal("kv: layer %d out of range", layer);
    layer = (int)xe_gemma4_kv_donor(s->engine->model, (uint32_t)layer);
    if (XE_IS_GLOBAL(layer)) {
        if (s->cow) {
            _Float16 *base = value ? s->cow_global_v[layer / 6]
                                   : s->cow_global_k[layer / 6];
            if (!base) xe_fatal("kv: unallocated COW layer %d", layer);
            return base;
        }
        _Float16 *base = value ? s->global_v : s->global_k;
        return base + (size_t)(layer / 6) * xe_global_layer_elems(s->engine);
    }
    _Float16 *base = value ? s->swa_v : s->swa_k;
    return base + (size_t)(layer - layer / 6) * XE_SWA_LAYER_ELEMS;
}

void xe_kv_append_f16(_Float16 *k_cache, _Float16 *v_cache, int n_kv_heads,
                      int head_dim, int capacity, int slot, const float *k, const float *v) {
    if (slot < 0 || slot >= capacity) xe_fatal("kv: slot %d out of range [0, %d]", slot, capacity - 1);
    for (int h = 0; h < n_kv_heads; h++) {
        _Float16 *kd = k_cache + ((size_t)h * capacity + slot) * head_dim;
        _Float16 *vd = v_cache + ((size_t)h * capacity + slot) * head_dim;
        const float *ks = k + (size_t)h * head_dim;
        const float *vs = v + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; d += 8) {
            __m128i kh = _mm256_cvtps_ph(_mm256_loadu_ps(ks + d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i vh = _mm256_cvtps_ph(_mm256_loadu_ps(vs + d), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i *)(kd + d), kh);
            _mm_storeu_si128((__m128i *)(vd + d), vh);
        }
    }
}

void xe_session_kv_append(xe_session *s,int layer,int pos,const float *k,const float *v) {
    if(layer<0 || layer>=XE_KV_LAYERS) xe_fatal("kv: only donor layers own cache writes");
    if(pos<0 || pos>=s->engine->context) xe_fatal("kv: invalid position");
    int global=XE_IS_GLOBAL(layer),dim=global?XE_GLOBAL_HEAD_DIM:XE_SWA_HEAD_DIM;
    int capacity=global?(s->cow?s->cow_capacity:s->engine->context):XE_SWA_WINDOW;
    int slot=global?(s->cow?pos-s->cow_split:pos):pos&(XE_SWA_WINDOW-1);
    xe_kv_append_f16(xe_kv_layer_ptr(s,layer,0),xe_kv_layer_ptr(s,layer,1),2,dim,capacity,slot,k,v);
}


/* Reference-shaped GQA attention. Eight query heads share two KV heads.
 * Cache access keeps immutable global prefixes and COW tails separate. */
static float xe_f16_dot(const float *q,const _Float16 *k,int n) {
    float sum=0; for(int i=0;i<n;i++) sum+=q[i]*(float)k[i]; return sum;
}
void xe_kv_attention_f16(const float *q,const _Float16 *kc,const _Float16 *vc,
                         int heads,int dim,int capacity,int pos,int window,
                         float *scores,float *out) {
    if(heads!=2 || (dim!=256 && dim!=512) || pos<0 ||
       (!window && pos>=capacity) || (window && capacity!=window))
        xe_fatal("kv: unsupported E4B attention shape/position");
    int count=pos+1; if(window && count>window) count=window;
    int first=pos+1-count;
    for(int h=0;h<XE_Q_HEADS;h++) {
        float *sc=scores+(size_t)h*count,*dst=out+(size_t)h*dim;
        for(int i=0;i<count;i++) {
            int slot=window?(first+i)%capacity:first+i;
            sc[i]=xe_f16_dot(q+(size_t)h*dim,kc+((size_t)(h/4)*capacity+slot)*dim,dim);
        }
        xe_softmax_f32(sc,count); memset(dst,0,dim*sizeof(float));
        for(int i=0;i<count;i++) {
            int slot=window?(first+i)%capacity:first+i;
            const _Float16 *v=vc+((size_t)(h/4)*capacity+slot)*dim;
            for(int d=0;d<dim;d++) dst[d]+=sc[i]*(float)v[d];
        }
    }
}
typedef struct {int layer,pos; const float *q; float *scores,*out;} xe_e4b_attention_arg;
static const _Float16 *xe_e4b_cache_row(xe_session *s,int layer,int head,int pos,int value) {
    int donor=(int)xe_gemma4_kv_donor(s->engine->model,(uint32_t)layer);
    int global=XE_IS_GLOBAL(donor),dim=global?XE_GLOBAL_HEAD_DIM:XE_SWA_HEAD_DIM;
    int cap=global?s->engine->context:XE_SWA_WINDOW,slot=global?pos:pos&(XE_SWA_WINDOW-1);
    const _Float16 *base;
    if(global && s->cow && pos<s->cow_split) {
        base=(value?s->cow_source->global_v:s->cow_source->global_k) +
             (size_t)(donor/6)*xe_global_layer_elems(s->engine);
    } else {
        base=xe_kv_layer_ptr(s,donor,value);
        if(global && s->cow) {cap=s->cow_capacity;slot=pos-s->cow_split;}
    }
    return base+((size_t)head*cap+slot)*dim;
}
static void xe_e4b_attention_phase(xe_session *s,const void *opaque,int worker,int workers) {
    const xe_e4b_attention_arg *a=opaque;
    int global=XE_IS_GLOBAL(a->layer),dim=global?XE_GLOBAL_HEAD_DIM:XE_SWA_HEAD_DIM;
    int count=a->pos+1;if(!global && count>XE_SWA_WINDOW) count=XE_SWA_WINDOW;
    int first=a->pos+1-count;
    for(int h=XE_Q_HEADS*worker/workers;h<XE_Q_HEADS*(worker+1)/workers;h++) {
        float *sc=a->scores+(size_t)h*count,*dst=a->out+(size_t)h*dim;
        for(int i=0;i<count;i++)
            sc[i]=xe_f16_dot(a->q+(size_t)h*dim,xe_e4b_cache_row(s,a->layer,h/4,first+i,0),dim);
        xe_softmax_f32(sc,count);memset(dst,0,dim*sizeof(float));
        for(int i=0;i<count;i++) {
            const _Float16 *v=xe_e4b_cache_row(s,a->layer,h/4,first+i,1);
            for(int d=0;d<dim;d++) dst[d]+=sc[i]*(float)v[d];
        }
    }
}
void xe_session_kv_attention(xe_session *s,int layer,int pos,const float *q,float *scores,float *out) {
    if(layer<0 || layer>=XE_LAYERS || pos<0 || pos>=s->engine->context)
        xe_fatal("kv: invalid layer/position");
    xe_e4b_attention_arg a={layer,pos,q,scores,out};
    xe_dispatch(s->engine,s,xe_e4b_attention_phase,&a,1);
}

static void xe_embed_decode(const xe_engine *e, int32_t token, float *x) {
    int blocks = XE_EMBD / 32;
    float factor = sqrtf((float)XE_EMBD);
    for (int b = 0; b < blocks; b++) {
        _Alignas(16) uint8_t packed[16];
        xe_q4_copy_block(&e->tok_embd, (uint64_t)token, blocks, b, packed);
        float d = _cvtsh_ss(xe_q4_scale(&e->tok_embd, (uint64_t)token, blocks, b)) * factor;
        for (int i = 0; i < 16; i++) {
            x[32 * b + i] = (float)((packed[i] & 15) - 8) * d;
            x[32 * b + i + 16] = (float)((packed[i] >> 4) - 8) * d;
        }
    }
}

static void xe_dispatch_matvec(xe_session *s, const xe_q4 *w, uint64_t first_row,
                               int n_in, int n_out, const xe_q8 *q, float *y) {
    xe_matvec_arg a = { w, first_row, n_in, n_out, q, y };
    xe_dispatch(s->engine, s, xe_matvec_phase, &a, 1);
}

/* All E4B matrices use the same row-interleaved Q4_0 layout.
 * The reference option deliberately avoids activation quantization, but shares
 * the graph/cache implementation: it is not an independent correctness oracle. */
static void xe_e4b_project(xe_session *s,const xe_q4 *w,int ni,int no,const float *x,float *y) {
    if(s->reference_mode) {
        for(int row=0;row<no;row++) {
            float sum=0;
            for(int b=0;b<ni/32;b++) {
                uint8_t q[16];xe_q4_copy_block(w,row,ni/32,b,q);
                float d=_cvtsh_ss(xe_q4_scale(w,row,ni/32,b));
                for(int j=0;j<16;j++) {
                    sum+=d*((int)(q[j]&15)-8)*x[b*32+j];
                    sum+=d*((int)(q[j]>>4)-8)*x[b*32+j+16];
                }
            }
            y[row]=sum;
        }
        return;
    }
    xe_quantize_q8(x,ni,&s->q8_main);
    if(!s->gpu_prefill) {xe_dispatch_matvec(s,w,0,ni,no,&s->q8_main,y);return;}
    ze_kernel_handle_t k=s->engine->gpu.e4b_q4q8;
    xe_gpu_pointer_arg(k,0,w->qs);xe_gpu_pointer_arg(k,1,w->d);
    xe_gpu_pointer_arg(k,2,s->q8_main.qs);xe_gpu_pointer_arg(k,3,s->q8_main.d);
    xe_gpu_pointer_arg(k,4,y);xe_gpu_int_arg(k,5,ni/32);xe_gpu_int_arg(k,6,no);
    ze_group_count_t groups={(uint32_t)((no+63)/64),1,1};
    xe_ze_check("E4B projection",zeCommandListAppendLaunchKernel(s->engine->gpu.commands,k,&groups,NULL,0,NULL));
    xe_ze_check("E4B projection synchronize",zeCommandListHostSynchronize(s->engine->gpu.commands,UINT64_MAX));
}
static float xe_e4b_gelu(float x) {
    return 0.5f*x*(1.0f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)));
}
static void xe_e4b_prepare_ple(xe_session *s,int32_t token) {
    xe_engine *e=s->engine;
    const uint8_t *row=e->per_layer_token_embd.qs+(size_t)token*(XE_PLE_WIDTH/32)*18;
    for(int b=0;b<XE_PLE_WIDTH/32;b++) {
        uint16_t bits;memcpy(&bits,row+b*18,2);
        float d=_cvtsh_ss(bits)*sqrtf((float)XE_PLE);
        const uint8_t *q=row+b*18+2;
        for(int j=0;j<16;j++) {
            s->ple_lookup[b*32+j]=d*((int)(q[j]&15)-8);
            s->ple_lookup[b*32+j+16]=d*((int)(q[j]>>4)-8);
        }
    }
    xe_e4b_project(s,&e->per_layer_model_proj,XE_EMBD,XE_PLE_WIDTH,s->hidden,s->ple_inputs);
    for(int i=0;i<XE_PLE_WIDTH;i++) s->ple_inputs[i]/=sqrtf((float)XE_EMBD);
    for(int l=0;l<XE_LAYERS;l++) {
        float *p=s->ple_inputs+l*XE_PLE;
        xe_rmsnorm_engine(e,p,e->per_layer_proj_norm,XE_PLE,p);
    }
    for(int i=0;i<XE_PLE_WIDTH;i++)
        s->ple_inputs[i]=(s->ple_inputs[i]+s->ple_lookup[i])*0.7071067811865475f;
}
static void xe_attention_half_decode(xe_session *s,int layer,int pos) {
    const xe_layer *l=&s->engine->layers[layer];
    int global=XE_IS_GLOBAL(layer),dim=global?XE_GLOBAL_HEAD_DIM:XE_SWA_HEAD_DIM;
    const float *co=global?s->rope_global_cos:s->rope_swa_cos;
    const float *si=global?s->rope_global_sin:s->rope_swa_sin;
    xe_rmsnorm_engine(s->engine,s->hidden,l->attn_norm,XE_EMBD,s->normed);
    xe_e4b_project(s,&l->attn_q,XE_EMBD,XE_Q_HEADS*dim,s->normed,s->q);
    for(int h=0;h<XE_Q_HEADS;h++)
        xe_rmsnorm_engine(s->engine,s->q+h*dim,l->q_norm,dim,s->q+h*dim);
    xe_rope_apply(s->q,XE_Q_HEADS,dim,co,si);
    if(layer<XE_KV_LAYERS) {
        xe_e4b_project(s,&l->attn_k,XE_EMBD,2*dim,s->normed,s->k);
        xe_e4b_project(s,&l->attn_v,XE_EMBD,2*dim,s->normed,s->v);
        for(int h=0;h<2;h++) {
            xe_rmsnorm_engine(s->engine,s->k+h*dim,l->k_norm,dim,s->k+h*dim);
            xe_rmsnorm_engine(s->engine,s->v+h*dim,NULL,dim,s->v+h*dim);
        }
        xe_rope_apply(s->k,2,dim,co,si);
        xe_session_kv_append(s,layer,pos,s->k,s->v);
    }
    xe_session_kv_attention(s,layer,pos,s->q,s->scores,s->attn_heads);
    xe_e4b_project(s,&l->attn_o,XE_Q_HEADS*dim,XE_EMBD,s->attn_heads,s->attn_proj);
    xe_rmsnorm_engine(s->engine,s->attn_proj,l->post_attn_norm,XE_EMBD,s->attn_proj);
    for(int i=0;i<XE_EMBD;i++) s->attn_out[i]=s->hidden[i]+s->attn_proj[i];
}


static void xe_ffn_half_decode(xe_session *s,int layer) {
    const xe_layer *l=&s->engine->layers[layer];
    xe_rmsnorm_engine(s->engine,s->attn_out,l->ffn_norm,XE_EMBD,s->normed);
    xe_e4b_project(s,&l->ffn_gate,XE_EMBD,XE_DENSE_FFN,s->normed,s->gate);
    xe_e4b_project(s,&l->ffn_up,XE_EMBD,XE_DENSE_FFN,s->normed,s->up);
    for(int i=0;i<XE_DENSE_FFN;i++) s->activation[i]=xe_e4b_gelu(s->gate[i])*s->up[i];
    xe_e4b_project(s,&l->ffn_down,XE_DENSE_FFN,XE_EMBD,s->activation,s->dense_out);
    xe_rmsnorm_engine(s->engine,s->dense_out,l->post_ffw_norm,XE_EMBD,s->dense_out);
    for(int i=0;i<XE_EMBD;i++) s->hidden[i]=s->attn_out[i]+s->dense_out[i];
    xe_e4b_project(s,&l->inp_gate,XE_EMBD,XE_PLE,s->hidden,s->ple_gate);
    for(int i=0;i<XE_PLE;i++) s->ple_gate[i]=xe_e4b_gelu(s->ple_gate[i])*s->ple_inputs[layer*XE_PLE+i];
    xe_e4b_project(s,&l->proj,XE_PLE,XE_EMBD,s->ple_gate,s->combined);
    xe_rmsnorm_engine(s->engine,s->combined,l->post_norm,XE_EMBD,s->combined);
    float scale=l->layer_out_scale?*l->layer_out_scale:1.0f;
    for(int i=0;i<XE_EMBD;i++) s->hidden[i]=(s->hidden[i]+s->combined[i])*scale;
}


typedef struct {
    int mode;
} xe_output_arg;




static void xe_output_decode(xe_session *s,int softcap_mode) {
    (void)softcap_mode;
#ifdef XE_TEST_OUTPUT_COUNT
    xe_test_output_calls++;
#endif
    xe_rmsnorm_engine(s->engine,s->hidden,s->engine->out_norm,XE_EMBD,s->normed);
    const xe_q4 *w=s->engine->output.qs?&s->engine->output:&s->engine->tok_embd;
    xe_e4b_project(s,w,XE_EMBD,XE_VOCAB,s->normed,s->logits);
    for(int i=0;i<XE_VOCAB;i++) s->logits[i]=XE_LOGIT_SOFTCAP*tanhf(s->logits[i]/XE_LOGIT_SOFTCAP);
}


static void xe_decode_token_mode(xe_session *,int32_t,int,int,int);

static void xe_prefill_batch_append_run(xe_session *s,const int32_t *tokens,int rows,int start) {
    if(rows<1 || rows>512 || s->prefill_pending || start!=s->n_tokens ||
       rows>s->engine->context-start) xe_fatal("prefill: invalid batch");
#ifdef XE_TEST_SESSION
    extern size_t xe_test_prefill_batches;xe_test_prefill_batches++;
#endif
    /* Token-serial baseline prevents shared donor ring overwrite across a batch.
     * Projections run on GPU; attention/normalization/PLE lookup run on CPU. */
    s->gpu_prefill=1;
    for(int row=0;row<rows;row++) xe_decode_token_mode(s,tokens[row],start+row,XE_SOFTCAP_SECOND_LOOP,0);
    s->gpu_prefill=0;s->prefill_pending=1;s->prefill_pending_start=start;s->prefill_pending_rows=rows;
    s->prefill_pending_hidden=s->hidden;
}


static void xe_prefill_batch_complete(xe_session *s, int output_logits) {
    if (!s->prefill_pending) xe_fatal("prefill batch is not pending");
    /* Baseline prefill leaves its last token in the session workspace. */
    s->n_tokens = s->prefill_pending_start + s->prefill_pending_rows;
    s->prefill_pending = 0;
    s->prefill_pending_hidden = NULL;
    if (output_logits) {
        xe_workers_begin(s->engine);
        xe_output_decode(s, XE_SOFTCAP_SECOND_LOOP);
        xe_workers_end(s->engine);
    }
}

static void xe_prefill_batch_run(xe_session *s, const int32_t *tokens,
                                 int rows, int batch_start,
                                 int output_logits) {
    xe_prefill_batch_append_run(s, tokens, rows, batch_start);
    xe_ze_check("zeCommandListHostSynchronize prefill batch",
                zeCommandListHostSynchronize(s->engine->gpu.commands,
                                             UINT64_MAX));
    xe_prefill_batch_complete(s, output_logits);
}

static void __attribute__((unused)) xe_prefill_initial_run(
        xe_session *s, const int32_t *tokens, int rows, int output_logits) {
    if (s->n_tokens != 0) xe_fatal("prefill initial requires an empty session");
    xe_prefill_batch_run(s, tokens, rows, 0, output_logits);
}

static void xe_decode_token_mode(xe_session *s,int32_t token,int pos,int softcap_mode,int output_logits) {
    if(token<0 || token>=XE_VOCAB || pos<0 || pos>=s->engine->context || pos!=s->n_tokens)
        xe_fatal("decode: invalid token/position");
#ifdef XE_TEST_SESSION
    extern size_t xe_test_decode_tokens;xe_test_decode_tokens++;
#endif
    xe_embed_decode(s->engine,token,s->hidden);xe_rope_prepare(s,pos);
    xe_workers_begin(s->engine);
    xe_e4b_prepare_ple(s,token);
    for(int layer=0;layer<XE_LAYERS;layer++) {
        xe_attention_half_decode(s,layer,pos);xe_ffn_half_decode(s,layer);
        if(s->dump_layers && pos==s->dump_position) {
            char path[PATH_MAX];
            int n=snprintf(path,sizeof path,"%s/layer_%02d.f32",s->dump_layers,layer);
            if(n<0 || n>=(int)sizeof path) xe_fatal("layer dump path too long");
            FILE *f=fopen(path,"wb");if(!f) xe_fatal("%s: %s",path,strerror(errno));
            if(fwrite(s->hidden,sizeof(float),XE_EMBD,f)!=XE_EMBD) xe_fatal("layer dump short write");
            if(fclose(f)) xe_fatal("layer dump close failed");
        }
    }
    if(output_logits) xe_output_decode(s,softcap_mode);
    xe_workers_end(s->engine);s->tokens[pos]=token;s->n_tokens=pos+1;
}


static void __attribute__((unused)) xe_decode_token(
        xe_session *s, int32_t token, int pos) {
    xe_decode_token_mode(s, token, pos, 
                         XE_SOFTCAP_SECOND_LOOP, 1);
}

static int xe_sample_better(xe_sample_candidate a, xe_sample_candidate b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

static void xe_sample_heap_up(xe_sample_candidate *heap, int at) {
    while (at > 0) {
        int parent = (at - 1) / 2;
        if (!xe_sample_better(heap[parent], heap[at])) break;
        xe_sample_candidate tmp = heap[parent];
        heap[parent] = heap[at];
        heap[at] = tmp;
        at = parent;
    }
}

static void xe_sample_heap_down(xe_sample_candidate *heap, int n, int at) {
    for (;;) {
        int left = 2 * at + 1;
        int right = left + 1;
        int worst = at;
        if (left < n && xe_sample_better(heap[worst], heap[left])) worst = left;
        if (right < n && xe_sample_better(heap[worst], heap[right])) worst = right;
        if (worst == at) break;
        xe_sample_candidate tmp = heap[at];
        heap[at] = heap[worst];
        heap[worst] = tmp;
        at = worst;
    }
}

static int xe_sample_compare(const void *pa, const void *pb) {
    const xe_sample_candidate *a = pa;
    const xe_sample_candidate *b = pb;
    if (a->logit < b->logit) return 1;
    if (a->logit > b->logit) return -1;
    return (a->id > b->id) - (a->id < b->id);
}

static uint64_t xe_sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = UINT64_C(0x9e3779b97f4a7c15);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(0x2545f4914f6cdd1d);
}

static float xe_sample_rng_f32(uint64_t *state) {
    uint64_t x = xe_sample_rng_next(state);
    return (float)((x >> 40) & UINT64_C(0xffffff)) / 16777216.0f;
}

static int32_t xe_sample_logits(const float *logits, int n_vocab,
                                xe_sample_candidate *candidates, xe_sampler *sp) {
    if (!logits || !candidates || !sp) xe_fatal("sample: missing input");
    if (!isfinite(sp->temperature) || sp->temperature < 0.0f)
        xe_fatal("sample: temperature must be finite and non-negative");
    if (sp->top_k < 0 || sp->top_k > n_vocab)
        xe_fatal("sample: top-k %d out of range [0, %d]", sp->top_k, n_vocab);
    if (!isfinite(sp->top_p) || sp->top_p <= 0.0f || sp->top_p > 1.0f)
        xe_fatal("sample: top-p must be in (0, 1]");

    int32_t best = 0;
    float best_logit = -INFINITY;
    for (int32_t id = 0; id < n_vocab; id++) {
        float v = logits[id];
        if (!isfinite(v)) xe_fatal("sample: logit %d is not finite (%g)", id, (double)v);
        if (v > best_logit) {
            best_logit = v;
            best = id;
        }
    }
    if (sp->temperature == 0.0f || sp->top_k == 1) return best;

    int n = 0;
    if (sp->top_k == 0) {
        for (int32_t id = 0; id < n_vocab; id++)
            candidates[n++] = (xe_sample_candidate){ id, logits[id], 0.0f };
    } else {
        for (int32_t id = 0; id < n_vocab; id++) {
            xe_sample_candidate candidate = { id, logits[id], 0.0f };
            if (n < sp->top_k) {
                candidates[n] = candidate;
                xe_sample_heap_up(candidates, n);
                n++;
            } else if (xe_sample_better(candidate, candidates[0])) {
                candidates[0] = candidate;
                xe_sample_heap_down(candidates, n, 0);
            }
        }
    }

    if (sp->top_k != 0 || sp->top_p < 1.0f)
        qsort(candidates, (size_t)n, sizeof(*candidates), xe_sample_compare);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = expf((candidates[i].logit - best_logit) / sp->temperature);
        candidates[i].prob = p;
        sum += p;
    }
    if (!(sum > 0.0f) || !isfinite(sum)) xe_fatal("sample: invalid probability sum %g", (double)sum);

    int filtered = n;
    float filtered_sum = sum;
    if (sp->top_p < 1.0f) {
        filtered = 0;
        filtered_sum = 0.0f;
        float target = sp->top_p * sum;
        do {
            filtered_sum += candidates[filtered].prob;
            filtered++;
        } while (filtered < n && filtered_sum < target);
    }

    float draw = xe_sample_rng_f32(&sp->rng_state) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        draw -= candidates[i].prob;
        if (draw <= 0.0f) return candidates[i].id;
    }
    return candidates[filtered - 1].id;
}

static void xe_session_swa_copy_slot(xe_session *s, int slot) {
    size_t bytes = XE_SWA_HEAD_DIM * sizeof(*s->swa_k);
    for (int layer = 0; layer < XE_SWA_LAYERS; layer++) {
        size_t layer_offset = (size_t)layer * XE_SWA_LAYER_ELEMS;
        for (int head = 0; head < XE_SWA_KV_HEADS; head++) {
            size_t offset = layer_offset +
                ((size_t)head * XE_SWA_WINDOW + slot) * XE_SWA_HEAD_DIM;
            memcpy(s->swa_spare_k + offset, s->swa_k + offset, bytes);
            memcpy(s->swa_spare_v + offset, s->swa_v + offset, bytes);
        }
    }
    s->swa_dirty[slot] = 0;
}

static void xe_session_swa_mark(xe_session *s, int first, int end) {
    int rows = end - first;
    if (rows <= 0) return;
    if (rows >= XE_SWA_WINDOW) {
        memset(s->swa_dirty, 1, sizeof s->swa_dirty);
        return;
    }
    for (int pos = first; pos < end; pos++)
        s->swa_dirty[pos & (XE_SWA_WINDOW - 1)] = 1;
}

static void xe_session_swa_mirror(xe_session *s, int first, int end) {
    if (!s->swa_spare_k || !s->swa_spare_v) return;
    if (end - first > XE_SWA_WINDOW) first = end - XE_SWA_WINDOW;
    for (int pos = first; pos < end; pos++)
        xe_session_swa_copy_slot(s, pos & (XE_SWA_WINDOW - 1));
}

static void xe_session_swa_repair(xe_session *s) {
    int first = s->n_tokens > XE_SWA_WINDOW
                ? s->n_tokens - XE_SWA_WINDOW : 0;
    for (int pos = first; pos < s->n_tokens; pos++) {
        int slot = pos & (XE_SWA_WINDOW - 1);
        if (s->swa_dirty[slot]) xe_session_swa_copy_slot(s, slot);
    }
}

xe_session *xe_session_new(xe_engine *e) {
    if (!e || e->context < XE_CONTEXT_MIN || e->context > XE_CONTEXT_MAX)
        xe_fatal("session: invalid engine context capacity");
    if (e->vocab_only) xe_fatal("session: engine opened vocab-only, no weights available");
    xe_worker_pool_init(e);
    xe_session *s = xe_alloc(e, sizeof *s, XE_MEM_HOST);
    memset(s, 0, sizeof *s);
    s->engine = e;
    s->swa_k = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_k), XE_MEM_SHARED);
    s->swa_v = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_v), XE_MEM_SHARED);
    s->swa_spare_k = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_spare_k),
                              XE_MEM_SHARED);
    s->swa_spare_v = xe_alloc(e, XE_SWA_SLAB_ELEMS * sizeof(*s->swa_spare_v),
                              XE_MEM_SHARED);
    memset(s->swa_dirty, 1, sizeof s->swa_dirty);
    s->global_k = xe_alloc(e, xe_global_slab_elems(e) * sizeof(*s->global_k), XE_MEM_SHARED);
    s->global_v = xe_alloc(e, xe_global_slab_elems(e) * sizeof(*s->global_v), XE_MEM_SHARED);
    s->tokens = xe_alloc(e, (size_t)e->context * sizeof(*s->tokens), XE_MEM_HOST);
    s->workspace_size = xe_workspace_layout(s, NULL);
    s->workspace = xe_alloc(e, s->workspace_size, XE_MEM_SHARED);
    xe_workspace_layout(s, s->workspace);
    return s;
}

xe_session *xe_session_shadow_new(xe_session *source) {
    if (!source || source->cow || source->n_tokens <= 0 ||
        !source->swa_spare_k || !source->swa_spare_v)
        return NULL;
    xe_require_owner(source->engine);
    xe_session_swa_repair(source);
    xe_engine *e = source->engine;
    xe_session *s = xe_alloc(e, sizeof *s, XE_MEM_HOST);
    memset(s, 0, sizeof *s);
    s->engine = e;
    s->cow = 1;
    s->cow_source = source;
    s->cow_split = source->n_tokens;
    s->global_k = source->global_k;
    s->global_v = source->global_v;
    s->swa_k = source->swa_spare_k;
    s->swa_v = source->swa_spare_v;
    source->swa_spare_k = NULL;
    source->swa_spare_v = NULL;
    s->tokens = xe_alloc(e, (size_t)e->context * sizeof(*s->tokens), XE_MEM_HOST);
    s->workspace_size = xe_workspace_layout(s, NULL);
    s->workspace = xe_alloc(e, s->workspace_size, XE_MEM_SHARED);
    xe_workspace_layout(s, s->workspace);
    memcpy(s->tokens, source->tokens,
           (size_t)source->n_tokens * sizeof(*s->tokens));
    memcpy(s->hidden, source->hidden, XE_EMBD * sizeof(*s->hidden));
    memcpy(s->logits, source->logits, XE_VOCAB * sizeof(*s->logits));
    s->n_tokens = source->n_tokens;
    return s;
}

void xe_session_free(xe_session *s) {
    if (!s) return;
    xe_require_owner(s->engine);
    xe_engine *e = s->engine;
    if (s->prefill_pending)
        xe_ze_check("zeCommandListHostSynchronize free pending prefill",
                    zeCommandListHostSynchronize(e->gpu.commands,
                                                 UINT64_MAX));
    if (s->cow && s->cow_source && !s->cow_source->swa_spare_k &&
        !s->cow_source->swa_spare_v) {
        xe_session *source = s->cow_source;
        source->swa_spare_k = s->swa_k;
        source->swa_spare_v = s->swa_v;
        s->swa_k = NULL;
        s->swa_v = NULL;
        int end = source->n_tokens > s->n_tokens
                  ? source->n_tokens : s->n_tokens;
        xe_session_swa_mark(source, s->cow_split, end);
    }
    xe_free(e, s->anchor_logits, XE_MEM_HOST);
    xe_free(e, s->anchor_swa_v, XE_MEM_HOST);
    xe_free(e, s->anchor_swa_k, XE_MEM_HOST);
    xe_free(e, s->workspace, XE_MEM_SHARED);
    xe_free(e, s->tokens, XE_MEM_HOST);
    if (s->cow) {
        for (int i = 0; i < XE_GLOBAL_LAYERS; i++) {
            xe_free(e, s->cow_global_v[i], XE_MEM_SHARED);
            xe_free(e, s->cow_global_k[i], XE_MEM_SHARED);
        }
    } else {
        xe_free(e, s->global_v, XE_MEM_SHARED);
        xe_free(e, s->global_k, XE_MEM_SHARED);
    }
    xe_free(e, s->swa_v, XE_MEM_SHARED);
    xe_free(e, s->swa_k, XE_MEM_SHARED);
    xe_free(e, s->swa_spare_v, XE_MEM_SHARED);
    xe_free(e, s->swa_spare_k, XE_MEM_SHARED);
    xe_free(e, s, XE_MEM_HOST);
}

static void xe_session_shadow_reserve(xe_session *s, int end) {
    if (!s->cow || end < s->cow_split || end > s->engine->context)
        xe_fatal("shadow reserve: invalid end %d", end);
    int needed = end - s->cow_split;
    if (needed <= s->cow_capacity) return;
    int capacity = (needed + 511) & ~511;
    int rows = s->n_tokens - s->cow_split;
    for (int layer = 0; layer < XE_GLOBAL_LAYERS; layer++) {
        size_t elements = (size_t)XE_GLOBAL_KV_HEADS * capacity *
                          XE_GLOBAL_HEAD_DIM;
        _Float16 *next_k = xe_alloc(s->engine,
                                    elements * sizeof(*next_k),
                                    XE_MEM_SHARED);
        _Float16 *next_v = xe_alloc(s->engine,
                                    elements * sizeof(*next_v),
                                    XE_MEM_SHARED);
        if (s->cow_capacity) {
            size_t bytes = (size_t)rows * XE_GLOBAL_HEAD_DIM *
                           sizeof(*next_k);
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                memcpy(next_k + (size_t)head * capacity *
                                XE_GLOBAL_HEAD_DIM,
                       s->cow_global_k[layer] +
                           (size_t)head * s->cow_capacity *
                           XE_GLOBAL_HEAD_DIM,
                       bytes);
                memcpy(next_v + (size_t)head * capacity *
                                XE_GLOBAL_HEAD_DIM,
                       s->cow_global_v[layer] +
                           (size_t)head * s->cow_capacity *
                           XE_GLOBAL_HEAD_DIM,
                       bytes);
            }
        }
        xe_free(s->engine, s->cow_global_k[layer], XE_MEM_SHARED);
        xe_free(s->engine, s->cow_global_v[layer], XE_MEM_SHARED);
        s->cow_global_k[layer] = next_k;
        s->cow_global_v[layer] = next_v;
    }
    s->cow_capacity = capacity;
}

uint64_t xe_session_shadow_kv_bytes(const xe_session *s) {
    if (!s || !s->cow) return 0;
    uint64_t swa = UINT64_C(2) * XE_SWA_SLAB_ELEMS * sizeof(_Float16);
    uint64_t global = UINT64_C(2) * XE_GLOBAL_LAYERS *
                      XE_GLOBAL_KV_HEADS * (uint64_t)s->cow_capacity *
                      XE_GLOBAL_HEAD_DIM * sizeof(_Float16);
    return swa + global;
}

int xe_session_shadow_split(const xe_session *s) {
    return s && s->cow ? s->cow_split : 0;
}

void xe_session_reset(xe_session *s) {
    if (!s) xe_fatal("session_reset: missing session");
    xe_require_owner(s->engine);
    s->n_tokens = 0;
}

int xe_session_position(xe_session *s) {
    if (!s) xe_fatal("session_position: missing session");
    xe_require_owner(s->engine);
    return s->n_tokens;
}

int xe_session_context_size(const xe_session *s) {
    if (!s) xe_fatal("session_context_size: missing session");
    return xe_context_size(s->engine);
}

static int xe_session_swa_can_resume(int current, int resume) {
    if (resume <= 0) return 1;
    int oldest = current > XE_SWA_WINDOW ? current - XE_SWA_WINDOW : 0;
    int needed = resume >= XE_SWA_WINDOW
                 ? resume - XE_SWA_WINDOW + 1 : 0;
    return needed >= oldest;
}

static void xe_session_extend(xe_session *s, const int32_t *tokens, int end,
                              int cpu_single) {
    int mirror_start = s->n_tokens;
    while (s->n_tokens < end) {
        int start = s->n_tokens;
        int rows = end - start;
        if (rows > 512) rows = 512;
        if (rows == 1 && cpu_single) {
            int pos = s->n_tokens;
            xe_decode_token_mode(s, tokens[pos], pos,
                                 
                                 XE_SOFTCAP_SECOND_LOOP, 1);
            s->tokens[pos] = tokens[pos];
            continue;
        }
        xe_prefill_batch_run(s, tokens + start, rows, start,
                             start + rows == end);
    }
    if (!s->cow) xe_session_swa_mirror(s, mirror_start, s->n_tokens);
}

void xe_session_rewind(xe_session *s, int position) {
    if (!s) xe_fatal("session_rewind: missing session");
    xe_require_owner(s->engine);
    if (position < 0 || position > s->n_tokens)
        xe_fatal("session_rewind: position %d out of range [0, %d]",
                 position, s->n_tokens);
    if (position == 0) {
        s->n_tokens = 0;
        return;
    }
    xe_tokens prefix = { s->tokens, position, s->engine->context };
    xe_session_sync(s, &prefix);
}

void xe_session_sync_report(xe_session *s, const xe_tokens *prefix,
                            xe_sync_report *report) {
    if (!s || !prefix) xe_fatal("session_sync: missing session or prefix");
    if (!xe_tokens_valid(prefix)) xe_fatal("session_sync: invalid prefix");
    xe_require_owner(s->engine);
    int n = prefix->len;
    if (n <= 0 || n > s->engine->context)
        xe_fatal("session_sync: prefix length %d out of range [1, %d]", n, s->engine->context);
    for (int i = 0; i < n; i++)
        if (prefix->v[i] < 0 || prefix->v[i] >= XE_VOCAB)
            xe_fatal("session_sync: token %d at position %d out of range [0, %d]",
                     prefix->v[i], i, XE_VOCAB - 1);

    int current = s->n_tokens;
    int common_limit = n < current ? n : current;
    int common = 0;
    while (common < common_limit && prefix->v[common] == s->tokens[common])
        common++;
    if (report) {
        report->reused = 0;
        report->prefilled = 0;
        report->restarted = 0;
    }
    if (common == n && common == current) {
        if (report) report->reused = n;
        return;
    }

    if (common < current) {
        int resume = common < n ? common : n - 1;
        if (xe_session_swa_can_resume(current, resume)) {
            s->n_tokens = resume;
        } else {
            s->n_tokens = 0;
            if (report) report->restarted = 1;
        }
    }
    if (report) {
        report->reused = s->n_tokens;
        report->prefilled = n - s->n_tokens;
    }
    xe_session_extend(s, prefix->v, n,
                      common == current && n == current + 1);
}

void xe_session_sync(xe_session *s, const xe_tokens *prefix) {
    xe_session_sync_report(s, prefix, NULL);
}

int xe_session_shadow_start(xe_session *s, const xe_tokens *prefix,
                            int max_rows) {
    if (!s || !s->cow || !prefix || !xe_tokens_valid(prefix) ||
        max_rows < 1 || max_rows > 512 || s->prefill_pending)
        return -1;
    xe_require_owner(s->engine);
    if (prefix->len < s->cow_split || prefix->len > s->engine->context)
        return -1;
    int limit = prefix->len < s->n_tokens ? prefix->len : s->n_tokens;
    int common = 0;
    while (common < limit && prefix->v[common] == s->tokens[common])
        common++;
    if (common < s->cow_split) return -1;
    if (common < s->n_tokens) {
        if (!xe_session_swa_can_resume(s->n_tokens, common)) return -1;
        s->n_tokens = common;
    }
    if (s->n_tokens >= prefix->len) return 0;
    xe_session_shadow_reserve(s, prefix->len);
    int rows = prefix->len - s->n_tokens;
    if (rows > max_rows) rows = max_rows;
    xe_prefill_batch_append_run(s, prefix->v + s->n_tokens,
                                rows, s->n_tokens);
    return rows;
}

int xe_session_shadow_poll(xe_session *s) {
    if (!s || !s->cow || !s->prefill_pending) return 0;
    xe_require_owner(s->engine);
    ze_result_t result = zeCommandListHostSynchronize(s->engine->gpu.commands,
                                                      0);
    if (result == ZE_RESULT_NOT_READY) return 0;
    xe_ze_check("zeCommandListHostSynchronize shadow poll", result);
    int rows = s->prefill_pending_rows;
    xe_prefill_batch_complete(s, 0);
    return rows;
}

int xe_session_shadow_wait(xe_session *s) {
    if (!s || !s->cow || !s->prefill_pending) return 0;
    xe_require_owner(s->engine);
    xe_ze_check("zeCommandListHostSynchronize shadow wait",
                zeCommandListHostSynchronize(s->engine->gpu.commands,
                                             UINT64_MAX));
    int rows = s->prefill_pending_rows;
    xe_prefill_batch_complete(s, 0);
    return rows;
}

int xe_session_shadow_sync(xe_session *s, const xe_tokens *prefix) {
    if (!s || !s->cow || !prefix) return -1;
    int prefilled = 0;
    int completed = xe_session_shadow_wait(s);
    if (completed > 0) prefilled += completed;
    for (;;) {
        int rows = xe_session_shadow_start(s, prefix, 512);
        if (rows < 0) return -1;
        if (rows == 0) return prefilled;
        completed = xe_session_shadow_wait(s);
        if (completed != rows) return -1;
        prefilled += completed;
    }
}

void xe_session_shadow_refresh_logits(xe_session *s) {
    if (!s || !s->cow || s->prefill_pending || s->n_tokens <= 0)
        xe_fatal("shadow logits: invalid session");
    xe_require_owner(s->engine);
    xe_workers_begin(s->engine);
    xe_output_decode(s, XE_SOFTCAP_SECOND_LOOP);
    xe_workers_end(s->engine);
}

int xe_session_shadow_promote(xe_session *source, xe_session *shadow) {
    if (!source || !shadow || source->cow || !shadow->cow ||
        source->engine != shadow->engine || shadow->prefill_pending ||
        shadow->cow_source != source || source->swa_spare_k ||
        source->swa_spare_v ||
        shadow->n_tokens < shadow->cow_split ||
        source->global_k != shadow->global_k ||
        source->global_v != shadow->global_v)
        return 0;
    xe_require_owner(source->engine);
    xe_engine *e = source->engine;
    int source_position = source->n_tokens;
    int rows = shadow->n_tokens - shadow->cow_split;
    size_t row_bytes = (size_t)rows * XE_GLOBAL_HEAD_DIM *
                       sizeof(_Float16);
    if (rows > 0) {
        for (int layer = 0; layer < XE_GLOBAL_LAYERS; layer++) {
            _Float16 *target_k = source->global_k +
                (size_t)layer * xe_global_layer_elems(e);
            _Float16 *target_v = source->global_v +
                (size_t)layer * xe_global_layer_elems(e);
            for (int head = 0; head < XE_GLOBAL_KV_HEADS; head++) {
                size_t target_offset =
                    ((size_t)head * e->context + shadow->cow_split) *
                    XE_GLOBAL_HEAD_DIM;
                size_t source_offset =
                    (size_t)head * shadow->cow_capacity * XE_GLOBAL_HEAD_DIM;
                xe_ze_check("zeCommandListAppendMemoryCopy promote K",
                            zeCommandListAppendMemoryCopy(
                                e->gpu.commands, target_k + target_offset,
                                shadow->cow_global_k[layer] + source_offset,
                                row_bytes, NULL, 0, NULL));
                xe_ze_check("zeCommandListAppendMemoryCopy promote V",
                            zeCommandListAppendMemoryCopy(
                                e->gpu.commands, target_v + target_offset,
                                shadow->cow_global_v[layer] + source_offset,
                                row_bytes, NULL, 0, NULL));
            }
        }
    }
    xe_ze_check("zeCommandListAppendMemoryCopy promote hidden",
                zeCommandListAppendMemoryCopy(
                    e->gpu.commands, source->hidden, shadow->hidden,
                    XE_EMBD * sizeof(*source->hidden), NULL, 0, NULL));
    xe_ze_check("zeCommandListAppendMemoryCopy promote logits",
                zeCommandListAppendMemoryCopy(
                    e->gpu.commands, source->logits, shadow->logits,
                    XE_VOCAB * sizeof(*source->logits), NULL, 0, NULL));
    xe_ze_check("zeCommandListHostSynchronize promote",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    source->swa_spare_k = source->swa_k;
    source->swa_spare_v = source->swa_v;
    source->swa_k = shadow->swa_k;
    source->swa_v = shadow->swa_v;
    shadow->swa_k = NULL;
    shadow->swa_v = NULL;
    shadow->cow_source = NULL;
    int end = source_position > shadow->n_tokens
              ? source_position : shadow->n_tokens;
    xe_session_swa_mark(source, shadow->cow_split, end);
    memcpy(source->tokens, shadow->tokens,
           (size_t)shadow->n_tokens * sizeof(*source->tokens));
    source->n_tokens = shadow->n_tokens;
    source->anchor_valid = 0;
    return 1;
}

int xe_session_common(const xe_session *s, const xe_tokens *prefix) {
    if (!s || !prefix) xe_fatal("session_common: missing session or prefix");
    if (!xe_tokens_valid(prefix)) xe_fatal("session_common: invalid prefix");
    int limit = prefix->len < s->n_tokens ? prefix->len : s->n_tokens;
    int common = 0;
    while (common < limit && prefix->v[common] == s->tokens[common])
        common++;
    return common;
}

const float *xe_session_logits(xe_session *s) {
    if (!s) xe_fatal("session_logits: missing session");
    xe_require_owner(s->engine);
    if (s->n_tokens == 0) xe_fatal("session_logits: session has no evaluated tokens");
    return s->logits;
}

int32_t xe_session_next(xe_session *s, xe_sampler *sp) {
    const float *logits = xe_session_logits(s);
    return xe_sample_logits(logits, XE_VOCAB, s->sample_candidates, sp);
}

int xe_session_anchor_capture(xe_session *s) {
    if (!s || s->n_tokens <= 0) return 0;
    xe_require_owner(s->engine);
    if (!s->anchor_swa_k) {
        s->anchor_swa_k = xe_alloc(s->engine,
            XE_SWA_SLAB_ELEMS * sizeof(*s->anchor_swa_k), XE_MEM_HOST);
        s->anchor_swa_v = xe_alloc(s->engine,
            XE_SWA_SLAB_ELEMS * sizeof(*s->anchor_swa_v), XE_MEM_HOST);
        s->anchor_logits = xe_alloc(s->engine,
            XE_VOCAB * sizeof(*s->anchor_logits), XE_MEM_HOST);
    }
    memcpy(s->anchor_swa_k, s->swa_k,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_k));
    memcpy(s->anchor_swa_v, s->swa_v,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_v));
    memcpy(s->anchor_logits, s->logits,
           XE_VOCAB * sizeof(*s->logits));
    s->anchor_position = s->n_tokens;
    s->anchor_valid = 1;
    return 1;
}

int xe_session_anchor_restore(xe_session *s) {
    if (!s || !s->anchor_valid) return 0;
    xe_require_owner(s->engine);
    memcpy(s->swa_k, s->anchor_swa_k,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_k));
    memcpy(s->swa_v, s->anchor_swa_v,
           XE_SWA_SLAB_ELEMS * sizeof(*s->swa_v));
    memcpy(s->logits, s->anchor_logits,
           XE_VOCAB * sizeof(*s->logits));
    s->n_tokens = s->anchor_position;
    if (s->swa_spare_k && s->swa_spare_v)
        memset(s->swa_dirty, 1, sizeof s->swa_dirty);
    return s->anchor_position;
}

void xe_session_anchor_clear(xe_session *s) {
    if (s) s->anchor_valid = 0;
}

int xe_session_anchor_valid(const xe_session *s) {
    return s && s->anchor_valid;
}

int xe_session_token_near_top(xe_session *s, int32_t token,
                              int max_rank, float max_margin) {
    const float *logits = xe_session_logits(s);
    if (token < 0 || token >= XE_VOCAB || max_rank < 1) return 0;
    float value = logits[token];
    float best = value;
    int rank = 1;
    for (int i = 0; i < XE_VOCAB; i++) {
        if (logits[i] > best) best = logits[i];
        if (logits[i] > value) rank++;
    }
    return rank <= max_rank && best - value <= max_margin;
}

enum {
    XE_SNAPSHOT_FORMAT_VERSION = 2,
    /* State compatibility, not build identity. Bump when persisted KV/logits
     * can no longer be continued under the same numerical contract. Review
     * inference changes explicitly; edits to I/O, CLI or docs do not bump it.
     * Supporting a new ABI also requires reviewing the legacy reader below. */
    XE_SNAPSHOT_NUMERIC_ABI = 1,
    XE_SNAPSHOT_HEADER_SIZE = 384,
    XE_SNAPSHOT_ENTRY_SIZE = 40,
    XE_SNAPSHOT_REQUIRED_SECTIONS = 6,
    XE_SNAPSHOT_MAX_SECTIONS = 64,
    XE_SNAPSHOT_BUFFER_SIZE = 8 * 1024 * 1024,
    XE_SNAPSHOT_SECTION_REQUIRED = 1
};

enum {
    XE_SNAPSHOT_FP_MODEL,
    XE_SNAPSHOT_FP_TOKENIZER,
    XE_SNAPSHOT_FP_TEMPLATE,
    XE_SNAPSHOT_FP_LAYOUT,
    XE_SNAPSHOT_FP_CONTEXT,
    XE_SNAPSHOT_FP_NUMERIC,
    XE_SNAPSHOT_FP_COUNT
};

enum {
    XE_SNAPSHOT_TOKENS = 1,
    XE_SNAPSHOT_LOGITS,
    XE_SNAPSHOT_SWA_K,
    XE_SNAPSHOT_SWA_V,
    XE_SNAPSHOT_GLOBAL_K,
    XE_SNAPSHOT_GLOBAL_V
};

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t length;
    uint64_t count;
    uint64_t crc;
} xe_snapshot_section;

typedef struct {
    uint64_t file_size;
    uint64_t position;
    uint64_t swa_rows;
    uint64_t global_rows;
    uint64_t directory_offset;
    uint64_t directory_length;
    uint64_t payload_offset;
    uint32_t section_count;
    xe_snapshot_section sections[XE_SNAPSHOT_MAX_SECTIONS];
    int known[XE_SNAPSHOT_REQUIRED_SECTIONS];
} xe_snapshot_layout;

static const uint8_t xe_snapshot_magic[8] = {
    'X', 'E', 'N', 'X', 'K', 'V', '\r', '\n'
};

static void xe_snapshot_hash_u32(format_sha256 *hash, uint32_t value) {
    uint8_t bytes[4];
    format_put_u32le(bytes, value);
    format_sha256_update(hash, bytes, sizeof bytes);
}

static void xe_snapshot_hash_u64(format_sha256 *hash, uint64_t value) {
    uint8_t bytes[8];
    format_put_u64le(bytes, value);
    format_sha256_update(hash, bytes, sizeof bytes);
}

static void xe_snapshot_hash_string(format_sha256 *hash, const char *s) {
    size_t len = strlen(s);
    xe_snapshot_hash_u64(hash, len);
    format_sha256_update(hash, s, len);
}

/* The v1 hash is opaque: it does not encode a recoverable commit/compiler.
 * Recognize only the audited baseline, with this build's compiler, flags,
 * workers, RMS mode and compiled kernel. Other v1 producers remain rejected.
 * The source hash is Makefile's hash of `sha256sum xenolith.c xenolith.cl`
 * at 68b8fbf55787eda392c8cbfa32bab5ad14622e8a, not of the current sources. */
static int xe_snapshot_legacy_numeric_matches(const xe_engine *e,
                                               const uint8_t saved[32]) {
    if (XE_SNAPSHOT_NUMERIC_ABI != 1 ||
        !strcmp(XE_NUMERIC_CFLAGS_HASH, "unknown")) return 0;
    format_sha256 hash;
    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-numeric-v1",
                         sizeof("xenolith-numeric-v1") - 1);
    xe_snapshot_hash_u32(&hash, 1);
    xe_snapshot_hash_string(&hash, "68b8fbf55");
    xe_snapshot_hash_string(&hash,
        "528b6db377397368158a28dc155047628ac30a5a06c38aa6c32a5680ccd56d84");
    xe_snapshot_hash_string(&hash, XE_NUMERIC_CFLAGS_HASH);
    xe_snapshot_hash_string(&hash, __VERSION__);
    xe_snapshot_hash_u32(&hash, XE_WORKERS);
    xe_snapshot_hash_u32(&hash, e->scalar_rms);
    size_t spv_size = (size_t)(_binary_xenolith_gpu_spv_end -
                               _binary_xenolith_gpu_spv_start);
    xe_snapshot_hash_u64(&hash, spv_size);
    format_sha256_update(&hash, _binary_xenolith_gpu_spv_start, spv_size);
    uint8_t expected[32];
    format_sha256_final(&hash, expected);
    return memcmp(saved, expected, sizeof expected) == 0;
}

static void xe_snapshot_fingerprints_init(xe_engine *e) {
    if (e->snapshot_fingerprint_ready) return;
    format_sha256 hash;

    /* Raw SHA-256 of all GGUF bytes, computed once per engine. Older
     * header-only MODEL identities intentionally fail as model-mismatch:
     * their tensor contents cannot be established retroactively. The wire
     * layout and numerical ABI are unchanged. */
    memcpy(e->snapshot_fingerprint[XE_SNAPSHOT_FP_MODEL],
           xe_engine_model_fingerprint(e), 32);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-tokenizer-v1",
                         sizeof("xenolith-tokenizer-v1") - 1);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->bos_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->eos_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->eot_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->unk_id);
    xe_snapshot_hash_u32(&hash, (uint32_t)e->pad_id);
    xe_snapshot_hash_u64(&hash, e->tok_tokens_count);
    for (uint64_t i = 0; i < e->tok_tokens_count; i++) {
        const xe_str *piece = &e->tok_piece[i];
        xe_snapshot_hash_u64(&hash, piece->len);
        format_sha256_update(&hash, piece->p, (size_t)piece->len);
    }
    xe_snapshot_hash_u64(&hash, e->tok_scores_count);
    if (e->tok_scores_count)
        format_sha256_update(&hash, e->tok_scores,
                             (size_t)e->tok_scores_count * sizeof(float));
    xe_snapshot_hash_u64(&hash, e->tok_token_type_count);
    if (e->tok_token_type_count)
        format_sha256_update(&hash, e->tok_token_type,
                             (size_t)e->tok_token_type_count * sizeof(int32_t));
    xe_snapshot_hash_u64(&hash, e->tok_merges_count);
    if (e->tok_merges_count) {
        xe_cur cursor = xe_cur_at(e, e->tok_merges);
        for (uint64_t i = 0; i < e->tok_merges_count; i++) {
            xe_str merge = xe_read_str(&cursor);
            xe_snapshot_hash_u64(&hash, merge.len);
            format_sha256_update(&hash, merge.p, (size_t)merge.len);
        }
    }
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_TOKENIZER]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-template-v1",
                         sizeof("xenolith-template-v1") - 1);
    xe_snapshot_hash_u64(&hash, e->chat_template.len);
    if (e->chat_template.len)
        format_sha256_update(&hash, e->chat_template.p,
                             (size_t)e->chat_template.len);
    xe_snapshot_hash_string(&hash, "xe_profile_render/v1");
    xe_snapshot_hash_u32(&hash, XE_CHANNEL_BEGIN_ID);
    xe_snapshot_hash_u32(&hash, XE_CHANNEL_END_ID);
    xe_snapshot_hash_u32(&hash, XE_TURN_BEGIN_ID);
    xe_snapshot_hash_u32(&hash, XE_TURN_END_ID);
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_TEMPLATE]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-e4b-kv-layout-v2",
                         sizeof("xenolith-e4b-kv-layout-v2") - 1);
    xe_snapshot_hash_u32(&hash, XE_LAYERS);
    xe_snapshot_hash_u32(&hash, XE_SWA_WINDOW);
    xe_snapshot_hash_u32(&hash, XE_SWA_KV_HEADS);
    xe_snapshot_hash_u32(&hash, XE_SWA_HEAD_DIM);
    xe_snapshot_hash_u32(&hash, XE_GLOBAL_KV_HEADS);
    xe_snapshot_hash_u32(&hash, XE_GLOBAL_HEAD_DIM);
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        xe_snapshot_hash_u32(&hash, XE_IS_GLOBAL(layer));
        xe_snapshot_hash_u32(&hash, xe_gemma4_kv_donor(e->model,(uint32_t)layer));
    }
    xe_snapshot_hash_u32(&hash, sizeof(_Float16));
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_LAYOUT]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-context-v1",
                         sizeof("xenolith-context-v1") - 1);
    xe_snapshot_hash_u32(&hash, e->context);
    xe_snapshot_hash_u32(&hash, XE_EMBD);
    xe_snapshot_hash_u32(&hash, XE_VOCAB);
    xe_snapshot_hash_u32(&hash, XE_Q_HEADS);
    xe_snapshot_hash_u32(&hash, XE_KV_LAYERS);
    xe_snapshot_hash_u32(&hash, XE_PLE);
    xe_snapshot_hash_u32(&hash, XE_DENSE_FFN);
    uint32_t bits;
    memcpy(&bits, &(float){XE_SWA_ROPE_BASE}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    memcpy(&bits, &(float){XE_GLOBAL_ROPE_BASE}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    memcpy(&bits, &(float){XE_RMS_EPS}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    memcpy(&bits, &(float){XE_LOGIT_SOFTCAP}, sizeof bits);
    xe_snapshot_hash_u32(&hash, bits);
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_CONTEXT]);

    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-numeric-abi-v1",
                         sizeof("xenolith-numeric-abi-v1") - 1);
    xe_snapshot_hash_u32(&hash, XE_SNAPSHOT_NUMERIC_ABI);
    xe_snapshot_hash_u32(&hash, XE_WORKERS);
    xe_snapshot_hash_u32(&hash, e->scalar_rms);
#ifdef __FAST_MATH__
    xe_snapshot_hash_u32(&hash, 1);
#else
    xe_snapshot_hash_u32(&hash, 0);
#endif
    format_sha256_final(&hash,
                        e->snapshot_fingerprint[XE_SNAPSHOT_FP_NUMERIC]);
    e->snapshot_fingerprint_ready = 1;
}

static void xe_snapshot_boundary_hash(const int32_t *tokens, int n,
                                      uint8_t out[32]) {
    format_sha256 hash;
    format_sha256_init(&hash);
    format_sha256_update(&hash, "xenolith-boundary-v1",
                         sizeof("xenolith-boundary-v1") - 1);
    xe_snapshot_hash_u64(&hash, (uint64_t)n);
    for (int i = 0; i < n; i++) xe_snapshot_hash_u32(&hash, (uint32_t)tokens[i]);
    format_sha256_final(&hash, out);
}

static int xe_snapshot_expected_valid(const xe_tokens *tokens) {
    if (!xe_tokens_valid(tokens) || tokens->len <= 0) return 0;
    for (int i = 0; i < tokens->len; i++)
        if (tokens->v[i] < 0 || tokens->v[i] >= XE_VOCAB) return 0;
    return 1;
}

static int xe_snapshot_layout_build(int capacity, uint64_t position,
                                    xe_snapshot_layout *layout) {
    if (!position || position > (uint64_t)capacity) return 0;
    memset(layout, 0, sizeof *layout);
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++)
        layout->known[i] = i;
    layout->section_count = XE_SNAPSHOT_REQUIRED_SECTIONS;
    layout->position = position;
    layout->swa_rows = position < XE_SWA_WINDOW ? position : XE_SWA_WINDOW;
    layout->global_rows = position;
    layout->directory_offset = XE_SNAPSHOT_HEADER_SIZE;
    layout->directory_length = (uint64_t)XE_SNAPSHOT_REQUIRED_SECTIONS *
                               XE_SNAPSHOT_ENTRY_SIZE;
    uint64_t end;
    if (!format_add_u64(layout->directory_offset, layout->directory_length,
                        &end) ||
        !format_align_u64(end, 64, &layout->payload_offset)) return 0;

    uint64_t lengths[XE_SNAPSHOT_REQUIRED_SECTIONS];
    uint64_t counts[XE_SNAPSHOT_REQUIRED_SECTIONS];
    counts[0] = position;
    counts[1] = XE_VOCAB;
    if (!format_mul_u64(position, 4, &lengths[0]) ||
        !format_mul_u64(XE_VOCAB, 4, &lengths[1]) ||
        !format_mul_u64(XE_SWA_LAYERS * XE_SWA_KV_HEADS * XE_SWA_HEAD_DIM,
                        layout->swa_rows, &counts[2]) ||
        !format_mul_u64(counts[2], sizeof(_Float16), &lengths[2]) ||
        !format_mul_u64(XE_GLOBAL_LAYERS * XE_GLOBAL_KV_HEADS * XE_GLOBAL_HEAD_DIM,
                        position, &counts[4]) ||
        !format_mul_u64(counts[4], sizeof(_Float16), &lengths[4])) return 0;
    counts[3] = counts[2];
    lengths[3] = lengths[2];
    counts[5] = counts[4];
    lengths[5] = lengths[4];

    uint64_t offset = layout->payload_offset;
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++) {
        if (!format_align_u64(offset, 64, &offset)) return 0;
        layout->sections[i].type = (uint32_t)i + 1;
        layout->sections[i].flags = XE_SNAPSHOT_SECTION_REQUIRED;
        layout->sections[i].offset = offset;
        layout->sections[i].length = lengths[i];
        layout->sections[i].count = counts[i];
        if (!format_add_u64(offset, lengths[i], &offset)) return 0;
    }
    layout->file_size = offset;
    return 1;
}

xe_snapshot_status xe_session_snapshot_size(xe_session *s, uint64_t *size) {
    if (!s || !size || s->cow) return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    if (!s->n_tokens) return XE_SNAPSHOT_EMPTY;
    xe_snapshot_layout layout;
    if (!xe_snapshot_layout_build(s->engine->context, (uint64_t)s->n_tokens, &layout))
        return XE_SNAPSHOT_FORMAT;
    *size = layout.file_size;
    return XE_SNAPSHOT_OK;
}

static xe_snapshot_status xe_snapshot_gpu_sync(xe_engine *e) {
    if (!e->gpu.commands) return XE_SNAPSHOT_OK;
    ze_result_t result = zeCommandListHostSynchronize(e->gpu.commands,
                                                       UINT64_MAX);
    return result == ZE_RESULT_SUCCESS ? XE_SNAPSHOT_OK : XE_SNAPSHOT_BACKEND;
}

static int xe_snapshot_stream_write(FILE *out, const void *data,
                                    uint64_t length, uint64_t *crc) {
    const uint8_t *p = data;
    while (length) {
        size_t chunk = length > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)length;
        if (!format_write(out, p, chunk)) return 0;
        *crc = format_crc64(*crc, p, chunk);
        p += chunk;
        length -= chunk;
    }
    return 1;
}

static int xe_snapshot_write_tokens(FILE *out, const int32_t *tokens, int n,
                                    uint8_t *buffer, uint64_t *crc) {
    int at = 0;
    int cap = XE_SNAPSHOT_BUFFER_SIZE / 4;
    while (at < n) {
        int count = n - at < cap ? n - at : cap;
        for (int i = 0; i < count; i++)
            format_put_u32le(buffer + 4 * i, (uint32_t)tokens[at + i]);
        size_t bytes = (size_t)count * 4;
        if (!format_write(out, buffer, bytes)) return 0;
        *crc = format_crc64(*crc, buffer, bytes);
        at += count;
    }
    return 1;
}

static int xe_snapshot_write_kv(FILE *out, xe_session *s, int global,
                                int value, uint64_t rows, uint64_t *crc) {
    int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    int first = global ? 0 : s->n_tokens - (int)rows;
    for (int layer = 0; layer < XE_KV_LAYERS; layer++) {
        if (XE_IS_GLOBAL(layer) != global) continue;
        _Float16 *base = xe_kv_layer_ptr(s, layer, value);
        for (int head = 0; head < heads; head++) {
            _Float16 *head_base = base + (size_t)head * capacity * dimension;
            if (global) {
                uint64_t bytes = rows * (uint64_t)dimension * sizeof(_Float16);
                if (!xe_snapshot_stream_write(out, head_base, bytes, crc))
                    return 0;
            } else {
                int slot = first & (XE_SWA_WINDOW - 1);
                uint64_t first_rows = rows;
                if (first_rows > (uint64_t)(capacity - slot))
                    first_rows = (uint64_t)(capacity - slot);
                uint64_t first_bytes = first_rows * (uint64_t)dimension *
                                       sizeof(_Float16);
                if (!xe_snapshot_stream_write(out,
                        head_base + (size_t)slot * dimension,
                        first_bytes, crc)) return 0;
                uint64_t second_rows = rows - first_rows;
                uint64_t second_bytes = second_rows * (uint64_t)dimension *
                                        sizeof(_Float16);
                if (second_bytes && !xe_snapshot_stream_write(out, head_base,
                                                               second_bytes,
                                                               crc)) return 0;
            }
        }
    }
    return 1;
}

static void xe_snapshot_encode_directory(const xe_snapshot_layout *layout,
                                         uint8_t *directory) {
    for (uint32_t i = 0; i < layout->section_count; i++) {
        uint8_t *entry = directory + (size_t)i * XE_SNAPSHOT_ENTRY_SIZE;
        const xe_snapshot_section *section = &layout->sections[i];
        format_put_u32le(entry, section->type);
        format_put_u32le(entry + 4, section->flags);
        format_put_u64le(entry + 8, section->offset);
        format_put_u64le(entry + 16, section->length);
        format_put_u64le(entry + 24, section->count);
        format_put_u64le(entry + 32, section->crc);
    }
}

static void xe_snapshot_encode_header(xe_engine *e,
                                      const xe_snapshot_layout *layout,
                                      const uint8_t boundary[32],
                                      const uint8_t *directory,
                                      uint8_t header[XE_SNAPSHOT_HEADER_SIZE]) {
    memset(header, 0, XE_SNAPSHOT_HEADER_SIZE);
    memcpy(header, xe_snapshot_magic, sizeof xe_snapshot_magic);
    format_put_u16le(header + 8, XE_SNAPSHOT_FORMAT_VERSION);
    format_put_u16le(header + 10, 0);
    format_put_u32le(header + 12, XE_SNAPSHOT_HEADER_SIZE);
    format_put_u32le(header + 16, XE_SNAPSHOT_ENTRY_SIZE);
    format_put_u32le(header + 20, layout->section_count);
    format_put_u64le(header + 32, layout->file_size);
    format_put_u64le(header + 40, layout->position);
    format_put_u64le(header + 48, layout->swa_rows);
    format_put_u64le(header + 56, layout->global_rows);
    format_put_u64le(header + 64, layout->directory_offset);
    format_put_u64le(header + 72, layout->directory_length);
    format_put_u64le(header + 80, layout->payload_offset);
    format_put_u64le(header + 88,
                     format_crc64(0, directory,
                                  (size_t)layout->directory_length));
    for (int i = 0; i < XE_SNAPSHOT_FP_COUNT; i++)
        memcpy(header + 96 + 32 * i, e->snapshot_fingerprint[i], 32);
    memcpy(header + 288, boundary, 32);
    format_put_u32le(header + 320, XE_SNAPSHOT_NUMERIC_ABI);
    /* Diagnostic strings are NUL-terminated and truncated to fit. They are
     * covered by the checksum, but never used to decide compatibility. */
    snprintf((char *)header + 328, 16, "%.15s", XE_BUILD_COMMIT);
    snprintf((char *)header + 344, 32, "%.31s", __VERSION__);
    format_put_u64le(header + 376, format_crc64(0, header, 376));
}

xe_snapshot_status xe_session_snapshot_stream(xe_session *s, FILE *out) {
    if (!s || !out || s->cow) return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    if (!s->n_tokens) return XE_SNAPSHOT_EMPTY;
    xe_snapshot_status status = xe_snapshot_gpu_sync(s->engine);
    if (status != XE_SNAPSHOT_OK) return status;
    xe_snapshot_fingerprints_init(s->engine);

    xe_snapshot_layout layout;
    if (!xe_snapshot_layout_build(s->engine->context, (uint64_t)s->n_tokens, &layout))
        return XE_SNAPSHOT_FORMAT;
    uint8_t *buffer = malloc(XE_SNAPSHOT_BUFFER_SIZE);
    if (!buffer) return XE_SNAPSHOT_NOMEM;
    uint8_t directory[XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE];
    uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t boundary[32];
    memset(buffer, 0, (size_t)layout.payload_offset);
    if (!format_seek(out, 0) ||
        !format_write(out, buffer, (size_t)layout.payload_offset)) {
        free(buffer);
        return XE_SNAPSHOT_IO;
    }

    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++) {
        xe_snapshot_section *section = &layout.sections[i];
        if (!format_seek(out, section->offset)) {
            free(buffer);
            return XE_SNAPSHOT_IO;
        }
        uint64_t crc = 0;
        int ok = 0;
        if (section->type == XE_SNAPSHOT_TOKENS)
            ok = xe_snapshot_write_tokens(out, s->tokens, s->n_tokens,
                                          buffer, &crc);
        else if (section->type == XE_SNAPSHOT_LOGITS)
            ok = xe_snapshot_stream_write(out, s->logits, section->length,
                                          &crc);
        else if (section->type == XE_SNAPSHOT_SWA_K)
            ok = xe_snapshot_write_kv(out, s, 0, 0, layout.swa_rows, &crc);
        else if (section->type == XE_SNAPSHOT_SWA_V)
            ok = xe_snapshot_write_kv(out, s, 0, 1, layout.swa_rows, &crc);
        else if (section->type == XE_SNAPSHOT_GLOBAL_K)
            ok = xe_snapshot_write_kv(out, s, 1, 0, layout.global_rows, &crc);
        else if (section->type == XE_SNAPSHOT_GLOBAL_V)
            ok = xe_snapshot_write_kv(out, s, 1, 1, layout.global_rows, &crc);
        if (!ok) {
            free(buffer);
            return XE_SNAPSHOT_IO;
        }
        section->crc = crc;
    }

    xe_snapshot_boundary_hash(s->tokens, s->n_tokens, boundary);
    xe_snapshot_encode_directory(&layout, directory);
    xe_snapshot_encode_header(s->engine, &layout, boundary, directory, header);
    int ok = format_seek(out, 0) &&
             format_write(out, header, sizeof header) &&
             format_write(out, directory, sizeof directory) &&
             fflush(out) == 0;
    free(buffer);
    return ok ? XE_SNAPSHOT_OK : XE_SNAPSHOT_IO;
}

xe_snapshot_status xe_session_snapshot_save(xe_session *s, FILE *out) {
    xe_snapshot_status status = xe_session_snapshot_stream(s, out);
    if (status != XE_SNAPSHOT_OK) return status;
    uint64_t size;
    status = xe_session_snapshot_size(s, &size);
    if (status != XE_SNAPSHOT_OK) return status;
    return ftruncate(fileno(out), (off_t)size) == 0 ? XE_SNAPSHOT_OK : XE_SNAPSHOT_IO;
}

static xe_snapshot_status xe_snapshot_fingerprint_status(int index) {
    static const xe_snapshot_status status[XE_SNAPSHOT_FP_COUNT] = {
        XE_SNAPSHOT_MODEL_MISMATCH,
        XE_SNAPSHOT_TOKENIZER_MISMATCH,
        XE_SNAPSHOT_TEMPLATE_MISMATCH,
        XE_SNAPSHOT_LAYOUT_MISMATCH,
        XE_SNAPSHOT_CONTEXT_MISMATCH,
        XE_SNAPSHOT_NUMERIC_MISMATCH
    };
    return status[index];
}

static int xe_snapshot_all_zero(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) if (p[i]) return 0;
    return 1;
}

static xe_snapshot_status xe_snapshot_parse_header(
        xe_engine *e, FILE *in, const xe_tokens *expected,
        xe_snapshot_layout *layout,
        uint8_t saved_header[XE_SNAPSHOT_HEADER_SIZE],
        uint8_t saved_directory[XE_SNAPSHOT_MAX_SECTIONS *
                                XE_SNAPSHOT_ENTRY_SIZE]) {
    uint64_t actual_size;
    if (!format_file_size(in, &actual_size) || !format_seek(in, 0) ||
        !format_read(in, saved_header, XE_SNAPSHOT_HEADER_SIZE))
        return XE_SNAPSHOT_IO;
    if (memcmp(saved_header, xe_snapshot_magic, sizeof xe_snapshot_magic) != 0)
        return XE_SNAPSHOT_FORMAT;
    uint16_t version = format_get_u16le(saved_header + 8);
    if ((version != 1 && version != XE_SNAPSHOT_FORMAT_VERSION) ||
        format_get_u16le(saved_header + 10) != 0)
        return XE_SNAPSHOT_VERSION;
    if (format_get_u32le(saved_header + 12) != XE_SNAPSHOT_HEADER_SIZE ||
        format_get_u32le(saved_header + 16) != XE_SNAPSHOT_ENTRY_SIZE)
        return XE_SNAPSHOT_VERSION;
    if (format_get_u32le(saved_header + 24) != 0 ||
        format_get_u32le(saved_header + 28) != 0 ||
        (version == 1 && !xe_snapshot_all_zero(saved_header + 320, 56)) ||
        (version == XE_SNAPSHOT_FORMAT_VERSION &&
         (format_get_u32le(saved_header + 324) != 0 ||
          saved_header[343] != 0 || saved_header[375] != 0)))
        return XE_SNAPSHOT_FORMAT;
    if (format_get_u64le(saved_header + 376) !=
        format_crc64(0, saved_header, 376)) return XE_SNAPSHOT_FORMAT;

    memset(layout, 0, sizeof *layout);
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++)
        layout->known[i] = -1;
    layout->section_count = format_get_u32le(saved_header + 20);
    layout->file_size = format_get_u64le(saved_header + 32);
    layout->position = format_get_u64le(saved_header + 40);
    layout->swa_rows = format_get_u64le(saved_header + 48);
    layout->global_rows = format_get_u64le(saved_header + 56);
    layout->directory_offset = format_get_u64le(saved_header + 64);
    layout->directory_length = format_get_u64le(saved_header + 72);
    layout->payload_offset = format_get_u64le(saved_header + 80);
    if (layout->position != (uint64_t)expected->len)
        return XE_SNAPSHOT_TOKEN_MISMATCH;
    if (layout->section_count < XE_SNAPSHOT_REQUIRED_SECTIONS ||
        layout->section_count > XE_SNAPSHOT_MAX_SECTIONS ||
        layout->file_size != actual_size ||
        layout->global_rows != layout->position ||
        layout->swa_rows != (layout->position < XE_SWA_WINDOW
                            ? layout->position : XE_SWA_WINDOW) ||
        layout->directory_offset != XE_SNAPSHOT_HEADER_SIZE ||
        !format_mul_u64(layout->section_count, XE_SNAPSHOT_ENTRY_SIZE,
                        &layout->directory_length) ||
        format_get_u64le(saved_header + 72) != layout->directory_length)
        return XE_SNAPSHOT_FORMAT;
    uint64_t directory_end;
    if (!format_add_u64(layout->directory_offset, layout->directory_length,
                        &directory_end) ||
        layout->payload_offset < directory_end ||
        (layout->payload_offset & 63) || layout->payload_offset > actual_size)
        return XE_SNAPSHOT_FORMAT;
    if (!format_seek(in, layout->directory_offset) ||
        !format_read(in, saved_directory, (size_t)layout->directory_length))
        return XE_SNAPSHOT_IO;
    if (format_get_u64le(saved_header + 88) !=
        format_crc64(0, saved_directory, (size_t)layout->directory_length))
        return XE_SNAPSHOT_FORMAT;

    for (uint32_t i = 0; i < layout->section_count; i++) {
        const uint8_t *entry = saved_directory +
                               (size_t)i * XE_SNAPSHOT_ENTRY_SIZE;
        xe_snapshot_section *section = &layout->sections[i];
        section->type = format_get_u32le(entry);
        section->flags = format_get_u32le(entry + 4);
        section->offset = format_get_u64le(entry + 8);
        section->length = format_get_u64le(entry + 16);
        section->count = format_get_u64le(entry + 24);
        section->crc = format_get_u64le(entry + 32);
        uint64_t section_end;
        if (!section->type || (section->flags & ~XE_SNAPSHOT_SECTION_REQUIRED) ||
            section->offset < layout->payload_offset ||
            !format_add_u64(section->offset, section->length, &section_end) ||
            section_end > layout->file_size) return XE_SNAPSHOT_FORMAT;
        for (uint32_t j = 0; j < i; j++) {
            const xe_snapshot_section *other = &layout->sections[j];
            uint64_t other_end = other->offset + other->length;
            if (section->type == other->type ||
                (section->offset < other_end && other->offset < section_end))
                return XE_SNAPSHOT_FORMAT;
        }
        if (section->type <= XE_SNAPSHOT_REQUIRED_SECTIONS) {
            if (section->flags != XE_SNAPSHOT_SECTION_REQUIRED)
                return XE_SNAPSHOT_FORMAT;
            layout->known[section->type - 1] = (int)i;
        } else if (section->flags & XE_SNAPSHOT_SECTION_REQUIRED) {
            return XE_SNAPSHOT_VERSION;
        }
    }

    if (memcmp(saved_header + 96 + 32 * XE_SNAPSHOT_FP_CONTEXT,
               e->snapshot_fingerprint[XE_SNAPSHOT_FP_CONTEXT], 32))
        return XE_SNAPSHOT_CONTEXT_MISMATCH;
    xe_snapshot_layout expected_layout;
    if (!xe_snapshot_layout_build(e->context, layout->position, &expected_layout))
        return XE_SNAPSHOT_FORMAT;
    for (int type = 0; type < XE_SNAPSHOT_REQUIRED_SECTIONS; type++) {
        int index = layout->known[type];
        if (index < 0) return XE_SNAPSHOT_FORMAT;
        const xe_snapshot_section *got = &layout->sections[index];
        const xe_snapshot_section *want = &expected_layout.sections[type];
        if (got->length != want->length || got->count != want->count)
            return XE_SNAPSHOT_FORMAT;
    }

    for (int i = 0; i < XE_SNAPSHOT_FP_NUMERIC; i++)
        if (memcmp(saved_header + 96 + 32 * i,
                   e->snapshot_fingerprint[i], 32) != 0)
            return xe_snapshot_fingerprint_status(i);
    const uint8_t *numeric = saved_header + 96 + 32 * XE_SNAPSHOT_FP_NUMERIC;
    if (version == 1) {
        if (!xe_snapshot_legacy_numeric_matches(e, numeric))
            return XE_SNAPSHOT_NUMERIC_MISMATCH;
    } else if (format_get_u32le(saved_header + 320) != XE_SNAPSHOT_NUMERIC_ABI ||
               memcmp(numeric, e->snapshot_fingerprint[XE_SNAPSHOT_FP_NUMERIC], 32)) {
        return XE_SNAPSHOT_NUMERIC_MISMATCH;
    }
    uint8_t boundary[32];
    xe_snapshot_boundary_hash(expected->v, expected->len, boundary);
    if (memcmp(saved_header + 288, boundary, 32) != 0)
        return XE_SNAPSHOT_TOKEN_MISMATCH;
    return XE_SNAPSHOT_OK;
}

static xe_snapshot_status xe_snapshot_verify_section(
        FILE *in, const xe_snapshot_section *section, uint8_t *buffer,
        const xe_tokens *expected) {
    if (!format_seek(in, section->offset)) return XE_SNAPSHOT_IO;
    uint64_t remaining = section->length;
    uint64_t crc = 0;
    int token_at = 0;
    while (remaining) {
        size_t chunk = remaining > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)remaining;
        if (!format_read(in, buffer, chunk)) return XE_SNAPSHOT_IO;
        crc = format_crc64(crc, buffer, chunk);
        if (section->type == XE_SNAPSHOT_TOKENS) {
            if (chunk & 3) return XE_SNAPSHOT_FORMAT;
            for (size_t i = 0; i < chunk; i += 4, token_at++)
                if (token_at >= expected->len ||
                    format_get_u32le(buffer + i) !=
                    (uint32_t)expected->v[token_at])
                    return XE_SNAPSHOT_TOKEN_MISMATCH;
        }
        remaining -= chunk;
    }
    if (section->type == XE_SNAPSHOT_TOKENS && token_at != expected->len)
        return XE_SNAPSHOT_TOKEN_MISMATCH;
    return crc == section->crc ? XE_SNAPSHOT_OK : XE_SNAPSHOT_FORMAT;
}

static xe_snapshot_status xe_snapshot_read_raw(
        FILE *in, const xe_snapshot_section *section, void *destination) {
    if (!format_seek(in, section->offset)) return XE_SNAPSHOT_IO;
    uint8_t *p = destination;
    uint64_t remaining = section->length;
    uint64_t crc = 0;
    while (remaining) {
        size_t chunk = remaining > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)remaining;
        if (!format_read(in, p, chunk)) return XE_SNAPSHOT_IO;
        crc = format_crc64(crc, p, chunk);
        p += chunk;
        remaining -= chunk;
    }
    return crc == section->crc ? XE_SNAPSHOT_OK : XE_SNAPSHOT_FORMAT;
}

static int xe_snapshot_stream_read(FILE *in, void *data, uint64_t length,
                                   uint64_t *crc) {
    uint8_t *p = data;
    while (length) {
        size_t chunk = length > XE_SNAPSHOT_BUFFER_SIZE
                       ? XE_SNAPSHOT_BUFFER_SIZE : (size_t)length;
        if (!format_read(in, p, chunk)) return 0;
        *crc = format_crc64(*crc, p, chunk);
        p += chunk;
        length -= chunk;
    }
    return 1;
}

static xe_snapshot_status xe_snapshot_read_kv(
        FILE *in, const xe_snapshot_section *section, xe_session *s,
        int global, int value, uint64_t rows, uint64_t position) {
    if (!format_seek(in, section->offset)) return XE_SNAPSHOT_IO;
    int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
    int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int capacity = global ? s->engine->context : XE_SWA_WINDOW;
    int logical_first = global ? 0 : (int)(position - rows);
    uint64_t crc = 0;
    for (int layer = 0; layer < XE_KV_LAYERS; layer++) {
        if (XE_IS_GLOBAL(layer) != global) continue;
        _Float16 *base = xe_kv_layer_ptr(s, layer, value);
        for (int head = 0; head < heads; head++) {
            _Float16 *head_base = base + (size_t)head * capacity * dimension;
            if (global) {
                uint64_t bytes = rows * (uint64_t)dimension * sizeof(_Float16);
                if (!xe_snapshot_stream_read(in, head_base, bytes, &crc))
                    return XE_SNAPSHOT_IO;
            } else {
                int slot = logical_first & (XE_SWA_WINDOW - 1);
                uint64_t first_rows = rows;
                if (first_rows > (uint64_t)(capacity - slot))
                    first_rows = (uint64_t)(capacity - slot);
                uint64_t first_bytes = first_rows * (uint64_t)dimension *
                                       sizeof(_Float16);
                if (!xe_snapshot_stream_read(in,
                        head_base + (size_t)slot * dimension,
                        first_bytes, &crc)) return XE_SNAPSHOT_IO;
                uint64_t second_rows = rows - first_rows;
                uint64_t second_bytes = second_rows * (uint64_t)dimension *
                                        sizeof(_Float16);
                if (second_bytes && !xe_snapshot_stream_read(in, head_base,
                                                              second_bytes,
                                                              &crc))
                    return XE_SNAPSHOT_IO;
            }
        }
    }
    return crc == section->crc ? XE_SNAPSHOT_OK : XE_SNAPSHOT_FORMAT;
}

xe_snapshot_status xe_session_snapshot_check_prefix(xe_session *s, FILE *in,
                                                     uint64_t position) {
    if (!s || s->cow || !in || !position || position > (uint64_t)s->n_tokens)
        return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    xe_snapshot_fingerprints_init(s->engine);
    xe_tokens expected = {s->tokens, (int)position, (int)position};
    uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t directory[XE_SNAPSHOT_MAX_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE];
    xe_snapshot_layout layout;
    xe_snapshot_status status = xe_snapshot_parse_header(
        s->engine, in, &expected, &layout, header, directory);
    if (status != XE_SNAPSHOT_OK) return status;
    uint8_t *buffer = malloc(XE_SNAPSHOT_BUFFER_SIZE);
    if (!buffer) return XE_SNAPSHOT_NOMEM;
    status = xe_snapshot_verify_section(in,
        &layout.sections[layout.known[XE_SNAPSHOT_TOKENS - 1]], buffer, &expected);
    free(buffer);
    return status;
}

xe_snapshot_status xe_session_snapshot_load(xe_session *s, FILE *in,
                                            const xe_tokens *expected) {
    if (!s || s->cow || !in || !expected) return XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_require_owner(s->engine);
    if (!xe_snapshot_expected_valid(expected) || expected->len > s->engine->context)
        return expected->len == 0 ? XE_SNAPSHOT_EMPTY
                                  : XE_SNAPSHOT_INVALID_ARGUMENT;
    xe_snapshot_fingerprints_init(s->engine);
    uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t directory[XE_SNAPSHOT_MAX_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE];
    xe_snapshot_layout layout;
    xe_snapshot_status status = xe_snapshot_parse_header(
        s->engine, in, expected, &layout, header, directory);
    if (status != XE_SNAPSHOT_OK) return status;
    uint8_t *buffer = malloc(XE_SNAPSHOT_BUFFER_SIZE);
    if (!buffer) return XE_SNAPSHOT_NOMEM;
    status = xe_snapshot_verify_section(
        in, &layout.sections[layout.known[XE_SNAPSHOT_TOKENS - 1]], buffer,
        expected);
    if (status != XE_SNAPSHOT_OK) {
        free(buffer);
        return status;
    }
    for (uint32_t i = 0; i < layout.section_count; i++) {
        if (layout.sections[i].type <= XE_SNAPSHOT_REQUIRED_SECTIONS) continue;
        status = xe_snapshot_verify_section(in, &layout.sections[i], buffer,
                                            expected);
        if (status != XE_SNAPSHOT_OK) {
            free(buffer);
            return status;
        }
    }
    free(buffer);
    status = xe_snapshot_gpu_sync(s->engine);
    if (status != XE_SNAPSHOT_OK) return status;
    s->n_tokens = 0;
    status = xe_snapshot_read_raw(
        in, &layout.sections[layout.known[XE_SNAPSHOT_LOGITS - 1]], s->logits);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_SWA_K - 1]], s,
            0, 0, layout.swa_rows, layout.position);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_SWA_V - 1]], s,
            0, 1, layout.swa_rows, layout.position);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_GLOBAL_K - 1]], s,
            1, 0, layout.global_rows, layout.position);
    if (status == XE_SNAPSHOT_OK)
        status = xe_snapshot_read_kv(
            in, &layout.sections[layout.known[XE_SNAPSHOT_GLOBAL_V - 1]], s,
            1, 1, layout.global_rows, layout.position);
    if (status != XE_SNAPSHOT_OK) return status;
    memcpy(s->tokens, expected->v,
           (size_t)expected->len * sizeof(*s->tokens));
    s->n_tokens = expected->len;
    if (s->swa_spare_k && s->swa_spare_v)
        memset(s->swa_dirty, 1, sizeof s->swa_dirty);
    return XE_SNAPSHOT_OK;
}

const char *xe_snapshot_status_name(xe_snapshot_status status) {
    static const char *names[] = {
        "ok", "invalid-argument", "empty", "io", "format", "version",
        "model-mismatch", "tokenizer-mismatch", "template-mismatch",
        "layout-mismatch", "context-mismatch", "numeric-mismatch",
        "token-mismatch", "backend", "no-memory", "changed"
    };
    return (unsigned)status < sizeof names / sizeof names[0]
           ? names[status] : "unknown";
}


/* Diagnostic dump hook for a separate validation environment. This is not an
 * independent model oracle; compare these dumps with HF/llama.cpp externally. */
void xe_oracle(xe_engine *e,const int32_t *tokens,int count,const char *dump_path,
               const char *layers_dir,int q8_mode,FILE *out) {
    if(!tokens || count<1 || count>e->context) xe_fatal("oracle: invalid tokens");
    if(layers_dir && mkdir(layers_dir,0777)!=0 && errno!=EEXIST)
        xe_fatal("%s: %s",layers_dir,strerror(errno));
    xe_session *s=xe_session_new(e);
    s->reference_mode=!q8_mode;s->dump_layers=layers_dir;s->dump_position=count-1;
    for(int i=0;i<count;i++) xe_decode_token_mode(s,tokens[i],i,XE_SOFTCAP_SECOND_LOOP,i==count-1);
    for(int i=0;i<XE_VOCAB;i++) if(!isfinite(s->logits[i])) xe_fatal("nonfinite logit %d",i);
    xe_sampler sampler={.temperature=0,.top_k=1,.top_p=1};
    int32_t best=xe_sample_logits(s->logits,XE_VOCAB,s->sample_candidates,&sampler);
    fprintf(out,"E4B diagnostic (%s activations): top token %d, logit %.6f\n",
            q8_mode?"Q8":"F32",best,(double)s->logits[best]);
    if(dump_path) {
        FILE *f=fopen(dump_path,"wb");if(!f) xe_fatal("%s: %s",dump_path,strerror(errno));
        if(fwrite(s->logits,sizeof(float),XE_VOCAB,f)!=XE_VOCAB) xe_fatal("logit dump short write");
        if(fclose(f)) xe_fatal("logit dump close failed");
    }
    xe_session_free(s);
}
