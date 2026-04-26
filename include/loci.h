#ifndef LOCI_H
#define LOCI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOCI_ABI_VERSION 1u

typedef struct LociEngine LociEngine;

typedef enum LociGpuSplitMode {
    LOCI_GPU_SPLIT_NONE = 0,
    LOCI_GPU_SPLIT_LAYER = 1,
    LOCI_GPU_SPLIT_ROW = 2
} LociGpuSplitMode;

typedef enum LociModelLoadStrategyKind {
    LOCI_MODEL_LOAD_STRICT = 0,
    LOCI_MODEL_LOAD_AUTO_REDUCE_GPU_LAYERS = 1
} LociModelLoadStrategyKind;

typedef struct LociModelLoadOptions {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint8_t has_n_threads;
    uint32_t n_threads;
    uint8_t use_gpu;
    int32_t n_gpu_layers;
    uint8_t use_mmap;
    uint8_t use_mlock;
    uint8_t kv_offload;
    uint8_t op_offload;
    LociGpuSplitMode split_mode;
    uint32_t main_gpu;
    const float* tensor_split;
    uint32_t tensor_split_len;
    LociModelLoadStrategyKind load_strategy;
    uint32_t load_strategy_step;
} LociModelLoadOptions;

typedef struct LociGenerationOptions {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint8_t has_n_threads;
    uint32_t n_threads;
    uint32_t max_tokens;
    float temperature;
    float top_p;
    float min_p;
    uint32_t top_k;
    float repeat_penalty;
} LociGenerationOptions;

uint32_t loci_abi_version(void);
const char* loci_version(void);
const char* loci_get_last_error(void);

LociModelLoadOptions loci_default_model_load_options(void);
LociGenerationOptions loci_default_generation_options(void);

LociEngine* loci_engine_new(void);
void loci_engine_free(LociEngine* engine);

char* loci_engine_load_model_json(
    LociEngine* engine,
    const char* backend_name,
    const char* model_path,
    const LociModelLoadOptions* options
);

char* loci_generate_with_len_and_options(
    LociEngine* engine,
    const char* prompt,
    uint32_t prompt_len,
    const LociGenerationOptions* options
);

char* loci_engine_runtime_snapshot_json(LociEngine* engine);
char* loci_engine_backend_capabilities_json(LociEngine* engine);

void loci_free_string(char* value);

#ifdef __cplusplus
}
#endif

#endif
