#ifndef LOCI_FFI_STUB_H
#define LOCI_FFI_STUB_H

#include <stdbool.h>
#include <stdint.h>

typedef int32_t llama_token;

typedef enum ggml_numa_strategy {
    ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED = 0,
} ggml_numa_strategy;

typedef enum llama_flash_attn_type {
    llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
} llama_flash_attn_type;

typedef struct llama_vocab {
    int32_t n_vocab;
} llama_vocab;

typedef struct llama_model {
    struct llama_vocab vocab;
    int32_t n_ctx_train;
    int32_t n_embd;
    bool has_encoder;
    bool has_decoder;
} llama_model;

typedef struct llama_memory {
    float *logits;
    int32_t n_vocab;
} llama_memory;

typedef struct llama_context {
    struct llama_model *model;
    float logits[512];
    float embeddings[64];
    struct llama_memory memory;
} llama_context;

typedef struct llama_model_params {
    int32_t n_gpu_layers;
} llama_model_params;

typedef struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    int32_t n_threads;
    enum llama_flash_attn_type flash_attn_type;
} llama_context_params;

typedef struct llama_batch {
    int32_t n_tokens;
    llama_token *token;
    float *embd;
    int32_t *pos;
    int32_t *n_seq_id;
    int32_t **seq_id;
    int8_t *logits;
} llama_batch;

void loci_llama_model_default_params(struct llama_model_params *out_params);
void loci_llama_context_default_params(struct llama_context_params *out_params);
struct llama_model *loci_llama_model_load_from_file(
    const char *path,
    const struct llama_model_params *params
);
struct llama_context *loci_llama_init_from_model(
    struct llama_model *model,
    const struct llama_context_params *params
);
void loci_llama_batch_init(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max,
    struct llama_batch *out_batch
);
void loci_llama_batch_free(const struct llama_batch *batch);
int32_t loci_llama_encode(struct llama_context *ctx, const struct llama_batch *batch);
int32_t loci_llama_decode(struct llama_context *ctx, const struct llama_batch *batch);
int32_t loci_llama_tokenize(
    const struct llama_vocab *vocab,
    const char *text,
    int32_t text_len,
    llama_token *tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
);

void llama_backend_init(void);
void llama_numa_init(enum ggml_numa_strategy strategy);
void llama_backend_free(void);
struct llama_vocab *llama_model_get_vocab(struct llama_model *model);
int32_t llama_n_vocab(const struct llama_vocab *vocab);
int32_t llama_model_n_ctx_train(struct llama_model *model);
int32_t llama_n_embd(struct llama_model *model);
bool llama_model_has_encoder(struct llama_model *model);
bool llama_model_has_decoder(struct llama_model *model);
int32_t llama_detokenize(
    const struct llama_vocab *vocab,
    const llama_token *tokens,
    int32_t n_tokens,
    char *text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special
);
bool llama_token_is_eog(const struct llama_vocab *vocab, llama_token token);
void llama_model_free(struct llama_model *model);
void llama_free(struct llama_context *ctx);
float *llama_get_logits_ith(struct llama_context *ctx, int32_t index);
float *llama_get_embeddings(struct llama_context *ctx);
struct llama_memory *llama_get_memory(const struct llama_context *ctx);
void llama_memory_clear(struct llama_memory *memory, bool reset);

#endif
