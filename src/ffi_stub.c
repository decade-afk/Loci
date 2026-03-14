#include "ffi_stub.h"

#include <stdlib.h>
#include <string.h>

static int32_t clamp_token(int32_t token, int32_t n_vocab) {
    if (n_vocab <= 0) {
        return 0;
    }
    if (token < 0) {
        return 0;
    }
    if (token >= n_vocab) {
        return token % n_vocab;
    }
    return token;
}

void loci_llama_model_default_params(struct llama_model_params *out_params) {
    if (out_params == NULL) {
        return;
    }
    memset(out_params, 0, sizeof(*out_params));
}

void loci_llama_context_default_params(struct llama_context_params *out_params) {
    if (out_params == NULL) {
        return;
    }
    memset(out_params, 0, sizeof(*out_params));
    out_params->n_ctx = 4096;
    out_params->n_batch = 512;
    out_params->flash_attn_type = llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
}

struct llama_model *loci_llama_model_load_from_file(
    const char *path,
    const struct llama_model_params *params
) {
    (void)path;
    (void)params;

    struct llama_model *model = (struct llama_model *)calloc(1, sizeof(struct llama_model));
    if (model == NULL) {
        return NULL;
    }

    model->vocab.n_vocab = 256;
    model->n_ctx_train = 4096;
    model->n_embd = 16;
    model->has_encoder = false;
    model->has_decoder = true;
    return model;
}

struct llama_context *loci_llama_init_from_model(
    struct llama_model *model,
    const struct llama_context_params *params
) {
    (void)params;

    if (model == NULL) {
        return NULL;
    }

    struct llama_context *ctx = (struct llama_context *)calloc(1, sizeof(struct llama_context));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->model = model;
    ctx->memory.logits = ctx->logits;
    ctx->memory.n_vocab = model->vocab.n_vocab;
    ctx->logits[0] = 1.0f;
    return ctx;
}

void loci_llama_batch_init(
    int32_t n_tokens,
    int32_t embd,
    int32_t n_seq_max,
    struct llama_batch *out_batch
) {
    if (out_batch == NULL || n_tokens <= 0) {
        return;
    }

    memset(out_batch, 0, sizeof(*out_batch));
    out_batch->n_tokens = n_tokens;
    out_batch->token = (llama_token *)calloc((size_t)n_tokens, sizeof(llama_token));
    if (embd > 0) {
        out_batch->embd = (float *)calloc((size_t)n_tokens * (size_t)embd, sizeof(float));
    }
    out_batch->pos = (int32_t *)calloc((size_t)n_tokens, sizeof(int32_t));
    out_batch->n_seq_id = (int32_t *)calloc((size_t)n_tokens, sizeof(int32_t));
    out_batch->seq_id = (int32_t **)calloc((size_t)n_tokens, sizeof(int32_t *));
    out_batch->logits = (int8_t *)calloc((size_t)n_tokens, sizeof(int8_t));

    if (out_batch->seq_id != NULL && n_seq_max > 0) {
        for (int32_t i = 0; i < n_tokens; ++i) {
            out_batch->seq_id[i] = (int32_t *)calloc((size_t)n_seq_max, sizeof(int32_t));
        }
    }
}

void loci_llama_batch_free(const struct llama_batch *batch) {
    if (batch == NULL) {
        return;
    }

    if (batch->seq_id != NULL) {
        for (int32_t i = 0; i < batch->n_tokens; ++i) {
            free(batch->seq_id[i]);
        }
    }

    free(batch->token);
    free(batch->embd);
    free(batch->pos);
    free(batch->n_seq_id);
    free(batch->seq_id);
    free(batch->logits);
}

int32_t loci_llama_encode(struct llama_context *ctx, const struct llama_batch *batch) {
    return loci_llama_decode(ctx, batch);
}

int32_t loci_llama_decode(struct llama_context *ctx, const struct llama_batch *batch) {
    if (ctx == NULL || batch == NULL || ctx->model == NULL) {
        return -1;
    }

    memset(ctx->logits, 0, sizeof(ctx->logits));
    memset(ctx->embeddings, 0, sizeof(ctx->embeddings));

    int32_t selected = 0;
    if (batch->n_tokens > 0 && batch->token != NULL) {
        selected = clamp_token(batch->token[batch->n_tokens - 1], ctx->model->vocab.n_vocab);
    }

    ctx->logits[selected] = 10.0f;
    for (int i = 0; i < ctx->model->n_embd && i < (int)(sizeof(ctx->embeddings) / sizeof(ctx->embeddings[0])); ++i) {
        ctx->embeddings[i] = (float)(selected + i);
    }

    return 0;
}

int32_t loci_llama_tokenize(
    const struct llama_vocab *vocab,
    const char *text,
    int32_t text_len,
    llama_token *tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
) {
    (void)parse_special;

    if (vocab == NULL || text == NULL || tokens == NULL || text_len < 0) {
        return -1;
    }

    int32_t required = text_len + (add_special ? 1 : 0);
    if (required <= 0) {
        required = 1;
    }

    if (n_tokens_max < required) {
        return -required;
    }

    int32_t index = 0;
    if (add_special) {
        tokens[index++] = 1;
    }

    for (int32_t i = 0; i < text_len; ++i) {
        unsigned char value = (unsigned char)text[i];
        tokens[index++] = clamp_token((int32_t)value, vocab->n_vocab);
    }

    if (index == 0) {
        tokens[index++] = 1;
    }

    return index;
}

void llama_backend_init(void) {}

void llama_numa_init(enum ggml_numa_strategy strategy) {
    (void)strategy;
}

void llama_backend_free(void) {}

struct llama_vocab *llama_model_get_vocab(struct llama_model *model) {
    if (model == NULL) {
        return NULL;
    }
    return &model->vocab;
}

int32_t llama_n_vocab(const struct llama_vocab *vocab) {
    return vocab == NULL ? 0 : vocab->n_vocab;
}

int32_t llama_model_n_ctx_train(struct llama_model *model) {
    return model == NULL ? 0 : model->n_ctx_train;
}

int32_t llama_n_embd(struct llama_model *model) {
    return model == NULL ? 0 : model->n_embd;
}

bool llama_model_has_encoder(struct llama_model *model) {
    return model != NULL && model->has_encoder;
}

bool llama_model_has_decoder(struct llama_model *model) {
    return model != NULL && model->has_decoder;
}

int32_t llama_detokenize(
    const struct llama_vocab *vocab,
    const llama_token *tokens,
    int32_t n_tokens,
    char *text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special
) {
    (void)vocab;
    (void)remove_special;
    (void)unparse_special;

    if (tokens == NULL || text == NULL || n_tokens < 0 || text_len_max < 0) {
        return -1;
    }

    int32_t written = 0;
    for (int32_t i = 0; i < n_tokens; ++i) {
        if (tokens[i] == 1) {
            continue;
        }
        if (written >= text_len_max) {
            return -(written + 1);
        }

        unsigned char raw = (unsigned char)(tokens[i] & 0xFF);
        text[written++] = (raw >= 32 && raw <= 126) ? (char)raw : '?';
    }

    return written;
}

bool llama_token_is_eog(const struct llama_vocab *vocab, llama_token token) {
    (void)vocab;
    return token == 0;
}

void llama_model_free(struct llama_model *model) {
    free(model);
}

void llama_free(struct llama_context *ctx) {
    free(ctx);
}

float *llama_get_logits_ith(struct llama_context *ctx, int32_t index) {
    (void)index;
    if (ctx == NULL) {
        return NULL;
    }
    return ctx->logits;
}

float *llama_get_embeddings(struct llama_context *ctx) {
    if (ctx == NULL) {
        return NULL;
    }
    return ctx->embeddings;
}

struct llama_memory *llama_get_memory(const struct llama_context *ctx) {
    if (ctx == NULL) {
        return NULL;
    }
    return (struct llama_memory *)&ctx->memory;
}

void llama_memory_clear(struct llama_memory *memory, bool reset) {
    if (!reset || memory == NULL || memory->logits == NULL || memory->n_vocab <= 0) {
        return;
    }

    memset(memory->logits, 0, (size_t)memory->n_vocab * sizeof(float));
}
