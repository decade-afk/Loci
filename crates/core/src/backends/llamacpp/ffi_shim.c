#include <stdint.h>
#include "llama.h"

void loci_llama_model_default_params(struct llama_model_params * out_params) {
    if (out_params == NULL) {
        return;
    }
    *out_params = llama_model_default_params();
}

void loci_llama_context_default_params(struct llama_context_params * out_params) {
    if (out_params == NULL) {
        return;
    }
    *out_params = llama_context_default_params();
}

struct llama_model * loci_llama_model_load_from_file(const char * path, const struct llama_model_params * params) {
    if (path == NULL || params == NULL) {
        return NULL;
    }
    return llama_model_load_from_file(path, *params);
}

struct llama_context * loci_llama_init_from_model(struct llama_model * model, const struct llama_context_params * params) {
    if (model == NULL || params == NULL) {
        return NULL;
    }
    return llama_init_from_model(model, *params);
}

void loci_llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max, llama_batch * out_batch) {
    if (out_batch == NULL) {
        return;
    }
    *out_batch = llama_batch_init(n_tokens, embd, n_seq_max);
}

void loci_llama_batch_free(const llama_batch * batch) {
    if (batch == NULL) {
        return;
    }
    llama_batch_free(*batch);
}

int32_t loci_llama_encode(struct llama_context * ctx, const llama_batch * batch) {
    if (ctx == NULL || batch == NULL) {
        return -1;
    }
    return llama_encode(ctx, *batch);
}

int32_t loci_llama_decode(struct llama_context * ctx, const llama_batch * batch) {
    if (ctx == NULL || batch == NULL) {
        return -1;
    }
    return llama_decode(ctx, *batch);
}

int32_t loci_llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special
) {
    if (vocab == NULL || text == NULL || tokens == NULL || text_len < 0 || n_tokens_max <= 0) {
        return -1;
    }
    return llama_tokenize(
        vocab,
        text,
        text_len,
        tokens,
        n_tokens_max,
        add_special,
        parse_special
    );
}
