#ifndef LOCI_H
#define LOCI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Opaque handle to the inference engine
typedef struct LociEngine LociEngine;

/**
 * Create a new inference engine
 *
 * @param model_path Path to the GGUF model file
 * @param n_ctx Context size (e.g., 4096)
 * @param n_gpu_layers Number of GPU layers to offload (-1 for all, 0 for CPU only)
 * @return Pointer to engine, or NULL on error
 */
LociEngine* loci_engine_new(const char* model_path, uint32_t n_ctx, int32_t n_gpu_layers);

/**
 * Generate text from a prompt
 *
 * @param engine The inference engine
 * @param prompt Input prompt text
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature (0.0 = greedy, higher = more random)
 * @return Generated text (must be freed with loci_free_string), or NULL on error
 */
char* loci_generate(LociEngine* engine, const char* prompt, uint32_t max_tokens, float temperature);

/**
 * Free a string returned by loci_generate
 *
 * @param s String to free
 */
void loci_free_string(char* s);

/**
 * Destroy an inference engine
 *
 * @param engine The engine to destroy
 */
void loci_engine_free(LociEngine* engine);

/**
 * Get vocabulary size of the loaded model
 *
 * @param engine The inference engine
 * @return Vocabulary size, or 0 on error
 */
uint32_t loci_get_vocab_size(const LociEngine* engine);

#ifdef __cplusplus
}
#endif

#endif // LOCI_H
