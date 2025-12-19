/**
 * Loci Core - FFI C Header
 *
 * C/C++ header file for the Loci AI inference library.
 * Use this header when linking against loci from C, C++, or other languages.
 *
 * Library files:
 * - Linux:   libloci.so (dynamic), libloci.a (static)
 * - macOS:   libloci.dylib (dynamic), libloci.a (static)
 * - Windows: loci.dll (dynamic), loci.lib (static)
 *
 * Usage example (C):
 * ```c
 * #include "loci.h"
 *
 * int main() {
 *     // Create AI service
 *     void* service = loci_ai_service_new();
 *
 *     // Configure model
 *     const char* config = "{\"model_path\":\"/path/to/model.gguf\",\"context_size\":4096}";
 *     char* error = NULL;
 *     int result = loci_ai_service_update_config(service, config, &error);
 *     if (result != 0) {
 *         printf("Config error: %s\n", error);
 *         loci_string_free(error);
 *         return 1;
 *     }
 *
 *     // Load model
 *     result = loci_ai_service_load_model(service, &error);
 *     if (result != 0) {
 *         printf("Load error: %s\n", error);
 *         loci_string_free(error);
 *         return 1;
 *     }
 *
 *     // Generate text
 *     const char* request = "{\"prompt\":\"Hello\",\"max_tokens\":100}";
 *     char* response = NULL;
 *     result = loci_ai_service_generate(service, request, &response, &error);
 *     if (result == 0) {
 *         printf("Response: %s\n", response);
 *         loci_string_free(response);
 *     } else {
 *         printf("Generate error: %s\n", error);
 *         loci_string_free(error);
 *     }
 *
 *     // Cleanup
 *     loci_ai_service_free(service);
 *     return 0;
 * }
 * ```
 *
 * Compile: gcc main.c -L. -lloci -o main
 * Run:     LD_LIBRARY_PATH=. ./main  (Linux/macOS)
 */

#ifndef LOCI_H
#define LOCI_H

#ifdef __cplusplus
extern "C" {
#endif

// ==================== Error Codes ====================

#define LOCI_SUCCESS                 0
#define LOCI_ERROR_NULL_POINTER     -1
#define LOCI_ERROR_MODEL_NOT_LOADED -2
#define LOCI_ERROR_CONFIG_ERROR     -3
#define LOCI_ERROR_INFERENCE_ERROR  -4
#define LOCI_ERROR_IO_ERROR         -5
#define LOCI_ERROR_UNKNOWN          -99

// ==================== Version ====================

/**
 * Get the version of the loci library
 *
 * Returns: Version string (e.g., "0.1.0"), do NOT free this string
 */
const char* loci_version(void);

// ==================== String Management ====================

/**
 * Free a string allocated by loci
 *
 * @param s String pointer returned by loci functions
 *
 * Safety: Only call this on strings returned by loci functions
 */
void loci_string_free(char* s);

// ==================== AI Service ====================

/**
 * Create a new AI service instance
 *
 * Returns: Opaque pointer to AIService, or NULL on failure
 *          Must be freed with loci_ai_service_free()
 */
void* loci_ai_service_new(void);

/**
 * Free an AI service instance
 *
 * @param service AI service pointer from loci_ai_service_new()
 *
 * Safety: Do not use the pointer after calling this function
 */
void loci_ai_service_free(void* service);

/**
 * Update AI service configuration
 *
 * @param service     AI service instance
 * @param config_json JSON configuration string
 * @param error_out   Output parameter for error message (optional, pass NULL to ignore)
 *
 * Returns: 0 on success, negative error code on failure
 *
 * Config JSON format:
 * {
 *   "model_path": "/path/to/model.gguf",
 *   "context_size": 4096,
 *   "gpu_layers": 32,
 *   "threads": 8,
 *   "batch_size": 512,
 *   "temperature": 0.7,
 *   "top_k": 40,
 *   "top_p": 0.95
 * }
 */
int loci_ai_service_update_config(
    void* service,
    const char* config_json,
    char** error_out
);

/**
 * Load the AI model
 *
 * @param service   AI service instance
 * @param error_out Output parameter for error message (optional)
 *
 * Returns: 0 on success, negative error code on failure
 */
int loci_ai_service_load_model(void* service, char** error_out);

/**
 * Unload the current model
 *
 * @param service AI service instance
 *
 * Returns: 0 on success, negative error code on failure
 */
int loci_ai_service_unload_model(void* service);

/**
 * Check if model is loaded
 *
 * @param service AI service instance
 *
 * Returns: 1 if loaded, 0 if not loaded, negative error code on failure
 */
int loci_ai_service_is_model_loaded(void* service);

/**
 * Generate text
 *
 * @param service      AI service instance
 * @param request_json JSON request string
 * @param response_out Output parameter for response JSON (caller must free with loci_string_free)
 * @param error_out    Output parameter for error message (optional)
 *
 * Returns: 0 on success, negative error code on failure
 *
 * Request JSON format:
 * {
 *   "prompt": "Your prompt here",
 *   "system_prompt": "Optional system prompt",
 *   "max_tokens": 512,
 *   "temperature": 0.7,
 *   "stop_words": ["Optional", "stop", "words"]
 * }
 *
 * Response JSON format:
 * {
 *   "content": "Generated text",
 *   "tokens_generated": 123,
 *   "stopped_early": false
 * }
 */
int loci_ai_service_generate(
    void* service,
    const char* request_json,
    char** response_out,
    char** error_out
);

/**
 * Get current configuration
 *
 * @param service    AI service instance
 * @param config_out Output parameter for config JSON (caller must free with loci_string_free)
 *
 * Returns: 0 on success, negative error code on failure
 */
int loci_ai_service_get_config(void* service, char** config_out);

// ==================== Agent System ====================

/**
 * Create a new Agent system instance
 *
 * Returns: Opaque pointer to AgentSystem, or NULL on failure
 *          Must be freed with loci_agent_system_free()
 */
void* loci_agent_system_new(void);

/**
 * Free an Agent system instance
 *
 * @param system Agent system pointer from loci_agent_system_new()
 */
void loci_agent_system_free(void* system);

// ==================== System Information ====================

/**
 * Get system information
 *
 * @param info_out  Output parameter for system info JSON (caller must free with loci_string_free)
 * @param error_out Output parameter for error message (optional)
 *
 * Returns: 0 on success, negative error code on failure
 *
 * Response JSON format:
 * {
 *   "cpu": {
 *     "name": "CPU model",
 *     "cores": 8,
 *     "threads": 16
 *   },
 *   "memory": {
 *     "total": 16.0,
 *     "available": 8.0
 *   },
 *   "gpu": {
 *     "available": true,
 *     "name": "GPU model",
 *     "memory": 8.0
 *   }
 * }
 */
int loci_get_system_info(char** info_out, char** error_out);

#ifdef __cplusplus
}
#endif

#endif // LOCI_H
