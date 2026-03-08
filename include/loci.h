/**
 * @file loci.h
 * @brief Loci C API - Embeddable AI Inference Engine
 * @version 0.1.0
 *
 * This header provides a complete C API for embedding Loci into third-party applications.
 * Supports C/C++/Python/Node.js integration via FFI.
 *
 * Features:
 * - LLM inference with llama.cpp backend
 * - Streaming generation
 * - Plugin system (hot-swappable)
 * - Persistent configuration
 * - Thread-safe serialized engine calls
 *
 * Build artifacts:
 * - Static library: libloci.a (Linux/macOS), loci.lib (Windows)
 * - Dynamic library: libloci.so (Linux), libloci.dylib (macOS), loci.dll (Windows)
 *
 * @example Basic usage:
 * @code
 * LociEngine* engine = loci_engine_new("model.gguf", 2048, 0);
 * char* response = loci_generate(engine, "Hello", 100, 0.7f);
 * printf("%s\n", response);
 * loci_free_string(response);
 * loci_engine_free(engine);
 * @endcode
 */

#ifndef LOCI_H
#define LOCI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Opaque handle to an inference engine instance
 */
typedef struct LociEngine LociEngine;

/**
 * @brief Opaque handle to a plugin registry
 */
typedef struct LociPluginRegistry LociPluginRegistry;

/**
 * @brief Opaque handle to a device selector
 */
typedef struct LociDeviceSelector LociDeviceSelector;

/**
 * @brief Device type enumeration
 */
typedef enum {
    LOCI_DEVICE_CPU = 0,      /**< CPU only (fallback) */
    LOCI_DEVICE_CUDA = 1,     /**< NVIDIA CUDA */
    LOCI_DEVICE_METAL = 2,    /**< Apple Metal (Apple Silicon) */
    LOCI_DEVICE_VULKAN = 3,   /**< Cross-platform Vulkan */
    LOCI_DEVICE_ROCM = 4,     /**< AMD ROCm */
    LOCI_DEVICE_OPENCL = 5    /**< OpenCL (fallback GPU) */
} LociDeviceType;

/**
 * @brief Device information structure
 */
typedef struct {
    int32_t device_id;           /**< Device ID (0-based) */
    char name[256];               /**< Device name/model */
    uint64_t memory_bytes;        /**< Total memory in bytes */
    int32_t device_type;          /**< Device type (LociDeviceType) */
    float compute_capability;     /**< Compute capability (CUDA) or equivalent */
    bool available;               /**< Is device available and usable */
} LociDeviceInfo;

/**
 * @brief Callback function for streaming inference
 * @param token The generated token (null-terminated string)
 * @param user_data User-provided data passed through
 * @return true to continue generation, false to stop
 *
 * Note: callbacks should not call generation/destroy APIs on the same
 * `LociEngine*` reentrantly. Engine calls are serialized.
 */
typedef bool (*LociStreamCallback)(const char* token, void* user_data);

// ============================================================================
// Inference Engine API
// ============================================================================

/**
 * @brief Create a new inference engine
 * @param model_path Path to the GGUF model file
 * @param n_ctx Context size (e.g., 2048, 4096)
 * @param n_gpu_layers Number of layers to offload to GPU (0 for CPU-only, -1 for all)
 * @return Pointer to engine, or NULL on error
 *
 * @example
 * LociEngine* engine = loci_engine_new("model.gguf", 4096, -1); // Use GPU
 */
LociEngine* loci_engine_new(const char* model_path, uint32_t n_ctx, int32_t n_gpu_layers);

/**
 * @brief Generate text from a prompt
 * @param engine Valid engine pointer
 * @param prompt Input prompt (null-terminated)
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature (0.0 to 2.0, typical 0.7)
 * @return Generated text (must be freed with loci_free_string), or NULL on error
 *
 * Current safety guard defaults to ~24 KiB UTF-8 bytes and can be configured
 * with environment variable `LOCI_MAX_PROMPT_BYTES` (minimum 1024).
 * Long prompts are tokenized internally in UTF-8-safe chunks for stability.
 *
 * If the same engine is currently executing another call, this returns NULL
 * and `loci_get_last_error()` reports that the engine is busy.
 *
 * @example
 * char* response = loci_generate(engine, "What is AI?", 200, 0.8f);
 * printf("%s\n", response);
 * loci_free_string(response);
 */
char* loci_generate(
    LociEngine* engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
);

/**
 * @brief Generate text from an explicit UTF-8 byte buffer
 * @param engine Valid engine pointer
 * @param prompt UTF-8 byte buffer
 * @param prompt_len Prompt byte length
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @return Generated text (must be freed with loci_free_string), or NULL on error
 */
char* loci_generate_with_len(
    LociEngine* engine,
    const char* prompt,
    uint32_t prompt_len,
    uint32_t max_tokens,
    float temperature
);

/**
 * @brief Generate text from a prompt and wait for engine lock if busy
 * @param engine Valid engine pointer
 * @param prompt Input prompt (null-terminated)
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @param wait_timeout_ms Lock wait timeout in milliseconds (0 = no wait)
 * @return Generated text (must be freed with loci_free_string), or NULL on error
 *
 * If waiting times out, `loci_get_last_error()` reports lock timeout.
 */
char* loci_generate_wait(
    LociEngine* engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature,
    uint32_t wait_timeout_ms
);

/**
 * @brief Generate text from explicit UTF-8 bytes and wait for engine lock if busy
 * @param engine Valid engine pointer
 * @param prompt UTF-8 byte buffer
 * @param prompt_len Prompt byte length
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @param wait_timeout_ms Lock wait timeout in milliseconds (0 = no wait)
 * @return Generated text (must be freed with loci_free_string), or NULL on error
 */
char* loci_generate_wait_with_len(
    LociEngine* engine,
    const char* prompt,
    uint32_t prompt_len,
    uint32_t max_tokens,
    float temperature,
    uint32_t wait_timeout_ms
);

/**
 * @brief Generate text with streaming output
 * @param engine Valid engine pointer
 * @param prompt Input prompt (null-terminated)
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @param callback Function called for each generated token
 * @param user_data User data passed to callback
 * @return 0 on success, -1 on error
 *
 * If the same engine is currently executing another call, returns -1 and sets
 * last error to busy.
 *
 * @example
 * bool my_callback(const char* token, void* data) {
 *     printf("%s", token);
 *     fflush(stdout);
 *     return true; // Continue
 * }
 * loci_generate_stream(engine, "Hello", 100, 0.7f, my_callback, NULL);
 */
int loci_generate_stream(
    LociEngine* engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature,
    LociStreamCallback callback,
    void* user_data
);

/**
 * @brief Stream generate from explicit UTF-8 bytes
 * @param engine Valid engine pointer
 * @param prompt UTF-8 byte buffer
 * @param prompt_len Prompt byte length
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @param callback Function called for each generated token
 * @param user_data User data passed to callback
 * @return 0 on success, -1 on error
 */
int loci_generate_stream_with_len(
    LociEngine* engine,
    const char* prompt,
    uint32_t prompt_len,
    uint32_t max_tokens,
    float temperature,
    LociStreamCallback callback,
    void* user_data
);

/**
 * @brief Generate text with streaming output and wait for engine lock if busy
 * @param engine Valid engine pointer
 * @param prompt Input prompt (null-terminated)
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @param callback Function called for each generated token
 * @param user_data User data passed to callback
 * @param wait_timeout_ms Lock wait timeout in milliseconds (0 = no wait)
 * @return 0 on success, -1 on error
 */
int loci_generate_stream_wait(
    LociEngine* engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature,
    LociStreamCallback callback,
    void* user_data,
    uint32_t wait_timeout_ms
);

/**
 * @brief Stream generate from explicit UTF-8 bytes and wait for engine lock if busy
 * @param engine Valid engine pointer
 * @param prompt UTF-8 byte buffer
 * @param prompt_len Prompt byte length
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature
 * @param callback Function called for each generated token
 * @param user_data User data passed to callback
 * @param wait_timeout_ms Lock wait timeout in milliseconds (0 = no wait)
 * @return 0 on success, -1 on error
 */
int loci_generate_stream_wait_with_len(
    LociEngine* engine,
    const char* prompt,
    uint32_t prompt_len,
    uint32_t max_tokens,
    float temperature,
    LociStreamCallback callback,
    void* user_data,
    uint32_t wait_timeout_ms
);

/**
 * @brief Free a string returned by loci_generate
 * @param s String pointer (can be NULL)
 */
void loci_free_string(char* s);

/**
 * @brief Get model vocabulary size
 * @param engine Valid engine pointer
 * @return Vocabulary size, or 0 on error
 */
uint32_t loci_get_vocab_size(const LociEngine* engine);

/**
 * @brief Get model context size
 * @param engine Valid engine pointer
 * @return Context size, or 0 on error
 */
uint32_t loci_get_context_size(const LociEngine* engine);

/**
 * @brief Destroy an inference engine and free resources
 * @param engine Engine pointer (can be NULL)
 *
 * This call waits for in-flight engine work to finish (up to an internal
 * timeout). If timeout happens, the engine is not freed and
 * `loci_get_last_error()` reports lock-timeout.
 */
void loci_engine_free(LociEngine* engine);

/**
 * @brief Destroy an inference engine, then set caller pointer to NULL
 * @param engine_ptr Address of engine pointer (LociEngine**)
 *
 * This helper reduces use-after-free/double-free risks in host applications.
 * Pointer is set to NULL only when free succeeds.
 *
 * @example
 * LociEngine* engine = loci_engine_new("model.gguf", 2048, 0);
 * loci_engine_free_safe(&engine);
 * // engine is now NULL
 */
void loci_engine_free_safe(LociEngine** engine_ptr);

// ============================================================================
// Plugin Registry API
// ============================================================================

/**
 * @brief Create a new plugin registry
 * @return Pointer to registry, or NULL on error
 *
 * The registry manages both static (compiled-in) and dynamic (hot-swappable) plugins.
 */
LociPluginRegistry* loci_registry_new(void);

/**
 * @brief Load a plugin from a file path
 * @param registry Valid registry pointer
 * @param plugin_path Path to plugin (.dll/.so/.dylib/.wasm)
 * @return 0 on success, -1 on error
 *
 * `.wasm` extension is loaded via WASM sandbox path; others use dynamic loader.
 *
 * @example
 * loci_registry_load_plugin(registry, "plugins/filter.dll");
 */
int loci_registry_load_plugin(LociPluginRegistry* registry, const char* plugin_path);

/**
 * @brief Unload a hot-swappable plugin by name
 * @param registry Valid registry pointer
 * @param plugin_name Plugin name (null-terminated)
 * @return 0 on success, -1 on error
 *
 * Dynamic/WASM plugins are unloadable. Static plugins cannot be unloaded.
 */
int loci_registry_unload_plugin(LociPluginRegistry* registry, const char* plugin_name);

/**
 * @brief Reload a hot-swappable plugin by name
 * @param registry Valid registry pointer
 * @param plugin_name Plugin name (null-terminated)
 * @return 0 on success, -1 on error
 *
 * Dynamic/WASM plugins are reloadable. Static plugins cannot be reloaded.
 */
int loci_registry_reload_plugin(LociPluginRegistry* registry, const char* plugin_name);

/**
 * @brief Enable a plugin by name
 * @param registry Valid registry pointer
 * @param plugin_name Plugin name (null-terminated)
 * @return 0 on success, -1 on error
 */
int loci_registry_enable_plugin(LociPluginRegistry* registry, const char* plugin_name);

/**
 * @brief Disable a plugin by name (keeps it loaded but inactive)
 * @param registry Valid registry pointer
 * @param plugin_name Plugin name (null-terminated)
 * @return 0 on success, -1 on error
 */
int loci_registry_disable_plugin(LociPluginRegistry* registry, const char* plugin_name);

/**
 * @brief Get total number of registered plugins
 * @param registry Valid registry pointer
 * @return Plugin count, or -1 on error
 */
int loci_registry_count(const LociPluginRegistry* registry);

/**
 * @brief Get detailed plugin list as JSON
 * @param registry Valid registry pointer
 * @return JSON string (must be freed with loci_free_string), or NULL on error
 *
 * JSON shape:
 * [
 *   {
 *     "name":"...",
 *     "version":"...",
 *     "enabled":true,
 *     "plugin_type":"dynamic|wasm|static",
 *     "source":"path-or-null",
 *     "hot_reloadable":true
 *   }
 * ]
 */
char* loci_registry_list_json(const LociPluginRegistry* registry);

/**
 * @brief Save plugin configuration to TOML file
 * @param registry Valid registry pointer
 * @param config_path Path to config file
 * @return 0 on success, -1 on error
 *
 * @example
 * loci_registry_save(registry, "plugins.toml");
 */
int loci_registry_save(LociPluginRegistry* registry, const char* config_path);

/**
 * @brief Load plugin configuration from TOML file
 * @param registry Valid registry pointer
 * @param config_path Path to config file
 * @return 0 on success, -1 on error
 *
 * @example
 * loci_registry_load(registry, "plugins.toml");
 */
int loci_registry_load(LociPluginRegistry* registry, const char* config_path);

/**
 * @brief Destroy a plugin registry and free resources
 * @param registry Registry pointer (can be NULL)
 */
void loci_registry_free(LociPluginRegistry* registry);

// ============================================================================
// Version and Info API
// ============================================================================

/**
 * @brief Get Loci version string
 * @return Version string (e.g., "0.1.0")
 */
const char* loci_version(void);

/**
 * @brief Check if GPU support is available
 * @return true if GPU is available, false otherwise
 */
bool loci_has_gpu_support(void);

/**
 * @brief Get last error message (thread-local)
 * @return Error message, or NULL if no error
 *
 * Call this after any function returns an error code to get details.
 */
const char* loci_get_last_error(void);

// ============================================================================
// Device Detection and Auto-Selection API (NEW)
// ============================================================================

/**
 * @brief Create a new device selector for automatic hardware detection
 * @return Pointer to device selector, or NULL on error
 *
 * The selector automatically detects all available devices (CPU, CUDA, Metal, etc.)
 *
 * @example
 * LociDeviceSelector* selector = loci_device_selector_new();
 */
LociDeviceSelector* loci_device_selector_new(void);

/**
 * @brief Free a device selector
 * @param selector Selector pointer (can be NULL)
 */
void loci_device_selector_free(LociDeviceSelector* selector);

/**
 * @brief Get number of detected devices
 * @param selector Valid selector pointer
 * @return Number of devices, or -1 on error
 *
 * @example
 * int count = loci_get_device_count(selector);
 * printf("Found %d devices\n", count);
 */
int32_t loci_get_device_count(const LociDeviceSelector* selector);

/**
 * @brief Get device information by index
 * @param selector Valid selector pointer
 * @param index Device index (0-based)
 * @param info Pointer to LociDeviceInfo structure to fill
 * @return 0 on success, -1 on error
 *
 * @example
 * LociDeviceInfo info;
 * if (loci_get_device_info(selector, 0, &info) == 0) {
 *     printf("Device: %s (%llu GB)\n", info.name, info.memory_bytes / (1024*1024*1024));
 * }
 */
int32_t loci_get_device_info(
    const LociDeviceSelector* selector,
    int32_t index,
    LociDeviceInfo* info
);

/**
 * @brief Automatically select the best available device
 * @param selector Valid selector pointer
 * @return Device ID of best device, or -1 on error
 *
 * Priority: CUDA > Metal > Vulkan > ROCm > OpenCL > CPU
 *
 * @example
 * int device_id = loci_auto_select_device(selector);
 */
int32_t loci_auto_select_device(const LociDeviceSelector* selector);

/**
 * @brief Get recommended device for a specific model size
 * @param selector Valid selector pointer
 * @param model_size_gb Estimated model size in GB (e.g., 7.0 for 7B model)
 * @return Device ID, or -1 on error
 *
 * Considers available memory and automatically selects partial GPU offloading if needed.
 *
 * @example
 * int device_id = loci_recommend_device_for_model(selector, 7.0); // For 7GB model
 */
int32_t loci_recommend_device_for_model(
    const LociDeviceSelector* selector,
    float model_size_gb
);

/**
 * @brief Check if a specific backend is available
 * @param selector Valid selector pointer
 * @param device_type Device type to check (LociDeviceType)
 * @return true if backend is available, false otherwise
 *
 * @example
 * if (loci_has_backend(selector, LOCI_DEVICE_CUDA)) {
 *     printf("CUDA is available!\n");
 * }
 */
bool loci_has_backend(
    const LociDeviceSelector* selector,
    int32_t device_type
);

/**
 * @brief Create inference engine with automatic device selection
 * @param model_path Path to the GGUF model file
 * @param n_ctx Context size (e.g., 2048, 4096)
 * @return Pointer to engine, or NULL on error
 *
 * Automatically detects and selects the best available device (GPU or CPU)
 *
 * @example
 * LociEngine* engine = loci_engine_new_auto("model.gguf", 4096);
 * // Automatically uses GPU if available, otherwise CPU
 */
LociEngine* loci_engine_new_auto(const char* model_path, uint32_t n_ctx);

/**
 * @brief Create inference engine with specific device
 * @param model_path Path to the GGUF model file
 * @param n_ctx Context size
 * @param device_id Device ID (from loci_get_device_info)
 * @param n_gpu_layers Number of GPU layers (-1 for all)
 * @return Pointer to engine, or NULL on error
 *
 * @example
 * LociEngine* engine = loci_engine_new_with_device("model.gguf", 4096, 0, -1);
 */
LociEngine* loci_engine_new_with_device(
    const char* model_path,
    uint32_t n_ctx,
    int32_t device_id,
    int32_t n_gpu_layers
);

// ============================================================================
// Integration Examples
// ============================================================================

/**
 * @example Complete C++ Integration Example
 * @code
 * #include "loci.h"
 * #include <iostream>
 *
 * int main() {
 *     // Create engine
 *     LociEngine* engine = loci_engine_new("model.gguf", 4096, -1);
 *     if (!engine) {
 *         std::cerr << "Failed to load model" << std::endl;
 *         return 1;
 *     }
 *
 *     // Generate text
 *     char* response = loci_generate(engine, "Explain AI in simple terms", 200, 0.7f);
 *     if (response) {
 *         std::cout << response << std::endl;
 *         loci_free_string(response);
 *     }
 *
 *     // Cleanup
 *     loci_engine_free(engine);
 *     return 0;
 * }
 * @endcode
 */

/**
 * @example Python Integration (via ctypes)
 * @code
 * from ctypes import *
 *
 * # Load library
 * loci = CDLL("loci.dll")  # or libloci.so on Linux
 *
 * # Setup function signatures
 * loci.loci_engine_new.restype = c_void_p
 * loci.loci_generate.restype = c_char_p
 *
 * # Create engine
 * engine = loci.loci_engine_new(b"model.gguf", 2048, 0)
 *
 * # Generate
 * response = loci.loci_generate(engine, b"Hello", 100, 0.7)
 * print(response.decode('utf-8'))
 *
 * # Cleanup
 * loci.loci_free_string(response)
 * loci.loci_engine_free(engine)
 * @endcode
 */

/**
 * @example Plugin System Usage
 * @code
 * // Create registry
 * LociPluginRegistry* registry = loci_registry_new();
 *
 * // Load plugins
 * loci_registry_load_plugin(registry, "plugins/filter.dll");
 * loci_registry_load_plugin(registry, "plugins/logger.dll");
 *
 * // Control plugins
 * loci_registry_disable_plugin(registry, "logger");
 *
 * // Save configuration
 * loci_registry_save(registry, "plugins.toml");
 *
 * // Cleanup
 * loci_registry_free(registry);
 * @endcode
 */

#ifdef __cplusplus
}
#endif

#endif /* LOCI_H */
