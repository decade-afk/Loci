package io.github.decadeafk.loci.sdk

import java.io.Closeable

class LociEngine private constructor(
    private var nativeHandle: Long,
) : Closeable {

    companion object {
        fun create(config: LociEngineConfig): LociEngine {
            return if (config.autoDetectDevice) {
                createAuto(config.modelPath, config.contextSize)
            } else {
                create(config.modelPath, config.contextSize, config.gpuLayers)
            }
        }

        fun create(modelPath: String, contextSize: Int, gpuLayers: Int = 0): LociEngine {
            require(modelPath.isNotBlank()) { "modelPath must not be blank" }
            require(contextSize > 0) { "contextSize must be positive" }
            val handle = LociNative.nativeCreateEngine(modelPath, contextSize, gpuLayers)
            check(handle != 0L) { "native engine handle was null" }
            return LociEngine(handle)
        }

        fun createAuto(modelPath: String, contextSize: Int): LociEngine {
            require(modelPath.isNotBlank()) { "modelPath must not be blank" }
            require(contextSize > 0) { "contextSize must be positive" }
            val handle = LociNative.nativeCreateEngineAuto(modelPath, contextSize)
            check(handle != 0L) { "native engine handle was null" }
            return LociEngine(handle)
        }
    }

    fun generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7f,
    ): String {
        validatePrompt(prompt)
        return LociNative.nativeGenerate(
            handleOrThrow(),
            prompt,
            maxTokens,
            temperature,
        )
    }

    fun generate(prompt: String, config: LociGenerationConfig): String {
        return if (config.waitTimeoutMs > 0) {
            generateWait(prompt, config.maxTokens, config.temperature, config.waitTimeoutMs)
        } else {
            generate(prompt, config.maxTokens, config.temperature)
        }
    }

    fun generateWait(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7f,
        waitTimeoutMs: Int = 0,
    ): String {
        validatePrompt(prompt)
        require(waitTimeoutMs >= 0) { "waitTimeoutMs must be >= 0" }
        return LociNative.nativeGenerateWait(
            handleOrThrow(),
            prompt,
            maxTokens,
            temperature,
            waitTimeoutMs,
        )
    }

    fun generateStream(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7f,
        callback: TokenCallback,
    ) {
        validatePrompt(prompt)
        LociNative.nativeGenerateStream(
            handleOrThrow(),
            prompt,
            maxTokens,
            temperature,
            callback,
        )
    }

    fun generateStream(
        prompt: String,
        config: LociGenerationConfig,
        callback: TokenCallback,
    ) {
        generateStream(prompt, config.maxTokens, config.temperature, callback)
    }

    override fun close() {
        val handle = nativeHandle
        if (handle != 0L) {
            nativeHandle = 0L
            LociNative.nativeCloseEngine(handle)
        }
    }

    private fun handleOrThrow(): Long {
        val handle = nativeHandle
        if (handle == 0L) {
            throw LociException("LociEngine is already closed")
        }
        return handle
    }

    private fun validatePrompt(prompt: String) {
        require(prompt.isNotEmpty()) { "prompt must not be empty" }
    }
}
