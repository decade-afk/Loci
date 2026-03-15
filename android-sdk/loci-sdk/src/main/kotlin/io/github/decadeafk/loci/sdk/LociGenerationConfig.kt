package io.github.decadeafk.loci.sdk

data class LociGenerationConfig(
    val maxTokens: Int = 256,
    val temperature: Float = 0.7f,
    val waitTimeoutMs: Int = 0,
)
