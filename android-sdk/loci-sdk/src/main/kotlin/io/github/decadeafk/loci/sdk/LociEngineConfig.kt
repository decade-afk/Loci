package io.github.decadeafk.loci.sdk

data class LociEngineConfig(
    val modelPath: String,
    val contextSize: Int = 2048,
    val gpuLayers: Int = 0,
    val autoDetectDevice: Boolean = false,
)
