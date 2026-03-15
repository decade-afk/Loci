package io.github.decadeafk.loci.sdk

internal object LociNative {
    init {
        System.loadLibrary("c++_shared")
        System.loadLibrary("loci")
        System.loadLibrary("loci_android_jni")
    }

    external fun nativeCreateEngine(modelPath: String, contextSize: Int, gpuLayers: Int): Long

    external fun nativeCreateEngineAuto(modelPath: String, contextSize: Int): Long

    external fun nativeCloseEngine(handle: Long)

    external fun nativeVersion(): String

    external fun nativeCreateDeviceSelector(): Long

    external fun nativeCloseDeviceSelector(handle: Long)

    external fun nativeGetDeviceCount(handle: Long): Int

    external fun nativeGetDeviceInfo(handle: Long, index: Int): LociDeviceInfo

    external fun nativeAutoSelectDevice(handle: Long): Int

    external fun nativeRecommendDeviceForModel(handle: Long, modelSizeGb: Float): Int

    external fun nativeHasBackend(handle: Long, deviceType: Int): Boolean

    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
    ): String

    external fun nativeGenerateWait(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        waitTimeoutMs: Int,
    ): String

    external fun nativeGenerateStream(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        callback: TokenCallback,
    )
}
