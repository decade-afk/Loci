package io.github.decadeafk.loci.sdk

import java.io.Closeable

class LociDeviceSelector private constructor(
    private var nativeHandle: Long,
) : Closeable {

    companion object {
        fun create(): LociDeviceSelector {
            val handle = LociNative.nativeCreateDeviceSelector()
            check(handle != 0L) { "native device selector handle was null" }
            return LociDeviceSelector(handle)
        }
    }

    fun getDeviceCount(): Int = LociNative.nativeGetDeviceCount(handleOrThrow())

    fun getDeviceInfo(index: Int): LociDeviceInfo {
        require(index >= 0) { "index must be >= 0" }
        return LociNative.nativeGetDeviceInfo(handleOrThrow(), index)
    }

    fun listDevices(): List<LociDeviceInfo> {
        return List(getDeviceCount()) { index -> getDeviceInfo(index) }
    }

    fun autoSelectDeviceId(): Int = LociNative.nativeAutoSelectDevice(handleOrThrow())

    fun recommendDeviceForModel(modelSizeGb: Float): Int {
        require(modelSizeGb >= 0f) { "modelSizeGb must be >= 0" }
        return LociNative.nativeRecommendDeviceForModel(handleOrThrow(), modelSizeGb)
    }

    fun hasBackend(deviceType: LociDeviceType): Boolean {
        return LociNative.nativeHasBackend(handleOrThrow(), deviceType.value)
    }

    override fun close() {
        val handle = nativeHandle
        if (handle != 0L) {
            nativeHandle = 0L
            LociNative.nativeCloseDeviceSelector(handle)
        }
    }

    private fun handleOrThrow(): Long {
        val handle = nativeHandle
        if (handle == 0L) {
            throw LociException("LociDeviceSelector is already closed")
        }
        return handle
    }
}
