package io.github.decadeafk.loci.sdk

object LociRuntime {
    fun version(): String = LociNative.nativeVersion()

    fun enumerateDevices(): List<LociDeviceInfo> {
        LociDeviceSelector.create().use { selector ->
            return selector.listDevices()
        }
    }
}
