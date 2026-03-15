package io.github.decadeafk.loci.sdk

data class LociDeviceInfo(
    val id: Int,
    val name: String,
    val memoryBytes: Long,
    val deviceTypeValue: Int,
    val computeCapability: Float,
    val available: Boolean,
) {
    val deviceType: LociDeviceType
        get() = LociDeviceType.fromValue(deviceTypeValue)
}
