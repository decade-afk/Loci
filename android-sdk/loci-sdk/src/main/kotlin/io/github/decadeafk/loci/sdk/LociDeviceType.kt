package io.github.decadeafk.loci.sdk

enum class LociDeviceType(val value: Int) {
    CPU(0),
    CUDA(1),
    METAL(2),
    VULKAN(3),
    ROCM(4),
    OPENCL(5);

    companion object {
        fun fromValue(value: Int): LociDeviceType {
            return entries.firstOrNull { it.value == value } ?: CPU
        }
    }
}
