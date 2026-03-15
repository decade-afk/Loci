package io.github.decadeafk.loci.sdk

enum class LociErrorCode(val value: Int) {
    UNKNOWN(0),
    INVALID_ARGUMENT(1),
    ENGINE_BUSY(2),
    ENGINE_TIMEOUT(3),
    UTF8(4),
    MODEL_LOAD(5),
    GENERATION(6),
    STREAM_CALLBACK(7);

    companion object {
        fun fromValue(value: Int): LociErrorCode {
            return entries.firstOrNull { it.value == value } ?: UNKNOWN
        }
    }
}
