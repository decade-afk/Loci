package io.github.decadeafk.loci.sdk

class LociException(
    message: String,
    val codeValue: Int = LociErrorCode.UNKNOWN.value,
) : RuntimeException(message) {
    val code: LociErrorCode
        get() = LociErrorCode.fromValue(codeValue)
}
