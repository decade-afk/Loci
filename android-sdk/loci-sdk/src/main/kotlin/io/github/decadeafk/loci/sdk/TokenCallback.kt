package io.github.decadeafk.loci.sdk

fun interface TokenCallback {
    fun onToken(token: String): Boolean
}
