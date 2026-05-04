#include "bridge.h"

#include <cstring>
#include <cstdlib>

const char* loci_ov_bridge_version(void) {
    return "0.1.0-bridge-skeleton";
}

void loci_ov_bridge_free_string(char* ptr) {
    if (ptr != nullptr) {
        std::free(ptr);
    }
}

char* loci_ov_bridge_dup_cstr(const char* input) {
    if (input == nullptr) {
        return nullptr;
    }

    const size_t len = std::strlen(input);
    char* out = static_cast<char*>(std::malloc(len + 1));
    if (out == nullptr) {
        return nullptr;
    }

    std::memcpy(out, input, len);
    out[len] = '\0';
    return out;
}

