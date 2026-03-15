#include <jni.h>

#include <cstdint>
#include <limits>
#include <string>

#include "loci.h"

namespace {

constexpr const char *kExceptionClass = "io/github/decadeafk/loci/sdk/LociException";
constexpr const char *kCallbackClass = "io/github/decadeafk/loci/sdk/TokenCallback";
constexpr const char *kCallbackMethod = "onToken";
constexpr const char *kCallbackSignature = "(Ljava/lang/String;)Z";
constexpr const char *kDeviceInfoClass = "io/github/decadeafk/loci/sdk/LociDeviceInfo";
constexpr const char *kDeviceInfoCtor = "(ILjava/lang/String;JIFZ)V";

enum LociAndroidErrorCode : jint {
    kUnknown = 0,
    kInvalidArgument = 1,
    kEngineBusy = 2,
    kEngineTimeout = 3,
    kUtf8 = 4,
    kModelLoad = 5,
    kGeneration = 6,
    kStreamCallback = 7,
};

std::string last_error_message() {
    const char *error = loci_get_last_error();
    if (error != nullptr && error[0] != '\0') {
        return std::string(error);
    }
    return "unknown Loci native error";
}

LociAndroidErrorCode classify_error_code(const std::string &message) {
    if (message.find("busy") != std::string::npos) {
        return kEngineBusy;
    }
    if (message.find("timeout") != std::string::npos) {
        return kEngineTimeout;
    }
    if (message.find("UTF-8") != std::string::npos || message.find("utf-8") != std::string::npos) {
        return kUtf8;
    }
    if (message.find("null") != std::string::npos || message.find("must not be null") != std::string::npos ||
        message.find("out of bounds") != std::string::npos || message.find("negative") != std::string::npos) {
        return kInvalidArgument;
    }
    if (message.find("failed to create inference engine") != std::string::npos ||
        message.find("Failed to load model") != std::string::npos ||
        message.find("model") != std::string::npos) {
        return kModelLoad;
    }
    if (message.find("stream callback failed") != std::string::npos) {
        return kStreamCallback;
    }
    if (message.find("generation failed") != std::string::npos ||
        message.find("stream generation failed") != std::string::npos) {
        return kGeneration;
    }
    return kUnknown;
}

void throw_java_exception(JNIEnv *env, const std::string &message) {
    jclass exception_class = env->FindClass(kExceptionClass);
    if (exception_class == nullptr) {
        env->ExceptionClear();
        exception_class = env->FindClass("java/lang/RuntimeException");
        if (exception_class == nullptr) {
            return;
        }
    }

    jmethodID ctor = env->GetMethodID(exception_class, "<init>", "(Ljava/lang/String;I)V");
    if (ctor == nullptr) {
        env->ExceptionClear();
        env->ThrowNew(exception_class, message.c_str());
        return;
    }

    jstring java_message = env->NewStringUTF(message.c_str());
    if (java_message == nullptr) {
        return;
    }

    jobject exception_obj = env->NewObject(
        exception_class,
        ctor,
        java_message,
        static_cast<jint>(classify_error_code(message)));
    env->DeleteLocalRef(java_message);
    if (exception_obj == nullptr) {
        return;
    }
    env->Throw(reinterpret_cast<jthrowable>(exception_obj));
}

bool jstring_to_utf8(JNIEnv *env, jstring value, std::string *out) {
    if (value == nullptr) {
        throw_java_exception(env, "string argument must not be null");
        return false;
    }

    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return false;
    }

    out->assign(chars);
    env->ReleaseStringUTFChars(value, chars);
    return true;
}

jstring utf8_to_jstring(JNIEnv *env, const char *value) {
    if (value == nullptr) {
        return nullptr;
    }
    return env->NewStringUTF(value);
}

jobject make_device_info(JNIEnv *env, const LociDeviceInfo &info) {
    jclass device_info_class = env->FindClass(kDeviceInfoClass);
    if (device_info_class == nullptr) {
        return nullptr;
    }

    jmethodID ctor = env->GetMethodID(device_info_class, "<init>", kDeviceInfoCtor);
    if (ctor == nullptr) {
        env->DeleteLocalRef(device_info_class);
        return nullptr;
    }

    jstring name = env->NewStringUTF(info.name);
    if (name == nullptr) {
        env->DeleteLocalRef(device_info_class);
        return nullptr;
    }

    jobject result = env->NewObject(
        device_info_class,
        ctor,
        static_cast<jint>(info.device_id),
        name,
        static_cast<jlong>(info.memory_bytes),
        static_cast<jint>(info.device_type),
        static_cast<jfloat>(info.compute_capability),
        static_cast<jboolean>(info.available));
    env->DeleteLocalRef(name);
    env->DeleteLocalRef(device_info_class);
    return result;
}

::LociDeviceSelector *require_selector(JNIEnv *env, jlong handle) {
    if (handle == 0) {
        throw_java_exception(env, "LociDeviceSelector is already closed");
        return nullptr;
    }
    return reinterpret_cast<::LociDeviceSelector *>(static_cast<intptr_t>(handle));
}

::LociEngine *require_engine(JNIEnv *env, jlong handle) {
    if (handle == 0) {
        throw_java_exception(env, "LociEngine is already closed");
        return nullptr;
    }
    return reinterpret_cast<::LociEngine *>(static_cast<intptr_t>(handle));
}

struct StreamCallbackContext {
    JNIEnv *env;
    jobject callback_ref;
    jmethodID on_token_method;
    bool callback_failed;
};

bool forward_token_callback(const char *token, void *user_data) {
    auto *context = reinterpret_cast<StreamCallbackContext *>(user_data);
    jstring token_string = utf8_to_jstring(context->env, token != nullptr ? token : "");
    if (token_string == nullptr) {
        context->callback_failed = true;
        return false;
    }

    jboolean keep_going = context->env->CallBooleanMethod(
        context->callback_ref,
        context->on_token_method,
        token_string);
    context->env->DeleteLocalRef(token_string);

    if (context->env->ExceptionCheck()) {
        context->env->ExceptionClear();
        context->callback_failed = true;
        return false;
    }

    return keep_going == JNI_TRUE;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeCreateEngine(
    JNIEnv *env,
    jclass,
    jstring model_path,
    jint context_size,
    jint gpu_layers) {
    std::string model_path_utf8;
    if (!jstring_to_utf8(env, model_path, &model_path_utf8)) {
        return 0;
    }

    auto *engine = loci_engine_new(
        model_path_utf8.c_str(),
        static_cast<uint32_t>(context_size),
        static_cast<int32_t>(gpu_layers));
    if (engine == nullptr) {
        throw_java_exception(env, last_error_message());
        return 0;
    }

    return static_cast<jlong>(reinterpret_cast<intptr_t>(engine));
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeCreateEngineAuto(
    JNIEnv *env,
    jclass,
    jstring model_path,
    jint context_size) {
    std::string model_path_utf8;
    if (!jstring_to_utf8(env, model_path, &model_path_utf8)) {
        return 0;
    }

    auto *engine = loci_engine_new_auto(
        model_path_utf8.c_str(),
        static_cast<uint32_t>(context_size));
    if (engine == nullptr) {
        throw_java_exception(env, last_error_message());
        return 0;
    }

    return static_cast<jlong>(reinterpret_cast<intptr_t>(engine));
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeCloseEngine(
    JNIEnv *,
    jclass,
    jlong handle) {
    auto *engine = reinterpret_cast<::LociEngine *>(static_cast<intptr_t>(handle));
    if (engine != nullptr) {
        loci_engine_free(engine);
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeVersion(
    JNIEnv *env,
    jclass) {
    return utf8_to_jstring(env, loci_version());
}

extern "C" JNIEXPORT jlong JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeCreateDeviceSelector(
    JNIEnv *env,
    jclass) {
    auto *selector = loci_device_selector_new();
    if (selector == nullptr) {
        throw_java_exception(env, last_error_message());
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(selector));
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeCloseDeviceSelector(
    JNIEnv *,
    jclass,
    jlong handle) {
    auto *selector = reinterpret_cast<::LociDeviceSelector *>(static_cast<intptr_t>(handle));
    if (selector != nullptr) {
        loci_device_selector_free(selector);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeGetDeviceCount(
    JNIEnv *env,
    jclass,
    jlong handle) {
    auto *selector = require_selector(env, handle);
    if (selector == nullptr) {
        return -1;
    }

    int32_t count = loci_get_device_count(selector);
    if (count < 0) {
        throw_java_exception(env, last_error_message());
    }
    return static_cast<jint>(count);
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeGetDeviceInfo(
    JNIEnv *env,
    jclass,
    jlong handle,
    jint index) {
    auto *selector = require_selector(env, handle);
    if (selector == nullptr) {
        return nullptr;
    }

    LociDeviceInfo info{};
    if (loci_get_device_info(selector, static_cast<int32_t>(index), &info) != 0) {
        throw_java_exception(env, last_error_message());
        return nullptr;
    }

    return make_device_info(env, info);
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeAutoSelectDevice(
    JNIEnv *env,
    jclass,
    jlong handle) {
    auto *selector = require_selector(env, handle);
    if (selector == nullptr) {
        return -1;
    }

    int32_t device_id = loci_auto_select_device(selector);
    if (device_id < 0) {
        throw_java_exception(env, last_error_message());
    }
    return static_cast<jint>(device_id);
}

extern "C" JNIEXPORT jint JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeRecommendDeviceForModel(
    JNIEnv *env,
    jclass,
    jlong handle,
    jfloat model_size_gb) {
    auto *selector = require_selector(env, handle);
    if (selector == nullptr) {
        return -1;
    }

    int32_t device_id = loci_recommend_device_for_model(selector, model_size_gb);
    if (device_id < 0) {
        throw_java_exception(env, last_error_message());
    }
    return static_cast<jint>(device_id);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeHasBackend(
    JNIEnv *env,
    jclass,
    jlong handle,
    jint device_type) {
    auto *selector = require_selector(env, handle);
    if (selector == nullptr) {
        return JNI_FALSE;
    }

    return loci_has_backend(selector, static_cast<int32_t>(device_type)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeGenerate(
    JNIEnv *env,
    jclass,
    jlong handle,
    jstring prompt,
    jint max_tokens,
    jfloat temperature) {
    auto *engine = require_engine(env, handle);
    if (engine == nullptr) {
        return nullptr;
    }

    std::string prompt_utf8;
    if (!jstring_to_utf8(env, prompt, &prompt_utf8)) {
        return nullptr;
    }

    if (prompt_utf8.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw_java_exception(env, "prompt exceeds native API length limit");
        return nullptr;
    }

    char *result = loci_generate_with_len(
        engine,
        prompt_utf8.data(),
        static_cast<uint32_t>(prompt_utf8.size()),
        static_cast<uint32_t>(max_tokens),
        temperature);
    if (result == nullptr) {
        throw_java_exception(env, last_error_message());
        return nullptr;
    }

    jstring response = utf8_to_jstring(env, result);
    loci_free_string(result);
    return response;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeGenerateWait(
    JNIEnv *env,
    jclass,
    jlong handle,
    jstring prompt,
    jint max_tokens,
    jfloat temperature,
    jint wait_timeout_ms) {
    auto *engine = require_engine(env, handle);
    if (engine == nullptr) {
        return nullptr;
    }

    std::string prompt_utf8;
    if (!jstring_to_utf8(env, prompt, &prompt_utf8)) {
        return nullptr;
    }

    if (prompt_utf8.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw_java_exception(env, "prompt exceeds native API length limit");
        return nullptr;
    }

    char *result = loci_generate_wait_with_len(
        engine,
        prompt_utf8.data(),
        static_cast<uint32_t>(prompt_utf8.size()),
        static_cast<uint32_t>(max_tokens),
        temperature,
        static_cast<uint32_t>(wait_timeout_ms));
    if (result == nullptr) {
        throw_java_exception(env, last_error_message());
        return nullptr;
    }

    jstring response = utf8_to_jstring(env, result);
    loci_free_string(result);
    return response;
}

extern "C" JNIEXPORT void JNICALL
Java_io_github_decadeafk_loci_sdk_LociNative_nativeGenerateStream(
    JNIEnv *env,
    jclass,
    jlong handle,
    jstring prompt,
    jint max_tokens,
    jfloat temperature,
    jobject callback) {
    auto *engine = require_engine(env, handle);
    if (engine == nullptr) {
        return;
    }

    if (callback == nullptr) {
        throw_java_exception(env, "stream callback must not be null");
        return;
    }

    std::string prompt_utf8;
    if (!jstring_to_utf8(env, prompt, &prompt_utf8)) {
        return;
    }

    if (prompt_utf8.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw_java_exception(env, "prompt exceeds native API length limit");
        return;
    }

    jclass callback_class = env->FindClass(kCallbackClass);
    if (callback_class == nullptr) {
        throw_java_exception(env, "TokenCallback class was not found");
        return;
    }

    jmethodID on_token = env->GetMethodID(callback_class, kCallbackMethod, kCallbackSignature);
    env->DeleteLocalRef(callback_class);
    if (on_token == nullptr) {
        throw_java_exception(env, "TokenCallback.onToken(String) method was not found");
        return;
    }

    jobject callback_ref = env->NewGlobalRef(callback);
    if (callback_ref == nullptr) {
        throw_java_exception(env, "failed to create global reference for stream callback");
        return;
    }

    StreamCallbackContext context{
        env,
        callback_ref,
        on_token,
        false,
    };

    int rc = loci_generate_stream_with_len(
        engine,
        prompt_utf8.data(),
        static_cast<uint32_t>(prompt_utf8.size()),
        static_cast<uint32_t>(max_tokens),
        temperature,
        forward_token_callback,
        &context);

    env->DeleteGlobalRef(callback_ref);

    if (context.callback_failed) {
        throw_java_exception(env, "stream callback failed");
        return;
    }

    if (rc != 0) {
        throw_java_exception(env, last_error_message());
    }
}
