cmake_minimum_required(VERSION 3.22.1)
project(loci_android_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LOCI_REPO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include")
set(LOCI_PREBUILT_SO "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libloci.so")

if (NOT EXISTS "${LOCI_PREBUILT_SO}")
    message(FATAL_ERROR "Missing ${LOCI_PREBUILT_SO}. Run android-sdk/scripts/sync-prebuilt-loci.sh or .ps1 first.")
endif()

add_library(loci SHARED IMPORTED)
set_target_properties(
    loci
    PROPERTIES
    IMPORTED_LOCATION "${LOCI_PREBUILT_SO}"
)

add_library(loci_android_jni SHARED loci_jni.cpp)

target_include_directories(
    loci_android_jni
    PRIVATE
    "${LOCI_REPO_INCLUDE_DIR}"
)

target_link_libraries(
    loci_android_jni
    PRIVATE
    loci
    log
)
