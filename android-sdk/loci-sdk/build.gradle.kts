import org.gradle.api.GradleException
import org.gradle.api.publish.maven.MavenPublication

plugins {
    id("com.android.library")
    kotlin("android")
    `maven-publish`
}

group = "io.github.decade-afk"
version = providers.gradleProperty("lociSdkVersion")
    .orElse(providers.environmentVariable("LOCI_ANDROID_SDK_VERSION"))
    .orElse("0.1.0-SNAPSHOT")
    .get()

val supportedLociAbis = listOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
val availableLociAbis = supportedLociAbis.filter { abi ->
    layout.projectDirectory.file("src/main/jniLibs/$abi/libloci.so").asFile.exists()
}
val configuredLociAbis = providers.gradleProperty("lociAbiFilters")
    .orNull
    ?.split(',')
    ?.map { it.trim() }
    ?.filter { it.isNotEmpty() }
    ?.distinct()
    ?: availableLociAbis

val unsupportedConfiguredAbis = configuredLociAbis.filterNot(supportedLociAbis::contains)
if (unsupportedConfiguredAbis.isNotEmpty()) {
    throw GradleException(
        "Unsupported Android ABI values in lociAbiFilters: ${unsupportedConfiguredAbis.joinToString(", ")}. " +
            "Supported values: ${supportedLociAbis.joinToString(", ")}"
    )
}

android {
    namespace = "io.github.decadeafk.loci.sdk"
    compileSdk = 35
    ndkVersion = "27.3.13750724"

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            if (configuredLociAbis.isNotEmpty()) {
                abiFilters += configuredLociAbis
            }
        }
        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17"
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        buildConfig = false
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

tasks.register("verifyLociPrebuilt") {
    doLast {
        if (configuredLociAbis.isEmpty()) {
            throw GradleException(
                "No prebuilt Android native libraries were found under src/main/jniLibs. " +
                    "Run android-sdk/scripts/sync-prebuilt-loci.ps1 or .sh after building libloci.so from the repository root."
            )
        }

        val missingLibs = configuredLociAbis.filter { abi ->
            !layout.projectDirectory.file("src/main/jniLibs/$abi/libloci.so").asFile.exists()
        }
        if (missingLibs.isNotEmpty()) {
            throw GradleException(
                "Missing prebuilt Android native libraries for ABI(s): ${missingLibs.joinToString(", ")}. " +
                    "Expected files under src/main/jniLibs/<abi>/libloci.so."
            )
        }
    }
}

tasks.named("preBuild").configure {
    dependsOn("verifyLociPrebuilt")
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                groupId = project.group.toString()
                artifactId = "loci-sdk"
                version = project.version.toString()
                from(components["release"])
            }
        }
    }
}

dependencies {
    implementation("androidx.annotation:annotation:1.9.1")
}
