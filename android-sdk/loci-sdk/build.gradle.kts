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

android {
    namespace = "io.github.decadeafk.loci.sdk"
    compileSdk = 35
    ndkVersion = "27.3.13750724"

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
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

val requiredArm64Lib = layout.projectDirectory.file("src/main/jniLibs/arm64-v8a/libloci.so")

tasks.register("verifyLociPrebuilt") {
    doLast {
        if (!requiredArm64Lib.asFile.exists()) {
            throw GradleException(
                "Missing prebuilt Android native library: ${requiredArm64Lib.asFile}. " +
                    "Run android-sdk/scripts/sync-prebuilt-loci.ps1 or .sh after building libloci.so from the repository root."
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
