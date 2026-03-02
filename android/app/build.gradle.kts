plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.my_ai"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion
    compileSdk = 36

    dependencies {
    // 1. The Official ExecuTorch Android Library (as per docs)
    implementation("org.pytorch:executorch-android:1.0.0")
    
    // 2. Mandatory helper libraries for the AAR to function
    implementation("com.facebook.soloader:soloader:0.10.5")
    implementation("com.facebook.fbjni:fbjni:0.7.0")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
    jvmTarget = "17"
}

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.my_ai"
        // ExecuTorch requires API 23+. Do not lower this value.
        minSdk = 23
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // ExecuTorch ships arm64-v8a native libs only.
        // Limiting to arm64-v8a prevents install on x86/x86_64 emulators
        // (which would crash with UnsatisfiedLinkError at startup).
        // Use a real device or an ARM64 emulator (e.g. on Apple Silicon).
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}
