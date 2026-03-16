# ExecuTorch - keep all classes including native methods
-keep class org.pytorch.executorch.** { *; }
-keepclassmembers class org.pytorch.executorch.** { *; }

# fbjni - required for ExecuTorch native bridge
-keep class com.facebook.jni.** { *; }
-keepclassmembers class com.facebook.jni.** { *; }
-keep class com.facebook.jni.HybridData { *; }

# SoLoader - required for loading native libraries
-keep class com.facebook.soloader.** { *; }
-keepclassmembers class com.facebook.soloader.** { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep all classes that have @DoNotStrip annotation
-keep @com.facebook.jni.annotations.DoNotStrip class *
-keepclassmembers class * {
    @com.facebook.jni.annotations.DoNotStrip *;
}
-keep @com.facebook.proguard.annotations.DoNotStrip class *
-keepclassmembers class * {
    @com.facebook.proguard.annotations.DoNotStrip *;
}
