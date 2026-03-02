package com.example.my_ai

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.my_ai/executorch"

    // Lazy references: avoids UnsatisfiedLinkError at class-load time
    // when the native SO is absent (e.g. wrong ABI on emulator).
    private var module: Any? = null  // Will hold org.pytorch.executorch.Module

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "loadModel" -> {
                    val path = call.argument<String>("path")
                    if (path == null) {
                        result.error("INVALID_ARGUMENT", "Model path is null", null)
                        return@setMethodCallHandler
                    }
                    try {
                        // Reflective load to avoid crashing if native lib missing.
                        val moduleClass = Class.forName("org.pytorch.executorch.Module")
                        val loadMethod = moduleClass.getMethod("load", String::class.java)
                        module = loadMethod.invoke(null, path)
                        result.success("Model Ready")
                    } catch (e: UnsatisfiedLinkError) {
                        result.error("NO_NATIVE_LIB",
                            "ExecuTorch native library not available on this ABI: ${e.message}", null)
                    } catch (e: ClassNotFoundException) {
                        result.error("NO_EXECUTORCH",
                            "ExecuTorch AAR not found: ${e.message}", null)
                    } catch (e: Exception) {
                        result.error("LOAD_ERROR", "Failed to load .pte: ${e.message}", null)
                    }
                }
                "runInference" -> {
                    val prompt = call.argument<String>("prompt")
                    if (prompt.isNullOrEmpty()) {
                        result.error("EMPTY_INPUT", "Prompt cannot be empty", null)
                        return@setMethodCallHandler
                    }
                    val mod = module
                    if (mod == null) {
                        result.error("NOT_LOADED", "Model not loaded yet. Call loadModel first.", null)
                        return@setMethodCallHandler
                    }
                    try {
                        val evalueClass  = Class.forName("org.pytorch.executorch.EValue")
                        val fromMethod   = evalueClass.getMethod("from", String::class.java)
                        val inputEValue  = fromMethod.invoke(null, prompt)
                        val forwardMethod = mod.javaClass.getMethod("forward",
                            Array<Any>::class.java)
                        @Suppress("UNCHECKED_CAST")
                        val output = forwardMethod.invoke(mod,
                            arrayOf(inputEValue)) as? Array<Any>
                        val response = output?.getOrNull(0)?.toString() ?: "No response"
                        result.success(response)
                    } catch (e: UnsatisfiedLinkError) {
                        result.error("NO_NATIVE_LIB",
                            "ExecuTorch native library not available: ${e.message}", null)
                    } catch (e: Exception) {
                        result.error("EXEC_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }
}
}