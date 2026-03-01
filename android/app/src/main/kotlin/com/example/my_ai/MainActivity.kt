package com.example.my_ai

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.my_ai/executorch"
    private var module: Module? = null

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
                        module = Module.load(path)
                        result.success("Model Ready")
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

                    try {
                        // Models usually expect an ARRAY of inputs.
                        // We create an array containing one EValue (your prompt).
                        val inputEValue = EValue.from(prompt)
                        val inputs = arrayOf(inputEValue) 

                        
                        // Pass the array to forward
                        val output = module?.forward(*inputs) 
                        
                        val response = output?.get(0)?.toString() ?: "No response"
                        result.success(response)
                    } catch (e: Exception) {
                        // Detailed logging as recommended by ExecuTorch contributors
                        val logs = module?.readLogBuffer()?.joinToString("\n")
                        result.error("EXEC_ERROR", "${e.message}\nLogs: $logs", null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }
}