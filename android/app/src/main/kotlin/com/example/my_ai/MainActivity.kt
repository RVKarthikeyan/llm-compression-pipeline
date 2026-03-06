package com.example.my_ai

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.json.JSONObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.concurrent.Executors

/**
 * ExecuTorch inference via LlmModule high-level API.
 *
 * LlmModule handles tokenization, KV-cache, and sampling internally.
 * We only need to:
 *   1. Pass the model + tokenizer file paths
 *   2. Format the prompt with the correct chat template
 *   3. Call generate() and collect streamed tokens
 *
 * Supports Llama 3 and Gemma chat templates (auto-detected from
 * tokenizer_config.json).
 */
class MainActivity : FlutterActivity() {
    companion object {
        private const val TAG = "ExecuTorch"
        private const val CHANNEL = "com.example.my_ai/executorch"
        private const val MAX_NEW_TOKENS = 512
        private const val SEQ_LEN = 2048
        private const val TEMPERATURE = 0.0f

        /** Special tokens to strip from final output. */
        private val SPECIAL_TOKENS = listOf(
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>", "<|finetune_right_pad_id|>",
            "<bos>", "<eos>",
            "<start_of_turn>", "<end_of_turn>",
        )
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private var flutterChannel: MethodChannel? = null

    private var llmModule: LlmModule? = null
    private var isLoaded = false
    @Volatile private var stopRequested = false

    // "llama3" or "gemma" — detected from tokenizer_config.json
    private var chatTemplateType = "llama3"

    private fun log(msg: String) {
        Log.i(TAG, msg)
        mainHandler.post {
            try { flutterChannel?.invokeMethod("log", msg) } catch (_: Exception) {}
        }
    }

    private fun logError(msg: String, e: Throwable? = null) {
        Log.e(TAG, msg, e)
        mainHandler.post {
            try { flutterChannel?.invokeMethod("log", "ERROR: $msg") } catch (_: Exception) {}
        }
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
        flutterChannel = channel

        channel.setMethodCallHandler { call, result ->
            when (call.method) {
                "loadModel" -> {
                    val modelPath = call.argument<String>("path")
                    val tokenizerPath = call.argument<String>("vocabPath")
                        ?: call.argument<String>("tokenizerPath")
                    val configPath = call.argument<String>("configPath")
                    if (modelPath == null) {
                        result.error("INVALID_ARG", "Model path is null", null)
                        return@setMethodCallHandler
                    }
                    if (tokenizerPath == null) {
                        result.error("INVALID_ARG", "Tokenizer path is null", null)
                        return@setMethodCallHandler
                    }
                    handleLoadModel(modelPath, tokenizerPath, configPath, result)
                }
                "runInference" -> {
                    val prompt = call.argument<String>("prompt")
                    val context = call.argument<String>("context")
                    if (prompt.isNullOrEmpty()) {
                        result.error("EMPTY_INPUT", "Prompt is empty", null)
                        return@setMethodCallHandler
                    }
                    handleRunInference(prompt, context, result)
                }
                "resetCache" -> {
                    try {
                        llmModule?.resetNative()
                        log("Cache/context reset OK")
                    } catch (e: Exception) {
                        log("Reset failed: ${e.message}")
                    }
                    result.success("reset_ok")
                }
                "stop" -> {
                    stopRequested = true
                    try { llmModule?.stop() } catch (_: Exception) {}
                    result.success("stop_ok")
                }
                else -> result.notImplemented()
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  MODEL LOADING
    // ═══════════════════════════════════════════════════════════════════════

    private var soLoaderInitialized = false

    private fun initSoLoader() {
        if (soLoaderInitialized) return
        try {
            val soLoaderClass = Class.forName("com.facebook.soloader.SoLoader")
            val initMethod = soLoaderClass.getMethod(
                "init", android.content.Context::class.java, Boolean::class.javaPrimitiveType
            )
            initMethod.invoke(null, this@MainActivity, false)
            log("SoLoader initialized OK")
            soLoaderInitialized = true
        } catch (e: Exception) {
            log("SoLoader init: ${e.message} (may not be needed)")
        }
    }

    private fun handleLoadModel(
        modelPath: String,
        tokenizerPath: String,
        configPath: String?,
        result: MethodChannel.Result
    ) {
        executor.execute {
            try {
                log("=== LOADING MODEL ===")
                log("Model: $modelPath")
                log("Tokenizer: $tokenizerPath")
                log("Config: $configPath")

                // Validate tokenizer format before passing to LlmModule
                if (!validateTokenizerFile(tokenizerPath)) {
                    val name = File(tokenizerPath).name
                    logError("Unsupported tokenizer format: $name")
                    mainHandler.post {
                        result.success("demo: Unsupported tokenizer format '$name'. " +
                            "LlmModule requires tokenizer.json, tokenizer.bin, or tokenizer.model. " +
                            "A custom vocab.json is not compatible.")
                    }
                    return@execute
                }

                // Clean up previous
                try { llmModule?.resetNative() } catch (_: Exception) {}
                llmModule = null
                isLoaded = false
                System.gc()

                // Detect chat template
                chatTemplateType = detectChatTemplate(configPath)
                log("Chat template: $chatTemplateType")

                // Log memory
                val runtime = Runtime.getRuntime()
                val freeMemMB = (runtime.freeMemory() + runtime.maxMemory() - runtime.totalMemory()) / (1024 * 1024)
                log("Java heap free: ${freeMemMB}MB")
                val am = getSystemService(android.content.Context.ACTIVITY_SERVICE) as android.app.ActivityManager
                val memInfo = android.app.ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                log("System: avail=${memInfo.availMem / (1024*1024)}MB, lowMemory=${memInfo.lowMemory}")

                // Initialize SoLoader
                initSoLoader()

                // Create LlmModule
                log("Creating LlmModule...")
                llmModule = LlmModule(
                    LlmModule.MODEL_TYPE_TEXT,
                    modelPath,
                    tokenizerPath,
                    TEMPERATURE
                )

                // Load model weights
                log("Loading model weights...")
                val t0 = System.currentTimeMillis()
                val loadResult = llmModule!!.load()
                val elapsed = System.currentTimeMillis() - t0
                log("Model loaded in ${elapsed}ms (result=$loadResult)")

                isLoaded = true
                log("=== MODEL READY ===")
                mainHandler.post { result.success("native_loaded") }

            } catch (e: Exception) {
                val root = findRootCause(e)
                logError("Load failed: ${root.javaClass.simpleName}: ${root.message}", e)
                isLoaded = false
                mainHandler.post {
                    result.success("demo: load failed - ${root.message}")
                }
            }
        }
    }

    /**
     * Detect chat template from tokenizer_config.json.
     * Returns "llama3" or "gemma".
     */
    private fun detectChatTemplate(configPath: String?): String {
        if (configPath == null) return "llama3"
        try {
            val json = JSONObject(File(configPath).readText(Charsets.UTF_8))
            val template = json.optString("chat_template", "")
            return when {
                template.contains("start_header_id") -> "llama3"
                template.contains("start_of_turn") -> "gemma"
                else -> "llama3"
            }
        } catch (e: Exception) {
            log("Could not read config: ${e.message}")
            return "llama3"
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  INFERENCE
    // ═══════════════════════════════════════════════════════════════════════

    private fun handleRunInference(
        prompt: String,
        context: String?,
        result: MethodChannel.Result
    ) {
        if (!isLoaded || llmModule == null) {
            result.error("NOT_LOADED", "Model not loaded", null)
            return
        }

        executor.execute {
            try {
                stopRequested = false
                log("=== INFERENCE START === prompt: \"${prompt.take(80)}\"")

                val formattedPrompt = formatChatPrompt(prompt, context)
                log("Formatted prompt: ${formattedPrompt.length} chars, template=$chatTemplateType")

                val genConfig = LlmGenerationConfig.create()
                    .maxNewTokens(MAX_NEW_TOKENS)
                    .seqLen(SEQ_LEN)
                    .temperature(TEMPERATURE)
                    .echo(false)
                    .build()

                val outputBuilder = StringBuilder()
                val t0 = System.currentTimeMillis()
                var tokenCount = 0

                val callback = object : LlmCallback {
                    override fun onResult(token: String) {
                        if (!stopRequested) {
                            outputBuilder.append(token)
                            tokenCount++
                            if (tokenCount <= 5 || tokenCount % 20 == 0) {
                                log("Token $tokenCount: \"${token.replace("\n", "\\n")}\"")
                            }
                        }
                    }

                    override fun onStats(stats: String?) {
                        log("Stats: $stats")
                    }
                }

                log("Calling generate()...")
                llmModule!!.generate(formattedPrompt, genConfig, callback)

                val elapsed = System.currentTimeMillis() - t0
                var output = outputBuilder.toString().trim()

                // Strip any special tokens that leaked through
                output = stripSpecialTokens(output)



                log("Generated $tokenCount tokens, ${output.length} chars in ${elapsed}ms")
                if (elapsed > 0 && tokenCount > 0) {
                    log("Speed: ${String.format("%.2f", tokenCount * 1000.0 / elapsed)} tok/s")
                }
                log("Output: \"${output.take(300)}\"")

                mainHandler.post { result.success(output) }

            } catch (e: Exception) {
                val root = findRootCause(e)
                logError("Inference failed: ${root.javaClass.simpleName}: ${root.message}", e)
                mainHandler.post {
                    result.error("INFERENCE_ERROR", root.message, null)
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  CHAT TEMPLATE FORMATTING
    // ═══════════════════════════════════════════════════════════════════════

    private fun formatChatPrompt(userText: String, context: String?): String {
        return when (chatTemplateType) {
            "llama3" -> formatLlama3(userText, context)
            "gemma" -> formatGemma(userText, context)
            else -> formatLlama3(userText, context)
        }
    }

    private fun formatLlama3(userText: String, context: String?): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")

        // System message
        sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
        if (!context.isNullOrBlank()) {
            sb.append("You are a helpful document assistant. ")
            sb.append("A document is provided below. Answer the user's question using the document first. ")
            sb.append("Clearly distinguish what comes from the document vs your own knowledge. ")
            sb.append("If the document contains the answer, quote or reference the relevant parts. ")
            sb.append("If the document does not fully answer the question, say so, then add anything you know that might help.")
        } else {
            sb.append("You are a helpful assistant. Provide a detailed and complete answer.")
        }
        sb.append("<|eot_id|>")

        // User message
        sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
        if (!context.isNullOrBlank()) {
            sb.append("DOCUMENT:\n")
            sb.append(context)
            sb.append("\n\nQUESTION: ")
            sb.append(userText)
        } else {
            sb.append(userText)
        }
        sb.append("<|eot_id|>")

        // Assistant turn
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return sb.toString()
    }

    private fun formatGemma(userText: String, context: String?): String {
        val sb = StringBuilder()
        sb.append("<bos><start_of_turn>user\n")
        if (!context.isNullOrBlank()) {
            sb.append("You are a helpful document assistant. ")
            sb.append("Answer using the document first, then add your own knowledge if helpful. ")
            sb.append("Clearly distinguish what comes from the document vs your own knowledge.\n\n")
            sb.append("DOCUMENT:\n")
            sb.append(context)
            sb.append("\n\nQUESTION: ")
            sb.append(userText)
        } else {
            sb.append(userText)
        }
        sb.append("<end_of_turn>\n<start_of_turn>model\n")
        return sb.toString()
    }

    /**
     * Remove all known special tokens from model output.
     */
    private fun stripSpecialTokens(text: String): String {
        var result = text
        for (token in SPECIAL_TOKENS) {
            result = result.replace(token, "")
        }
        // Also strip "user", "assistant", "model" that appear as role markers
        // right after header tokens, but only if at start of output
        result = result
            .trimStart()
            .removePrefix("user")
            .removePrefix("assistant")
            .removePrefix("model")
            .trim()
        return result
    }

    /**
     * Validate tokenizer file format. LlmModule supports:
     * - tokenizer.json (HuggingFace format)
     * - tokenizer.bin (SentencePiece binary)
     * - tokenizer.model (SentencePiece model)
     * It does NOT support custom vocab.json format.
     */
    private fun validateTokenizerFile(path: String): Boolean {
        val name = File(path).name.lowercase()
        // Known good formats
        if (name == "tokenizer.json" || name == "tokenizer.bin" || name == "tokenizer.model") {
            return true
        }
        // Check if the file content looks like a HuggingFace tokenizer
        try {
            val head = File(path).bufferedReader().use { it.readLine() ?: "" }
            if (head.contains("\"model\"") || head.contains("\"version\"") || head.contains("\"vocab\"")) {
                // Looks like a HuggingFace tokenizer.json content — allow it
                return true
            }
        } catch (_: Exception) {}
        return false
    }

    private fun findRootCause(e: Throwable): Throwable {
        var cause: Throwable = e
        while (cause.cause != null && cause.cause !== cause) {
            cause = cause.cause!!
        }
        return cause
    }
}