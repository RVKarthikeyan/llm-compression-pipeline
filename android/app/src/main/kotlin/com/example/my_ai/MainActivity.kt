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
        private const val MAX_NEW_TOKENS = 300
        private const val SEQ_LEN = 2048
        private const val TEMPERATURE = 0.1f

        /** EOS markers — if any of these appear as a token, stop generation. */
        private val EOS_TOKENS = listOf(
            "<|eot_id|>", "<|end_of_text|>", "<eos>", "<end_of_turn>",
        )

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

    // Store paths for module re-creation between inferences
    private var storedModelPath: String? = null
    private var storedTokenizerPath: String? = null

    // "llama3" or "gemma" — detected from tokenizer_config.json
    private var chatTemplateType = "llama3"

    // On-device embedding model for query vectorization
    private var embeddingService: OnnxEmbeddingService? = null

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
                "initEmbeddingModel" -> {
                    handleInitEmbeddingModel(result)
                }
                "embedQuery" -> {
                    val text = call.argument<String>("text")
                    if (text.isNullOrEmpty()) {
                        result.error("EMPTY_INPUT", "Text is empty", null)
                        return@setMethodCallHandler
                    }
                    handleEmbedQuery(text, result)
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

                // Store paths for re-creation between inferences
                storedModelPath = modelPath
                storedTokenizerPath = tokenizerPath

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

                // Recreate LlmModule before each inference to fully reset
                // the KV-cache. resetNative() destroys the module, so we
                // must create a fresh instance. The OS page cache keeps the
                // model file in memory, so reload is fast (~1-3s).
                val mPath = storedModelPath
                val tPath = storedTokenizerPath
                if (mPath != null && tPath != null) {
                    try {
                        llmModule?.stop()
                    } catch (_: Exception) {}
                    try {
                        llmModule?.resetNative()
                    } catch (_: Exception) {}
                    llmModule = null

                    val t0r = System.currentTimeMillis()
                    llmModule = LlmModule(
                        LlmModule.MODEL_TYPE_TEXT,
                        mPath,
                        tPath,
                        TEMPERATURE
                    )
                    llmModule!!.load()
                    val reloadMs = System.currentTimeMillis() - t0r
                    log("Module re-created in ${reloadMs}ms (KV-cache fresh)")
                }

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
                var lastToken = ""
                var repeatCount = 0
                val MAX_REPEATS = 4  // stop if same token repeats this many times

                val callback = object : LlmCallback {
                    override fun onResult(token: String) {
                        if (!stopRequested) {
                            // Stop if model emits an EOS-like token
                            if (EOS_TOKENS.any { token.contains(it) }) {
                                log("EOS token detected at token $tokenCount: \"$token\"")
                                stopRequested = true
                                try { llmModule?.stop() } catch (_: Exception) {}
                                return
                            }

                            // Detect degeneration loops (e.g. "from from from from...")
                            val trimmed = token.trim()
                            if (trimmed.isNotEmpty() && trimmed == lastToken.trim()) {
                                repeatCount++
                                if (repeatCount >= MAX_REPEATS) {
                                    log("Repetition loop detected at token $tokenCount: \"$trimmed\" x$repeatCount — stopping")
                                    stopRequested = true
                                    try { llmModule?.stop() } catch (_: Exception) {}
                                    return
                                }
                            } else {
                                repeatCount = 0
                            }
                            lastToken = token

                            outputBuilder.append(token)
                            tokenCount++

                            // Stream token to Dart UI in real-time
                            mainHandler.post {
                                try { flutterChannel?.invokeMethod("onToken", token) } catch (_: Exception) {}
                            }

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

                // Truncate at first EOS token (in case stop didn't fire fast enough)
                for (eos in EOS_TOKENS) {
                    val idx = output.indexOf(eos)
                    if (idx >= 0) output = output.substring(0, idx)
                }

                // Strip any remaining special tokens
                output = stripSpecialTokens(output)

                // Clean up any trailing repetition that slipped through
                output = stripTrailingRepetition(output)

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

    /** System prompt for RAG document Q&A — shared between all chat templates. */
    private fun ragSystemPrompt(): String = """
You are a document assistant. Read the DOCUMENT context and answer the question in your own words. Don't repeat this prompt back to the user.
1. Summarize relevant facts from the document. Do NOT copy sentences from it.
2. Keep key details like names, numbers, dates, and medications or other important info.
3. If the answer is not in the document, say "Not found in the document.""".trimIndent()

    private fun formatLlama3(userText: String, context: String?): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")

        // System message
        sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
        if (!context.isNullOrBlank()) {
            sb.append(ragSystemPrompt())
        } else {
            sb.append("You are a helpful assistant. Be concise and direct.")
        }
        sb.append("<|eot_id|>")

        // User message
        sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
        if (!context.isNullOrBlank()) {
            sb.append("DOCUMENT:\n")
            sb.append(context)
            sb.append("\n\nUsing ONLY the document above, answer this question: ")
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
            sb.append(ragSystemPrompt())
            sb.append("\n\nDOCUMENT:\n")
            sb.append(context)
            sb.append("\n\nUsing ONLY the document above, answer this question: ")
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
        // Strip role markers only when they appear as isolated labels
        // (not as part of regular words like "User satisfaction" or "Model A")
        result = result.trimStart()
        result = result
            .replaceFirst(Regex("^(user|assistant|model)\\s*[\n:]\\s*", RegexOption.IGNORE_CASE), "")
            .trim()
        return result
    }

    /**
     * Detect and remove trailing word/phrase repetition (degeneration loops).
     * E.g. "The answer is from from from from" → "The answer is from"
     * Also handles multi-word patterns like "the contract the contract the contract".
     */
    private fun stripTrailingRepetition(text: String): String {
        if (text.length < 20) return text

        // Check for repeated patterns of 1-5 words at the end
        val words = text.split(Regex("\\s+"))
        if (words.size < 4) return text

        for (patLen in 1..5) {
            if (words.size < patLen * 3) continue
            val pattern = words.takeLast(patLen).joinToString(" ")
            var count = 0
            var i = words.size
            while (i >= patLen) {
                val segment = words.subList(i - patLen, i).joinToString(" ")
                if (segment == pattern) {
                    count++
                    i -= patLen
                } else {
                    break
                }
            }
            if (count >= 3) {
                // Keep one instance of the pattern, remove the rest
                val keepUpTo = words.size - (count - 1) * patLen
                return words.take(keepUpTo).joinToString(" ").trim()
            }
        }
        return text
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

    // ═══════════════════════════════════════════════════════════════════════
    //  EMBEDDING MODEL (ONNX)
    // ═══════════════════════════════════════════════════════════════════════

    private fun handleInitEmbeddingModel(result: MethodChannel.Result) {
        executor.execute {
            try {
                if (embeddingService?.ready() == true) {
                    mainHandler.post { result.success("already_loaded") }
                    return@execute
                }
                log("Loading ONNX embedding model...")
                embeddingService = OnnxEmbeddingService(this@MainActivity)
                embeddingService!!.initialize()
                if (embeddingService!!.ready()) {
                    log("Embedding model ready")
                    mainHandler.post { result.success("loaded") }
                } else {
                    mainHandler.post {
                        result.error("LOAD_FAILED", "Embedding model failed to initialize", null)
                    }
                }
            } catch (e: Exception) {
                logError("Embedding model init failed: ${e.message}", e)
                mainHandler.post {
                    result.error("LOAD_FAILED", e.message, null)
                }
            }
        }
    }

    private fun handleEmbedQuery(text: String, result: MethodChannel.Result) {
        executor.execute {
            try {
                // Auto-initialize if not yet loaded
                if (embeddingService?.ready() != true) {
                    log("Auto-initializing embedding model for query...")
                    embeddingService = OnnxEmbeddingService(this@MainActivity)
                    embeddingService!!.initialize()
                }

                val embedding = embeddingService?.embed(text)
                if (embedding != null) {
                    // Return as List<Double> for Flutter
                    val doubleList = embedding.map { it.toDouble() }
                    mainHandler.post { result.success(doubleList) }
                } else {
                    mainHandler.post {
                        result.error("EMBED_FAILED", "Failed to generate embedding", null)
                    }
                }
            } catch (e: Exception) {
                logError("Embed query failed: ${e.message}", e)
                mainHandler.post {
                    result.error("EMBED_FAILED", e.message, null)
                }
            }
        }
    }

    private fun findRootCause(e: Throwable): Throwable {
        var cause: Throwable = e
        while (cause.cause != null && cause.cause !== cause) {
            cause = cause.cause!!
        }
        return cause
    }
}