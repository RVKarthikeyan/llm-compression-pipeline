package com.example.my_ai

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.concurrent.Executors

/**
 * ExecuTorch inference via MethodChannel.
 *
 * The .pte model produced by the Colab notebook takes:
 *   input_ids     : Tensor<long> [1, seq_len]
 *   attention_mask : Tensor<long> [1, seq_len]
 * and returns:
 *   logits        : Tensor<float32> [1, seq_len, vocab_size]
 *
 * We tokenize text → long[], run forward(), sample from logits, and decode back.
 * Everything runs on a background thread to avoid blocking the UI.
 */
class MainActivity : FlutterActivity() {
    companion object {
        private const val TAG = "ExecuTorch"
        private const val CHANNEL = "com.example.my_ai/executorch"
        private const val MAX_NEW_TOKENS = 150
        private const val TEMPERATURE = 0.7f
        private const val TOP_K = 40
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    // Native ExecuTorch module (org.pytorch.executorch.Module)
    private var module: Any? = null
    private var tokenizer: SimpleTokenizer? = null
    private var isNativeLoaded = false

    // Reflection caches (resolved once after model load)
    private var tensorClass: Class<*>? = null
    private var evalueClass: Class<*>? = null
    private var fromTensorMethod: java.lang.reflect.Method? = null
    private var forwardMethod: java.lang.reflect.Method? = null
    private var toTensorMethod: java.lang.reflect.Method? = null
    private var getDataAsFloatArrayMethod: java.lang.reflect.Method? = null
    private var tensorFromBlobLongMethod: java.lang.reflect.Method? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "loadModel" -> {
                        val path = call.argument<String>("path")
                        val vocabPath = call.argument<String>("vocabPath")
                        val configPath = call.argument<String>("configPath")
                        if (path == null) {
                            result.error("INVALID_ARG", "Model path is null", null)
                            return@setMethodCallHandler
                        }
                        handleLoadModel(path, vocabPath, configPath, result)
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
                    else -> result.notImplemented()
                }
            }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NATIVE LIBRARY INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════

    private var nativeLibInitialized = false

    /**
     * Initialize native libraries (SoLoader + ExecuTorch JNI).
     * Must be called BEFORE any ExecuTorch class is touched.
     */
    private fun ensureNativeLibsLoaded() {
        if (nativeLibInitialized) return

        // Try SoLoader first (handles transitive .so dependencies)
        try {
            val soLoaderClass = Class.forName("com.facebook.soloader.SoLoader")
            val initMethod = soLoaderClass.getMethod(
                "init", android.content.Context::class.java, Boolean::class.javaPrimitiveType
            )
            initMethod.invoke(null, this@MainActivity, false)
            Log.i(TAG, "SoLoader initialized successfully")
            nativeLibInitialized = true
        } catch (e: Exception) {
            Log.w(TAG, "SoLoader init failed: ${e.message}, trying System.loadLibrary")
        }

        // Fallback: try direct System.loadLibrary
        if (!nativeLibInitialized) {
            try {
                System.loadLibrary("executorch_jni")
                Log.i(TAG, "System.loadLibrary(executorch_jni) succeeded")
                nativeLibInitialized = true
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "System.loadLibrary failed: ${e.message}")
                // Try alternate library names
                try {
                    System.loadLibrary("executorch")
                    Log.i(TAG, "System.loadLibrary(executorch) succeeded")
                    nativeLibInitialized = true
                } catch (e2: UnsatisfiedLinkError) {
                    Log.e(TAG, "All native library loading failed: ${e2.message}")
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MODEL LOADING
    // ═══════════════════════════════════════════════════════════════════════

    private fun handleLoadModel(
        modelPath: String,
        vocabPath: String?,
        configPath: String?,
        result: MethodChannel.Result
    ) {
        executor.execute {
            try {
                Log.i(TAG, "Loading model: $modelPath")
                Log.i(TAG, "Vocab path: $vocabPath")
                Log.i(TAG, "Config path: $configPath")

                // 0. Initialize native libraries FIRST
                ensureNativeLibsLoaded()

                // 1. Load tokenizer (if vocab files provided)
                if (vocabPath != null && configPath != null) {
                    try {
                        tokenizer = SimpleTokenizer(vocabPath, configPath)
                        Log.i(TAG, "Tokenizer loaded: vocab=${tokenizer!!.vocabSize}, " +
                                "bos=${tokenizer!!.bosTokenId}, eos=${tokenizer!!.eosTokenId}, " +
                                "startTurn=${tokenizer!!.startOfTurnId}, endTurn=${tokenizer!!.endOfTurnId}")
                    } catch (e: Exception) {
                        Log.w(TAG, "Tokenizer load failed: ${e.message}", e)
                        tokenizer = null
                    }
                }

                // 2. Load ExecuTorch Module via reflection (for graceful fallback)
                val moduleClass = Class.forName("org.pytorch.executorch.Module")
                Log.i(TAG, "Module class found: ${moduleClass.name}")

                // Try load(String) first, then load(String, int)
                val loadMethod = try {
                    moduleClass.getMethod("load", String::class.java)
                } catch (e: NoSuchMethodException) {
                    Log.w(TAG, "Module.load(String) not found, trying load(String, int)")
                    moduleClass.getMethod("load", String::class.java, Int::class.javaPrimitiveType)
                }
                Log.i(TAG, "Load method found: ${loadMethod.name}(${loadMethod.parameterTypes.joinToString { it.name }})")

                module = if (loadMethod.parameterCount == 1) {
                    loadMethod.invoke(null, modelPath)
                } else {
                    loadMethod.invoke(null, modelPath, 0) // 0 = MMAP mode
                }
                Log.i(TAG, "Module loaded successfully: ${module!!.javaClass.name}")

                // 3. Resolve reflection handles for Tensor/EValue
                resolveReflection()

                isNativeLoaded = true

                val status = if (tokenizer != null) "native_loaded" else "native_loaded_no_tokenizer"
                mainHandler.post { result.success(status) }

            } catch (e: ClassNotFoundException) {
                Log.e(TAG, "ExecuTorch classes not found: ${e.message}", e)
                isNativeLoaded = false
                mainHandler.post { result.success("demo: ExecuTorch classes not found - ${e.message}") }
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Native lib missing: ${e.message}", e)
                isNativeLoaded = false
                mainHandler.post { result.success("demo: native lib not loaded - ${e.message}") }
            } catch (e: Exception) {
                val rootCause = findRootCause(e)
                val msg = "${rootCause.javaClass.simpleName}: ${rootCause.message}"
                Log.e(TAG, "Load error: $msg", e)
                // Still try to enter demo mode instead of hard failing
                isNativeLoaded = false
                mainHandler.post { result.success("demo: $msg") }
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

    /**
     * Resolve all reflection Method handles once, so forward() calls are fast.
     */
    private fun resolveReflection() {
        // org.pytorch.executorch.Tensor
        tensorClass = Class.forName("org.pytorch.executorch.Tensor")
        Log.i(TAG, "Tensor class found. Methods: ${tensorClass!!.methods.map { it.name }.distinct().sorted()}")

        // Tensor.fromBlob(long[] data, long[] shape) — NOTE: method is "fromBlob", NOT "fromBlobLong"
        tensorFromBlobLongMethod = try {
            tensorClass!!.getMethod("fromBlob", LongArray::class.java, LongArray::class.java)
        } catch (e: NoSuchMethodException) {
            Log.w(TAG, "Tensor.fromBlob(long[], long[]) not found, trying fromBlobLong")
            tensorClass!!.getMethod("fromBlobLong", LongArray::class.java, LongArray::class.java)
        }
        Log.i(TAG, "Tensor create method: ${tensorFromBlobLongMethod!!.name}")

        // tensor.getDataAsFloatArray()
        getDataAsFloatArrayMethod = try {
            tensorClass!!.getMethod("getDataAsFloatArray")
        } catch (e: NoSuchMethodException) {
            Log.w(TAG, "getDataAsFloatArray not found, trying dataAsFloatArray")
            tensorClass!!.getMethod("dataAsFloatArray")
        }
        Log.i(TAG, "Float data method: ${getDataAsFloatArrayMethod!!.name}")

        // org.pytorch.executorch.EValue
        evalueClass = Class.forName("org.pytorch.executorch.EValue")
        Log.i(TAG, "EValue class found. Methods: ${evalueClass!!.methods.map { it.name }.distinct().sorted()}")

        // EValue.from(Tensor t)
        fromTensorMethod = evalueClass!!.getMethod("from", tensorClass)
        // evalue.toTensor()
        toTensorMethod = evalueClass!!.getMethod("toTensor")

        // Module.forward(EValue[])
        val moduleClass = module!!.javaClass
        val evalueArrayType = java.lang.reflect.Array.newInstance(evalueClass!!, 0).javaClass

        forwardMethod = try {
            moduleClass.getMethod("forward", evalueArrayType)
        } catch (e: NoSuchMethodException) {
            // Some versions use execute("forward", EValue[])
            Log.w(TAG, "Module.forward(EValue[]) not found, trying execute(String, EValue[])")
            try {
                moduleClass.getMethod("execute", String::class.java, evalueArrayType)
            } catch (e2: NoSuchMethodException) {
                Log.e(TAG, "Available methods on Module: ${moduleClass.methods.map { "${it.name}(${it.parameterTypes.joinToString { t -> t.simpleName }})" }}")
                throw e2
            }
        }
        Log.i(TAG, "Forward method: ${forwardMethod!!.name}(${forwardMethod!!.parameterTypes.joinToString { it.simpleName }})")

        Log.i(TAG, "Reflection resolved successfully")
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INFERENCE — autoregressive generation with tensor I/O
    // ═══════════════════════════════════════════════════════════════════════

    private fun handleRunInference(
        prompt: String,
        context: String?,
        result: MethodChannel.Result
    ) {
        if (!isNativeLoaded) {
            result.error("DEMO_MODE", "Native ExecuTorch not loaded", null)
            return
        }
        if (tokenizer == null) {
            result.error("NO_TOKENIZER",
                "Tokenizer not loaded. Please load vocab.json alongside the model.", null)
            return
        }

        executor.execute {
            try {
                val response = autoregressiveGenerate(prompt, context, MAX_NEW_TOKENS)
                mainHandler.post { result.success(response) }
            } catch (e: Exception) {
                val rootCause = findRootCause(e)
                val msg = "${rootCause.javaClass.simpleName}: ${rootCause.message}"
                Log.e(TAG, "Inference error: $msg", e)
                mainHandler.post { result.error("INFERENCE_ERROR", msg, null) }
            }
        }
    }

    /**
     * Full autoregressive decoding loop using Gemma chat template:
     *   <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
     * Then generate tokens until EOS or max length.
     */
    private fun autoregressiveGenerate(
        prompt: String,
        context: String?,
        maxTokens: Int
    ): String {
        val tok = tokenizer!!

        // Format with Gemma chat template
        val inputIds = tok.formatGemmaChat(prompt, context).toMutableList()
        val generatedIds = mutableListOf<Int>()

        Log.i(TAG, "Generating: prompt_tokens=${inputIds.size}, max_new=$maxTokens")
        val t0 = System.currentTimeMillis()

        for (step in 0 until maxTokens) {
            val seqLen = inputIds.size
            val idsArray = inputIds.toLongArray()
            val maskArray = LongArray(seqLen) { 1L }

            // Create Tensor objects via reflection
            val shape = longArrayOf(1L, seqLen.toLong())
            val idsTensor = tensorFromBlobLongMethod!!.invoke(null, idsArray, shape)
            val maskTensor = tensorFromBlobLongMethod!!.invoke(null, maskArray, shape)

            // Wrap in EValue
            val idsEValue = fromTensorMethod!!.invoke(null, idsTensor)
            val maskEValue = fromTensorMethod!!.invoke(null, maskTensor)

            // Build EValue[] array
            val evalueArray = java.lang.reflect.Array.newInstance(evalueClass!!, 2)
            java.lang.reflect.Array.set(evalueArray, 0, idsEValue)
            java.lang.reflect.Array.set(evalueArray, 1, maskEValue)

            // module.forward(EValue[]) → EValue[]
            // Cast to Object to prevent Java varargs spreading the array
            val outputs = if (forwardMethod!!.name == "execute") {
                forwardMethod!!.invoke(module, "forward", evalueArray)
            } else {
                forwardMethod!!.invoke(module, evalueArray as Any)
            }

            // Get first output → Tensor → float[]
            val firstEValue = java.lang.reflect.Array.get(outputs, 0)
            val logitsTensor = toTensorMethod!!.invoke(firstEValue)
            val logitsFlat = getDataAsFloatArrayMethod!!.invoke(logitsTensor) as FloatArray

            // logitsFlat is [1, seq_len, vocab_size] flattened
            // We want the last position's logits: offset = (seq_len - 1) * vocab_size
            val vocabSize = tok.vocabSize
            val lastOffset = (seqLen - 1) * vocabSize
            val lastLogits = FloatArray(vocabSize)
            System.arraycopy(logitsFlat, lastOffset, lastLogits, 0, vocabSize)

            // Sample next token
            val nextTokenId = sampleToken(lastLogits)

            // Check for EOS or end_of_turn
            if (nextTokenId == tok.eosTokenId || nextTokenId == tok.endOfTurnId) {
                Log.i(TAG, "Stop token at step $step (id=$nextTokenId)")
                break
            }

            generatedIds.add(nextTokenId)
            inputIds.add(nextTokenId.toLong())

            // Log progress every 10 tokens
            if (step % 10 == 0 && step > 0) {
                val elapsed = System.currentTimeMillis() - t0
                val tokPerSec = if (elapsed > 0) step * 1000.0 / elapsed else 0.0
                Log.d(TAG, "Step $step: ${String.format("%.1f", tokPerSec)} tok/s")
            }
        }

        val elapsed = System.currentTimeMillis() - t0
        val tokPerSec = if (elapsed > 0) generatedIds.size * 1000.0 / elapsed else 0.0
        Log.i(TAG, "Generated ${generatedIds.size} tokens in ${elapsed}ms (${String.format("%.1f", tokPerSec)} tok/s)")

        return tok.decode(generatedIds)
    }

    /**
     * Temperature-scaled top-k sampling.
     * temperature=0 → greedy argmax.
     */
    private fun sampleToken(logits: FloatArray): Int {
        if (TEMPERATURE <= 0f) {
            // Greedy
            return logits.indices.maxByOrNull { logits[it] } ?: 0
        }

        // Apply temperature
        val scaled = FloatArray(logits.size) { logits[it] / TEMPERATURE }

        // Top-k filtering
        val k = minOf(TOP_K, scaled.size)
        data class IdxVal(val idx: Int, val value: Float)

        val topK = scaled.mapIndexed { i, v -> IdxVal(i, v) }
            .sortedByDescending { it.value }
            .take(k)

        // Softmax over top-k
        val maxVal = topK.first().value
        val exps = topK.map { Math.exp((it.value - maxVal).toDouble()).toFloat() }
        val sumExps = exps.sum()
        val probs = exps.map { it / sumExps }

        // Sample
        val r = Math.random().toFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (r < cumulative) return topK[i].idx
        }
        return topK.last().idx
    }
}