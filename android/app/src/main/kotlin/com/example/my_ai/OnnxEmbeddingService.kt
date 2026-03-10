package com.example.my_ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * On-device sentence embedding using all-MiniLM-L6-v2 (ONNX).
 *
 * Loads the ONNX model and tokenizer.json from Android assets at
 * `assets/embedding/`. Produces 384-dim L2-normalized embeddings
 * compatible with the backend's sentence-transformers output.
 *
 * The model expects three inputs:
 *   - input_ids:      (1, seq_len) INT64
 *   - attention_mask:  (1, seq_len) INT64
 *   - token_type_ids:  (1, seq_len) INT64
 *
 * It outputs token_embeddings (1, seq_len, 384). We mean-pool over
 * the attention mask and L2-normalize to get a single 384-dim vector.
 */
class OnnxEmbeddingService(private val context: Context) {
    companion object {
        private const val TAG = "OnnxEmbed"
        private const val MODEL_PATH = "embedding/all-MiniLM-L6-v2.onnx"
        private const val TOKENIZER_PATH = "embedding/tokenizer.json"
        private const val MAX_SEQ_LEN = 128
        private const val EMBEDDING_DIM = 384

        // Special token IDs for all-MiniLM-L6-v2 (BERT-based WordPiece)
        private const val CLS_ID = 101L
        private const val SEP_ID = 102L
        private const val PAD_ID = 0L
        private const val UNK_ID = 100L
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var vocab: Map<String, Long> = emptyMap()
    private var isReady = false

    /**
     * Load the ONNX model and tokenizer from assets.
     * Call this once during app initialization.
     */
    fun initialize() {
        try {
            // Load tokenizer vocabulary
            vocab = loadTokenizerVocab()
            Log.i(TAG, "Tokenizer loaded: ${vocab.size} tokens")

            // Load ONNX model
            env = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(MODEL_PATH).use { it.readBytes() }
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            session = env!!.createSession(modelBytes, opts)
            isReady = true
            Log.i(TAG, "ONNX model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize embedding model: ${e.message}", e)
            isReady = false
        }
    }

    /**
     * Generate a 384-dim embedding for a single text string.
     * Returns null if the model isn't loaded.
     */
    fun embed(text: String): FloatArray? {
        if (!isReady || session == null || env == null) {
            Log.w(TAG, "Model not ready, returning null")
            return null
        }

        try {
            // Tokenize
            val tokens = tokenize(text)

            // Prepare tensors
            val seqLen = tokens.size.toLong()
            val shape = longArrayOf(1, seqLen)

            val inputIds = OnnxTensor.createTensor(
                env!!, LongBuffer.wrap(tokens.toLongArray()), shape
            )
            val attentionMask = OnnxTensor.createTensor(
                env!!,
                LongBuffer.wrap(LongArray(tokens.size) { 1L }),
                shape
            )
            val tokenTypeIds = OnnxTensor.createTensor(
                env!!,
                LongBuffer.wrap(LongArray(tokens.size) { 0L }),
                shape
            )

            val inputs = mapOf(
                "input_ids" to inputIds,
                "attention_mask" to attentionMask,
                "token_type_ids" to tokenTypeIds,
            )

            // Run inference
            val output = session!!.run(inputs)

            // Get token embeddings: shape (1, seq_len, 384)
            // The output name is typically "last_hidden_state" or the first output
            val tensor = output[0].value
            @Suppress("UNCHECKED_CAST")
            val embeddings = tensor as Array<Array<FloatArray>>
            val tokenEmbeddings = embeddings[0] // (seq_len, 384)

            // Mean pooling over attended tokens (skip [CLS] and [SEP] padding)
            val pooled = FloatArray(EMBEDDING_DIM)
            val tokenCount = tokens.size // all are attended (no padding)
            for (i in 0 until tokenCount) {
                for (d in 0 until EMBEDDING_DIM) {
                    pooled[d] += tokenEmbeddings[i][d]
                }
            }
            for (d in 0 until EMBEDDING_DIM) {
                pooled[d] /= tokenCount
            }

            // L2 normalize
            var norm = 0.0f
            for (d in 0 until EMBEDDING_DIM) {
                norm += pooled[d] * pooled[d]
            }
            norm = sqrt(norm)
            if (norm > 0) {
                for (d in 0 until EMBEDDING_DIM) {
                    pooled[d] /= norm
                }
            }

            // Clean up tensors
            inputIds.close()
            attentionMask.close()
            tokenTypeIds.close()
            output.close()

            return pooled
        } catch (e: Exception) {
            Log.e(TAG, "Embedding failed: ${e.message}", e)
            return null
        }
    }

    /**
     * Whether the model is loaded and ready for inference.
     */
    fun ready(): Boolean = isReady

    /**
     * Release resources.
     */
    fun close() {
        try {
            session?.close()
            env?.close()
        } catch (_: Exception) {}
        session = null
        env = null
        isReady = false
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  TOKENIZER (WordPiece, compatible with all-MiniLM-L6-v2)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Load vocab from HuggingFace tokenizer.json format.
     * Extracts the "vocab" map from model.vocab in tokenizer.json.
     */
    private fun loadTokenizerVocab(): Map<String, Long> {
        val json = context.assets.open(TOKENIZER_PATH).bufferedReader().readText()
        val parsed = Gson().fromJson<Map<String, Any>>(
            json, object : TypeToken<Map<String, Any>>() {}.type
        )

        @Suppress("UNCHECKED_CAST")
        val model = parsed["model"] as? Map<String, Any> ?: return emptyMap()

        @Suppress("UNCHECKED_CAST")
        val vocabRaw = model["vocab"] as? Map<String, Double> ?: return emptyMap()

        return vocabRaw.mapValues { it.value.toLong() }
    }

    /**
     * Tokenize text using WordPiece algorithm.
     * Returns token IDs including [CLS] and [SEP], truncated to MAX_SEQ_LEN.
     */
    private fun tokenize(text: String): List<Long> {
        val tokens = mutableListOf(CLS_ID)

        // Basic pre-tokenization: lowercase, split on whitespace and punctuation
        val words = text.lowercase().trim()
            .replace(Regex("[^\\w\\s']"), " $0 ")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }

        for (word in words) {
            val subTokens = wordPieceTokenize(word)
            // Check if adding these tokens would exceed limit (reserve 1 for [SEP])
            if (tokens.size + subTokens.size + 1 > MAX_SEQ_LEN) break
            tokens.addAll(subTokens)
        }

        tokens.add(SEP_ID)
        return tokens
    }

    /**
     * WordPiece tokenization for a single word.
     * Splits into known subwords using "##" prefix for continuation tokens.
     */
    private fun wordPieceTokenize(word: String): List<Long> {
        if (word.isEmpty()) return emptyList()

        // Check if whole word is in vocab
        val wholeId = vocab[word]
        if (wholeId != null) return listOf(wholeId)

        val result = mutableListOf<Long>()
        var start = 0

        while (start < word.length) {
            var end = word.length
            var found = false

            while (start < end) {
                val sub = if (start == 0) {
                    word.substring(start, end)
                } else {
                    "##${word.substring(start, end)}"
                }

                val id = vocab[sub]
                if (id != null) {
                    result.add(id)
                    start = end
                    found = true
                    break
                }
                end--
            }

            if (!found) {
                // Unknown character, use [UNK]
                result.add(UNK_ID)
                start++
            }
        }

        return result
    }
}
