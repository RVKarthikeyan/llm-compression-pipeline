package com.example.my_ai

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Minimal on-device tokenizer matching SentencePiece (Unigram) behaviour
 * used by Gemma models. Uses vocab.json + tokenizer_config.json.
 *
 * Supports two vocab.json formats:
 *   Format A (id→token): { "0": "<pad>", "1": "<eos>", ... }
 *   Format B (token→id): { "<pad>": 0, "<eos>": 1, ... }
 *
 * Encoding:  SentencePiece-style preprocessing (space→▁, prepend ▁)
 *            then greedy longest-prefix matching.
 * Decoding:  direct id → string lookup, ▁ → space.
 */
class SimpleTokenizer(vocabJsonPath: String, configJsonPath: String) {

    companion object {
        private const val TAG = "Tokenizer"
        // Gemma default special token IDs
        private const val DEFAULT_BOS = 2
        private const val DEFAULT_EOS = 1
        private const val DEFAULT_PAD = 0
        private const val DEFAULT_START_OF_TURN = 106
        private const val DEFAULT_END_OF_TURN = 107
    }

    // id → token string  (for decoding)
    private val id2token: Map<Int, String>
    // token string → id  (for encoding)
    private val token2id: Map<String, Int>

    val vocabSize: Int
    val bosTokenId: Int
    val eosTokenId: Int
    val padTokenId: Int
    val startOfTurnId: Int
    val endOfTurnId: Int

    // max token length (used for greedy prefix encode)
    private val maxTokenLen: Int

    init {
        // ── Load vocab.json ─────────────────────────────────────────────
        val vocabText = File(vocabJsonPath).readText(Charsets.UTF_8)
        val vocabObj = JSONObject(vocabText)

        val id2tok = mutableMapOf<Int, String>()
        val tok2id = mutableMapOf<String, Int>()

        // Auto-detect format: check first key
        val firstKey = vocabObj.keys().next()
        val isIdToToken = firstKey.toIntOrNull() != null

        Log.i(TAG, "Vocab format: ${if (isIdToToken) "id->token" else "token->id"}, first key: $firstKey")

        val keys = vocabObj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            if (isIdToToken) {
                // Format A: { "0": "<pad>", ... }
                val id = key.toInt()
                val token = vocabObj.getString(key)
                id2tok[id] = token
                if (token !in tok2id) tok2id[token] = id
            } else {
                // Format B: { "<pad>": 0, ... }
                val token = key
                val id = vocabObj.getInt(key)
                id2tok[id] = token
                if (token !in tok2id) tok2id[token] = id
            }
        }
        id2token = id2tok
        token2id = tok2id
        vocabSize = id2tok.size
        maxTokenLen = tok2id.keys.maxOfOrNull { it.length } ?: 1

        Log.i(TAG, "Vocab loaded: $vocabSize entries, maxTokenLen=$maxTokenLen")
        val samples = id2tok.entries.take(5).joinToString { "${it.key}->'${it.value}'" }
        Log.i(TAG, "Sample vocab entries: $samples")

        // ── Load tokenizer_config.json ──────────────────────────────────
        val configText = File(configJsonPath).readText(Charsets.UTF_8)
        val configObj = JSONObject(configText)

        bosTokenId = configObj.optInt("bos_token_id", DEFAULT_BOS)
        eosTokenId = configObj.optInt("eos_token_id", DEFAULT_EOS)
        padTokenId = configObj.optInt("pad_token_id", DEFAULT_PAD)

        // Resolve special tokens for Gemma chat template
        startOfTurnId = configObj.optInt("start_of_turn_id",
            tok2id["<start_of_turn>"] ?: DEFAULT_START_OF_TURN)
        endOfTurnId = configObj.optInt("end_of_turn_id",
            tok2id["<end_of_turn>"] ?: DEFAULT_END_OF_TURN)

        Log.i(TAG, "Config: bos=$bosTokenId, eos=$eosTokenId, pad=$padTokenId, " +
                "startTurn=$startOfTurnId, endTurn=$endOfTurnId")
        Log.i(TAG, "BOS='${id2tok[bosTokenId]}', EOS='${id2tok[eosTokenId]}', " +
                "startTurn='${id2tok[startOfTurnId]}', endTurn='${id2tok[endOfTurnId]}'")
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Core text→IDs encoding (SentencePiece-style)
    // ─────────────────────────────────────────────────────────────────────

    /**
     * SentencePiece preprocessing: replace every space with ▁ and
     * prepend ▁ to the beginning. Then greedy longest-prefix match.
     *
     * "Hello world" → "▁Hello▁world"
     * "user\nHi"    → "▁user\nHi"     (newlines are NOT replaced)
     */
    private fun encodeToIds(text: String): MutableList<Int> {
        // SentencePiece normalization: ▁ replaces spaces, prepend ▁
        val normalized = "\u2581" + text.replace(" ", "\u2581")
        val ids = mutableListOf<Int>()
        var i = 0
        while (i < normalized.length) {
            var matched = false
            val end = minOf(i + maxTokenLen, normalized.length)
            // Try longest match first
            for (len in (end - i) downTo 1) {
                val sub = normalized.substring(i, i + len)
                val id = token2id[sub]
                if (id != null) {
                    ids.add(id)
                    i += len
                    matched = true
                    break
                }
            }
            if (!matched) {
                // Single char lookup
                val ch = normalized.substring(i, i + 1)
                val id = token2id[ch]
                if (id != null) {
                    ids.add(id)
                } else {
                    // Byte fallback: <0xNN> for each UTF-8 byte
                    for (b in ch.toByteArray(Charsets.UTF_8)) {
                        val hex = String.format("<0x%02X>", b.toInt() and 0xFF)
                        val byteId = token2id[hex]
                        if (byteId != null) ids.add(byteId)
                    }
                }
                i += 1
            }
        }
        return ids
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────────────

    fun encode(text: String, addBos: Boolean = true): LongArray {
        val ids = mutableListOf<Int>()
        if (addBos) ids.add(bosTokenId)
        ids.addAll(encodeToIds(text))
        return ids.map { it.toLong() }.toLongArray()
    }

    /**
     * Format with Gemma chat template:
     *   <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
     */
    fun formatGemmaChat(userText: String, context: String? = null): LongArray {
        val ids = mutableListOf<Int>()

        ids.add(bosTokenId)
        ids.add(startOfTurnId)

        val userContent = if (!context.isNullOrBlank()) {
            "user\nYou are a precise document assistant. Follow these rules strictly:\n" +
            "1. Answer ONLY using facts explicitly stated in the DOCUMENT below.\n" +
            "2. Do NOT infer, assume, or add any information not present in the document.\n" +
            "3. If the document mentions multiple related facts, include ALL of them.\n" +
            "4. If the answer is not in the document, say \"The document does not contain this information.\"\n\n" +
            "DOCUMENT:\n$context\n\nUsing ONLY the document above, answer this question: $userText"
        } else {
            "user\n$userText"
        }
        ids.addAll(encodeToIds(userContent))

        ids.add(endOfTurnId)
        ids.addAll(encodeToIds("\n"))
        ids.add(startOfTurnId)
        ids.addAll(encodeToIds("model\n"))

        Log.i(TAG, "Chat formatted: ${ids.size} tokens, first 20 IDs: ${ids.take(20)}")
        // Log the text representation of first tokens for verification
        val firstTokenTexts = ids.take(20).map { id -> "${id}='${id2token[id] ?: "?"}'" }
        Log.i(TAG, "Token text: $firstTokenTexts")
        return ids.map { it.toLong() }.toLongArray()
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id == bosTokenId || id == padTokenId) continue
            if (id == eosTokenId || id == endOfTurnId) break
            if (id == startOfTurnId) continue
            val token = id2token[id] ?: ""
            sb.append(token)
        }
        return sb.toString()
            .replace("\u2581", " ")
            .trimStart()
    }
}
