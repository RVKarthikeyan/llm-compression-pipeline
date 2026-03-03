package com.example.my_ai

import android.util.Log
import org.json.JSONObject
import java.io.File

/**
 * Minimal on-device tokenizer that uses the vocab.json + tokenizer_config.json
 * exported from the Colab notebook.
 *
 * Encoding:  greedy longest-prefix matching on the vocab (good enough for Gemma
 *            SentencePiece-based models when the full vocab is available).
 * Decoding:  direct id → string lookup.
 */
class SimpleTokenizer(vocabJsonPath: String, configJsonPath: String) {

    companion object {
        private const val TAG = "Tokenizer"
        // Gemma default special token IDs
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
        // ── Load vocab.json: { "0": "<pad>", "1": "<eos>", ... } ────────
        val vocabText = File(vocabJsonPath).readText(Charsets.UTF_8)
        val vocabObj = JSONObject(vocabText)

        val id2tok = mutableMapOf<Int, String>()
        val tok2id = mutableMapOf<String, Int>()

        val keys = vocabObj.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            val id = key.toInt()
            val token = vocabObj.getString(key)
            id2tok[id] = token
            // For encoding, prefer the LOWEST id when there are duplicates
            if (token !in tok2id) {
                tok2id[token] = id
            }
        }
        id2token = id2tok
        token2id = tok2id
        vocabSize = id2tok.size
        maxTokenLen = tok2id.keys.maxOfOrNull { it.length } ?: 1

        // ── Load tokenizer_config.json ──────────────────────────────────
        val configText = File(configJsonPath).readText(Charsets.UTF_8)
        val configObj = JSONObject(configText)

        bosTokenId = configObj.optInt("bos_token_id", 2)
        eosTokenId = configObj.optInt("eos_token_id", 1)
        padTokenId = configObj.optInt("pad_token_id", 0)

        // Resolve special tokens for Gemma chat template
        startOfTurnId = configObj.optInt("start_of_turn_id",
            tok2id["<start_of_turn>"] ?: DEFAULT_START_OF_TURN)
        endOfTurnId = configObj.optInt("end_of_turn_id",
            tok2id["<end_of_turn>"] ?: DEFAULT_END_OF_TURN)

        Log.i(TAG, "Vocab loaded: size=$vocabSize, bos=$bosTokenId, eos=$eosTokenId, " +
                "startTurn=$startOfTurnId, endTurn=$endOfTurnId, maxTokenLen=$maxTokenLen")
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Core text→IDs encoding (no special tokens)
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Encode raw text into token IDs using greedy longest-prefix matching.
     * Does NOT add BOS/EOS or any special tokens.
     */
    private fun encodeToIds(text: String): MutableList<Int> {
        val ids = mutableListOf<Int>()
        var i = 0
        while (i < text.length) {
            var matched = false
            // Try longest prefix first
            val end = minOf(i + maxTokenLen, text.length)
            for (len in (end - i) downTo 1) {
                val sub = text.substring(i, i + len)
                val id = token2id[sub]
                if (id != null) {
                    ids.add(id)
                    i += len
                    matched = true
                    break
                }
            }
            if (!matched) {
                // Character-level fallback: try single char
                val ch = text.substring(i, i + 1)
                val id = token2id[ch]
                if (id != null) {
                    ids.add(id)
                } else {
                    // For SentencePiece, the '▁' prefix is common
                    val withPrefix = "▁$ch"
                    val prefixId = token2id[withPrefix]
                    if (prefixId != null) {
                        ids.add(prefixId)
                    }
                    // else: truly unknown, skip
                }
                i += 1
            }
        }
        return ids
    }

    // ─────────────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Encode text → list of token IDs.
     * Uses greedy longest-prefix matching (character-level fallback).
     * Prepends BOS token.
     */
    fun encode(text: String, addBos: Boolean = true): LongArray {
        val ids = mutableListOf<Int>()
        if (addBos) ids.add(bosTokenId)
        ids.addAll(encodeToIds(text))
        return ids.map { it.toLong() }.toLongArray()
    }

    /**
     * Format a user message + optional RAG context using the Gemma chat template:
     *
     *   <bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
     *
     * Returns token IDs ready for the model's forward() method.
     */
    fun formatGemmaChat(userText: String, context: String? = null): LongArray {
        val ids = mutableListOf<Int>()

        // <bos>
        ids.add(bosTokenId)

        // <start_of_turn>
        ids.add(startOfTurnId)

        // "user\n" + content
        val userContent = if (!context.isNullOrBlank()) {
            "user\nContext:\n$context\n\nQuestion: $userText\nAnswer based on the context above."
        } else {
            "user\n$userText"
        }
        ids.addAll(encodeToIds(userContent))

        // <end_of_turn>
        ids.add(endOfTurnId)

        // newline between turns
        val nlIds = encodeToIds("\n")
        ids.addAll(nlIds)

        // <start_of_turn>model\n
        ids.add(startOfTurnId)
        ids.addAll(encodeToIds("model\n"))

        Log.d(TAG, "Chat formatted: ${ids.size} tokens")
        return ids.map { it.toLong() }.toLongArray()
    }

    /**
     * Decode token IDs → text string.
     * Strips SentencePiece '▁' prefix markers and replaces with spaces.
     */
    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id == bosTokenId || id == padTokenId) continue
            if (id == eosTokenId || id == endOfTurnId) break
            if (id == startOfTurnId) continue  // skip special tokens in output
            val token = id2token[id] ?: ""
            sb.append(token)
        }
        // SentencePiece uses '▁' (U+2581) as word boundary
        return sb.toString()
            .replace("▁", " ")
            .trimStart()
    }
}
