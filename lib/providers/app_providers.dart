import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../services/backend_service.dart';
import '../services/objectbox_service.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure providers
// ─────────────────────────────────────────────────────────────────────────────

final objectBoxProvider = Provider<ObjectBoxService>((ref) {
  throw UnimplementedError('objectBoxProvider must be overridden in main()');
});

final backendServiceProvider =
    Provider<BackendService>((_) => BackendService());

// ─────────────────────────────────────────────────────────────────────────────
// Secure Storage (HF Token)
// ─────────────────────────────────────────────────────────────────────────────

const _storage = FlutterSecureStorage();
const _hfTokenKey = 'hf_access_token';
const _wandbKeyKey = 'wandb_api_key';
const _backendUrlKey = 'backend_url';

class SettingsNotifier extends AsyncNotifier<String> {
  @override
  Future<String> build() async =>
      await _storage.read(key: _hfTokenKey) ?? '';

  Future<void> saveToken(String token) async {
    await _storage.write(key: _hfTokenKey, value: token);
    state = AsyncData(token);
  }
}

final settingsProvider =
    AsyncNotifierProvider<SettingsNotifier, String>(SettingsNotifier.new);

/// Stores the W&B API key in secure storage.
final wandbKeyProvider = FutureProvider<String>((ref) async {
  return await _storage.read(key: _wandbKeyKey) ?? '';
});

Future<void> saveWandbKey(String key) async {
  await _storage.write(key: _wandbKeyKey, value: key);
}

Future<String> readWandbKey() async {
  return await _storage.read(key: _wandbKeyKey) ?? '';
}

/// Stores the backend URL in secure storage.
Future<void> saveBackendUrl(String url) async {
  await _storage.write(key: _backendUrlKey, value: url);
}

Future<String> readBackendUrl() async {
  return await _storage.read(key: _backendUrlKey) ?? '';
}

// ─────────────────────────────────────────────────────────────────────────────
// Model state (loading / loaded / error)
// ─────────────────────────────────────────────────────────────────────────────

enum ModelLoadState { idle, loading, loaded, error }

class ModelState {
  final ModelLoadState loadState;
  final String? modelPath;
  final String? vocabPath;      // *_vocab.json
  final String? configPath;     // *_tokenizer_config.json
  final String? statusMessage;
  final bool hasTokenizer;

  const ModelState({
    this.loadState = ModelLoadState.idle,
    this.modelPath,
    this.vocabPath,
    this.configPath,
    this.statusMessage,
    this.hasTokenizer = false,
  });

  ModelState copyWith({
    ModelLoadState? loadState,
    String? modelPath,
    String? vocabPath,
    String? configPath,
    String? statusMessage,
    bool? hasTokenizer,
  }) =>
      ModelState(
        loadState: loadState ?? this.loadState,
        modelPath: modelPath ?? this.modelPath,
        vocabPath: vocabPath ?? this.vocabPath,
        configPath: configPath ?? this.configPath,
        statusMessage: statusMessage ?? this.statusMessage,
        hasTokenizer: hasTokenizer ?? this.hasTokenizer,
      );

  bool get isLoaded => loadState == ModelLoadState.loaded;
  bool get isLoading => loadState == ModelLoadState.loading;
  bool get isDemoMode =>
      statusMessage != null && statusMessage!.contains('demo');
  bool get needsTokenizer => isLoaded && !hasTokenizer && !isDemoMode;
}

class ModelNotifier extends Notifier<ModelState> {
  static const _channel = MethodChannel('com.example.my_ai/executorch');

  @override
  ModelState build() => const ModelState();

  /// Load the .pte model + tokenizer via the native ExecuTorch channel.
  /// [vocabPath] and [configPath] are the vocab.json and tokenizer_config.json
  /// files exported from the Colab notebook.
  Future<void> loadModel(
    String path, {
    String? vocabPath,
    String? configPath,
  }) async {
    state = ModelState(
      loadState: ModelLoadState.loading,
      modelPath: path,
      vocabPath: vocabPath,
      configPath: configPath,
      statusMessage: 'Loading model into memory…',
    );

    try {
      final args = <String, dynamic>{'path': path};
      if (vocabPath != null) args['vocabPath'] = vocabPath;
      if (configPath != null) args['configPath'] = configPath;

      debugPrint('[MODEL] Loading model: $path');
      debugPrint('[MODEL] Vocab: $vocabPath');
      debugPrint('[MODEL] Config: $configPath');

      final result = await _channel
          .invokeMethod<String>('loadModel', args)
          .timeout(
            const Duration(seconds: 120),
            onTimeout: () => 'demo: model load timed out after 120s',
          );

      debugPrint('[MODEL] Load result: $result');

      final hasTok = result != null &&
          !result.contains('no_tokenizer') &&
          !result.contains('demo');

      state = state.copyWith(
        loadState: ModelLoadState.loaded,
        statusMessage: result,
        hasTokenizer: hasTok,
      );
    } on PlatformException catch (e) {
      debugPrint('[MODEL] PlatformException: ${e.message}');
      state = state.copyWith(
        loadState: ModelLoadState.loaded,
        statusMessage: 'demo: ${e.message}',
      );
    } on MissingPluginException {
      debugPrint('[MODEL] MissingPluginException');
      state = state.copyWith(
        loadState: ModelLoadState.loaded,
        statusMessage: 'demo: native channel not available',
      );
    } catch (e) {
      debugPrint('[MODEL] Error: $e');
      state = state.copyWith(
        loadState: ModelLoadState.error,
        statusMessage: 'Error: $e',
      );
    }
  }

  void clearModel() => state = const ModelState();
}

final modelProvider =
    NotifierProvider<ModelNotifier, ModelState>(ModelNotifier.new);

// ─────────────────────────────────────────────────────────────────────────────
// Chat state
// ─────────────────────────────────────────────────────────────────────────────

class ChatMessage {
  final String role; // 'user' | 'ai'
  final String content;
  final List<String>? ragContext;
  final bool noContextWarning;
  const ChatMessage({
    required this.role,
    required this.content,
    this.ragContext,
    this.noContextWarning = false,
  });
}

class ChatState {
  final List<ChatMessage> messages;
  final bool isInferencing;

  const ChatState({this.messages = const [], this.isInferencing = false});

  ChatState copyWith({List<ChatMessage>? messages, bool? isInferencing}) =>
      ChatState(
        messages: messages ?? this.messages,
        isInferencing: isInferencing ?? this.isInferencing,
      );
}

class ChatNotifier extends Notifier<ChatState> {
  static const _channel = MethodChannel('com.example.my_ai/executorch');

  /// Accumulates streamed tokens during inference.
  final StringBuffer _streamBuffer = StringBuffer();

  @override
  ChatState build() {
    // Listen for native log messages and streamed tokens
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'log') {
        debugPrint('[NATIVE] ${call.arguments}');
      } else if (call.method == 'onToken') {
        _onStreamedToken(call.arguments as String);
      }
    });
    return const ChatState();
  }

  /// Called for each token streamed from native side.
  void _onStreamedToken(String token) {
    _streamBuffer.write(token);
    final partial = _cleanSpecialTokens(_streamBuffer.toString());

    // Update the last message (the AI placeholder) in-place
    final msgs = [...state.messages];
    if (msgs.isNotEmpty && msgs.last.role == 'ai') {
      msgs[msgs.length - 1] = ChatMessage(
        role: 'ai',
        content: partial,
        ragContext: msgs.last.ragContext,
        noContextWarning: msgs.last.noContextWarning,
      );
      state = state.copyWith(messages: msgs);
    }
  }

  Future<void> sendMessage(String userText) async {
    if (userText.trim().isEmpty) return;

    state = state.copyWith(
      messages: [
        ...state.messages,
        ChatMessage(role: 'user', content: userText),
      ],
      isInferencing: true,
    );

    // RAG: retrieve context from ObjectBox
    final obs = ref.read(objectBoxProvider);
    final totalChunks = obs.count;
    debugPrint('[RAG] ObjectBox has $totalChunks chunks');

    // Always score chunks by relevance to the query.
    // For small docs (≤30 chunks), score ALL chunks so we don't miss anything.
    // For larger docs, use DB-level keyword search first.
    List<String> contextChunks;
    if (totalChunks > 0 && totalChunks <= 30) {
      contextChunks = obs.scoredSearchAll(userText);
      debugPrint('[RAG] Scored ALL chunks → ${contextChunks.length} selected');
    } else {
      contextChunks = obs.searchContext(userText);
      debugPrint('[RAG] Keyword search returned ${contextChunks.length} chunks');
    }
    final hasContext = contextChunks.isNotEmpty;

    // Build context string, capped at 1500 chars to fit within
    // model's actual max_context_len (~940 tokens) with prompt overhead.
    String? ctx;
    if (hasContext) {
      final buf = StringBuffer();
      for (final chunk in contextChunks) {
        if (buf.length + chunk.length + 2 > 1500 && buf.isNotEmpty) break;
        if (buf.isNotEmpty) buf.write('\n\n');
        buf.write(chunk);
      }
      ctx = buf.toString();
      debugPrint('[RAG] Context: ${ctx.length} chars from ${contextChunks.length} chunks');
      debugPrint('[RAG] Context preview: ${ctx.substring(0, ctx.length < 300 ? ctx.length : 300)}');
    } else {
      debugPrint('[RAG] NO context found!');
    }

    String aiResponse;
    bool usedNative = false;

    // Add an empty AI placeholder for streaming — tokens will fill it
    _streamBuffer.clear();
    state = state.copyWith(
      messages: [
        ...state.messages,
        ChatMessage(
          role: 'ai',
          content: '',
          ragContext: hasContext ? contextChunks : null,
          noContextWarning: !hasContext,
        ),
      ],
    );

    debugPrint('[CHAT] Sending to native: prompt="$userText", hasContext=$hasContext');

    try {
      aiResponse = await _channel.invokeMethod<String>(
            'runInference',
            {
              'prompt': userText,
              'context': ctx,
            },
          )
          .timeout(
            const Duration(seconds: 900),
            onTimeout: () {
              debugPrint('[CHAT] Inference timed out after 900s');
              return '(Inference timed out — model may be too slow on this device. '
                  'Try a simpler prompt.)';
            },
          ) ??
          '(no response)';
      usedNative = true;
      debugPrint('[CHAT] Native response: ${aiResponse.length} chars');

      // Strip any leftover special tokens the native side missed
      aiResponse = _cleanSpecialTokens(aiResponse);
    } on PlatformException catch (e) {
      debugPrint('[CHAT] PlatformException: ${e.code} - ${e.message}');
      aiResponse = _demo(userText, contextChunks, e.message);
    } on MissingPluginException {
      debugPrint('[CHAT] MissingPluginException');
      aiResponse = _demo(userText, contextChunks, null);
    } catch (e) {
      debugPrint('[CHAT] Error: $e');
      aiResponse = _demo(userText, contextChunks, e.toString());
    }

    // Small delay for demo mode so it feels natural
    if (!usedNative) {
      await Future.delayed(
          Duration(milliseconds: 300 + Random().nextInt(500)));
    }

    // Finalize the AI placeholder with the complete response
    final msgs = [...state.messages];
    if (msgs.isNotEmpty && msgs.last.role == 'ai') {
      msgs[msgs.length - 1] = ChatMessage(
        role: 'ai',
        content: aiResponse,
        ragContext: hasContext ? contextChunks : null,
        noContextWarning: !hasContext,
      );
    }
    state = state.copyWith(
      messages: msgs,
      isInferencing: false,
    );
  }

  String _demo(String query, List<String> chunks, String? error) {
    if (chunks.isNotEmpty) {
      final preview = chunks.first.length > 200
          ? '${chunks.first.substring(0, 200)}…'
          : chunks.first;
      return 'Based on your documents:\n\n$preview\n\n'
          '(Demo mode${error != null ? ': $error' : ''})';
    }
    return '(Demo mode) I don\'t have relevant context for: "$query".\n'
        'Upload a PDF for better answers.'
        '${error != null ? '\nNative: $error' : ''}';
  }

  static const _specialTokens = [
    '<|begin_of_text|>', '<|end_of_text|>',
    '<|start_header_id|>', '<|end_header_id|>',
    '<|eot_id|>', '<|finetune_right_pad_id|>',
    '<bos>', '<eos>',
    '<start_of_turn>', '<end_of_turn>',
  ];

  String _cleanSpecialTokens(String text) {
    var result = text;
    for (final token in _specialTokens) {
      result = result.replaceAll(token, '');
    }
    return result.trim();
  }

  void clearHistory() {
    state = const ChatState();
    // Reset KV-cache position counter (no model reload — safe)
    try {
      _channel.invokeMethod('resetCache');
    } catch (_) {}
  }
}

final chatProvider =
    NotifierProvider<ChatNotifier, ChatState>(ChatNotifier.new);
