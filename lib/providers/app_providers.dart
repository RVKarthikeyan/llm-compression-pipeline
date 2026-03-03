import 'dart:math';

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

      final result = await _channel.invokeMethod<String>('loadModel', args);

      final hasTok = result != null &&
          !result.contains('no_tokenizer') &&
          !result.contains('demo');

      state = state.copyWith(
        loadState: ModelLoadState.loaded,
        statusMessage: result,
        hasTokenizer: hasTok,
      );
    } on PlatformException catch (e) {
      state = state.copyWith(
        loadState: ModelLoadState.loaded,
        statusMessage: 'demo: ${e.message}',
      );
    } on MissingPluginException {
      state = state.copyWith(
        loadState: ModelLoadState.loaded,
        statusMessage: 'demo: native channel not available',
      );
    } catch (e) {
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

  @override
  ChatState build() => const ChatState();

  Future<void> sendMessage(String userText) async {
    if (userText.trim().isEmpty) return;

    state = state.copyWith(
      messages: [
        ...state.messages,
        ChatMessage(role: 'user', content: userText),
      ],
      isInferencing: true,
    );

    // RAG: retrieve context from ObjectBox vector DB
    final obs = ref.read(objectBoxProvider);
    final contextChunks = obs.searchContext(userText);
    final hasContext = contextChunks.isNotEmpty;

    // Send user text + context separately — native side applies Gemma template
    final ctx = hasContext ? contextChunks.join('\n') : null;

    String aiResponse;
    bool usedNative = false;

    try {
      aiResponse = await _channel.invokeMethod<String>(
            'runInference',
            {
              'prompt': userText,
              'context': ctx,
            },
          ) ??
          '(no response)';
      usedNative = true;
    } on PlatformException catch (e) {
      aiResponse = _demo(userText, contextChunks, e.message);
    } on MissingPluginException {
      aiResponse = _demo(userText, contextChunks, null);
    } catch (e) {
      aiResponse = _demo(userText, contextChunks, e.toString());
    }

    // Small delay for demo mode so it feels natural
    if (!usedNative) {
      await Future.delayed(
          Duration(milliseconds: 300 + Random().nextInt(500)));
    }

    state = state.copyWith(
      messages: [
        ...state.messages,
        ChatMessage(
          role: 'ai',
          content: aiResponse,
          ragContext: contextChunks,
          noContextWarning: !hasContext,
        ),
      ],
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

  void clearHistory() => state = const ChatState();
}

final chatProvider =
    NotifierProvider<ChatNotifier, ChatState>(ChatNotifier.new);
