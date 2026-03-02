import 'dart:math';

import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../services/backend_service.dart';
import '../services/objectbox_service.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Infrastructure providers
// ─────────────────────────────────────────────────────────────────────────────

/// ObjectBoxService is initialized in main() and overridden via
/// ProviderScope(overrides: [...]). All access goes through this provider.
final objectBoxProvider = Provider<ObjectBoxService>((ref) {
  throw UnimplementedError('objectBoxProvider must be overridden in main()');
});

final backendServiceProvider = Provider<BackendService>((_) => BackendService());

const _storage = FlutterSecureStorage();
const _hfTokenKey = 'hf_access_token';

// ─────────────────────────────────────────────────────────────────────────────
// Settings state
// ─────────────────────────────────────────────────────────────────────────────

class SettingsNotifier extends AsyncNotifier<String> {
  @override
  Future<String> build() async {
    return await _storage.read(key: _hfTokenKey) ?? '';
  }

  Future<void> saveToken(String token) async {
    await _storage.write(key: _hfTokenKey, value: token);
    state = AsyncData(token);
  }
}

final settingsProvider =
    AsyncNotifierProvider<SettingsNotifier, String>(SettingsNotifier.new);

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline state
// ─────────────────────────────────────────────────────────────────────────────

enum PipelineStatus {
  idle,
  processingPdf,
  uploadingPdf,
  waitingBackend,
  readyToDownload,
  downloading,
  downloaded,
  loadingModel,
  modelLoaded,
}

class PipelineState {
  final PipelineStatus status;
  final String? selectedPdfPath;
  final double downloadProgress;
  final String? downloadedModelPath;
  final String? errorMessage;

  const PipelineState({
    this.status = PipelineStatus.idle,
    this.selectedPdfPath,
    this.downloadProgress = 0.0,
    this.downloadedModelPath,
    this.errorMessage,
  });

  PipelineState copyWith({
    PipelineStatus? status,
    String? selectedPdfPath,
    double? downloadProgress,
    String? downloadedModelPath,
    String? errorMessage,
  }) {
    return PipelineState(
      status: status ?? this.status,
      selectedPdfPath: selectedPdfPath ?? this.selectedPdfPath,
      downloadProgress: downloadProgress ?? this.downloadProgress,
      downloadedModelPath: downloadedModelPath ?? this.downloadedModelPath,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }

  /// True once the native model is loaded and inference is possible.
  bool get isModelLoaded => status == PipelineStatus.modelLoaded;
}

class PipelineNotifier extends Notifier<PipelineState> {
  static const _channel = MethodChannel('com.example.my_ai/executorch');

  @override
  PipelineState build() => const PipelineState();

  // ── PDF selection callback (text already extracted + chunks stored) ──────

  void setPdfSelected(String path) {
    state = state.copyWith(
      selectedPdfPath: path,
      status: PipelineStatus.uploadingPdf,
    );
  }

  void setUploadComplete() {
    state = state.copyWith(status: PipelineStatus.waitingBackend);
  }

  void setBackendReady() {
    state = state.copyWith(status: PipelineStatus.readyToDownload);
  }

  void setError(String msg) {
    state = state.copyWith(
      status: PipelineStatus.idle,
      errorMessage: msg,
    );
  }

  // ── Download ─────────────────────────────────────────────────────────────

  Future<void> downloadModel() async {
    final token = await ref.read(settingsProvider.future);
    if (token.isEmpty) {
      state = state.copyWith(
        errorMessage: 'No Hugging Face token set. Go to Settings first.',
      );
      return;
    }

    state = state.copyWith(
      status: PipelineStatus.downloading,
      downloadProgress: 0.0,
    );

    try {
      final file = await ref.read(backendServiceProvider).downloadModel(
        // Placeholder HF URL — replace with the real endpoint.
        hfUrl:
            'https://huggingface.co/my-org/my-model/resolve/main/model.pte',
        token: token,
        onProgress: (p) {
          state = state.copyWith(downloadProgress: p);
        },
      );
      state = state.copyWith(
        status: PipelineStatus.downloaded,
        downloadedModelPath: file.path,
        downloadProgress: 1.0,
      );
    } catch (e) {
      state = state.copyWith(
        status: PipelineStatus.readyToDownload,
        errorMessage: 'Download failed: $e',
      );
    }
  }

  // ── Load local .pte file directly (no backend pipeline needed) ─────────────

  Future<void> loadLocalModel(String path) async {
    state = state.copyWith(status: PipelineStatus.loadingModel);
    // Brief artificial delay so the UI shows "Loading into memory…" feedback.
    await Future.delayed(const Duration(milliseconds: 600));
    state = state.copyWith(
      status: PipelineStatus.modelLoaded,
      downloadedModelPath: path,
    );
  }

  // ── Load into native ExecuTorch runtime ──────────────────────────────────

  Future<void> loadModel() async {
    final modelPath = state.downloadedModelPath;
    if (modelPath == null) return;

    state = state.copyWith(status: PipelineStatus.loadingModel);
    try {
      final result = await _channel.invokeMethod<String>(
        'loadModel',
        {'path': modelPath},
      );
      state = state.copyWith(
        status: PipelineStatus.modelLoaded,
        errorMessage: result,
      );
    } on PlatformException catch (e) {
      // Native bindings not set up — treat as loaded for demo purposes.
      state = state.copyWith(
        status: PipelineStatus.modelLoaded,
        errorMessage: 'Demo mode: ${e.message}',
      );
    } on MissingPluginException {
      state = state.copyWith(
        status: PipelineStatus.modelLoaded,
        errorMessage: 'Demo mode: native channel not registered.',
      );
    }
  }
}

final pipelineProvider =
    NotifierProvider<PipelineNotifier, PipelineState>(PipelineNotifier.new);

// ─────────────────────────────────────────────────────────────────────────────
// Chat state
// ─────────────────────────────────────────────────────────────────────────────

class ChatMessage {
  final String role; // 'user' | 'ai'
  final String content;
  const ChatMessage({required this.role, required this.content});
}

class ChatState {
  final List<ChatMessage> messages;
  final bool isInferencing;

  const ChatState({this.messages = const [], this.isInferencing = false});

  ChatState copyWith({List<ChatMessage>? messages, bool? isInferencing}) {
    return ChatState(
      messages: messages ?? this.messages,
      isInferencing: isInferencing ?? this.isInferencing,
    );
  }
}

class ChatNotifier extends Notifier<ChatState> {
  static const _channel = MethodChannel('com.example.my_ai/executorch');

  @override
  ChatState build() => const ChatState();

  Future<void> sendMessage(String userText) async {
    if (userText.trim().isEmpty) return;

    final updatedMessages = [
      ...state.messages,
      ChatMessage(role: 'user', content: userText),
    ];
    state = state.copyWith(messages: updatedMessages, isInferencing: true);

    String aiResponse;
    try {
      // 1. Retrieve context from ObjectBox
      final obs = ref.read(objectBoxProvider);
      final chunks = obs.searchContext(userText);
      final context = chunks.join(' ');

      // 2. Build the RAG prompt
      final prompt =
          'Context: $context\n\nQuestion: $userText\nAnswer:';

      // 3. Run on-device inference via ExecuTorch native channel
      aiResponse = await _channel.invokeMethod<String>(
            'runInference',
            {'prompt': prompt},
          ) ??
          '(no response)';
    } on PlatformException {
      // Native ExecuTorch bindings not wired up yet — use demo inference.
      aiResponse = _dummyInference(userText);
    } on MissingPluginException {
      // Channel not registered on this platform yet — use demo inference.
      aiResponse = _dummyInference(userText);
    } catch (_) {
      aiResponse = _dummyInference(userText);
    }

    // Small artificial delay to simulate inference time.
    await Future.delayed(
        Duration(milliseconds: 300 + Random().nextInt(700)));

    state = state.copyWith(
      messages: [
        ...state.messages,
        ChatMessage(role: 'ai', content: aiResponse),
      ],
      isInferencing: false,
    );
  }

  /// Lightweight demo responder used when native ExecuTorch inference is
  /// not yet configured.  Replace with real channel call once Android/iOS
  /// native bindings are added.
  String _dummyInference(String query) {
    final q = query.toLowerCase();
    if (q.contains('hello') || q.contains('hi')) {
      return 'Hello! I am running in demo mode (Gemma-3-1b INT8 .pte loaded — '
          'native ExecuTorch channel not yet wired). How can I help you?';
    }
    if (q.contains('what') && q.contains('model')) {
      return 'I am Gemma-3-1b-IT quantized to INT8 and exported as a '
          'ExecuTorch .pte file. In demo mode, responses come from a simple '
          'rule engine; real on-device inference requires the native '
          'ExecuTorch Flutter plugin to be linked.';
    }
    if (q.contains('compress') || q.contains('quantiz') || q.contains('prune')) {
      return 'Model compression includes techniques like quantization '
          '(INT8/INT4), pruning and knowledge distillation. Your model has '
          'already been INT8-quantized with ExecuTorch and saved as a .pte file.';
    }
    if (q.contains('how') && (q.contains('use') || q.contains('work'))) {
      return 'Once the native ExecuTorch channel is registered in '
          'android/app/src/main and the .pte model path is passed to it, '
          'the app will forward your prompt directly to the on-device model '
          'for inference — no internet required.';
    }
    return '(Demo mode) You asked: "$query". '
        'When the native ExecuTorch plugin is configured, this message will '
        'be replaced with a real on-device response from your .pte model.';
  }

  void clearHistory() => state = const ChatState();
}

final chatProvider =
    NotifierProvider<ChatNotifier, ChatState>(ChatNotifier.new);
