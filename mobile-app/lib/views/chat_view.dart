import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';

import '../providers/app_providers.dart';

class ChatView extends ConsumerStatefulWidget {
  const ChatView({super.key});

  @override
  ConsumerState<ChatView> createState() => _ChatViewState();
}

class _ChatViewState extends ConsumerState<ChatView>
    with SingleTickerProviderStateMixin {
  final _ctrl = TextEditingController();
  final _scroll = ScrollController();
  late final AnimationController _pulseCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    _scroll.dispose();
    _pulseCtrl.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _send() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty) return;
    _ctrl.clear();
    await ref.read(chatProvider.notifier).sendMessage(text);
    _scrollToBottom();
  }

  /// Pick a PDF and load it into the knowledge base for RAG.
  Future<void> _selectKnowledge() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'txt'],
      dialogTitle: 'Select Knowledge PDF',
    );
    if (res == null) return;

    final file = File(res.files.single.path!);
    final fileName = res.files.single.name;

    try {
      String text;
      if (fileName.toLowerCase().endsWith('.pdf')) {
        final doc = PdfDocument(inputBytes: file.readAsBytesSync());
        text = PdfTextExtractor(doc).extractText();
        doc.dispose();
      } else {
        text = await file.readAsString();
      }

      final obs = ref.read(objectBoxProvider);
      await obs.replaceChunksFromText(text);

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Loaded "$fileName" — ${obs.count} chunks stored'),
          duration: const Duration(seconds: 3),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load PDF: $e')),
      );
    }
  }

  Future<void> _loadModel() async {
    // Let user pick all files at once: .pte + tokenizer.json (+ optional tokenizer_config.json)
    final res = await FilePicker.platform.pickFiles(
      type: FileType.any,
      allowMultiple: true,
      dialogTitle: 'Select .pte model + tokenizer files',
    );
    if (res == null || res.files.isEmpty) return;

    String? ptePath;
    String? vocabPath;
    String? configPath;

    // Sort files by extension / naming convention
    for (final file in res.files) {
      final path = file.path;
      if (path == null) continue;
      final name = path.split(Platform.pathSeparator).last.toLowerCase();
      if (name.endsWith('.pte')) {
        ptePath = path;
      } else if (name == 'tokenizer.json' ||
          name.endsWith('_vocab.json') ||
          name == 'vocab.json') {
        vocabPath = path;
      } else if (name.endsWith('_tokenizer_config.json') ||
          name == 'tokenizer_config.json') {
        configPath = path;
      }
    }

    // If only one file picked, try auto-detect in same directory
    if (ptePath == null) {
      // Maybe user picked a single .pte file (old flow)
      final singlePath = res.files.first.path;
      if (singlePath != null &&
          singlePath.toLowerCase().endsWith('.pte')) {
        ptePath = singlePath;
      }
    }

    if (ptePath == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('No .pte model file found in selection.'),
        ),
      );
      return;
    }

    // Try auto-detect from cache directory if not explicitly selected
    if (vocabPath == null) {
      final dir = File(ptePath).parent;
      try {
        final files = dir.listSync().whereType<File>();
        for (final f in files) {
          final name =
              f.path.split(Platform.pathSeparator).last.toLowerCase();
          if (vocabPath == null &&
              (name == 'tokenizer.json' ||
                  name.endsWith('_vocab.json') ||
                  name == 'vocab.json')) {
            vocabPath = f.path;
          }
          if (configPath == null &&
              (name.endsWith('_tokenizer_config.json') ||
                  name == 'tokenizer_config.json')) {
            configPath = f.path;
          }
        }
      } catch (_) {}
    }

    if (!mounted) return;

    if (vocabPath != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Model + tokenizer files found! Loading…'),
          duration: Duration(seconds: 2),
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Tokenizer not found — select .pte + tokenizer.json '
            '(or vocab.json + tokenizer_config.json) together.',
          ),
          duration: Duration(seconds: 4),
        ),
      );
    }

    // Load model + tokenizer via native channel
    await ref.read(modelProvider.notifier).loadModel(
          ptePath,
          vocabPath: vocabPath,
          configPath: configPath,
        );
  }

  @override
  Widget build(BuildContext context) {
    final chat = ref.watch(chatProvider);
    final model = ref.watch(modelProvider);
    final hasModel = model.isLoaded;
    final dbCount = ref.watch(objectBoxProvider).count;

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('Chat'),
        backgroundColor: Colors.white,
        surfaceTintColor: Colors.white,
        foregroundColor: const Color(0xFF1A1A2E),
        elevation: 0,
        actions: [
          // Select Knowledge PDF button — always enabled
          IconButton(
            icon: const Icon(Icons.menu_book_outlined, size: 20),
            onPressed: _selectKnowledge,
            tooltip: 'Select Knowledge',
          ),
          if (chat.messages.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.delete_outline, size: 20),
              onPressed: () =>
                  ref.read(chatProvider.notifier).clearHistory(),
              tooltip: 'Clear chat',
            ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              // ── Model status bar ──────────────────────────────────────
              if (!hasModel && !model.isLoading)
                Container(
                  width: double.infinity,
                  margin: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 8),
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF3E0),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: const Color(0xFFFFCC80)),
                  ),
                  child: Column(
                    children: [
                      const Row(children: [
                        Icon(Icons.info_outline,
                            size: 18, color: Color(0xFFE65100)),
                        SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'No model loaded. Select files together:\n'
                            '.pte model + tokenizer.json',
                            style: TextStyle(
                                fontSize: 13, color: Color(0xFFE65100)),
                          ),
                        ),
                      ]),
                      const SizedBox(height: 10),
                      SizedBox(
                        width: double.infinity,
                        child: OutlinedButton.icon(
                          onPressed: _loadModel,
                          icon: const Icon(Icons.folder_open, size: 18),
                          label: const Text('Load Model + Tokenizer'),
                        ),
                      ),
                    ],
                  ),
                )
              else if (hasModel)
                Container(
                  width: double.infinity,
                  margin: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 8),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFE8F5E9),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(children: [
                    const Icon(Icons.check_circle,
                        size: 16, color: Color(0xFF43A047)),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        '${model.modelPath!.split(Platform.pathSeparator).last}'
                        '${dbCount > 0 ? '  •  $dbCount chunks in DB' : ''}'
                        '${model.isDemoMode ? '  •  Demo mode' : ''}',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                    TextButton(
                      onPressed: _loadModel,
                      child: const Text('Change',
                          style: TextStyle(fontSize: 12)),
                    ),
                  ]),
                ),

              // ── Tokenizer warning ─────────────────────────────────────
              if (model.needsTokenizer)
                Container(
                  width: double.infinity,
                  margin: const EdgeInsets.symmetric(horizontal: 16),
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF8E1),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                        color: const Color(0xFFFFE082), width: 0.5),
                  ),
                  child: const Row(children: [
                    Icon(Icons.warning_amber_rounded,
                        size: 16, color: Color(0xFFF9A825)),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Tokenizer not found. Place tokenizer.json '
                        'in the same folder as your .pte file.',
                        style: TextStyle(
                            fontSize: 11, color: Color(0xFFF57F17)),
                      ),
                    ),
                  ]),
                ),

              // ── Messages ──────────────────────────────────────────────
              Expanded(
                child: chat.messages.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.chat_bubble_outline,
                                size: 48, color: Colors.grey.shade300),
                            const SizedBox(height: 12),
                            Text(
                              hasModel
                                  ? 'Ask anything about your document'
                                  : 'Load a model to start',
                              style: TextStyle(
                                  color: Colors.grey.shade500,
                                  fontSize: 14),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        controller: _scroll,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 14, vertical: 8),
                        itemCount: chat.messages.length,
                        itemBuilder: (_, i) {
                          // Auto-scroll when the last message is being streamed
                          if (i == chat.messages.length - 1 && chat.isInferencing) {
                            _scrollToBottom();
                          }
                          return _Bubble(message: chat.messages[i]);
                        },
                      ),
              ),

              // ── Typing indicator ──────────────────────────────────────
              if (chat.isInferencing)
                Padding(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 4),
                  child: Row(children: [
                    SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.grey.shade400,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text('Generating…',
                        style: TextStyle(
                            fontSize: 12, color: Colors.grey.shade500)),
                  ]),
                ),

              // ── Input bar ─────────────────────────────────────────────
              Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  border:
                      Border(top: BorderSide(color: Colors.grey.shade200)),
                ),
                child: SafeArea(
                  top: false,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 8),
                    child: Row(children: [
                      Expanded(
                        child: TextField(
                          controller: _ctrl,
                          enabled: hasModel && !chat.isInferencing,
                          maxLines: null,
                          textInputAction: TextInputAction.send,
                          onSubmitted: (_) => _send(),
                          decoration: InputDecoration(
                            hintText: hasModel
                                ? 'Ask about your document…'
                                : 'Load a model first',
                            filled: true,
                            fillColor: const Color(0xFFF5F5F5),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(24),
                              borderSide: BorderSide.none,
                            ),
                            isDense: true,
                            contentPadding: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 10),
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        decoration: BoxDecoration(
                          color: hasModel
                              ? const Color(0xFF4F56C7)
                              : Colors.grey.shade300,
                          shape: BoxShape.circle,
                        ),
                        child: IconButton(
                          onPressed: hasModel && !chat.isInferencing
                              ? _send
                              : null,
                          icon: const Icon(Icons.send, size: 18),
                          color: Colors.white,
                        ),
                      ),
                    ]),
                  ),
                ),
              ),
            ],
          ),

          // ── Full-screen loading overlay while model is loading ─────
          if (model.isLoading)
            Container(
              color: Colors.white.withOpacity(0.92),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    FadeTransition(
                      opacity: Tween(begin: 0.3, end: 1.0)
                          .animate(_pulseCtrl),
                      child: Container(
                        padding: const EdgeInsets.all(28),
                        decoration: const BoxDecoration(
                          color: Color(0xFFEEF0FF),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.psychology_outlined,
                          size: 56,
                          color: Color(0xFF4F56C7),
                        ),
                      ),
                    ),
                    const SizedBox(height: 32),
                    const Text(
                      'Loading Model…',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFF1A1A2E),
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      width: 180,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: const LinearProgressIndicator(
                          minHeight: 4,
                          backgroundColor: Color(0xFFEEF0FF),
                          color: Color(0xFF4F56C7),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'This may take 1-2 minutes.\nPlease keep the app open.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey.shade500,
                        height: 1.4,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat bubble
// ─────────────────────────────────────────────────────────────────────────────

class _Bubble extends StatelessWidget {
  final ChatMessage message;
  const _Bubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == 'user';

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 4),
          padding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: isUser
                ? const Color(0xFF4F56C7)
                : const Color(0xFFF5F5F5),
            borderRadius: BorderRadius.only(
              topLeft: const Radius.circular(16),
              topRight: const Radius.circular(16),
              bottomLeft: Radius.circular(isUser ? 16 : 4),
              bottomRight: Radius.circular(isUser ? 4 : 16),
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // No-context warning badge
              if (!isUser && message.noContextWarning) ...[
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 4),
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF8E1),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                        color: const Color(0xFFFFE082), width: 0.5),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.warning_amber_rounded,
                          size: 14, color: Color(0xFFF9A825)),
                      SizedBox(width: 4),
                      Flexible(
                        child: Text(
                          'No relevant context found — answering from model knowledge',
                          style: TextStyle(
                            fontSize: 10,
                            color: Color(0xFFF57F17),
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              Text(
                message.content,
                style: TextStyle(
                  color:
                      isUser ? Colors.white : const Color(0xFF1A1A2E),
                  fontSize: 14,
                ),
              ),
              if (!isUser &&
                  message.ragContext != null &&
                  message.ragContext!.isNotEmpty) ...[
                const SizedBox(height: 6),
                Text(
                  '📚 ${message.ragContext!.length} context chunk(s) used',
                  style: TextStyle(
                    fontSize: 10,
                    color: Colors.grey.shade500,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
