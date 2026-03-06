import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';

import '../providers/app_providers.dart';
import '../services/encryption_service.dart';

class TrainDownloadView extends ConsumerStatefulWidget {
  const TrainDownloadView({super.key});

  @override
  ConsumerState<TrainDownloadView> createState() => _TrainDownloadViewState();
}

class _TrainDownloadViewState extends ConsumerState<TrainDownloadView> {
  final _tokenCtrl = TextEditingController();

  // ── Local state ─────────────────────────────────────────────────────────

  // Token
  bool _tokenSaved = false;
  bool _savingToken = false;

  // PDF
  String? _pdfFileName;
  bool _extracting = false;
  String? _extractedText;
  int _chunkCount = 0;
  List<String> _sampleChunks = [];

  // Encryption
  EncryptionTestResult? _encTest;
  EncryptionService? _encService;

  // Upload
  bool _uploading = false;
  bool _uploaded = false;

  // Pipeline
  bool _pipelineRunning = false;
  bool _pipelineDone = false;

  // Model
  bool _downloading = false;
  double _dlProgress = 0;
  String? _modelPath;

  String? _error;

  @override
  void initState() {
    super.initState();
    // Pre-populate token if already saved.
    ref.listenManual(settingsProvider, (_, next) {
      if (next is AsyncData<String> && _tokenCtrl.text.isEmpty) {
        _tokenCtrl.text = next.value;
        if (next.value.isNotEmpty) setState(() => _tokenSaved = true);
      }
    }, fireImmediately: true);

    WidgetsBinding.instance.addPostFrameCallback((_) {
      final m = ref.read(modelProvider);
      if (m.isLoaded && m.modelPath != null) {
        setState(() => _modelPath = m.modelPath!);
      }
    });
  }

  @override
  void dispose() {
    _tokenCtrl.dispose();
    super.dispose();
  }

  // ── Actions ─────────────────────────────────────────────────────────────

  Future<void> _saveToken() async {
    final t = _tokenCtrl.text.trim();
    if (t.isEmpty) return;

    setState(() {
      _savingToken = true;
      _error = null;
    });
    try {
      await ref.read(settingsProvider.notifier).saveToken(t);
      await ref.read(backendServiceProvider).postHfToken(t);
      setState(() {
        _tokenSaved = true;
        _savingToken = false;
      });
    } catch (e) {
      setState(() {
        _error = '$e';
        _savingToken = false;
      });
    }
  }

  Future<void> _pickPdf() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'txt'],
    );
    if (res == null) return;

    final file = File(res.files.single.path!);

    setState(() {
      _extracting = true;
      _pdfFileName = res.files.single.name;
      _extractedText = null;
      _encTest = null;
      _encService = null;
      _chunkCount = 0;
      _sampleChunks = [];
      _uploaded = false;
      _pipelineDone = false;
      _error = null;
    });

    try {
      // 1. Extract text
      String text;
      if (res.files.single.extension?.toLowerCase() == 'pdf') {
        final doc = PdfDocument(inputBytes: file.readAsBytesSync());
        text = PdfTextExtractor(doc).extractText();
        doc.dispose();
      } else {
        text = await file.readAsString();
      }

      // 2. Chunk & store in ObjectBox (vector DB)
      final chunks = text
          .split(RegExp(r'(?<=[.!?])\s+'))
          .where((c) => c.trim().length > 10)
          .map((c) => c.trim())
          .toList();
      await ref.read(objectBoxProvider).replaceChunks(chunks);

      // 3. Encrypt a snippet for the self-test visual
      final snippet =
          text.length > 300 ? text.substring(0, 300) : text;
      final testResult = EncryptionService.selfTest(snippet);

      // Keep a service instance for the full-content upload
      final svc = EncryptionService();

      setState(() {
        _extracting = false;
        _extractedText = text;
        _chunkCount = ref.read(objectBoxProvider).count;
        _sampleChunks = chunks.take(3).toList();
        _encTest = testResult;
        _encService = svc;
      });
    } catch (e) {
      setState(() {
        _extracting = false;
        _error = 'Extraction failed: $e';
      });
    }
  }

  Future<void> _upload() async {
    if (_encService == null || _extractedText == null) return;

    setState(() {
      _uploading = true;
      _error = null;
    });
    try {
      final toEncrypt = _extractedText!.length > 50000
          ? _extractedText!.substring(0, 50000)
          : _extractedText!;
      final enc = _encService!.encrypt(toEncrypt);

      await ref.read(backendServiceProvider).uploadEncryptedContent(
            encryptedBase64: enc,
            keyBase64: _encService!.keyBase64,
            ivBase64: _encService!.ivBase64,
          );
      setState(() {
        _uploading = false;
        _uploaded = true;
      });
    } catch (e) {
      setState(() {
        _uploading = false;
        _error = '$e';
      });
    }
  }

  Future<void> _runPipeline() async {
    setState(() {
      _pipelineRunning = true;
      _error = null;
    });
    try {
      await ref.read(backendServiceProvider).triggerPipeline();
      setState(() {
        _pipelineRunning = false;
        _pipelineDone = true;
      });
    } catch (e) {
      setState(() {
        _pipelineRunning = false;
        _error = '$e';
      });
    }
  }

  Future<void> _downloadModel() async {
    setState(() {
      _downloading = true;
      _dlProgress = 0;
      _error = null;
    });
    try {
      await ref.read(backendServiceProvider).downloadModel(
            url: 'dummy',
            savePath: 'dummy',
            onProgress: (p) => setState(() => _dlProgress = p),
          );
      setState(() => _downloading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Pipeline download simulated. Use "Load Local .pte" to load a real model.',
            ),
          ),
        );
      }
    } catch (e) {
      setState(() {
        _downloading = false;
        _error = '$e';
      });
    }
  }

  Future<void> _loadLocal() async {
    final res = await FilePicker.platform.pickFiles(
      type: FileType.any,
      dialogTitle: 'Select .pte model',
    );
    if (res == null) return;
    final path = res.files.single.path!;
    if (!path.toLowerCase().endsWith('.pte')) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a .pte file.')),
      );
      return;
    }

    // Auto-detect tokenizer files in same directory
    final dir = File(path).parent;
    String? vocabPath;
    String? configPath;
    try {
      final files = dir.listSync().whereType<File>();
      for (final f in files) {
        final name = f.path.split(Platform.pathSeparator).last.toLowerCase();
        if (name == 'tokenizer.json' ||
            name.endsWith('_vocab.json') ||
            name == 'vocab.json') {
          vocabPath = f.path;
        }
        if (name.endsWith('_tokenizer_config.json') ||
            name == 'tokenizer_config.json') {
          configPath = f.path;
        }
      }
    } catch (_) {}

    // Load Model + tokenizer via native ExecuTorch channel
    ref.read(modelProvider.notifier).loadModel(
          path,
          vocabPath: vocabPath,
          configPath: configPath,
        );
    setState(() => _modelPath = path);
  }

  // ── Build ───────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      appBar: AppBar(
        title: const Text('Train & Download'),
        backgroundColor: Colors.white,
        surfaceTintColor: Colors.white,
        foregroundColor: const Color(0xFF1A1A2E),
        elevation: 0,
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // ── 1. Token ────────────────────────────────────────────────────
          _SectionCard(
            step: 1,
            title: 'Hugging Face Token',
            subtitle: 'Required to access models & backend',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _tokenCtrl,
                  obscureText: true,
                  decoration: InputDecoration(
                    hintText: 'hf_xxxxxxxxxxxxxxxxxx',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    prefixIcon: const Icon(Icons.key_outlined, size: 20),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 12),
                  ),
                ),
                const SizedBox(height: 12),
                _savingToken
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : FilledButton.icon(
                        onPressed: _saveToken,
                        icon: Icon(
                            _tokenSaved ? Icons.check : Icons.save,
                            size: 18),
                        label: Text(_tokenSaved
                            ? 'Saved'
                            : 'Save & Send to Backend'),
                        style: FilledButton.styleFrom(
                          backgroundColor: _tokenSaved
                              ? const Color(0xFF43A047)
                              : null,
                        ),
                      ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── 2. Knowledge Document ───────────────────────────────────────
          _SectionCard(
            step: 2,
            title: 'Upload Knowledge',
            subtitle: 'Select a PDF/TXT for knowledge distillation',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                FilledButton.icon(
                  onPressed: _extracting ? null : _pickPdf,
                  icon: const Icon(Icons.upload_file, size: 18),
                  label: Text(_pdfFileName ?? 'Pick PDF / TXT'),
                ),

                if (_extracting) ...[
                  const SizedBox(height: 16),
                  const Row(children: [
                    SizedBox(
                        width: 16,
                        height: 16,
                        child:
                            CircularProgressIndicator(strokeWidth: 2)),
                    SizedBox(width: 10),
                    Text('Extracting & processing…'),
                  ]),
                ],

                // ── Extraction results ─────────────────────────────────
                if (_extractedText != null) ...[
                  const SizedBox(height: 16),
                  _InfoChip(
                      '${_extractedText!.length} characters extracted'),
                  const SizedBox(height: 6),
                  _InfoChip(
                      '$_chunkCount chunks stored in Vector DB'),

                  // Sample chunks
                  if (_sampleChunks.isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Text('Sample chunks:',
                        style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: Colors.grey.shade700)),
                    const SizedBox(height: 6),
                    ..._sampleChunks.map((c) => Container(
                          margin: const EdgeInsets.only(bottom: 4),
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: const Color(0xFFF5F5F5),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            c.length > 120
                                ? '${c.substring(0, 120)}…'
                                : c,
                            style: const TextStyle(fontSize: 11),
                          ),
                        )),
                  ],

                  // ── Encryption test ──────────────────────────────────
                  if (_encTest != null) ...[
                    const SizedBox(height: 16),
                    _EncryptionTestCard(result: _encTest!),
                  ],

                  // ── Upload button ────────────────────────────────────
                  const SizedBox(height: 16),
                  if (_uploading)
                    const Row(children: [
                      SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(
                              strokeWidth: 2)),
                      SizedBox(width: 10),
                      Text('Uploading encrypted data…'),
                    ])
                  else if (_uploaded)
                    const _InfoChip(
                        '✓ Encrypted content uploaded to backend',
                        color: Color(0xFF43A047))
                  else
                    FilledButton.icon(
                      onPressed: _upload,
                      icon: const Icon(Icons.cloud_upload_outlined,
                          size: 18),
                      label:
                          const Text('Upload Encrypted to Backend'),
                    ),
                ],
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── 3. Pipeline ─────────────────────────────────────────────────
          _SectionCard(
            step: 3,
            title: 'Run Pipeline',
            subtitle: 'Knowledge distillation & quantization',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_pipelineRunning) ...[
                  const LinearProgressIndicator(),
                  const SizedBox(height: 8),
                  const Text('Running pipeline… this may take minutes.'),
                ] else if (_pipelineDone)
                  const _InfoChip(
                      '✓ Pipeline complete — model ready',
                      color: Color(0xFF43A047))
                else
                  FilledButton.icon(
                    onPressed: _uploaded ? _runPipeline : null,
                    icon: const Icon(Icons.play_arrow, size: 18),
                    label: const Text('Start Pipeline'),
                  ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── 4. Model ────────────────────────────────────────────────────
          _SectionCard(
            step: 4,
            title: 'Download & Load Model',
            subtitle: 'Get your .pte model onto the device',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_downloading) ...[
                  LinearProgressIndicator(value: _dlProgress),
                  const SizedBox(height: 8),
                  Text('${(_dlProgress * 100).toStringAsFixed(0)}%'),
                ] else ...[
                  FilledButton.icon(
                    onPressed:
                        _pipelineDone ? _downloadModel : null,
                    icon: const Icon(Icons.download, size: 18),
                    label: const Text('Download from Pipeline'),
                  ),
                  const SizedBox(height: 8),
                  Row(children: [
                    Expanded(
                        child: Divider(color: Colors.grey.shade300)),
                    Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12),
                      child: Text('or',
                          style: TextStyle(
                              color: Colors.grey.shade500,
                              fontSize: 12)),
                    ),
                    Expanded(
                        child: Divider(color: Colors.grey.shade300)),
                  ]),
                  const SizedBox(height: 8),
                  OutlinedButton.icon(
                    onPressed: _loadLocal,
                    icon: const Icon(Icons.folder_open, size: 18),
                    label: const Text('Load Local .pte File'),
                  ),
                ],
                if (_modelPath != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0xFFE8F5E9),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Row(children: [
                      const Icon(Icons.check_circle,
                          color: Color(0xFF43A047), size: 18),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _modelPath!
                              .split(Platform.pathSeparator)
                              .last,
                          style: const TextStyle(
                              fontWeight: FontWeight.w500,
                              fontSize: 13),
                        ),
                      ),
                    ]),
                  ),
                ],
              ],
            ),
          ),

          // ── Error ───────────────────────────────────────────────────────
          if (_error != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFFFFEBEE),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(children: [
                const Icon(Icons.error_outline,
                    color: Colors.red, size: 18),
                const SizedBox(width: 8),
                Expanded(
                    child: Text(_error!,
                        style: const TextStyle(fontSize: 13))),
              ]),
            ),
          ],

          const SizedBox(height: 32),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper widgets
// ─────────────────────────────────────────────────────────────────────────────

class _SectionCard extends StatelessWidget {
  final int step;
  final String title;
  final String subtitle;
  final Widget child;

  const _SectionCard({
    required this.step,
    required this.title,
    required this.subtitle,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 0,
      color: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: BorderSide(color: Colors.grey.shade200),
      ),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Container(
                width: 28,
                height: 28,
                decoration: const BoxDecoration(
                  color: Color(0xFFEEF0FF),
                  shape: BoxShape.circle,
                ),
                alignment: Alignment.center,
                child: Text(
                  '$step',
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                    color: Color(0xFF4F56C7),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: const TextStyle(
                            fontWeight: FontWeight.w600, fontSize: 15)),
                    Text(subtitle,
                        style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey.shade500)),
                  ],
                ),
              ),
            ]),
            const SizedBox(height: 14),
            child,
          ],
        ),
      ),
    );
  }
}

class _InfoChip extends StatelessWidget {
  final String text;
  final Color? color;
  const _InfoChip(this.text, {this.color});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Icon(Icons.check_circle,
          size: 15, color: color ?? const Color(0xFF4F56C7)),
      const SizedBox(width: 6),
      Expanded(
        child: Text(text,
            style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w500,
                color: color ?? Colors.grey.shade700)),
      ),
    ]);
  }
}

class _EncryptionTestCard extends StatelessWidget {
  final EncryptionTestResult result;
  const _EncryptionTestCard({required this.result});

  String _trunc(String s, int max) =>
      s.length > max ? '${s.substring(0, max)}…' : s;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color:
            result.passed ? const Color(0xFFE8F5E9) : const Color(0xFFFFEBEE),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: result.passed
              ? const Color(0xFF81C784)
              : const Color(0xFFEF9A9A),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(
              result.passed ? Icons.lock_outlined : Icons.lock_open,
              size: 18,
              color: result.passed
                  ? const Color(0xFF43A047)
                  : Colors.red,
            ),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                'AES-256-CBC Encryption Test',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                  color: result.passed
                      ? const Color(0xFF2E7D32)
                      : Colors.red.shade700,
                ),
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(
                  horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: result.passed
                    ? const Color(0xFF43A047)
                    : Colors.red,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                result.passed ? 'PASS' : 'FAIL',
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w600),
              ),
            ),
          ]),
          const SizedBox(height: 10),
          _EncRow('Original', _trunc(result.original, 80)),
          _EncRow('Encrypted', _trunc(result.encrypted, 80)),
          _EncRow('Decrypted', _trunc(result.decrypted, 80)),
          const Divider(height: 16),
          _EncRow('Key', _trunc(result.keyBase64, 44)),
          _EncRow('IV', _trunc(result.ivBase64, 24)),
        ],
      ),
    );
  }
}

class _EncRow extends StatelessWidget {
  final String label;
  final String value;
  const _EncRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 70,
            child: Text(
              label,
              style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: Colors.grey.shade700),
            ),
          ),
          Expanded(
            child: Text(value,
                style: const TextStyle(
                    fontSize: 11, fontFamily: 'monospace')),
          ),
        ],
      ),
    );
  }
}
