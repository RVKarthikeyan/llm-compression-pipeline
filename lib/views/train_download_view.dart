import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:syncfusion_flutter_pdf/pdf.dart';

import '../providers/app_providers.dart';
import '../services/backend_service.dart';

class TrainDownloadView extends ConsumerStatefulWidget {
  const TrainDownloadView({super.key});

  @override
  ConsumerState<TrainDownloadView> createState() => _TrainDownloadViewState();
}

class _TrainDownloadViewState extends ConsumerState<TrainDownloadView> {
  final _tokenCtrl = TextEditingController();
  final _wandbCtrl = TextEditingController();
  final _urlCtrl = TextEditingController();

  // ── Local state ─────────────────────────────────────────────────────────

  // Auth
  bool _authenticating = false;
  bool _authenticated = false;
  String? _username;

  // PDF
  bool _pickingPdf = false;
  String? _pdfFileName;
  File? _pdfFile;
  int _chunkCount = 0;

  // Pipeline
  bool _triggeringPipeline = false;
  String? _jobId;
  bool _polling = false;
  Timer? _pollTimer;
  PipelineStatusResponse? _lastStatus;

  // Download
  bool _downloading = false;
  double _downloadProgress = 0;

  // Model
  String? _modelPath;

  String? _error;

  @override
  void initState() {
    super.initState();
    // Pre-populate tokens if already saved.
    ref.listenManual(settingsProvider, (_, next) {
      if (next is AsyncData<String> && _tokenCtrl.text.isEmpty) {
        _tokenCtrl.text = next.value;
      }
    }, fireImmediately: true);

    _loadSavedKeys();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      final m = ref.read(modelProvider);
      if (m.isLoaded && m.modelPath != null) {
        setState(() => _modelPath = m.modelPath!);
      }
    });
  }

  Future<void> _loadSavedKeys() async {
    final wandb = await readWandbKey();
    if (wandb.isNotEmpty && _wandbCtrl.text.isEmpty) {
      _wandbCtrl.text = wandb;
    }
    final url = await readBackendUrl();
    if (url.isNotEmpty && _urlCtrl.text.isEmpty) {
      _urlCtrl.text = url;
    } else if (_urlCtrl.text.isEmpty) {
      _urlCtrl.text = 'http://10.0.2.2:8000';
    }
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _tokenCtrl.dispose();
    _wandbCtrl.dispose();
    _urlCtrl.dispose();
    super.dispose();
  }

  // ── Actions ─────────────────────────────────────────────────────────────

  Future<void> _authenticate() async {
    final hfToken = _tokenCtrl.text.trim();
    final wandbKey = _wandbCtrl.text.trim();
    final backendUrl = _urlCtrl.text.trim();

    if (hfToken.isEmpty || wandbKey.isEmpty) {
      setState(() => _error = 'Both HF token and W&B API key are required.');
      return;
    }

    setState(() {
      _authenticating = true;
      _error = null;
    });

    try {
      // Save credentials locally
      await ref.read(settingsProvider.notifier).saveToken(hfToken);
      await saveWandbKey(wandbKey);
      if (backendUrl.isNotEmpty) {
        await saveBackendUrl(backendUrl);
      }

      final backend = ref.read(backendServiceProvider);
      if (backendUrl.isNotEmpty) {
        backend.setBaseUrl(backendUrl);
      }

      final authResp = await backend.authenticate(
        hfToken: hfToken,
        wandbApiKey: wandbKey,
      );

      setState(() {
        _authenticating = false;
        _authenticated = true;
        _username = authResp.username;
      });
    } catch (e) {
      setState(() {
        _authenticating = false;
        _error = 'Authentication failed: $e';
      });
    }
  }

  Future<void> _pickPdf() async {
    if (_pickingPdf) return;
    setState(() => _pickingPdf = true);

    try {
      final res = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
      );
      if (res == null) {
        setState(() => _pickingPdf = false);
        return;
      }

      final file = File(res.files.single.path!);
      setState(() {
        _pdfFile = file;
        _pdfFileName = res.files.single.name;
        _error = null;
      });

      // Extract text and store chunks in ObjectBox for RAG
      try {
        final doc = PdfDocument(inputBytes: file.readAsBytesSync());
        final text = PdfTextExtractor(doc).extractText();
        doc.dispose();

        debugPrint('[RAG] Extracted ${text.length} chars from PDF');
        debugPrint(
          '[RAG] First 500 chars: ${text.substring(0, text.length < 500 ? text.length : 500)}',
        );

        // Use smart chunking that preserves patient sections and logical blocks
        final obs = ref.read(objectBoxProvider);
        await obs.replaceChunksFromText(text);

        final storedCount = obs.count;
        debugPrint('[RAG] Stored $storedCount chunks in ObjectBox');

        setState(() => _chunkCount = storedCount);
      } catch (e) {
        debugPrint('[RAG] PDF chunking FAILED: $e');
      }
    } finally {
      setState(() => _pickingPdf = false);
    }
  }

  Future<void> _triggerPipeline() async {
    if (_pdfFile == null) return;
    final hfToken = _tokenCtrl.text.trim();

    setState(() {
      _triggeringPipeline = true;
      _error = null;
    });

    try {
      final backend = ref.read(backendServiceProvider);
      final resp = await backend.triggerPipeline(
        hfToken: hfToken,
        pdfFile: _pdfFile!,
      );

      setState(() {
        _triggeringPipeline = false;
        _jobId = resp.jobId;
      });

      // Start polling
      _startPolling();
    } catch (e) {
      setState(() {
        _triggeringPipeline = false;
        _error = 'Failed to trigger pipeline: $e';
      });
    }
  }

  void _startPolling() {
    _polling = true;
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 5), (_) async {
      await _pollStatus();
    });
    // Also poll immediately
    _pollStatus();
  }

  Future<void> _pollStatus() async {
    final hfToken = _tokenCtrl.text.trim();
    try {
      final backend = ref.read(backendServiceProvider);
      final status = await backend.getStatus(hfToken: hfToken);

      if (!mounted) return;
      setState(() => _lastStatus = status);

      if (status.isCompleted || status.isFailed) {
        _pollTimer?.cancel();
        setState(() => _polling = false);
      }
    } catch (e) {
      // Silently continue polling on transient errors
    }
  }

  Future<void> _downloadFromHub() async {
    if (_username == null || _jobId == null) return;
    final hfToken = _tokenCtrl.text.trim();

    // Ask user to pick a save directory
    final selectedDir = await FilePicker.platform.getDirectoryPath(
      dialogTitle: 'Choose where to save the model',
    );
    if (selectedDir == null) return; // User cancelled

    setState(() {
      _downloading = true;
      _downloadProgress = 0;
      _error = null;
    });

    try {
      final backend = ref.read(backendServiceProvider);
      final saveDir = selectedDir;

      final file = await backend.downloadModelFromHub(
        hfToken: hfToken,
        username: _username!,
        jobId: _jobId!,
        saveDir: saveDir,
        onProgress: (p) {
          if (mounted) setState(() => _downloadProgress = p);
        },
      );

      if (!mounted) return;

      // Auto-detect tokenizer files next to the downloaded .pte
      final dir = file.parent;
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

      ref
          .read(modelProvider.notifier)
          .loadModel(file.path, vocabPath: vocabPath, configPath: configPath);

      setState(() {
        _downloading = false;
        _modelPath = file.path;
      });
    } catch (e) {
      setState(() {
        _downloading = false;
        _error = 'Download failed: $e';
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

    ref
        .read(modelProvider.notifier)
        .loadModel(path, vocabPath: vocabPath, configPath: configPath);
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
          // ── 1. Authentication ───────────────────────────────────────────
          _SectionCard(
            step: 1,
            title: 'Authenticate',
            subtitle: 'HuggingFace token, W&B key & backend URL',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TextField(
                  controller: _urlCtrl,
                  decoration: InputDecoration(
                    hintText: 'http://10.0.2.2:8000',
                    labelText: 'Backend URL',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    prefixIcon: const Icon(Icons.dns_outlined, size: 20),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 12,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _tokenCtrl,
                  obscureText: true,
                  decoration: InputDecoration(
                    hintText: 'hf_xxxxxxxxxxxxxxxxxx',
                    labelText: 'HuggingFace Token',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    prefixIcon: const Icon(Icons.key_outlined, size: 20),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 12,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _wandbCtrl,
                  obscureText: true,
                  decoration: InputDecoration(
                    hintText: 'W&B API key',
                    labelText: 'Weights & Biases API Key',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    prefixIcon: const Icon(Icons.analytics_outlined, size: 20),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 12,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                if (_authenticating)
                  const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else if (_authenticated)
                  Row(
                    children: [
                      const Icon(
                        Icons.check_circle,
                        color: Color(0xFF43A047),
                        size: 18,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Authenticated as $_username',
                        style: const TextStyle(
                          fontWeight: FontWeight.w500,
                          fontSize: 13,
                        ),
                      ),
                    ],
                  )
                else
                  FilledButton.icon(
                    onPressed: _authenticate,
                    icon: const Icon(Icons.login, size: 18),
                    label: const Text('Authenticate'),
                  ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── 2. Select PDF ───────────────────────────────────────────────
          _SectionCard(
            step: 2,
            title: 'Select Document',
            subtitle: 'PDF for knowledge distillation',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                FilledButton.icon(
                  onPressed: _pickingPdf ? null : _pickPdf,
                  icon: _pickingPdf
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.upload_file, size: 18),
                  label: Text(
                    _pickingPdf ? 'Processing…' : (_pdfFileName ?? 'Pick PDF'),
                  ),
                ),
                if (_pdfFileName != null) ...[
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      const Icon(
                        Icons.description,
                        size: 16,
                        color: Color(0xFF4F56C7),
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          _pdfFileName!,
                          style: const TextStyle(fontSize: 12),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                  if (_chunkCount > 0) ...[
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        const Icon(
                          Icons.storage,
                          size: 14,
                          color: Color(0xFF43A047),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '$_chunkCount chunks stored for RAG',
                          style: const TextStyle(
                            fontSize: 11,
                            color: Color(0xFF43A047),
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── 3. Trigger Pipeline ─────────────────────────────────────────
          _SectionCard(
            step: 3,
            title: 'Run Pipeline',
            subtitle:
                'Pruning → Knowledge Distillation → Quantization → Upload',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_triggeringPipeline) ...[
                  const Row(
                    children: [
                      SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                      SizedBox(width: 10),
                      Text('Submitting pipeline job…'),
                    ],
                  ),
                ] else if (_jobId != null) ...[
                  Row(
                    children: [
                      const Icon(
                        Icons.check_circle,
                        color: Color(0xFF43A047),
                        size: 16,
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          'Job: ${_jobId!.substring(0, 8)}…',
                          style: const TextStyle(
                            fontSize: 12,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ),
                    ],
                  ),

                  // ── Status display ────────────────────────────────────
                  if (_lastStatus != null) ...[
                    const SizedBox(height: 12),
                    _PipelineProgressCard(status: _lastStatus!),
                  ],

                  if (_polling) ...[
                    const SizedBox(height: 8),
                    const Row(
                      children: [
                        SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                        SizedBox(width: 8),
                        Text(
                          'Polling for updates…',
                          style: TextStyle(fontSize: 12, color: Colors.grey),
                        ),
                      ],
                    ),
                  ],

                  // Manual refresh button
                  const SizedBox(height: 8),
                  OutlinedButton.icon(
                    onPressed: _pollStatus,
                    icon: const Icon(Icons.refresh, size: 16),
                    label: const Text('Refresh Status'),
                  ),
                ] else
                  FilledButton.icon(
                    onPressed: (_authenticated && _pdfFile != null)
                        ? _triggerPipeline
                        : null,
                    icon: const Icon(Icons.play_arrow, size: 18),
                    label: const Text('Start Pipeline'),
                  ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // ── 4. Download & Load Model ────────────────────────────────────
          _SectionCard(
            step: 4,
            title: 'Download & Load Model',
            subtitle: 'Download .pte from HuggingFace or load a local file',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Download from HuggingFace
                if (_downloading) ...[
                  LinearProgressIndicator(value: _downloadProgress),
                  const SizedBox(height: 8),
                  Text(
                    '${(_downloadProgress * 100).toStringAsFixed(1)}% downloaded',
                    style: const TextStyle(fontSize: 12),
                  ),
                ] else ...[
                  FilledButton.icon(
                    onPressed:
                        (_lastStatus != null &&
                            _lastStatus!.isCompleted &&
                            _username != null &&
                            _jobId != null)
                        ? _downloadFromHub
                        : null,
                    icon: const Icon(Icons.download, size: 18),
                    label: const Text('Download from HuggingFace'),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(child: Divider(color: Colors.grey.shade300)),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        child: Text(
                          'or',
                          style: TextStyle(
                            color: Colors.grey.shade500,
                            fontSize: 12,
                          ),
                        ),
                      ),
                      Expanded(child: Divider(color: Colors.grey.shade300)),
                    ],
                  ),
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
                    child: Row(
                      children: [
                        const Icon(
                          Icons.check_circle,
                          color: Color(0xFF43A047),
                          size: 18,
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            _modelPath!.split(Platform.pathSeparator).last,
                            style: const TextStyle(
                              fontWeight: FontWeight.w500,
                              fontSize: 13,
                            ),
                          ),
                        ),
                      ],
                    ),
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
              child: Row(
                children: [
                  const Icon(Icons.error_outline, color: Colors.red, size: 18),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(_error!, style: const TextStyle(fontSize: 13)),
                  ),
                ],
              ),
            ),
          ],

          const SizedBox(height: 32),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline progress card — shows the current step with visual indicator
// ─────────────────────────────────────────────────────────────────────────────

class _PipelineProgressCard extends StatelessWidget {
  final PipelineStatusResponse status;
  const _PipelineProgressCard({required this.status});

  @override
  Widget build(BuildContext context) {
    final steps = [
      'queued',
      'pruning',
      'distilling',
      'quantizing',
      'uploading',
      'completed',
    ];
    final currentIdx = steps.indexOf(status.status);

    Color statusColor;
    IconData statusIcon;
    if (status.isCompleted) {
      statusColor = const Color(0xFF43A047);
      statusIcon = Icons.check_circle;
    } else if (status.isFailed) {
      statusColor = Colors.red;
      statusIcon = Icons.error;
    } else {
      statusColor = const Color(0xFF4F56C7);
      statusIcon = Icons.hourglass_top;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: status.isFailed
            ? const Color(0xFFFFEBEE)
            : status.isCompleted
            ? const Color(0xFFE8F5E9)
            : const Color(0xFFEEF0FF),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: statusColor.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(statusIcon, size: 20, color: statusColor),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  status.status.toUpperCase(),
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                    color: statusColor,
                  ),
                ),
              ),
              if (status.pteReady == true)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFF43A047),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Text(
                    'PTE READY',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          Text(status.message, style: const TextStyle(fontSize: 12)),
          if (status.isRunning && currentIdx >= 0) ...[
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: (currentIdx + 1) / steps.length,
                minHeight: 6,
                backgroundColor: Colors.grey.shade200,
                valueColor: AlwaysStoppedAnimation<Color>(statusColor),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Step ${currentIdx + 1} of ${steps.length}',
              style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
            ),
          ],
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
            Row(
              children: [
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
                      Text(
                        title,
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 15,
                        ),
                      ),
                      Text(
                        subtitle,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade500,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            child,
          ],
        ),
      ),
    );
  }
}
