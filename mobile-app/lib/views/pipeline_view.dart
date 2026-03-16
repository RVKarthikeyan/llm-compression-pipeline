import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_pdf/pdf.dart';

import '../providers/app_providers.dart';

class PipelineView extends ConsumerWidget {
  const PipelineView({super.key});

  // ── Load a local .pte file directly ─────────────────────────────────────

  Future<void> _loadLocalPteFile(BuildContext context, WidgetRef ref) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any, // .pte isn't a standard MIME type
      dialogTitle: 'Select your .pte model file',
    );
    if (result == null) return;

    final path = result.files.single.path!;
    if (!path.toLowerCase().endsWith('.pte')) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select a .pte ExecuTorch model file.'),
        ),
      );
      return;
    }

    await ref.read(pipelineProvider.notifier).loadLocalModel(path);
  }

  // ── PDF selection: extract → store → upload ──────────────────────────────

  Future<void> _selectAndProcessPdf(BuildContext context, WidgetRef ref) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'txt'],
    );
    if (result == null) return;

    final filePath = result.files.single.path!;
    final file = File(filePath);

    // 1. Extract text
    String content = '';
    if (result.files.single.extension == 'pdf') {
      final doc = PdfDocument(inputBytes: file.readAsBytesSync());
      content = PdfTextExtractor(doc).extractText();
      doc.dispose();
    } else {
      content = await file.readAsString();
    }

    // 2. Chunk & store in ObjectBox using smart chunking
    await ref.read(objectBoxProvider).replaceChunksFromText(content);

    // Update pipeline state: mark PDF selected
    ref.read(pipelineProvider.notifier).setPdfSelected(filePath);

    // 3. Upload PDF to backend (mock)
    final uploaded =
        await ref.read(backendServiceProvider).uploadPdf(file);
    if (!uploaded) {
      ref.read(pipelineProvider.notifier).setError('PDF upload failed.');
      return;
    }
    ref.read(pipelineProvider.notifier).setUploadComplete();

    // 4. Poll backend (mock delay)
    final ready = await ref.read(backendServiceProvider).pollModelReady();
    if (ready) {
      ref.read(pipelineProvider.notifier).setBackendReady();
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final pipeline = ref.watch(pipelineProvider);
    final status = pipeline.status;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Pipeline', style: Theme.of(context).textTheme.headlineSmall),
          const SizedBox(height: 8),
          Text(
            'Select a document, let the backend compress the LLM, '
            'then download and load your custom model.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 32),

          // ── Local model shortcut ────────────────────────────────────────
          Card(
            color: Theme.of(context).colorScheme.primaryContainer,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.folder_open_outlined,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Load Local .pte Model',
                        style: Theme.of(context)
                            .textTheme
                            .titleSmall
                            ?.copyWith(
                              color: Theme.of(context).colorScheme.primary,
                            ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Already have a .pte file? Skip the pipeline and load it directly.',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                  const SizedBox(height: 12),
                  if (status == PipelineStatus.loadingModel)
                    const Row(
                      children: [
                        SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                        SizedBox(width: 10),
                        Text('Loading into memory…'),
                      ],
                    )
                  else if (status == PipelineStatus.modelLoaded &&
                      pipeline.downloadedModelPath != null)
                    Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            '✓ ${pipeline.downloadedModelPath!.split(Platform.pathSeparator).last}',
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        TextButton(
                          onPressed: () =>
                              _loadLocalPteFile(context, ref),
                          child: const Text('Change'),
                        ),
                      ],
                    )
                  else
                    FilledButton.icon(
                      onPressed: () => _loadLocalPteFile(context, ref),
                      icon: const Icon(Icons.drive_folder_upload_outlined),
                      label: const Text('Pick .pte File'),
                    ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          Divider(color: Theme.of(context).colorScheme.outlineVariant),
          const SizedBox(height: 8),
          Text(
            'Or use the full pipeline:',
            style: Theme.of(context).textTheme.labelMedium,
          ),
          const SizedBox(height: 16),

          // ── Step 1: Select PDF ──────────────────────────────────────────
          _StepCard(
            stepNumber: 1,
            title: 'Select Document',
            subtitle: pipeline.selectedPdfPath != null
                ? '📄 ${pipeline.selectedPdfPath!.split(Platform.pathSeparator).last}'
                : 'No document selected',
            child: FilledButton.icon(
              onPressed: (status == PipelineStatus.idle ||
                      status == PipelineStatus.modelLoaded)
                  ? () => _selectAndProcessPdf(context, ref)
                  : null,
              icon: const Icon(Icons.upload_file_outlined),
              label: const Text('Select PDF / TXT'),
            ),
          ),

          const SizedBox(height: 16),

          // ── Step 2: Backend processing indicator ────────────────────────
          if (status == PipelineStatus.uploadingPdf ||
              status == PipelineStatus.waitingBackend)
            _StepCard(
              stepNumber: 2,
              title: status == PipelineStatus.uploadingPdf
                  ? 'Uploading Document…'
                  : 'Distilling & Quantizing…',
              subtitle: 'This may take a few minutes.',
              child: const LinearProgressIndicator(),
            ),

          // ── Step 3: Download model ──────────────────────────────────────
          if (status == PipelineStatus.readyToDownload ||
              status == PipelineStatus.downloading ||
              status == PipelineStatus.downloaded ||
              status == PipelineStatus.loadingModel ||
              status == PipelineStatus.modelLoaded)
            _StepCard(
              stepNumber: 2,
              title: 'Download Custom Model',
              subtitle: '.pte file from Hugging Face',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (status == PipelineStatus.downloading) ...[
                    const SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: pipeline.downloadProgress,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${(pipeline.downloadProgress * 100).toStringAsFixed(1)} %',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ] else if (status == PipelineStatus.readyToDownload)
                    FilledButton.icon(
                      onPressed: () =>
                          ref.read(pipelineProvider.notifier).downloadModel(),
                      icon: const Icon(Icons.download_outlined),
                      label: const Text('Download Custom Model (.pte)'),
                    )
                  else
                    Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green),
                        const SizedBox(width: 8),
                        const Text('Model downloaded'),
                      ],
                    ),
                ],
              ),
            ),

          const SizedBox(height: 16),

          // ── Step 4: Load model into ExecuTorch ──────────────────────────
          if (status == PipelineStatus.downloaded ||
              status == PipelineStatus.loadingModel ||
              status == PipelineStatus.modelLoaded)
            _StepCard(
              stepNumber: 3,
              title: 'Load Model',
              subtitle: 'Initialise the ExecuTorch runtime',
              child: status == PipelineStatus.loadingModel
                  ? const SizedBox(
                      height: 36,
                      child: Row(
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(width: 12),
                          Text('Loading into memory…'),
                        ],
                      ),
                    )
                  : status == PipelineStatus.modelLoaded
                      ? Row(
                          children: const [
                            Icon(Icons.memory, color: Colors.green),
                            SizedBox(width: 8),
                            Text('Model ready — head to Chat!'),
                          ],
                        )
                      : FilledButton.icon(
                          onPressed: () =>
                              ref.read(pipelineProvider.notifier).loadModel(),
                          icon: const Icon(Icons.memory_outlined),
                          label: const Text('Load Model to Memory'),
                        ),
            ),

          // ── Error banner ────────────────────────────────────────────────
          if (pipeline.errorMessage != null &&
              pipeline.errorMessage!.isNotEmpty &&
              status != PipelineStatus.modelLoaded) ...[
            const SizedBox(height: 16),
            Card(
              color: Theme.of(context).colorScheme.errorContainer,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline),
                    const SizedBox(width: 8),
                    Expanded(child: Text(pipeline.errorMessage!)),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared helper widget
// ─────────────────────────────────────────────────────────────────────────────

class _StepCard extends StatelessWidget {
  final int stepNumber;
  final String title;
  final String subtitle;
  final Widget child;

  const _StepCard({
    required this.stepNumber,
    required this.title,
    required this.subtitle,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 12,
                  child: Text(
                    '$stepNumber',
                    style: const TextStyle(fontSize: 12),
                  ),
                ),
                const SizedBox(width: 10),
                Text(title,
                    style: Theme.of(context).textTheme.titleSmall),
              ],
            ),
            const SizedBox(height: 4),
            Padding(
              padding: const EdgeInsets.only(left: 34),
              child: Text(subtitle,
                  style: Theme.of(context).textTheme.bodySmall),
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.only(left: 34),
              child: child,
            ),
          ],
        ),
      ),
    );
  }
}
