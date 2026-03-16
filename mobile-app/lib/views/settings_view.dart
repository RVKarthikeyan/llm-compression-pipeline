import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/app_providers.dart';

class SettingsView extends ConsumerStatefulWidget {
  const SettingsView({super.key});

  @override
  ConsumerState<SettingsView> createState() => _SettingsViewState();
}

class _SettingsViewState extends ConsumerState<SettingsView> {
  final _formKey = GlobalKey<FormState>();
  final _tokenController = TextEditingController();
  final _wandbController = TextEditingController();
  final _urlController = TextEditingController();
  bool _saved = false;

  @override
  void initState() {
    super.initState();
    // Pre-populate with the stored token once loaded.
    ref.listenManual(settingsProvider, (_, next) {
      if (next is AsyncData<String> && _tokenController.text.isEmpty) {
        _tokenController.text = next.value;
      }
    }, fireImmediately: true);
    _loadKeys();
  }

  Future<void> _loadKeys() async {
    final wandb = await readWandbKey();
    if (wandb.isNotEmpty && _wandbController.text.isEmpty) {
      _wandbController.text = wandb;
    }
    final url = await readBackendUrl();
    if (url.isNotEmpty && _urlController.text.isEmpty) {
      _urlController.text = url;
    }
  }

  @override
  void dispose() {
    _tokenController.dispose();
    _wandbController.dispose();
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    await ref
        .read(settingsProvider.notifier)
        .saveToken(_tokenController.text.trim());
    await saveWandbKey(_wandbController.text.trim());
    final url = _urlController.text.trim();
    if (url.isNotEmpty) {
      await saveBackendUrl(url);
      ref.read(backendServiceProvider).setBaseUrl(url);
    }
    setState(() => _saved = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _saved = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    final tokenState = ref.watch(settingsProvider);

    return Padding(
      padding: const EdgeInsets.all(24),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Settings',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              'Your credentials are stored securely on-device.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 32),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Hugging Face Access Token',
                      style: Theme.of(context).textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    TextFormField(
                      controller: _tokenController,
                      obscureText: true,
                      decoration: const InputDecoration(
                        hintText: 'hf_xxxxxxxxxxxxxxxxxx',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.key_outlined),
                      ),
                      validator: (v) =>
                          (v == null || v.trim().isEmpty)
                              ? 'Token is required'
                              : null,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Weights & Biases API Key',
                      style: Theme.of(context).textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    TextFormField(
                      controller: _wandbController,
                      obscureText: true,
                      decoration: const InputDecoration(
                        hintText: 'W&B API key',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.analytics_outlined),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Backend URL',
                      style: Theme.of(context).textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    TextFormField(
                      controller: _urlController,
                      decoration: const InputDecoration(
                        hintText: 'http://10.0.2.2:8000',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.dns_outlined),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            tokenState.isLoading
                ? const CircularProgressIndicator()
                : FilledButton.icon(
                    onPressed: _save,
                    icon: Icon(_saved ? Icons.check : Icons.save_outlined),
                    label: Text(_saved ? 'Saved!' : 'Save Token'),
                  ),
          ],
        ),
      ),
    );
  }
}
