import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'providers/app_providers.dart';
import 'services/objectbox_service.dart';
import 'views/chat_view.dart';
import 'views/pipeline_view.dart';
import 'views/settings_view.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialise ObjectBox before the widget tree is built.
  final objectBox = await ObjectBoxService.create();

  runApp(
    ProviderScope(
      overrides: [
        // Inject the initialised ObjectBoxService into the provider graph.
        objectBoxProvider.overrideWithValue(objectBox),
      ],
      child: const MyAiApp(),
    ),
  );
}

class MyAiApp extends StatelessWidget {
  const MyAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Local AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      home: const _Shell(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bottom-navigation shell
// ─────────────────────────────────────────────────────────────────────────────

class _Shell extends ConsumerStatefulWidget {
  const _Shell();

  @override
  ConsumerState<_Shell> createState() => _ShellState();
}

class _ShellState extends ConsumerState<_Shell> {
  int _selectedIndex = 1; // Start on the Pipeline tab

  static const _pages = <Widget>[
    SettingsView(),
    PipelineView(),
    ChatView(),
  ];

  @override
  Widget build(BuildContext context) {
    final pipeline = ref.watch(pipelineProvider);

    return Scaffold(
      appBar: AppBar(
        title: Text(
          pipeline.isModelLoaded
              ? 'Local AI  •  Model Ready'
              : 'Local AI  •  Setup Required',
          style: const TextStyle(fontSize: 15),
        ),
        actions: [
          if (pipeline.isModelLoaded)
            const Padding(
              padding: EdgeInsets.only(right: 12),
              child: Icon(Icons.check_circle, color: Colors.green),
            ),
        ],
      ),
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
          NavigationDestination(
            icon: Icon(Icons.hub_outlined),
            selectedIcon: Icon(Icons.hub),
            label: 'Pipeline',
          ),
          NavigationDestination(
            icon: Icon(Icons.chat_outlined),
            selectedIcon: Icon(Icons.chat),
            label: 'Chat',
          ),
        ],
      ),
    );
  }
}
