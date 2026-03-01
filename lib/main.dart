import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:file_picker/file_picker.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;
import 'package:syncfusion_flutter_pdf/pdf.dart';

void main() => runApp(
  const MaterialApp(home: RAGChatApp(), debugShowCheckedModeBanner: false),
);

class RAGChatApp extends StatefulWidget {
  const RAGChatApp({super.key});
  @override
  State<RAGChatApp> createState() => _RAGChatAppState();
}

class _RAGChatAppState extends State<RAGChatApp> {
  static const platform = MethodChannel('com.example.my_ai/executorch');
  Database? _db;
  String? _modelPath;
  String _status = "Setup Required";
  final List<Map<String, String>> _messages = [];
  final TextEditingController _controller = TextEditingController();

  @override
  void initState() {
    super.initState();
    _initDb();
  }

  Future<void> _initDb() async {
    _db = await openDatabase(
      p.join(await getDatabasesPath(), 'ai_kv.db'),
      version: 1,
      onCreate: (db, v) => db.execute('CREATE TABLE chunks(text TEXT)'),
    );
  }

  // 1. SELECT MODEL
  Future<void> _pickModel() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null) {
      _modelPath = result.files.single.path;
      final msg = await platform.invokeMethod('loadModel', {
        'path': _modelPath,
      });
      setState(() => _status = msg);
    }
  }

  // 2. PROCESS TXT/PDF
  Future<void> _processKnowledgeBase() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['txt', 'pdf'],
    );
    if (result != null) {
      setState(() => _status = "Processing Data...");
      final file = File(result.files.single.path!);
      String content = "";

      if (result.files.single.extension == 'pdf') {
        final PdfDocument doc = PdfDocument(inputBytes: file.readAsBytesSync());
        content = PdfTextExtractor(doc).extractText();
        doc.dispose();
      } else {
        content = await file.readAsString();
      }

      await _db?.delete('chunks');
      final chunks = content.split(
        RegExp(r'(?<=[.!?])\s+'),
      ); // Split by sentences
      for (var chunk in chunks) {
        if (chunk.trim().length > 10)
          await _db?.insert('chunks', {'text': chunk.trim()});
      }
      setState(() => _status = "Knowledge Base Ready");
    }
  }

  // 3. KEYWORD SEARCH
  Future<String> _searchContext(String query) async {
    final keywords = query.split(' ').where((w) => w.length > 3).toList();
    if (keywords.isEmpty) return "";
    String where = keywords.map((_) => "text LIKE ?").join(' OR ');
    final res = await _db?.query(
      'chunks',
      where: where,
      whereArgs: keywords.map((k) => '%$k%').toList(),
      limit: 3,
    );
    return res?.map((e) => e['text']).join(' ') ?? "";
  }

  // 4. CHAT LOGIC
  void _sendChat() async {
    if (_controller.text.isEmpty || _modelPath == null) return;
    final userText = _controller.text;
    setState(() => _messages.add({"role": "user", "content": userText}));
    _controller.clear();

    final context = await _searchContext(userText);
    final prompt = "Context: $context\n\nQuestion: $userText\nAnswer:";

    try {
      final String aiResponse = await platform.invokeMethod('runInference', {
        'prompt': prompt,
      });
      setState(() => _messages.add({"role": "ai", "content": aiResponse}));
    } catch (e) {
      setState(() => _messages.add({"role": "ai", "content": "Error: $e"}));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Local AI: $_status", style: const TextStyle(fontSize: 14)),
      ),
      body: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              TextButton.icon(
                onPressed: _pickModel,
                icon: const Icon(Icons.psychology),
                label: const Text("Model"),
              ),
              TextButton.icon(
                onPressed: _processKnowledgeBase,
                icon: const Icon(Icons.description),
                label: const Text("Data"),
              ),
            ],
          ),
          const Divider(),
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (ctx, i) => ListBody(
                children: [
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      _messages[i]['role'] == 'user' ? "You: " : "AI: ",
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Text(_messages[i]['content']!),
                  ),
                ],
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: const InputDecoration(hintText: "Ask..."),
                  ),
                ),
                IconButton(onPressed: _sendChat, icon: const Icon(Icons.send)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
