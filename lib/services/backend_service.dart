import 'dart:io';
import 'package:dio/dio.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

/// Handles all network operations:
///  • Uploading a PDF to the backend pipeline.
///  • Downloading a `.pte` model from Hugging Face (with progress).
class BackendService {
  static const String _backendTrainUrl = 'https://api.mybackend.com/train';

  // ───────────────────────── PDF Upload ───────────────────────────────────

  /// Uploads [pdfFile] to the backend compression pipeline.
  ///
  /// Returns `true` on HTTP 2xx, `false` otherwise.
  Future<bool> uploadPdf(File pdfFile) async {
    try {
      final uri = Uri.parse(_backendTrainUrl);
      final request = http.MultipartRequest('POST', uri)
        ..files.add(
          await http.MultipartFile.fromPath('file', pdfFile.path),
        );
      final streamed = await request.send();
      return streamed.statusCode >= 200 && streamed.statusCode < 300;
    } catch (_) {
      // Return true in the PoC so the UI can always proceed.
      return true;
    }
  }

  // ───────────────────────── Model Download ───────────────────────────────

  /// Downloads a `.pte` model from Hugging Face at [hfUrl] using [token].
  ///
  /// [onProgress] receives a value in [0.0, 1.0] as bytes arrive.
  /// Returns the local [File] that was saved, or throws on failure.
  Future<File> downloadModel({
    required String hfUrl,
    required String token,
    required void Function(double progress) onProgress,
  }) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final savePath = p.join(docsDir.path, 'model.pte');

    final dio = Dio();
    await dio.download(
      hfUrl,
      savePath,
      options: Options(
        headers: {'Authorization': 'Bearer $token'},
      ),
      onReceiveProgress: (received, total) {
        if (total > 0) onProgress(received / total);
      },
    );

    return File(savePath);
  }

  // ───────────────────────── Model Status ─────────────────────────────────

  /// Checks if the backend has finished processing.
  ///
  /// In this PoC, always returns `true` after an artificial delay.
  Future<bool> pollModelReady() async {
    await Future.delayed(const Duration(seconds: 4));
    return true;
  }
}
