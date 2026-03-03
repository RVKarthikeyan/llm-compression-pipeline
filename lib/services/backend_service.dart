import 'dart:io';

/// Backend API service.
///
/// All 3 APIs are **dummy stubs** for the MVP — they simulate network
/// delays and always return success. Replace with real HTTP calls when
/// the backend is deployed.
class BackendService {
  // ignore: unused_field
  static const String _baseUrl = 'https://api.localai-backend.com/v1';

  // ── API 1: Post HF Token ────────────────────────────────────────────────

  /// Posts the Hugging Face access token to the backend.
  /// Returns `true` on success.
  Future<bool> postHfToken(String token) async {
    // TODO: POST $_baseUrl/auth/token  body: { "hf_token": token }
    await Future.delayed(const Duration(seconds: 1));
    return true;
  }

  // ── API 2: Upload Encrypted PDF Content ─────────────────────────────────

  /// Uploads AES-encrypted PDF content + key/IV to the backend.
  /// Returns `true` on success.
  Future<bool> uploadEncryptedContent({
    required String encryptedBase64,
    required String keyBase64,
    required String ivBase64,
  }) async {
    // TODO: POST $_baseUrl/pipeline/upload
    // body: { "data": encryptedBase64, "key": keyBase64, "iv": ivBase64 }
    await Future.delayed(const Duration(seconds: 2));
    return true;
  }

  // ── API 3: Trigger Pipeline & Get Model ─────────────────────────────────

  /// Triggers the distillation pipeline on the backend.
  /// Returns a URL for the resulting .pte model (or null on failure).
  Future<String?> triggerPipeline() async {
    // TODO: POST $_baseUrl/pipeline/run
    // response: { "model_url": "https://..." }
    await Future.delayed(const Duration(seconds: 3));
    return '$_baseUrl/models/latest.pte';
  }

  /// Downloads the .pte model file.
  /// [onProgress] receives values in [0.0 .. 1.0] as bytes arrive.
  Future<File?> downloadModel({
    required String url,
    required String savePath,
    void Function(double)? onProgress,
  }) async {
    // TODO: use Dio to download from url to savePath
    for (var i = 1; i <= 10; i++) {
      await Future.delayed(const Duration(milliseconds: 200));
      onProgress?.call(i / 10);
    }
    return null; // dummy — no real file in MVP
  }
}
