import 'dart:io';

import 'package:dio/dio.dart';

/// Response from POST /auth.
class AuthResponse {
  final String status;
  final String username;
  final String userHash;

  const AuthResponse({
    required this.status,
    required this.username,
    required this.userHash,
  });

  factory AuthResponse.fromJson(Map<String, dynamic> json) => AuthResponse(
        status: json['status'] as String,
        username: json['username'] as String,
        userHash: json['user_hash'] as String,
      );
}

/// Response from POST /trigger-pipeline.
class TriggerPipelineResponse {
  final String status;
  final String jobId;
  final String message;

  const TriggerPipelineResponse({
    required this.status,
    required this.jobId,
    required this.message,
  });

  factory TriggerPipelineResponse.fromJson(Map<String, dynamic> json) =>
      TriggerPipelineResponse(
        status: json['status'] as String,
        jobId: json['job_id'] as String,
        message: json['message'] as String,
      );
}

/// Response from GET /status.
class PipelineStatusResponse {
  final String? jobId;
  final String status;
  final String message;
  final bool? pteReady;

  const PipelineStatusResponse({
    this.jobId,
    required this.status,
    required this.message,
    this.pteReady,
  });

  factory PipelineStatusResponse.fromJson(Map<String, dynamic> json) =>
      PipelineStatusResponse(
        jobId: json['job_id'] as String?,
        status: json['status'] as String,
        message: json['message'] as String,
        pteReady: json['pte_ready'] as bool?,
      );

  bool get isRunning => const {
        'queued',
        'pruning',
        'distilling',
        'quantizing',
        'uploading',
      }.contains(status);

  bool get isCompleted => status == 'completed';
  bool get isFailed => status == 'failed';
  bool get noJob => status == 'no_job';
}

/// Backend API service — talks to the FastAPI compression pipeline.
class BackendService {
  static const String _defaultBaseUrl = 'http://10.0.2.2:8000';

  final Dio _dio;

  BackendService({String? baseUrl})
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl ?? _defaultBaseUrl,
          connectTimeout: const Duration(seconds: 30),
          receiveTimeout: const Duration(seconds: 60),
        ));

  /// Override the base URL at runtime (e.g. from settings).
  void setBaseUrl(String url) {
    _dio.options.baseUrl = url;
  }

  // ── POST /auth ──────────────────────────────────────────────────────────

  /// Authenticate with HuggingFace token and W&B API key.
  Future<AuthResponse> authenticate({
    required String hfToken,
    required String wandbApiKey,
  }) async {
    final response = await _dio.post(
      '/auth',
      data: FormData.fromMap({
        'hf_token': hfToken,
        'wandb_api_key': wandbApiKey,
      }),
    );
    return AuthResponse.fromJson(response.data as Map<String, dynamic>);
  }

  // ── POST /trigger-pipeline ──────────────────────────────────────────────

  /// Trigger the compression pipeline with a PDF file.
  Future<TriggerPipelineResponse> triggerPipeline({
    required String hfToken,
    required File pdfFile,
  }) async {
    final response = await _dio.post(
      '/trigger-pipeline',
      data: FormData.fromMap({
        'hf_token': hfToken,
        'pdf_file': await MultipartFile.fromFile(
          pdfFile.path,
          filename: pdfFile.path.split(Platform.pathSeparator).last,
        ),
      }),
    );
    return TriggerPipelineResponse.fromJson(
        response.data as Map<String, dynamic>);
  }

  // ── GET /status ─────────────────────────────────────────────────────────

  /// Poll the pipeline status.
  Future<PipelineStatusResponse> getStatus({
    required String hfToken,
  }) async {
    final response = await _dio.get(
      '/status',
      queryParameters: {'hf_token': hfToken},
    );
    return PipelineStatusResponse.fromJson(
        response.data as Map<String, dynamic>);
  }

  // ── HuggingFace model download ────────────────────────────────────────

  static const String _hfBaseUrl = 'https://huggingface.co';

  /// Lists files in the HF repo at `{username}/compressed_models/{jobId}/`
  /// and returns the filename of the first `.pte` file found.
  Future<String?> findPteFileName({
    required String hfToken,
    required String username,
    required String jobId,
  }) async {
    final repoId = '$username/compressed_models';
    final dio = Dio();
    final response = await dio.get(
      '$_hfBaseUrl/api/models/$repoId/tree/main/$jobId',
      options: Options(headers: {'Authorization': 'Bearer $hfToken'}),
    );

    final files = response.data as List<dynamic>;
    for (final entry in files) {
      final path = entry['path'] as String?;
      if (path != null && path.toLowerCase().endsWith('.pte')) {
        // path may be "jobId/model.pte" — extract just the filename
        return path.contains('/') ? path.split('/').last : path;
      }
    }
    return null;
  }

  /// Downloads a `.pte` model file from HuggingFace Hub.
  ///
  /// The file lives at `{username}/compressed_models/{jobId}/{pteFileName}`.
  /// [savePath] is the full local path where the file will be saved.
  /// [onProgress] receives values in `[0.0 .. 1.0]`.
  Future<File> downloadPteFromHub({
    required String hfToken,
    required String username,
    required String jobId,
    required String pteFileName,
    required String savePath,
    void Function(double)? onProgress,
  }) async {
    final repoId = '$username/compressed_models';
    final url = '$_hfBaseUrl/$repoId/resolve/main/$jobId/$pteFileName';

    final dio = Dio();
    await dio.download(
      url,
      savePath,
      options: Options(
        headers: {'Authorization': 'Bearer $hfToken'},
        receiveTimeout: const Duration(minutes: 30),
      ),
      onReceiveProgress: (received, total) {
        if (total > 0) {
          onProgress?.call(received / total);
        }
      },
    );

    return File(savePath);
  }

  /// Convenience: find the `.pte` in the job folder and download it.
  ///
  /// Returns the local [File] path, or throws if no `.pte` is found.
  Future<File> downloadModelFromHub({
    required String hfToken,
    required String username,
    required String jobId,
    required String saveDir,
    void Function(double)? onProgress,
  }) async {
    final pteFileName = await findPteFileName(
      hfToken: hfToken,
      username: username,
      jobId: jobId,
    );
    if (pteFileName == null) {
      throw Exception(
        'No .pte file found in $username/compressed_models/$jobId/',
      );
    }

    final dir = Directory(saveDir);
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }

    final savePath = '${dir.path}${Platform.pathSeparator}$pteFileName';
    return downloadPteFromHub(
      hfToken: hfToken,
      username: username,
      jobId: jobId,
      pteFileName: pteFileName,
      savePath: savePath,
      onProgress: onProgress,
    );
  }
}
