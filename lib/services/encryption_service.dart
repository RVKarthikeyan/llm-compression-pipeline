import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:pointycastle/export.dart';

/// AES-256-CBC symmetric encryption service (PKCS7 padding).
///
/// Encrypts PDF content before transmitting to the backend so that the
/// transport layer never sees plaintext data.
class EncryptionService {
  final Uint8List _key;
  final Uint8List _iv;

  /// Creates a service with a secure-random 256-bit key and 128-bit IV.
  EncryptionService() : _key = _randomBytes(32), _iv = _randomBytes(16);

  static Uint8List _randomBytes(int n) {
    final rng = Random.secure();
    return Uint8List.fromList(List.generate(n, (_) => rng.nextInt(256)));
  }

  /// Encrypts [plainText] → Base64-encoded cipher text.
  String encrypt(String plainText) {
    final cipher = PaddedBlockCipher('AES/CBC/PKCS7')
      ..init(
        true,
        PaddedBlockCipherParameters(
          ParametersWithIV(KeyParameter(_key), _iv),
          null,
        ),
      );
    final input = Uint8List.fromList(utf8.encode(plainText));
    return base64.encode(cipher.process(input));
  }

  /// Decrypts a Base64-encoded cipher text → original plain text.
  String decrypt(String base64Cipher) {
    final cipher = PaddedBlockCipher('AES/CBC/PKCS7')
      ..init(
        false,
        PaddedBlockCipherParameters(
          ParametersWithIV(KeyParameter(_key), _iv),
          null,
        ),
      );
    final input = base64.decode(base64Cipher);
    return utf8.decode(cipher.process(input));
  }

  /// Base64-encoded key (for transmission alongside encrypted payload).
  String get keyBase64 => base64.encode(_key);

  /// Base64-encoded IV.
  String get ivBase64 => base64.encode(_iv);

  // ── Self-test ───────────────────────────────────────────────────────────

  /// Runs an encrypt → decrypt round-trip and returns the result.
  /// Use this to visually demonstrate that encryption works.
  static EncryptionTestResult selfTest(String input) {
    final svc = EncryptionService();
    final encrypted = svc.encrypt(input);
    final decrypted = svc.decrypt(encrypted);
    return EncryptionTestResult(
      original: input,
      encrypted: encrypted,
      decrypted: decrypted,
      passed: input == decrypted,
      keyBase64: svc.keyBase64,
      ivBase64: svc.ivBase64,
    );
  }
}

/// Result of an encryption self-test.
class EncryptionTestResult {
  final String original;
  final String encrypted;
  final String decrypted;
  final bool passed;
  final String keyBase64;
  final String ivBase64;

  const EncryptionTestResult({
    required this.original,
    required this.encrypted,
    required this.decrypted,
    required this.passed,
    required this.keyBase64,
    required this.ivBase64,
  });
}
