import 'package:objectbox/objectbox.dart';

@Entity()
class DocumentChunk {
  /// ObjectBox-managed primary key.
  @Id()
  int id;

  /// Raw text of the chunk extracted from the PDF.
  String text;

  /// Placeholder for future dense vector embeddings.
  /// Marked @Transient so ObjectBox ignores it until a proper
  /// float-vector property (or separate embedding store) is wired up.
  @Transient()
  List<double>? embedding;

  DocumentChunk({
    this.id = 0,
    required this.text,
    this.embedding,
  });
}
