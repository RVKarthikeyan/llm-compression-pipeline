import 'package:objectbox/objectbox.dart';

@Entity()
class DocumentChunk {
  /// ObjectBox-managed primary key.
  @Id()
  int id;

  /// Raw text of the chunk extracted from the PDF.
  String text;

  /// Dense vector embedding (384 dimensions, from all-MiniLM-L6-v2).
  /// Indexed with HNSW for fast approximate nearest-neighbor search.
  @HnswIndex(dimensions: 384, distanceType: VectorDistanceType.cosine)
  @Property(type: PropertyType.floatVector)
  List<double>? embedding;

  DocumentChunk({this.id = 0, required this.text, this.embedding});
}
