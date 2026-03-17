"""
ingestion/pipeline.py

Orchestrates the full ingestion pipeline:
  1. Load regulation documents
  2. Chunk them
  3. Embed chunks → ChromaDB
  4. Extract KG entities → Memgraph
"""

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field

from src.ingestion.document_loader import RegulationDocumentLoader, RegulationChunk
from src.ingestion.kg_extractor import KGExtractor
from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore
from src.graph.schema import ExtractedGraph

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    regulation: str
    total_chunks: int = 0
    embedded_chunks: int = 0
    graph_nodes: int = 0
    graph_relationships: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.regulation}] "
            f"{self.embedded_chunks}/{self.total_chunks} chunks embedded, "
            f"{self.graph_nodes} graph nodes, "
            f"{self.graph_relationships} relationships, "
            f"{self.duration_seconds:.1f}s"
        )


class IngestionPipeline:
    """
    Full ingestion pipeline for privacy regulation documents.

    For each document:
      - Parallel path 1: embed chunks → store in ChromaDB
      - Parallel path 2: zero-shot KG extraction → store in Memgraph

    Both paths use the same chunks so chunk_ids link the two stores.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        kg_extractor: KGExtractor,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.kg_extractor = kg_extractor
        self.loader = RegulationDocumentLoader(chunk_size, chunk_overlap)

    def ingest_file(self, path: Path, regulation: str) -> IngestionStats:
        """Ingest a single regulation file."""
        stats = IngestionStats(regulation=regulation)
        start = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"Ingesting: {path.name} ({regulation.upper()})")
        print(f"{'='*60}")

        # Step 1: Load and chunk
        print("Step 1/3: Loading and chunking document...")
        chunks = self.loader.load_file(path, regulation)
        stats.total_chunks = len(chunks)
        print(f"  → {len(chunks)} chunks created")

        # Step 2: Embed and store in ChromaDB
        print("Step 2/3: Embedding chunks → ChromaDB...")
        docs = [chunk.to_langchain_doc() for chunk in chunks]
        self.vector_store.add_documents(docs)
        stats.embedded_chunks = len(docs)
        print(f"  → {len(docs)} chunks embedded and stored")

        # Step 3: KG extraction → Memgraph
        print("Step 3/3: Extracting knowledge graph → Memgraph...")
        extracted: ExtractedGraph = self.kg_extractor.extract_batch(chunks)
        print(f"  → {extracted.summary()}")
        self.graph_store.ingest_graph(extracted)

        stats.graph_nodes = (
            len(extracted.articles) + len(extracted.obligations) +
            len(extracted.rights) + len(extracted.parties) +
            len(extracted.concepts)
        )
        stats.graph_relationships = len(extracted.relationships)
        stats.duration_seconds = time.perf_counter() - start

        print(f"\n✓ Done: {stats}")
        return stats

    def ingest_directory(self, directory: Path) -> list[IngestionStats]:
        """Ingest all .txt regulation files in a directory."""
        all_stats = []
        files = sorted(directory.glob("*.txt"))

        if not files:
            logger.warning(f"No .txt files found in {directory}")
            return []

        print(f"\nFound {len(files)} regulation files: {[f.name for f in files]}")

        for path in files:
            regulation = path.stem.lower()
            try:
                stats = self.ingest_file(path, regulation)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed to ingest {path.name}: {e}", exc_info=True)
                all_stats.append(IngestionStats(
                    regulation=regulation,
                    errors=[str(e)]
                ))

        self._print_summary(all_stats)
        return all_stats

    def ingest_text(self, text: str, regulation: str) -> IngestionStats:
        """Ingest regulation text directly (useful for testing)."""
        stats = IngestionStats(regulation=regulation)
        start = time.perf_counter()

        chunks = self.loader.load_text(text, regulation)
        stats.total_chunks = len(chunks)

        docs = [chunk.to_langchain_doc() for chunk in chunks]
        self.vector_store.add_documents(docs)
        stats.embedded_chunks = len(docs)

        extracted = self.kg_extractor.extract_batch(chunks, verbose=False)
        self.graph_store.ingest_graph(extracted)

        stats.graph_nodes = (
            len(extracted.articles) + len(extracted.obligations) +
            len(extracted.rights) + len(extracted.parties) +
            len(extracted.concepts)
        )
        stats.graph_relationships = len(extracted.relationships)
        stats.duration_seconds = time.perf_counter() - start
        return stats

    @staticmethod
    def _print_summary(stats_list: list[IngestionStats]) -> None:
        print(f"\n{'='*60}")
        print("INGESTION SUMMARY")
        print(f"{'='*60}")
        total_chunks = sum(s.total_chunks for s in stats_list)
        total_nodes = sum(s.graph_nodes for s in stats_list)
        total_rels = sum(s.graph_relationships for s in stats_list)
        total_time = sum(s.duration_seconds for s in stats_list)

        for s in stats_list:
            status = "✓" if not s.errors else "✗"
            print(f"  {status} {s}")

        print(f"\n  Total: {total_chunks} chunks, {total_nodes} nodes, "
              f"{total_rels} relationships in {total_time:.1f}s")
