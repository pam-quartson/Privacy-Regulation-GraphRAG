"""
scripts/ingest.py

CLI script to run the full ingestion pipeline.

Usage:
  python scripts/ingest.py --source data/regulations/
  python scripts/ingest.py --file data/regulations/gdpr.txt --regulation gdpr
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore
from src.ingestion.kg_extractor import KGExtractor
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingest privacy regulation documents into GraphRAG system"
    )
    parser.add_argument("--source", type=Path, help="Directory of regulation .txt files")
    parser.add_argument("--file", type=Path, help="Single regulation file")
    parser.add_argument("--regulation", type=str, help="Regulation name (required with --file)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    if not args.source and not args.file:
        parser.print_help()
        sys.exit(1)

    if args.file and not args.regulation:
        print("Error: --regulation is required when using --file")
        sys.exit(1)

    print("Initializing stores...")
    vector_store = VectorStore(
        collection_name=os.getenv("CHROMA_COLLECTION", "privacy_regulations"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_host=os.getenv("CHROMA_HOST") or None,
    )

    graph_store = GraphStore(
        host=os.getenv("MEMGRAPH_HOST", "localhost"),
        port=int(os.getenv("MEMGRAPH_PORT", "7687")),
    )

    kg_extractor = KGExtractor(
        model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    )

    pipeline = IngestionPipeline(
        vector_store=vector_store,
        graph_store=graph_store,
        kg_extractor=kg_extractor,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.source:
        pipeline.ingest_directory(args.source)
    elif args.file:
        pipeline.ingest_file(args.file, args.regulation)

    graph_store.close()
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
