"""
retrieval/vector_store.py

ChromaDB-backed vector store for semantic similarity search over
regulation chunks. Uses OpenAI text-embedding-3-small for dense embeddings.
"""

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB + LangChain Chroma integration.

    Responsibilities:
      - Add regulation chunks (as LangChain Documents)
      - Semantic similarity search with metadata filtering
      - MMR (Maximal Marginal Relevance) search to reduce redundancy
    """

    def __init__(
        self,
        collection_name: str = "privacy_regulations",
        embedding_model: str = "text-embedding-3-small",
        chroma_host: Optional[str] = None,
        chroma_port: int = 8001,
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name

        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # Connect to ChromaDB — HTTP client if host provided, else local persistent
        if chroma_host:
            client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Connected to ChromaDB at {chroma_host}:{chroma_port}")
        else:
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Using local ChromaDB at {persist_directory}")

        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and add documents to the collection."""
        if not documents:
            return

        # ChromaDB has a batch limit; chunk into groups of 100
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            # Use chunk_id as the document ID for deduplication
            ids = [doc.metadata.get("chunk_id", f"doc_{i+j}") for j, doc in enumerate(batch)]
            self.vectorstore.add_documents(documents=batch, ids=ids)
        logger.info(f"Added {len(documents)} documents to ChromaDB")

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Standard cosine similarity search.

        Args:
            query: Natural language query
            k: Number of results to return
            filter: Optional ChromaDB metadata filter, e.g. {"regulation": "gdpr"}
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Returns (document, similarity_score) pairs."""
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )

    def mmr_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """
        Maximal Marginal Relevance search.
        Balances relevance and diversity to reduce redundant chunks.

        lambda_mult=1.0 → pure similarity, 0.0 → pure diversity
        """
        return self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )

    def get_by_ids(self, chunk_ids: list[str]) -> list[Document]:
        """Retrieve specific documents by their chunk IDs."""
        if not chunk_ids:
            return []
        results = self.vectorstore.get(ids=chunk_ids, include=["documents", "metadatas"])
        docs = []
        for i, doc_text in enumerate(results.get("documents", [])):
            metadata = results["metadatas"][i] if results.get("metadatas") else {}
            docs.append(Document(page_content=doc_text, metadata=metadata))
        return docs

    def collection_stats(self) -> dict:
        """Return basic stats about the collection."""
        count = self.vectorstore._collection.count()
        return {
            "collection": self.collection_name,
            "total_documents": count,
        }
