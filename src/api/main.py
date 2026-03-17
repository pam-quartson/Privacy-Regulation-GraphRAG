"""
api/main.py

FastAPI application for the Privacy Regulation GraphRAG system.

Endpoints:
  POST /query         - Ask a question about privacy regulations
  POST /ingest        - Ingest raw regulation text
  GET  /health        - System health check
  GET  /regulations   - List ingested regulations
  GET  /stats         - Graph and vector store stats
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QueryRequest, QueryResponse, Citation, RetrievalStats,
    IngestRequest, IngestResponse, HealthResponse,
)
from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore
from src.retrieval.graphrag_chain import GraphRAGChain
from src.ingestion.kg_extractor import KGExtractor
from src.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


# ─── App State ────────────────────────────────────────────────────────────────

class AppState:
    vector_store: VectorStore = None
    graph_store: GraphStore = None
    chain: GraphRAGChain = None
    pipeline: IngestionPipeline = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all stores and chains on startup."""
    logger.info("Starting GraphRAG system...")

    # Vector store
    state.vector_store = VectorStore(
        collection_name=os.getenv("CHROMA_COLLECTION", "privacy_regulations"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_host=os.getenv("CHROMA_HOST") or None,
        chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
    )

    # Graph store
    state.graph_store = GraphStore(
        host=os.getenv("MEMGRAPH_HOST", "localhost"),
        port=int(os.getenv("MEMGRAPH_PORT", "7687")),
        username=os.getenv("MEMGRAPH_USERNAME", ""),
        password=os.getenv("MEMGRAPH_PASSWORD", ""),
    )

    # GraphRAG chain
    state.chain = GraphRAGChain(
        vector_store=state.vector_store,
        graph_store=state.graph_store,
        model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        top_k_vector=int(os.getenv("TOP_K_VECTOR", "5")),
        top_k_graph=int(os.getenv("TOP_K_GRAPH", "5")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "6000")),
    )

    # Ingestion pipeline
    kg_extractor = KGExtractor(
        model_name=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    )
    state.pipeline = IngestionPipeline(
        vector_store=state.vector_store,
        graph_store=state.graph_store,
        kg_extractor=kg_extractor,
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
    )

    logger.info("GraphRAG system ready")
    yield

    # Cleanup
    state.graph_store.close()
    logger.info("GraphRAG system shut down")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Privacy Regulation GraphRAG",
    description=(
        "Hybrid RAG system combining semantic vector search (ChromaDB) "
        "and knowledge graph traversal (Memgraph) for privacy regulation queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_regulations(request: QueryRequest) -> QueryResponse:
    """
    Ask a question about privacy regulations.

    The system retrieves relevant context via parallel vector search and
    graph traversal, fuses results with RRF, then generates an LLM answer.
    """
    if state.chain is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        if request.include_trace:
            result = state.chain.query_with_trace(
                question=request.question,
            )
            response_data = result
            trace = result.get("retrieval_trace", {}).get("vector_docs")
        else:
            response = state.chain.query(
                question=request.question,
                regulation_filter=request.regulation_filter,
            )
            response_data = response.to_dict()
            trace = None

        return QueryResponse(
            query=response_data["query"],
            answer=response_data["answer"],
            citations=[Citation(**c) for c in response_data["citations"]],
            retrieval_stats=RetrievalStats(**response_data["retrieval_stats"]),
            retrieval_trace=trace,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_regulation(request: IngestRequest) -> IngestResponse:
    """Ingest raw regulation text into the system."""
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        stats = state.pipeline.ingest_text(request.text, request.regulation)
        return IngestResponse(
            regulation=stats.regulation,
            chunks_created=stats.total_chunks,
            graph_nodes=stats.graph_nodes,
            graph_relationships=stats.graph_relationships,
            duration_seconds=stats.duration_seconds,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check system health."""
    chroma_ok = False
    memgraph_ok = False
    vector_count = 0
    graph_stats = {}

    try:
        stats = state.vector_store.collection_stats()
        vector_count = stats["total_documents"]
        chroma_ok = True
    except Exception:
        pass

    try:
        memgraph_ok = state.graph_store.is_healthy()
        if memgraph_ok:
            graph_stats = state.graph_store.get_stats()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (chroma_ok and memgraph_ok) else "degraded",
        chromadb=chroma_ok,
        memgraph=memgraph_ok,
        vector_store_docs=vector_count,
        graph_stats=graph_stats,
    )


@app.get("/regulations")
async def list_regulations() -> dict:
    """List all ingested regulations in the graph."""
    try:
        regs = state.graph_store.query(
            "MATCH (a:Article) RETURN DISTINCT a.regulation AS regulation, "
            "count(a) AS article_count ORDER BY regulation"
        )
        return {"regulations": regs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats() -> dict:
    """Return detailed system statistics."""
    try:
        vector_stats = state.vector_store.collection_stats()
        graph_stats = state.graph_store.get_stats()
        return {
            "vector_store": vector_stats,
            "knowledge_graph": graph_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
