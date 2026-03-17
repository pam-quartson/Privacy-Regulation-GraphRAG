"""
api/models.py

Pydantic models for the GraphRAG API request/response contracts.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language question about privacy regulations",
        min_length=5,
        max_length=1000,
        examples=["What are GDPR obligations for data controllers after a breach?"],
    )
    regulation_filter: Optional[str] = Field(
        None,
        description="Restrict search to a specific regulation (gdpr, ccpa, hipaa)",
        pattern="^[a-z]{2,10}$",
    )
    include_trace: bool = Field(
        False,
        description="Include full retrieval trace in response (for debugging)",
    )


class Citation(BaseModel):
    regulation: Optional[str]
    article: Optional[str]
    title: Optional[str]
    source: Optional[str]  # "vector(#1) + graph(#2)"
    chunk_id: Optional[str]


class RetrievalStats(BaseModel):
    vector_results: int
    graph_results: int
    fused_results: int
    latency_ms: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieval_stats: RetrievalStats
    retrieval_trace: Optional[list[dict]] = None


class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw regulation text to ingest")
    regulation: str = Field(..., description="Regulation identifier (e.g. gdpr)")


class IngestResponse(BaseModel):
    regulation: str
    chunks_created: int
    graph_nodes: int
    graph_relationships: int
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    chromadb: bool
    memgraph: bool
    vector_store_docs: int
    graph_stats: dict[str, int]
