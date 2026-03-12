"""
retrieval/fusion.py

Reciprocal Rank Fusion (RRF) for merging results from vector search
and graph traversal into a single ranked context list.

RRF Formula:
  score(d) = Σ 1 / (k + rank_i(d))

where k=60 is the standard smoothing constant that prevents high scores
from dominating when a document ranks first in only one list.

Why RRF?
  - Parameter-free (no score normalization needed between heterogeneous sources)
  - Robust to outliers
  - Proven effective in hybrid retrieval (Cormack et al., 2009)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document


@dataclass
class FusedResult:
    """A single fused result combining vector and graph sources."""
    content: str
    rrf_score: float
    vector_rank: Optional[int] = None   # rank in vector results (1-indexed)
    graph_rank: Optional[int] = None    # rank in graph results (1-indexed)
    regulation: Optional[str] = None
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    chunk_id: Optional[str] = None
    retrieval_strategy: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        sources = []
        if self.vector_rank is not None:
            sources.append(f"vector(#{self.vector_rank})")
        if self.graph_rank is not None:
            sources.append(f"graph(#{self.graph_rank})")
        return " + ".join(sources) if sources else "unknown"

    def to_context_string(self) -> str:
        """Format as an LLM-friendly context block."""
        header_parts = []
        if self.regulation:
            header_parts.append(self.regulation.upper())
        if self.article_number:
            header_parts.append(f"Art. {self.article_number}")
        if self.article_title:
            header_parts.append(self.article_title)

        header = " | ".join(header_parts) if header_parts else "Regulation Text"
        return f"[{header}]\n{self.content}"


class ContextFusion:
    """
    Fuses vector search results and graph traversal results using RRF.

    Usage:
        fusion = ContextFusion(rrf_k=60)
        fused = fusion.fuse(vector_docs, graph_results)
        context_str = fusion.build_context_string(fused, max_tokens=4000)
    """

    def __init__(self, rrf_k: int = 60, max_context_tokens: int = 6000):
        self.rrf_k = rrf_k
        self.max_context_tokens = max_context_tokens

    def fuse(
        self,
        vector_docs: list[Document],
        graph_results: list[dict[str, Any]],
        vector_weight: float = 1.0,
        graph_weight: float = 1.0,
    ) -> list[FusedResult]:
        """
        Merge and rerank results from both retrieval paths.

        Returns list of FusedResult sorted by RRF score (descending).
        """
        scores: dict[str, float] = {}
        result_map: dict[str, FusedResult] = {}

        # --- Process vector results ---
        for rank, doc in enumerate(vector_docs, start=1):
            key = self._doc_key(doc)
            rrf_contrib = vector_weight / (self.rrf_k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_contrib

            if key not in result_map:
                result_map[key] = FusedResult(
                    content=doc.page_content,
                    rrf_score=0.0,
                    vector_rank=rank,
                    regulation=doc.metadata.get("regulation"),
                    article_number=doc.metadata.get("article_number"),
                    article_title=doc.metadata.get("article_title"),
                    chunk_id=doc.metadata.get("chunk_id"),
                    metadata=doc.metadata,
                )
            else:
                result_map[key].vector_rank = rank

        # --- Process graph results ---
        graph_texts = self._graph_results_to_texts(graph_results)
        for rank, (text, meta) in enumerate(graph_texts, start=1):
            key = self._text_key(text)
            rrf_contrib = graph_weight / (self.rrf_k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_contrib

            if key not in result_map:
                result_map[key] = FusedResult(
                    content=text,
                    rrf_score=0.0,
                    graph_rank=rank,
                    regulation=meta.get("regulation"),
                    article_number=meta.get("article_number"),
                    article_title=meta.get("article_title"),
                    retrieval_strategy=meta.get("retrieval_strategy"),
                    metadata=meta,
                )
            else:
                result_map[key].graph_rank = rank

        # Apply scores and sort
        for key, score in scores.items():
            if key in result_map:
                result_map[key].rrf_score = score

        ranked = sorted(result_map.values(), key=lambda r: r.rrf_score, reverse=True)
        return ranked

    def build_context_string(
        self, fused_results: list[FusedResult], max_tokens: Optional[int] = None
    ) -> str:
        """
        Build the context string to pass to the LLM.
        Truncates to max_tokens (approximate, using 4 chars/token heuristic).
        """
        max_chars = (max_tokens or self.max_context_tokens) * 4
        blocks = []
        total_chars = 0

        for i, result in enumerate(fused_results, start=1):
            block = f"[Context {i} | {result.source_label}]\n{result.to_context_string()}"
            block_chars = len(block)

            if total_chars + block_chars > max_chars and blocks:
                break

            blocks.append(block)
            total_chars += block_chars

        return "\n\n---\n\n".join(blocks)

    def get_citations(self, fused_results: list[FusedResult]) -> list[dict]:
        """Return citation metadata for the top results."""
        citations = []
        for r in fused_results:
            if r.regulation or r.article_number:
                citations.append({
                    "regulation": r.regulation,
                    "article": r.article_number,
                    "title": r.article_title,
                    "source": r.source_label,
                    "chunk_id": r.chunk_id,
                })
        return citations

    # ─── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _doc_key(doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return f"chunk::{chunk_id}"
        return f"text::{hash(doc.page_content[:100])}"

    @staticmethod
    def _text_key(text: str) -> str:
        return f"text::{hash(text[:100])}"

    @staticmethod
    def _graph_results_to_texts(
        graph_results: list[dict[str, Any]]
    ) -> list[tuple[str, dict]]:
        """
        Convert raw Cypher result dicts to (text, metadata) pairs
        suitable for RRF ranking.
        """
        texts = []
        for result in graph_results:
            parts = []
            meta = {
                "regulation": result.get("regulation"),
                "article_number": result.get("article") or result.get("number"),
                "article_title": result.get("title") or result.get("article_title"),
                "retrieval_strategy": result.get("retrieval_strategy", "graph"),
            }

            # Build a human-readable text from the graph result
            if result.get("title"):
                parts.append(f"Article {result.get('article', '')}: {result['title']}")

            if result.get("obligation"):
                parts.append(f"Obligation: {result['obligation']}")
                if result.get("severity"):
                    parts.append(f"Severity: {result['severity']}")

            if result.get("obligations"):
                for ob in result["obligations"]:
                    if isinstance(ob, str):
                        parts.append(f"• {ob}")
                    elif isinstance(ob, dict):
                        parts.append(f"• {ob.get('description', ob)}")

            if result.get("rights"):
                for r in result["rights"]:
                    if isinstance(r, str):
                        parts.append(f"Right: {r}")
                    elif isinstance(r, dict):
                        parts.append(f"Right: {r.get('right_name', '')} - {r.get('right_description', '')}")

            if result.get("responsible_party"):
                parts.append(f"Responsible party: {result['responsible_party']}")

            if result.get("detail_reg1"):
                parts.append(f"Under {result.get('regulation', 'Regulation 1')}: {result['detail_reg1']}")
            if result.get("detail_reg2"):
                parts.append(f"Comparison: {result['detail_reg2']}")

            text = "\n".join(parts)
            if text.strip():
                texts.append((text, meta))

        return texts
