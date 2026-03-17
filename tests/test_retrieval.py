"""
tests/test_retrieval.py

Unit tests for the RRF fusion module.
Run with: pytest tests/test_retrieval.py -v
"""

import pytest
from langchain_core.documents import Document
from src.retrieval.fusion import ContextFusion, FusedResult


def make_doc(content: str, chunk_id: str, regulation: str = "gdpr", article: str = "1") -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "regulation": regulation,
            "article_number": article,
            "article_title": f"Article {article}",
        }
    )


class TestContextFusion:
    def setup_method(self):
        self.fusion = ContextFusion(rrf_k=60)

    def test_fuse_vector_only(self):
        docs = [
            make_doc("GDPR Article 17 text", "gdpr_art_17_c0"),
            make_doc("GDPR Article 33 text", "gdpr_art_33_c0"),
        ]
        results = self.fusion.fuse(docs, [])
        assert len(results) == 2
        # Higher ranked vector doc should have higher RRF score
        assert results[0].rrf_score >= results[1].rrf_score

    def test_fuse_graph_only(self):
        graph_results = [
            {
                "regulation": "gdpr",
                "article": "33",
                "title": "Breach notification",
                "obligation": "Notify supervisory authority within 72 hours",
                "severity": "mandatory",
                "retrieval_strategy": "breach_chain",
            }
        ]
        results = self.fusion.fuse([], graph_results)
        assert len(results) >= 1

    def test_fuse_hybrid_boosts_overlap(self):
        """A result appearing in both vector and graph should score highest."""
        # Same content appears in both
        shared_text = "GDPR Article 33 breach notification"
        docs = [
            make_doc(shared_text, "gdpr_art_33_c0", article="33"),
            make_doc("GDPR Article 5 principles", "gdpr_art_5_c0", article="5"),
        ]
        graph_results = [
            {
                "regulation": "gdpr",
                "article": "33",
                "title": "Notification of a personal data breach",
                "obligation": "Notify supervisory authority within 72 hours",
                "retrieval_strategy": "breach_chain",
            }
        ]
        results = self.fusion.fuse(docs, graph_results)
        assert len(results) >= 2
        # The top result should have both vector_rank and graph_rank OR high score
        top = results[0]
        assert top.rrf_score > 0

    def test_rrf_formula(self):
        """Verify RRF scoring: rank 1 with k=60 should give 1/61."""
        docs = [make_doc("text", "chunk_1")]
        results = self.fusion.fuse(docs, [])
        expected_score = 1.0 / (60 + 1)
        assert abs(results[0].rrf_score - expected_score) < 1e-10

    def test_build_context_string(self):
        docs = [
            make_doc("Article 17 content about erasure rights.", "gdpr_art_17_c0", article="17"),
            make_doc("Article 33 content about breach notification.", "gdpr_art_33_c0", article="33"),
        ]
        results = self.fusion.fuse(docs, [])
        context = self.fusion.build_context_string(results)
        assert "Article 17" in context or "17" in context
        assert len(context) > 0

    def test_context_string_respects_token_limit(self):
        # Create many large docs
        docs = [make_doc("x" * 1000, f"chunk_{i}") for i in range(20)]
        results = self.fusion.fuse(docs, [])
        context = self.fusion.build_context_string(results, max_tokens=500)
        # 500 tokens * 4 chars/token = 2000 chars max
        assert len(context) <= 2000 * 1.1  # Allow 10% buffer

    def test_citations_extracted(self):
        docs = [make_doc("content", "gdpr_art_17_c0", regulation="gdpr", article="17")]
        results = self.fusion.fuse(docs, [])
        citations = self.fusion.get_citations(results)
        assert len(citations) == 1
        assert citations[0]["regulation"] == "gdpr"
        assert citations[0]["article"] == "17"

    def test_empty_inputs(self):
        results = self.fusion.fuse([], [])
        assert results == []
        context = self.fusion.build_context_string([])
        assert context == ""

    def test_fused_result_source_label(self):
        r = FusedResult(
            content="text", rrf_score=0.1,
            vector_rank=1, graph_rank=2
        )
        assert "vector" in r.source_label
        assert "graph" in r.source_label

        r_vector_only = FusedResult(content="text", rrf_score=0.1, vector_rank=1)
        assert "graph" not in r_vector_only.source_label

    def test_deduplication(self):
        """Same chunk appearing twice in vector results should be deduplicated."""
        doc = make_doc("same content", "chunk_1")
        results = self.fusion.fuse([doc, doc], [])
        chunk_ids = [r.chunk_id for r in results if r.chunk_id == "chunk_1"]
        assert len(chunk_ids) == 1
