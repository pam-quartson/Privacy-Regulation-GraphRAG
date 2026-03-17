"""
tests/test_ingestion.py

Unit tests for the document loader and KG extractor.
Run with: pytest tests/test_ingestion.py -v
"""

import pytest
from src.ingestion.document_loader import RegulationDocumentLoader, RegulationChunk


# ─── Document Loader Tests ────────────────────────────────────────────────────

SAMPLE_GDPR_TEXT = """
Article 1 - Subject-matter and objectives

This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data and rules relating to the free movement of personal data.

This Regulation protects fundamental rights and freedoms of natural persons and in particular their right to the protection of personal data.

Article 17 - Right to erasure ('right to be forgotten')

The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay and the controller shall have the obligation to erase personal data without undue delay where one of the following grounds applies:
(a) the personal data are no longer necessary in relation to the purposes for which they were collected or otherwise processed;
(b) the data subject withdraws consent on which the processing is based according to point (a) of Article 6(1), or point (a) of Article 9(2), and where there is no other legal ground for the processing;
(c) the data subject objects to the processing pursuant to Article 21(1) and there are no overriding legitimate grounds for the processing.

Article 33 - Notification of a personal data breach to the supervisory authority

In the case of a personal data breach, the controller shall without undue delay and, where feasible, not later than 72 hours after having become aware of it, notify the personal data breach to the supervisory authority competent in accordance with Article 55.

The notification referred to in paragraph 1 shall at least contain the nature of the personal data breach including where possible, the categories and approximate number of data subjects concerned and the categories and approximate number of personal data records concerned.
"""


class TestRegulationDocumentLoader:
    def setup_method(self):
        self.loader = RegulationDocumentLoader(chunk_size=512, chunk_overlap=64)

    def test_load_text_returns_chunks(self):
        chunks = self.loader.load_text(SAMPLE_GDPR_TEXT, "gdpr")
        assert len(chunks) > 0
        assert all(isinstance(c, RegulationChunk) for c in chunks)

    def test_chunks_have_required_fields(self):
        chunks = self.loader.load_text(SAMPLE_GDPR_TEXT, "gdpr")
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.regulation == "gdpr"
            assert chunk.text
            assert chunk.article_number
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_article_numbers_detected(self):
        chunks = self.loader.load_text(SAMPLE_GDPR_TEXT, "gdpr")
        article_numbers = {c.article_number for c in chunks}
        # Should detect articles 1, 17, 33
        assert "17" in article_numbers or any("17" in n for n in article_numbers)

    def test_chunk_size_respected(self):
        loader = RegulationDocumentLoader(chunk_size=200, chunk_overlap=20)
        chunks = loader.load_text(SAMPLE_GDPR_TEXT, "gdpr")
        # Chunks should generally not exceed chunk_size significantly
        for chunk in chunks:
            assert len(chunk.text) <= 400  # Allow some overflow for article header

    def test_to_langchain_doc(self):
        chunks = self.loader.load_text(SAMPLE_GDPR_TEXT, "gdpr")
        for chunk in chunks:
            doc = chunk.to_langchain_doc()
            assert doc.page_content == chunk.text
            assert doc.metadata["regulation"] == "gdpr"
            assert doc.metadata["chunk_id"] == chunk.chunk_id

    def test_chunk_ids_unique(self):
        chunks = self.loader.load_text(SAMPLE_GDPR_TEXT, "gdpr")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_ccpa_pattern(self):
        ccpa_text = """
Section 1798.100. (a) A consumer shall have the right to request that a business that collects a consumer's personal information disclose to that consumer the categories and specific pieces of personal information the business has collected.

Section 1798.105. (a) A consumer shall have the right to request that a business delete any personal information about the consumer which the business has collected from the consumer.
"""
        chunks = self.loader.load_text(ccpa_text, "ccpa")
        article_numbers = {c.article_number for c in chunks}
        assert any("1798" in n for n in article_numbers)

    def test_empty_text_handled(self):
        chunks = self.loader.load_text("", "gdpr")
        # Should return empty or single chunk without error
        assert isinstance(chunks, list)

    def test_short_text_single_chunk(self):
        text = "Article 1 - Title\nThis is a short article."
        chunks = self.loader.load_text(text, "gdpr")
        assert len(chunks) >= 1
