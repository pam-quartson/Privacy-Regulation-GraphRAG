"""
ingestion/document_loader.py

Loads privacy regulation documents (PDF, TXT, or raw text),
splits them into semantically coherent chunks, and prepares
them for both vector embedding and KG extraction.
"""

import re
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class RegulationChunk:
    """A single chunk ready for embedding and KG extraction."""
    chunk_id: str
    regulation: str          # e.g. "gdpr"
    article_number: str      # e.g. "17", "5(1)(a)"
    article_title: str       # e.g. "Right to erasure"
    text: str                # The chunk text
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "regulation": self.regulation,
                "article_number": self.article_number,
                "article_title": self.article_title,
                "char_start": self.char_start,
                "char_end": self.char_end,
                **self.metadata,
            }
        )


# ─── Article Patterns ─────────────────────────────────────────────────────────

# Regex patterns to detect article headers in different regulations
ARTICLE_PATTERNS = {
    "gdpr": re.compile(
        r"Article\s+(\d+(?:\(\d+\))?(?:\([a-z]\))?)\s*[\n\r]+([^\n\r]{5,120})",
        re.IGNORECASE
    ),
    "ccpa": re.compile(
        r"(?:Section|§)\s*(1798\.\d+(?:\.\d+)?)\s*[\.\-]?\s*([^\n\r]{5,120})",
        re.IGNORECASE
    ),
    "hipaa": re.compile(
        r"§\s*(164\.\d+)\s*([^\n\r]{5,120})",
        re.IGNORECASE
    ),
}

# Generic fallback
GENERIC_ARTICLE_PATTERN = re.compile(
    r"(?:Article|Section|§)\s+(\d+(?:\.\d+)?(?:[a-z])?)\s*[\n\r.]+([^\n\r]{0,120})",
    re.IGNORECASE
)


# ─── Loader ───────────────────────────────────────────────────────────────────

class RegulationDocumentLoader:
    """
    Loads and chunks privacy regulation documents.

    Chunking strategy:
      1. First split by article/section boundaries (preserving article context)
      2. Then apply recursive character splitting for oversized articles
      3. Tag each chunk with its regulation, article number and title
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def load_text(
        self,
        text: str,
        regulation: str,
        source_path: Optional[str] = None,
    ) -> list[RegulationChunk]:
        """Parse a raw regulation text string into chunks."""
        articles = self._split_into_articles(text, regulation)
        chunks: list[RegulationChunk] = []

        for article_num, article_title, article_text, char_start in articles:
            sub_chunks = self._chunk_article(
                article_text, regulation, article_num, article_title,
                char_start, source_path
            )
            chunks.extend(sub_chunks)

        return chunks

    def load_file(self, path: Path, regulation: str) -> list[RegulationChunk]:
        """Load a .txt or .md regulation file."""
        text = path.read_text(encoding="utf-8")
        return self.load_text(text, regulation, source_path=str(path))

    def load_directory(self, directory: Path) -> list[RegulationChunk]:
        """
        Load all regulation files from a directory.
        Files must be named after their regulation, e.g. gdpr.txt, ccpa.txt
        """
        all_chunks: list[RegulationChunk] = []
        for path in sorted(directory.glob("*.txt")):
            regulation = path.stem.lower()
            chunks = self.load_file(path, regulation)
            all_chunks.extend(chunks)
            print(f"  Loaded {regulation}: {len(chunks)} chunks")
        return all_chunks

    # ─── Private ──────────────────────────────────────────────────────────────

    def _split_into_articles(
        self, text: str, regulation: str
    ) -> list[tuple[str, str, str, int]]:
        """
        Returns list of (article_number, article_title, article_text, char_start).
        Falls back to treating the whole document as one article.
        """
        pattern = ARTICLE_PATTERNS.get(regulation, GENERIC_ARTICLE_PATTERN)
        matches = list(pattern.finditer(text))

        if not matches:
            return [("1", "Full Document", text, 0)]

        articles = []
        for i, match in enumerate(matches):
            art_num = match.group(1).strip()
            art_title = match.group(2).strip().rstrip(".")
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[start:end].strip()
            articles.append((art_num, art_title, article_text, start))

        return articles

    def _chunk_article(
        self,
        article_text: str,
        regulation: str,
        article_number: str,
        article_title: str,
        char_start: int,
        source_path: Optional[str],
    ) -> list[RegulationChunk]:
        """Split a single article into token-bounded chunks."""
        if len(article_text) <= self.chunk_size:
            return [RegulationChunk(
                chunk_id=self._make_id(regulation, article_number, 0),
                regulation=regulation,
                article_number=article_number,
                article_title=article_title,
                text=article_text,
                char_start=char_start,
                char_end=char_start + len(article_text),
                metadata={"source": source_path or regulation, "chunk_index": 0},
            )]

        raw_chunks = self._splitter.split_text(article_text)
        chunks = []
        offset = 0
        for i, chunk_text in enumerate(raw_chunks):
            chunk_start = article_text.find(chunk_text, offset)
            if chunk_start == -1:
                chunk_start = offset
            offset = chunk_start + 1

            chunks.append(RegulationChunk(
                chunk_id=self._make_id(regulation, article_number, i),
                regulation=regulation,
                article_number=article_number,
                article_title=article_title,
                text=chunk_text,
                char_start=char_start + chunk_start,
                char_end=char_start + chunk_start + len(chunk_text),
                metadata={
                    "source": source_path or regulation,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                },
            ))
        return chunks

    @staticmethod
    def _make_id(regulation: str, article_number: str, chunk_index: int) -> str:
        safe_num = re.sub(r"[^a-z0-9]", "_", article_number.lower())
        return f"{regulation}_art_{safe_num}_c{chunk_index}"
