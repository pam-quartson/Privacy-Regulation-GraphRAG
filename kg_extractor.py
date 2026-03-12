"""
ingestion/kg_extractor.py

Zero-shot knowledge graph extraction from regulation text chunks.
Uses an LLM (GPT-4o) with a structured JSON prompt to extract:
  - Articles, Obligations, Rights, Parties, Concepts
  - Relationships between them

This is the core of what makes GraphRAG outperform vanilla RAG:
we capture structured relational information that embeddings alone miss.
"""

import json
import re
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.ingestion.document_loader import RegulationChunk
from src.graph.schema import (
    ArticleNode, ObligationNode, RightNode, PartyNode, ConceptNode,
    ExtractedGraph, RelationshipType, PartyType, ObligationSeverity,
)

logger = logging.getLogger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal knowledge graph extraction expert specializing in privacy regulations.

Extract structured knowledge graph data from the given regulation text.

Return ONLY valid JSON (no markdown, no explanation) matching this exact schema:

{
  "articles": [
    {
      "id": "string (e.g. gdpr_art_17)",
      "regulation": "string (e.g. gdpr)",
      "number": "string (e.g. 17 or 5.1.a)",
      "title": "string",
      "text": "string (first 300 chars of article text)"
    }
  ],
  "obligations": [
    {
      "id": "string (e.g. gdpr_art_17_ob_1)",
      "description": "string (specific actionable obligation)",
      "regulation": "string",
      "article_ref": "string (article id this belongs to)",
      "severity": "mandatory | conditional | recommended",
      "conditions": "string or null (when this obligation is triggered)"
    }
  ],
  "rights": [
    {
      "id": "string (e.g. gdpr_right_erasure)",
      "name": "string (e.g. Right to Erasure)",
      "description": "string",
      "regulation": "string",
      "article_ref": "string"
    }
  ],
  "parties": [
    {
      "id": "string (e.g. party_controller)",
      "name": "string",
      "party_type": "Controller | Processor | DataSubject | SupervisoryAuthority | ThirdParty",
      "description": "string or null"
    }
  ],
  "concepts": [
    {
      "id": "string (e.g. concept_personal_data)",
      "name": "string (e.g. Personal Data)",
      "definition": "string",
      "regulation": "string",
      "synonyms": ["string"]
    }
  ],
  "relationships": [
    {
      "source_id": "string",
      "relationship_type": "REQUIRES | GRANTS | SUPERSEDES | APPLIES_TO | EXERCISED_AGAINST | RELATES_TO | DEFINED_IN | PART_OF",
      "target_id": "string"
    }
  ]
}

Rules:
- Only extract entities that are clearly present in the text
- Use snake_case IDs that are globally unique across regulations
- Obligations must be specific and actionable, not vague
- Every entity needs at least one relationship
- For cross-regulation links (e.g. GDPR Art 17 SUPERSEDES CCPA 1798.105), only include if explicitly mentioned
- Return {} if text contains no extractable entities
"""

EXTRACTION_PROMPT = """Extract knowledge graph entities and relationships from this privacy regulation text:

REGULATION: {regulation}
ARTICLE: {article_number} - {article_title}
SOURCE CHUNK ID: {chunk_id}

TEXT:
{text}

Remember: Return ONLY valid JSON matching the schema. No markdown fences."""


# ─── Extractor ────────────────────────────────────────────────────────────────

class KGExtractor:
    """
    Zero-shot knowledge graph extractor.

    For each regulation chunk, calls GPT-4o to extract structured
    entities and relationships, then converts them to typed dataclasses
    compatible with Memgraph ingestion.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._party_cache: dict[str, PartyNode] = {}  # Deduplicate parties across chunks

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def extract(self, chunk: RegulationChunk) -> ExtractedGraph:
        """Extract a knowledge graph from a single regulation chunk."""
        prompt = EXTRACTION_PROMPT.format(
            regulation=chunk.regulation.upper(),
            article_number=chunk.article_number,
            article_title=chunk.article_title,
            chunk_id=chunk.chunk_id,
            text=chunk.text[:3000],  # Safety cap
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        raw_json = self._clean_json(response.content)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for chunk {chunk.chunk_id}: {e}")
            return ExtractedGraph()

        return self._parse_extracted_data(data, chunk)

    def extract_batch(
        self,
        chunks: list[RegulationChunk],
        verbose: bool = True,
    ) -> ExtractedGraph:
        """Extract KG from a list of chunks and merge into one graph."""
        merged = ExtractedGraph()
        seen_ids: set[str] = set()

        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"  Extracting [{i+1}/{len(chunks)}] {chunk.chunk_id}...")

            graph = self.extract(chunk)

            # Merge, deduplicating by id
            for article in graph.articles:
                if article.id not in seen_ids:
                    # Append chunk_id to link back to ChromaDB
                    if chunk.chunk_id not in article.chunk_ids:
                        article.chunk_ids.append(chunk.chunk_id)
                    merged.articles.append(article)
                    seen_ids.add(article.id)

            for obligation in graph.obligations:
                if obligation.id not in seen_ids:
                    merged.obligations.append(obligation)
                    seen_ids.add(obligation.id)

            for right in graph.rights:
                if right.id not in seen_ids:
                    merged.rights.append(right)
                    seen_ids.add(right.id)

            for party in graph.parties:
                if party.id not in seen_ids:
                    merged.parties.append(party)
                    seen_ids.add(party.id)

            for concept in graph.concepts:
                if concept.id not in seen_ids:
                    merged.concepts.append(concept)
                    seen_ids.add(concept.id)

            for rel in graph.relationships:
                if rel not in merged.relationships:
                    merged.relationships.append(rel)

        return merged

    # ─── Parsing helpers ──────────────────────────────────────────────────────

    def _parse_extracted_data(
        self, data: dict[str, Any], chunk: RegulationChunk
    ) -> ExtractedGraph:
        graph = ExtractedGraph()

        # Articles
        for a in data.get("articles", []):
            try:
                graph.articles.append(ArticleNode(
                    id=a["id"],
                    regulation=a.get("regulation", chunk.regulation),
                    number=a.get("number", chunk.article_number),
                    title=a.get("title", chunk.article_title),
                    text=a.get("text", chunk.text[:300]),
                    chunk_ids=[chunk.chunk_id],
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed article: {e}")

        # Obligations
        for o in data.get("obligations", []):
            try:
                severity_str = o.get("severity", "mandatory").lower()
                severity = ObligationSeverity(severity_str) if severity_str in [s.value for s in ObligationSeverity] else ObligationSeverity.MANDATORY
                graph.obligations.append(ObligationNode(
                    id=o["id"],
                    description=o["description"],
                    regulation=o.get("regulation", chunk.regulation),
                    article_ref=o.get("article_ref", ""),
                    severity=severity,
                    conditions=o.get("conditions"),
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed obligation: {e}")

        # Rights
        for r in data.get("rights", []):
            try:
                graph.rights.append(RightNode(
                    id=r["id"],
                    name=r["name"],
                    description=r["description"],
                    regulation=r.get("regulation", chunk.regulation),
                    article_ref=r.get("article_ref", ""),
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed right: {e}")

        # Parties
        for p in data.get("parties", []):
            try:
                pt_str = p.get("party_type", "ThirdParty")
                try:
                    pt = PartyType(pt_str)
                except ValueError:
                    pt = PartyType.THIRD_PARTY
                graph.parties.append(PartyNode(
                    id=p["id"],
                    name=p["name"],
                    party_type=pt,
                    description=p.get("description"),
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed party: {e}")

        # Concepts
        for c in data.get("concepts", []):
            try:
                graph.concepts.append(ConceptNode(
                    id=c["id"],
                    name=c["name"],
                    definition=c["definition"],
                    regulation=c.get("regulation", chunk.regulation),
                    synonyms=c.get("synonyms", []),
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed concept: {e}")

        # Relationships
        valid_rel_types = {r.value for r in RelationshipType}
        for rel in data.get("relationships", []):
            try:
                rel_type_str = rel.get("relationship_type", "").upper()
                if rel_type_str not in valid_rel_types:
                    continue
                graph.relationships.append((
                    rel["source_id"],
                    RelationshipType(rel_type_str),
                    rel["target_id"],
                ))
            except (KeyError, TypeError) as e:
                logger.debug(f"Skipping malformed relationship: {e}")

        return graph

    @staticmethod
    def _clean_json(text: str) -> str:
        """Strip markdown code fences and whitespace from LLM JSON output."""
        text = text.strip()
        # Remove ```json ... ``` fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()
