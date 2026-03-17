"""
graph/schema.py

Defines the knowledge graph node and relationship types for
the Privacy Regulation GraphRAG system.

Graph Model:
  Nodes:   Article, Obligation, Right, Party, Regulation, Concept
  Edges:   REQUIRES, GRANTS, SUPERSEDES, APPLIES_TO, RELATES_TO,
           EXERCISED_AGAINST, DEFINED_IN, PART_OF
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── Node Types ──────────────────────────────────────────────────────────────

class NodeLabel(str, Enum):
    ARTICLE     = "Article"
    OBLIGATION  = "Obligation"
    RIGHT       = "Right"
    PARTY       = "Party"
    REGULATION  = "Regulation"
    CONCEPT     = "Concept"


class RelationshipType(str, Enum):
    REQUIRES           = "REQUIRES"           # Article → Obligation
    GRANTS             = "GRANTS"             # Article → Right
    SUPERSEDES         = "SUPERSEDES"         # Article → Article (cross-regulation)
    APPLIES_TO         = "APPLIES_TO"         # Obligation → Party
    EXERCISED_AGAINST  = "EXERCISED_AGAINST"  # Right → Party
    RELATES_TO         = "RELATES_TO"         # Concept ↔ Concept
    DEFINED_IN         = "DEFINED_IN"         # Concept → Article
    PART_OF            = "PART_OF"            # Article → Regulation


class PartyType(str, Enum):
    CONTROLLER      = "Controller"
    PROCESSOR       = "Processor"
    DATA_SUBJECT    = "DataSubject"
    SUPERVISORY     = "SupervisoryAuthority"
    THIRD_PARTY     = "ThirdParty"


class ObligationSeverity(str, Enum):
    MANDATORY   = "mandatory"    # "must", "shall"
    CONDITIONAL = "conditional"  # "should", "where applicable"
    RECOMMENDED = "recommended"  # "may", "is encouraged"


# ─── Node Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class RegulationNode:
    id: str                     # e.g. "gdpr", "ccpa", "hipaa"
    name: str                   # e.g. "General Data Protection Regulation"
    abbreviation: str           # e.g. "GDPR"
    jurisdiction: str           # e.g. "EU", "California", "US Healthcare"
    effective_date: Optional[str] = None
    source_url: Optional[str] = None

    def to_cypher_props(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ArticleNode:
    id: str                     # e.g. "gdpr_art_17"
    regulation: str             # e.g. "gdpr"
    number: str                 # e.g. "17", "5(1)(a)"
    title: str                  # e.g. "Right to erasure"
    text: str                   # Full article text (truncated)
    chunk_ids: list[str] = field(default_factory=list)  # Links back to ChromaDB

    def to_cypher_props(self) -> dict:
        props = self.__dict__.copy()
        props["chunk_ids"] = ",".join(self.chunk_ids)
        return props


@dataclass
class ObligationNode:
    id: str                     # e.g. "gdpr_art_17_ob_1"
    description: str            # e.g. "Erase personal data without undue delay"
    regulation: str
    article_ref: str
    severity: ObligationSeverity = ObligationSeverity.MANDATORY
    conditions: Optional[str] = None  # When this obligation is triggered

    def to_cypher_props(self) -> dict:
        props = self.__dict__.copy()
        props["severity"] = self.severity.value
        return {k: v for k, v in props.items() if v is not None}


@dataclass
class RightNode:
    id: str                     # e.g. "gdpr_right_erasure"
    name: str                   # e.g. "Right to Erasure"
    description: str
    regulation: str
    article_ref: str
    conditions: Optional[str] = None

    def to_cypher_props(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class PartyNode:
    id: str                     # e.g. "party_controller"
    name: str                   # e.g. "Data Controller"
    party_type: PartyType
    description: Optional[str] = None

    def to_cypher_props(self) -> dict:
        props = self.__dict__.copy()
        props["party_type"] = self.party_type.value
        return {k: v for k, v in props.items() if v is not None}


@dataclass
class ConceptNode:
    id: str                     # e.g. "concept_personal_data"
    name: str                   # e.g. "Personal Data"
    definition: str
    regulation: str
    synonyms: list[str] = field(default_factory=list)

    def to_cypher_props(self) -> dict:
        props = self.__dict__.copy()
        props["synonyms"] = ",".join(self.synonyms)
        return props


# ─── Extracted Graph (output of KG extractor) ────────────────────────────────

@dataclass
class ExtractedGraph:
    """Structured output from the zero-shot KG extraction prompt."""
    articles: list[ArticleNode] = field(default_factory=list)
    obligations: list[ObligationNode] = field(default_factory=list)
    rights: list[RightNode] = field(default_factory=list)
    parties: list[PartyNode] = field(default_factory=list)
    concepts: list[ConceptNode] = field(default_factory=list)

    # Relationships as (source_id, relationship_type, target_id) tuples
    relationships: list[tuple[str, RelationshipType, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Extracted: {len(self.articles)} articles, "
            f"{len(self.obligations)} obligations, "
            f"{len(self.rights)} rights, "
            f"{len(self.parties)} parties, "
            f"{len(self.concepts)} concepts, "
            f"{len(self.relationships)} relationships"
        )


# ─── Cypher index creation ────────────────────────────────────────────────────

SCHEMA_INDEXES = [
    "CREATE INDEX ON :Article(id);",
    "CREATE INDEX ON :Article(regulation);",
    "CREATE INDEX ON :Obligation(id);",
    "CREATE INDEX ON :Right(id);",
    "CREATE INDEX ON :Party(id);",
    "CREATE INDEX ON :Regulation(id);",
    "CREATE INDEX ON :Concept(id);",
    "CREATE INDEX ON :Concept(name);",
]
