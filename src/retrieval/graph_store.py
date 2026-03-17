"""
retrieval/graph_store.py

Memgraph knowledge graph store interface.
Uses the neo4j Python driver (Memgraph is Bolt-protocol compatible).

Handles:
  - Graph ingestion (nodes + relationships)
  - Cypher-based retrieval for structured queries
  - Entity extraction from natural language queries (for graph routing)
"""

import logging
import re
from contextlib import contextmanager
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable

from src.graph.schema import (
    ExtractedGraph, ArticleNode, ObligationNode, RightNode,
    PartyNode, ConceptNode, RelationshipType, SCHEMA_INDEXES,
)
from src.graph import cypher_templates as Q

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Memgraph knowledge graph store.

    Uses the Neo4j Bolt driver for all operations.
    Memgraph is a drop-in replacement for Neo4j CE for our purposes.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "",
        password: str = "",
    ):
        uri = f"bolt://{host}:{port}"
        auth = (username, password) if username else None

        self._driver: Driver = GraphDatabase.driver(uri, auth=auth)
        logger.info(f"Connected to Memgraph at {uri}")
        self._ensure_indexes()

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def _session(self):
        with self._driver.session() as session:
            yield session

    # ─── Schema ───────────────────────────────────────────────────────────────

    def _ensure_indexes(self) -> None:
        """Create indexes if they don't exist."""
        with self._session() as session:
            for query in SCHEMA_INDEXES:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Index creation skipped (may already exist): {e}")

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def ingest_graph(self, graph: ExtractedGraph) -> None:
        """Ingest a full ExtractedGraph into Memgraph."""
        with self._session() as session:
            self._ingest_articles(session, graph.articles)
            self._ingest_obligations(session, graph.obligations)
            self._ingest_rights(session, graph.rights)
            self._ingest_parties(session, graph.parties)
            self._ingest_concepts(session, graph.concepts)
            self._ingest_relationships(session, graph.relationships)

        logger.info(f"Ingested graph: {graph.summary()}")

    def _ingest_articles(self, session: Session, articles: list[ArticleNode]) -> None:
        for a in articles:
            session.run(Q.CREATE_ARTICLE, **a.to_cypher_props())

    def _ingest_obligations(self, session: Session, obligations: list[ObligationNode]) -> None:
        for o in obligations:
            session.run(Q.CREATE_OBLIGATION, **o.to_cypher_props())

    def _ingest_rights(self, session: Session, rights: list[RightNode]) -> None:
        for r in rights:
            session.run(Q.CREATE_RIGHT, **r.to_cypher_props())

    def _ingest_parties(self, session: Session, parties: list[PartyNode]) -> None:
        for p in parties:
            session.run(Q.CREATE_PARTY, **p.to_cypher_props())

    def _ingest_concepts(self, session: Session, concepts: list[ConceptNode]) -> None:
        for c in concepts:
            session.run(Q.CREATE_CONCEPT, **c.to_cypher_props())

    def _ingest_relationships(
        self,
        session: Session,
        relationships: list[tuple[str, RelationshipType, str]],
    ) -> None:
        for source_id, rel_type, target_id in relationships:
            cypher = Q.CREATE_RELATIONSHIP.format(rel_type=rel_type.value)
            try:
                session.run(cypher, source_id=source_id, target_id=target_id)
            except Exception as e:
                logger.debug(f"Relationship creation failed {source_id}->{target_id}: {e}")

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def search_by_keyword(
        self, keyword: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find articles and their related entities matching a keyword."""
        with self._session() as session:
            result = session.run(Q.KEYWORD_ARTICLE_SEARCH, keyword=keyword, limit=limit)
            return [record.data() for record in result]

    def get_obligations_for_party(
        self, party_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get all obligations that apply to a given party type."""
        with self._session() as session:
            result = session.run(
                Q.OBLIGATIONS_FOR_PARTY, party_name=party_name, limit=limit
            )
            return [record.data() for record in result]

    def get_concept_coverage(self, concept: str) -> list[dict[str, Any]]:
        """Find all articles covering a given concept across regulations."""
        with self._session() as session:
            result = session.run(Q.CONCEPT_COVERAGE, concept=concept)
            return [record.data() for record in result]

    def get_breach_notification_chain(self, concept: str = "data breach") -> list[dict]:
        """Multi-hop traversal for breach notification obligations."""
        with self._session() as session:
            result = session.run(Q.BREACH_NOTIFICATION_CHAIN, concept_name=concept)
            return [record.data() for record in result]

    def get_cross_regulation_comparison(
        self, reg1: str, reg2: str, keyword: str
    ) -> list[dict]:
        """Compare how two regulations treat the same concept."""
        with self._session() as session:
            result = session.run(
                Q.CROSS_REGULATION_COMPARISON,
                reg1=reg1, reg2=reg2, keyword=keyword
            )
            return [record.data() for record in result]

    def get_article_neighborhood(self, article_id: str) -> list[dict]:
        """Get full 2-hop neighborhood of an article."""
        with self._session() as session:
            result = session.run(Q.ARTICLE_NEIGHBORHOOD, article_id=article_id)
            return [record.data() for record in result]

    def get_rights_for_party(self, party_type: str) -> list[dict]:
        """Get all rights exercisable by a party type."""
        with self._session() as session:
            result = session.run(Q.GET_RIGHTS_FOR_PARTY, party_type=party_type)
            return [record.data() for record in result]

    def query(self, cypher: str, **params) -> list[dict[str, Any]]:
        """Execute a raw Cypher query."""
        with self._session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def get_stats(self) -> dict[str, int]:
        """Return node counts by label."""
        with self._session() as session:
            result = session.run(Q.GRAPH_STATS)
            return {record["node_type"]: record["count"] for record in result}

    def is_healthy(self) -> bool:
        """Check if Memgraph is reachable."""
        try:
            with self._session() as session:
                session.run("RETURN 1")
            return True
        except ServiceUnavailable:
            return False

    # ─── Query routing helpers ─────────────────────────────────────────────────

    def retrieve_for_query(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Smart retrieval: routes a natural language query to the most
        appropriate Cypher template(s).

        Returns a flat list of result dicts with a 'source' key added.
        """
        results = []
        q_lower = query.lower()

        # Breach/incident queries → multi-hop chain
        if any(w in q_lower for w in ["breach", "incident", "notify", "notification"]):
            breach_results = self.get_breach_notification_chain()
            for r in breach_results:
                r["retrieval_strategy"] = "breach_chain"
            results.extend(breach_results)

        # Party-specific queries
        for party in ["controller", "processor", "data subject", "supervisor"]:
            if party in q_lower:
                party_results = self.get_obligations_for_party(party, limit=limit)
                for r in party_results:
                    r["retrieval_strategy"] = "party_obligations"
                results.extend(party_results)
                break

        # Cross-regulation comparison
        regs = []
        for reg in ["gdpr", "ccpa", "hipaa", "lgpd", "pdpa"]:
            if reg in q_lower:
                regs.append(reg)
        if len(regs) >= 2:
            # Extract keyword: remove regulation names from query
            keyword = re.sub(r"\b(gdpr|ccpa|hipaa|lgpd|pdpa)\b", "", q_lower).strip()
            keyword = keyword[:50] if keyword else "data"
            comp_results = self.get_cross_regulation_comparison(regs[0], regs[1], keyword)
            for r in comp_results:
                r["retrieval_strategy"] = "cross_regulation"
            results.extend(comp_results)

        # Concept coverage (fallback keyword search)
        if not results:
            # Extract meaningful keywords
            stopwords = {"what", "how", "does", "do", "the", "a", "an", "is", "are",
                        "and", "or", "for", "to", "of", "in", "under", "about"}
            words = [w for w in re.findall(r"\b\w{4,}\b", q_lower) if w not in stopwords]
            keyword = words[0] if words else query[:30]

            kw_results = self.search_by_keyword(keyword, limit=limit)
            for r in kw_results:
                r["retrieval_strategy"] = "keyword_search"
            results.extend(kw_results)

        return results[:limit * 2]  # Cap total
