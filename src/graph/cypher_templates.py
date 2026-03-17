"""
graph/cypher_templates.py

Reusable Cypher query templates for graph traversal in Memgraph.
Each template is designed to answer a class of privacy regulation query.
"""

from string import Template


# ─── Schema Setup ─────────────────────────────────────────────────────────────

CREATE_ARTICLE = """
MERGE (a:Article {id: $id})
SET a.regulation = $regulation,
    a.number = $number,
    a.title = $title,
    a.text = $text,
    a.chunk_ids = $chunk_ids
"""

CREATE_OBLIGATION = """
MERGE (o:Obligation {id: $id})
SET o.description = $description,
    o.regulation = $regulation,
    o.article_ref = $article_ref,
    o.severity = $severity
"""

CREATE_RIGHT = """
MERGE (r:Right {id: $id})
SET r.name = $name,
    r.description = $description,
    r.regulation = $regulation,
    r.article_ref = $article_ref
"""

CREATE_PARTY = """
MERGE (p:Party {id: $id})
SET p.name = $name,
    p.party_type = $party_type,
    p.description = $description
"""

CREATE_CONCEPT = """
MERGE (c:Concept {id: $id})
SET c.name = $name,
    c.definition = $definition,
    c.regulation = $regulation,
    c.synonyms = $synonyms
"""

CREATE_REGULATION = """
MERGE (r:Regulation {id: $id})
SET r.name = $name,
    r.abbreviation = $abbreviation,
    r.jurisdiction = $jurisdiction
"""

# Generic relationship creation
CREATE_RELATIONSHIP = """
MATCH (a {{id: $source_id}})
MATCH (b {{id: $target_id}})
MERGE (a)-[:{rel_type}]->(b)
"""


# ─── Retrieval Queries ────────────────────────────────────────────────────────

# 1. Find all obligations for a given article
GET_ARTICLE_OBLIGATIONS = """
MATCH (a:Article {id: $article_id})-[:REQUIRES]->(o:Obligation)
RETURN a.title AS article_title,
       a.number AS article_number,
       a.regulation AS regulation,
       collect({
           id: o.id,
           description: o.description,
           severity: o.severity
       }) AS obligations
"""

# 2. Find all rights granted to a party type
GET_RIGHTS_FOR_PARTY = """
MATCH (r:Right)-[:EXERCISED_AGAINST]->(p:Party {party_type: $party_type})
MATCH (a:Article)-[:GRANTS]->(r)
RETURN p.name AS party,
       collect({
           right_name: r.name,
           right_description: r.description,
           article: a.number,
           regulation: a.regulation
       }) AS rights
"""

# 3. Cross-regulation comparison (e.g. GDPR Art 17 vs CCPA right to delete)
CROSS_REGULATION_COMPARISON = """
MATCH (a1:Article {regulation: $reg1})-[:GRANTS|REQUIRES]->(n1)
MATCH (a2:Article {regulation: $reg2})-[:GRANTS|REQUIRES]->(n2)
WHERE (a1)-[:SUPERSEDES]->(a2) OR (a2)-[:SUPERSEDES]->(a1)
   OR toLower(n1.name) CONTAINS toLower($keyword)
   OR toLower(n1.description) CONTAINS toLower($keyword)
RETURN a1.number AS article_reg1,
       a1.title AS title_reg1,
       n1.description AS detail_reg1,
       a2.number AS article_reg2,
       a2.title AS title_reg2,
       n2.description AS detail_reg2
LIMIT 10
"""

# 4. Breach notification obligations (multi-hop traversal)
BREACH_NOTIFICATION_CHAIN = """
MATCH (c:Concept {name: $concept_name})<-[:RELATES_TO|DEFINED_IN*1..2]-(a:Article)
MATCH (a)-[:REQUIRES]->(o:Obligation)-[:APPLIES_TO]->(p:Party)
RETURN a.regulation AS regulation,
       a.number AS article,
       a.title AS title,
       o.description AS obligation,
       o.severity AS severity,
       p.name AS responsible_party
ORDER BY a.regulation, a.number
"""

# 5. Semantic entity search — find articles mentioning a concept by keyword
KEYWORD_ARTICLE_SEARCH = """
MATCH (a:Article)
WHERE toLower(a.title) CONTAINS toLower($keyword)
   OR toLower(a.text) CONTAINS toLower($keyword)
OPTIONAL MATCH (a)-[:REQUIRES]->(o:Obligation)
OPTIONAL MATCH (a)-[:GRANTS]->(r:Right)
RETURN a.id AS article_id,
       a.regulation AS regulation,
       a.number AS number,
       a.title AS title,
       a.chunk_ids AS chunk_ids,
       collect(DISTINCT o.description) AS obligations,
       collect(DISTINCT r.name) AS rights
LIMIT $limit
"""

# 6. Full neighborhood of an article (1-hop context)
ARTICLE_NEIGHBORHOOD = """
MATCH (a:Article {id: $article_id})
OPTIONAL MATCH (a)-[r1]->(n1)
OPTIONAL MATCH (n1)-[r2]->(n2)
RETURN a,
       collect(DISTINCT {
           rel_type: type(r1),
           node_label: labels(n1)[0],
           node_id: n1.id,
           node_name: coalesce(n1.title, n1.name, n1.description, n1.id)
       }) AS direct_neighbors,
       collect(DISTINCT {
           hop2_rel: type(r2),
           node_label: labels(n2)[0],
           node_id: n2.id,
           node_name: coalesce(n2.title, n2.name, n2.description, n2.id)
       }) AS second_neighbors
"""

# 7. Obligations that apply to a specific party
OBLIGATIONS_FOR_PARTY = """
MATCH (o:Obligation)-[:APPLIES_TO]->(p:Party)
WHERE toLower(p.name) CONTAINS toLower($party_name)
   OR toLower(p.party_type) CONTAINS toLower($party_name)
MATCH (a:Article)-[:REQUIRES]->(o)
RETURN p.name AS party,
       a.regulation AS regulation,
       a.number AS article,
       a.title AS article_title,
       o.description AS obligation,
       o.severity AS severity,
       a.chunk_ids AS chunk_ids
ORDER BY o.severity DESC, a.regulation
LIMIT $limit
"""

# 8. Find related regulations/articles by concept
CONCEPT_COVERAGE = """
MATCH (c:Concept)
WHERE toLower(c.name) CONTAINS toLower($concept)
   OR toLower(c.definition) CONTAINS toLower($concept)
MATCH (c)<-[:DEFINED_IN|RELATES_TO*1..2]-(a:Article)
RETURN c.name AS concept,
       collect(DISTINCT {
           regulation: a.regulation,
           article: a.number,
           title: a.title,
           article_id: a.id,
           chunk_ids: a.chunk_ids
       }) AS covered_in
"""

# 9. List all regulations in the graph
LIST_REGULATIONS = """
MATCH (r:Regulation)
RETURN r.id AS id, r.name AS name, r.abbreviation AS abbreviation,
       r.jurisdiction AS jurisdiction
ORDER BY r.abbreviation
"""

# 10. Graph statistics
GRAPH_STATS = """
MATCH (n)
RETURN labels(n)[0] AS node_type, count(n) AS count
ORDER BY count DESC
"""
