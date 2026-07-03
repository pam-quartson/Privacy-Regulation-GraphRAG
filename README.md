# Privacy Regulation GraphRAG System

A hybrid retrieval-augmented generation system that answers complex privacy regulation queries by combining **semantic vector search** (ChromaDB) with **knowledge graph traversal** (Memgraph). Runs entirely on local, self-hosted infrastructure — no external LLM API required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      Query Interface                          │
│                   (FastAPI REST + React UI)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   GraphRAG Query Pipeline                      │
│                                                               │
│   ┌─────────────────┐         ┌─────────────────────────┐    │
│   │  Vector Search  │         │   Graph Traversal       │    │
│   │   (ChromaDB)    │         │    (Memgraph/Cypher)    │    │
│   │                 │         │                         │    │
│   │ - Semantic sim. │         │ - Entity relationships  │    │
│   │ - Dense embeds  │         │ - Multi-hop reasoning   │    │
│   │ - Top-k chunks  │         │ - Regulatory links      │    │
│   └────────┬────────┘         └───────────┬─────────────┘    │
│            │                              │                   │
│   ┌────────▼──────────────────────────────▼─────────────┐    │
│   │              Context Fusion & Reranking              │    │
│   └────────────────────────┬─────────────────────────────┘    │
│                            │                                   │
│   ┌────────────────────────▼─────────────────────────────┐    │
│   │        LLM Answer Generation (Local via Ollama)      │    │
│   └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                    Ingestion Pipeline                          │
│                                                               │
│   Regulation Docs → Chunking → Embeddings → ChromaDB         │
│                             → KG Extraction → Memgraph        │
└──────────────────────────────────────────────────────────────┘
```

## Key Results

Benchmark results are pending a full evaluation run — `scripts/evaluate.py` compares GraphRAG against a vector-only baseline RAG using an LLM-as-judge. Run it yourself after ingestion completes:

```bash
python scripts/evaluate.py --questions data/eval_questions.json
```

## Tech Stack

| Component         | Technology                             |
|--------------------|-----------------------------------------|
| LLM Orchestration | LangChain                              |
| Vector Store      | ChromaDB                               |
| Knowledge Graph   | Memgraph (Cypher)                      |
| Embeddings        | sentence-transformers (local, `all-MiniLM-L6-v2`) |
| LLM               | Ollama (local, `llama3.1:8b`)          |
| API               | FastAPI                                |
| Frontend          | React + TypeScript (Vite)               |

Everything runs locally except the source regulation text fetch — no OpenAI/Anthropic API key is required to ingest, query, or evaluate.

## Project Structure

```
graphrag-privacy/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py      # Load & chunk regulation text into articles/sections
│   │   ├── kg_extractor.py         # Zero-shot entity/relation extraction (local LLM)
│   │   └── pipeline.py             # Ingestion orchestration
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB interface (local embeddings)
│   │   ├── graph_store.py          # Memgraph Cypher queries
│   │   ├── fusion.py               # RRF context fusion
│   │   └── graphrag_chain.py       # LangChain RAG chain (local LLM)
│   ├── graph/
│   │   ├── schema.py               # Graph node/edge types
│   │   └── cypher_templates.py     # Reusable Cypher queries
│   └── api/
│       ├── main.py                 # FastAPI app + all routes
│       └── models.py               # Pydantic request/response models
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── package.json
├── data/
│   └── regulations/                # GDPR/CCPA/HIPAA text (populated by fetch_regulations.py)
├── tests/
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── scripts/
│   ├── fetch_regulations.py        # One-time fetch of GDPR/CCPA/HIPAA source text
│   ├── ingest.py                   # Run ingestion pipeline
│   └── evaluate.py                 # Benchmark vs baseline RAG
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- [Ollama](https://ollama.com) (local LLM runtime — no API key needed)
- ~5GB disk space for the local LLM model, ~2GB for embeddings/ML libraries

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/graphrag-privacy
cd graphrag-privacy

# 2. Start infrastructure (Memgraph + ChromaDB)
docker-compose up -d

# 3. Install Ollama and pull the local LLM
brew install ollama          # or see https://ollama.com/download
brew services start ollama
ollama pull llama3.1:8b

# 4. Create a virtualenv and install Python dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Defaults work out of the box (local LLM, local embeddings, memgraph/memgraph creds
# matching docker-compose.yml) - no API keys required.

# 6. Fetch regulation source text (GDPR, CCPA, HIPAA from official sources)
.venv/bin/python scripts/fetch_regulations.py

# 7. Ingest into ChromaDB + Memgraph
# Note: KG extraction runs one LLM call per chunk (~1,900 chunks total).
# On an 8B local model this takes several hours - let it run in the background.
.venv/bin/python scripts/ingest.py --source data/regulations/

# 8. Start the API
.venv/bin/uvicorn src.api.main:app --reload

# 9. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### Environment variables

See `.env.example` for the full list. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Local LLM for KG extraction, answer generation, and eval judging |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model for ChromaDB |
| `MEMGRAPH_USERNAME` / `MEMGRAPH_PASSWORD` | `memgraph` / `memgraph` | Must match `docker-compose.yml` |
| `CHROMA_HOST` / `CHROMA_PORT` | `localhost` / `8001` | ChromaDB connection |

## How It Works

### 1. Fetching Source Text

`scripts/fetch_regulations.py` is a one-time fetcher (not a scheduled job) — GDPR/CCPA/HIPAA text changes on the order of years, not days. It pulls:
- **GDPR**: all 99 articles from [gdpr-info.eu](https://gdpr-info.eu)
- **CCPA**: California Civil Code Title 1.81.5 from [leginfo.legislature.ca.gov](https://leginfo.legislature.ca.gov)
- **HIPAA**: 45 CFR Part 164 from the official [eCFR API](https://www.ecfr.gov)

Re-run it (optionally with `--only gdpr,hipaa`) if a regulation is amended and you want to refresh the corpus.

### 2. Ingestion Pipeline

Regulation documents are processed in two parallel paths:

**Vector Path**: Documents → article/section-aware chunking (512 tokens, 64 overlap) → local sentence-transformers embeddings → ChromaDB

**Graph Path**: Same chunks → zero-shot LLM extraction prompt (local Ollama model, constrained to JSON output) → structured entities (Article, Obligation, Right, Party) and relationships (REQUIRES, GRANTS, SUPERSEDES, APPLIES_TO) → Memgraph

### 3. Knowledge Graph Schema

```cypher
// Nodes
(:Article {id, title, regulation, number, text})
(:Obligation {id, description, regulation, severity})
(:Right {id, name, description, regulation})
(:Party {id, name, type})  // Controller, Processor, DataSubject

// Relationships
(a:Article)-[:REQUIRES]->(o:Obligation)
(a:Article)-[:GRANTS]->(r:Right)
(a:Article)-[:SUPERSEDES]->(b:Article)
(o:Obligation)-[:APPLIES_TO]->(p:Party)
(r:Right)-[:EXERCISED_AGAINST]->(p:Party)
```

### 4. Query Processing

For each incoming query:

1. **Parallel retrieval**: ChromaDB top-k semantic search + Memgraph Cypher traversal
2. **Fusion**: Reciprocal Rank Fusion (RRF) merges and reranks results
3. **Generation**: Fused context fed to the local LLM with a structured prompt
4. **Response**: Answer + source citations + graph path explanation

### 5. Why GraphRAG Outperforms RAG

Plain RAG fails on queries like:
- *"What are all obligations triggered when a GDPR data breach occurs?"* (requires graph traversal)
- *"How does CCPA's right to deletion compare to GDPR Article 17?"* (requires cross-regulation linking)
- *"Which parties are affected by HIPAA's minimum necessary standard?"* (requires relationship hops)

The knowledge graph captures **structural relationships** between regulatory concepts that embeddings alone cannot reliably represent.

## Evaluation

Run the benchmark comparing GraphRAG vs baseline RAG:

```bash
.venv/bin/python scripts/evaluate.py --questions data/eval_questions.json
```

Metrics computed: Accuracy (LLM-as-judge), Precision (retrieved chunk relevance), MRR, latency p50/p95. The judge and both systems' answer generation run on the same local Ollama model, so results reflect the *relative* GraphRAG-vs-baseline gap rather than absolute quality against a frontier model.

## Running Tests

```bash
.venv/bin/pytest tests/ -v
```

## License

MIT
