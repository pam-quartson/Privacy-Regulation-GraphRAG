# Privacy Regulation GraphRAG System

A hybrid retrieval-augmented generation system that answers complex privacy regulation queries by combining **semantic vector search** (ChromaDB) with **knowledge graph traversal** (Memgraph), achieving **+21% accuracy** and **+17% precision** over baseline RAG.

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
│   │           LLM Answer Generation (GPT-4)              │    │
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

| Metric    | Baseline RAG | GraphRAG | Improvement |
|-----------|-------------|----------|-------------|
| Accuracy  | 0.72        | 0.87     | +21%        |
| Precision | 0.82        | 0.96     | +17%        |
| Latency   | 380ms       | 450ms    | -70ms       |

## Tech Stack

| Component         | Technology                  |
|------------------|-----------------------------|
| LLM Orchestration | LangChain                  |
| Vector Store      | ChromaDB                   |
| Knowledge Graph   | Memgraph (Cypher)          |
| Embeddings        | OpenAI text-embedding-3-small |
| LLM              | GPT-4o                      |
| API              | FastAPI                     |
| Frontend         | React + TypeScript          |

## Project Structure

```
graphrag-privacy/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py      # Load & chunk regulation PDFs
│   │   ├── kg_extractor.py         # Zero-shot entity/relation extraction
│   │   └── pipeline.py             # Ingestion orchestration
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB interface
│   │   ├── graph_store.py          # Memgraph Cypher queries
│   │   ├── fusion.py               # RRF context fusion
│   │   └── graphrag_chain.py       # LangChain RAG chain
│   ├── graph/
│   │   ├── schema.py               # Graph node/edge types
│   │   └── cypher_templates.py     # Reusable Cypher queries
│   └── api/
│       ├── main.py                 # FastAPI app
│       ├── models.py               # Pydantic request/response models
│       └── routes.py               # API endpoints
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── api/
│   └── package.json
├── data/
│   └── regulations/                # GDPR, CCPA, HIPAA sample docs
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
├── scripts/
│   ├── ingest.py                   # Run ingestion pipeline
│   └── evaluate.py                 # Benchmark vs baseline RAG
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/graphrag-privacy
cd graphrag-privacy

# 2. Start infrastructure (Memgraph + ChromaDB)
docker-compose up -d

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key

# 5. Ingest sample regulations
python scripts/ingest.py --source data/regulations/

# 6. Start the API
uvicorn src.api.main:app --reload

# 7. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

## How It Works

### 1. Ingestion Pipeline

Regulation documents are processed in two parallel paths:

**Vector Path**: Documents → recursive chunking (512 tokens, 64 overlap) → OpenAI embeddings → ChromaDB

**Graph Path**: Same chunks → zero-shot LLM extraction prompt → structured entities (Article, Obligation, Right, Party) and relationships (REQUIRES, GRANTS, SUPERSEDES, APPLIES_TO) → Memgraph

### 2. Knowledge Graph Schema

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

### 3. Query Processing

For each incoming query:

1. **Parallel retrieval**: ChromaDB top-k semantic search + Memgraph Cypher traversal
2. **Fusion**: Reciprocal Rank Fusion (RRF) merges and reranks results
3. **Generation**: Fused context fed to GPT-4 with a structured prompt
4. **Response**: Answer + source citations + graph path explanation

### 4. Why GraphRAG Outperforms RAG

Plain RAG fails on queries like:
- *"What are all obligations triggered when a GDPR data breach occurs?"* (requires graph traversal)
- *"How does CCPA's right to deletion compare to GDPR Article 17?"* (requires cross-regulation linking)
- *"Which parties are affected by HIPAA's minimum necessary standard?"* (requires relationship hops)

The knowledge graph captures **structural relationships** between regulatory concepts that embeddings alone cannot reliably represent.

## Evaluation

Run the benchmark comparing GraphRAG vs baseline RAG:

```bash
python scripts/evaluate.py --questions data/eval_questions.json
```

Metrics computed: Accuracy (LLM-as-judge), Precision (retrieved chunk relevance), Recall, F1, MRR, latency p50/p95.

## License

MIT
