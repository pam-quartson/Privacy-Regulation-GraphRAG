"""
scripts/evaluate.py

Benchmarks GraphRAG vs baseline RAG (vector-only) across:
  - Accuracy  (LLM-as-judge: 0 or 1 per question)
  - Precision (fraction of retrieved chunks that are relevant)
  - Recall    (fraction of relevant chunks that are retrieved)
  - MRR       (Mean Reciprocal Rank)
  - Latency   (p50, p95)

Usage:
  python scripts/evaluate.py --questions data/eval_questions.json

Eval question format (data/eval_questions.json):
  [
    {
      "question": "What must a GDPR data controller do after a breach?",
      "ground_truth": "Notify supervisory authority within 72 hours...",
      "relevant_articles": ["gdpr_art_33", "gdpr_art_34"],
      "regulation": "gdpr"
    },
    ...
  ]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval.vector_store import VectorStore
from src.retrieval.graph_store import GraphStore
from src.retrieval.graphrag_chain import GraphRAGChain


# ─── LLM Judge ────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator for question-answering systems.
Given a question, a ground truth answer, and a system answer, rate the system answer
on a scale of 0.0 to 1.0 for accuracy.

1.0 = Fully correct, covers all key points
0.7 = Mostly correct, minor gaps
0.4 = Partially correct, significant gaps
0.1 = Mostly incorrect
0.0 = Completely wrong or irrelevant

Return ONLY a JSON object: {"score": <float>, "reasoning": "<brief explanation>"}"""

JUDGE_PROMPT = """Question: {question}
Ground Truth: {ground_truth}
System Answer: {answer}

Rate the system answer (return only JSON):"""


def llm_judge_score(
    question: str, ground_truth: str, answer: str, llm: ChatOpenAI
) -> tuple[float, str]:
    """Use GPT-4o to judge the quality of an answer (0.0 - 1.0)."""
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )),
    ]
    response = llm.invoke(messages)
    try:
        data = json.loads(response.content.strip())
        return float(data.get("score", 0.0)), data.get("reasoning", "")
    except Exception:
        return 0.0, "Parse error"


# ─── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question: str
    system: str  # "graphrag" or "baseline_rag"
    accuracy: float
    precision: float
    latency_ms: float
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    relevant_articles: list[str] = field(default_factory=list)
    judge_reasoning: str = ""


def compute_precision(retrieved_chunk_ids: list[str], relevant_articles: list[str]) -> float:
    """Fraction of retrieved chunks that belong to relevant articles."""
    if not retrieved_chunk_ids:
        return 0.0
    relevant_hits = sum(
        1 for cid in retrieved_chunk_ids
        if any(art in cid for art in relevant_articles)
    )
    return relevant_hits / len(retrieved_chunk_ids)


def compute_mrr(results_list: list[EvalResult]) -> float:
    """Mean Reciprocal Rank across all queries."""
    reciprocal_ranks = []
    for r in results_list:
        for i, cid in enumerate(r.retrieved_chunk_ids, start=1):
            if any(art in cid for art in r.relevant_articles):
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


# ─── Baseline RAG (vector only) ───────────────────────────────────────────────

class BaselineRAGChain:
    """Standard RAG: vector search only, no graph component."""

    def __init__(self, vector_store: VectorStore, model_name: str = "gpt-4o"):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)

    def query(self, question: str) -> dict:
        start = time.perf_counter()
        docs = self.vector_store.similarity_search(question, k=5)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"Answer based on the following regulatory context:\n\n{context}\n\nQuestion: {question}"
        response = self.llm.invoke([HumanMessage(content=prompt)])

        latency_ms = (time.perf_counter() - start) * 1000
        chunk_ids = [d.metadata.get("chunk_id", "") for d in docs]
        return {
            "answer": response.content,
            "chunk_ids": chunk_ids,
            "latency_ms": latency_ms,
        }


# ─── Evaluator ────────────────────────────────────────────────────────────────

def run_evaluation(questions_path: Path) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    print("Initializing systems...")
    vector_store = VectorStore()
    graph_store = GraphStore()
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    graphrag = GraphRAGChain(vector_store=vector_store, graph_store=graph_store)
    baseline = BaselineRAGChain(vector_store=vector_store)

    with open(questions_path) as f:
        eval_questions = json.load(f)

    print(f"\nRunning evaluation on {len(eval_questions)} questions...\n")

    graphrag_results: list[EvalResult] = []
    baseline_results: list[EvalResult] = []

    for i, q in enumerate(eval_questions, start=1):
        question = q["question"]
        ground_truth = q["ground_truth"]
        relevant_articles = q.get("relevant_articles", [])
        print(f"[{i}/{len(eval_questions)}] {question[:60]}...")

        # GraphRAG
        gr_response = graphrag.query(question)
        gr_acc, gr_reasoning = llm_judge_score(question, ground_truth, gr_response.answer, judge_llm)
        gr_chunk_ids = [r.chunk_id for r in gr_response.fused_results if r.chunk_id]
        gr_precision = compute_precision(gr_chunk_ids, relevant_articles)
        graphrag_results.append(EvalResult(
            question=question,
            system="graphrag",
            accuracy=gr_acc,
            precision=gr_precision,
            latency_ms=gr_response.latency_ms,
            retrieved_chunk_ids=gr_chunk_ids,
            relevant_articles=relevant_articles,
            judge_reasoning=gr_reasoning,
        ))

        # Baseline RAG
        bl_response = baseline.query(question)
        bl_acc, _ = llm_judge_score(question, ground_truth, bl_response["answer"], judge_llm)
        bl_precision = compute_precision(bl_response["chunk_ids"], relevant_articles)
        baseline_results.append(EvalResult(
            question=question,
            system="baseline_rag",
            accuracy=bl_acc,
            precision=bl_precision,
            latency_ms=bl_response["latency_ms"],
            retrieved_chunk_ids=bl_response["chunk_ids"],
            relevant_articles=relevant_articles,
        ))

        print(f"  GraphRAG accuracy={gr_acc:.2f} precision={gr_precision:.2f} | "
              f"Baseline accuracy={bl_acc:.2f} precision={bl_precision:.2f}")

    # ── Print Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    for label, results in [("GraphRAG", graphrag_results), ("Baseline RAG", baseline_results)]:
        accuracies = [r.accuracy for r in results]
        precisions = [r.precision for r in results]
        latencies = [r.latency_ms for r in results]
        mrr = compute_mrr(results)

        print(f"\n{label}:")
        print(f"  Accuracy  (mean): {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"  Precision (mean): {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        print(f"  MRR:              {mrr:.3f}")
        print(f"  Latency p50:      {np.percentile(latencies, 50):.0f}ms")
        print(f"  Latency p95:      {np.percentile(latencies, 95):.0f}ms")

    # ── Deltas ────────────────────────────────────────────────────────────────
    gr_acc = np.mean([r.accuracy for r in graphrag_results])
    bl_acc = np.mean([r.accuracy for r in baseline_results])
    gr_prec = np.mean([r.precision for r in graphrag_results])
    bl_prec = np.mean([r.precision for r in baseline_results])

    print(f"\n{'='*60}")
    print("IMPROVEMENT (GraphRAG over Baseline RAG)")
    print(f"  Accuracy:  {gr_acc - bl_acc:+.3f} ({(gr_acc/bl_acc - 1)*100:+.1f}%)")
    print(f"  Precision: {gr_prec - bl_prec:+.3f} ({(gr_prec/bl_prec - 1)*100:+.1f}%)")

    graph_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG vs Baseline RAG")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/eval_questions.json"),
        help="Path to evaluation questions JSON file",
    )
    args = parser.parse_args()
    run_evaluation(args.questions)
