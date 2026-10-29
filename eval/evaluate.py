"""
eval/evaluate.py - measure how good your retrieval is, with NUMBERS instead of "it seems fine".

Run (from the project folder, after indexing your PDF):
    python -m eval.evaluate                 # uses config.py defaults
    python -m eval.evaluate --sweep         # tries many thresholds to help you choose min_score

It checks retrieval only, so it is free (no LLM key needed) and takes seconds.

Metrics
  Hit@k    : for questions that HAVE an answer, how often is a chunk from a correct page in the top k?
  MRR      : like Hit@k but rewards a higher position. 1.0 = correct chunk always ranked first.
  Rejected : for questions that are NOT in the document, how often do we correctly return nothing?
             (a good "I don't know" behaviour)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import get_settings
from rag.pipeline import RAGPipeline


def first_hit_rank(chunks, question: dict) -> int | None:
    """Position (1 = best) of the first chunk that comes from a correct page, else None."""
    for position, chunk in enumerate(chunks, start=1):
        right_page = chunk.page in question["pages"]
        right_file = (not question.get("source")) or chunk.source == question["source"]
        if right_page and right_file:
            return position
    return None


def evaluate(pipeline: RAGPipeline, questions: list[dict], k: int, thresholds: list[float]):
    # Search ONCE with no threshold (-1 keeps everything), then apply each threshold in memory.
    # That makes the sweep fast.
    cache = {q["question"]: pipeline.retriever.retrieve(q["question"], top_k=k, min_score=-1.0)
             for q in questions}

    rows, misses = [], {}
    for t in thresholds:
        hits, rr_sum, n_ans, rejected, n_unans = 0, 0.0, 0, 0, 0
        for q in questions:
            kept = [c for c in cache[q["question"]] if c.score >= t]
            if q.get("answerable", True):
                n_ans += 1
                rank = first_hit_rank(kept, q)
                if rank:
                    hits += 1
                    rr_sum += 1.0 / rank
                else:
                    misses.setdefault(t, []).append(q["question"])
            else:
                n_unans += 1
                rejected += int(len(kept) == 0)
        rows.append((t, hits / max(n_ans, 1), rr_sum / max(n_ans, 1), rejected / max(n_unans, 1)))
    return rows, misses


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--min-score", type=float,default=settings.rerank_min_score if settings.use_reranker else settings.min_score)
    parser.add_argument("--sweep", action="store_true", help="try many thresholds")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    pipeline = RAGPipeline(settings)
    if pipeline.store.count() == 0:
        raise SystemExit("The index is empty. Run:  python -m scripts.ingest data/pdfs")

    thresholds =   [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9] if args.sweep else [args.min_score]
    rows, misses = evaluate(pipeline, questions, args.k, thresholds)

    # Printed as a markdown table, so you can paste it straight into your README.
    print(f"\nchunks in index: {pipeline.store.count()} | k={args.k} | questions: {len(questions)}\n")
    print(f"| min_score | Hit@{args.k} | MRR | Rejected (out-of-doc) |")
    print("|---|---|---|---|")
    for t, hit, mrr, rej in rows:
        print(f"| {t:.2f} | {hit:.0%} | {mrr:.2f} | {rej:.0%} |")

    shown = misses.get(thresholds[-1] if not args.sweep else args.min_score, [])
    if shown:
        print("\nQuestions that missed (look at these to improve chunking/embeddings):")
        for question in shown:
            print(" -", question)


if __name__ == "__main__":
    main()
