"""
rag/retriever.py - STEP 5: question  ->  the most relevant chunks.

Two stages (this is how most production systems work):
  1. FAST SEARCH (vectors): pick ~30 candidates. Cheap, but it compares the question and each
     chunk SEPARATELY, so generic intro chunks full of the word "RAG" can outrank the exact answer.
  2. RERANK (optional cross-encoder): reads question + chunk TOGETHER and re-scores them.
     Slower, but much better at judging "does this chunk really answer the question?".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints, so this file stays light to import
    from rag.embeddings import EmbeddingManager
    from rag.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result. A dataclass is cleaner than a dict: fewer typos like r["conent"]."""
    chunk_id: str
    text: str
    source: str                          # file name
    page: int                            # page number (starts at 1)
    score: float                         # final score used for the threshold (cosine, or reranker score)
    rank: int                            # 1 = best match
    vector_score: float | None = None    # the original cosine score, kept for debugging


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Retriever:
    def __init__(self, embedder: "EmbeddingManager", store: "VectorStore",
                 top_k: int, min_score: float, candidate_k: int = 30):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score            # threshold for COSINE scores (reranker is off)
        self.rerank_min_score = 0.05          # threshold for RERANKER scores (tune with the sweep!)
        self.candidate_k = candidate_k        # how many candidates the reranker gets to look at
        self.reranker = None                  # off by default

    # ---------- switch the reranker on/off ----------
    def enable_reranker(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                        min_score: float | None = None) -> None:
        # Imported here so you only need it when you use it. The model (~90 MB) downloads once.
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(model_name)
        if min_score is not None:
            self.rerank_min_score = min_score

    def disable_reranker(self) -> None:
        self.reranker = None

    # ---------- main function ----------
    def retrieve(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        if min_score is None:
            min_score = self.rerank_min_score if self.reranker else self.min_score

        total = self.store.count()
        if not query.strip() or total == 0:
            return []

        # Stage 1: vector search. With a reranker we fetch MORE candidates (e.g. 30), so the
        # right chunk has a chance to be in the list even if it was ranked #14 by the vectors.
        n_candidates = max(top_k, self.candidate_k) if self.reranker else top_k
        hits = self.store.search(self.embedder.embed_query(query), n_results=min(n_candidates, total))

        candidates, seen_texts = [], set()
        for hit in hits:
            text_key = hit["text"].strip()
            if text_key in seen_texts:
                continue                      # skip exact duplicate text
            seen_texts.add(text_key)
            # cosine similarity. Valid ONLY because the collection uses cosine distance.
            candidates.append((hit, 1.0 - hit["distance"]))
        if not candidates:
            return []

        # Stage 2 (optional): rerank
        if self.reranker:
            raw = self.reranker.predict([(query, hit["text"]) for hit, _ in candidates], show_progress_bar=False)
            scores = [float(s) for s in raw]
            if min(scores) < 0 or max(scores) > 1:      # raw logits -> squeeze into 0..1
                scores = [_sigmoid(s) for s in scores]
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            ranked = [(candidates[i][0], candidates[i][1], scores[i]) for i in order]
        else:
            ranked = [(hit, vec, vec) for hit, vec in candidates]

        results: list[RetrievedChunk] = []
        for hit, vector_score, score in ranked:
            if score < min_score:
                continue                      # too weak: probably not about our question
            results.append(RetrievedChunk(
                chunk_id=hit["id"], text=hit["text"],
                source=hit["metadata"]["source"], page=int(hit["metadata"]["page"]),
                score=score, rank=len(results) + 1, vector_score=vector_score,
            ))
            if len(results) == top_k:
                break
        return results
