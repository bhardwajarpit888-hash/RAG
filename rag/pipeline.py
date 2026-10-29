"""
rag/pipeline.py - connects all the steps. app.py, the CLI and the evaluation all use THIS class,
so the logic exists in exactly one place.

  ingest:  file -> pages -> chunks -> vectors -> vector store
  ask:     question -> vector -> top chunks -> LLM -> answer + sources
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import Settings, get_settings
from rag.chunker import split_documents
from rag.embeddings import EmbeddingManager
from rag.generator import REFUSAL, Answer, AnswerGenerator
from rag.llm import build_llm
from rag.loader import load_file
from rag.retriever import Retriever
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.embedder = EmbeddingManager(s.embedding_model)
        self.store = VectorStore(s.vector_dir, s.collection_name)
        self.retriever = Retriever(self.embedder, self.store, s.top_k, s.min_score)
        if s.use_reranker:
            self.retriever.enable_reranker(s.reranker_model, s.rerank_min_score)
        self._generator: AnswerGenerator | None = None   # built only when first needed

    @property
    def generator(self) -> AnswerGenerator:
        # Lazy: you can index and run the evaluation even before you add an LLM key.
        if self._generator is None:
            self._generator = AnswerGenerator(build_llm(self.settings))
        return self._generator

    # ---------- indexing ----------
    def ingest_file(self, path: str | Path) -> int:
        """Index one file. Returns the number of chunks stored."""
        path = Path(path)
        docs = load_file(path)
        if not docs:
            raise ValueError(f"No text found in {path.name}. Is it a scanned PDF? (OCR is not supported yet)")

        chunks = split_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        vectors = self.embedder.embed_documents([c.page_content for c in chunks])
        self.store.replace_source(path.name, chunks, vectors)
        return len(chunks)

    def ingest_folder(self, folder: str | Path) -> dict[str, int]:
        """Index every .pdf/.txt/.md file in a folder. Returns {file name: chunk count}."""
        report: dict[str, int] = {}
        for file in sorted(Path(folder).iterdir()):
            if file.suffix.lower() in {".pdf", ".txt", ".md"}:
                try:
                    report[file.name] = self.ingest_file(file)
                except Exception as exc:   # one bad file must not stop the others
                    logger.error("Could not index %s: %s", file.name, exc)
        return report

    # ---------- asking ----------
    def ask(self, question: str, top_k: int | None = None, min_score: float | None = None) -> Answer:
        chunks = self.retriever.retrieve(question, top_k=top_k, min_score=min_score)
        if not chunks:
            return Answer(REFUSAL, [], grounded=False)   # skip the LLM completely
        return self.generator.answer(question, chunks)
