"""
rag/vector_store.py - STEP 4: save vectors in a database (ChromaDB) and search them.

THIS FILE FIXES THREE BUGS FROM YOUR NOTEBOOK
1. Wrong distance type.
   Chroma's default is L2 distance, but your code did `score = 1 - distance`, which is only
   correct for COSINE distance. With L2 the scores came out too low (even negative), so good
   chunks were thrown away. That is why "what is hallucination?" returned 0 documents.
   Fix: we create the collection with cosine distance.
2. Duplicates.
   You used a random uuid as the ID, so every notebook re-run added the same chunks again
   (402 chunks became 1608). Fix: IDs are built from file name + chunk number, and before
   adding a file we delete its old chunks. Re-running is now always safe.
3. `collection.add(...)` was indented INSIDE the for-loop, so it ran once per chunk.
   Now we add in a few big batches.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

BATCH_SIZE = 256  # how many chunks we send to Chroma per call


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        # PersistentClient = the database is saved on disk, so it survives a restart.
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self._open_collection()
        logger.info("Vector store ready: %s (%d chunks)", collection_name, self.count())

    def _open_collection(self):
        """Open (or create) the collection with COSINE distance."""
        try:
            # Newer Chroma versions use `configuration`
            return self.client.get_or_create_collection(
                name=self.collection_name, configuration={"hnsw": {"space": "cosine"}}
            )
        except (TypeError, ValueError):
            # Older Chroma versions use metadata for the same thing
            return self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    # ---------- writing ----------
    def replace_source(self, source: str, chunks: list[Document], vectors: list[list[float]]) -> None:
        """Store all chunks of ONE file, replacing any older version of that file.

        This is what makes ingestion "idempotent": running it 1 time or 10 times gives the same result.
        """
        if len(chunks) != len(vectors):
            raise ValueError("number of chunks and number of vectors must match")

        # 1) remove the previous version of this file (if any)
        self.collection.delete(where={"source": source})

        # 2) add the new chunks in batches
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            self.collection.upsert(
                # Stable ID: same file + same chunk number => same ID (never random)
                ids=[f"{source}::{c.metadata['chunk_index']}" for c in batch],
                documents=[c.page_content for c in batch],
                embeddings=vectors[start:start + BATCH_SIZE],
                metadatas=[
                    {"source": c.metadata["source"], "page": c.metadata["page"],
                     "chunk_index": c.metadata["chunk_index"]}
                    for c in batch
                ],
            )
        logger.info("Stored %d chunks from %s", len(chunks), source)

    def reset(self) -> None:
        """Delete everything and start with an empty collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._open_collection()

    # ---------- reading ----------
    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        """Names of all files currently indexed."""
        metadatas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return sorted({m["source"] for m in metadatas if m})

    def search(self, query_vector: list[float], n_results: int) -> list[dict]:
        """Return the n closest chunks. `distance` is cosine distance (0 = identical)."""
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({"id": chunk_id, "text": text, "metadata": meta, "distance": dist})
        return hits
