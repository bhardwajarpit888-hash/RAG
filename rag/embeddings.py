"""
rag/embeddings.py - STEP 3: text  ->  vectors (lists of numbers).

Texts with similar MEANING get vectors that point in similar directions.
That is what makes "semantic search" possible.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, model_name: str):
        # Imported here (not at the top) so tests that do not need the model start fast.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=64,
            # normalize_embeddings=True makes every vector length 1.
            # Then "cosine similarity" is a clean number between -1 and 1.
            normalize_embeddings=True,
            show_progress_bar=False,   # your notebook printed a progress bar for every query; this removes the noise
        )
        return vectors.tolist()        # Chroma wants plain Python lists, not numpy arrays

    def embed_query(self, text: str) -> list[float]:
        # IMPORTANT: queries and documents must use the SAME model, otherwise their vectors are not comparable.
        return self.embed_documents([text])[0]
