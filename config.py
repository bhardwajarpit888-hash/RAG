"""
config.py - ONE place for every setting in the project.

WHY THIS FILE EXISTS
In your notebook, values like chunk_size=500, the model name and the folder paths
were spread over many cells. In real projects we keep them in a single file, so if
you want to try chunk_size=1000 you change ONE line and the whole app follows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Secrets (API keys) must NEVER be written inside code, because code goes to GitHub.
# We read them from a local file instead. We try ".env" and also your "api.env".
# Both files are listed in .gitignore, so they are never uploaded.
load_dotenv(".env")
load_dotenv("api.env")


@dataclass(frozen=True)  # frozen = settings cannot be changed by accident while running
class Settings:
    # ---- where files live ----
    upload_dir: Path = Path("data/pdfs")                 # uploaded PDFs are saved here
    vector_dir: Path = Path("data/vector_store_v2")      # Chroma saves its database here
    # NEW collection name on purpose: your old "pdf_documents" collection was created with
    # the wrong distance type (L2). A collection's distance type cannot be changed later.
    collection_name: str = "documents_cosine"

    # ---- embeddings ----
    embedding_model: str = "all-MiniLM-L6-v2"            # same model you used in the notebook

    # ---- chunking ----
    chunk_size: int = 500        # max characters per chunk
    chunk_overlap: int = 50      # characters shared between neighbouring chunks

    # ---- retrieval ----
    top_k: int = 5               # how many chunks we ask the vector DB for
    # Chunks with cosine similarity below this are treated as "not relevant".
    # Tune it with `python -m eval.evaluate --sweep` instead of guessing.
    min_score: float = 0.25
    use_reranker: bool = True                               # second-stage reranking (big accuracy gain in our eval)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_min_score: float = 0.05                          # chosen from the sweep: plateau 0.01 to 0.10

    # ---- LLM (read from environment so you can switch without touching code) ----
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-20b"))
    temperature: float = 0.1     # low = more factual, less creative
    max_tokens: int = 1024       # gpt-oss / qwen3 "think" first, so do not set this too low


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Create the Settings object once and reuse it everywhere."""
    return Settings()
