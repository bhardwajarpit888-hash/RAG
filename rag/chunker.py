"""
rag/chunker.py - STEP 2: long pages  ->  small chunks.

WHY CHUNKS? An embedding model turns text into ONE vector. If the text is a full page,
that vector is a vague "average" of many topics, and search gets worse.
Small focused chunks give sharper vectors, so retrieval finds the right place.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHUNK_CHARS = 40

# A bibliography entry starts like "[53] J. Ala" or "[1] Martin Abadi,".
# Table rows like "CoG [29] Wikipedia Text" do NOT match, because the word after [29] is not a name.
_REFERENCE_ENTRY = re.compile(r"\[\d{1,3}\]\s+(?:[A-Z]\.\s?[A-Z]?|[A-Z][a-zà-ÿ]+\s+[A-Z][a-zà-ÿ]+,)")


def looks_like_references(text: str, min_hits: int = 2) -> bool:
    """True if the chunk is part of a reference list. Such chunks are noise for search:
    they contain author names and paper titles, so they match many questions by accident."""
    return len(_REFERENCE_ENTRY.findall(text)) >= min_hits


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    chunks = [
        c for c in chunks
        if len(c.page_content.strip()) >= MIN_CHUNK_CHARS and not looks_like_references(c.page_content)
    ]

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks