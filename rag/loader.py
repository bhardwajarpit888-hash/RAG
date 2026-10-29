"""
rag/loader.py - STEP 1 of the pipeline: files  ->  Document objects.

A "Document" (from LangChain) is just: text + metadata (a dict of extra info).
We store the file name and the PAGE NUMBER in metadata, because later we show
them as citations ("answer came from paper.pdf, page 4").
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    """Light cleaning. We keep newlines because the text splitter uses them as split points."""
    text = text.replace("\x00", "")          # null bytes sometimes appear in PDFs
    text = re.sub(r"[ \t]+", " ", text)      # many spaces/tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)   # many blank lines -> one blank line
    return text.strip()


def load_pdf(path: str | Path) -> list[Document]:
    """One Document per PDF page. Page numbers start at 1 (same as the number you see in a PDF viewer)."""
    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if not text:
            # A blank page or a scanned image page has no text layer. Skip it.
            logger.warning("%s page %d has no extractable text, skipping", path.name, page_number)
            continue
        # source = file NAME only (not full path), so citations look clean and
        # the same file gives the same source on every computer.
        docs.append(Document(page_content=text, metadata={"source": path.name, "page": page_number}))
    return docs


def load_text(path: str | Path) -> list[Document]:
    """A .txt file is treated as a single page."""
    path = Path(path)
    text = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]


def load_file(path: str | Path) -> list[Document]:
    """Pick the right loader from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {suffix} (use .pdf, .txt or .md)")
