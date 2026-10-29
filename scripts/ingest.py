"""
Index documents from the command line.

    python -m scripts.ingest data/pdfs            # a whole folder
    python -m scripts.ingest data/pdfs/paper.pdf  # a single file

Safe to run again and again: each file REPLACES its old chunks, so no duplicates.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag.pipeline import RAGPipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Index PDFs/TXT files into the vector store")
    parser.add_argument("path", help="a file or a folder")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    target = Path(args.path)
    report = pipeline.ingest_folder(target) if target.is_dir() else {target.name: pipeline.ingest_file(target)}

    for name, n_chunks in report.items():
        print(f"{name}: {n_chunks} chunks")
    print(f"Total chunks in index: {pipeline.store.count()}")


if __name__ == "__main__":
    main()
