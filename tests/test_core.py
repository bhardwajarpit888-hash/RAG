"""
Automated tests. Run with:   pytest -q

Why tests? They prove the important rules still work after you change code.
Two of these tests lock in the bug fixes from your notebook (cosine distance, no duplicates).
None of them need an API key or the embedding model, so they run in seconds.
"""
import pytest

from rag.chunker import split_documents
from rag.loader import load_file
from rag.vector_store import VectorStore


@pytest.fixture
def sample_chunks(tmp_path):
    file = tmp_path / "notes.txt"
    file.write_text(("RAG combines retrieval with generation to reduce hallucination. " * 30), encoding="utf-8")
    return split_documents(load_file(file), chunk_size=200, chunk_overlap=20)


def test_chunks_have_metadata_and_respect_size(sample_chunks):
    assert len(sample_chunks) > 1
    for i, chunk in enumerate(sample_chunks):
        assert chunk.metadata["source"] == "notes.txt"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["chunk_index"] == i
        assert len(chunk.page_content) <= 200


def test_unsupported_file_type_is_rejected(tmp_path):
    bad = tmp_path / "image.png"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(bad)


def _fake_vectors(n):
    # tiny made-up vectors, enough to test storage without the real model
    return [[1.0, 0.0, 0.0, 0.0] for _ in range(n)]


def test_ingesting_same_file_twice_does_not_duplicate(tmp_path, sample_chunks):
    store = VectorStore(tmp_path / "db", "test_collection")
    store.replace_source("notes.txt", sample_chunks, _fake_vectors(len(sample_chunks)))
    first_count = store.count()
    store.replace_source("notes.txt", sample_chunks, _fake_vectors(len(sample_chunks)))
    assert store.count() == first_count          # your notebook would have doubled this number


def test_collection_uses_cosine_distance(tmp_path, sample_chunks):
    store = VectorStore(tmp_path / "db", "test_cosine")
    store.replace_source("notes.txt", sample_chunks[:1], [[1.0, 0.0, 0.0, 0.0]])
    # A vector at 90 degrees has cosine distance 1.0. With the default L2 distance it would be 2.0.
    hit = store.search([0.0, 1.0, 0.0, 0.0], n_results=1)[0]
    assert hit["distance"] == pytest.approx(1.0, abs=1e-3)
