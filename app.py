"""
app.py - the web UI.   Run with:   streamlit run app.py

Streamlit re-runs this whole file from top to bottom every time you click something.
That is why we keep chat history in `st.session_state` (it survives re-runs)
and why the heavy pipeline is created once with @st.cache_resource.
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config import get_settings
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_pipeline() -> RAGPipeline:
    # cache_resource = build this ONCE and share it, instead of reloading the model on every click.
    return RAGPipeline(get_settings())


def show_sources(sources) -> None:
    """Render the retrieved chunks under an answer. Numbers match the [1], [2] citations in the text."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s.source} - page {s.page}**  (similarity {s.score:.2f})")
            st.caption(s.text[:500] + ("..." if len(s.text) > 500 else ""))


pipeline = get_pipeline()
settings = pipeline.settings

# ---------------- sidebar: documents + settings ----------------
with st.sidebar:
    st.header("1. Add documents")
    uploaded = st.file_uploader("PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploaded, type="primary"):
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        for file in uploaded:
            # Path(...).name keeps only the file name. This blocks tricks like "../../secret.pdf".
            destination = settings.upload_dir / Path(file.name).name
            destination.write_bytes(file.getbuffer())
            with st.spinner(f"Indexing {destination.name}..."):
                try:
                    n_chunks = pipeline.ingest_file(destination)
                    st.success(f"{destination.name}: {n_chunks} chunks")
                except Exception as exc:
                    st.error(f"{destination.name}: {exc}")

    indexed = pipeline.store.list_sources()
    st.caption(f"Indexed files: {', '.join(indexed) if indexed else 'none yet'}")
    st.caption(f"Total chunks: {pipeline.store.count()}")

    st.header("2. Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, settings.top_k)
    default_min = settings.rerank_min_score if settings.use_reranker else settings.min_score
    min_score = st.slider("Minimum relevance", 0.0, 0.9, default_min, 0.01, 
                          help="Higher = stricter. Chunks below this are ignored and the app says it cannot find the answer.")
    if st.button("Clear all documents"):
        pipeline.store.reset()
        st.session_state.messages = []
        st.rerun()

# ---------------- main area: chat ----------------
st.title("📄 Chat with your documents")
st.caption("Answers come only from your files, with page citations. If it is not in the files, the app says so.")

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "sources"}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

if pipeline.store.count() == 0:
    st.info("Upload and index a document in the sidebar to start.")

question = st.chat_input("Ask a question about your documents", disabled=pipeline.store.count() == 0)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                answer = pipeline.ask(question, top_k=top_k, min_score=min_score)
                st.markdown(answer.text)
                show_sources(answer.sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer.text, "sources": answer.sources})
            except Exception as exc:   # e.g. missing API key, no internet, rate limit
                st.error(f"Something went wrong: {exc}")
