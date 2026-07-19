"""
app.py
======
Streamlit front-end for the RAG-System project.

Provides:
    * Sidebar: model info, chunk-size / overlap / top-k sliders,
      index creation & loading controls, chat reset.
    * Main page: question input, generated answer, retrieved context
      with similarity scores and metadata (expandable), and full
      chat history for the session.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config, get_logger  # noqa: E402
from ingest import index_exists, run_ingestion_pipeline  # noqa: E402
from rag import RAGPipeline, RAGResponse  # noqa: E402
from utils.llm import (  # noqa: E402
    NoModelAvailableError,
    OllamaConnectionError,
    list_installed_models,
    resolve_model,
)

logger = get_logger(__name__)

st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    """Ensure all session-state keys used across the app exist up front."""
    defaults = {
        "pipeline": None,           # RAGPipeline instance once index is loaded
        "chat_history": [],         # List[RAGResponse]
        "index_ready": index_exists(),
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_pipeline_if_needed() -> None:
    """Lazily load the RAGPipeline from disk into session state, once."""
    if st.session_state.pipeline is None and st.session_state.index_ready:
        with st.spinner("Loading FAISS index..."):
            try:
                st.session_state.pipeline = RAGPipeline.from_disk()
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_error = f"Failed to load index: {exc}"
                st.session_state.index_ready = False


def _get_ollama_status() -> tuple[str, str]:
    """
    Return (status_label, detail) describing the local Ollama setup,
    used to populate the sidebar's model info panel.
    """
    try:
        installed = list_installed_models()
        if not installed:
            return "⚠️ No models installed", f"Run: ollama pull {Config.OLLAMA_MODEL}"
        model = resolve_model()
        return "✅ Connected", f"Active model: {model}"
    except OllamaConnectionError:
        return "❌ Ollama not reachable", "Start it with: ollama serve"
    except NoModelAvailableError as exc:
        return "⚠️ No supported model", str(exc)
    except Exception as exc:  # noqa: BLE001
        return "❌ Error", str(exc)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 RAG Control Panel")

    # --- Model info -----------------------------------------------------------
    st.subheader("🤖 Model Status")
    status_label, status_detail = _get_ollama_status()
    st.markdown(f"**Ollama:** {status_label}")
    st.caption(status_detail)
    st.caption(f"Embedding model: `{Config.EMBEDDING_MODEL}`")

    st.divider()

    # --- Retrieval / chunking parameters --------------------------------------
    st.subheader("⚙️ Pipeline Settings")

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=128, max_value=1024, value=Config.CHUNK_SIZE, step=32,
        help="Maximum characters per chunk during ingestion.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0, max_value=256, value=Config.CHUNK_OVERLAP, step=16,
        help="Overlapping characters between consecutive chunks.",
    )
    top_k = st.slider(
        "Top-K retrieved chunks",
        min_value=1, max_value=15, value=Config.TOP_K, step=1,
        help="Number of chunks retrieved from FAISS per question.",
    )

    if chunk_overlap >= chunk_size:
        st.warning("Chunk overlap must be smaller than chunk size.")

    st.divider()

    # --- Index management -------------------------------------------------------
    st.subheader("🗂️ Vector Index")

    index_status = "✅ Index found on disk" if index_exists() else "⚠️ No index found"
    st.caption(index_status)

    col1, col2 = st.columns(2)

    with col1:
        create_clicked = st.button(
            "🏗️ Create Index", use_container_width=True,
            disabled=(chunk_overlap >= chunk_size),
        )
    with col2:
        load_clicked = st.button("📥 Load Index", use_container_width=True)

    if create_clicked:
        progress_bar = st.progress(0.0, text="Starting ingestion...")

        def _on_progress(stage: str, fraction: float) -> None:
            progress_bar.progress(fraction, text=stage)

        try:
            summary = run_ingestion_pipeline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                progress_callback=_on_progress,
            )
            st.session_state.index_ready = True
            st.session_state.pipeline = RAGPipeline.from_disk()
            st.success(
                f"Index created! {summary['num_documents']} docs -> "
                f"{summary['num_chunks']} chunks in {summary['elapsed_seconds']}s."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ingestion failed: {exc}")
            logger.exception("Ingestion failed")

    if load_clicked:
        if not index_exists():
            st.error("No saved index found. Click 'Create Index' first.")
        else:
            with st.spinner("Loading index from disk..."):
                try:
                    st.session_state.pipeline = RAGPipeline.from_disk()
                    st.session_state.index_ready = True
                    st.success("Index loaded successfully.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to load index: {exc}")
                    logger.exception("Failed to load index")

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📚 Retrieval-Augmented Generation QA System")
st.caption("Powered by Open RAGBench · FAISS · Sentence-Transformers · Ollama")

_load_pipeline_if_needed()

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if not st.session_state.index_ready or st.session_state.pipeline is None:
    st.info(
        "👈 No vector index is loaded yet. Use the sidebar to **Create Index** "
        "(first run) or **Load Index** (if one already exists on disk)."
    )
else:
    question = st.text_input(
        "Ask a question about the indexed documents:",
        placeholder="e.g. What is retrieval-augmented generation?",
        key="question_input",
    )
    ask_clicked = st.button("🔍 Ask", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Retrieving context and generating answer..."):
            try:
                response: RAGResponse = st.session_state.pipeline.ask(
                    question=question,
                    top_k=top_k,
                    max_context_chunks=min(top_k, Config.MAX_CONTEXT_CHUNKS),
                )
                st.session_state.chat_history.append(response)
            except (OllamaConnectionError, NoModelAvailableError) as exc:
                st.error(f"LLM error: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                logger.exception("Error while answering question")
    elif ask_clicked:
        st.warning("Please enter a question first.")

    st.divider()

    # --- Chat history (most recent first) ----------------------------------------
    if not st.session_state.chat_history:
        st.caption("No questions asked yet this session.")

    for turn_idx, turn in enumerate(reversed(st.session_state.chat_history)):
        with st.container(border=True):
            st.markdown(f"**Q: {turn.question}**")

            if not turn.grounded:
                st.warning(turn.answer)
            else:
                st.markdown(turn.answer)

            # --- Metrics row -----------------------------------------------------
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieval time", f"{turn.retrieval_time:.2f}s")
            m2.metric("Generation time", f"{turn.generation_time:.2f}s")
            m3.metric("Total time", f"{turn.total_time:.2f}s")
            m4.metric("Chunks used", len(turn.retrieved_chunks))

            if turn.model_used:
                st.caption(f"Model: `{turn.model_used}`")

            # --- Retrieved chunks (expandable, with scores + metadata) -----------
            if turn.retrieved_chunks:
                with st.expander(f"📄 View {len(turn.retrieved_chunks)} retrieved chunks"):
                    for result in turn.retrieved_chunks:
                        title = result.chunk.metadata.get("title") or result.chunk.doc_id
                        st.markdown(
                            f"**#{result.rank} — {title}** "
                            f"&nbsp;·&nbsp; similarity: `{result.score:.3f}` "
                            f"&nbsp;·&nbsp; id: `{result.chunk.chunk_id}`"
                        )
                        st.text(result.chunk.text)
                        st.caption(f"Metadata: {result.chunk.metadata}")
                        st.markdown("---")

        st.caption("")  # small spacer between turns


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "RAG-System · Dataset: vectara/open_ragbench · "
    "Embeddings: all-MiniLM-L6-v2 · Vector store: FAISS · LLM: Ollama (local, free)"
)
