"""
ingest.py
=========
End-to-end ingestion pipeline for the RAG-System project.

Pipeline stages
---------------
    1. Load the Open RAGBench dataset
    2. Clean documents
    3. Chunk documents
    4. Embed chunks
    5. Build a FAISS index
    6. Persist the index + chunk metadata to disk

Run directly with:

    python ingest.py

Or import and call ``run_ingestion_pipeline()`` from other modules
(e.g. the Streamlit app's "Create Index" button).
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import faiss
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config, get_logger  # noqa: E402
from utils.chunking import Chunk, chunk_documents  # noqa: E402
from utils.embedding import embed_texts  # noqa: E402
from utils.loader import DatasetLoadError, load_open_ragbench  # noqa: E402

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[str, float], None]]


def _report(callback: ProgressCallback, stage: str, fraction: float) -> None:
    """
    Forward progress updates to an optional callback (e.g. a Streamlit
    progress bar) without coupling this module to Streamlit directly.

    Parameters
    ----------
    callback : Optional[Callable[[str, float], None]]
        Function accepting ``(stage_description, fraction_complete)``.
    stage : str
        Human-readable description of the current stage.
    fraction : float
        Progress within the overall pipeline, from 0.0 to 1.0.
    """
    logger.info("[%s] %.0f%%", stage, fraction * 100)
    if callback is not None:
        callback(stage, fraction)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index over the given embeddings using inner-product
    search (equivalent to cosine similarity, since embeddings are
    L2-normalized upstream in ``utils.embedding``).

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape ``(n_chunks, embedding_dim)``, dtype float32.

    Returns
    -------
    faiss.Index
        A flat (exact search) FAISS index, well-suited to the small-
        to-medium corpus sizes typical of a university project.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings array, got shape {embeddings.shape}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product == cosine (normalized vectors)
    index.add(embeddings)
    logger.info("Built FAISS IndexFlatIP with %d vectors (dim=%d).", index.ntotal, dimension)
    return index


def save_index(index: faiss.Index, chunks: List[Chunk]) -> None:
    """
    Persist the FAISS index and its associated chunk metadata to disk.

    The index is saved via ``faiss.write_index`` and the chunk objects
    (needed to map FAISS result positions back to text + metadata) are
    pickled separately, since FAISS itself only stores raw vectors.

    Parameters
    ----------
    index : faiss.Index
        The FAISS index to persist.
    chunks : List[Chunk]
        The chunks corresponding, in order, to each vector in ``index``.
    """
    Config.VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(Config.VECTOR_DB_PATH))
    with open(Config.METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(
        "Saved FAISS index to '%s' and metadata for %d chunks to '%s'.",
        Config.VECTOR_DB_PATH, len(chunks), Config.METADATA_PATH,
    )


def load_index() -> tuple[faiss.Index, List[Chunk]]:
    """
    Load a previously saved FAISS index and its chunk metadata.

    Returns
    -------
    tuple[faiss.Index, List[Chunk]]
        The FAISS index and the list of chunks aligned to its vectors.

    Raises
    ------
    FileNotFoundError
        If either the index file or the metadata file does not exist.
    """
    if not index_exists():
        raise FileNotFoundError(
            f"No index found at '{Config.VECTOR_DB_PATH}'. Run `python "
            f"ingest.py` (or click 'Create Index' in the app) first."
        )

    index = faiss.read_index(str(Config.VECTOR_DB_PATH))
    with open(Config.METADATA_PATH, "rb") as f:
        chunks: List[Chunk] = pickle.load(f)

    logger.info("Loaded FAISS index (%d vectors) and %d chunks.", index.ntotal, len(chunks))
    return index, chunks


def index_exists() -> bool:
    """Return True if a saved FAISS index and metadata file both exist."""
    return Config.VECTOR_DB_PATH.exists() and Config.METADATA_PATH.exists()


def run_ingestion_pipeline(
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
    max_documents: Optional[int] = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    """
    Run the full ingest pipeline: load -> clean -> chunk -> embed ->
    index -> save.

    Parameters
    ----------
    chunk_size : int
        Max characters per chunk (forwarded to the text splitter).
    chunk_overlap : int
        Overlap in characters between consecutive chunks.
    max_documents : Optional[int]
        Cap on number of documents pulled from the dataset. Defaults
        to ``Config.MAX_DOCUMENTS`` when ``None``.
    progress_callback : Optional[Callable[[str, float], None]]
        Optional hook for reporting progress to a UI (e.g. Streamlit).

    Returns
    -------
    dict
        Summary statistics: number of documents, chunks, embedding
        dimension, and total elapsed time in seconds.
    """
    start_time = time.time()

    # --- Stage 1: Load -----------------------------------------------------
    _report(progress_callback, "Loading Open RAGBench dataset", 0.05)
    try:
        documents = load_open_ragbench(max_documents=max_documents)
    except DatasetLoadError as exc:
        logger.error("Ingestion aborted: %s", exc)
        raise

    # --- Stage 2: Chunk ------------------------------------------------------
    _report(progress_callback, "Chunking documents", 0.30)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        raise RuntimeError("No chunks were produced from the loaded documents.")

    # --- Stage 3: Embed --------------------------------------------------------
    _report(progress_callback, "Generating embeddings", 0.50)
    chunk_texts = [c.text for c in chunks]

    # Manual batching with a progress bar so long runs give visible feedback,
    # matching the "display progress" requirement even for large corpora.
    embeddings_list = []
    batch_size = Config.EMBEDDING_BATCH_SIZE
    n_batches = (len(chunk_texts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Embedding batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(chunk_texts))
        batch_embeddings = embed_texts(chunk_texts[start:end], batch_size=batch_size)
        embeddings_list.append(batch_embeddings)

        fraction = 0.50 + 0.35 * ((batch_idx + 1) / n_batches)
        _report(progress_callback, f"Embedding batch {batch_idx + 1}/{n_batches}", fraction)

    embeddings = np.vstack(embeddings_list)

    # --- Stage 4: Build FAISS index --------------------------------------------
    _report(progress_callback, "Building FAISS index", 0.90)
    index = build_faiss_index(embeddings)

    # --- Stage 5: Save -----------------------------------------------------------
    _report(progress_callback, "Saving index to disk", 0.97)
    save_index(index, chunks)

    elapsed = time.time() - start_time
    _report(progress_callback, "Ingestion complete", 1.0)

    summary = {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "elapsed_seconds": round(elapsed, 2),
    }
    logger.info("Ingestion pipeline finished: %s", summary)
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("RAG-System Ingestion Pipeline")
    print("=" * 60)

    try:
        result = run_ingestion_pipeline()
    except (DatasetLoadError, RuntimeError) as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}")
        sys.exit(1)

    print("\nIngestion complete!")
    print(f"  Documents processed : {result['num_documents']}")
    print(f"  Chunks created       : {result['num_chunks']}")
    print(f"  Embedding dimension  : {result['embedding_dim']}")
    print(f"  Time elapsed          : {result['elapsed_seconds']}s")
    print(f"\nIndex saved to: {Config.VECTOR_DB_PATH}")
    print("You can now run: streamlit run app.py")
