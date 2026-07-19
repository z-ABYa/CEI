"""
utils/chunking.py
==================
Splits cleaned :class:`~utils.loader.Document` objects into smaller,
overlapping text chunks suitable for embedding and retrieval.

We deliberately reuse LangChain's ``RecursiveCharacterTextSplitter``
here (and only here) because reimplementing a robust recursive
splitter adds little pedagogical value and LangChain's version is
well-tested. Every other stage of the pipeline is implemented by hand.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config, get_logger  # noqa: E402
from utils.loader import Document  # noqa: E402

logger = get_logger(__name__)


@dataclass
class Chunk:
    """
    A single chunk of text produced from a parent :class:`Document`.

    Attributes
    ----------
    chunk_id : str
        Unique identifier, formatted as ``"{doc_id}_chunk_{index}"``.
    text : str
        The chunk's text content.
    doc_id : str
        Identifier of the parent document this chunk was extracted from.
    chunk_index : int
        Position of this chunk within its parent document (0-indexed).
    metadata : Dict[str, Any]
        Extra context carried over from the parent document (title,
        source index, etc.) plus chunk-specific stats, useful for
        citations in the UI.
    """

    chunk_id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _build_splitter(chunk_size: int, chunk_overlap: int):
    """
    Instantiate a LangChain ``RecursiveCharacterTextSplitter`` configured
    with the project's separators and size/overlap settings.

    Kept as its own function so ``ingest.py`` and tests can build a
    splitter with custom parameters (e.g. from Streamlit sliders)
    without duplicating configuration logic.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=Config.CHUNK_SEPARATORS,
        length_function=len,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split a list of documents into overlapping chunks.

    Parameters
    ----------
    documents : List[Document]
        Cleaned documents as produced by ``utils.loader.load_open_ragbench``.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks,
        used to avoid losing context at chunk boundaries.

    Returns
    -------
    List[Chunk]
        Flat list of chunks across all input documents, each tagged
        with metadata linking it back to its parent document.

    Raises
    ------
    ValueError
        If ``chunk_overlap`` is greater than or equal to ``chunk_size``,
        which would produce degenerate or infinite splitting behaviour.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )

    if not documents:
        logger.warning("chunk_documents called with an empty document list.")
        return []

    splitter = _build_splitter(chunk_size, chunk_overlap)
    all_chunks: List[Chunk] = []

    logger.info(
        "Chunking %d documents (chunk_size=%d, overlap=%d)...",
        len(documents), chunk_size, chunk_overlap,
    )

    for document in documents:
        # Skip documents too short to be meaningfully chunked; keep them
        # as a single chunk instead of discarding useful short passages.
        raw_pieces = splitter.split_text(document.text)

        if not raw_pieces:
            continue

        for idx, piece in enumerate(raw_pieces):
            piece = piece.strip()
            if not piece:
                continue

            chunk = Chunk(
                chunk_id=f"{document.doc_id}_chunk_{idx}",
                text=piece,
                doc_id=document.doc_id,
                chunk_index=idx,
                metadata={
                    "title": document.title,
                    "source_question": document.question,
                    "char_length": len(piece),
                    "total_chunks_in_doc": len(raw_pieces),
                    **document.metadata,
                },
            )
            all_chunks.append(chunk)

    logger.info(
        "Produced %d chunks from %d documents (avg %.1f chunks/doc).",
        len(all_chunks), len(documents),
        len(all_chunks) / max(len(documents), 1),
    )
    return all_chunks


if __name__ == "__main__":
    # Quick manual smoke test: `python utils/chunking.py`
    from utils.loader import load_open_ragbench

    docs = load_open_ragbench(max_documents=3)
    chunks = chunk_documents(docs)
    for c in chunks[:5]:
        print(f"[{c.chunk_id}] ({c.metadata['char_length']} chars) {c.text[:80]}...")
